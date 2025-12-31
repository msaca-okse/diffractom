import numpy as np
from collections import defaultdict


class MemoryCounter:
    """
    Deterministic memory accounting for GPU-like arrays.

    This class does NOT allocate memory.
    It only tracks sizes based on shapes and dtypes.
    """

    def __init__(self):

        # current allocated bytes
        self.current_bytes = 0

        # peak allocated bytes
        self.peak_bytes = 0

        # allocation name -> bytes (LIVE)
        self._allocations = {}

        # allocation snapshot at peak
        self._peak_allocations = {}

        # stack of scopes; each scope is a list of allocation names
        self._scope_stack = []

        # optional timeline (for debugging / reporting)
        self._events = []




    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    @staticmethod
    def _dtype_nbytes(dtype):
        return np.dtype(dtype).itemsize

    @staticmethod
    def _shape_nbytes(shape, dtype):
        n = 1
        for s in shape:
            n *= int(s)
        return n * np.dtype(dtype).itemsize

    def _update_peak(self):
        if self.current_bytes > self.peak_bytes:
            self.peak_bytes = self.current_bytes
            # snapshot live allocations at peak
            self._peak_allocations = dict(self._allocations)


    # ------------------------------------------------------------
    # allocation / free
    # ------------------------------------------------------------
    def alloc(self, name, shape, dtype=np.float32):
        """
        Register an allocation.

        Parameters
        ----------
        name : str
            Unique allocation name
        shape : tuple[int]
            Array shape
        dtype : numpy dtype
        """
        if name in self._allocations:
            raise RuntimeError(f"Allocation '{name}' already exists")

        nbytes = self._shape_nbytes(shape, dtype)

        self._allocations[name] = nbytes
        self.current_bytes += nbytes
        self._update_peak()

        if self._scope_stack:
            self._scope_stack[-1].append(name)

        self._events.append(("alloc", name, nbytes, self.current_bytes))

    def free(self, name):
        """
        Explicitly free an allocation.
        """
        if name not in self._allocations:
            raise RuntimeError(f"Cannot free unknown allocation '{name}'")

        nbytes = self._allocations.pop(name)
        self.current_bytes -= nbytes

        self._events.append(("free", name, nbytes, self.current_bytes))

    # ------------------------------------------------------------
    # scopes
    # ------------------------------------------------------------
    def enter_scope(self, label=None):
        """
        Enter a new allocation scope.
        All allocations in this scope are freed on exit.
        """
        self._scope_stack.append([])
        self._events.append(("enter_scope", label, 0, self.current_bytes))

    def exit_scope(self):
        """
        Exit current scope and free all allocations created in it.
        """
        if not self._scope_stack:
            raise RuntimeError("No active scope to exit")

        names = self._scope_stack.pop()
        for name in reversed(names):
            self.free(name)

        self._events.append(("exit_scope", None, 0, self.current_bytes))

    # ------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------
    def bytes_to_gb(self, nbytes):
        return nbytes / (1024 ** 3)

    def summary(self):
        """
        Return a concise summary dict.
        """
        return {
            "current_bytes": self.current_bytes,
            "peak_bytes": self.peak_bytes,
            "current_gb": self.bytes_to_gb(self.current_bytes),
            "peak_gb": self.bytes_to_gb(self.peak_bytes),
            "n_allocations": len(self._allocations),
        }

    def report(self, verbose=True, show_allocations=False, show_peak=False, min_mb=0.0):
        s = self.summary()
        if verbose:
            print("=== MemoryCounter Report ===")
            print(f"Current : {s['current_gb']:.3f} GB")
            print(f"Peak    : {s['peak_gb']:.3f} GB")
            print(f"Live allocations : {s['n_allocations']}")
            print("============================")

            if show_allocations:
                self.report_allocations(min_mb=min_mb)

            if show_peak:
                self.report_peak_allocations(min_mb=min_mb)

        return s



    def reset(self):
        """
        Reset everything.
        """
        self.current_bytes = 0
        self.peak_bytes = 0
        self._allocations.clear()
        self._scope_stack.clear()
        self._events.clear()


    def allocation_breakdown(self, sort=True):
        """
        Return a list of (name, bytes) for current live allocations.
        """
        items = list(self._allocations.items())
        if sort:
            items.sort(key=lambda x: x[1], reverse=True)
        return items


    def report_allocations(self, min_mb=0.0):
        """
        Print a table of current live allocations.

        Parameters
        ----------
        min_mb : float
            Only show allocations >= min_mb
        """
        items = self.allocation_breakdown()

        if not items:
            print("No live allocations.")
            return

        print("\n=== Live Allocation Breakdown ===")
        print(f"{'Name':40s} {'Size (MB)':>12s}")
        print("-" * 55)

        total = 0
        for name, nbytes in items:
            mb = nbytes / (1024 ** 2)
            if mb < min_mb:
                continue
            total += nbytes
            print(f"{name:40s} {mb:12.3f}")

        print("-" * 55)
        print(f"{'TOTAL':40s} {total / (1024 ** 2):12.3f}")
        print("================================\n")



    def peak_allocation_breakdown(self, sort=True):
        """
        Return a list of (name, bytes) for allocations live at peak.
        """
        items = list(self._peak_allocations.items())
        if sort:
            items.sort(key=lambda x: x[1], reverse=True)
        return items




    def report_peak_allocations(self, min_mb=0.0):
        """
        Print allocations that were live at peak memory usage.

        Parameters
        ----------
        min_mb : float
            Only show allocations >= min_mb
        """
        items = self.peak_allocation_breakdown()

        if not items:
            print("No peak allocation data available.")
            return

        print("\n=== Peak Allocation Breakdown ===")
        print(f"{'Name':40s} {'Size (MB)':>12s}")
        print("-" * 55)

        total = 0
        for name, nbytes in items:
            mb = nbytes / (1024 ** 2)
            if mb < min_mb:
                continue
            total += nbytes
            print(f"{name:40s} {mb:12.3f}")

        print("-" * 55)
        print(f"{'TOTAL @ PEAK':40s} {total / (1024 ** 2):12.3f}")
        print("=================================\n")
