import subprocess
import time
import threading

class GPUMemoryLogger:
    """Poll GPU memory usage in a background thread via nvidia-smi."""

    def __init__(self, interval=0.2, gpu_id = 0):
        """Start a logger that polls every *interval* seconds on *gpu_id*."""
        self.interval = interval
        self.times = []      # seconds since start
        self.memory_mb = []  # GPU memory usage in MB
        self._running = False
        self._thread = None
        self._t0 = None
        self.gpu_id = gpu_id

    def _worker(self):
        self._t0 = time.perf_counter()
        while self._running:
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu_id}",  # <-- IMPORTANT
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=subprocess.DEVNULL,
                )
                mem = float(out.decode().strip())
                t = time.perf_counter() - self._t0

                self.times.append(t)
                self.memory_mb.append(mem)
            except Exception:
                pass

            time.sleep(self.interval)


    def start(self):
        """Begin background polling."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background polling and join the thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def as_arrays(self):
        """Return (times, memory_mb) as numpy arrays."""
        import numpy as np
        return np.asarray(self.times), np.asarray(self.memory_mb)
