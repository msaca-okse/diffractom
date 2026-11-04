"""The forward model of raw detector counts: coefficients -> the expected counts of the detector cells of every
frame, through a forward operator whose eta bins are finer than the pixels' azimuthal extent.

    mu_cell(frame) = exposure(frame) * sum_seg w_cell,seg M[frame, seg] + background,    M = A x

A is the operator (SinglePhaseForwardOperator or LocalSinglePhaseForwardOperator: PF products and projections, with
its rings as the data channels), w the cell weights (cells.py: pixel factor x ring scale x radial profile x eta
interpolation). The data are an EventList (the cells with counts > 0); the negative Poisson log-likelihood of all
cells of all frames is

    sum_cells (mu - n log mu)  =  <s, x>  -  sum_events n_e log mu_e  +  const,    s = B^T 1 (the sensitivity)

B = W A the composed model, so only the events enter the data term; cells with zero counts are summed through s,
computed once with the operator's adjoint (sensitivity). The model lives on the GPU: the operator's data-sized work
array M (op.data_shape, float32) and the events (frame, cell, count; 12 bytes each) plus the solver's per-event
arrays.

Subsets of the rotation angles (set_subsets): the events are ordered by subset, and the subset forms of the
operator compute only those angles (op.set_omega_subsets; sparse or generated PF and the native or FFT projector).
"""
import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

from ..utils.arrays import check_device

KERNELS = r"""
inline void atomic_add_f(volatile __global float *p, const float v) {
    union { unsigned int u; float f; } old, nw;
    do { old.f = *p; nw.f = old.f + v; } while (atomic_cmpxchg((volatile __global unsigned int *)p, old.u, nw.u) != old.u);
}

// mu[e] = exposure[frame] * sum_t w[t] M[frame * nseg + seg[t]] over the cell's entries, events e0 .. e0 + n - 1
__kernel void ev_gather(__global const float *M, __global const int *frame, __global const int *cell,
                        __global const int *ptr, __global const int *seg, __global const float *w,
                        __global const float *expo, __global float *mu, const ulong e0, const ulong n,
                        const ulong nseg)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const size_t e = e0 + i;
        const int f = frame[e], c = cell[e];
        const __global float *row = M + (size_t)f * nseg;
        float acc = 0.0f;
        for (int t = ptr[c]; t < ptr[c + 1]; ++t) acc += w[t] * row[seg[t]];
        mu[e] = expo[f] * acc;
    }
}

// M[frame * nseg + seg[t]] += exposure[frame] * w[t] * r[e]   (the transpose of ev_gather)
__kernel void ev_scatter(__global float *M, __global const int *frame, __global const int *cell,
                         __global const int *ptr, __global const int *seg, __global const float *w,
                         __global const float *expo, __global const float *r, const ulong e0, const ulong n,
                         const ulong nseg)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const size_t e = e0 + i;
        const float re = r[e];
        if (re == 0.0f) continue;
        const int f = frame[e], c = cell[e];
        const float v = expo[f] * re;
        __global float *row = M + (size_t)f * nseg;
        for (int t = ptr[c]; t < ptr[c + 1]; ++t) atomic_add_f(row + seg[t], w[t] * v);
    }
}

// rows of the angles omegas[0 .. n_om - 1] (all angles: omegas = 0 .. N_Omega - 1): M[i] = exposure[frame] * tot[seg]
__kernel void fill_sensitivity(__global float *M, __global const int *omegas, __global const float *expo,
                               __global const float *tot, const ulong My, const ulong nseg, const ulong n)
{
    const ulong row = My * nseg;
    for (size_t t = get_global_id(0); t < n; t += get_global_size(0)) {
        const size_t s = t / row, rem = t - s * row;
        const size_t i = (size_t)omegas[s] * row + rem;
        M[i] = expo[i / nseg] * tot[i % nseg];
    }
}
"""


class RawDetectorModel:
    """Raw-detector forward model (module docstring).

    operator : the forward operator (its rings = the cells' rings, its eta bins = the cells' eta bins)
    cells : DetectorCells; events : EventList (frames = the operator's (N_Omega, My) data rows)
    background : (E,) expected background counts of the events (input order), or None
    background_total : the background summed over all cells of all frames (enters the deviance only)
    subsets : number of interleaved rotation-angle subsets (or a list of index arrays), see set_subsets
    storage : "gpu" (the events and the solvers' per-event arrays on the GPU) or "host" (in host memory, sent to the
        GPU in chunks of chunk_events in every pass; a second data-sized GPU array holds the back-projection)
    """

    def __init__(self, operator, cells, events, background=None, background_total=None, subsets=1, storage="gpu",
                 chunk_events=1 << 24):
        op = operator
        self.op = op
        self.queue = op.queue
        self.ctx = op.ctx
        if cells.n_eta != op.N_eta or cells.n_rings != op.N_peaks:
            raise ValueError(f"cells ({cells.n_eta} eta bins x {cells.n_rings} rings) do not match the operator "
                             f"({op.N_eta} x {op.N_peaks})")
        if not np.allclose(cells.eta_angle_range, op.eta_angle_range):
            raise ValueError("cells and operator have different eta ranges")
        if (events.n_omega, events.n_y) != (op.N_Omega, op.My):
            raise ValueError(f"events have {events.n_omega} x {events.n_y} frames, the operator "
                             f"{op.N_Omega} x {op.My}")
        if getattr(events, "unit", "cells") != "cells":
            raise ValueError("pixel events: map them onto the model's cells first (events.to_cells(cells))")
        if events.n_cells != cells.n_cells:
            raise ValueError("events and cells have different numbers of cells")
        if cells.nnz >= 2 ** 31:
            raise ValueError("too many cell weights for 32-bit indices")
        if storage not in ("gpu", "host"):
            raise ValueError("storage must be 'gpu' or 'host'")
        self.storage = storage
        self.chunk_events = int(chunk_events)
        self.cells, self.events = cells, events
        self.n_seg = int(op.N_seg)
        self.n_frames = int(op.N_Omega) * int(op.My)
        self.n_events = len(events)
        self._bg_in = None if background is None else np.asarray(background, dtype=np.float32)
        if self._bg_in is not None and self._bg_in.shape != (self.n_events,):
            raise ValueError("background: one value per event")
        self.background_total = float(background_total) if background_total is not None else (
            0.0 if background is None else float("nan"))
        self.prg = cl.Program(self.ctx, KERNELS).build()
        self.k = {n: cl.Kernel(self.prg, n) for n in ("ev_gather", "ev_scatter", "fill_sensitivity")}
        self.expo_gpu = clarray.to_device(self.queue, events.exposure.ravel().astype(np.float32))
        self._upload_cells()
        self.M = clarray.zeros(self.queue, op.data_shape, np.float32)
        self.G = clarray.zeros(self.queue, op.data_shape, np.float32) if storage == "host" else self.M
        self._all_omegas = clarray.to_device(self.queue, np.arange(op.N_Omega, dtype=np.int32))
        self.set_subsets(subsets)

    # ---------------------------------------------------------------- setup
    def _upload_cells(self):
        c = self.cells
        self.ptr_gpu = clarray.to_device(self.queue, c.ptr.astype(np.int32))
        self.seg_gpu = clarray.to_device(self.queue, c.seg.astype(np.int32))
        self.w_gpu = clarray.to_device(self.queue, c.w.astype(np.float32))
        self.tot_gpu = clarray.to_device(self.queue, c.seg_total.astype(np.float32))

    def set_ring_scales(self, ring_scales):
        """New ring scales s_r (cells.set_ring_scales); the sensitivity changes with them."""
        self.cells.set_ring_scales(ring_scales)
        self._upload_cells()

    def set_subsets(self, subsets):
        """Order the events by rotation-angle subset (M interleaved subsets for an int, or index arrays). Per-event
        arrays of event_array are dropped (their order changes)."""
        op = self.op
        if np.isscalar(subsets) and int(subsets) == 1:
            self.subsets = [np.arange(op.N_Omega)]
            self._op_subsets = False
        else:
            self.subsets = op.set_omega_subsets(subsets)
            self._op_subsets = True
        M = len(self.subsets)
        sub_of_omega = np.full(op.N_Omega, -1, dtype=np.int64)
        for m, s in enumerate(self.subsets):
            sub_of_omega[s] = m
        if np.any(sub_of_omega < 0):
            raise ValueError("the subsets must cover every rotation angle")
        ev = self.events
        sub = sub_of_omega[ev.frame // op.My]
        self.order = np.argsort(sub, kind="stable")
        self.ev_start = np.searchsorted(sub[self.order], np.arange(M + 1)).astype(np.int64)
        self._sub_omegas = [clarray.to_device(self.queue, s.astype(np.int32)) for s in self.subsets]
        q = self.queue
        self._ev_arrays = {}
        self._work = None
        if self.storage == "gpu":
            self.frame_gpu = clarray.to_device(q, ev.frame[self.order])
            self.cell_gpu = clarray.to_device(q, ev.cell[self.order])
            self.count_gpu = clarray.to_device(q, ev.count[self.order])
            self.bg_gpu = None if self._bg_in is None else clarray.to_device(q, self._bg_in[self.order])
        else:
            self._host = {"frame": ev.frame[self.order], "cell": ev.cell[self.order], "count": ev.count[self.order]}
            if self._bg_in is not None:
                self._host["bg"] = self._bg_in[self.order]
            n = int(min(self.chunk_events, max(self.n_events, 1)))
            self._chunk_n = n
            self._dev = {name: clarray.empty(q, n, a.dtype) for name, a in self._host.items()}
            self._dev["mu"] = clarray.empty(q, n, np.float32)
            self.bg_gpu = self._dev.get("bg")

    @property
    def n_subsets(self):
        return len(self.subsets)

    @property
    def has_background(self):
        return self._bg_in is not None

    def ev_range(self, subset=None):
        if subset is None:
            return 0, self.n_events
        return int(self.ev_start[subset]), int(self.ev_start[subset + 1])

    def chunks(self, subset=None):
        """The event ranges (a, b) of one pass over the subset (one range on the GPU)."""
        a, b = self.ev_range(subset)
        if self.storage == "gpu":
            return [(a, b)] if b > a else []
        return [(i, min(i + self._chunk_n, b)) for i in range(a, b, self._chunk_n)]

    def _op_subset(self, subset):
        return subset if (subset is not None and self._op_subsets) else None

    def _launch(self, name, n, *args):
        if n == 0:
            return
        k = self.k[name]
        ls = 256
        gs = min(int(n) + (-int(n)) % ls, ls * 4096)
        k.set_args(*args)
        cl.enqueue_nd_range_kernel(self.queue, k, (gs,), (ls,))

    # ---------------------------------------------------------------- per-event arrays (for the solvers)
    def event_array(self, name, fill=0.0):
        """A float32 array with one value per event, in the model's order, kept with the events (GPU: a pyopencl
        array; host: a NumPy array, sent to the GPU chunk by chunk by event_pass)."""
        if self.storage == "gpu":
            a = clarray.empty(self.queue, self.n_events, np.float32)
            a.fill(np.float32(fill))
        else:
            a = np.full(self.n_events, fill, np.float32)
            if name not in self._dev:
                self._dev[name] = clarray.empty(self.queue, self._chunk_n, np.float32)
        self._ev_arrays[name] = a
        return a

    def _bind(self, a, b, names, upload=True):
        """{name: device buffer}, offset and length for the events a .. b - 1."""
        n = b - a
        if self.storage == "gpu":
            if self._work is None:
                self._work = clarray.empty(self.queue, self.n_events, np.float32)
            arr = {"frame": self.frame_gpu, "cell": self.cell_gpu, "count": self.count_gpu, "mu": self._work}
            if self.bg_gpu is not None:
                arr["bg"] = self.bg_gpu
            for nm in names:
                arr[nm] = self._ev_arrays[nm]
            return arr, a, n
        arr = {}
        for nm in ("frame", "cell", "count", "bg") + tuple(names):
            if nm == "bg" and "bg" not in self._host:
                continue
            src = self._host[nm] if nm in self._host else self._ev_arrays[nm]
            if upload:
                cl.enqueue_copy(self.queue, self._dev[nm].data, src[a:b])
            arr[nm] = self._dev[nm]
        arr["mu"] = self._dev["mu"]
        return arr, 0, n

    def _gather(self, arr, off, n):
        self._launch("ev_gather", n, self.M.data, arr["frame"].data, arr["cell"].data, self.ptr_gpu.data,
                     self.seg_gpu.data, self.w_gpu.data, self.expo_gpu.data, arr["mu"].data, np.uint64(off),
                     np.uint64(n), np.uint64(self.n_seg))

    def _scatter(self, arr, off, n, target):
        self._launch("ev_scatter", n, target.data, arr["frame"].data, arr["cell"].data, self.ptr_gpu.data,
                     self.seg_gpu.data, self.w_gpu.data, self.expo_gpu.data, arr["mu"].data, np.uint64(off),
                     np.uint64(n), np.uint64(self.n_seg))

    def _zero_rows(self, target, subset):
        sub = self._op_subset(subset)
        if sub is None:
            target.fill(0.0)
        else:
            self.op.zero_subset_rows(target, sub)

    def event_pass(self, coeffs, out, subset, fn, names=(), write_back=()):
        """One pass of a solver over the events of a subset:
            M = A_m x;  per chunk: mu = W M (arrays["mu"]), fn(arrays, off, n), back-projection of arrays["mu"];
            out = A_m^T (the back-projection).
        fn launches the per-event kernels on events off .. off + n - 1 of the device buffers in arrays ("frame",
        "cell", "count", "bg" if any, "mu" and the event arrays in names) and leaves the values to back-project in
        "mu". Event arrays in write_back are copied back to host memory (storage "host")."""
        sub = self._op_subset(subset)
        self.op.direct_cl(coeffs, self.M, subset=sub)
        chunks = self.chunks(subset)
        if self.storage == "gpu":
            for a, b in chunks:
                arr, off, n = self._bind(a, b, names)
                self._gather(arr, off, n)
                fn(arr, off, n)
            self._zero_rows(self.M, subset)
            for a, b in chunks:
                arr, off, n = self._bind(a, b, names)
                self._scatter(arr, off, n, self.M)
            self.op.adjoint_cl(self.M, out, subset=sub)
            return
        self._zero_rows(self.G, subset)
        for a, b in chunks:
            arr, off, n = self._bind(a, b, names)
            self._gather(arr, off, n)
            fn(arr, off, n)
            self._scatter(arr, off, n, self.G)
            for nm in write_back:
                cl.enqueue_copy(self.queue, self._ev_arrays[nm][a:b], arr[nm].data)
        self.op.adjoint_cl(self.G, out, subset=sub)

    def event_values(self, coeffs, fn, reduce, subset=None):
        """sum over chunks of reduce(arrays, off, n) after mu = B x and fn(arrays, off, n) (fn may be None), over the
        subset's events (all events for None): e.g. the log-likelihood of the solvers' objective."""
        total = 0.0
        subs = range(self.n_subsets) if subset is None else [subset]
        for m in subs:
            msub = m if self.n_subsets > 1 else None
            self.op.direct_cl(coeffs, self.M, subset=self._op_subset(msub))
            for a, b in self.chunks(msub):
                arr, off, n = self._bind(a, b, ())
                self._gather(arr, off, n)
                if fn is not None:
                    fn(arr, off, n)
                total += float(reduce(arr, off, n))
        return total

    # ---------------------------------------------------------------- the model as arrays
    def forward_cl(self, coeffs, mu, subset=None):
        """mu[events of the subset] = (B x) there (no background), in the model's event order: mu is an
        (n_events,) float32 pyopencl array (storage "gpu") or NumPy array (storage "host")."""
        if self.storage == "gpu":
            check_device(mu, (self.n_events,), "mu")
        self.op.direct_cl(coeffs, self.M, subset=self._op_subset(subset))
        for a, b in self.chunks(subset):
            arr, off, n = self._bind(a, b, ())
            if self.storage == "gpu":
                arr["mu"] = mu
            self._gather(arr, off, n)
            if self.storage == "host":
                cl.enqueue_copy(self.queue, mu[a:b], arr["mu"].data)

    def adjoint_cl(self, r, coeffs, subset=None):
        """coeffs = B_m^T r (r: (n_events,) in the model's order, a pyopencl array for storage "gpu", NumPy for
        "host"; only the subset's events are read)."""
        target = self.G
        self._zero_rows(target, subset)
        for a, b in self.chunks(subset):
            arr, off, n = self._bind(a, b, ())
            if self.storage == "gpu":
                check_device(r, (self.n_events,), "r")
                arr["mu"] = r
            else:
                cl.enqueue_copy(self.queue, arr["mu"].data, np.ascontiguousarray(r[a:b], dtype=np.float32))
            self._scatter(arr, off, n, target)
        self.op.adjoint_cl(target, coeffs, subset=self._op_subset(subset))

    def sensitivity_cl(self, coeffs, subset=None):
        """coeffs = B_m^T 1 over ALL cells of the subset's frames (zero counts included)."""
        om = self._all_omegas if (subset is None or not self._op_subsets) else self._sub_omegas[subset]
        n_om = self.op.N_Omega if (subset is None or not self._op_subsets) else len(self.subsets[subset])
        self._launch("fill_sensitivity", n_om * self.op.My * self.n_seg, self.M.data, om.data, self.expo_gpu.data,
                     self.tot_gpu.data, np.uint64(self.op.My), np.uint64(self.n_seg),
                     np.uint64(n_om * self.op.My * self.n_seg))
        self.op.adjoint_cl(self.M, coeffs, subset=self._op_subset(subset))

    # ---------------------------------------------------------------- NumPy
    def _host_events(self):
        return np.zeros(self.n_events, np.float32)

    def forward(self, coeffs):
        """(n_events,) B x in the model's event order (model.count_cpu has the counts in that order)."""
        x = clarray.to_device(self.queue, np.ascontiguousarray(coeffs, dtype=np.float32))
        if self.storage == "gpu":
            mu = clarray.zeros(self.queue, (self.n_events,), np.float32)
            for m in range(self.n_subsets):
                self.forward_cl(x, mu, subset=m if self.n_subsets > 1 else None)
            return mu.get()
        mu = self._host_events()
        for m in range(self.n_subsets):
            self.forward_cl(x, mu, subset=m if self.n_subsets > 1 else None)
        return mu

    def adjoint(self, r):
        out = clarray.zeros(self.queue, self.op.coeff_shape, np.float32)
        acc = np.zeros(self.op.coeff_shape)
        rr = np.ascontiguousarray(r, dtype=np.float32)
        rd = clarray.to_device(self.queue, rr) if self.storage == "gpu" else rr
        for m in range(self.n_subsets):
            self.adjoint_cl(rd, out, subset=m if self.n_subsets > 1 else None)
            acc += out.get()
        return acc.astype(np.float32)

    def sensitivity(self):
        out = clarray.zeros(self.queue, self.op.coeff_shape, np.float32)
        self.sensitivity_cl(out)
        return out.get()

    @property
    def count_cpu(self):
        return self.events.count[self.order]

    @property
    def background_cpu(self):
        return None if self._bg_in is None else self._bg_in[self.order]

    def to_input_order(self, a):
        out = np.empty_like(a)
        out[self.order] = a
        return out

    def deviance(self, coeffs, eps=1e-6):
        """Poisson deviance of all cells (>= 0): sum (mu - n + n log(n / mu)) = <s, x> + bg_total - sum n
        + sum_events n log(n / mu). NumPy, for checks (the solvers compute it on the GPU)."""
        mu = self.forward(coeffs).astype(np.float64)
        bg = self.background_cpu
        if bg is not None:
            mu += bg
        n = self.count_cpu.astype(np.float64)
        s = self.sensitivity().astype(np.float64)
        lin = float((s * np.asarray(coeffs, np.float64)).sum()) + self.background_total
        mu = np.maximum(mu, eps)
        return lin - n.sum() + float(np.sum(np.where(n > 0, n * np.log(np.where(n > 0, n, 1.0) / mu), 0.0)))

    def frame_prediction(self, M_frame):
        """The model of every used pixel of one frame from that frame's data row M_frame (N_seg,) (e.g.
        op.direct(x)[i_omega, i_y]): (P,) expected counts per pixel (before exposure), for displays. With cells of
        several pixels, every pixel shows its cell's total."""
        c = self.cells
        P = c.pixels.n_pixels
        out = np.zeros(P)
        if c.nnz:
            vals = c.w.astype(np.float64) * np.asarray(M_frame, np.float64)[c.seg]
            cell_mu = np.add.reduceat(vals, c.ptr[:-1].clip(max=len(vals) - 1))
            cell_mu[np.diff(c.ptr) == 0] = 0.0
            ok = c.pixel_cell >= 0
            out[ok] = cell_mu[c.pixel_cell[ok]]
        return out


def powder_ring_scales(cells, events):
    """Ring scales from the data: s_r = (counts in ring r) / (the ring's model weight summed over all frames), the
    scale at which a powder (the same density M in every channel) reproduces every ring's total; the counterpart of
    normalising integrated data per ring. Needs cells that lie in one ring each (disjoint windows)."""
    c = cells
    ring_of_entry = c.ring_of_seg[c.seg]
    cell_ring = np.full(c.n_cells, -1, dtype=np.int64)
    first = c.ptr[:-1]
    has = np.diff(c.ptr) > 0
    cell_ring[has] = ring_of_entry[first[has]]
    # every cell's entries in one ring
    if np.any(ring_of_entry != np.repeat(cell_ring, np.diff(c.ptr))):
        raise ValueError("a cell spans several rings: ring scales from totals need disjoint windows")
    counts = np.bincount(cell_ring[events.cell], weights=events.count.astype(np.float64), minlength=c.n_rings)
    unscaled = np.bincount(c.ring_of_seg, weights=c.seg_total, minlength=c.n_rings) / np.maximum(c.ring_scales, 1e-300)
    weight = unscaled * float(events.exposure.astype(np.float64).sum())
    return counts / np.maximum(weight, 1e-300)
