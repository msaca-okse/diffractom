"""Scale factors of fixed model components by EM for the Poisson likelihood (em_scales).

The prediction is a sum of J components (e.g. the predictions of the mineral phases, or of a phase and a background
model) with one non-negative scale per component and channel:

    mu_i = bg_i + max(sum_j a[j, c(s_i)] P_j,i, 0),

c(s) the channel of segment s (e.g. its q window, or 0 for all). With the components fixed, the scales are a small
Poisson problem, and EM solves it with a closed-form update (Poisson deviance as in the solvers):

    a[j, c] <- a[j, c] sum_{i in c} (w/g) P_j,i b_i / mu_i'  /  sum_{i in c} (w/g) P_j,i

Components can share their scales (ties: components with the same label get one common factor per channel, their
ratios fixed, e.g. window scales shared by all phases). The sums are per segment on the GPU (one work item per
segment, looping over the rows and translations), then per channel on the host.

Uses in the rock fits (DanMAX 2026): one scale per phase (the phases' intensity shares), one scale per q window shared
by all phases (window scales: intensity model and integration errors), one per phase and window (per-mineral window
scales: meteorite held-out deviance -2.4 %), the background level. Fit the scales on the fitted angles only (rows):
scales fitted on held-out angles bias the comparison. A subsample of the rows (e.g. every second fitted angle) gives the
same scales at a fraction of the cost.
"""
import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

_SRC = r"""
#define PER(a, p, i) ((p) ? (a)[(i) % (p)] : 0.0f)
__kernel void seg_sums(__global const float *G, __global const float *b, __global const float *w, const ulong pw,
                       __global const float *g, const ulong pg, __global const float *bg, const ulong pbg,
                       __global const long *rows, const long R, const long My, const long ns,
                       __global const float *a, const float eps, const int with_den,
                       __global double *num, __global double *den)
{
    // one work item per segment s; G: (NJ, R, My, ns) the components on the selected rows; a: (NJ, ns)
    const long s = get_global_id(0);
    if (s >= ns) return;
    double nu[NJ], de[NJ];
    float aj[NJ];
    for (int j = 0; j < NJ; ++j) { nu[j] = 0.0; de[j] = 0.0; aj[j] = a[j * ns + s]; }
    const long stride = R * My * ns;
    for (long r = 0; r < R; ++r) {
        for (long y = 0; y < My; ++y) {
            const long loc = (r * My + y) * ns + s;
            const long full = (rows[r] * My + y) * ns + s;
            const float wi = PER(w, pw, full);
            if (wi == 0.0f) continue;
            const float gi = PER(g, pg, full);
            float p = 0.0f;
            for (int j = 0; j < NJ; ++j) p += aj[j] * G[j * stride + loc];
            const float mu = fmax(p, 0.0f) + PER(bg, pbg, full) + eps * gi;
            const double f = (double)(wi / gi);
            const double rat = f * b[full] / mu;
            for (int j = 0; j < NJ; ++j) {
                const double pj = G[j * stride + loc];
                nu[j] += pj * rat;
                de[j] += pj * f;
            }
        }
    }
    for (int j = 0; j < NJ; ++j) {
        num[j * ns + s] = nu[j];
        if (with_den) den[j * ns + s] = de[j];
    }
}
"""


def _periodic_dev(queue, a, dshape, name, fill):
    from .fista_poisson import _periodic
    return _periodic(queue, a if a is not None else (np.full(1, fill, np.float32) if fill is not None else None),
                     dshape, name, [])


def em_scales(queue, components, b, channels=None, ties=None, rows=None, weights=None, gain=None, background=None,
              a0=None, iters=30, eps_counts=1e-3, tol=None, verbose=False):
    """EM for the scales of fixed components (module docstring).

    queue: a pyopencl CommandQueue. components: a list of J data-shaped arrays (N_Omega, My, N_seg) (NumPy or
    pyopencl; the predictions A_j x_j of the parts, at their current scale). b: the data (NumPy or pyopencl).
    channels: (N_seg,) int, the channel of every segment (e.g. the window index of each (eta, window) segment:
    np.tile(np.arange(N_windows), N_eta)); None: one channel. ties: (J,) labels; components with equal labels share one
    factor per channel (None: each its own). rows: the angles to fit (bool mask or indices; None: all). weights, gain,
    background: as in the solvers (per segment, per (translation, segment) or data-shaped). a0: (J, C) start (default
    ones). tol: stop when no scale changes by more than tol (relative) in an iteration.

    Returns a (J, C) array of scales: multiply component j by a[j, channels] (e.g. the coefficients of phase j by
    a[j, 0] with one channel)."""
    ctx = queue.context
    J = len(components)
    if J == 0:
        raise ValueError("no components")
    bd = b if isinstance(b, clarray.Array) else clarray.to_device(queue, np.ascontiguousarray(b, np.float32))
    dshape = tuple(bd.shape) if bd.ndim == 3 else tuple(np.shape(components[0]))
    n_om, My, ns = dshape
    rowlen = My * ns
    if rows is None:
        ridx = np.arange(n_om, dtype=np.int64)
    else:
        r = np.asarray(rows)
        ridx = np.flatnonzero(r).astype(np.int64) if r.dtype == bool else r.astype(np.int64)
    R = len(ridx)
    ch = np.zeros(ns, np.int64) if channels is None else np.asarray(channels, np.int64).ravel()
    if ch.size != ns:
        raise ValueError(f"channels must have N_seg = {ns} entries")
    C = int(ch.max()) + 1
    lab = np.arange(J) if ties is None else np.asarray(ties)
    if lab.size != J:
        raise ValueError("ties must have one label per component")

    # the components on the selected rows, stacked (J, R, My, ns)
    G = clarray.empty(queue, (J, R, My, ns), np.float32)
    for j, P in enumerate(components):
        if isinstance(P, clarray.Array):
            for k, a_ in enumerate(ridx):
                cl.enqueue_copy(queue, G.data, P.data, src_offset=int(4 * a_ * rowlen),
                                dst_offset=int(4 * (j * R + k) * rowlen), byte_count=int(4 * rowlen))
        else:
            Ph = np.asarray(P, np.float32).reshape(dshape)[ridx]
            cl.enqueue_copy(queue, G.data, np.ascontiguousarray(Ph), dst_offset=int(4 * j * R * rowlen))
    w, pw = _periodic_dev(queue, weights, dshape, "weights", 1.0)
    g, pg = _periodic_dev(queue, gain, dshape, "gain", 1.0)
    bg, pbg = _periodic_dev(queue, background, dshape, "background", None)
    dummy = clarray.zeros(queue, (1,), np.float32)
    bg = bg if bg is not None else dummy
    rows_d = clarray.to_device(queue, ridx)
    num = clarray.empty(queue, (J, ns), np.float64)
    den = clarray.empty(queue, (J, ns), np.float64)
    seg_sums = cl.Program(ctx, _SRC).build(options=[f"-DNJ={J}"]).seg_sums
    ls = 128
    gs = -(-ns // ls) * ls

    a = np.ones((J, C)) if a0 is None else np.array(a0, np.float64).reshape(J, C)
    dens = None
    for it in range(int(iters)):
        a_seg = clarray.to_device(queue, np.ascontiguousarray(a[:, ch], np.float32))
        seg_sums(queue, (gs,), (ls,), G.data, bd.data, w.data, np.uint64(pw), g.data, np.uint64(pg), bg.data,
                     np.uint64(pbg), rows_d.data, np.int64(R), np.int64(My), np.int64(ns), a_seg.data,
                     np.float32(eps_counts), np.int32(dens is None), num.data, den.data)
        nseg = num.get()
        if dens is None:
            dseg = den.get()
            dens = np.zeros((J, C))
            for j in range(J):
                dens[j] = np.bincount(ch, weights=dseg[j], minlength=C)
        nums = np.zeros((J, C))
        for j in range(J):
            nums[j] = np.bincount(ch, weights=nseg[j], minlength=C)
        new = a.copy()
        for t in np.unique(lab):
            js = np.flatnonzero(lab == t)
            nu = (a[js] * nums[js]).sum(axis=0)
            de = (a[js] * dens[js]).sum(axis=0)
            fac = np.where(de > 0, nu / np.where(de > 0, de, 1.0), 1.0)
            new[js] = a[js] * fac
        change = np.max(np.abs(new / np.where(a > 0, a, 1.0) - 1.0)) if a.size else 0.0
        a = new
        a_seg.base_data.release()
        if verbose:
            print(f"em_scales iteration {it + 1}: max relative change {change:.3g}", flush=True)
        if tol is not None and change <= tol:
            break
    for arr in (G, num, den, rows_d, dummy):
        arr.base_data.release()
    return a
