"""Nonnegative reconstruction with the Poisson likelihood of photon-counting data (FISTAPoisson).

Model and loss
--------------
The detector counts photons, so a data point b_i (a sum of pixel counts divided by the solid-angle and
polarization normalisation, possibly scaled, summed or averaged) is a scaled Poisson variable with mean
mu_i = (A x)_i + bg_i and variance g_i mu_i. The gain g_i (data units per count, var / mean) is set by the
integration; see diffractom.utils.counting. The negative log-likelihood, as the Poisson deviance (zero at a
perfect fit, so the objective is the data misfit in "counts"):

    f(x) = sum_i w_i / g_i [ mu_i' - b_i + b_i log(b_i / mu_i') ],   mu_i' = max(mu_i, 0) + bg_i + eps_counts g_i

w_i are the usual weights (0 excludes a data point); eps_counts (default 1e-3 counts) keeps the logarithm finite
where the model predicts nothing. Compared with least squares, every data point is weighted by its own
counting statistics: bright and weak points have relative errors 1 / sqrt(counts), and zero counts carry
information. Rescaling b, bg and g by the same factor rescales x and leaves the minimiser otherwise unchanged.

Solver
------
f has no global Lipschitz constant (its curvature b / mu^2 grows without bound as mu -> 0), and the weights 1 /
(g mu) span many orders of magnitude, so FISTA with one step 1 / L stalls. Instead FISTA runs in a diagonal
metric D, the separable quadratic surrogate (SQS) of CT reconstruction (Erdogan & Fessler 1999):

    D_j = 1 / (A^T (c * A 1))_j,    c_i = w_i / g_i * max(b_i, m_i) / m_i^2,   m_i = max(mu_i, g_i)

c is the Poisson curvature b / m^2 where the prediction mu is below the data and its Pearson value 1 / m above (at
least one count); for A >= 0 the metric majorizes the quadratic model with curvature c. It needs one forward and
one adjoint pass and is computed at the starting prediction and again at the iterations in `refresh` (the momentum
restarts there). Because c is the curvature at one point only, each step is checked for sufficient decrease: a
rejected step first recomputes the metric at the extrapolated point and is retried, then the step factor theta
(x <- prox(y - theta D grad)) halves until it holds; theta doubles again per iteration up to 1. The momentum
restarts when the objective increases (O'Donoghue & Candes). The prediction at the extrapolated point is formed
from the two last predictions (A is linear), so an iteration costs one forward and one adjoint pass, as in
FISTAHuber, plus one forward per rejected step (and one forward and adjoint per metric update).

Start: at x = 0 the prediction is eps, where the likelihood has its barrier. By default the first n_start
iterations (20) fit the quadratic approximation of the deviance at the data instead (weighted least squares with
weights w / (g max(b, g)), the chi-square of the data: no barrier, the metric is an exact majorizer), then the
deviance takes over.

Memory (GPU): five coefficient-sized arrays (x, x_new, y, gradient, D) and four data-sized arrays (data,
three predictions); the coefficients are not streamed from host memory (FISTAHuber can stream them).
"""
import time

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

from .launch import elementwise, reduction
from .prox import support_mask_to_gpu
from ..utils.arrays import as_host, check_device

KERNELS = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// value (and gradient, in place of mu) of the Poisson deviance: w, g and bg are periodic (index i % period;
// period 0: the constant 1, 1 and 0)
inline float wv(__global const float *a, const ulong p, const ulong i, const float dflt){
    return p ? a[i % p] : dflt;
}

// mode 0: the deviance; mode 1: its quadratic approximation at the data (weighted least squares, weights
// w / (g max(b, floor g)), halved squares), used for the first iterations
__kernel void poisson_value_grad(
    __global float *mu, __global const float *b,
    __global const float *w, const ulong pw, __global const float *g, const ulong pg,
    __global const float *bg, const ulong pbg, const float eps, const int write_grad, const int mode,
    const float floor_counts, const ulong n, __global double *partial, __local double *scratch)
{
    double s = 0.0;
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const float wi = wv(w, pw, i, 1.0f);
        float gr = 0.0f;
        if (wi != 0.0f) {
            const float gi = wv(g, pg, i, 1.0f);
            const double bi = (double)b[i];
            const double sc = (double)wi / (double)gi;
            if (mode == 0) {
                const double m = (double)fmax(mu[i], 0.0f) + (double)wv(bg, pbg, i, 0.0f) + (double)eps * gi;
                double v = m - bi;
                if (bi > 0.0) v += bi * log(bi / m);
                s += sc * v;
                gr = (float)(sc * (1.0 - bi / m));
            } else {
                const double r = (double)mu[i] + (double)wv(bg, pbg, i, 0.0f) - bi;
                const double c = sc / fmax(bi, (double)floor_counts * gi);
                s += 0.5 * c * r * r;
                gr = (float)(c * r);
            }
        }
        if (write_grad) mu[i] = gr;
    }
    const int lid = get_local_id(0);
    scratch[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int h = get_local_size(0) / 2; h > 0; h >>= 1) {
        if (lid < h) scratch[lid] += scratch[lid + h];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial[get_group_id(0)] = scratch[0];
}

// u <- u * c, c = w / g * max(b, m) / max(m, floor g)^2 with m = max(mu, 0) + bg: the Poisson curvature
// b / m^2 where the model is below the data, its Pearson value 1 / m above (m at least floor counts)
// (mode 1: the curvature of the weighted least squares of the start, w / (g max(b, floor g)))
__kernel void times_curvature(
    __global float *u, __global const float *mu, __global const float *b,
    __global const float *w, const ulong pw, __global const float *g, const ulong pg,
    __global const float *bg, const ulong pbg, const float floor_counts, const int mode, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const float wi = wv(w, pw, i, 1.0f);
        const float gi = wv(g, pg, i, 1.0f);
        if (mode == 0) {
            const float m = fmax(fmax(mu[i], 0.0f) + wv(bg, pbg, i, 0.0f), floor_counts * gi);
            u[i] = wi != 0.0f ? u[i] * wi / gi * fmax(b[i], m) / (m * m) : 0.0f;
        } else {
            u[i] = wi != 0.0f ? u[i] * wi / (gi * fmax(b[i], floor_counts * gi)) : 0.0f;
        }
    }
}

// partial sums of w b and w mu (the uniform start: x0 = sum w b / sum w A 1)
__kernel void weighted_sums(__global const float *mu, __global const float *b, __global const float *w,
                            const ulong pw, const ulong n, __global double *partial, __local double *scratch)
{
    double s0 = 0.0, s1 = 0.0;
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const float wi = wv(w, pw, i, 1.0f);
        s0 += (double)wi * b[i];
        s1 += (double)wi * mu[i];
    }
    const int lid = get_local_id(0);
    const int ls = get_local_size(0);
    scratch[lid] = s0; scratch[ls + lid] = s1;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int h = ls / 2; h > 0; h >>= 1) {
        if (lid < h) { scratch[lid] += scratch[lid + h]; scratch[ls + lid] += scratch[ls + lid + h]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) { partial[get_group_id(0)] = scratch[0]; partial[get_num_groups(0) + get_group_id(0)] = scratch[ls]; }
}

__kernel void scale_buf(__global float *x, const float a, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) x[i] *= a;
}

// D = 1 / d where d > 0 and the pixel is in the support (support: uint8 per pixel, period npix; npix 0: none)
__kernel void invert_metric(__global float *D, __global const float *d, __global const uchar *supp,
                            const ulong npix, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const int inside = npix ? supp[i % npix] : 1;
        D[i] = (inside && d[i] > 0.0f) ? 1.0f / d[i] : 0.0f;
    }
}

// ones on the support (the SQS row sums are A applied to it)
__kernel void support_ones(__global float *x, __global const uchar *supp, const ulong npix, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0))
        x[i] = (npix ? supp[i % npix] : 1) ? 1.0f : 0.0f;
}

// xn = max(0, y - th D (grad + lam))   (nonnegativity, optional l1 weight lam; D = 0 outside the support)
__kernel void metric_step(__global float *xn, __global const float *y, __global const float *grad,
                          __global const float *D, const float th, const float lam, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0))
        xn[i] = D[i] > 0.0f ? fmax(0.0f, y[i] - th * D[i] * (grad[i] + lam)) : 0.0f;
}

// partial sums of <grad, xn - y>, sum (xn - y)^2 / D (D > 0) and sum xn
__kernel void step_sums(__global const float *xn, __global const float *y, __global const float *grad,
                        __global const float *D, const ulong n, __global double *partial, __local double *scratch)
{
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const double d = (double)xn[i] - (double)y[i];
        s0 += (double)grad[i] * d;
        if (D[i] > 0.0f) s1 += d * d / (double)D[i];
        s2 += (double)xn[i];
    }
    const int lid = get_local_id(0);
    const int ls = get_local_size(0);
    scratch[lid] = s0; scratch[ls + lid] = s1; scratch[2 * ls + lid] = s2;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int h = ls / 2; h > 0; h >>= 1) {
        if (lid < h) {
            scratch[lid] += scratch[lid + h];
            scratch[ls + lid] += scratch[ls + lid + h];
            scratch[2 * ls + lid] += scratch[2 * ls + lid + h];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        const size_t gid = get_group_id(0), ng = get_num_groups(0);
        partial[gid] = scratch[0]; partial[ng + gid] = scratch[ls]; partial[2 * ng + gid] = scratch[2 * ls];
    }
}

// out = a + beta (a - c)
__kernel void extrapolate(__global float *out, __global const float *a, __global const float *c, const float beta,
                          const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const float ai = a[i];
        out[i] = ai + beta * (ai - c[i]);
    }
}

__kernel void copy_buf(__global const float *src, __global float *dst, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) dst[i] = src[i];
}

__kernel void project_support(__global float *x, __global const uchar *supp, const ulong npix, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0))
        if ((npix && !supp[i % npix]) || x[i] < 0.0f) x[i] = 0.0f;
}
"""

DEFAULT_REFRESH = (5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120)


def _periodic(queue, a, dshape, name, uploaded):
    """A per-segment, per-(translation, segment) or per-data-point array as (device array, period);
    None -> (None, 0)."""
    if a is None:
        return None, 0
    n_seg, n_ys, n_all = dshape[-1], dshape[-2] * dshape[-1], int(np.prod(dshape))
    if isinstance(a, clarray.Array):
        if a.dtype != np.float32 or not a.flags.c_contiguous or a.size not in (n_seg, n_ys, n_all):
            raise ValueError(f"{name} must be C-contiguous float32 with N_seg = {n_seg}, My * N_seg = {n_ys} or "
                             f"{n_all} elements; got shape {a.shape}, {a.dtype}")
        return a, int(a.size)
    h = np.ascontiguousarray(a, dtype=np.float32)
    if h.size == 1:
        h = np.full(n_seg, float(h.ravel()[0]), np.float32)
    if h.size not in (n_seg, n_ys, n_all):
        raise ValueError(f"{name} must have N_seg = {n_seg} elements (one per segment), My * N_seg = {n_ys} (one per "
                         f"translation and segment, e.g. shaped (My, N_eta, N_rings)) or the data shape "
                         f"{tuple(dshape)}; "
                         f"got shape {h.shape}")
    d = clarray.to_device(queue, h.ravel())
    uploaded.append(d)
    return d, int(h.size)


class FISTAPoisson:
    """
    Solve  min_x  sum_i w_i / g_i [mu_i' - b_i + b_i log(b_i / mu_i')] + lam sum(x),  x >= 0 (in the support),
    mu' = A x + bg + eps_counts g, with FISTA in the SQS metric and backtracking (module docstring).

    x layout: (K, Ny, Nx) C order (dense operators) or the operator's flat coefficients (local basis)
    data layout: (N_Omega, My, N_seg) C order
    """

    def __init__(self, operator, prox_kind="nonneg", lam=0.0, support="fov", eps_counts=1e-3, floor_counts=1.0,
                 refresh=DEFAULT_REFRESH, refresh_on_failure=True, max_backtracks=60, start="wls", n_start=20):
        """
        Parameters
        ----------
        operator : SinglePhaseForwardOperator or another operator with direct_cl / adjoint_cl, coeff_shape and
            data_shape (A >= 0 elementwise, as for every diffraction operator here).
        prox_kind : 'nonneg' or 'nonneg_l1' (adds lam * sum(x)); the SQS metric is diagonal, so only element-wise
            proximal operators are available (no TV).
        support : "fov" (default), None or an (Ny, Nx) boolean array, as in FISTAHuber.
        eps_counts : offset of the prediction in counts (eps_counts * g_i data units): keeps the logarithm finite
            where the model predicts nothing; small compared with one count.
        floor_counts : the curvature of the metric (and of the start's weighted least squares) is taken at no fewer
            than floor_counts counts (default one).
        refresh : iterations (1-based) after which the metric is recomputed at the current prediction and the
            momentum restarts; () keeps the starting metric (updates on rejected steps aside).
        refresh_on_failure : when a step is rejected, first recompute the metric at the extrapolated point (one
            more forward and adjoint) and retry the full step, before halving it.
        max_backtracks : rejected steps per iteration before giving up (an error).
        start, n_start : at x = 0 the prediction is eps, where the likelihood has its barrier (curvature b / eps^2): a
            gradient method from there needs tiny steps. "wls" (default): a zero x0 is first fitted for n_start
            iterations with the quadratic approximation of the deviance at the data (weighted least squares, weights
            w / (g max(b, g)): no barrier, the metric is a majorizer, no backtracking), then the deviance takes over
            (metric at the prediction, momentum restart). "uniform": a zero x0 is replaced by the uniform image on the
            support with the data's total weighted intensity (as in EM). "x0": x0 as given.
        """
        if prox_kind not in ("nonneg", "nonneg_l1"):
            raise ValueError("FISTAPoisson supports prox_kind 'nonneg' and 'nonneg_l1' (element-wise proximal "
                             "operators in the diagonal metric)")
        self.op = operator
        self.ctx, self.queue = operator.ctx, operator.queue
        self.prox_kind = prox_kind
        self.lam = float(lam) if prox_kind == "nonneg_l1" else 0.0
        self.eps_counts = float(eps_counts)
        self.floor_counts = float(floor_counts)
        self.refresh = tuple(int(r) for r in refresh)
        self.max_backtracks = int(max_backtracks)
        self.refresh_on_failure = bool(refresh_on_failure)
        if start not in ("wls", "uniform", "x0"):
            raise ValueError("start must be 'wls', 'uniform' or 'x0'")
        self.start = start
        self.n_start = int(n_start)
        self._mode = 0
        self.support_gpu = support_mask_to_gpu(self.queue, operator, support)
        self.prg = cl.Program(self.ctx, KERNELS).build()
        self.k = {name: cl.Kernel(self.prg, name) for name in (
            "poisson_value_grad", "times_curvature", "invert_metric", "support_ones", "metric_step", "step_sums",
            "extrapolate", "copy_buf", "project_support", "weighted_sums", "scale_buf")}
        self.state = None

    # ---- small helpers ----
    def _npix(self):
        return (np.uint64(self.support_gpu.size), self.support_gpu.data) if self.support_gpu is not None \
            else (np.uint64(0), self._dummy_u8.data)

    def _data_term(self, mu, write_grad):
        """f at the prediction mu (the deviance, or in the start phase its quadratic approximation at the data); with
        write_grad, mu is overwritten by df/dmu."""
        g, l, n, ng = reduction(mu.size)
        self.k["poisson_value_grad"](
            self.queue, g, l, mu.data, self.b.data, self.w.data, np.uint64(self.pw), self.g.data, np.uint64(self.pg),
            self.bg.data, np.uint64(self.pbg), np.float32(self.eps_counts), np.int32(write_grad), np.int32(self._mode),
            np.float32(self.floor_counts), n, self._part.data, cl.LocalMemory(8 * l[0]))
        return float(self._part[:ng].get().sum())

    def _ones_forward(self, scratch, x_tmp):
        """scratch = A 1_S (the SQS row sums); x_tmp = 1_S."""
        npix, supp = self._npix()
        gx, nx = elementwise(x_tmp.size)
        self.k["support_ones"](self.queue, gx, None, x_tmp.data, supp, npix, nx)
        self.op.direct_cl(x_tmp, scratch)

    def _metric(self, mu, scratch, x_tmp, D):
        """D = 1 / A^T(c * A 1_S), c = the Poisson curvature at the prediction mu (kernel times_curvature); scratch:
        a data-sized buffer, x_tmp a coefficient-sized one (both overwritten)."""
        q = self.queue
        npix, supp = self._npix()
        gx, nx = elementwise(x_tmp.size)
        gd, nd = elementwise(scratch.size)
        self._ones_forward(scratch, x_tmp)
        self.k["times_curvature"](q, gd, None, scratch.data, mu.data, self.b.data, self.w.data,
                                  np.uint64(self.pw), self.g.data, np.uint64(self.pg), self.bg.data,
                                  np.uint64(self.pbg), np.float32(self.floor_counts), np.int32(self._mode), nd)
        self.op.adjoint_cl(scratch, x_tmp)
        self.k["invert_metric"](q, gx, None, D.data, x_tmp.data, supp, npix, nx)

    def run(self, x0, b, niter, weights=None, gain=None, background=None, verbose=0, diagnostics_interval=10,
            state=None, callback=None):
        """
        x0 : starting point (NumPy or pyopencl, the operator's coeff_shape, float32; zeros are fine). A NumPy array is
            uploaded and the result copied back into it (or into a converted copy) and returned; a pyopencl array is
            updated in place and returned.
        b : the data (N_Omega, My, N_seg), NumPy or pyopencl (in the units of the gain).
        weights : as in FISTAHuber (N_seg, My * N_seg or the data shape); 0 excludes a data point.
        gain : counting gain g = var / mean in data units per count (N_seg, My * N_seg or the data shape; a scalar
            broadcasts); None: the data are counts (g = 1). See diffractom.utils.counting.
        background : known additive background bg >= 0 of the prediction (same shapes), None: 0.
        state : the solver.state of an earlier run to continue it (y, momentum, step factor, iteration count), e.g.
            in chunks on time-limited GPU queues; x0 must then be that run's result.
        callback : called as callback(it, x_gpu, stats) after every iteration (it 1-based, counted over resumed runs);
            a true return value stops the run there (e.g. at a time limit; solver.state continues it).
        returns x (see x0). solver.iter_stats: per iteration f (deviance), obj (with lam sum(x)), theta, backtracks,
        restarts, time; solver.state: to continue.
        """
        t_start = time.perf_counter()
        q = self.queue
        op = self.op
        cshape, dshape = tuple(op.coeff_shape), tuple(op.data_shape)
        uploaded = []
        host_x0 = None
        if isinstance(x0, clarray.Array):
            x = check_device(x0, cshape, "x0")
        else:
            host_x0 = as_host(x0, cshape, "x0")
            x = clarray.to_device(q, host_x0)
        if isinstance(b, clarray.Array):
            self.b = check_device(b, dshape, "b")
        else:
            self.b = clarray.to_device(q, as_host(b, dshape, "b"))
            uploaded.append(self.b)
        self._dummy = clarray.zeros(q, (1,), np.float32)
        self._dummy_u8 = clarray.zeros(q, (1,), np.uint8)
        self.w, self.pw = _periodic(q, weights, dshape, "weights", uploaded)
        self.g, self.pg = _periodic(q, gain, dshape, "gain", uploaded)
        self.bg, self.pbg = _periodic(q, background, dshape, "background", uploaded)
        self.w = self.w if self.w is not None else self._dummy
        self.g = self.g if self.g is not None else self._dummy
        self.bg = self.bg if self.bg is not None else self._dummy
        if self.pg:
            gmin = float(clarray.min(self.g).get())
            if not gmin > 0:
                raise ValueError("the gain must be positive everywhere")
        _, _, _, ng = reduction(max(int(np.prod(dshape)), int(np.prod(cshape))))
        self._part = clarray.empty(q, (3 * ng,), np.float64)

        # ---- buffers ----
        xn = clarray.empty(q, cshape, np.float32)
        grad = clarray.empty(q, cshape, np.float32)
        D = clarray.empty(q, cshape, np.float32)
        Ax = clarray.empty(q, dshape, np.float32)
        Axn = clarray.empty(q, dshape, np.float32)
        Ay = clarray.empty(q, dshape, np.float32)
        gx, nx = elementwise(x.size)
        gd, nd = elementwise(Ax.size)
        npix, supp = self._npix()
        self.k["project_support"](q, gx, None, x.data, supp, npix, nx)
        zero_start = state is None and float(clarray.max(x).get()) == 0.0
        # the weighted least squares start: from zero, or continued from a state that is still in it
        wls_until = 0
        if self.start == "wls" and self.n_start > 0:
            if zero_start:
                wls_until = self.n_start
            elif state is not None and state.get("wls_until", 0) > int(state["it"]):
                wls_until = int(state["wls_until"])
        self._mode = 1 if wls_until else 0
        if zero_start and self.start == "uniform":
            # x0 = 0: the log-likelihood has its barrier there (prediction eps). Start instead, as EM does, from the
            # uniform image on the support with the data's total (weighted) intensity.
            self._ones_forward(Axn, xn)
            rg, rl, rn, rng_ = reduction(Axn.size)
            self.k["weighted_sums"](q, rg, rl, Axn.data, self.b.data, self.w.data, np.uint64(self.pw), rn,
                                    self._part.data, cl.LocalMemory(2 * 8 * rl[0]))
            sb, sa = self._part[:2 * rng_].get().reshape(2, rng_).sum(axis=1)
            if sa > 0 and sb > 0:
                self.k["copy_buf"](q, gx, None, xn.data, x.data, nx)
                self.k["scale_buf"](q, gx, None, x.data, np.float32(sb / sa), nx)

        it0, t_mom, theta = 0, 1.0, 1.0
        if state is not None:
            it0, t_mom, theta = int(state["it"]), float(state["t"]), float(state["theta"])
            y = clarray.to_device(q, as_host(state["y"], cshape, "state['y']"))
        else:
            y = clarray.empty(q, cshape, np.float32)
            self.k["copy_buf"](q, gx, None, x.data, y.data, nx)
        op.direct_cl(x, Ax)
        if state is not None:
            op.direct_cl(y, Ay)
        else:
            self.k["copy_buf"](q, gd, None, Ax.data, Ay.data, nd)
        self._metric(Ax, Axn, xn, D)  # at the starting prediction
        t_metric = time.perf_counter() - t_start

        f_x = self._data_term(Ax, False)
        F_x = f_x + self.lam * float(clarray.sum(x).get())
        self.iter_stats = []
        n_bt_total, n_restart, n_metric = 0, 0, 0
        red_g, red_l, red_n, red_ng = reduction(x.size)
        for k in range(niter):
            it = it0 + k + 1
            # gradient at y: df/dmu in place of A y, then A^T
            f_y = self._data_term(Ay, True)
            op.adjoint_cl(Ay, grad)
            theta = min(1.0, 2.0 * theta)
            n_bt = 0
            local = False  # metric recomputed at y in this iteration
            while True:
                self.k["metric_step"](q, gx, None, xn.data, y.data, grad.data, D.data, np.float32(theta),
                                      np.float32(self.lam), nx)
                op.direct_cl(xn, Axn)
                f_n = self._data_term(Axn, False)
                self.k["step_sums"](q, red_g, red_l, xn.data, y.data, grad.data, D.data, red_n, self._part.data,
                                    cl.LocalMemory(3 * 8 * red_l[0]))
                p = self._part[:3 * red_ng].get().reshape(3, red_ng).sum(axis=1)
                lin, quad, xsum = float(p[0]), float(p[1]) / theta, float(p[2])
                if f_n <= f_y + lin + 0.5 * quad + 1e-6 * abs(f_y) or quad == 0.0:
                    break
                n_bt += 1
                if not local and self.refresh_on_failure:
                    # the metric is the curvature at one point: first take it at y (where the step starts), then halve
                    # (free here: the candidate xn, its prediction Axn, and Ay, whose gradient is already in grad)
                    op.direct_cl(y, Axn)
                    self._metric(Axn, Ay, xn, D)
                    local, theta = True, 1.0
                    n_metric += 1
                    continue
                theta *= 0.5
                if n_bt > self.max_backtracks:
                    raise RuntimeError(f"FISTAPoisson: no sufficient decrease after {n_bt} step reductions")
            n_bt_total += n_bt
            F_n = f_n + self.lam * xsum
            restarted = F_n > F_x
            if restarted:  # function-value restart
                t_mom = 1.0
                n_restart += 1
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_mom * t_mom))
            beta = np.float32((t_mom - 1.0) / t_new)
            t_mom = t_new
            # y = xn + beta (xn - x), A y likewise; then x <- xn
            self.k["extrapolate"](q, gx, None, y.data, xn.data, x.data, beta, nx)
            self.k["extrapolate"](q, gd, None, Ay.data, Axn.data, Ax.data, beta, nd)
            self.k["copy_buf"](q, gx, None, xn.data, x.data, nx)
            Ax, Axn = Axn, Ax
            f_x, F_x = f_n, F_n
            switch = self._mode == 1 and it >= wls_until
            if switch:  # end of the start phase: the deviance from here
                self._mode = 0
                f_x = self._data_term(Ax, False)
                F_x = f_x + self.lam * xsum
            refreshed = (it in self.refresh or switch) and k + 1 < niter and self._mode == 0
            if refreshed:  # metric at the current prediction, momentum restart
                self._metric(Ax, Axn, xn, D)
                t_mom, theta = 1.0, 1.0
                self.k["copy_buf"](q, gx, None, x.data, y.data, nx)
                self.k["copy_buf"](q, gd, None, Ax.data, Ay.data, nd)
            elif it % 50 == 0:  # float32 drift of the combined A y: recompute it
                op.direct_cl(y, Ay)
            stats = {"iter": it, "f": f_n, "obj": F_n, "phase": "wls" if it <= wls_until else "poisson",
                     "theta": theta, "backtracks": n_bt, "local_metric": local,
                     "restart": bool(restarted),
                     "metric_refresh": bool(refreshed), "time": time.perf_counter() - t_start}
            self.iter_stats.append(stats)
            stop = bool(callback(it, x, stats)) if callback is not None else False
            if verbose and (k == 0 or (k + 1) % diagnostics_interval == 0 or k + 1 == niter or stop):
                print(f"[iter {it:5d}] f={f_n:.8e} obj={F_n:.8e} theta={theta:.3g} backtracks={n_bt_total} "
                      f"metric updates={n_metric} restarts={n_restart} {time.perf_counter() - t_start:.1f} s",
                      flush=True)
            if stop:
                break

        q.finish()
        done = len(self.iter_stats)
        self.state = {"it": it0 + done, "t": t_mom, "theta": theta, "y": y.get(), "wls_until": wls_until}
        self.final_stats = {"niter": done, "final_f": f_x, "final_obj": F_x, "backtracks": n_bt_total,
                            "metric_updates_on_failure": n_metric,
                            "restarts": n_restart, "metric_setup_s": t_metric,
                            "time": time.perf_counter() - t_start}
        del xn, grad, D, Ax, Axn, Ay, y, uploaded
        self.b = self.w = self.g = self.bg = self._part = None
        if host_x0 is not None:
            x.get(ary=host_x0)
            return host_x0
        return x
