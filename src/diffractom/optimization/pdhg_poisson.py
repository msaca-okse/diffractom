"""The Poisson likelihood problem of FISTAPoisson solved with the primal-dual hybrid gradient method of Chambolle &
Pock (2011) with the diagonal preconditioning of Pock & Chambolle (2011) (PDHGPoisson).

    min_x  sum_i s_i [mu_i' - b_i log mu_i'] + lam sum(x),  x >= 0 (in the support),   mu' = A x + bg + eps_counts g,
    s = w / g

Iteration (theta = 1):
    y     <- prox_{Sigma F*}(y + Sigma A x_bar)                  (closed form, one quadratic root per data point)
    x_new <- max(0, x - T (A^T y + lam))
    x_bar <- 2 x_new - x

with T_j = tau0 / (A^T 1)_j (column sums) and Sigma_i = sigma0 / (A 1_S)_i (row sums), tau0 sigma0 = 1 (rho = sigma0 /
tau0 balances the primal and dual steps). The dual step handles the likelihood's barrier at mu = 0 exactly, so there
is no line search and no Lipschitz constant; a positive multiple of the transpose as the adjoint cancels in T A^T.
One forward (A x_bar) and one adjoint (A^T y) per iteration; A x of the iterate, for the objective, follows from
A x_k = (A x_bar_k + A x_(k-1)) / 2. Memory (GPU): four coefficient-sized arrays (x, x_bar, T, A^T y) and five
data-sized ones (data, dual, Sigma, A x_bar, A x).
"""
import time

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

from .fista_poisson import KERNELS as POISSON_KERNELS, _periodic
from .launch import elementwise, reduction
from .prox import support_mask_to_gpu
from ..utils.arrays import as_host, check_device

RHO_AUTO = 3e4

KERNELS = r"""
inline float pv(__global const float *a, const ulong p, const ulong i, const float dflt){
    return p ? a[i % p] : dflt;
}

// y <- prox_{sig F*}(y + sig Axb): for F_i(z) = s (z + r - b log(z + r)), s = w / g, r = bg + eps g:
// y = (v + sig r + s - sqrt((v + sig r - s)^2 + 4 sig s b)) / 2 with v = y + sig Axb; 0 where s = 0
__kernel void dual_update(__global float *y, __global const float *Axb, __global const float *sig,
                          __global const float *b, __global const float *w, const ulong pw, __global const float *g,
                          const ulong pg, __global const float *bg, const ulong pbg, const float eps, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const float wi = pv(w, pw, i, 1.0f);
        const float si = sig[i];
        if (wi == 0.0f || si == 0.0f) { y[i] = 0.0f; continue; }
        const float gi = pv(g, pg, i, 1.0f);
        const double s = (double)wi / (double)gi;
        const double r = (double)pv(bg, pbg, i, 0.0f) + (double)eps * gi;
        const double v = (double)y[i] + (double)si * Axb[i] + (double)si * r;
        const double d = v - s;
        y[i] = (float)(0.5 * (v + s - sqrt(d * d + 4.0 * (double)si * s * (double)b[i])));
    }
}

// x_new = max(0, x - T (grad + lam)) (0 where T = 0, outside the support); xb = 2 x_new - x; x = x_new
__kernel void primal_update(__global float *x, __global float *xb, __global const float *grad,
                            __global const float *T, const float lam, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const float t = T[i];
        const float xo = x[i];
        const float xn = t > 0.0f ? fmax(0.0f, xo - t * (grad[i] + lam)) : 0.0f;
        xb[i] = 2.0f * xn - xo;
        x[i] = xn;
    }
}

// Ax <- (Axb + Ax) / 2   (A x_k from A x_bar_k = 2 A x_k - A x_(k-1))
__kernel void average(__global float *Ax, __global const float *Axb, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) Ax[i] = 0.5f * (Axb[i] + Ax[i]);
}

// sig = sigma0 / rowsum where rowsum > 0 and the point counts (w != 0), else 0
__kernel void dual_steps(__global float *sig, __global const float *rowsum, __global const float *w, const ulong pw,
                         const float sigma0, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0))
        sig[i] = (pv(w, pw, i, 1.0f) != 0.0f && rowsum[i] > 0.0f) ? sigma0 / rowsum[i] : 0.0f;
}

// T = tau0 / colsum on the support where colsum > 0
__kernel void primal_steps(__global float *T, __global const float *colsum, __global const uchar *supp,
                           const ulong npix, const float tau0, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const int inside = npix ? supp[i % npix] : 1;
        T[i] = (inside && colsum[i] > 0.0f) ? tau0 / colsum[i] : 0.0f;
    }
}

// partial sums of b and of the count of points with w != 0 (the mean of the counted data)
__kernel void mean_counted(__global const float *b, __global const float *w, const ulong pw, const ulong n,
                           __global double *partial, __local double *scratch)
{
    double s = 0.0, c = 0.0;
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0))
        if (pv(w, pw, i, 1.0f) != 0.0f) { s += (double)b[i]; c += 1.0; }
    const int lid = get_local_id(0), ls = get_local_size(0);
    scratch[lid] = s; scratch[ls + lid] = c;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int h = ls / 2; h > 0; h >>= 1) {
        if (lid < h) { scratch[lid] += scratch[lid + h]; scratch[ls + lid] += scratch[ls + lid + h]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) { partial[get_group_id(0)] = scratch[0]; partial[get_num_groups(0) + get_group_id(0)] = scratch[ls]; }
}

__kernel void set_mask_ones(__global float *u, __global const float *w, const ulong pw, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) u[i] = pv(w, pw, i, 1.0f) != 0.0f ? 1.0f : 0.0f;
}
"""


class PDHGPoisson:
    """Chambolle-Pock for the Poisson deviance (module docstring): an alternative to FISTAPoisson with the same model
    and arguments (run(x0, b, niter, weights, gain, background, ...)); coefficients on the GPU."""

    def __init__(self, operator, prox_kind="nonneg", lam=0.0, support="fov", eps_counts=1e-3, rho="auto"):
        """rho = sigma0 / tau0 (tau0 sigma0 = 1): larger rho, larger dual steps and smaller primal steps. "auto"
        (default): rho = RHO_AUTO / mean(b)^2, the mean over the counted data points (the primal variable scales with
        the data, the dual one does not). RHO_AUTO = 3e4 is calibrated for SinglePhaseForwardOperator (normalized=True)
        on two datasets (10 deg simulation, data mean 0.19, best rho ~1e6 at two noise levels whose gains differ
        400-fold; Al1050, data mean 1376, best rho ~1e-2): a factor 100 from there was slower (larger rho) or unstable
        at low counts (smaller). It depends on the operator's scale as well (not only the data's), so other operators
        need their own value. A number sets it."""
        if prox_kind not in ("nonneg", "nonneg_l1"):
            raise ValueError("PDHGPoisson supports prox_kind 'nonneg' and 'nonneg_l1'")
        self.op = operator
        self.ctx, self.queue = operator.ctx, operator.queue
        self.lam = float(lam) if prox_kind == "nonneg_l1" else 0.0
        self.eps_counts = float(eps_counts)
        self.rho = rho if rho == "auto" else float(rho)
        self.support_gpu = support_mask_to_gpu(self.queue, operator, support)
        class _K:  # kernels retrieved once (each attribute access of a Program builds a new kernel object)
            def __init__(self, prg, names):
                for nm in names:
                    setattr(self, nm, cl.Kernel(prg, nm))
        self.prg = _K(cl.Program(self.ctx, KERNELS).build(),
                      ("dual_update", "primal_update", "average", "dual_steps", "primal_steps", "set_mask_ones",
                       "mean_counted"))
        self.pk = _K(cl.Program(self.ctx, POISSON_KERNELS).build(),
                     ("poisson_value_grad", "support_ones", "project_support", "copy_buf"))
        self.state = None

    def _deviance(self, mu):
        g, l, n, ng = reduction(mu.size)
        self.pk.poisson_value_grad(self.queue, g, l, mu.data, self.b.data, self.w.data, np.uint64(self.pw),
                                   self.g.data, np.uint64(self.pg), self.bg.data, np.uint64(self.pbg),
                                   np.float32(self.eps_counts), np.int32(0), np.int32(0), np.float32(1.0), n,
                                   self._part.data, cl.LocalMemory(8 * l[0]))
        return float(self._part[:ng].get().sum())

    def run(self, x0, b, niter, weights=None, gain=None, background=None, verbose=0, diagnostics_interval=10,
            callback=None, state=None):
        """As FISTAPoisson.run (GPU-resident: a NumPy x0 is uploaded and the result copied back). state: the
        solver.state of an earlier run (x_bar and the dual variable) to continue it."""
        t_start = time.perf_counter()
        q, op = self.queue, self.op
        cshape, dshape = tuple(op.coeff_shape), tuple(op.data_shape)
        uploaded, host_x0 = [], None
        if isinstance(x0, clarray.Array):
            x = check_device(x0, cshape, "x0")
        else:
            host_x0 = as_host(x0, cshape, "x0")
            x = clarray.to_device(q, host_x0)
        self.b = check_device(b, dshape, "b") if isinstance(b, clarray.Array) else \
            clarray.to_device(q, as_host(b, dshape, "b"))
        dummy = clarray.zeros(q, (1,), np.float32)
        dummy_u8 = clarray.zeros(q, (1,), np.uint8)
        self.w, self.pw = _periodic(q, weights, dshape, "weights", uploaded)
        self.g, self.pg = _periodic(q, gain, dshape, "gain", uploaded)
        self.bg, self.pbg = _periodic(q, background, dshape, "background", uploaded)
        self.w = self.w if self.w is not None else dummy
        self.g = self.g if self.g is not None else dummy
        self.bg = self.bg if self.bg is not None else dummy
        _, _, _, ng = reduction(max(int(np.prod(dshape)), int(np.prod(cshape))))
        self._part = clarray.empty(q, (ng,), np.float64)
        self._part2 = clarray.empty(q, (2 * ng,), np.float64)
        npix = np.uint64(self.support_gpu.size) if self.support_gpu is not None else np.uint64(0)
        supp = self.support_gpu.data if self.support_gpu is not None else dummy_u8.data
        gx, nx = elementwise(x.size)
        gd, nd = elementwise(int(np.prod(dshape)))

        xb = clarray.empty(q, cshape, np.float32)
        T = clarray.empty(q, cshape, np.float32)
        grad = clarray.empty(q, cshape, np.float32)
        ydual = clarray.zeros(q, dshape, np.float32)
        sig = clarray.empty(q, dshape, np.float32)
        Axb = clarray.empty(q, dshape, np.float32)
        Ax = clarray.empty(q, dshape, np.float32)
        rho = self.rho
        if rho == "auto":
            rg, rl, rn, rng_ = reduction(int(np.prod(dshape)))
            self.prg.mean_counted(q, rg, rl, self.b.data, self.w.data, np.uint64(self.pw), rn, self._part2.data,
                                  cl.LocalMemory(2 * 8 * rl[0]))
            sb, sn = self._part2[:2 * rng_].get().reshape(2, rng_).sum(axis=1)
            mean_b = sb / max(sn, 1.0)
            rho = RHO_AUTO / mean_b ** 2 if mean_b > 0 else 1.0
        self.rho_used = float(rho)
        tau0, sigma0 = 1.0 / np.sqrt(rho), np.sqrt(rho)
        # row sums A 1_S (Sigma) and column sums A^T 1 over the counted data points (T)
        self.pk.support_ones(q, gx, None, xb.data, supp, npix, nx)
        op.direct_cl(xb, Axb)
        self.prg.dual_steps(q, gd, None, sig.data, Axb.data, self.w.data, np.uint64(self.pw), np.float32(sigma0), nd)
        self.prg.set_mask_ones(q, gd, None, Axb.data, self.w.data, np.uint64(self.pw), nd)
        op.adjoint_cl(Axb, grad)
        self.prg.primal_steps(q, gx, None, T.data, grad.data, supp, npix, np.float32(tau0), nx)
        self.pk.project_support(q, gx, None, x.data, supp, npix, nx)
        if state is not None:
            xb.set(as_host(state["x_bar"], cshape, "state['x_bar']"))
            ydual.set(as_host(state["y"], dshape, "state['y']"))
        else:
            self.pk.copy_buf(q, gx, None, x.data, xb.data, nx)
        op.direct_cl(x, Ax)
        t_setup = time.perf_counter() - t_start
        it0 = int(state["it"]) if state is not None else 0
        self.iter_stats = []
        f = self._deviance(Ax)
        for k in range(niter):
            it = it0 + k + 1
            op.direct_cl(xb, Axb)
            self.prg.dual_update(q, gd, None, ydual.data, Axb.data, sig.data, self.b.data, self.w.data,
                                 np.uint64(self.pw), self.g.data, np.uint64(self.pg), self.bg.data, np.uint64(self.pbg),
                                 np.float32(self.eps_counts), nd)
            op.adjoint_cl(ydual, grad)
            self.prg.primal_update(q, gx, None, x.data, xb.data, grad.data, T.data, np.float32(self.lam), nx)
            self.prg.average(q, gd, None, Ax.data, Axb.data, nd)  # A x of the new iterate
            if it % 50 == 0:  # float32 drift of the recursion
                op.direct_cl(x, Ax)
            f = self._deviance(Ax)
            stats = {"iter": it, "f": f, "rho": self.rho_used, "time": time.perf_counter() - t_start}
            self.iter_stats.append(stats)
            stop = bool(callback(it, x, stats)) if callback is not None else False
            if verbose and (k == 0 or (k + 1) % diagnostics_interval == 0 or k + 1 == niter or stop):
                print(f"[pdhg {it:5d}] f={f:.8e} {time.perf_counter() - t_start:.1f} s", flush=True)
            if stop:
                break
        q.finish()
        self.state = {"it": it0 + len(self.iter_stats), "x_bar": xb.get(), "y": ydual.get()}
        self.final_stats = {"niter": len(self.iter_stats), "final_f": f, "setup_s": t_setup, "rho": self.rho_used,
                            "time": time.perf_counter() - t_start}
        for a in (xb, T, grad, ydual, sig, Axb, Ax):
            a.base_data.release()
        self.b = self.w = self.g = self.bg = self._part = self._part2 = None
        if host_x0 is not None:
            x.get(ary=host_x0)
            return host_x0
        return x
