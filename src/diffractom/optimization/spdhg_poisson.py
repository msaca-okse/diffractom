"""The Poisson likelihood problem of FISTAPoisson / PDHGPoisson solved with stochastic primal-dual hybrid gradient
(SPDHG; Chambolle, Ehrhardt, Richtarik & Schoenlieb 2018) over subsets of the rotation angles, with the diagonal
preconditioning of Ehrhardt et al. (2019, PET) (SPDHGPoisson).

The data are split into M subsets of angles (operator.set_omega_subsets); F = sum_m F_m. Iteration (theta = 1):

    x     <- max(0, x - T (z_bar + lam))                         (support)
    m     <- the next subset (a random permutation every epoch; p_m = 1 / M)
    y_m'  <- prox_{S F_m*}(y_m + S A_m x)                        (rows of subset m, closed form)
    dz    <- A_m^T (y_m' - y_m);  y_m <- y_m';  z <- z + dz;  z_bar <- z + dz / p_m

with S_i = gamma / (A 1)_i and T_j = min_m p_m / (gamma (A_m^T 1_counted)_j), gamma = sqrt(rho). z = A^T y is kept up to
date, so an iteration costs one subset forward and one subset adjoint: an epoch (M iterations) costs about one
iteration of PDHGPoisson, and visits every data point once. rho as in PDHGPoisson ("auto": 3e4 / mean(b)^2).

Memory (GPU): five coefficient-sized arrays (x, z, z_bar, T, A_m^T dy) and four data-sized ones (data, dual, S, the
subset prediction / dual change). The objective of the full data needs a full forward: it is computed every
diagnostics_interval epochs only (and at the end).
"""
import time

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

from .fista_poisson import KERNELS as POISSON_KERNELS, _periodic
from .launch import elementwise, reduction
from .pdhg_poisson import KERNELS as PDHG_KERNELS, RHO_AUTO
from .prox import support_mask_to_gpu
from ..utils.arrays import as_host, check_device

KERNELS = r"""
inline float pv2(__global const float *a, const ulong p, const ulong i, const float dflt){
    return p ? a[i % p] : dflt;
}

// on the rows of subset omegas: y' = prox_{S F*}(y + S Ax), dy = y' - y, y = y'   (as dual_update of PDHGPoisson)
__kernel void dual_update_sub(__global float *y, __global float *dy, __global const float *Ax,
                              __global const float *sig, __global const float *b, __global const float *w,
                              const ulong pw, __global const float *g, const ulong pg, __global const float *bg,
                              const ulong pbg, const float eps, __global const int *omegas, const ulong row,
                              const ulong n)
{
    for (size_t t = get_global_id(0); t < n; t += get_global_size(0)) {
        const size_t s = t / row;
        const size_t i = (size_t)omegas[s] * row + (t - s * row);
        const float wi = pv2(w, pw, i, 1.0f);
        const float si = sig[i];
        float yn = 0.0f;
        if (wi != 0.0f && si != 0.0f) {
            const float gi = pv2(g, pg, i, 1.0f);
            const double sc = (double)wi / (double)gi;
            const double r = (double)pv2(bg, pbg, i, 0.0f) + (double)eps * gi;
            const double v = (double)y[i] + (double)si * Ax[i] + (double)si * r;
            const double d = v - sc;
            yn = (float)(0.5 * (v + sc - sqrt(d * d + 4.0 * (double)si * sc * (double)b[i])));
        }
        dy[i] = yn - y[i];
        y[i] = yn;
    }
}

// x = max(0, x - T (zb + lam)) (0 where T = 0)
__kernel void primal_spdhg(__global float *x, __global const float *zb, __global const float *T, const float lam,
                           const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const float t = T[i];
        x[i] = t > 0.0f ? fmax(0.0f, x[i] - t * (zb[i] + lam)) : 0.0f;
    }
}

// z += dz; zb = z + dz / p
__kernel void update_z(__global float *z, __global float *zb, __global const float *dz, const float inv_p,
                       const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const float zi = z[i] + dz[i];
        z[i] = zi;
        zb[i] = zi + inv_p * dz[i];
    }
}

// T = min(T, c / colsum) on the support where colsum > 0 (T starts at +inf; c = p_m / gamma)
__kernel void tmin(__global float *T, __global const float *colsum, __global const uchar *supp, const ulong npix,
                   const float c, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const int inside = npix ? supp[i % npix] : 1;
        if (inside && colsum[i] > 0.0f) T[i] = fmin(T[i], c / colsum[i]);
    }
}

// T = 0 where it is still +inf (no counted data point sees the coefficient, or outside the support)
__kernel void tfinal(__global float *T, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) if (isinf(T[i])) T[i] = 0.0f;
}
"""


class _K:
    def __init__(self, prg, names):
        for nm in names:
            setattr(self, nm, cl.Kernel(prg, nm))


class SPDHGPoisson:
    """SPDHG over omega subsets for the Poisson deviance (module docstring); coefficients on the GPU."""

    def __init__(self, operator, subsets=8, prox_kind="nonneg", lam=0.0, support="fov", eps_counts=1e-3, rho="auto",
                 seed=0):
        """subsets: the number of interleaved angle subsets, or a list of angle-index arrays (both passed to
        operator.set_omega_subsets, which needs the native projector and a sparse PF mode); seed: the order of the
        subsets within each epoch."""
        if prox_kind not in ("nonneg", "nonneg_l1"):
            raise ValueError("SPDHGPoisson supports prox_kind 'nonneg' and 'nonneg_l1'")
        self.op = operator
        self.ctx, self.queue = operator.ctx, operator.queue
        self.subsets = operator.set_omega_subsets(subsets)
        self.M = len(self.subsets)
        self.lam = float(lam) if prox_kind == "nonneg_l1" else 0.0
        self.eps_counts = float(eps_counts)
        self.rho = rho if rho == "auto" else float(rho)
        self.rng = np.random.default_rng(seed)
        self.support_gpu = support_mask_to_gpu(self.queue, operator, support)
        self.k = _K(cl.Program(self.ctx, KERNELS).build(),
                    ("dual_update_sub", "primal_spdhg", "update_z", "tmin", "tfinal"))
        self.pk = _K(cl.Program(self.ctx, POISSON_KERNELS).build(),
                     ("poisson_value_grad", "support_ones", "project_support", "copy_buf"))
        self.dk = _K(cl.Program(self.ctx, PDHG_KERNELS).build(), ("dual_steps", "set_mask_ones", "mean_counted"))
        self.state = None

    def _deviance(self, mu):
        g, l, n, ng = reduction(mu.size)
        self.pk.poisson_value_grad(self.queue, g, l, mu.data, self.b.data, self.w.data, np.uint64(self.pw),
                                   self.g.data, np.uint64(self.pg), self.bg.data, np.uint64(self.pbg),
                                   np.float32(self.eps_counts), np.int32(0), np.int32(0), np.float32(1.0), n,
                                   self._part.data, cl.LocalMemory(8 * l[0]))
        return float(self._part[:ng].get().sum())

    def run(self, x0, b, n_epochs, weights=None, gain=None, background=None, verbose=0, diagnostics_interval=1,
            callback=None, state=None):
        """As PDHGPoisson.run, in epochs (n_epochs passes over all subsets). A NumPy x0 is uploaded (the coefficients
        stay on the GPU) and the result copied back. callback(epoch, x, stats) after every epoch (a true return
        value stops the run). state: solver.state of an earlier run (z, z_bar, the dual variable) to continue it."""
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
        self._part = clarray.empty(q, (2 * ng,), np.float64)
        npix = np.uint64(self.support_gpu.size) if self.support_gpu is not None else np.uint64(0)
        supp = self.support_gpu.data if self.support_gpu is not None else dummy_u8.data
        gx, nx = elementwise(x.size)
        gd, nd = elementwise(int(np.prod(dshape)))
        row = np.uint64(dshape[1] * dshape[2])

        rho = self.rho
        if rho == "auto":
            rg, rl, rn, rng_ = reduction(int(np.prod(dshape)))
            self.dk.mean_counted(q, rg, rl, self.b.data, self.w.data, np.uint64(self.pw), rn, self._part.data,
                                 cl.LocalMemory(2 * 8 * rl[0]))
            sb, sn = self._part[:2 * rng_].get().reshape(2, rng_).sum(axis=1)
            mean_b = sb / max(sn, 1.0)
            rho = RHO_AUTO / mean_b ** 2 if mean_b > 0 else 1.0
        self.rho_used = float(rho)
        gamma = np.sqrt(rho)
        p = 1.0 / self.M

        z = clarray.zeros(q, cshape, np.float32)
        zb = clarray.zeros(q, cshape, np.float32)
        T = clarray.empty(q, cshape, np.float32)
        dz = clarray.empty(q, cshape, np.float32)
        ydual = clarray.zeros(q, dshape, np.float32)
        sig = clarray.empty(q, dshape, np.float32)
        Ax = clarray.empty(q, dshape, np.float32)
        dy = clarray.empty(q, dshape, np.float32)
        # S from the row sums A 1_S; T from the subsets' column sums over the counted points
        self.pk.support_ones(q, gx, None, dz.data, supp, npix, nx)
        op.direct_cl(dz, Ax)
        self.dk.dual_steps(q, gd, None, sig.data, Ax.data, self.w.data, np.uint64(self.pw), np.float32(gamma), nd)
        self.dk.set_mask_ones(q, gd, None, Ax.data, self.w.data, np.uint64(self.pw), nd)
        T.fill(np.float32(np.inf))
        for m in range(self.M):
            op.adjoint_cl(Ax, dz, subset=m)
            self.k.tmin(q, gx, None, T.data, dz.data, supp, npix, np.float32(p / gamma), nx)
        self.k.tfinal(q, gx, None, T.data, nx)
        self.pk.project_support(q, gx, None, x.data, supp, npix, nx)
        if state is not None:
            z.set(as_host(state["z"], cshape, "state['z']"))
            zb.set(as_host(state["z_bar"], cshape, "state['z_bar']"))
            ydual.set(as_host(state["y"], dshape, "state['y']"))
        t_setup = time.perf_counter() - t_start
        ep0 = int(state["epoch"]) if state is not None else 0
        self.iter_stats = []
        f = float("nan")
        for e in range(n_epochs):
            ep = ep0 + e + 1
            for m in self.rng.permutation(self.M):
                self.k.primal_spdhg(q, gx, None, x.data, zb.data, T.data, np.float32(self.lam), nx)
                op.direct_cl(x, Ax, subset=int(m))
                s = self.subsets[m]
                n_sub = np.uint64(len(s) * int(row))
                self.k.dual_update_sub(q, (min(int(n_sub), 1 << 24),), None, ydual.data, dy.data, Ax.data, sig.data,
                                       self.b.data, self.w.data, np.uint64(self.pw), self.g.data, np.uint64(self.pg),
                                       self.bg.data, np.uint64(self.pbg), np.float32(self.eps_counts),
                                       op._sub_omegas[m].data, row, n_sub)
                op.adjoint_cl(dy, dz, subset=int(m))
                self.k.update_z(q, gx, None, z.data, zb.data, dz.data, np.float32(1.0 / p), nx)
            want_f = (e + 1) % max(1, diagnostics_interval) == 0 or e + 1 == n_epochs
            if want_f:
                op.direct_cl(x, Ax)
                f = self._deviance(Ax)
            q.finish()
            stats = {"epoch": ep, "f": f if want_f else None, "rho": self.rho_used,
                     "time": time.perf_counter() - t_start}
            self.iter_stats.append(stats)
            stop = bool(callback(ep, x, stats)) if callback is not None else False
            if verbose and want_f:
                print(f"[spdhg epoch {ep:4d}] f={f:.8e} {time.perf_counter() - t_start:.1f} s", flush=True)
            if stop:
                break
        q.finish()
        self.state = {"epoch": ep0 + len(self.iter_stats), "z": z.get(), "z_bar": zb.get(), "y": ydual.get()}
        self.final_stats = {"epochs": len(self.iter_stats), "final_f": f, "setup_s": t_setup, "rho": self.rho_used,
                            "subsets": self.M, "time": time.perf_counter() - t_start}
        for a in (z, zb, T, dz, ydual, sig, Ax, dy):
            a.base_data.release()
        self.b = self.w = self.g = self.bg = self._part = None
        if host_x0 is not None:
            x.get(ary=host_x0)
            return host_x0
        return x
