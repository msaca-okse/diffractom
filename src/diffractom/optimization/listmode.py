"""Solvers for the Poisson likelihood of raw detector counts in list mode (detector.RawDetectorModel):

    min_x>=0  <s, x> - sum_events n_e log(mu_e + r_e),    mu = B x,  s = B^T 1 (all cells),  r = background

ListModeSPDHG: stochastic primal-dual hybrid gradient over rotation-angle subsets with a dual variable per event
(the list-mode SPDHG of Schramm & Holler, NeuroImage 2022, for PET): the cells without counts have the dual value
of a zero count, which never changes, so they appear only through the constant gradient s. With F_e(u) = -n_e
log(u + r_e) the dual step is the closed form

    v = y + S (B x + r);   y' = (v - sqrt(v^2 + 4 S n)) / 2   (y <= 0),

and the primal step x <- max(0, x - T (z_bar + s + lam)), z = B^T y (z_bar = z + dz / p). Preconditioning as in
Ehrhardt et al. (2019): S_e = gamma / (B 1)_e, T_j = min_m p_m / (gamma (B_m^T 1_events)_j), gamma = sqrt(rho).
An iteration costs one subset forward and one subset adjoint of the operator; an epoch visits every event once.

ListModeMLEM: the EM algorithm (ordered subsets with subsets > 1, i.e. list-mode OSEM): x <- x B_m^T(n / mu) / s_m.
No step size to choose; it converges slowly (MLEM) or stops converging with many subsets (OSEM), and it cannot leave
a zero coefficient.

Memory: the coefficient-sized x, z, T, s on the GPU; the per-event y and S (8 bytes per event) with the model's
events (RawDetectorModel storage "gpu": on the GPU, next to the model's 12 bytes per event, a work array and the
data-sized M; "host": in host memory, streamed through the GPU in chunks).
"""
import time

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel

from ..utils.arrays import as_host

KERNELS = r"""
// events off .. off + n - 1; d holds B x on entry and the dual change on exit
__kernel void lm_dual(__global float *y, __global float *d, __global const float *S, __global const float *cnt,
                      __global const float *bg, const int has_bg, const float eps, const ulong off, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const size_t e = off + i;
        const float s = S[e];
        float yn = 0.0f;
        if (s > 0.0f) {
            const double r = (has_bg ? (double)bg[e] : 0.0) + (double)eps;
            const double v = (double)y[e] + (double)s * ((double)d[e] + r);
            yn = (float)(0.5 * (v - sqrt(v * v + 4.0 * (double)s * (double)cnt[e])));
        }
        d[e] = yn - y[e];
        y[e] = yn;
    }
}

// mu -> n / (mu + r) (0 where the model is 0)
__kernel void lm_ratio(__global float *mu, __global const float *cnt, __global const float *bg, const int has_bg,
                       const float eps, const ulong off, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const size_t e = off + i;
        const float m = mu[e] + (has_bg ? bg[e] : 0.0f) + eps;
        mu[e] = m > 0.0f ? cnt[e] / m : 0.0f;
    }
}

// S = gamma / (B 1) (0 where B 1 = 0); mu holds B 1
__kernel void lm_recip(__global float *S, __global const float *mu, const float g, const ulong off, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const size_t e = off + i;
        S[e] = mu[e] > 0.0f ? g / mu[e] : 0.0f;
    }
}

// mu -> 1 (the back-projection of ones)
__kernel void lm_ones(__global float *mu, const ulong off, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) mu[off + i] = 1.0f;
}

// mu -> n log(n / (mu + r)) (0 for n = 0)
__kernel void lm_loglik(__global float *mu, __global const float *cnt, __global const float *bg, const int has_bg,
                        const float eps, const ulong off, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const size_t e = off + i;
        const double c = cnt[e];
        const double m = fmax((double)mu[e] + (has_bg ? (double)bg[e] : 0.0) + (double)eps, 1e-30);
        mu[e] = c > 0.0 ? (float)(c * log(c / m)) : 0.0f;
    }
}
"""


def _launch(queue, kernel, n, *args):
    if n == 0:
        return
    ls = 256
    gs = min(int(n) + (-int(n)) % ls, ls * 4096)
    kernel.set_args(*args)
    cl.enqueue_nd_range_kernel(queue, kernel, (gs,), (ls,))


class _Common:
    def _setup(self, model, support):
        self.model = model
        self.op = model.op
        self.queue = model.queue
        q = self.queue
        ctx = model.ctx
        self.prg = cl.Program(ctx, KERNELS).build()
        self.kk = {n: cl.Kernel(self.prg, n) for n in ("lm_dual", "lm_ratio", "lm_recip", "lm_ones", "lm_loglik")}
        self._update = ElementwiseKernel(
            ctx, "float *x, float *z, const float *T, const float *dz, const float *s, float inv_p, float lam",
            "const float zi = z[i] + dz[i]; z[i] = zi; const float t = T[i];"
            "x[i] = t > 0.0f ? fmax(0.0f, x[i] - t * (zi + inv_p * dz[i] + s[i] + lam)) : 0.0f;", "lm_update")
        self._tmin = ElementwiseKernel(
            ctx, "float *T, const float *colsum, const float *supp, float c",
            "if (supp[i] > 0.0f && colsum[i] > 0.0f) T[i] = fmin(T[i], c / colsum[i]);", "lm_tmin")
        self._tfin = ElementwiseKernel(ctx, "float *T", "if (isinf(T[i])) T[i] = 0.0f;", "lm_tfin")
        self._em = ElementwiseKernel(ctx, "float *x, const float *g, const float *s, const float *supp",
                                     "x[i] = (supp[i] > 0.0f && s[i] > 0.0f) ? x[i] * g[i] / s[i] : 0.0f;", "lm_em")
        self._em_scaled = ElementwiseKernel(
            ctx, "float *x, const float *g, const float *s, const float *supp, float share",
            "x[i] = (supp[i] > 0.0f && s[i] > 0.0f) ? x[i] * g[i] / (share * s[i]) : 0.0f;", "lm_em_scaled")
        self._sum = ReductionKernel(ctx, np.float64, neutral="0", reduce_expr="a+b", map_expr="(double)v[i]",
                                    arguments="const float *v")
        shape = self.op.coeff_shape
        if support == "fov":
            m = np.broadcast_to(self.op.support_mask()[None].astype(np.float32), shape)
        elif support is None:
            m = np.ones(shape, np.float32)
        else:
            m = np.broadcast_to(np.asarray(support, np.float32), shape)
        self.supp = clarray.to_device(q, np.ascontiguousarray(m, dtype=np.float32))
        self.n_total = float(model.n_total) if hasattr(model, "n_total") else \
            float(model.events.count.astype(np.float64).sum())
        self.eps = 0.0

    def _sens(self):
        s = clarray.zeros(self.queue, self.op.coeff_shape, np.float32)
        self.model.sensitivity_cl(s)
        return s * self.supp

    def _bg(self, arr):
        if "bg" in arr:
            return arr["bg"].data, np.int32(1)
        return arr["count"].data, np.int32(0)

    def _sum_range(self, a, off, n):
        if off == 0 and n == a.size:
            return float(self._sum(a).get())
        return float(self._sum(a[off:off + n]).get())

    def objective(self, x, xb=None):
        """Poisson deviance of all cells at x (and background coefficients xb) (GPU, one full forward):
        <s, x> + <s_b, xb> + bg_total - sum n + sum_events n log(n / (mu + r))."""
        def loglik(arr, off, n):
            bgd, hb = self._bg(arr)
            _launch(self.queue, self.kk["lm_loglik"], n, arr["mu"].data, arr["count"].data, bgd, hb,
                    np.float32(self.eps), np.uint64(off), np.uint64(n))
        ll = self.model.event_values(x, loglik, lambda arr, off, n: self._sum_range(arr["mu"], off, n), beta=xb)
        if not hasattr(self, "s"):
            self.s = self._sens()
        lin = float(clarray.dot(self.s, x, dtype=np.float64).get())
        if getattr(self.model, "parallax", None) is not None:
            lin += self.model.parallax_total(x)
        if xb is not None:
            lin += float(clarray.dot(self.sb, xb, dtype=np.float64).get())
        return lin + self.model.background_total - self.n_total + ll

    def _bg_setup(self, beta0):
        """The background block (model with a background basis): coefficients on the GPU, from beta0 or a flat
        start that predicts a share of the counts."""
        m, q = self.model, self.queue
        if m.bgb is None:
            self.has_bg = False
            return
        self.has_bg = True
        self.sb = clarray.to_device(q, m.bg_sens.astype(np.float32))
        if beta0 is None:
            beta0 = np.zeros(m.bgb.shape, np.float32)
        self.xb = clarray.to_device(q, np.ascontiguousarray(beta0, dtype=np.float32).reshape(m.bgb.shape))

    @property
    def beta(self):
        return self.xb.get() if getattr(self, "has_bg", False) else None


class ListModeSPDHG(_Common):
    """List-mode SPDHG (module docstring).

    model : RawDetectorModel; subsets : number of rotation-angle subsets (model.set_subsets; None: keep the model's);
    rho : gamma^2 of the preconditioning, or "auto": gamma = rho_factor / x_ref with x_ref = sum(n) / sum(s)
        (the coefficient value of a uniform image that predicts the total counts). On the simulated samples
        rho_factor 0.1 was best (GT misorientation after 200 epochs: 0.208 deg vs 0.214 at 1 and 0.233 at 10);
    lam : l1 weight (>= 0); support : "fov" (the operator's field of view), None, or a mask of the coefficients;
    eps : added to the model in the dual step and the objective (counts); seed : the random subset order.
    """

    def __init__(self, model, subsets=8, rho="auto", rho_factor=0.1, lam=0.0, support="fov", eps=0.0, seed=0,
                 beta0=None):
        if subsets is not None:
            model.set_subsets(subsets)
        self._setup(model, support)
        self._bg_setup(beta0)
        self.lam = float(lam)
        self.eps = float(eps)
        self.rng = np.random.default_rng(seed)
        self.rho, self.rho_factor = rho, float(rho_factor)
        self.iter_stats = []
        self.state = None

    def _prepare(self):
        m, q = self.model, self.queue
        M = m.n_subsets
        self.p = 1.0 / M
        self.s = self._sens()
        s_sum = float(clarray.sum(self.s, dtype=np.float64).get())
        self.x_ref = self.n_total / max(s_sum, 1e-300)
        gamma = (self.rho_factor / self.x_ref) if self.rho == "auto" else float(np.sqrt(float(self.rho)))
        self.gamma = gamma
        self.y = m.event_array("y", 0.0)
        self.S = m.event_array("S", 0.0)
        col = clarray.zeros(q, self.op.coeff_shape, np.float32)
        T = clarray.zeros(q, self.op.coeff_shape, np.float32) + np.float32(np.inf)
        if self.has_bg:
            ones_b = clarray.zeros(q, m.bgb.shape, np.float32) + 1.0
            colb = clarray.zeros(q, m.bgb.shape, np.float32)
            Tb = clarray.zeros(q, m.bgb.shape, np.float32) + np.float32(np.inf)
            supp_b = ones_b
        else:
            ones_b = colb = None

        # one pass per subset: S = gamma / (B 1) from the forward of the support (and of the background block), and
        # the column sums of the subset's events (T_j = min_m p / (gamma (B_m^T 1)_j))
        def steps(arr, off, n):
            _launch(q, self.kk["lm_recip"], n, arr["S"].data, arr["mu"].data, np.float32(gamma), np.uint64(off),
                    np.uint64(n))
            _launch(q, self.kk["lm_ones"], n, arr["mu"].data, np.uint64(off), np.uint64(n))
        for sub in range(M):
            m.event_pass(self.supp, col, sub if M > 1 else None, steps, names=("S",), write_back=("S",),
                         beta=ones_b, beta_out=colb)
            self._tmin(T, col, self.supp, np.float32(self.p / gamma))
            if self.has_bg:
                self._tmin(Tb, colb, supp_b, np.float32(self.p / gamma))
        self._tfin(T)
        self.T = T
        self.dz = col
        if self.has_bg:
            self._tfin(Tb)
            self.Tb = Tb
            self.dzb = colb

    def run(self, x0, n_epochs, verbose=0, diagnostics_interval=1, callback=None, state=None):
        """n_epochs epochs (each subset once, in random order) from x0 (NumPy or GPU, coefficient-shaped).
        state: self.state of an earlier run (z = B^T y; the dual y stays with the model) to continue it.
        Returns x (NumPy for a NumPy x0). callback(epoch, x) after every epoch."""
        m, q = self.model, self.queue
        host = not isinstance(x0, clarray.Array)
        x = clarray.to_device(q, np.ascontiguousarray(x0, dtype=np.float32)) if host else x0.copy()
        x *= self.supp
        t0 = time.perf_counter()
        if not hasattr(self, "T"):
            self._prepare()
        M = m.n_subsets
        if state is not None:
            z = state["z"]
            epoch0 = int(state.get("epoch", 0))
        else:
            if isinstance(self.y, np.ndarray):
                self.y[:] = 0.0
            else:
                self.y.fill(0.0)
            z = clarray.zeros(q, self.op.coeff_shape, np.float32)
            epoch0 = 0
        if self.has_bg:
            zb = state["zb"] if (state is not None and "zb" in state) else clarray.zeros(q, m.bgb.shape, np.float32)
        self.setup_time = time.perf_counter() - t0
        eps = np.float32(self.eps)

        def dual(arr, off, n):
            bgd, hb = self._bg(arr)
            _launch(q, self.kk["lm_dual"], n, arr["y"].data, arr["mu"].data, arr["S"].data, arr["count"].data, bgd,
                    hb, eps, np.uint64(off), np.uint64(n))
        for ep in range(n_epochs):
            te = time.perf_counter()
            for sub in self.rng.permutation(M):
                sub = int(sub) if M > 1 else None
                if self.has_bg:
                    m.event_pass(x, self.dz, sub, dual, names=("y", "S"), write_back=("y",), beta=self.xb,
                                 beta_out=self.dzb)
                    self._update(self.xb, zb, self.Tb, self.dzb, self.sb, np.float32(1.0 / self.p), np.float32(0.0))
                else:
                    m.event_pass(x, self.dz, sub, dual, names=("y", "S"), write_back=("y",))
                self._update(x, z, self.T, self.dz, self.s, np.float32(1.0 / self.p), np.float32(self.lam))
            q.finish()
            stats = {"epoch": epoch0 + ep + 1, "time": time.perf_counter() - te, "f": None}
            last = ep == n_epochs - 1
            if diagnostics_interval and ((ep + 1) % diagnostics_interval == 0 or last):
                stats["f"] = self.objective(x, self.xb if self.has_bg else None)
                if verbose:
                    print(f"epoch {stats['epoch']}: deviance {stats['f']:.6e} ({stats['time']:.2f} s)", flush=True)
            self.iter_stats.append(stats)
            if callback is not None:
                callback(stats["epoch"], x)
        self.state = {"z": z, "epoch": epoch0 + n_epochs}
        if self.has_bg:
            self.state["zb"] = zb
        return x.get() if host else x


class ListModeMLEM(_Common):
    """List-mode EM / OSEM (module docstring). x0 must be positive where coefficients may become non-zero."""

    def __init__(self, model, subsets=1, support="fov", eps=0.0, beta0=None, subset_sensitivity="exact", fix_x=False):
        """subset_sensitivity: "exact" (B_m^T 1 per subset, one coefficient-sized array each) or "scaled" (the full
        sensitivity times the subset's share of the exposure: one array; for interleaved angle subsets they are
        nearly equal)."""
        self.subset_sensitivity = subset_sensitivity
        self.fix_x = bool(fix_x)          # only the background block is updated (x fixed)
        if subsets is not None:
            model.set_subsets(subsets)
        self._setup(model, support)
        self.eps = float(eps)
        self.iter_stats = []
        self._bg_setup(beta0)
        self.sbm = None
        if self.has_bg and beta0 is None:
            self.xb.fill(np.float32(1.0))

    def reset_sensitivity(self):
        """Recompute the sensitivities in the next run (after the model's weights changed, e.g. ring scales)."""
        for name in ("sm", "s"):
            if hasattr(self, name):
                delattr(self, name)

    def run(self, x0, n_epochs, verbose=0, diagnostics_interval=1, callback=None):
        m, q = self.model, self.queue
        host = not isinstance(x0, clarray.Array)
        x = clarray.to_device(q, np.ascontiguousarray(x0, dtype=np.float32)) if host else x0.copy()
        x *= self.supp
        M = m.n_subsets
        if not hasattr(self, "sm"):
            if hasattr(self, "s"):
                del self.s
            self.s = self._sens()
            self.sm = []
            expo = m.events.exposure.ravel().astype(np.float64)
            om = np.arange(len(expo)) // m.op.My
            for sub in range(M):
                if self.subset_sensitivity == "scaled":
                    self.sm.append(float(expo[np.isin(om, m.subsets[sub])].sum() / max(expo.sum(), 1e-300)))
                    continue
                a = clarray.zeros(q, self.op.coeff_shape, np.float32)
                m.sensitivity_cl(a, subset=sub if M > 1 else None)
                self.sm.append(a * self.supp)
            if not hasattr(self, "g"):
                self.g = clarray.zeros(q, self.op.coeff_shape, np.float32)
            if self.has_bg:
                self.sbm = [clarray.to_device(q, a.astype(np.float32)) for a in m.bg_sens_sub]
                if not hasattr(self, "gb"):
                    self.gb = clarray.zeros(q, m.bgb.shape, np.float32)
                    self.ones_b = clarray.zeros(q, m.bgb.shape, np.float32) + 1.0
        eps = np.float32(self.eps)

        def ratio(arr, off, n):
            bgd, hb = self._bg(arr)
            _launch(q, self.kk["lm_ratio"], n, arr["mu"].data, arr["count"].data, bgd, hb, eps, np.uint64(off),
                    np.uint64(n))
        for ep in range(n_epochs):
            te = time.perf_counter()
            for sub in range(M):
                sub_arg = sub if M > 1 else None
                if self.has_bg:
                    m.event_pass(x, self.g, sub_arg, ratio, beta=self.xb, beta_out=self.gb)
                    self._em(self.xb, self.gb, self.sbm[sub], self.ones_b)
                else:
                    m.event_pass(x, self.g, sub_arg, ratio)
                if not self.fix_x:
                    if isinstance(self.sm[sub], float):
                        self._em_scaled(x, self.g, self.s, self.supp, np.float32(self.sm[sub]))
                    else:
                        self._em(x, self.g, self.sm[sub], self.supp)
            q.finish()
            stats = {"epoch": len(self.iter_stats) + 1, "time": time.perf_counter() - te, "f": None}
            if diagnostics_interval and ((ep + 1) % diagnostics_interval == 0 or ep == n_epochs - 1):
                stats["f"] = self.objective(x, self.xb if self.has_bg else None)
                if verbose:
                    print(f"epoch {stats['epoch']}: deviance {stats['f']:.6e} ({stats['time']:.2f} s)", flush=True)
            self.iter_stats.append(stats)
            if callback is not None:
                callback(stats["epoch"], x)
        return x.get() if host else x
