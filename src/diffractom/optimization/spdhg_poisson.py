"""The Poisson likelihood problem of FISTAPoisson / PDHGPoisson solved with stochastic primal-dual hybrid gradient
(SPDHG; Chambolle, Ehrhardt, Richtarik & Schoenlieb 2018) over subsets of the rotation angles, with the diagonal
preconditioning of Ehrhardt et al. (2019, PET) (SPDHGPoisson).

The data are split into M subsets of angles (operator.set_omega_subsets); F = sum_m F_m. Iteration (theta = 1), for
the next subset m (a random permutation every epoch, p_m = 1 / M):

    y_m'  <- prox_{S F_m*}(y_m + S A_m x)                        (rows of subset m, closed form)
    dz    <- A_m^T (y_m' - y_m);  y_m <- y_m';  z <- z + dz;  z_bar = z + dz / p_m
    x     <- max(0, x - T (z_bar + lam))                         (support)

with S_i = gamma / (A 1)_i and T_j = min_m p_m / (gamma (A_m^T 1_counted)_j), gamma = sqrt(rho). z = A^T y is kept up to
date, so an iteration costs one subset adjoint and one subset forward: an epoch (M iterations) costs about one
iteration of PDHGPoisson, and visits every data point once. rho as in PDHGPoisson ("auto": 3e4 / mean(b)^2).

One pass over the orientation batches per iteration: the adjoint of subset m, the updates of z and x, and the forward
of the next subset of the new x, batch by batch (z_bar and A_m^T dy are never stored). Two consecutive subsets
differ (an epoch does not start with the subset that ended the previous one), so the prediction of the next subset
and the dual change of the current one share one data-sized array.

Memory. GPU: four data-sized arrays (data, dual variable, S, prediction / dual change; a background of the data
shape is a fifth) and those of the coefficient arrays x, z, T that are resident there; the others stay in host
memory and are streamed through the GPU in every pass (x and z up and down, T up). resident="auto" keeps as many on
the GPU as fit, x first. The objective of the full data needs a full forward: it is computed every
diagnostics_interval epochs (and at the end), in a pass of its own.

Local basis (LocalSinglePhaseForwardOperator, flat coefficients): all coefficient arrays are on the GPU (they are
small), and an iteration is the subset adjoint, the update of the whole vector, and the next subset's forward (two
calls of the operator instead of one fused pass; the same work).

Warm starts: state (solver.state of an earlier run) holds the dual variable y and z = A^T y and continues the run (a
state without "z": z = A^T y recomputed). After the operator has changed (e.g. differentiable.set_pf_parameters),
start a new run from the current x without a state (y = 0): in the sandstone refinement the old dual variable made
the next run diverge (the spots had moved under it), a fresh one gave a steady descent. A dual start consistent
with x (y = F'(Ax)) diverged as well: where photons fall on a near-zero prediction it is huge.
"""
import time

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

from .fista_poisson import KERNELS as POISSON_KERNELS, _periodic
from .launch import elementwise, reduction
from .pdhg_poisson import KERNELS as PDHG_KERNELS, RHO_AUTO
from .prox import support_mask_to_gpu
from .streaming import _Pinned, batch_pass, host_map, streamer_for
from ..utils.arrays import as_host, check_device, supports_streaming
from ..utils.device_memory import free_device_bytes

KERNELS = r"""
inline float pv2(__global const float *a, const ulong p, const ulong i, const float dflt){
    return p ? a[i % p] : dflt;
}

// on the rows of subset omegas: d holds A x there on entry and the dual change y' - y on exit;
// y' = prox_{S F*}(y + S Ax), y = y'   (as dual_update of PDHGPoisson)
__kernel void dual_update_sub(__global float *y, __global float *d, __global const float *sig,
                              __global const float *b, __global const float *w, const ulong pw,
                              __global const float *g, const ulong pg, __global const float *bg, const ulong pbg,
                              const float eps, __global const int *omegas, const ulong row, const ulong n)
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
            const double v = (double)y[i] + (double)si * d[i] + (double)si * r;
            const double dd = v - sc;
            yn = (float)(0.5 * (v + sc - sqrt(dd * dd + 4.0 * (double)si * sc * (double)b[i])));
        }
        d[i] = yn - y[i];
        y[i] = yn;
    }
}

// one batch (n elements from offsets xo, zo, to): z += dz; z_bar = z + dz / p; x = max(0, x - T (z_bar + lam))
// (0 where T = 0)
__kernel void spdhg_update(__global float *x, const ulong xo, __global float *z, const ulong zo,
                           __global const float *T, const ulong to, __global const float *dz, const float inv_p,
                           const float lam, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const float zi = z[zo + i] + dz[i];
        z[zo + i] = zi;
        const float zb = zi + inv_p * dz[i];
        const float t = T[to + i];
        x[xo + i] = t > 0.0f ? fmax(0.0f, x[xo + i] - t * (zb + lam)) : 0.0f;
    }
}

__kernel void tinit(__global float *T, const ulong to, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) T[to + i] = INFINITY;
}

// T = min(T, c / colsum) on the support where colsum > 0 (c = p_m / gamma; a batch from offset to, a whole number
// of images)
__kernel void tmin(__global float *T, const ulong to, __global const float *colsum, __global const uchar *supp,
                   const ulong npix, const float c, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const int inside = npix ? supp[i % npix] : 1;
        if (inside && colsum[i] > 0.0f) T[to + i] = fmin(T[to + i], c / colsum[i]);
    }
}

// T = 0 where it is still +inf (no counted data point sees the coefficient, or outside the support)
__kernel void tfinal(__global float *T, const ulong to, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) if (isinf(T[to + i])) T[to + i] = 0.0f;
}
"""

NAMES = ("x", "z", "T")   # the coefficient-sized arrays, in the order resident="auto" keeps them on the GPU
MARGIN_GB = 1.5           # resident="auto": device memory left free


class _K:
    def __init__(self, prg, names):
        for nm in names:
            setattr(self, nm, cl.Kernel(prg, nm))


class _Coeff:
    """A coefficient-sized array: on the GPU (dev) or in host memory (flat host array, with staging buffers)."""

    def __init__(self, dev=None, host=None, pin_up=None, pin_dn=None, slots=None):
        self.dev, self.host, self.pin_up, self.pin_dn, self.slots = dev, host, pin_up, pin_dn, slots

    def buf(self, s, off):
        """(buffer, element offset) of the current batch: slot s, or offset off in the resident array."""
        return (self.dev.data, off) if self.dev is not None else (self.slots[s].data, 0)


class SPDHGPoisson:
    """SPDHG over omega subsets for the Poisson deviance (module docstring)."""

    def __init__(self, operator, subsets=8, prox_kind="nonneg", lam=0.0, support="fov", eps_counts=1e-3, rho="auto",
                 seed=0, resident="auto"):
        """subsets: the number of interleaved angle subsets, or a list of angle-index arrays (both passed to
        operator.set_omega_subsets, which needs the native or FFT projector and a sparse PF mode; also the
        local-basis operator). Angles in no subset are not fitted (e.g. held out for validation). seed: the order of
        the subsets within each epoch. rho: as PDHGPoisson's ("auto" = 3e4 / mean(b)^2). Measured (held-out
        deviance after 100 epochs, M 4-16): sandstone quartz rings (DanMAX 2026, local basis, gain ~20 data units
        per photon, ~3 photons per data point) best at 0.01-0.1 x auto, 0.15 % better than auto; the noise-free
        example data best at 1-100 x auto, ~3 % better than auto. M 4, 8, 16 within 0.02 %.

        resident: which of the coefficient arrays x, z (= A^T y) and T (the primal steps) stay on the GPU when x0
        is a NumPy array: "auto" (as many as fit in the free device memory, in this order), "all", "none" (all
        streamed from host memory), or a list of names. With a pyopencl x0 all three are on the GPU."""
        if prox_kind not in ("nonneg", "nonneg_l1"):
            raise ValueError("SPDHGPoisson supports prox_kind 'nonneg' and 'nonneg_l1'")
        if not (resident in ("auto", "all", "none") or set(resident) <= set(NAMES)):
            raise ValueError(f"resident must be 'auto', 'all', 'none' or a list of names from {NAMES}")
        self.op = operator
        self.ctx, self.queue = operator.ctx, operator.queue
        self.subsets = operator.set_omega_subsets(subsets)
        self.M = len(self.subsets)
        self.lam = float(lam) if prox_kind == "nonneg_l1" else 0.0
        self.eps_counts = float(eps_counts)
        self.rho = rho if rho == "auto" else float(rho)
        self.rng = np.random.default_rng(seed)
        self.resident = resident
        self._last = None   # the last subset of the previous epoch
        self.support_gpu = support_mask_to_gpu(self.queue, operator, support)
        self.k = _K(cl.Program(self.ctx, KERNELS).build(),
                    ("dual_update_sub", "spdhg_update", "tinit", "tmin", "tfinal"))
        self.pk = _K(cl.Program(self.ctx, POISSON_KERNELS).build(),
                     ("poisson_value_grad", "support_ones", "project_support"))
        self.dk = _K(cl.Program(self.ctx, PDHG_KERNELS).build(), ("dual_steps", "set_mask_ones", "mean_counted"))
        self.state = None

    # ---------------------------------------------------------------- helpers
    def _epoch_order(self):
        """The subsets of the next epoch: a random permutation that does not start with the last subset of the
        previous epoch (so that two consecutive subsets differ)."""
        perm = self.rng.permutation(self.M)
        if self.M > 1 and int(perm[0]) == self._last:
            perm[[0, 1]] = perm[[1, 0]]
        self._last = int(perm[-1])
        return [int(m) for m in perm]

    def _deviance(self, mu):
        g, l, n, ng = reduction(mu.size)
        self.pk.poisson_value_grad(self.queue, g, l, mu.data, self.b.data, self.w.data, np.uint64(self.pw),
                                   self.g.data, np.uint64(self.pg), self.bg.data, np.uint64(self.pbg),
                                   np.float32(self.eps_counts), np.int32(0), np.int32(0), np.float32(1.0), n,
                                   self._part.data, cl.LocalMemory(8 * l[0]))
        return float(self._part[:ng].get().sum())

    def _mean_counted(self, a):
        """Mean of a data-shaped GPU array over the counted points (weight != 0)."""
        rg, rl, rn, rng_ = reduction(int(a.size))
        self.dk.mean_counted(self.queue, rg, rl, a.data, self.w.data, np.uint64(self.pw), rn, self._part.data,
                             cl.LocalMemory(2 * 8 * rl[0]))
        sa, sn = self._part[:2 * rng_].get().reshape(2, rng_).sum(axis=1)
        return float(sa / max(sn, 1.0))

    def _gain_mean(self):
        return 1.0 if self.pg == 0 else float(clarray.sum(self.g).get()) / int(self.g.size)

    def _dual(self, m):
        """Dual update on the rows of subset m (A_m x in D there; afterwards the dual change)."""
        row = int(self.op.data_shape[1] * self.op.data_shape[2])
        n_sub = len(self.subsets[m]) * row
        self.k.dual_update_sub(self.queue, (min(n_sub, 1 << 24),), None, self.y.data, self.D.data, self.sig.data,
                               self.b.data, self.w.data, np.uint64(self.pw), self.g.data, np.uint64(self.pg),
                               self.bg.data, np.uint64(self.pbg), np.float32(self.eps_counts),
                               self.op._sub_omegas[m].data, np.uint64(row), np.uint64(n_sub))

    def _choose_resident(self, x_on_gpu, cshape):
        if x_on_gpu or not supports_streaming(self.op) or self.resident == "all":
            return set(NAMES)
        if self.resident == "none":
            return set()
        if self.resident != "auto":
            return set(self.resident)
        nbytes = 4 * int(np.prod(cshape))
        free = free_device_bytes(self.queue.device)
        if free is None:
            free = (self.queue.device.global_mem_size - self.op.device_bytes()
                    - 5 * 4 * int(np.prod(self.op.data_shape)))
        # staging: the streamer's 4 slots and 8 pinned buffers, T's 2 + 2 (NVIDIA backs pinned buffers with device
        # memory)
        staging = 16 * 4 * self.op.Nx * self.op.Ny * self.op.K_batch_max
        n = int(max(0, free - staging - MARGIN_GB * 1024**3) // nbytes)
        return set(NAMES[:min(n, len(NAMES))])

    def _pass(self, adj, fwd):
        """One pass over the orientation batches. adj (a subset whose dual change is in D, or None): its adjoint,
        z += it, x updated; fwd (a subset, "full" or None): the forward of the (new) x added to D."""
        op, q, A = self.op, self.queue, self.arrays
        npix, kmax = op.Nx * op.Ny, op.K_batch_max
        inv_p, lam = np.float32(self.M), np.float32(self.lam)
        sub = None if fwd == "full" else fwd
        x, z, T = A["x"], A["z"], A["T"]
        ups, dns = [], []
        if x.dev is None:
            ups.append((x.host, x.pin_up, x.slots))
            if adj is not None:
                dns.append((x.host, x.pin_dn, x.slots))
        if adj is not None:
            if z.dev is None:
                ups.append((z.host, z.pin_up, z.slots))
                dns.append((z.host, z.pin_dn, z.slots))
            if T.dev is None:
                ups.append((T.host, T.pin_up, T.slots))

        def compute(ib, s, n):
            k0 = op.batches[ib]["k_start"]
            off = k0 * npix
            if adj is not None:
                op._adjoint_batch(self.D, ib, self.dz, 0, kmax, subset=adj)
                (xb, xo), (zb, zo), (tb, to) = x.buf(s, off), z.buf(s, off), T.buf(s, off)
                g1, n1 = elementwise(n)
                self.k.spdhg_update(q, g1, None, xb, np.uint64(xo), zb, np.uint64(zo), tb, np.uint64(to),
                                    self.dz.data, inv_p, lam, n1)
            if fwd is not None:
                if x.dev is not None:
                    op._direct_batch(x.dev, k0, op.K, ib, self.D, sub)
                else:
                    op._direct_batch(x.slots[s], 0, kmax, ib, self.D, sub)

        self._run_pass(ups, dns, compute)

    def _run_pass(self, ups, dns, compute):
        op = self.op
        if ups or dns:
            batch_pass(self.st, ups, dns, compute)
        else:
            for ib, b in enumerate(op.batches):
                compute(ib, ib % 2, op.Nx * op.Ny * b["K_batch"])

    # ---------------------------------------------------------------- run
    def run(self, x0, b, n_epochs, weights=None, gain=None, background=None, verbose=0, diagnostics_interval=1,
            callback=None, state=None):
        """As PDHGPoisson.run, in epochs (n_epochs passes over all subsets). x0: a pyopencl array (all coefficient
        arrays on the GPU; returned, updated) or a NumPy array (updated in place if it is C-contiguous float32,
        else converted; the coefficient arrays not resident on the GPU, see `resident`, are streamed from host
        memory). callback(epoch, x, stats) after every epoch (a true return value stops the run; x is a host array
        for a NumPy x0). state: solver.state of an earlier run (z = A^T y and the dual variable y), to continue it
        from the x it returned (without "z": z recomputed as A^T y). After the operator has changed (e.g.
        differentiable.set_pf_parameters), start without a state (y = 0) from the current x: continuing the old dual
        variable diverged in tests (sandstone refinement), restarting it did not."""
        if getattr(self.op, "coefficient_layout", "dense") == "local":
            return self._run_local(x0, b, n_epochs, weights, gain, background, verbose, diagnostics_interval,
                                   callback, state)
        t_start = time.perf_counter()
        q, op = self.queue, self.op
        cshape, dshape = tuple(op.coeff_shape), tuple(op.data_shape)
        npix, kmax = op.Nx * op.Ny, op.K_batch_max
        uploaded = []
        x_on_gpu = isinstance(x0, clarray.Array)
        host_x = None if x_on_gpu else as_host(x0, cshape, "x0")
        x_dev = check_device(x0, cshape, "x0") if x_on_gpu else None
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
        _, _, _, ng = reduction(int(np.prod(dshape)))
        self._part = clarray.empty(q, (2 * ng,), np.float64)
        npx = np.uint64(self.support_gpu.size) if self.support_gpu is not None else np.uint64(0)
        supp = self.support_gpu.data if self.support_gpu is not None else dummy_u8.data
        gd, nd = elementwise(int(np.prod(dshape)))
        self.y = clarray.zeros(q, dshape, np.float32)
        self.sig = clarray.empty(q, dshape, np.float32)
        self.D = clarray.empty(q, dshape, np.float32)
        self.dz = clarray.empty(q, (npix * kmax,), np.float32)

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

        # the coefficient arrays: on the GPU or in host memory
        res = self._choose_resident(x_on_gpu, cshape)
        self.resident_used = tuple(n for n in NAMES if n in res)
        self.st = streamer_for(op) if len(res) < len(NAMES) else None
        extra = []   # T's staging buffers
        A = {}
        if "x" in res:
            if x_dev is None:
                x_dev = clarray.to_device(q, host_x)
            A["x"] = _Coeff(dev=x_dev)
        else:
            A["x"] = _Coeff(host=host_x.reshape(-1), pin_up=self.st.pin_up["x"], pin_dn=self.st.pin_dn["x"],
                            slots=self.st.xs)
        z_given = state is not None and state.get("z") is not None
        if "z" in res:
            zd = clarray.zeros(q, cshape, np.float32)
            if z_given:
                zd.set(as_host(state["z"], cshape, "state['z']"))
            A["z"] = _Coeff(dev=zd)
        else:
            zh = np.zeros(int(np.prod(cshape)), np.float32) if not z_given else \
                np.array(as_host(state["z"], cshape, "state['z']").reshape(-1))
            A["z"] = _Coeff(host=zh, pin_up=self.st.pin_up["y"], pin_dn=self.st.pin_dn["y"], slots=self.st.ys)
        if "T" in res:
            A["T"] = _Coeff(dev=clarray.empty(q, cshape, np.float32))
        else:
            tp = [_Pinned(self.ctx, q, npix * kmax) for _ in range(2)]
            ts = [clarray.empty(q, (npix * kmax,), np.float32) for _ in range(2)]
            extra = tp + ts
            A["T"] = _Coeff(host=np.empty(int(np.prod(cshape)), np.float32), pin_up=tp, pin_dn=self.st.pin_dn["y"],
                            slots=ts)
        self.arrays = A

        # S from the row sums A 1_S (batches of ones made on the GPU)
        self.D.fill(0.0)
        for ib, bt in enumerate(op.batches):
            g1, n1 = elementwise(npix * bt["K_batch"])
            self.pk.support_ones(q, g1, None, self.dz.data, supp, npx, n1)
            op._direct_batch(self.dz, 0, kmax, ib, self.D)
        self.scales = {"row_sum_mean": self._mean_counted(self.D), "b_mean": self._mean_counted(self.b),
                       "gain_mean": self._gain_mean()}
        self.dk.dual_steps(q, gd, None, self.sig.data, self.D.data, self.w.data, np.uint64(self.pw),
                           np.float32(gamma), nd)
        # T from the subsets' column sums over the counted points
        self.dk.set_mask_ones(q, gd, None, self.D.data, self.w.data, np.uint64(self.pw), nd)
        Tc = A["T"]

        def steps(ib, s, n):
            tb, to = Tc.buf(s, op.batches[ib]["k_start"] * npix)
            g1, n1 = elementwise(n)
            self.k.tinit(q, g1, None, tb, np.uint64(to), n1)
            for m in range(self.M):
                op._adjoint_batch(self.D, ib, self.dz, 0, kmax, subset=m)
                self.k.tmin(q, g1, None, tb, np.uint64(to), self.dz.data, supp, npx, np.float32(p / gamma), n1)
            self.k.tfinal(q, g1, None, tb, np.uint64(to), n1)
        self._run_pass([], [] if Tc.dev is not None else [(Tc.host, Tc.pin_dn, Tc.slots)], steps)
        # x on the support, >= 0
        if A["x"].dev is not None:
            gx, nx = elementwise(A["x"].dev.size)
            self.pk.project_support(q, gx, None, A["x"].dev.data, supp, npx, nx)
        else:
            xf = A["x"].host
            host_map(self.st, lambda a, b_: np.maximum(xf[a:b_], 0, out=xf[a:b_]), xf.size)
            if self.support_gpu is not None:
                inside = self.support_gpu.get().astype(bool).ravel()
                if not inside.all():
                    xf.reshape(op.K, npix)[:, ~inside] = 0.0
        if state is not None:
            self.y.set(as_host(state["y"], dshape, "state['y']"))
        if state is not None and not z_given:
            if A["z"].dev is not None:               # z = A^T y of this operator
                op.adjoint_cl(self.y, A["z"].dev)
            else:
                A["z"].host[:] = np.asarray(op.adjoint(self.y.get()), np.float32).reshape(-1)
        t_setup = time.perf_counter() - t_start

        ep0 = int(state["epoch"]) if state is not None else 0
        self.iter_stats = []
        f = float("nan")
        order = self._epoch_order()
        op.zero_subset_rows(self.D, order[0])
        self._pass(None, order[0])
        for e in range(n_epochs):
            ep = ep0 + e + 1
            last = e + 1 == n_epochs
            want_f = (e + 1) % max(1, diagnostics_interval) == 0 or last
            nxt = None
            for j, m in enumerate(order):
                self._dual(m)
                end_of_epoch = j + 1 == self.M
                if not end_of_epoch:
                    m_next = order[j + 1]
                elif not last:
                    nxt = self._epoch_order()
                    m_next = nxt[0]
                else:
                    m_next = None
                if (end_of_epoch and want_f) or m_next is None or m_next == m:
                    self._pass(m, None)              # adjoint, updates
                    if (end_of_epoch and want_f) or m_next is not None:
                        self.D.fill(0.0)
                        self._pass(None, "full")     # A x on every row (the next subset's among them)
                        if end_of_epoch and want_f:
                            f = self._deviance(self.D)
                else:
                    op.zero_subset_rows(self.D, m_next)
                    self._pass(m, m_next)            # adjoint, updates, forward of the next subset
            if nxt is not None:
                order = nxt
            q.finish()
            stats = {"epoch": ep, "f": f if want_f else None, "rho": self.rho_used,
                     "time": time.perf_counter() - t_start}
            self.iter_stats.append(stats)
            stop = False
            if callback is not None:
                if x_on_gpu or A["x"].dev is None:
                    xc = x_dev if x_on_gpu else host_x
                else:   # x resident for a NumPy x0: a host copy for the callback
                    A["x"].dev.get(ary=host_x)
                    xc = host_x
                stop = bool(callback(ep, xc, stats))
            if verbose and want_f:
                print(f"[spdhg epoch {ep:4d}] f={f:.8e} {time.perf_counter() - t_start:.1f} s", flush=True)
            if stop:
                break
        q.finish()
        z = A["z"]
        self.state = {"epoch": ep0 + len(self.iter_stats),
                      "z": z.dev.get() if z.dev is not None else z.host.reshape(cshape), "y": self.y.get()}
        self.final_stats = {"epochs": len(self.iter_stats), "final_f": f, "setup_s": t_setup, "rho": self.rho_used,
                            "subsets": self.M, "resident": self.resident_used, "time": time.perf_counter() - t_start,
                            **self.scales}
        if A["x"].dev is not None and not x_on_gpu:
            A["x"].dev.get(ary=host_x)
            A["x"].dev.base_data.release()
        for c in (A["z"], A["T"]):
            if c.dev is not None:
                c.dev.base_data.release()
        for a in extra:
            if isinstance(a, _Pinned):
                a.release()
            else:
                a.base_data.release()
        for a in (self.y, self.sig, self.D, self.dz):
            a.base_data.release()
        del uploaded
        self.b = self.w = self.g = self.bg = self._part = self.y = self.sig = self.D = self.dz = None
        self.arrays = self.st = None
        return x_dev if x_on_gpu else host_x

    # ---------------------------------------------------------------- local basis
    def _run_local(self, x0, b, n_epochs, weights, gain, background, verbose, diagnostics_interval, callback, state):
        """run() for an operator with flat coefficients (coefficient_layout "local"): every array on the GPU."""
        t_start = time.perf_counter()
        q, op = self.queue, self.op
        cshape, dshape = tuple(op.coeff_shape), tuple(op.data_shape)
        nc = int(np.prod(cshape))
        uploaded = []
        x_on_gpu = isinstance(x0, clarray.Array)
        host_x = None if x_on_gpu else as_host(x0, cshape, "x0")
        x = check_device(x0, cshape, "x0") if x_on_gpu else clarray.to_device(q, host_x)
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
        _, _, _, ng = reduction(int(np.prod(dshape)))
        self._part = clarray.empty(q, (2 * ng,), np.float64)
        gd, nd = elementwise(int(np.prod(dshape)))
        gc, ncc = elementwise(nc)
        self.y = clarray.zeros(q, dshape, np.float32)
        self.sig = clarray.empty(q, dshape, np.float32)
        self.D = clarray.empty(q, dshape, np.float32)
        z = clarray.zeros(q, cshape, np.float32)
        T = clarray.empty(q, cshape, np.float32)
        dz = clarray.empty(q, cshape, np.float32)

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
        self.resident_used = NAMES

        # S from the row sums A 1 (1 on the allowed coefficients), T from the subsets' column sums over the counted
        # points
        ones = clarray.to_device(q, np.ascontiguousarray(op.coefficient_mask(), np.float32))
        op.direct_cl(ones, self.D)
        ones.base_data.release()
        self.scales = {"row_sum_mean": self._mean_counted(self.D), "b_mean": self._mean_counted(self.b),
                       "gain_mean": self._gain_mean()}
        self.dk.dual_steps(q, gd, None, self.sig.data, self.D.data, self.w.data, np.uint64(self.pw),
                           np.float32(gamma), nd)
        self.dk.set_mask_ones(q, gd, None, self.D.data, self.w.data, np.uint64(self.pw), nd)
        self.k.tinit(q, gc, None, T.data, np.uint64(0), ncc)
        for m in range(self.M):
            op.adjoint_cl(self.D, dz, subset=m)
            self.k.tmin(q, gc, None, T.data, np.uint64(0), dz.data, dummy_u8.data, np.uint64(0),
                        np.float32(p / gamma), ncc)
        self.k.tfinal(q, gc, None, T.data, np.uint64(0), ncc)
        self.pk.project_support(q, gc, None, x.data, dummy_u8.data, np.uint64(0), ncc)
        if state is not None:
            self.y.set(as_host(state["y"], dshape, "state['y']"))
            if state.get("z") is not None:
                z.set(as_host(state["z"], cshape, "state['z']"))
            else:                                # z = A^T y of this operator
                op.adjoint_cl(self.y, z)
        t_setup = time.perf_counter() - t_start

        inv_p, lam = np.float32(self.M), np.float32(self.lam)
        ep0 = int(state["epoch"]) if state is not None else 0
        self.iter_stats = []
        f = float("nan")
        order = self._epoch_order()
        op.direct_cl(x, self.D, subset=order[0])
        for e in range(n_epochs):
            ep = ep0 + e + 1
            last = e + 1 == n_epochs
            want_f = (e + 1) % max(1, diagnostics_interval) == 0 or last
            nxt = None
            for j, m in enumerate(order):
                self._dual(m)
                op.adjoint_cl(self.D, dz, subset=m)
                self.k.spdhg_update(q, gc, None, x.data, np.uint64(0), z.data, np.uint64(0), T.data, np.uint64(0),
                                    dz.data, inv_p, lam, ncc)
                end_of_epoch = j + 1 == self.M
                if not end_of_epoch:
                    m_next = order[j + 1]
                elif not last:
                    nxt = self._epoch_order()
                    m_next = nxt[0]
                else:
                    m_next = None
                if end_of_epoch and want_f:
                    op.direct_cl(x, self.D)          # A x on every row (the next subset's among them)
                    f = self._deviance(self.D)
                elif m_next is not None:
                    op.direct_cl(x, self.D, subset=m_next)
            if nxt is not None:
                order = nxt
            q.finish()
            stats = {"epoch": ep, "f": f if want_f else None, "rho": self.rho_used,
                     "time": time.perf_counter() - t_start}
            self.iter_stats.append(stats)
            stop = False
            if callback is not None:
                if not x_on_gpu:
                    x.get(ary=host_x)
                stop = bool(callback(ep, x if x_on_gpu else host_x, stats))
            if verbose and want_f:
                print(f"[spdhg epoch {ep:4d}] f={f:.8e} {time.perf_counter() - t_start:.1f} s", flush=True)
            if stop:
                break
        q.finish()
        self.state = {"epoch": ep0 + len(self.iter_stats), "z": z.get(), "y": self.y.get()}
        self.final_stats = {"epochs": len(self.iter_stats), "final_f": f, "setup_s": t_setup, "rho": self.rho_used,
                            "subsets": self.M, "resident": self.resident_used, "time": time.perf_counter() - t_start,
                            **self.scales}
        if not x_on_gpu:
            x.get(ary=host_x)
            x.base_data.release()
        for a in (z, T, dz, self.y, self.sig, self.D):
            a.base_data.release()
        del uploaded
        self.b = self.w = self.g = self.bg = self._part = self.y = self.sig = self.D = None
        return x if x_on_gpu else host_x
