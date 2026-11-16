"""The Poisson likelihood problem of FISTAPoisson / SPDHGPoisson solved with the EM algorithm (MLEM; with subsets of the
rotation angles, ordered-subsets EM, OSEM; Shepp & Vardi 1982, Hudson & Larkin 1994) (OSEMPoisson).

Model and loss as in FISTAPoisson: mu_i = (A x)_i + bg_i, data b_i with gain g_i (data units per count) and weights w_i,

    f(x) = sum_i w_i / g_i [ mu_i' - b_i + b_i log(b_i / mu_i') ],   mu_i' = max((A x)_i, 0) + bg_i + eps_counts g_i,

over x >= 0 (+ lam sum(x) with lam > 0). For the next subset m (a random permutation every epoch):

    x <- x A_m^T(w b / (g mu')) / (A_m^T(w / g) + lam)

The update is multiplicative: no step size to choose, x stays non-negative, and (without background and lam) the
total prediction on the fitted angles matches the data after every MLEM iteration. With M subsets an epoch costs about
one forward and one adjoint (as an SPDHG epoch) and advances about M times further than an MLEM iteration early on;
OSEM does not converge to the maximum-likelihood point (it cycles near it, more with more subsets), and a coefficient
that is zero stays zero (start from a positive x: start_floor).

When to use it (rock fits, DanMAX 2026): on dense problems where many orientations overlap in every column (phyllite:
micas and chlorite, local basis), SPDHG barely moved at any rho, while 10 OSEM epochs (8 subsets, 40 s) lowered the
held-out deviance 45.6 -> 41.7, more than 20 SPDHG epochs + 6 Gauss-Newton steps (42.06). Elsewhere SPDHGPoisson
remains the default. Without regularization EM (as every maximum-likelihood fit) puts material where the data barely
constrain it (basalt: into the vesicles); lam > 0 (an l1 penalty, as prox_kind="nonneg_l1" in the other solvers)
damps it. (A one-step-late TV, tried on the phyllite, did not change the held-out deviance; not included.)

Memory: x and the back-projection (coefficient-sized, on the GPU), one sensitivity A_m^T(w / g) per subset
(store_sensitivities=True; False recomputes it in every iteration: an extra subset adjoint), and three data-sized
arrays. The coefficients are not streamed: they must fit on the GPU with these arrays.
"""
import time

import numpy as np
import pyopencl.array as clarray
from pyopencl.elementwise import ElementwiseKernel

from .poisson_term import PoissonTerm
from .prox import support_mask_to_gpu
from ..utils.arrays import as_host, check_device


class OSEMPoisson:
    """Ordered-subsets EM for the Poisson deviance (module docstring)."""

    def __init__(self, operator, subsets=8, lam=0.0, support="fov", eps_counts=1e-3, seed=0,
                 store_sensitivities=True):
        """subsets: the number of interleaved angle subsets, or a list of angle-index arrays (passed to
        operator.set_omega_subsets, as SPDHGPoisson's; evaluation.interleaved_subsets builds them from a held-out
        split; angles in no subset are not fitted), or None: MLEM over all angles (subset=None in the operator calls;
        any operator with direct_cl / adjoint_cl; hold data out with zero weights). lam: weight of the l1 penalty
        lam sum(x) (photons per coefficient unit, as the deviance). support: "fov", None or an (Ny, Nx) array (dense
        layout, as in the other solvers); the local-basis operator uses its own allowed coefficients
        (coefficient_mask). seed: the order of the subsets within each epoch."""
        self.op = operator
        self.ctx, self.queue = operator.ctx, operator.queue
        if subsets is None:
            self.subsets = None
            self.M = 1
        else:
            self.subsets = operator.set_omega_subsets(subsets)
            self.M = len(self.subsets)
        self.lam = float(lam)
        if self.lam < 0:
            raise ValueError("lam must be >= 0")
        self.eps_counts = float(eps_counts)
        self.rng = np.random.default_rng(seed)
        self.store_sensitivities = bool(store_sensitivities)
        q = self.queue
        if getattr(operator, "coefficient_layout", "dense") == "local":
            self.mask = clarray.to_device(q, (np.asarray(operator.coefficient_mask()) > 0).astype(np.uint8))
        else:
            self.mask = support_mask_to_gpu(q, operator, support)
        self.pmask = int(self.mask.size) if self.mask is not None else 0
        if self.mask is None:
            self.mask = clarray.zeros(q, (1,), np.uint8)
        self._update = ElementwiseKernel(
            self.ctx, "float *x, const float *back, const float *sens, const unsigned char *mask, unsigned long pm, "
                      "float lam",
            "const int ok = pm ? mask[i % pm] : 1; const float s = sens[i]; "
            "x[i] = !ok ? 0.0f : (s > 0.0f ? x[i] * back[i] / (s + lam) : x[i])", "osem_update")
        self.iter_stats = []
        self.final_f = None
        self.start_floor_used = None

    def _fwd(self, x, out, m):
        if self.subsets is None:
            self.op.direct_cl(x, out)
        else:
            self.op.direct_cl(x, out, subset=m)

    def _adj(self, r, out, m):
        if self.subsets is None:
            self.op.adjoint_cl(r, out)
        else:
            self.op.adjoint_cl(r, out, subset=m)

    def run(self, x0, b, n_epochs, weights=None, gain=None, background=None, start_floor=0.01, verbose=0,
            diagnostics_interval=1, callback=None):
        """n_epochs passes over all subsets. x0: a pyopencl array (updated in place and returned) or a NumPy array (a
        new NumPy array is returned); shape op.coeff_shape. b, weights, gain, background: as SPDHGPoisson.run.

        start_floor: EM cannot move a zero coefficient. Every allowed coefficient (support / local mask) starts at
        least at start_floor x the median of the positive allowed ones of x0 (0 or None: x0 as given, outside the
        allowed set zeroed; the rock fits used 0.01). If x0 has no positive allowed coefficient, x starts flat at the
        value whose prediction matches the fitted data's total (background subtracted), as MLEM does.

        Diagnostics: the deviance on the fitted angles (a full forward each time) every diagnostics_interval epochs
        and at the end, in iter_stats (epoch, f, time) and final_f. callback(epoch, x, stats) after every epoch
        (x a host copy for a NumPy x0); a true return value stops the run."""
        t0 = time.perf_counter()
        q, op = self.queue, self.op
        cshape, dshape = tuple(op.coeff_shape), tuple(op.data_shape)
        on_gpu = isinstance(x0, clarray.Array)
        x = check_device(x0, cshape, "x0") if on_gpu else clarray.to_device(q, as_host(x0, cshape, "x0"))
        fit = None
        if self.subsets is not None:
            fit = np.zeros(dshape[0], bool)
            for s in self.subsets:
                fit[np.asarray(s, np.int64)] = True
        pt = PoissonTerm(q, dshape, b, weights, gain, background, fit, self.eps_counts)
        pred = clarray.zeros(q, dshape, np.float32)
        r = clarray.empty(q, dshape, np.float32)
        wg = clarray.empty(q, dshape, np.float32)
        pt.weight_over_gain(wg)
        back = clarray.empty(q, cshape, np.float32)
        n_sens = self.M if self.store_sensitivities else 1
        sens = [clarray.empty(q, cshape, np.float32) for _ in range(n_sens)]
        if self.store_sensitivities:
            for m in range(self.M):
                self._adj(wg, sens[m], m)

        # start
        n = int(np.prod(cshape))
        xh = x.get().ravel()
        allowed = np.resize(self.mask.get().astype(bool), n) if self.pmask else np.ones(n, bool)
        xh = np.where(allowed, xh, 0.0)
        if start_floor:
            pos = xh[xh > 0]
            if pos.size:
                floor = float(start_floor) * float(np.median(pos))
            else:
                # flat start c: c sum_j (A_fit^T(w/g))_j = sum_fitted (w/g)(b - bg), the sum over the allowed j
                st = 0.0
                for m in range(self.M):
                    if not self.store_sensitivities:
                        self._adj(wg, sens[0], m)
                    st += float(sens[m if self.store_sensitivities else 0].get().ravel()[allowed].sum(dtype=np.float64))
                tot = float(clarray.dot(wg, pt.b, dtype=np.float64).get())
                if pt.pbg:
                    tot -= float((wg.get().ravel().astype(np.float64)
                                  * np.resize(pt.bg.get(), int(np.prod(dshape)))).sum())
                floor = max(tot, 0.0) / st if st > 0 else 1.0
            xh = np.where(allowed, np.maximum(xh, floor), 0.0)
            self.start_floor_used = floor
        x.set(xh.astype(np.float32).reshape(cshape))

        self.iter_stats = []
        f = None
        for ep in range(int(n_epochs)):
            order = self.rng.permutation(self.M) if self.M > 1 else [0]
            for m in order:
                m = int(m)
                self._fwd(x, pred, m)
                pt.em_ratio(pred, r)
                self._adj(r, back, m)
                if self.store_sensitivities:
                    s = sens[m]
                else:
                    self._adj(wg, sens[0], m)
                    s = sens[0]
                self._update(x, back, s, self.mask, np.uint64(self.pmask), np.float32(self.lam))
            stats = {"epoch": ep + 1, "time": time.perf_counter() - t0}
            f = None
            if diagnostics_interval and ((ep + 1) % int(diagnostics_interval) == 0 or ep + 1 == int(n_epochs)):
                op.direct_cl(x, pred)
                f = pt.deviance(pred)
                stats["f"] = f
                if verbose:
                    print(f"OSEM epoch {ep + 1}: deviance {f:.6g} ({stats['time']:.1f} s)", flush=True)
            self.iter_stats.append(stats)
            if callback is not None and callback(ep + 1, x if on_gpu else x.get(), stats):
                break
        if f is None:
            op.direct_cl(x, pred)
            f = pt.deviance(pred)
        self.final_f = f
        return x if on_gpu else x.get()
