"""Gauss-Newton refinement of the basis orientations and their kernel widths, alternated with coefficient fits
(OrientationRefiner).

Each step, at the current coefficients x and Poisson deviance f (optimization.poisson_term; on the fitted angles):
1. r = df/dpred and the Fisher curvature c = w / (g mu) of the prediction; PFGradient gives the gradient of f with
   respect to every orientation's rotation (3, a left increment) and width sigma, and the 4 x 4 Gauss-Newton block
   J_k^T diag(c) J_k of each orientation (the coupling between orientations ignored).
2. A damped Newton step per orientation (H_k + damping diag(H_k))^-1 grad_k, the rotation limited to max_rot_deg (a
   trust region) and the width change to max_dsigma x sigma, the width kept in [sigma_min_deg, sigma_max_deg].
3. A backtracking line search over the step length (alphas) at fixed x: the first that lowers f is kept, else the
   old parameters are restored.
Then the coefficients are refit (solve(x), e.g. some SPDHGPoisson or OSEMPoisson epochs; start a new solver each time:
the dual variable of an earlier SPDHG run diverged after the spots had moved, see SPDHGPoisson).

Measured (docs/differentiable_model.md): simulated 1 deg mosaicity, median misorientation to the truth 0.216 -> 0.097
deg; Al1050 held-out relative residual 2.3 % lower than an equally long fit without it (widths settle at a median
0.28 deg). The rock fits (DanMAX 2026) used max_rot_deg 0.2, max_dsigma 0.3, width floors 0.05-0.3 deg, 4-12 steps
between SPDHG or OSEM epochs.

The operator needs pf_mode "sparse" or "generated" (PFGradient), dense or local coefficient layout.
"""
import time

import numpy as np
import pyopencl.array as clarray

from .gradients import PFGradient
from .update import rotate_orientations, set_pf_parameters
from ..optimization.poisson_term import PoissonTerm
from ..utils.arrays import as_host


class OrientationRefiner:
    """Gauss-Newton steps on the orientations and widths of op's basis (module docstring)."""

    def __init__(self, op, b, weights=None, gain=None, background=None, fit_rows=None, eps_counts=1e-3,
                 max_rot_deg=0.2, max_dsigma=0.3, sigma_min_deg=0.05, sigma_max_deg=5.0, damping=1e-3,
                 alphas=(1.0, 0.5, 0.25, 0.1), refine_sigma=True, curv_floor_counts=0.25):
        """b, weights, gain, background: as in the Poisson solvers; fit_rows: bool (N_Omega,) or indices of the
        fitted angles (None: all; the held-out angles must not steer the refinement). refine_sigma: False keeps the
        widths and steps the rotations only."""
        self.op = op
        self.q = op.queue
        self.pt = PoissonTerm(self.q, op.data_shape, b, weights, gain, background, fit_rows, eps_counts)
        self.grad = PFGradient(op, hessian=True)
        self.max_rot = np.radians(float(max_rot_deg))
        self.max_dsigma = float(max_dsigma)
        self.sig_lo, self.sig_hi = np.radians(float(sigma_min_deg)), np.radians(float(sigma_max_deg))
        self.damping = float(damping)
        self.alphas = tuple(float(a) for a in alphas)
        self.refine_sigma = bool(refine_sigma)
        self.curv_floor = float(curv_floor_counts)
        self.pred = clarray.empty(self.q, tuple(op.data_shape), np.float32)
        self.history = []

    def _dev(self, x):
        if isinstance(x, clarray.Array):
            return x, None
        d = clarray.to_device(self.q, as_host(x, self.op.coeff_shape, "x"))
        return d, d

    def loss(self, x):
        """The deviance of A x on the fitted angles (x NumPy or pyopencl)."""
        xd, _ = self._dev(x)
        self.op.direct_cl(xd, self.pred)
        return self.pt.deviance(self.pred)

    def newton_steps(self, x):
        """(K, 4) steps (rotation increment in radians, width change in radians) at x, before limits; and f(x)."""
        xd, _ = self._dev(x)
        self.op.direct_cl(xd, self.pred)
        f0 = self.pt.deviance(self.pred)
        r = clarray.empty(self.q, self.pred.shape, np.float32)
        c = clarray.empty(self.q, self.pred.shape, np.float32)
        self.pt.grad_curv(self.pred, r, c, self.curv_floor)
        g = self.grad(xd, r, groups=("rot", "sigma"), curv=c)
        grad = np.concatenate([g["rot"], g["sigma"][:, None]], axis=1).astype(np.float64)
        H = np.asarray(g["hess"], np.float64)
        n = 4 if self.refine_sigma else 3
        Hn, gn = H[:, :n, :n], grad[:, :n]
        tr = np.trace(Hn, axis1=1, axis2=2)
        ok = tr > 0
        st = np.zeros((len(grad), 4))
        if ok.any():
            Hd = Hn[ok] + self.damping * np.einsum("kii->ki", Hn[ok])[:, :, None] * np.eye(n)[None] \
                + 1e-12 * tr[ok, None, None] * np.eye(n)[None]
            st[ok, :n] = -np.linalg.solve(Hd, gn[ok][..., None])[..., 0]
        return st, f0

    def step(self, x):
        """One Gauss-Newton step at fixed coefficients x (NumPy or pyopencl, unchanged). Returns a dict: alpha (the
        accepted step length, 0 = rejected), f0 and f (deviance before and after), the median and maximum rotation
        (deg) and the median width (deg) after the step."""
        t0 = time.perf_counter()
        op = self.op
        xd, _ = self._dev(x)
        st, f0 = self.newton_steps(xd)
        sig0 = np.asarray(op.sigma_cpu, np.float64).copy()
        U0 = np.asarray(op.grid_rot_cpu, np.float64).copy()
        st[:, 3] = np.clip(st[:, 3], -self.max_dsigma * sig0, self.max_dsigma * sig0)
        rn = np.linalg.norm(st[:, :3], axis=1)
        st[:, :3] *= np.minimum(1.0, self.max_rot / np.maximum(rn, 1e-30))[:, None]
        acc, f = 0.0, f0
        for a in self.alphas:
            set_pf_parameters(op, rotations=rotate_orientations(U0, a * st[:, :3]),
                              sigma=np.clip(sig0 + a * st[:, 3], self.sig_lo, self.sig_hi))
            fa = self.loss(xd)
            if fa < f0:
                acc, f = a, fa
                break
        if acc == 0.0:
            set_pf_parameters(op, rotations=U0, sigma=sig0)
        moved = np.degrees(acc * np.linalg.norm(st[:, :3], axis=1))
        info = {"alpha": acc, "f0": f0, "f": f, "rot_deg_median": float(np.median(moved)),
                "rot_deg_max": float(moved.max()) if moved.size else 0.0,
                "sigma_deg_median": float(np.degrees(np.median(op.sigma_cpu))), "time": time.perf_counter() - t0}
        return info

    def run(self, x, solve, n_steps, verbose=False):
        """n_steps of [Gauss-Newton step, then x = solve(x)] (solve: the coefficient fit, e.g. lambda x:
        SPDHGPoisson(op, subsets=subs).run(x, b, 20, weights=w, gain=g, background=bg); it returns the new x).
        Returns x; self.history holds every step's step() dict and "f_fit" (the deviance after the refit)."""
        for t in range(int(n_steps)):
            info = self.step(x)
            x = solve(x)
            info["f_fit"] = self.loss(x)
            info["step"] = t + 1
            self.history.append(info)
            if verbose:
                print(f"GN step {t + 1}: alpha {info['alpha']}, deviance {info['f0']:.6g} -> {info['f']:.6g} "
                      f"(refit {info['f_fit']:.6g}), rotation median {info['rot_deg_median']:.3f} deg, "
                      f"width median {info['sigma_deg_median']:.3f} deg", flush=True)
        return x
