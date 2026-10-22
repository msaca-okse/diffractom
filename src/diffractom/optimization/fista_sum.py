"""FISTA for data that are the sum of several operators' predictions, e.g. one SinglePhaseForwardOperator per
phase writing into the same integration windows (operator option ``rings``):

    min_{x_p >= 0}  sum_i w_i huber_delta((sum_p A_p x_p - b)_i) + sum_p l1_p sum(x_p)

or, with ``log_scale`` = c (scalar or broadcastable to b), the misfit of the transformed values
g(v) = c asinh(v / c): sum_i w_i / 2 (g(u_i) - g(b_i))^2, u = sum_p A_p x_p. That is least squares below c and a
difference of logarithms above it (c (ln u - ln b) for u, b >> c), so whether a bright spot is there counts more than how
bright it is. Not convex where the prediction exceeds the data many times; the step is 1 / (log_step L).

Every operator is used through its NumPy ``direct`` / ``adjoint`` (so each keeps its own PF storage, projector and
coefficient streaming); the coefficients x_p are NumPy arrays (K_p, Ny, Nx). huber_delta = inf is weighted least
squares. Pixels outside each operator's field of view are kept at zero (support="fov").
"""
import time

import numpy as np


def _huber_grad(r, delta):
    return r if not np.isfinite(delta) else np.clip(r, -delta, delta)


def _huber_value(r, delta, w):
    if not np.isfinite(delta):
        return 0.5 * float(np.sum(w * r * r, dtype=np.float64))
    a = np.abs(r)
    return float(np.sum(w * np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta)), dtype=np.float64))


def _data_term(u, b, w, delta, c=None, gb=None):
    """Weighted gradient of the data term with respect to the prediction u, and a function giving its value."""
    if c is None:
        r = u - b
        return (_huber_grad(r, delta) * w).astype(np.float32), lambda: _huber_value(r, delta, w)
    q = u / c
    r = c * np.arcsinh(q) - gb
    return (w * r / np.sqrt(1.0 + q * q)).astype(np.float32), lambda: 0.5 * float(np.sum(w * r * r, dtype=np.float64))


def estimate_L_sum(operators, niter=8, seed=0, verbose=0, weights=None):
    """Largest eigenvalue of A^T W A for A = [A_1 ... A_P] and data weights W (default 1; broadcastable to the data),
    by power iteration from random non-negative x: the Lipschitz constant of the weighted least-squares gradient."""
    rng = np.random.default_rng(seed)
    xs = [rng.random(op.coeff_shape, dtype=np.float32) for op in operators]
    lam = 0.0
    for it in range(niter):
        nrm = np.sqrt(sum(float(np.sum(x.astype(np.float64) ** 2)) for x in xs))
        xs = [x / nrm for x in xs]
        y = sum(op.direct(x) for op, x in zip(operators, xs))
        if weights is not None:
            y = (y * np.asarray(weights, dtype=np.float32)).astype(np.float32)
        zs = [op.adjoint(y) for op in operators]
        lam = sum(float(np.vdot(x.astype(np.float64), z)) for x, z in zip(xs, zs))
        xs = zs
        if verbose:
            print(f"[power {it + 1:02d}] L_est={lam:.6e}", flush=True)
    return lam


class FISTASum:
    """Non-negative FISTA with a Huber (or least-squares) data term for a sum of operators."""

    def __init__(self, operators, L, huber_delta=np.inf, support="fov", l1=0.0, volume_fractions=None,
                 volume_weight=1.0, log_scale=None, log_step=2.0):
        """l1: weight of a sparsity penalty l1 * sum(x) (x >= 0), a scalar or one value per operator.
        volume_fractions: target share of every operator's summed coefficients in the total of the operators that have
        a target (one value per operator, None or nan = free), held softly by the penalty
        lam / 2 * sum_p (S_p - f_p T)^2 with S_p = sum(x_p), T = the sum of S_p over the targeted operators and
        lam = volume_weight * L / (number of targeted coefficients): for phases whose shares the data cannot separate
        (reflections of two phases in the same channels) but which are known from elsewhere, e.g. a powder pattern.
        support: "fov" (coefficients only inside the field of view), None, or a mask (Ny, Nx): coefficients only
        where it is non-zero and inside the field of view (e.g. the sample outline).
        log_scale: c of the transformed misfit (module docstring), scalar or broadcastable to the data (e.g. (N_seg,));
        None = least squares (or Huber). log_step: the step is 1 / (log_step L) with it (the transformed term's
        curvature exceeds that of least squares where the data exceed the prediction many times)."""
        self.operators = list(operators)
        self.log_scale = None if log_scale is None else np.asarray(log_scale, dtype=np.float32)
        self.log_step = float(log_step)
        self.L = float(L)
        self.l1 = np.broadcast_to(np.asarray(l1, dtype=np.float64), (len(self.operators),)).copy()
        self.huber_delta = float(huber_delta)
        if isinstance(support, str) and support != "fov":
            raise ValueError("support: 'fov', None or a mask (Ny, Nx)")
        self.masks = [self._mask(op, support) for op in self.operators]
        self.vf = None
        if volume_fractions is not None:
            vf = np.array([np.nan if v is None else float(v) for v in volume_fractions], dtype=np.float64)
            if np.isfinite(vf).sum() >= 2:
                self.vf = vf / np.nansum(vf)
                n = sum(int(np.prod(op.coeff_shape)) for op, v in zip(self.operators, vf) if np.isfinite(v))
                self.vlam = float(volume_weight) * self.L / n
                self.L *= 1.0 + 2.0 * float(volume_weight)          # the penalty's curvature is at most 2 lam n
        self.history = []

    @staticmethod
    def _mask(op, support):
        """The support of an operator's coefficients, broadcastable to them (None: no constraint). Operators
        with a local basis (coefficient_layout "local") hold the field of view in their support already."""
        if support is None:
            return None
        if getattr(op, "coefficient_layout", "dense") != "dense":
            return None if isinstance(support, str) else op.coefficient_mask(np.asarray(support) != 0)
        m = op.support_mask().astype(np.float32)
        if not isinstance(support, str):
            m = m * (np.asarray(support) != 0).astype(np.float32)
        return m[None]

    def volume_gradient(self, xs):
        """Per operator the (constant) gradient of the volume-fraction penalty, and the penalty's value."""
        S = np.array([float(x.sum(dtype=np.float64)) for x in xs])
        on = np.isfinite(self.vf)
        e = np.where(on, S - np.where(on, self.vf, 0.0) * S[on].sum(), 0.0)
        g = np.where(on, e - np.sum(np.where(on, self.vf, 0.0) * e), 0.0)
        return self.vlam * g, 0.5 * self.vlam * float(np.sum(e ** 2))

    def direct(self, xs):
        return sum(op.direct(x) for op, x in zip(self.operators, xs))

    def run(self, xs, b, niter, weights=None, verbose=1, diagnostics_interval=10, offset=None):
        """xs: list of starting coefficients (NumPy, (K_p, Ny, Nx), or the operator's coeff_shape); b: data (N_Omega, My, N_seg); weights:
        broadcastable to b (e.g. (N_seg,)); offset: a fixed prediction added to sum_p A_p x_p (e.g. of parts that are
        not fitted; with log_scale it matters whether it is added here or subtracted from b). Returns the list of
        coefficients."""
        w = np.float32(1.0) if weights is None else np.asarray(weights, dtype=np.float32)
        xs = [np.ascontiguousarray(x, dtype=np.float32) for x in xs]
        ys = [x.copy() for x in xs]
        t = 1.0
        c = self.log_scale
        gb = None if c is None else (c * np.arcsinh(b / c)).astype(np.float32)
        tau = 1.0 / self.L if c is None else 1.0 / (self.log_step * self.L)
        t0 = time.time()
        for it in range(1, niter + 1):
            u = self.direct(ys)
            if offset is not None:
                u += offset
            g, value = _data_term(u, b, w, self.huber_delta, c, gb)
            del u
            report = verbose and (it == 1 or it % diagnostics_interval == 0 or it == niter)
            if report:
                f = value() + sum(float(l * x.sum(dtype=np.float64)) for l, x in zip(self.l1, ys))
            del value
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            beta = (t - 1.0) / t_new
            vg = None
            if self.vf is not None:
                vg, vval = self.volume_gradient(ys)
                if report:
                    f += vval
            for p, op in enumerate(self.operators):
                x_new = ys[p] - tau * op.adjoint(g)
                if vg is not None and vg[p] != 0.0:
                    x_new -= np.float32(tau * vg[p])
                if self.l1[p] > 0:
                    x_new -= np.float32(tau * self.l1[p])           # prox of l1 * sum(x) with x >= 0
                np.maximum(x_new, 0, out=x_new)
                if self.masks[p] is not None:
                    x_new *= self.masks[p]
                ys[p] = x_new + beta * (x_new - xs[p])
                xs[p] = x_new
            t = t_new
            if report:
                self.history.append((it, f))
                norms = " ".join(f"{np.sqrt(np.sum(x.astype(np.float64) ** 2)):.3e}" for x in xs)
                print(f"[iter {it:5d}/{niter}] f={f:.6e}  ||x_p||={norms}  ({time.time() - t0:.0f} s)", flush=True)
        return xs
