"""FISTA for data that are the sum of several operators' predictions, e.g. one SinglePhaseForwardOperator per
phase writing into the same integration windows (operator option ``rings``):

    min_{x_p >= 0}  sum_i w_i huber_delta((sum_p A_p x_p - b)_i) + sum_p l1_p sum(x_p)

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

    def __init__(self, operators, L, huber_delta=np.inf, support="fov", l1=0.0):
        """l1: weight of a sparsity penalty l1 * sum(x) (x >= 0), a scalar or one value per operator."""
        self.operators = list(operators)
        self.L = float(L)
        self.l1 = np.broadcast_to(np.asarray(l1, dtype=np.float64), (len(self.operators),)).copy()
        self.huber_delta = float(huber_delta)
        self.masks = [op.support_mask().astype(np.float32) if support == "fov" else None for op in self.operators]
        self.history = []

    def direct(self, xs):
        return sum(op.direct(x) for op, x in zip(self.operators, xs))

    def run(self, xs, b, niter, weights=None, verbose=1, diagnostics_interval=10):
        """xs: list of starting coefficients (NumPy, (K_p, Ny, Nx)); b: data (N_Omega, My, N_seg); weights:
        broadcastable to b (e.g. (N_seg,)). Returns the list of coefficients."""
        w = np.float32(1.0) if weights is None else np.asarray(weights, dtype=np.float32)
        xs = [np.ascontiguousarray(x, dtype=np.float32) for x in xs]
        ys = [x.copy() for x in xs]
        t = 1.0
        tau = 1.0 / self.L
        t0 = time.time()
        for it in range(1, niter + 1):
            r = self.direct(ys) - b
            g = (_huber_grad(r, self.huber_delta) * w).astype(np.float32)
            report = verbose and (it == 1 or it % diagnostics_interval == 0 or it == niter)
            if report:
                f = _huber_value(r, self.huber_delta, w) + sum(float(l * x.sum(dtype=np.float64)) for l, x in zip(self.l1, ys))
            del r
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            beta = (t - 1.0) / t_new
            for p, op in enumerate(self.operators):
                x_new = ys[p] - tau * op.adjoint(g)
                if self.l1[p] > 0:
                    x_new -= np.float32(tau * self.l1[p])           # prox of l1 * sum(x) with x >= 0
                np.maximum(x_new, 0, out=x_new)
                if self.masks[p] is not None:
                    x_new *= self.masks[p][None]
                ys[p] = x_new + beta * (x_new - xs[p])
                xs[p] = x_new
            t = t_new
            if report:
                self.history.append((it, f))
                norms = " ".join(f"{np.sqrt(np.sum(x.astype(np.float64) ** 2)):.3e}" for x in xs)
                print(f"[iter {it:5d}/{niter}] f={f:.6e}  ||x_p||={norms}  ({time.time() - t0:.0f} s)", flush=True)
        return xs
