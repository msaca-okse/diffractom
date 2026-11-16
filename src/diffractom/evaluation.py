"""Held-out evaluation: which rotation angles to fit, subsets of them for the stochastic solvers, and the Poisson
deviance of a prediction (the metric the model comparisons use: lower is better, comparable within one dataset).

The split the rock and Al1050 fits use holds out every fifth block of 5 consecutive angles (block_split()): the held-out
angles are spread over the whole rotation, and a block is wider than the kernel smearing between neighbouring angles,
so a fit cannot reproduce a held-out angle by interpolating its neighbours.

Noise floor of the deviance per point: for a perfect model with many counts per point, E[deviance] is about 1/2 per
data point (less for points with few counts); values far above it mean model error, not noise.
"""
import numpy as np


def block_split(n_omega, block=5, period=5, held=2):
    """Bool (n_omega,): True for the angles to fit. Blocks of `block` consecutive angles, every `period`-th block
    (block index % period == held) held out: the default holds out 20 % of the angles in blocks of 5."""
    i = np.arange(int(n_omega))
    return (i // int(block)) % int(period) != int(held)


def interleaved_subsets(fitted, n_subsets):
    """n_subsets interleaved subsets of the fitted angles (subset m: every n_subsets-th of them, from the m-th), for
    SPDHGPoisson / OSEMPoisson (subsets=...). fitted: a bool array of length N_Omega or the indices of the fitted
    angles. The angles in no subset are not fitted."""
    f = np.asarray(fitted)
    idx = np.flatnonzero(f) if f.dtype == bool else np.asarray(f, np.int64)
    return [idx[m::int(n_subsets)] for m in range(int(n_subsets))]


def _host(a):
    if a is None:
        return None
    return a.get() if hasattr(a, "get") and not isinstance(a, np.ndarray) else np.asarray(a)


def _per_row(a, dshape, name):
    """A per-segment, per-(translation, segment) or data-shaped array as an array broadcastable to one angle's
    (My, N_seg) or, for the data shape, returned whole."""
    if a is None:
        return None
    a = np.asarray(a, np.float64)
    n_seg, n_ys, n_all = dshape[-1], dshape[-2] * dshape[-1], int(np.prod(dshape))
    if a.size == 1:
        return np.full(n_seg, float(a.ravel()[0]))
    if a.size == n_seg:
        return a.reshape(n_seg)
    if a.size == n_ys:
        return a.reshape(dshape[-2], n_seg)
    if a.size == n_all:
        return a.reshape(dshape)
    raise ValueError(f"{name} must have N_seg, My * N_seg or {n_all} elements; got {a.shape}")


def poisson_deviance(b, mu, gain=None, weights=None, background=None, rows=None, eps_counts=1e-3,
                     per_point=False):
    """The Poisson deviance of the prediction mu (= A x, data shape (N_Omega, My, N_seg)) for the data b, as the
    solvers' objective (FISTAPoisson):

        sum_i w_i / g_i [ mu_i' - b_i + b_i log(b_i / mu_i') ],   mu_i' = max(mu_i, 0) + bg_i + eps_counts g_i,

    over the angles `rows` (a bool mask or indices; None: all), in photons. gain, weights, background: per segment,
    per (translation, segment) or data-shaped (as in the solvers; None: gain 1, weights 1, no background). NumPy or
    pyopencl arrays (copied to the host), summed in float64 one angle at a time. per_point: divided by the number of
    counted points (weight != 0) in those rows (the "held-out deviance per point" of the model comparisons).
    Returns a float."""
    b, mu = _host(b), _host(mu)
    dshape = tuple(b.shape) if b.ndim == 3 else tuple(mu.shape)
    b = b.reshape(dshape)
    mu = mu.reshape(dshape)
    g = _per_row(_host(gain), dshape, "gain")
    w = _per_row(_host(weights), dshape, "weights")
    bg = _per_row(_host(background), dshape, "background")
    if rows is None:
        idx = np.arange(dshape[0])
    else:
        r = np.asarray(rows)
        idx = np.flatnonzero(r) if r.dtype == bool else r.astype(np.int64)
    total, count = 0.0, 0
    for a in idx:
        ga = 1.0 if g is None else (g[a] if g.ndim == 3 else g)
        wa = 1.0 if w is None else (w[a] if w.ndim == 3 else w)
        ba = 0.0 if bg is None else (bg[a] if bg.ndim == 3 else bg)
        bi = b[a].astype(np.float64)
        m = np.maximum(mu[a].astype(np.float64), 0.0) + ba + eps_counts * np.asarray(ga, np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = m - bi + np.where(bi > 0, bi * np.log(np.where(bi > 0, bi, 1.0) / m), 0.0)
        wg = np.broadcast_to(np.asarray(wa, np.float64) / np.asarray(ga, np.float64), t.shape)
        total += float((wg * t)[wg != 0].sum())
        count += int(np.count_nonzero(wg))
    if per_point:
        return total / max(count, 1)
    return total
