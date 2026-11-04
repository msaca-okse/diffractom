"""Choosing a regularisation weight by the L-curve.

For a sequence of weights lam_1 < ... < lam_n, the L-curve is (log ||w (A x_lam - b)||, log R(x_lam)), with R the
regulariser (TV(x), ||x||_1, ...). Small weights fit the data and leave R large (the vertical branch), large weights
make R small at the cost of the fit (the horizontal branch); the corner between them balances the two (Hansen,
SIAM Review 34, 561-580, 1992; Hansen & O'Leary, SIAM J. Sci. Comput. 14, 1487-1503, 1993). lcurve_corner takes
the point of largest curvature, measured for each interior point as the curvature of the circle through it and
its two neighbours (the Menger curvature; cf. Cultrera & Callegaro, IOP SciNotes 1, 025004, 2020), on both log
axes scaled to unit range so that the result does not depend on units. A problem that needs little
regularisation has no clear corner, and the point found then over-regularises (the README has an example):
check the weight on held-out data.

A scan of the weight by the L-curve, as a routine of the package, follows textom 0.7.21 (scan_tikhonov;
M. Frewein, M. Allain, T. Grunewald et al., https://gitlab.fresnel.fr/textom/textom). This module is an
independent implementation for diffractom's solvers (no textom code is used).
"""
import numpy as np


def menger_curvature(x, y):
    """Signed curvature of the circle through consecutive points (x[i-1], y[i-1]), (x[i], y[i]), (x[i+1], y[i+1]);
    NaN at both ends. Positive for a counter-clockwise turn (an L-curve traversed with increasing weight turns
    counter-clockwise at its corner)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    k = np.full(x.size, np.nan)
    for i in range(1, x.size - 1):
        ax, ay = x[i] - x[i - 1], y[i] - y[i - 1]
        bx, by = x[i + 1] - x[i], y[i + 1] - y[i]
        cx, cy = x[i + 1] - x[i - 1], y[i + 1] - y[i - 1]
        den = np.sqrt((ax * ax + ay * ay) * (bx * bx + by * by) * (cx * cx + cy * cy))
        k[i] = 2.0 * (ax * by - ay * bx) / den if den > 0 else 0.0
    return k


def lcurve_corner(residual, penalty):
    """The corner of an L-curve.

    Parameters
    ----------
    residual, penalty : sequences of the data misfit (e.g. ||w (A x - b)||) and the regulariser value R(x), for
        increasing regularisation weights (at least three, all positive)

    Returns
    -------
    (index of the corner, signed curvature per point on the normalised log-log curve, NaN at the ends).
    If no interior point turns counter-clockwise, the index of the largest |curvature| is returned.
    """
    r = np.log10(np.asarray(residual, np.float64))
    p = np.log10(np.asarray(penalty, np.float64))
    if r.size < 3:
        raise ValueError("an L-curve corner needs at least three points")
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(p))):
        raise ValueError("the residuals and penalties must be positive and finite")
    span_r, span_p = np.ptp(r), np.ptp(p)
    k = menger_curvature((r - r.min()) / (span_r if span_r > 0 else 1.0), (p - p.min()) / (span_p if span_p > 0 else 1.0))
    inner = k[1:-1]
    i = int(np.nanargmax(inner)) + 1 if np.nanmax(inner) > 0 else int(np.nanargmax(np.abs(inner))) + 1
    return i, k


def lcurve_scan(make_solver, x0, b, lams, niter, weights=None, penalty=None, keep=False, verbose=0):
    """Fit once per weight and locate the L-curve corner.

    Parameters
    ----------
    make_solver : callable lam -> solver with run(x, b, niter, weights=...) and iter_stats (FISTAHuber, FISTAL2,
        FISTAGram, ... set up with regularisation weight lam)
    x0 : the starting point of every fit (NumPy or pyopencl array; copied for each weight)
    b, weights, niter : passed to run
    lams : the weights, all > 0 (sorted ascending here)
    penalty : optional callable x -> R(x); by default the solver's last g divided by lam (g = lam R(x), as the
        solvers report it in iter_stats)
    keep : keep every solution (otherwise only the one at the corner)

    Returns
    -------
    dict with lams, residual (sqrt(2 f) of the last iteration, f = 0.5 ||w (A x - b)||^2 as the solvers report
    it), penalty, corner (index), lam (the weight at the corner), curvature, solution (at the corner) and, with
    keep, solutions.
    """
    lams = np.sort(np.asarray(lams, np.float64))
    if lams.size < 3 or np.any(lams <= 0):
        raise ValueError("lcurve_scan needs at least three positive weights")
    residual, pen, sols = [], [], []
    for lam in lams:
        x = x0.copy()
        solver = make_solver(float(lam))
        if weights is None:
            out = solver.run(x, b, niter)
        else:
            out = solver.run(x, b, niter, weights=weights)
        x = out if out is not None else x
        st = solver.iter_stats[-1]
        residual.append(float(np.sqrt(2.0 * st["f"])))
        pen.append(float(penalty(x)) if penalty is not None else float(st["g"]) / float(lam))
        sols.append(x)
        if verbose:
            print(f"[lcurve] lam={lam:.3e}  residual={residual[-1]:.6e}  penalty={pen[-1]:.6e}")
    i, k = lcurve_corner(residual, pen)
    out = {"lams": lams, "residual": np.array(residual), "penalty": np.array(pen), "corner": i, "lam": float(lams[i]),
           "curvature": k, "solution": sols[i]}
    if keep:
        out["solutions"] = sols
    return out
