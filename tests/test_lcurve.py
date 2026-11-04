"""L-curve corner and scan."""
import numpy as np
import pytest

from diffractom.optimization.lcurve import lcurve_corner, lcurve_scan, menger_curvature


def test_curvature_of_a_circle_and_a_line():
    t = np.linspace(0, np.pi / 2, 9)
    k = menger_curvature(2 * np.cos(t), 2 * np.sin(t))   # counter-clockwise, radius 2
    np.testing.assert_allclose(k[1:-1], 0.5, rtol=1e-12)
    assert np.allclose(menger_curvature(np.arange(5.0), 2 * np.arange(5.0))[1:-1], 0)


def test_corner_of_a_tikhonov_l_curve():
    """Tikhonov solutions of a discrete ill-posed problem (singular values 1 .. 1e-6, coefficients decaying like
    them, noise 1e-3): the corner's weight is within a factor 10 of the one with the smallest error."""
    rng = np.random.default_rng(0)
    n = 60
    s = np.logspace(0, -6, n)
    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    x_true = V @ (s ** 0.5 * rng.normal(size=n))
    b = U @ (s * (V.T @ x_true)) + 1e-3 * rng.normal(size=n)
    beta = U.T @ b
    lams = np.logspace(-6, 0, 49)
    xs = [V @ (s / (s ** 2 + lam ** 2) * beta) for lam in lams]
    res = [np.linalg.norm(U @ (s * (V.T @ x)) - b) for x in xs]
    pen = [np.linalg.norm(x) for x in xs]
    i, k = lcurve_corner(res, pen)
    err = np.array([np.linalg.norm(x - x_true) for x in xs])
    assert 0 < i < len(lams) - 1
    assert abs(np.log10(lams[i] / lams[np.argmin(err)])) <= 1, (lams[i], lams[np.argmin(err)])


class _FakeSolver:
    """A solver whose fit traces the L-curve of a scalar problem: misfit grows and the penalty falls with lam."""

    def __init__(self, lam):
        self.lam = lam

    def run(self, x, b, niter, weights=None):
        x[...] = b / (1 + self.lam)
        self.iter_stats = [{"f": 0.5 * float(((x - b) ** 2).sum()) + 1e-12, "g": self.lam * float(np.abs(x).sum())}]
        return x


def test_scan_reports_the_curve_and_the_solution_at_the_corner():
    b = np.array([1.0, 2.0, 3.0])
    out = lcurve_scan(_FakeSolver, np.zeros(3), b, np.logspace(-3, 3, 13), niter=1, keep=True)
    assert np.all(np.diff(out["residual"]) > 0) and np.all(np.diff(out["penalty"]) < 0)
    assert out["lams"][out["corner"]] == out["lam"]
    np.testing.assert_allclose(out["solution"], b / (1 + out["lam"]))
    with pytest.raises(ValueError):
        lcurve_scan(_FakeSolver, np.zeros(3), b, [0.0, 1.0, 2.0], niter=1)
