"""Quadrature of the pole density over (omega, eta) bins: rules, the error bound and the automatic choice (CPU only)."""
import numpy as np
import pytest
from scipy.special import ive

from diffractom import QuadratureRule
from diffractom.operators.quadrature import (choose_rule, cut_share, direction_bound, gauss_constant, nodes_weights,
                                             peak_derivative_factor)

D = np.radians


@pytest.mark.parametrize("p,k", [(1, 1), (1, 4), (2, 1), (3, 2), (5, 1), (8, 3)])
def test_nodes_weights_exact_for_polynomials(p, k):
    x, w = nodes_weights(p, k)
    assert np.all((x > 0) & (x < 1)) and np.isclose(w.sum(), 1.0)
    for deg in range(2 * p):            # exact up to degree 2p - 1 on every panel, hence on the bin
        exact = 1.0 / (deg + 1)
        assert np.isclose((w * x ** deg).sum(), exact, rtol=1e-12, atol=1e-14)


def test_gauss_constant():
    assert gauss_constant(1) == pytest.approx(1 / 24)
    assert gauss_constant(2) == pytest.approx(1 / 4320)


def test_peak_derivative_factor_polynomials_and_bessel_sums():
    for a in (0.3, 5.0, 300.0):
        assert peak_derivative_factor(2, a) == pytest.approx(a)
        assert peak_derivative_factor(4, a) == pytest.approx(3 * a ** 2 + a)
        assert peak_derivative_factor(6, a) == pytest.approx(15 * a ** 3 + 15 * a ** 2 + a)
        assert peak_derivative_factor(8, a) == pytest.approx(105 * a ** 4 + 210 * a ** 3 + 63 * a ** 2 + a)
        k = np.arange(-600, 601)
        for m in (2, 4, 8, 12, 16):
            assert peak_derivative_factor(m, a) == pytest.approx((k.astype(float) ** m * ive(k, a)).sum(), rel=1e-9)


def test_peak_derivative_factor_is_the_maximum_derivative():
    # max over t of |d^m/dt^m exp(a (cos t - 1))|, by finite differences of the analytic derivatives (m = 2, 4)
    t = np.linspace(-np.pi, np.pi, 200001)
    for a in (0.5, 7.0, 2000.0):
        f = np.exp(a * (np.cos(t) - 1))
        d2 = f * (a ** 2 * np.sin(t) ** 2 - a * np.cos(t))
        assert np.abs(d2).max() == pytest.approx(peak_derivative_factor(2, a), rel=1e-6)
        s, c = np.sin(t), np.cos(t)
        d4 = f * (a ** 4 * s ** 4 - 6 * a ** 3 * s ** 2 * c + a ** 2 * (3 * c ** 2 - 4 * s ** 2) + a * c)
        assert np.abs(d4).max() == pytest.approx(peak_derivative_factor(4, a), rel=1e-6)


def test_bound_holds_along_a_great_circle():
    # worst case of the bound: the pole on a great circle, f(t) = exp(-(1 - cos t)/s^2); bin means vs Gauss-Legendre
    rng = np.random.default_rng(0)
    s = D(0.4)
    gx, gw = np.polynomial.legendre.leggauss(64)
    for width, p, k in ((D(0.5), 1, 1), (D(0.5), 2, 1), (D(2.0), 4, 1), (D(2.0), 3, 2), (D(2.0), 2, 3)):
        x, w = nodes_weights(p, k)
        b = direction_bound(p, k, width, s)
        worst = 0.0
        for u in rng.uniform(-3 * s, 3 * s, 200):        # bin start relative to the peak
            f = lambda t: np.exp(-(1 - np.cos(u + t * width)) / s ** 2)
            ref = sum((f((j + (gx + 1) / 2) / 8) * gw / 2 / 8).sum() for j in range(8))
            worst = max(worst, abs((f(x) * w).sum() - ref))
        assert worst <= b, (width, p, k, worst, b)
        assert worst >= 0.02 * b                          # and not absurdly loose


def test_choose_rule():
    share = cut_share(4.0)
    assert choose_rule(D(0.5), D(2.0), D(4.0), 0.05, share) == QuadratureRule(1, 1, 1, 1)
    assert choose_rule(D(0.5), D(2.0), D(0.4), 0.05, share) == QuadratureRule(2, 1, 4, 1)   # the simulations
    for s in (0.2, 0.4, 1.0, 2.0):
        r = choose_rule(D(0.5), D(1.0), D(s), 0.05, share)
        assert r.bound(D(0.5), D(1.0), D(s)) + share <= 0.05
        # no rule with fewer nodes meets the tolerance
        n = r.n_omega * r.n_eta
        for p1 in range(1, 5):
            for k1 in range(1, 9):
                for p2 in range(1, 5):
                    for k2 in range(1, 9):
                        if p1 * k1 * p2 * k2 < n:
                            q = QuadratureRule(p1, k1, p2, k2)
                            assert q.bound(D(0.5), D(1.0), D(s)) + share > 0.05
    with pytest.raises(ValueError):
        choose_rule(D(0.5), D(1.0), D(0.4), 0.01, cut_share(3.0))      # the 3 sigma cut alone is 1.1 %


def test_rule_basics():
    r = QuadratureRule.midpoint(2, 3)
    assert r.is_midpoint and (r.n_omega, r.n_eta) == (2, 3)
    assert not QuadratureRule(2, 1, 1, 1).is_midpoint
    with pytest.raises(ValueError):
        QuadratureRule(0, 1, 1, 1)
