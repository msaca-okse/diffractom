"""
Quadrature of the pole density over one (omega, eta) data bin, with an error bound.

A PF-matrix entry is the mean of the pole density over a bin [omega, omega + d_omega] x [eta, eta + d_eta].
It is approximated by a tensor-product rule: in each direction k panels of p-point Gauss-Legendre
(p = 1 is the midpoint rule). The error of the bin mean, relative to the peak of one pole's density,
is bounded by the sum over the two directions of

    c_p h^{2p} K_{2p},   h = d / k,   c_p = (p!)^4 / ((2p + 1) ((2p)!)^3)          (Gauss-Legendre)

where K_m bounds the m-th derivative along omega or eta. For the density exp(-(1 - cos a) / sigma^2), along
any omega or eta line, K_m <= mu_m(1 / sigma^2) times the peak, with mu_m(a) = e^{-a} sum_k k^m I_k(a), the
m-th moment of the symmetric Skellam distribution: mu_2 = a, mu_4 = 3a^2 + a, mu_6 = 15a^3 + 15a^2 + a, ...
(Jacobi-Anger: the maximum is at the peak; it is reached when the probed direction moves on a great circle).
The cut of the Gaussian at pf_cutoff_sigma adds at most exp(-pf_cutoff_sigma^2 / 2) of the peak.
Derivation and checks: the notes "Midpoint quadrature over (omega, eta)" (2026-10-29).

choose_rule() picks, for a tolerance, the rule with the fewest nodes per bin whose bound meets it.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import comb, factorial

import numpy as np

P_MAX = 8          # Gauss-Legendre points per panel considered by choose_rule
N_MAX = 64         # nodes per bin and direction considered by choose_rule
DEFAULT_TOL = 0.05


def gauss_constant(p: int) -> float:
    """c_p of the p-point Gauss-Legendre error term (c_1 = 1/24: the midpoint rule)."""
    return factorial(p) ** 4 / ((2 * p + 1) * factorial(2 * p) ** 3)


def peak_derivative_factor(m: int, a: float) -> float:
    """mu_m(a) = e^{-a} sum_k k^m I_k(a) for even m: max over t of |d^m/dt^m exp(a (cos t - 1))|.

    The moments of the symmetric Skellam distribution, whose cumulants are a (even orders) and 0 (odd).
    """
    if m % 2:
        raise ValueError("m must be even")
    mom = [1.0]
    for n in range(1, m + 1):
        mom.append(sum(comb(n - 1, j - 1) * a * mom[n - j] for j in range(2, n + 1, 2)))
    return mom[m]


def direction_bound(p: int, k: int, width: float, sigma: float) -> float:
    """Bound on the error of a bin mean (relative to the peak) from one direction: k panels of p points over
    a bin of `width` radians, Gaussian width sigma (radians)."""
    h = abs(float(width)) / k
    return gauss_constant(p) * h ** (2 * p) * peak_derivative_factor(2 * p, 1.0 / float(sigma) ** 2)


def nodes_weights(p: int, k: int):
    """Nodes (offsets in (0, 1) of the bin) and weights (summing to 1) of k panels of p-point Gauss-Legendre."""
    g, w = np.polynomial.legendre.leggauss(p)
    x = ((np.arange(k)[:, None] + (g[None, :] + 1.0) / 2.0) / k).ravel()
    return x, np.tile(w / 2.0 / k, k)


@dataclass(frozen=True)
class QuadratureRule:
    """Tensor-product rule over one (omega, eta) bin: k panels of p-point Gauss-Legendre per direction.

    p = 1 in both directions is the midpoint rule with k sub-bins (equal weights), as N_Omega_subdivisions = k_omega
    and N_eta_subdivisions = k_eta of earlier versions.
    """
    p_omega: int = 1
    k_omega: int = 1
    p_eta: int = 1
    k_eta: int = 1

    def __post_init__(self):
        for v in (self.p_omega, self.k_omega, self.p_eta, self.k_eta):
            if int(v) != v or v < 1:
                raise ValueError(f"QuadratureRule needs positive integers, not {self}")

    @classmethod
    def midpoint(cls, n_omega: int = 1, n_eta: int = 1) -> "QuadratureRule":
        return cls(1, int(n_omega), 1, int(n_eta))

    @property
    def n_omega(self) -> int:
        return self.p_omega * self.k_omega

    @property
    def n_eta(self) -> int:
        return self.p_eta * self.k_eta

    @property
    def is_midpoint(self) -> bool:
        """Equal weights in both directions (the PF kernels then average, as before)."""
        return self.p_omega == 1 and self.p_eta == 1

    def bound(self, d_omega: float, d_eta: float, sigma: float) -> float:
        """Bound on the quadrature error of a PF entry, relative to the peak (without the cut)."""
        return (direction_bound(self.p_omega, self.k_omega, d_omega, sigma)
                + direction_bound(self.p_eta, self.k_eta, d_eta, sigma))

    def describe(self) -> str:
        def one(p, k):
            name = "midpoint" if p == 1 else f"{p}-point Gauss"
            return f"{name} x {k} panel{'s' if k > 1 else ''}" if (p > 1 or k > 1) else "midpoint"
        return f"omega {one(self.p_omega, self.k_omega)}, eta {one(self.p_eta, self.k_eta)} " \
               f"({self.n_omega * self.n_eta} node{'s' if self.n_omega * self.n_eta > 1 else ''} per bin)"


def _direction_options(width, sigma, p_max, n_max):
    """For every node count n <= n_max: the smallest bound over k panels of p points with p k = n."""
    out = []
    for n in range(1, n_max + 1):
        best = None
        for p in range(1, min(p_max, n) + 1):
            if n % p:
                continue
            b = direction_bound(p, n // p, width, sigma)
            if best is None or b < best[0]:
                best = (b, p, n // p)
        out.append((n, *best))
    return out


def choose_rule(d_omega: float, d_eta: float, sigma: float, tol: float = DEFAULT_TOL, cut_share: float = 0.0,
                p_max: int = P_MAX, n_max: int = N_MAX) -> QuadratureRule:
    """The rule with the fewest nodes per bin whose error bound is at most tol - cut_share (then the smallest bound).

    d_omega, d_eta: bin widths (radians); sigma: the smallest Gaussian width of the basis (radians);
    tol: bound on the error of a PF entry relative to the peak of a pole; cut_share: the part of it taken by the cut
    of the Gaussian (exp(-pf_cutoff_sigma^2 / 2)).
    """
    budget = float(tol) - float(cut_share)
    if budget <= 0:
        raise ValueError(f"quadrature_tol = {tol:g} is not above the error of the Gaussian's cut ({cut_share:.2g} of the "
                         "peak): raise pf_cutoff_sigma or the tolerance")
    ow = _direction_options(d_omega, sigma, p_max, n_max)
    oe = _direction_options(d_eta, sigma, p_max, n_max)
    best = None
    for nw, bw, pw, kw in ow:
        for ne, be, pe, ke in oe:
            if bw + be <= budget:
                key = (nw * ne, bw + be)
                if best is None or key < best[0]:
                    best = (key, QuadratureRule(pw, kw, pe, ke))
                break                     # more eta nodes only cost more
    if best is None:
        raise ValueError(f"no rule with at most {n_max} nodes per direction meets quadrature_tol = {tol:g} for bins of "
                         f"{np.degrees(abs(d_omega)):.3g} x {np.degrees(abs(d_eta)):.3g} deg and sigma = "
                         f"{np.degrees(sigma):.3g} deg: use a larger tolerance or pass a QuadratureRule")
    return best[1]


def grid_sigma_min(grid) -> float:
    """Smallest Gaussian width (radians) of a grid's active orientations."""
    if hasattr(grid, "leaf_arrays"):
        return float(np.min(grid.leaf_arrays()[1]))
    return float(min(grid.nodes[i].sigma for i in grid.active_leaf_nodes()))


def cut_share(pf_cutoff_sigma) -> float:
    """Error of a PF entry (relative to the peak) that the cut of the Gaussian can cause: exp(-PF_CUT)."""
    return float(np.exp(-(6.0 if pf_cutoff_sigma is None else float(pf_cutoff_sigma) ** 2 / 2)))


def setup_quadrature(op, quadrature, tol, sigma_min):
    """Set an operator's angular quadrature from its N_Omega, angle_range, N_eta, eta_angle_range (radians), cfg
    and pf_cutoff_sigma.

    Sets quadrature (QuadratureRule), quadrature_bound (bound on the error of a PF entry relative to the peak,
    including the cut), N_Omega_subdivisions / N_eta_subdivisions (nodes per bin and direction), angles_subdivided
    and eta_angles_subdivided (the nodes, bin by bin), quad_weights_cpu ((S_omega * S_eta,) float32 product weights,
    None for the midpoint rule, whose kernels average) and quad_options (compile options of the PF kernels).
    """
    cfg = op.cfg
    a0, a1 = float(op.angle_range[0]), float(op.angle_range[1])
    e0, e1 = float(op.eta_angle_range[0]), float(op.eta_angle_range[1])
    delta = (a1 - a0) / op.N_Omega
    eta_delta = (e1 - e0) / op.N_eta
    share = cut_share(getattr(op, "pf_cutoff_sigma", None))
    given = [k for k in ("N_Omega_subdivisions", "N_eta_subdivisions") if k in cfg]
    if isinstance(quadrature, QuadratureRule):
        rule = quadrature
    elif quadrature == "midpoint":
        rule = QuadratureRule.midpoint(cfg.get("N_Omega_subdivisions", 1), cfg.get("N_eta_subdivisions", 1))
    elif quadrature == "auto":
        rule = choose_rule(delta, eta_delta, sigma_min, tol, share)
        if given and any(int(cfg[k]) != 1 for k in given):
            warnings.warn(f"{' and '.join(given)} are used only with quadrature='midpoint'; quadrature='auto' chose "
                          f"{rule.describe()}", stacklevel=3)
    else:
        raise ValueError(f"quadrature must be 'auto', 'midpoint' or a QuadratureRule, not {quadrature!r}")
    op.quadrature = rule
    op.quadrature_tol = float(tol)
    op.quadrature_bound = rule.bound(delta, eta_delta, sigma_min) + share
    S, S_eta = rule.n_omega, rule.n_eta
    op.N_Omega_subdivisions, op.N_eta_subdivisions = S, S_eta
    if rule.p_omega == 1:   # equal sub-bins: the expression of earlier versions (same angles to the last bit)
        op.angles_subdivided = np.linspace(a0, a1, op.N_Omega * S, endpoint=False) + delta / S / 2
    else:
        x, _ = nodes_weights(rule.p_omega, rule.k_omega)
        op.angles_subdivided = (a0 + (np.arange(op.N_Omega)[:, None] + x[None, :]) * delta).ravel()
    if rule.p_eta == 1:
        op.eta_angles_subdivided = np.linspace(e0, e1, op.N_eta * S_eta, endpoint=False) + eta_delta / S_eta / 2
    else:
        x, _ = nodes_weights(rule.p_eta, rule.k_eta)
        op.eta_angles_subdivided = (e0 + (np.arange(op.N_eta)[:, None] + x[None, :]) * eta_delta).ravel()
    if rule.is_midpoint:
        op.quad_weights_cpu = None
        op.quad_options = []
    else:
        _, ww = nodes_weights(rule.p_omega, rule.k_omega)
        _, we = nodes_weights(rule.p_eta, rule.k_eta)
        op.quad_weights_cpu = np.ascontiguousarray((ww[:, None] * we[None, :]).ravel(), dtype=np.float32)
        op.quad_options = ["-DQUAD_WEIGHTED"]
    if getattr(op, "verbose", False):
        print(f"quadrature: {rule.describe()}; error bound {100 * op.quadrature_bound:.2g} % of the peak "
              f"(sigma {np.degrees(sigma_min):.3g} deg, bins {np.degrees(abs(delta)):.3g} x "
              f"{np.degrees(abs(eta_delta)):.3g} deg)")
    return rule
