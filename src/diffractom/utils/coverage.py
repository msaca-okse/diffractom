"""
Detector coverage of the data segments (eta bin x ring window).

A segment that is partly off the detector, in a module gap or masked holds only part of the intensity of its ring
window. With ring-normalised data this mostly averages out; with unnormalised ring intensities (structure factors)
such segments are biased low and should get zero weight (or be corrected in the preprocessing). The coverage of a
segment is the solid angle of its valid pixels over the solid angle of the full segment,

    coverage = sum_{valid pixels p in segment} Omega_p  /  ( delta_eta (cos 2theta_lo - cos 2theta_hi) ),

with 2theta_lo, 2theta_hi the edges of the ring window. Computed from per-pixel arrays, so it works with any
integration package (pyFAI: Geometry.center_array, solidAngleArray; see segment_coverage_pyfai).
"""
from __future__ import annotations

import numpy as np


def segment_coverage(two_theta, eta_deg, solid_angle, valid, ring_two_theta_edges, eta_edges_deg):
    """Coverage (N_eta, N_rings) of every segment.

    two_theta : per-pixel 2theta (radians), any shape; eta_deg : per-pixel azimuth in the convention of the eta bins;
    solid_angle : per-pixel solid angle (steradian, absolute); valid : per-pixel bool (False: masked or gap);
    ring_two_theta_edges : (N_rings, 2) lower and upper 2theta (radians) of every ring window;
    eta_edges_deg : (N_eta + 1,) edges of the eta bins.
    """
    tt = np.asarray(two_theta, dtype=float).ravel()
    eta = np.asarray(eta_deg, dtype=float).ravel()
    om = np.asarray(solid_angle, dtype=float).ravel()
    ok = np.asarray(valid, dtype=bool).ravel()
    edges = np.asarray(ring_two_theta_edges, dtype=float)
    eta_edges = np.asarray(eta_edges_deg, dtype=float)
    n_eta, n_rings = len(eta_edges) - 1, len(edges)
    e_bin = np.searchsorted(eta_edges, eta, side="right") - 1
    cov = np.zeros((n_eta, n_rings))
    d_eta = np.radians(np.diff(eta_edges))
    for r, (lo, hi) in enumerate(edges):
        sel = ok & (tt >= lo) & (tt < hi) & (e_bin >= 0) & (e_bin < n_eta)
        got = np.bincount(e_bin[sel], weights=om[sel], minlength=n_eta)
        cov[:, r] = got / (d_eta * (np.cos(lo) - np.cos(hi)))
    return cov


def segment_coverage_pyfai(ai, shape, mask, ring_q_nm, q_half_width_nm, eta_edges_deg):
    """segment_coverage for a pyFAI AzimuthalIntegrator: rings given by their centre q (nm^-1) and the half width of
    the window (nm^-1), eta in pyFAI's chi convention (degrees), mask: True = masked (pyFAI convention)."""
    tt = ai.center_array(shape, unit="2th_rad", scale=False)
    chi = np.degrees(ai.center_array(shape, unit="chi_rad", scale=False))
    omega = ai.solidAngleArray(shape) * ai.pixel1 * ai.pixel2 / ai.dist ** 2   # relative -> absolute (sr)
    lam_nm = ai.wavelength * 1e9
    q = np.asarray(ring_q_nm, dtype=float)
    to_tt = lambda qq: 2 * np.arcsin(np.clip(qq * lam_nm / (4 * np.pi), 0, 1))  # noqa: E731
    edges = np.stack([to_tt(q - q_half_width_nm), to_tt(q + q_half_width_nm)], axis=1)
    return segment_coverage(tt, chi, omega, ~np.asarray(mask, dtype=bool), edges, eta_edges_deg)
