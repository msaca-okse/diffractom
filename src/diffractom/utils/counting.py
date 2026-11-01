"""Counting statistics of integrated diffraction data: the gain of a data point (variance / mean, in data units
per detected photon), as FISTAPoisson needs it.

A photon-counting detector gives Poisson counts c_p per pixel. pyFAI's integrate2d returns per (azimuthal, radial)
bin I = sum_p c_p / sum_p n_p, with n_p the pixel's normalisation (solid angle x polarization x flat field ...).
For counts proportional to n_p within the bin (a smooth signal), E[I] = lambda and var[I] = lambda / sum_p n_p, so
the bin has the gain 1 / sum_p n_p. Summing the radial bins of a ring window with a flat profile gives the mean of
their gains. Later reduction steps change the gain as follows:

    summing data points (frames, omega bins)        gain unchanged
    averaging n repeated measurements               gain / n
    multiplying the data by s (e.g. ring scaling)   gain x s

so for data summed over omega bins, with repeated translations averaged and the rings rescaled, the gain of a data
point (translation y, eta bin, ring r) is s_r * g(eta, r) / n_repeats(y).
"""
import numpy as np


def pyfai_gain(ai, mask, npt_rad, npt_azim, ring_windows, unit="q_nm^-1", radial_range=None, azimuth_range=None,
               polarization_factor=None, method=("no", "csr", "cython"), **integrate_kwargs):
    """
    Gain of ring-window sums of pyFAI integrate2d bins, per (azimuthal bin, ring).

    ai : pyFAI AzimuthalIntegrator (the geometry of the integration)
    mask : the detector mask of the integration (nonzero = masked)
    npt_rad, npt_azim, unit, radial_range, azimuth_range, polarization_factor, integrate_kwargs : exactly as passed to
        integrate2d for the data (they set the bins and the normalisation); method only needs to reproduce the
        pixel splitting ("no" splitting: every pixel counted in one bin).
    ring_windows : (lo, hi) of every ring in the radial unit; a ring is the sum of the radial bins with
        lo <= centre < hi.
    returns (gain, n_pixels): gain (npt_azim, n_rings), the mean over the window's non-empty radial bins of
        1 / sum(normalisation) (NaN where a window has no pixels), and the number of pixels per (azimuthal bin, ring).
    """
    ones = np.ones(ai.detector.shape, np.float32)
    kw = dict(npt_rad=npt_rad, npt_azim=npt_azim, unit=unit, mask=mask, method=method,
              polarization_factor=polarization_factor, **integrate_kwargs)
    if radial_range is not None:
        kw["radial_range"] = tuple(radial_range)
    if azimuth_range is not None:
        kw["azimuth_range"] = tuple(azimuth_range)
    res = ai.integrate2d_ng(ones, **kw)
    sum_norm = np.asarray(res.sum_normalization, dtype=np.float64)
    count = np.asarray(res.count, dtype=np.float64)
    if sum_norm.ndim == 3:
        sum_norm = sum_norm[..., 0]
    if count.ndim == 3:
        count = count[..., 0]
    radial = np.asarray(res.radial)
    if sum_norm.shape != (npt_azim, len(radial)):
        sum_norm, count = sum_norm.T, count.T
    gain = np.full((npt_azim, len(ring_windows)), np.nan)
    npix = np.zeros((npt_azim, len(ring_windows)))
    for r, (lo, hi) in enumerate(ring_windows):
        sel = (radial >= lo) & (radial < hi)
        sn = sum_norm[:, sel]
        ok = sn > 0
        g = np.where(ok, 1.0 / np.where(ok, sn, 1.0), 0.0)
        nb = ok.sum(axis=1)
        gain[:, r] = np.where(nb > 0, g.sum(axis=1) / np.maximum(nb, 1), np.nan)
        npix[:, r] = count[:, sel].sum(axis=1)
    return gain, npix


def gain_from_pairs(a, b, axis=None, max_mean=None):
    """
    Empirical gain from two measurements a, b of the same quantities (e.g. a translation recorded twice):
    sum((a - b)^2 / 2) / sum((a + b) / 2) over `axis` (None: all). Data points whose mean exceeds max_mean are left
    out (bright spots, where scatter between the measurements other than counting noise dominates). Includes any
    non-Poisson scatter; compare with pyfai_gain.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = 0.5 * (a + b)
    v = 0.5 * (a - b) ** 2
    ok = np.isfinite(m) & np.isfinite(v)
    if max_mean is not None:
        ok &= m <= max_mean
    return np.where(ok, v, 0).sum(axis=axis) / np.where(ok, m, 0).sum(axis=axis)
