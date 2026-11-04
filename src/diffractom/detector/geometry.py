"""Per-pixel geometry of an area detector: two-theta, the azimuth eta in the operator's convention, and the pixel's
intensity factor (solid angle x polarisation), from a pyFAI calibration.

The forward operator (SinglePhaseForwardOperator) probes the scattering vector of a ray that leaves the sample at the
Bragg angle of a ring in the detector direction  cos(eta) d0 + sin(eta) d90  (cfg detector_direction_origin /
detector_direction_positive_90, perpendicular to the beam). pyFAI's azimuth chi is measured in its own laboratory
frame; the two are related by eta = eta_sign * chi + eta_offset. For an integration made with pyFAI (integrate2d,
azimuth range [-180, 180) deg) whose bins are used as the operator's eta bins over cfg["eta_angle_range"] = [0, 360],
eta = chi + 180 deg (eta_offset = pi, eta_sign = +1): bin i of the integration is bin i of the operator. That is the
default here, so that a raw-detector model sees the detector exactly as an operator built for such integrated data.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectorPixels:
    """The used pixels of an area detector, flattened (index into the frame: ``index``).

    two_theta : (P,) float64, radians, at the pixel centre
    eta : (P,) float64, radians in [0, 2 pi), the operator's azimuth
    factor : (P,) float64, the pixel's relative response to an isotropic intensity per solid angle: solid angle
        (relative to that of a pixel at the point of normal incidence, as pyFAI's solidAngleArray) x polarisation
    index : (P,) int64, flat index of the pixel in a frame
    shape : the frame shape
    eta_extent : (P,) float64, the pixel's extent in eta (radians; pixel size / radius), for the model's eta nodes
    two_theta_extent : (P,) float64, the pixel's extent in two-theta (radians)
    """
    two_theta: np.ndarray
    eta: np.ndarray
    factor: np.ndarray
    index: np.ndarray
    shape: tuple
    eta_extent: np.ndarray
    two_theta_extent: np.ndarray

    @property
    def n_pixels(self):
        return len(self.index)

    def subset(self, keep):
        """The pixels where keep (boolean over the used pixels) is True."""
        keep = np.asarray(keep, dtype=bool)
        return DetectorPixels(self.two_theta[keep], self.eta[keep], self.factor[keep], self.index[keep], self.shape,
                              self.eta_extent[keep], self.two_theta_extent[keep])

    def in_windows(self, two_theta_rings, half_widths):
        """Boolean over the used pixels: within half_widths (radians, scalar or per ring) of a ring's two-theta."""
        tt = np.atleast_1d(np.asarray(two_theta_rings, dtype=np.float64))
        hw = np.broadcast_to(np.asarray(half_widths, dtype=np.float64), tt.shape)
        keep = np.zeros(self.n_pixels, dtype=bool)
        for c, h in zip(tt, hw):
            keep |= np.abs(self.two_theta - c) <= h
        return keep

    def frame_lookup(self):
        """(prod(shape),) int64: the used-pixel number of every frame pixel, -1 where the pixel is not used."""
        lut = np.full(int(np.prod(self.shape)), -1, dtype=np.int64)
        lut[self.index] = np.arange(self.n_pixels)
        return lut

    @classmethod
    def from_pyfai(cls, ai, shape=None, mask=None, polarization_factor=None, eta_offset=np.pi, eta_sign=1.0,
                   two_theta_range=None, exclude_eta=None):
        """From a pyFAI AzimuthalIntegrator (or the path of a .poni file).

        mask : boolean frame, True = not used (pyFAI convention)
        polarization_factor : pyFAI's (+1 horizontal, -1 vertical, 0 circular, None: no polarisation factor)
        eta_offset, eta_sign : eta = eta_sign * chi + eta_offset (module docstring)
        two_theta_range : (min, max) radians: only pixels in this range
        exclude_eta : list of (eta_min, eta_max) radians (operator convention): pixels left out (e.g. near the
            rotation axis, where the Lorentz factor diverges)
        """
        if isinstance(ai, str):
            import pyFAI
            ai = pyFAI.load(ai)
        if shape is None:
            shape = tuple(ai.detector.shape)
        shape = tuple(int(s) for s in shape)
        tth = np.asarray(ai.center_array(shape, unit="2th_rad"), dtype=np.float64)
        chi = np.asarray(ai.center_array(shape, unit="chi_rad"), dtype=np.float64)
        sa = np.asarray(ai.solidAngleArray(shape), dtype=np.float64)
        factor = sa.copy()
        if polarization_factor is not None:
            factor *= np.asarray(ai.polarization(shape, factor=polarization_factor), dtype=np.float64)
        # extents from the corner positions (pyFAI: corner_array(unit) -> (..., 4, 2) with (radial, chi))
        corners = np.asarray(ai.corner_array(shape, unit="2th_rad"), dtype=np.float64)
        tth_c = corners[..., 0]
        chi_c = np.unwrap(corners[..., 1], axis=-1)
        tth_ext = tth_c.max(axis=-1) - tth_c.min(axis=-1)
        chi_ext = chi_c.max(axis=-1) - chi_c.min(axis=-1)
        eta = np.mod(eta_sign * chi + eta_offset, 2 * np.pi)
        use = np.isfinite(tth) & np.isfinite(chi) & (factor > 0)
        if mask is not None:
            use &= ~np.asarray(mask, dtype=bool).reshape(shape)
        if two_theta_range is not None:
            use &= (tth >= two_theta_range[0]) & (tth <= two_theta_range[1])
        if exclude_eta is not None:
            for lo, hi in exclude_eta:
                use &= ~((eta >= np.mod(lo, 2 * np.pi)) & (eta <= np.mod(hi, 2 * np.pi)))
        idx = np.flatnonzero(use.ravel())
        f = lambda a: np.ascontiguousarray(a.ravel()[idx])
        return cls(f(tth), f(eta), f(factor), idx.astype(np.int64), shape, f(chi_ext), f(tth_ext))
