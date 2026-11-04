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


GEOMETRY_PARAMETERS = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength", "p2", "p5", "p4", "p0", "phi0",
                       "p1", "phi1", "parallax")


class DetectorGeometry:
    """A flat detector with pyFAI's geometry (dist, poni1, poni2 in m; rot1, rot2, rot3 in rad; the same formulas as
    pyFAI, so a .poni file gives the same two-theta and chi) plus the corrections of MIDAS's calibration model
    (FF_Calibration.md, CalibrantIntegratorOMP), as parameters that can be refined from the data:

        radial distortion   tan(2theta) -> tan(2theta) (1 + D),  D = p2 r^2 + p5 r^4 + p4 r^6
                            + p0 r^2 cos(2 eta + phi0) + p1 r^4 cos(4 eta + phi1),   r = tan(2theta) / tan_max
        parallax            a photon converting at depth in the sensor lands farther out: R = L tan(2theta)
                            + parallax sin(2theta) (MIDAS), so the true angle of a pixel at radius R is smaller
        wavelength          (m), sets the Bragg angles of the rings (Material) - not used for the pixel angles

    eta = eta_sign * chi + eta_offset (DetectorPixels). All corrections are 0 by default (= pyFAI).
    """

    def __init__(self, dist, poni1, poni2, rot1=0.0, rot2=0.0, rot3=0.0, pixel1=75e-6, pixel2=75e-6, shape=(2048, 2048),
                 wavelength=None, tan_max=None, eta_offset=np.pi, eta_sign=1.0, **corrections):
        self.params = {"dist": float(dist), "poni1": float(poni1), "poni2": float(poni2), "rot1": float(rot1),
                       "rot2": float(rot2), "rot3": float(rot3),
                       "wavelength": float(wavelength) if wavelength is not None else float("nan")}
        for k in ("p0", "p1", "p2", "p4", "p5", "phi0", "phi1", "parallax"):
            self.params[k] = float(corrections.pop(k, 0.0))
        if corrections:
            raise ValueError(f"unknown geometry parameters {sorted(corrections)}")
        self.pixel1, self.pixel2 = float(pixel1), float(pixel2)
        self.shape = tuple(int(s) for s in shape)
        self.eta_offset, self.eta_sign = float(eta_offset), float(eta_sign)
        self.tan_max = tan_max

    @classmethod
    def from_pyfai(cls, ai, shape=None, **kw):
        if isinstance(ai, str):
            import pyFAI
            ai = pyFAI.load(ai)
        if shape is None:
            shape = tuple(ai.detector.shape)
        return cls(ai.dist, ai.poni1, ai.poni2, ai.rot1, ai.rot2, ai.rot3, ai.detector.pixel1, ai.detector.pixel2,
                   shape, wavelength=ai.wavelength, **kw)

    def copy(self, **changes):
        g = DetectorGeometry.__new__(DetectorGeometry)
        g.__dict__ = {**self.__dict__, "params": {**self.params, **{k: float(v) for k, v in changes.items()}}}
        return g

    def _tilted(self, index, p):
        i, j = np.divmod(np.asarray(index, dtype=np.int64), self.shape[1])
        p1 = (i + 0.5) * self.pixel1 - p["poni1"]
        p2 = (j + 0.5) * self.pixel2 - p["poni2"]
        L = p["dist"]
        c1, s1 = np.cos(p["rot1"]), np.sin(p["rot1"])
        c2, s2 = np.cos(p["rot2"]), np.sin(p["rot2"])
        c3, s3 = np.cos(p["rot3"]), np.sin(p["rot3"])
        t1 = p1 * c2 * c3 + p2 * (c3 * s1 * s2 - c1 * s3) - L * (c1 * c3 * s2 + s1 * s3)
        t2 = p1 * c2 * s3 + p2 * (c1 * c3 + s1 * s2 * s3) - L * (-c3 * s1 + c1 * s2 * s3)
        t3 = p1 * s2 - p2 * c2 * s1 + L * c1 * c2
        return t1, t2, t3

    def angles(self, index, params=None):
        """(two_theta, eta) in radians at the centres of the frame pixels `index` (flat), with the corrections."""
        p = self.params if params is None else {**self.params, **params}
        t1, t2, t3 = self._tilted(index, p)
        tan2t = np.hypot(t1, t2) / t3
        chi = np.arctan2(t1, t2)
        eta = np.mod(self.eta_sign * chi + self.eta_offset, 2 * np.pi)
        tmax = self.tan_max if self.tan_max is not None else float(np.max(tan2t))
        if self.tan_max is None:
            self.tan_max = tmax
        if p["p0"] or p["p1"] or p["p2"] or p["p4"] or p["p5"]:
            r = tan2t / tmax
            r2 = r * r
            D = (p["p2"] * r2 + p["p5"] * r2 * r2 + p["p4"] * r2 * r2 * r2
                 + p["p0"] * r2 * np.cos(2 * eta + p["phi0"]) + p["p1"] * r2 * r2 * np.cos(4 * eta + p["phi1"]))
            tan2t = tan2t * (1.0 + D)
        two_theta = np.arctan(tan2t)
        if p["parallax"]:
            # L tan(2t) + par sin(2t) = L tan(2t_geom): two Newton steps from the geometric angle
            Lp = p["dist"]
            target = Lp * tan2t
            tt = two_theta
            for _ in range(2):
                f = Lp * np.tan(tt) + p["parallax"] * np.sin(tt) - target
                df = Lp / np.cos(tt) ** 2 + p["parallax"] * np.cos(tt)
                tt = tt - f / df
            two_theta = tt
        return two_theta, eta

    def jacobian(self, index, names, steps=None):
        """d two_theta / d param and d eta / d param at the pixels (central differences), as dicts of arrays."""
        default = {"dist": 1e-6, "poni1": 1e-7, "poni2": 1e-7, "rot1": 1e-6, "rot2": 1e-6, "rot3": 1e-6,
                   "p0": 1e-5, "p1": 1e-5, "p2": 1e-5, "p4": 1e-5, "p5": 1e-5, "phi0": 1e-4, "phi1": 1e-4,
                   "parallax": 1e-6}
        steps = {**default, **(steps or {})}
        dtt, deta = {}, {}
        for n in names:
            h = steps[n]
            a = self.angles(index, {n: self.params[n] + h})
            b = self.angles(index, {n: self.params[n] - h})
            dtt[n] = (a[0] - b[0]) / (2 * h)
            de = a[1] - b[1]
            deta[n] = (de - 2 * np.pi * np.round(de / (2 * np.pi))) / (2 * h)
        return dtt, deta
