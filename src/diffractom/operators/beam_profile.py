"""The profile of the pencil beam across the translation direction (the projector's detector coordinate).

The native projector's default kernel ("trapezoid") treats the beam as a box exactly one translation step wide: a
pixel's footprint is averaged over its detector bin. A real beam is set by slits (often wider than the step) and its
edges are blurred (slit penumbra, optics). With a BeamProfile the projector weight becomes the pixel footprint
convolved with the profile,

    K_a(d) = (1 / A_a) * integral_{-A_a/2}^{A_a/2} b(d - x) dx,      d = t - s (detector bins),

A_a the spacing of neighbouring pixel centres on the detector at angle a (as for the trapezoid), b the beam profile
with unit area: the data scale is unchanged, only the shape of the response. b = BeamProfile(width=1) gives the
trapezoid. All lengths are in detector bins, i.e. translation steps.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _gauss_cdf_integral(x, sigma):
    """Psi(x) = integral_{-inf}^x Phi(u / sigma) du (Phi the standard normal CDF); max(x, 0) for sigma = 0."""
    x = np.asarray(x, np.float64)
    if sigma <= 0:
        return np.maximum(x, 0.0)
    z = x / sigma
    return x * 0.5 * (1.0 + erf(z / np.sqrt(2.0))) + sigma * np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


class BeamProfile:
    """Beam profile across the translation direction, in detector bins (translation steps), unit area.

    BeamProfile(width, sigma): a box of the given width (the slit) convolved with a Gaussian of standard deviation sigma
    (edge blur). BeamProfile.tabulated(x, values): any profile, e.g. a measured one (values >= 0 on increasing x; it is
    normalised to unit area; linear between the points, 0 outside). BeamProfile.from_um(...) converts micrometres with
    the translation step.
    """

    def __init__(self, width: float = 1.0, sigma: float = 0.0):
        if width <= 0 or sigma < 0:
            raise ValueError("BeamProfile: width > 0 and sigma >= 0 (detector bins)")
        self.width, self.sigma = float(width), float(sigma)
        self._x = self._cum = None

    @classmethod
    def from_um(cls, width_um: float, sigma_um: float, step_um: float) -> "BeamProfile":
        return cls(width_um / step_um, sigma_um / step_um)

    @classmethod
    def tabulated(cls, x, values) -> "BeamProfile":
        x = np.asarray(x, np.float64)
        v = np.clip(np.asarray(values, np.float64), 0.0, None)
        if x.ndim != 1 or x.shape != v.shape or len(x) < 3 or np.any(np.diff(x) <= 0):
            raise ValueError("BeamProfile.tabulated: x increasing, values of the same length (>= 3)")
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(x))])
        if cum[-1] <= 0:
            raise ValueError("BeamProfile.tabulated: the profile has no positive part")
        self = cls.__new__(cls)
        self.width = self.sigma = None
        self._x, self._v, self._cum = x, v / cum[-1], cum / cum[-1]
        return self

    @property
    def is_tabulated(self) -> bool:
        return self._x is not None

    def __repr__(self):
        if self.is_tabulated:
            return f"BeamProfile.tabulated({len(self._x)} points over [{self._x[0]:.3g}, {self._x[-1]:.3g}] bins)"
        return f"BeamProfile(width={self.width:.4g}, sigma={self.sigma:.4g})"

    def half_support(self) -> float:
        """Half width (bins) outside which the profile is taken as 0 (box: w / 2 + 4.5 sigma: the Gaussian is 4e-5 of
        its peak there)."""
        if self.is_tabulated:
            return float(max(-self._x[0], self._x[-1]))
        return 0.5 * self.width + 4.5 * self.sigma

    def __call__(self, x):
        """The profile b(x) (unit area)."""
        x = np.asarray(x, np.float64)
        if self.is_tabulated:
            return np.interp(x, self._x, self._v, left=0.0, right=0.0)
        w, s = self.width, self.sigma
        if s == 0:
            return (np.abs(x) <= 0.5 * w) / w
        return (erf((x + 0.5 * w) / (np.sqrt(2) * s)) - erf((x - 0.5 * w) / (np.sqrt(2) * s))) / (2.0 * w)

    def cdf(self, x):
        """integral_{-inf}^x b."""
        x = np.asarray(x, np.float64)
        if self.is_tabulated:
            return np.interp(x, self._x, self._cum, left=0.0, right=1.0)
        w, s = self.width, self.sigma
        if s == 0:
            return np.clip((x + 0.5 * w) / w, 0.0, 1.0)
        return (_gauss_cdf_integral(x + 0.5 * w, s) - _gauss_cdf_integral(x - 0.5 * w, s)) / w

    def footprint(self, d, A):
        """K(d) = (pixel box of width A, unit area) convolved with b, at offsets d (bins)."""
        A = float(abs(A))
        return (self.cdf(np.asarray(d, np.float64) + 0.5 * A) - self.cdf(np.asarray(d, np.float64) - 0.5 * A)) / A

    def transform(self, om):
        """Fourier transform integral b(x) exp(-i om x) dx at angular frequencies om (rad per bin)."""
        om = np.asarray(om, np.float64)
        if self.is_tabulated:
            x = np.linspace(self._x[0], self._x[-1], max(4097, 16 * len(self._x)))
            b = self(x)
            return _trapz(b[None, :] * np.exp(-1j * om.reshape(-1, 1) * x[None, :]), x, axis=1).reshape(om.shape)
        return np.sinc(self.width * om / (2 * np.pi)) * np.exp(-0.5 * (self.sigma * om) ** 2)

    def tables(self, A, per_bin: int = 256):
        """Footprint tables for the projector: K_a(d) for every pixel spacing A_a (bins) on PNT points spanning
        [-PHW, PHW]. Returns (tables (R, PNT) float32, PHW, PNT)."""
        A = np.abs(np.asarray(A, np.float64))
        phw = float(0.5 * A.max() + self.half_support())
        pnt = int(np.ceil(2 * phw * per_bin)) + 1
        d = np.linspace(-phw, phw, pnt)
        tab = np.stack([self.footprint(d, a) for a in A]).astype(np.float32)
        tab[:, 0] = tab[:, -1] = 0.0
        return np.ascontiguousarray(tab), phw, pnt


def as_beam_profile(beam) -> BeamProfile | None:
    """None, a BeamProfile, a width (bins) or (width, sigma) in bins."""
    if beam is None or isinstance(beam, BeamProfile):
        return beam
    if np.isscalar(beam):
        return BeamProfile(float(beam))
    w, s = beam
    return BeamProfile(float(w), float(s))
