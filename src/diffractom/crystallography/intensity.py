"""
Integrated ring intensities for texture tomography without normalising the rings.

What the operator predicts
--------------------------
The texture operator evaluates, for every orientation and ring, a pole figure normalised to the same integral for
every ring (`SinglePhaseForwardOperator.transfer_pole_axes_to_gpu`), at the (omega, eta) of every data point. The
measured intensity of a ring also depends on the reflection itself and on how the data were reduced. For a crystal
volume V_x in the beam, rotated about an axis normal to the beam, the integrated intensity of one reflection is
(kinematic theory, e.g. Als-Nielsen & McMorrow ch. 5; Warren ch. 4)

    E = I0 r_e^2 (lambda^3 / V_cell^2) |F_hkl|^2 V_x P L,       L = 1 / (sin 2theta |sin eta_r|)

with eta_r the azimuth of the diffracted beam measured from the rotation axis. Integrating a texture over a data bin,
the pole density is sampled at n(omega, eta), and the area element on the pole sphere is
|dn/domega x dn/deta| = cos(theta) |sin eta_r| domega deta. The |sin eta_r| cancels the one in L, so the data are
the pole figure times a factor that is the same for the whole ring:

    I_ring(omega, eta) proportional to  [sum_families m |F|^2] (lambda^3 / V_cell^2) / sin(theta)  x  PF(omega, eta)

(1 / sin theta per ring is the Lorentz factor of a rotation scan in this form; the multiplicity m enters because the
operator's pole figure has the same integral for every ring). The Debye-Waller factor is inside |F|^2
(Material.structure_factors). Polarisation and absorption are assumed corrected in the data, or negligible.

How the data were reduced adds a factor per ring (`data` below), relative to "counts", the number of
(polarisation-corrected) photons in the ring window of a data bin:

    "counts"              sum of pixel values / polarisation factor over the window             1
    "solid_angle_sum"     sum of solid-angle-normalised pixels (pyFAI correctSolidAngle, the    1 / cos^3(2theta)
                          solid angle relative to the PONI pixel; flat detector normal to the
                          beam)
    "q_bin_mean"          pyFAI integrate2d means in bins uniform in q, summed over the bins    1 / sin(theta)
                          of the ring window (the integration of the companion repositories);
                          independent of the detector geometry
    "two_theta_bin_mean"  the same with bins uniform in 2theta                                   1 / sin(2theta)
    "none"                no factor

So with the companion repositories' integration, a ring scales as m |F|^2 / sin^2(theta) (x lambda^3 / V_cell^2).
The factors are relative: one global scale (beam intensity, exposure, detector efficiency, voxel volume) remains
and is absorbed in the reconstructed coefficients. The lambda^3 / V_cell^2 factor makes the coefficients of
different phases comparable (volume fractions).

Fitting
-------
Two ways to use the prediction, both after the preprocessing below:

* One phase: multiply the data of every ring by ring_scale_factors(material, wavelength_A, model) (1 / predicted
  ring intensity) and reconstruct with the ring-normalised operator (normalized=True). Every ring then weighs about
  equally in the fit, as with the usual normalisation by the measured ring totals, but the ratios between the rings are
  the physical ones. Dividing by the measured totals instead removes from every ring its texture factor (how much of
  that ring's pole figure the rotation sweeps through the detector), which the texture model then cannot reproduce.
  Al1050 (15 % deformed), same basis, weights and solver; per-ring scale between data and fitted model (rms) and
  relative residual: background-subtracted data, structure factors 0.96-1.03 (2 %), 0.255; measured totals 0.76-2.3
  (29 %), 0.354. Data with background: structure factors 0.92-1.17 (7 %), 0.276; totals 0.61-1.30 (21 %), 0.329.
  Simulated data: +-3-6 % vs +-10-14 %. The orientation maps hardly change (dominant orientation within 1 deg in
  > 93 % of the voxels, density correlation > 0.94): the support of the peaks decides the orientations.
* Several phases, or coefficients in absolute units: normalized=False with this intensity_model; the operator
  multiplies ring r by ring_intensities(...) and the coefficients of different phases share one scale. Rings then
  weigh by their intensity in a least-squares fit; per-segment weights can compensate.

Preprocessing that unnormalised intensities need (with normalised rings the errors mostly cancel; here they do not):

* Background. Under the rings lies the incoherent (Compton) and thermal diffuse scattering of the sample itself, plus
  air scattering. Its share of a ring window grows steeply with q, because the diffuse scattering grows with q while
  Bragg intensity falls (form factor, Debye-Waller, Lorentz). On Al1050 at 35 keV it was 4 % of the 111 window and
  70 % of the outermost window, 70-88 % of it from the sample, and its level matched Compton + thermal diffuse
  scattering of Al without a free parameter (within 0.9-1.9x). Subtract it per data point in the reduction, e.g.
  linearly in 2theta between narrow bands at the two edges of each ring window. Left in, it inflates the outer rings
  (+12-16 % on Al1050), biases a fitted Debye-Waller factor low and acts as a uniform orientation component in the
  reconstruction (per-voxel texture 20 % less sharp).
* Detector coverage. Segments (eta bin x ring window) partly off the detector or in module gaps hold only part of
  their intensity: give them zero weight (diffractom.utils.segment_coverage).
* Debye-Waller factor: set it (Material.set_displacement) if the CIF has no displacement parameters; the outer rings
  depend on it (Al at room temperature: B = 0.85 A^2).
* Form factors: the simulated data of the companion repository were rendered with xfab's form factor table, whose
  constant term is one electron low for Al (and wrong for several other elements); Material.anomalous = {"Al": (-1, 0)}
  reproduces it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LORENTZ = ("rotation", "none")
DATA_CONVENTIONS = ("counts", "solid_angle_sum", "q_bin_mean", "two_theta_bin_mean", "none")


@dataclass
class IntensityModel:
    """How measured ring intensities relate to |F|^2 (see the module docstring).

    lorentz : "rotation" (1/sin theta: a rotation scan, axis normal to the beam) or "none".
    data : the reduction convention of the data, one of DATA_CONVENTIONS.
    polarization : None if the data are polarisation-corrected (pyFAI polarization_factor); "unpolarized" applies
        (1 + cos^2 2theta) / 2 per ring (unpolarised source, data not corrected). Linear polarisation varies with
        eta and must be corrected in the data.
    absolute : include lambda^3 / V_cell^2, so that coefficients of different phases are comparable.
    """
    lorentz: str = "rotation"
    data: str = "counts"
    polarization: str | None = None
    absolute: bool = True

    def __post_init__(self):
        if self.lorentz not in LORENTZ:
            raise ValueError(f"lorentz must be one of {LORENTZ}, got {self.lorentz!r}")
        if self.data not in DATA_CONVENTIONS:
            raise ValueError(f"data must be one of {DATA_CONVENTIONS}, got {self.data!r}")
        if self.polarization not in (None, "unpolarized"):
            raise ValueError("polarization must be None (data corrected) or 'unpolarized'")

    def factor(self, two_theta) -> np.ndarray:
        """The per-ring factor multiplying m |F|^2 (without lambda^3 / V_cell^2), for 2theta in radians."""
        tt = np.asarray(two_theta, dtype=float)
        th = tt / 2.0
        f = np.ones_like(tt)
        if self.lorentz == "rotation":
            f = f / np.sin(th)
        if self.data == "solid_angle_sum":
            f = f / np.cos(tt) ** 3
        elif self.data == "q_bin_mean":
            f = f / np.sin(th)
        elif self.data == "two_theta_bin_mean":
            f = f / np.sin(tt)
        if self.polarization == "unpolarized":
            f = f * (1.0 + np.cos(tt) ** 2) / 2.0
        return f


def as_intensity_model(model) -> IntensityModel:
    """An IntensityModel from None (defaults), a dict of its fields, or an IntensityModel."""
    if model is None:
        return IntensityModel()
    if isinstance(model, IntensityModel):
        return model
    if isinstance(model, dict):
        return IntensityModel(**model)
    raise TypeError(f"intensity_model must be an IntensityModel, a dict or None, not {type(model).__name__}")


def reflection_intensities(material, wavelength_A: float, model=None) -> np.ndarray:
    """Integrated intensity of every reflection family of a material: m |F|^2 x model.factor(2theta)
    (x lambda^3 / V_cell^2 if model.absolute), in electrons^2 per Å^3 (relative units). NaN if the material has
    no atomic basis (built from lattice parameters only)."""
    model = as_intensity_model(model)
    r = material.reflections
    if "sf_squared" not in r.dtype.names or np.any(np.isnan(r["sf_squared"])):
        return np.full(len(r), np.nan)
    out = r["multiplicity"].astype(float) * r["sf_squared"] * model.factor(r["two_theta"])
    if model.absolute:
        out = out * float(wavelength_A) ** 3 / material.volume ** 2
    return out


def group_reflections_into_rings(material, rtol=1e-6):
    """
    Group the reflections of a material into rings of equal two-theta, sorted
    by increasing two-theta. Returns a list with, per ring, the indices of its
    reflections, e.g. [[0], [1], ..., [9, 10], ...] when (333) and (511) share a ring.
    """
    tt = np.asarray(material.reflections["two_theta"], dtype=np.float64)
    rings = []
    for i in np.argsort(tt, kind="stable"):
        if rings and np.isclose(tt[i], tt[rings[-1][0]], rtol=rtol, atol=0.0):
            rings[-1].append(int(i))
        else:
            rings.append([int(i)])
    return rings


def ring_intensities(material, rings, wavelength_A: float, model=None) -> np.ndarray:
    """Per ring (lists of reflection indices, as group_reflections_into_rings; None: the material's rings, as the
    operator groups them), the summed reflection_intensities."""
    if rings is None:
        rings = group_reflections_into_rings(material)
    I = reflection_intensities(material, wavelength_A, model)
    return np.array([I[list(r)].sum() for r in rings])


def ring_scale_factors(material, wavelength_A: float, model=None, rings=None) -> np.ndarray:
    """Factors that put the measured rings on a common scale with physical ring ratios: 1 / ring_intensities,
    normalised to a geometric mean of 1. Multiply the data of ring r by factor[r] (data[..., r] for data shaped
    (N_Omega, My, N_eta, N_rings)) and use the ring-normalised operator (normalized=True), instead of dividing every
    ring by its measured total (see "Fitting" in the module docstring)."""
    s = ring_intensities(material, rings, wavelength_A, model)
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("ring intensities are not all finite and positive (material without an atomic basis?)")
    f = 1.0 / s
    return f / np.exp(np.mean(np.log(f)))
