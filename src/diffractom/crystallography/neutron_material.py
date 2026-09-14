"""Neutron crystallographic material for Bragg-edge transmission modelling.

Stores atomic positions, neutron coherent scattering lengths, and
Debye-Waller parameters for a *cubic* crystal.  The only public factory
inputs are:

* ``a``       – cubic lattice parameter in Å
* ``atoms``   – (N_atoms, 5) array: [x, y, z (fractional), b_coh (fm), u² (Å²)]

Example — 316L stainless steel (FCC, SLD-averaged alloy)::

    import numpy as np
    from diffractom.crystallography.neutron_material import NeutronMaterial

    mat = NeutronMaterial(
        a=3.596,
        atoms=np.array([
            [0.0, 0.0, 0.0, 9.2, 0.0083],
            [0.5, 0.5, 0.0, 9.2, 0.0083],
            [0.0, 0.5, 0.5, 9.2, 0.0083],
            [0.5, 0.0, 0.5, 9.2, 0.0083],
        ]),
        name="316L",
        number_density=0.0833,  # atoms / Å³  (optional, for cross-section scaling)
    )
    table = mat.hkl_table(h_max=10)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NeutronMaterial:
    """Cubic crystal defined by neutron scattering parameters.

    Parameters
    ----------
    a : float
        Cubic lattice parameter in Å.
    atoms : np.ndarray, shape (N_atoms, 5)
        Each row: [x, y, z, b_coh, u²]
        * x, y, z  – fractional coordinates in the unit cell [0, 1)
        * b_coh    – coherent scattering length in fm
        * u²       – mean-square displacement ⟨u²⟩ in Å² (isotropic Debye-Waller)
    name : str
        Human-readable label.
    number_density : float
        Number of formula units per Å³ (N/V), used to scale incoherent
        background contributions if needed.  Defaults to 0.0.
    """

    a: float
    atoms: np.ndarray
    name: str = "material"
    number_density: float = 0.0

    def __post_init__(self):
        self.atoms = np.asarray(self.atoms, dtype=np.float64)
        if self.atoms.ndim != 2 or self.atoms.shape[1] != 5:
            raise ValueError("atoms must have shape (N_atoms, 5): [x, y, z, b_coh, u2]")

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------

    @property
    def unit_cell_volume(self) -> float:
        """Unit-cell volume V in Å³."""
        return self.a ** 3

    def d_spacing(self, h: np.ndarray, k: np.ndarray, l: np.ndarray) -> np.ndarray:
        """d-spacing in Å for cubic lattice: d = a / sqrt(h²+k²+l²)."""
        return self.a / np.sqrt(h ** 2 + k ** 2 + l ** 2)

    # ------------------------------------------------------------------
    # Structure factor |F_hkl|²  (kinematic, neutron)
    # ------------------------------------------------------------------

    def structure_factor_sq(
        self,
        h: np.ndarray,
        k: np.ndarray,
        l: np.ndarray,
    ) -> np.ndarray:
        """Compute |F_{hkl}|² in barns for arrays of Miller indices.

        Implements the same formula as the MATLAB ``factor.m``::

            F = Σ_j  b_j · exp(-q²·u²_j/2) · exp(2πi(h·xj + k·yj + l·zj))

        where q = 2π/d.

        The common prefactor ``exp(-q²·u²₀/2)`` (Debye-Waller of the first
        atom) is factored out; each atom's *extra* DW relative to atom 0 is
        kept.  This reproduces the numerical behaviour of ``factor.m`` exactly.

        Returns
        -------
        F2 : np.ndarray
            |F|² in barns (1 fm² = 0.01 barn → multiply by 0.01).
        """
        h = np.asarray(h, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64)
        l = np.asarray(l, dtype=np.float64)

        d = self.d_spacing(h, k, l)          # (N_hkl,)
        q = 2.0 * np.pi / d                  # (N_hkl,)
        q2 = q ** 2                           # (N_hkl,)

        # Atom-0 Debye-Waller (shared prefactor, same as factor.m)
        u2_0 = self.atoms[0, 4]
        dw_common = np.exp(-q2 * u2_0 * 0.5)  # (N_hkl,)

        xre = np.zeros_like(q2)
        xim = np.zeros_like(q2)

        for atom in self.atoms:
            xj, yj, zj, bj, u2j = atom
            phase = 2.0 * np.pi * (h * xj + k * yj + l * zj)
            extra_dw = np.exp(-q2 * (u2j - u2_0) * 0.5)
            xre += dw_common * bj * extra_dw * np.cos(phase)
            xim += dw_common * bj * extra_dw * np.sin(phase)

        # 0.01 converts fm² → barn
        return (xre ** 2 + xim ** 2) * 0.01

    # ------------------------------------------------------------------
    # HKL table
    # ------------------------------------------------------------------

    def hkl_table(
        self,
        h_max: int = 10,
        threshold: float = 1e-3,
        lam_min: float = -np.inf,
        lam_max: float = np.inf,
    ) -> np.ndarray:
        """Enumerate unique (h,k,l) planes with nonzero structure factor.

        Mirrors ``genera_indices_2022.m``: generates all integer triples
        $(h,k,l) \in [-h_\text{max}, h_\text{max}]^3 \setminus \{0,0,0\}$,
        computes $|F|^2$, removes extinct reflections, and deduplicates.

        Parameters
        ----------
        h_max : int
            Maximum absolute Miller index (inclusive).
        threshold : float
            Minimum |F|² (barns) to keep a reflection.
        lam_min, lam_max : float
            Optional wavelength filter in Å applied to the Bragg wavelength
            λ = 2d.  Default: keep everything.

        Returns
        -------
        table : np.ndarray, shape (N_hkl, 5)
            Columns: ``[h, k, l, F2_barns, d_Angstrom]``, sorted by d
            descending (largest d / longest wavelength first).
        """
        idx = np.arange(-h_max, h_max + 1)
        hh, kk, ll = np.meshgrid(idx, idx, idx, indexing="ij")
        hh = hh.ravel().astype(np.float64)
        kk = kk.ravel().astype(np.float64)
        ll = ll.ravel().astype(np.float64)

        # Remove (0, 0, 0)
        nonzero = (hh != 0) | (kk != 0) | (ll != 0)
        hh, kk, ll = hh[nonzero], kk[nonzero], ll[nonzero]

        F2 = self.structure_factor_sq(hh, kk, ll)

        # Filter extinct reflections
        active = F2 > threshold
        hh, kk, ll, F2 = hh[active], kk[active], ll[active], F2[active]

        d = self.d_spacing(hh, kk, ll)

        # Optional wavelength filter (Bragg edge at λ = 2d)
        if lam_min > -np.inf or lam_max < np.inf:
            lam_bragg = 2.0 * d
            in_range = (lam_bragg >= lam_min) & (lam_bragg <= lam_max)
            hh, kk, ll, F2, d = (
                hh[in_range], kk[in_range], ll[in_range],
                F2[in_range], d[in_range],
            )

        # Deduplicate: keep unique (h, k, l) combos
        # Stack as integers for fast uniqueness check
        hkl_int = np.stack([hh, kk, ll], axis=1).astype(np.int32)
        _, unique_idx = np.unique(hkl_int, axis=0, return_index=True)
        hh  = hh[unique_idx]
        kk  = kk[unique_idx]
        ll  = ll[unique_idx]
        F2  = F2[unique_idx]
        d   = d[unique_idx]

        # Sort by d descending
        order = np.argsort(-d)
        table = np.column_stack([hh[order], kk[order], ll[order], F2[order], d[order]])
        return table  # shape (N_hkl, 5)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NeutronMaterial(name={self.name!r}, a={self.a} Å, "
            f"n_atoms={len(self.atoms)}, n_density={self.number_density})"
        )

    # ------------------------------------------------------------------
    # Predefined materials
    # ------------------------------------------------------------------

    @classmethod
    def steel_316L(cls) -> "NeutronMaterial":
        """316L austenitic stainless steel (SLD-averaged FCC, a=3.596 Å).

        Atom parameters from Malamud et al. (atomos_316L_alloy.txt):
        b_coh = 9.2 fm, u² = 0.0083 Å².
        """
        atoms = np.array(
            [
                [0.0, 0.0, 0.0, 9.2, 0.0083],
                [0.5, 0.5, 0.0, 9.2, 0.0083],
                [0.0, 0.5, 0.5, 9.2, 0.0083],
                [0.5, 0.0, 0.5, 9.2, 0.0083],
            ]
        )
        return cls(
            a=3.596,
            atoms=atoms,
            name="316L",
            number_density=0.0833,
        )
