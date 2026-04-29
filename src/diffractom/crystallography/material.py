"""material.py — Crystal structure and powder diffraction material class.

Public API is fully backward-compatible with the old pymatgen-based Material
class so that all operators work without modification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .cif_parser import parse_cif, _parse_sym_op
from .form_factors import form_factor_array
from .lattice import reciprocal_lattice
from . import point_groups


# ─────────────────────────────────────────────────────────────────────────────
# Crystal-system helpers
# ─────────────────────────────────────────────────────────────────────────────

_SG_TO_CRYSTAL_SYSTEM: list[tuple[range, str]] = [
    (range(1,   3),   'triclinic'),
    (range(3,  16),   'monoclinic'),
    (range(16,  75),  'orthorhombic'),
    (range(75, 143),  'tetragonal'),
    (range(143, 168), 'trigonal'),
    (range(168, 195), 'hexagonal'),
    (range(195, 231), 'cubic'),
]


def _crystal_system_from_sg(sg: int) -> str:
    for rng, name in _SG_TO_CRYSTAL_SYSTEM:
        if sg in rng:
            return name
    raise ValueError(f"Space group number {sg} is outside the valid range 1–230.")


def _infer_crystal_system(a: float, b: float, c: float,
                          alpha: float, beta: float, gamma: float,
                          tol: float = 5e-2) -> str:
    """Infer crystal system from metric parameters (angles in degrees)."""
    def eq(x: float, y: float) -> bool:
        return abs(x - y) < tol

    all90 = eq(alpha, 90.) and eq(beta, 90.) and eq(gamma, 90.)

    if eq(a, b) and eq(b, c) and all90:
        return 'cubic'
    if eq(a, b) and eq(alpha, 90.) and eq(beta, 90.) and eq(gamma, 120.):
        return 'hexagonal'
    if eq(a, b) and all90:
        return 'tetragonal'
    if eq(a, b) and eq(b, c) and eq(alpha, beta) and eq(beta, gamma) and not eq(alpha, 90.):
        return 'trigonal'
    if all90:
        return 'orthorhombic'
    if eq(alpha, 90.) and eq(gamma, 90.):
        return 'monoclinic'
    return 'triclinic'


# ─────────────────────────────────────────────────────────────────────────────
# Lattice matrix helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lattice_matrix_from_params(a: float, b: float, c: float,
                                 alpha: float, beta: float, gamma: float) -> np.ndarray:
    """3×3 direct lattice matrix A where A[:, i] is the i-th lattice vector (Å).

    Convention (IUCr standard):
        a-vector along x
        b-vector in the xy-plane
        c-vector defined by (alpha, beta, gamma)
    """
    ar = np.deg2rad(alpha)
    br = np.deg2rad(beta)
    gr = np.deg2rad(gamma)
    A = np.zeros((3, 3))
    A[:, 0] = [a, 0.0, 0.0]
    A[:, 1] = [b * np.cos(gr), b * np.sin(gr), 0.0]
    cx = c * np.cos(br)
    cy = c * (np.cos(ar) - np.cos(br) * np.cos(gr)) / np.sin(gr)
    cz_sq = max(0.0, c*c - cx*cx - cy*cy)
    A[:, 2] = [cx, cy, np.sqrt(cz_sq)]
    return A


def _lattice_params_from_matrix_rows(M: np.ndarray):
    """Extract (a, b, c, alpha, beta, gamma) from matrix whose *rows* are lattice vectors."""
    M = np.asarray(M, dtype=float)
    a_v, b_v, c_v = M[0], M[1], M[2]
    a = np.linalg.norm(a_v)
    b = np.linalg.norm(b_v)
    c = np.linalg.norm(c_v)
    alpha = float(np.degrees(np.arccos(np.clip(np.dot(b_v, c_v) / (b * c), -1., 1.))))
    beta  = float(np.degrees(np.arccos(np.clip(np.dot(a_v, c_v) / (a * c), -1., 1.))))
    gamma = float(np.degrees(np.arccos(np.clip(np.dot(a_v, b_v) / (a * b), -1., 1.))))
    return float(a), float(b), float(c), alpha, beta, gamma


# ─────────────────────────────────────────────────────────────────────────────
# Point-group helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pg_int_matrices(pg_obj) -> list[np.ndarray]:
    """Convert a tuple of scipy Rotations to a list of 3×3 integer rotation matrices."""
    return [np.rint(r.as_matrix()).astype(int) for r in pg_obj]


def _sym_ops_to_hkl_ops(sym_op_strings: list[str]) -> list[np.ndarray]:
    """Convert CIF symmetry-operation strings to unique HKL-space integer rotation matrices.

    A direct-space operation W (in fractional coords) acts on reciprocal-space
    Miller indices as (W⁻¹)ᵀ.  Because all space-group rotations have det = ±1,
    W⁻¹ is always an integer matrix.
    """
    hkl_ops: list[np.ndarray] = []
    seen: set[tuple] = set()
    for op_str in sym_op_strings:
        rows = _parse_sym_op(op_str)          # list of 3 rows [cx, cy, cz, offset]
        W = np.array([[r[0], r[1], r[2]] for r in rows], dtype=float)
        # Skip if W is not close to an integer matrix (malformed op)
        if np.max(np.abs(W - np.rint(W))) > 0.1:
            continue
        W_int = np.rint(W).astype(int)
        det = int(round(np.linalg.det(W_int)))
        if abs(det) != 1:
            continue                           # not a valid symmetry rotation
        W_inv = np.rint(np.linalg.inv(W)).astype(int)
        W_hkl = W_inv.T
        key = tuple(W_hkl.ravel())
        if key not in seen:
            seen.add(key)
            hkl_ops.append(W_hkl)
    if not hkl_ops:
        hkl_ops.append(np.eye(3, dtype=int))  # at least the identity
    return hkl_ops


def _equiv_hkls(hkl: tuple, ops: list[np.ndarray]) -> list[tuple]:
    """All HKL equivalents under point-group ops + Friedel pairs, deduplicated."""
    v = np.asarray(hkl, dtype=int)
    seen: set[tuple] = set()
    result: list[tuple] = []
    for R in ops:
        w = tuple(int(x) for x in R @ v)
        for candidate in (w, tuple(-x for x in w)):
            if candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reflection dtype
# ─────────────────────────────────────────────────────────────────────────────

_REFLECTION_DTYPE = np.dtype([
    ('hkl',          'i4', (3,)),
    ('multiplicity',  'i4'),
    ('d_spacing',     'f8'),
    ('two_theta',     'f8'),
    ('intensity',     'f8'),
])


# ─────────────────────────────────────────────────────────────────────────────
# Input resolution helper
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_inputs(
    *,
    wavelength_kev=None,
    wavelength_A=None,
    energy_keV=None,
    min_two_theta=None,
    max_two_theta=None,
    q_min=None,
    q_max=None,
    hkl_list=None,
) -> tuple:
    """Return (q_min, q_max, wavelength_A).

    Raises ValueError if inputs are insufficient and no hkl_list is provided.
    """
    lam = None
    if wavelength_A is not None:
        lam = float(wavelength_A)
    elif energy_keV is not None:
        lam = 12.39842 / float(energy_keV)
    elif wavelength_kev is not None:
        lam = 12.39842 / float(wavelength_kev)

    if q_min is not None and q_max is not None:
        return float(q_min), float(q_max), lam

    if min_two_theta is not None and max_two_theta is not None and lam is not None:
        tt_min = float(min_two_theta)
        tt_max = float(max_two_theta)
        q_lo = (4.0 * np.pi / lam) * np.sin(tt_min / 2.0)
        q_hi = (4.0 * np.pi / lam) * np.sin(tt_max / 2.0)
        return max(0.0, q_lo), q_hi, lam

    if hkl_list is not None:
        return 0.0, np.inf, lam

    raise ValueError(
        "Insufficient input: provide q_min+q_max, or (wavelength/energy + "
        "two-theta range), or an explicit hkl_list."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Structure-factor computation (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def _xray_sf_sq(hkl_arr: np.ndarray, q_vals: np.ndarray,
                basis_frac: np.ndarray, elements: list,
                occupancies: np.ndarray) -> np.ndarray:
    """Compute |F(hkl)|² for each reflection using Cromer-Mann form factors.

    Parameters
    ----------
    hkl_arr    : (N, 3) int
    q_vals     : (N,) float — |G| in Å⁻¹ (physics, with 2π)
    basis_frac : (M, 3) float — fractional coordinates of all basis sites
    elements   : list of M element symbols
    occupancies: (M,) float — site occupancies

    Returns
    -------
    (N,) float — |F(hkl)|²
    """
    N, M = len(hkl_arr), len(elements)
    if N == 0 or M == 0:
        return np.ones(max(N, 1))

    occupancies = np.asarray(occupancies, dtype=float)

    # Form factor matrix (N, M): group by unique element to minimise evaluations
    ff_matrix: np.ndarray = np.zeros((N, M))
    _ff_cache: dict = {}
    for j, elem in enumerate(elements):
        if elem not in _ff_cache:
            _ff_cache[elem] = form_factor_array(elem, q_vals)
        ff_matrix[:, j] = _ff_cache[elem]

    # Phase matrix (N, M): 2π (h·x_j + k·y_j + l·z_j)
    phases = 2.0 * np.pi * (hkl_arr.astype(float) @ basis_frac.T)

    # Weighted structure factor
    w = occupancies * ff_matrix
    re_F = np.sum(w * np.cos(phases), axis=1)
    im_F = np.sum(w * np.sin(phases), axis=1)

    return re_F**2 + im_F**2


# ─────────────────────────────────────────────────────────────────────────────
# Material class
# ─────────────────────────────────────────────────────────────────────────────

class Material:
    """Crystallographic material container for powder diffraction.

    Can be initialised empty and populated incrementally, or built directly
    from a CIF file or from lattice parameters via the factory class methods.

    The public interface (accessor methods and array attributes) is fully
    backward-compatible with the old pymatgen-based Material class so that
    all forward operators work without changes.

    Parameters
    ----------
    name : str, optional
        Human-readable label.
    """

    _POINT_GROUP_MAP = {
        'triclinic':    point_groups.trivial,
        'monoclinic':   point_groups.cyclic_2,
        'orthorhombic': point_groups.orthorhombic,
        'tetragonal':   point_groups.tetragonal,
        'trigonal':     point_groups.trigonal,
        'hexagonal':    point_groups.hexagonal,
        'cubic':        point_groups.cubic,
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, name: str = ''):
        self.name: str = name

        # Crystallographic metadata
        self.crystal_system:     str | None = None
        self.space_group_number: int | None = None
        self.space_group_symbol: str | None = None
        self.lattice_params:     dict | None = None   # {a,b,c,alpha,beta,gamma}

        # Internal lattice matrices
        self._A: np.ndarray | None = None   # 3×3 direct  (Å, columns = lattice vectors)
        self._B: np.ndarray | None = None   # 3×3 physics reciprocal (with 2π factor)

        # Atomic basis (from CIF)
        self._basis_frac:  np.ndarray | None = None   # (M, 3) fractional coords
        self._elements:    list | None = None          # M element symbols
        self._occupancies: np.ndarray | None = None   # (M,)

        # Crystallographic HKL-space symmetry ops (from CIF sym_ops)
        # If set, these are used instead of Cartesian point-group matrices for
        # grouping equivalent reflections — necessary for non-cubic systems.
        self._hkl_grouping_ops: list[np.ndarray] | None = None

        # Reflection table (structured numpy array)
        self.reflections: np.ndarray | None = None

        # Operator-facing attributes (set by compute_h_vectors / attach_point_group)
        self.hkl:                 np.ndarray | None = None
        self.h_vecs:              np.ndarray | None = None
        self.h_vecs_normed:       np.ndarray | None = None
        self.point_group_matrices: np.ndarray | None = None
        self.num_sym_ops:          int | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_lattice(self, a: float, b: float, c: float,
                       alpha: float, beta: float, gamma: float) -> None:
        self.lattice_params = dict(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        if self.space_group_number is not None:
            self.crystal_system = _crystal_system_from_sg(self.space_group_number)
        else:
            self.crystal_system = _infer_crystal_system(a, b, c, alpha, beta, gamma)
        self._A = _lattice_matrix_from_params(a, b, c, alpha, beta, gamma)
        self._B = reciprocal_lattice(self._A)

    def _normalise_crystal_system(self) -> str:
        sys = (self.crystal_system or 'triclinic').lower()
        if 'rhombohedral' in sys:
            sys = 'trigonal'
        return sys

    # ------------------------------------------------------------------
    # Public setters (incremental construction)
    # ------------------------------------------------------------------

    def set_lattice(self, a: float, b: float, c: float,
                    alpha: float = 90.0, beta: float = 90.0, gamma: float = 90.0) -> None:
        """Set lattice parameters and rebuild lattice matrices."""
        self._setup_lattice(float(a), float(b), float(c),
                            float(alpha), float(beta), float(gamma))

    def set_space_group(self, number: int | None = None,
                        symbol: str | None = None) -> None:
        """Override space group; crystal system is updated from *number* if given."""
        self.space_group_number = number
        self.space_group_symbol = symbol
        if number is not None:
            self.crystal_system = _crystal_system_from_sg(number)

    def load_cif(self, cif_path) -> None:
        """Parse a CIF file and populate lattice + basis (no reflection computation)."""
        cif = parse_cif(cif_path)
        self.name = cif['name']
        self.space_group_number = cif['space_group_number']
        self.space_group_symbol = cif['space_group_symbol']
        self._setup_lattice(cif['a'], cif['b'], cif['c'],
                            cif['alpha'], cif['beta'], cif['gamma'])
        if len(cif['basis']) > 0:
            self._basis_frac  = cif['basis']
            self._elements    = cif['elements']
            self._occupancies = cif['occupancies']
        if cif.get('sym_ops'):
            self._hkl_grouping_ops = _sym_ops_to_hkl_ops(cif['sym_ops'])

    # ------------------------------------------------------------------
    # Factory: from CIF
    # ------------------------------------------------------------------

    @classmethod
    def from_cif(
        cls,
        cif_path,
        *,
        wavelength_kev:   float | None = None,
        wavelength_A:     float | None = None,
        energy_keV:       float | None = None,
        min_two_theta:    float | None = None,
        max_two_theta:    float | None = None,
        q_min:            float | None = None,
        q_max:            float | None = None,
        hkl_list:         Sequence | None = None,
        intensity_cutoff_fraction: float = 0.0,
        global_intensity_norm:     float | None = None,
        name:             str | None = None,
    ) -> 'Material':
        """Parse a CIF file and compute X-ray powder-diffraction reflections.

        Wavelength / energy
        -------------------
        Provide exactly one of:
        * ``wavelength_A``   — wavelength in Å
        * ``energy_keV``     — photon energy in keV   (λ = 12.39842 / E)
        * ``wavelength_kev`` — same as energy_keV (historical, confusingly named)

        Reflection selection
        --------------------
        Provide one of:
        * ``q_min`` + ``q_max``  (Å⁻¹, physics: q = 2π/d)
        * ``min_two_theta`` + ``max_two_theta`` (radians) together with a wavelength
        * ``hkl_list`` — explicit list of (h,k,l) tuples
        """
        mat = cls()
        mat.load_cif(cif_path)
        if name is not None:
            mat.name = name

        q_lo, q_hi, lam = _resolve_inputs(
            wavelength_kev=wavelength_kev, wavelength_A=wavelength_A,
            energy_keV=energy_keV,
            min_two_theta=min_two_theta, max_two_theta=max_two_theta,
            q_min=q_min, q_max=q_max, hkl_list=hkl_list,
        )
        mat.compute_reflections(
            q_min=q_lo, q_max=q_hi, wavelength_A=lam,
            hkl_list=hkl_list,
            intensity_cutoff_fraction=intensity_cutoff_fraction,
            global_intensity_norm=global_intensity_norm,
        )
        return mat

    # ------------------------------------------------------------------
    # Factory: from lattice parameters (backward-compatible with old API)
    # ------------------------------------------------------------------

    @classmethod
    def from_lattice_parameters(
        cls,
        *,
        # New API — explicit cell parameters
        a: float | None = None,
        b: float | None = None,
        c: float | None = None,
        alpha: float = 90.0,
        beta:  float = 90.0,
        gamma: float = 90.0,
        # Old API — 3×3 matrix (rows = direct lattice vectors, pymatgen convention)
        lattice_matrix:      np.ndarray | None = None,
        lattice_matrix_kind: str = 'direct',
        reciprocal_has_2pi:  bool = False,
        # Crystal system / space group
        symmetry_group:     str | None = None,   # old param name
        crystal_system:     str | None = None,   # new param name (alias)
        space_group_number: int | None = None,
        # Reflection selection
        wavelength_kev:   float | None = None,
        wavelength_A:     float | None = None,
        energy_keV:       float | None = None,
        min_two_theta:    float | None = None,
        max_two_theta:    float | None = None,
        q_min:            float | None = None,
        q_max:            float | None = None,
        hkl_list:         Sequence | None = None,
        global_intensity_norm: float | None = None,
        name: str = 'custom_lattice',
    ) -> 'Material':
        """Build a Material from lattice parameters without an atomic basis.

        Intensities are set to 1.0 (scaled by multiplicity).
        No structure factors are computed.

        Parameters
        ----------
        a, b, c : float
            Cell lengths in Å.  b defaults to a, c defaults to a.
        alpha, beta, gamma : float
            Cell angles in degrees (default 90°).
        lattice_matrix : (3, 3) array, optional
            Rows are lattice vectors (direct) or reciprocal vectors.
            Overrides a/b/c/alpha/beta/gamma when supplied.
        lattice_matrix_kind : {'direct', 'reciprocal'}
        reciprocal_has_2pi : bool
            True if the supplied reciprocal matrix already has the 2π factor.
        symmetry_group / crystal_system : str
            Crystal system used for multiplicity computation.
        space_group_number : int, optional
            Derive crystal system deterministically (takes priority over inference).
        wavelength_kev : float
            Energy in keV (λ = 12.39842 / E).
        min_two_theta, max_two_theta : float
            2θ range in radians.
        q_min, q_max : float
            q-range in Å⁻¹ (2π/d convention).
        hkl_list : sequence of (h,k,l), optional
            Explicit list of reflections.
        """
        mat = cls(name=name)

        # ---- lattice parameters ----
        if lattice_matrix is not None:
            M = np.asarray(lattice_matrix, dtype=float)
            if M.shape != (3, 3):
                raise ValueError(f"lattice_matrix must be (3, 3), got {M.shape}")
            if lattice_matrix_kind == 'direct':
                a_, b_, c_, alpha_, beta_, gamma_ = _lattice_params_from_matrix_rows(M)
            elif lattice_matrix_kind == 'reciprocal':
                rec = M / (2.0 * np.pi) if reciprocal_has_2pi else M
                try:
                    direct_rows = np.linalg.inv(rec).T
                except np.linalg.LinAlgError:
                    raise ValueError("Reciprocal lattice matrix is singular.")
                a_, b_, c_, alpha_, beta_, gamma_ = _lattice_params_from_matrix_rows(direct_rows)
            else:
                raise ValueError(f"Unknown lattice_matrix_kind: {lattice_matrix_kind!r}")
        elif a is not None:
            a_ = float(a)
            b_ = float(b) if b is not None else a_
            c_ = float(c) if c is not None else a_
            alpha_ = float(alpha); beta_ = float(beta); gamma_ = float(gamma)
        else:
            raise ValueError("Provide 'lattice_matrix' or 'a' (and optionally b, c, angles).")

        # ---- space group / crystal system ----
        mat.space_group_number = space_group_number
        if space_group_number is not None:
            mat.crystal_system = _crystal_system_from_sg(space_group_number)
        mat._setup_lattice(a_, b_, c_, alpha_, beta_, gamma_)
        # Explicit string overrides metric inference (but NOT SG-derived value)
        sys_str = crystal_system or symmetry_group
        if sys_str is not None and space_group_number is None:
            mat.crystal_system = sys_str.lower()

        # ---- reflections ----
        q_lo, q_hi, lam = _resolve_inputs(
            wavelength_kev=wavelength_kev, wavelength_A=wavelength_A,
            energy_keV=energy_keV,
            min_two_theta=min_two_theta, max_two_theta=max_two_theta,
            q_min=q_min, q_max=q_max, hkl_list=hkl_list,
        )
        mat.compute_reflections(
            q_min=q_lo, q_max=q_hi, wavelength_A=lam,
            hkl_list=hkl_list,
            intensity_cutoff_fraction=0.0,
            global_intensity_norm=global_intensity_norm,
        )
        return mat

    # ------------------------------------------------------------------
    # Core: reflection computation
    # ------------------------------------------------------------------

    def compute_reflections(
        self,
        *,
        q_min: float = 0.0,
        q_max: float,
        wavelength_A: float | None = None,
        hkl_list: Sequence | None = None,
        intensity_cutoff_fraction: float = 0.0,
        global_intensity_norm: float | None = None,
    ) -> None:
        """Enumerate reflections and compute (or estimate) intensities.

        Parameters
        ----------
        q_min, q_max : float
            q-range in Å⁻¹ (physics, 2π/d). Ignored if *hkl_list* is given.
        wavelength_A : float, optional
            Wavelength in Å, used to compute 2θ. If None, 2θ is stored as 0.
        hkl_list : sequence of (h, k, l), optional
            Use these HKLs instead of enumerating from the q range.
        intensity_cutoff_fraction : float
            Drop reflections with intensity < cutoff × max_intensity.
        global_intensity_norm : float, optional
            If given, intensities = |F|² / norm × 100 instead of norming to max.
        """
        if self._B is None:
            raise RuntimeError("Lattice not set. Call set_lattice() or load_cif() first.")

        B = self._B

        # ── 1. Enumerate candidate HKLs ───────────────────────────────────────
        if hkl_list is not None:
            raw_hkls = [tuple(int(x) for x in h) for h in hkl_list]
        else:
            if not np.isfinite(q_max) or q_max <= 0:
                raise ValueError("q_max must be a positive finite number.")
            # Per-axis upper bounds from reciprocal basis vector lengths
            recip_lengths = np.linalg.norm(B, axis=0)   # [|a*|, |b*|, |c*|]
            N_h = int(np.ceil(q_max / recip_lengths[0])) + 1
            N_k = int(np.ceil(q_max / recip_lengths[1])) + 1
            N_l = int(np.ceil(q_max / recip_lengths[2])) + 1

            hh, kk, ll = np.meshgrid(
                np.arange(-N_h, N_h + 1),
                np.arange(-N_k, N_k + 1),
                np.arange(-N_l, N_l + 1),
                indexing='ij',
            )
            all_hkl = np.stack([hh.ravel(), kk.ravel(), ll.ravel()], axis=1)
            all_hkl = all_hkl[np.any(all_hkl != 0, axis=1)]   # remove origin

            G_all  = B @ all_hkl.T
            q_all  = np.linalg.norm(G_all, axis=0)
            mask   = (q_all >= q_min) & (q_all <= q_max)
            raw_hkls = [tuple(int(x) for x in h) for h in all_hkl[mask]]

        if not raw_hkls:
            raise ValueError("No reflections found in the specified q range.")

        # ── 2. Group into symmetry families ───────────────────────────────────
        # Prefer CIF-derived HKL-space ops (correct for all crystal systems);
        # fall back to Cartesian point-group ops when no CIF was loaded.
        if self._hkl_grouping_ops is not None:
            pg_ops = self._hkl_grouping_ops
        else:
            pg_obj = self._POINT_GROUP_MAP.get(self._normalise_crystal_system(),
                                               point_groups.trivial)
            pg_ops = _pg_int_matrices(pg_obj)

        families:  list[tuple] = []
        mult_list: list[int]   = []
        seen:      set[tuple]  = set()

        for hkl_t in raw_hkls:
            if hkl_t in seen:
                continue
            equivs = _equiv_hkls(hkl_t, pg_ops)
            seen.update(equivs)
            families.append(min(equivs))
            mult_list.append(len(equivs))

        hkl_arr = np.array(families, dtype=np.int32)
        mults   = np.array(mult_list, dtype=np.int32)

        # ── 3. d-spacing and 2θ ───────────────────────────────────────────────
        G_fam  = B @ hkl_arr.T
        q_fam  = np.linalg.norm(G_fam, axis=0)
        d_fam  = 2.0 * np.pi / q_fam

        if wavelength_A is not None:
            lam = float(wavelength_A)
            sin_theta  = q_fam * lam / (4.0 * np.pi)
            accessible = sin_theta <= 1.0
            two_theta  = np.where(accessible,
                                  2.0 * np.arcsin(np.clip(sin_theta, 0., 1.)),
                                  0.0)
        else:
            accessible = np.ones(len(hkl_arr), dtype=bool)
            two_theta  = np.zeros(len(hkl_arr))

        # ── 4. Structure factors ──────────────────────────────────────────────
        has_basis = (
            self._basis_frac is not None and len(self._basis_frac) > 0 and
            self._elements   is not None and len(self._elements) > 0
        )
        if has_basis:
            sf_sq = _xray_sf_sq(hkl_arr, q_fam,
                                 self._basis_frac, self._elements, self._occupancies)
            raw_I = mults.astype(float) * sf_sq
        else:
            # Atom positions unknown — intensity cannot be predicted.
            # Store NaN so callers can detect the missing information.
            sf_sq = None
            raw_I = None

        # ── 5. Normalise intensities ──────────────────────────────────────────
        if raw_I is not None:
            max_I = raw_I.max() if raw_I.size else 1.0
            if max_I > 0:
                if global_intensity_norm is not None:
                    intensities = raw_I / float(global_intensity_norm) * 100.0
                else:
                    intensities = raw_I / max_I * 100.0
            else:
                intensities = raw_I.copy()
        else:
            intensities = np.full(len(hkl_arr), np.nan)

        # ── 6. Filter ─────────────────────────────────────────────────────────
        # Intensity-based cutoff only applies when intensities are known.
        if raw_I is not None and intensity_cutoff_fraction > 0:
            cutoff = intensity_cutoff_fraction * (intensities.max() if intensities.size else 0.0)
            mask   = accessible & (intensities >= cutoff)
        else:
            mask   = accessible

        if not np.any(mask):
            raise ValueError(
                "All reflections were filtered out.  "
                "Check intensity_cutoff_fraction, q range, or crystal system."
            )

        hkl_arr     = hkl_arr[mask]
        mults       = mults[mask]
        d_fam       = d_fam[mask]
        two_theta   = two_theta[mask]
        intensities = intensities[mask]

        # Sort by 2θ (or by d descending if no wavelength)
        order = np.argsort(two_theta if wavelength_A is not None else -d_fam)
        hkl_arr     = hkl_arr[order]
        mults       = mults[order]
        d_fam       = d_fam[order]
        two_theta   = two_theta[order]
        intensities = intensities[order]

        # ── 7. Store ──────────────────────────────────────────────────────────
        N = len(hkl_arr)
        self.reflections                 = np.empty(N, dtype=_REFLECTION_DTYPE)
        self.reflections['hkl']          = hkl_arr
        self.reflections['multiplicity'] = mults
        self.reflections['d_spacing']    = d_fam
        self.reflections['two_theta']    = two_theta
        self.reflections['intensity']    = intensities

        self.compute_h_vectors()
        self.attach_point_group()

    # ------------------------------------------------------------------
    # Operator-facing accessor methods (backward-compatible)
    # ------------------------------------------------------------------

    def hkls(self) -> np.ndarray:
        """Miller indices, shape (N, 3)."""
        return self.reflections['hkl']

    def multiplicities(self) -> np.ndarray:
        return self.reflections['multiplicity']

    def two_theta(self) -> np.ndarray:
        """2θ values in radians."""
        return self.reflections['two_theta']

    def d_spacings(self) -> np.ndarray:
        return self.reflections['d_spacing']

    def intensities(self) -> np.ndarray:
        return self.reflections['intensity']

    def reflection(self, idx: int) -> dict:
        r = self.reflections[idx]
        return {
            'hkl':          tuple(int(x) for x in r['hkl']),
            'multiplicity': int(r['multiplicity']),
            'd_spacing':    float(r['d_spacing']),
            'two_theta':    float(r['two_theta']),
            'intensity':    float(r['intensity']),
        }

    def filter_by_intensity(self, min_intensity: float) -> None:
        """Drop reflections with intensity < min_intensity."""
        mask = self.reflections['intensity'] >= min_intensity
        self.reflections = self.reflections[mask]

    def filter_by_two_theta(self, tth_min: float, tth_max: float) -> None:
        """Keep reflections with tth_min ≤ 2θ ≤ tth_max (radians)."""
        tt   = self.reflections['two_theta']
        mask = (tt >= tth_min) & (tt <= tth_max)
        self.reflections = self.reflections[mask]

    def __len__(self) -> int:
        return 0 if self.reflections is None else len(self.reflections)

    def summary(self) -> dict:
        return {
            'name':            self.name,
            'space_group':     f"{self.space_group_symbol} ({self.space_group_number})",
            'crystal_system':  self.crystal_system,
            'num_reflections': len(self),
        }

    # ------------------------------------------------------------------
    # Geometry / symmetry helpers (used by operators)
    # ------------------------------------------------------------------

    def _compute_reciprocal_lattice_matrix(self) -> np.ndarray:
        """Return B (builds from lattice_params if not yet cached)."""
        if self._B is not None:
            return self._B
        if self.lattice_params is None:
            raise RuntimeError("Lattice not set.")
        lp = self.lattice_params
        self._A = _lattice_matrix_from_params(
            lp['a'], lp['b'], lp['c'], lp['alpha'], lp['beta'], lp['gamma'])
        self._B = reciprocal_lattice(self._A)
        return self._B

    def compute_h_vectors(self) -> None:
        """Populate self.hkl, self.h_vecs, self.h_vecs_normed."""
        B      = self._compute_reciprocal_lattice_matrix()
        hkl_f  = self.reflections['hkl'].astype(float)
        h_vecs = (B @ hkl_f.T).T
        norms  = np.linalg.norm(h_vecs, axis=1, keepdims=True)

        self.hkl           = self.reflections['hkl'].copy()
        self.h_vecs        = h_vecs
        self.h_vecs_normed = h_vecs / (norms + 1e-12)

    def attach_point_group(self, point_group_map: dict | None = None) -> None:
        """Store the point-group rotation matrices (float32, shape (num_ops, 9)).

        Parameters
        ----------
        point_group_map : dict, optional
            Maps crystal system string → tuple of scipy Rotation objects.
            Defaults to the built-in map.
        """
        if point_group_map is None:
            point_group_map = self._POINT_GROUP_MAP

        system = self._normalise_crystal_system()
        if system not in point_group_map:
            raise ValueError(f"No point group defined for crystal system '{system}'.")

        rotations = point_group_map[system]
        mats = [r.as_matrix().astype(np.float32).reshape(-1) for r in rotations]
        self.point_group_matrices = np.stack(mats, axis=0)
        self.num_sym_ops = self.point_group_matrices.shape[0]

    @staticmethod
    def _hkil_to_hkl(hkil) -> np.ndarray:
        """Convert Miller–Bravais (h k i l) → Miller (h k l)."""
        h, k, i, l = hkil
        return np.array([(2*h + k) / 3.0, (h + 2*k) / 3.0, float(l)])
