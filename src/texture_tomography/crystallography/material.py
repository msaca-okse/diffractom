import numpy as np
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pathlib import Path
from .lattice import (
    cubic,
    tetragonal,
    orthorhombic,
    hexagonal,
    trigonal_rhombohedral,
    monoclinic,
    triclinic,
)

from . import point_groups



from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from pymatgen.core import Lattice, Structure



class Material:
    """
    Crystallographic material with powder diffraction reflections.

    Responsibilities:
    - Parse CIF
    - Determine symmetry and lattice
    - Compute powder diffraction reflections
    - Store and filter reflections
    """

    # ----------------------------
    # Construction
    # ----------------------------
    def __init__(
        self,
        name: str,
        structure,
        space_group_symbol: str,
        space_group_number: int,
        crystal_system: str,
        lattice_params: dict,
        reflections: np.ndarray,
    ):
        self.name = name
        self.structure = structure

        self.space_group_symbol = space_group_symbol
        self.space_group_number = space_group_number
        self.crystal_system = crystal_system

        self.lattice_params = lattice_params
        self.reflections = reflections  # structured array

    # ----------------------------
    # Factory: from CIF
    # ----------------------------
    @classmethod
    def from_cif(
        cls,
        cif_path: str,
        *,
        wavelength_kev: float,
        min_two_theta: float,
        max_two_theta: float,
        intensity_cutoff_fraction: float = 0.0,
        global_intensity_norm: float | None = None,
    ):
        """
        Parse CIF and compute diffraction reflections.

        Parameters
        ----------
        wavelength_kev : float
        min_two_theta, max_two_theta : radians
        intensity_cutoff_fraction : fraction of max intensity (0..1)
        global_intensity_norm : optional global normalization factor
        """


        def _hkil_to_hkl(hkil):
            h, k, i, l = hkil
            return (
                (2*h + k) / 3,
                (h + 2*k) / 3,
                l,
            )
        # ---- read structure ----
        atoms = read(cif_path)
        structure = AseAtomsAdaptor.get_structure(atoms)

        name = Path(cif_path).stem

        # ---- symmetry ----
        sga = SpacegroupAnalyzer(structure)
        sg_symbol = sga.get_space_group_symbol()
        sg_number = sga.get_space_group_number()
        crystal_system = sga.get_crystal_system()

        # ---- lattice ----
        lattice = structure.lattice
        lattice_params = dict(
            a=lattice.a,
            b=lattice.b,
            c=lattice.c,
            alpha=lattice.alpha,
            beta=lattice.beta,
            gamma=lattice.gamma,
        )

        # ---- diffraction ----
        wavelength = 12.398 / wavelength_kev
        calc = XRDCalculator(wavelength=wavelength)

        pattern = calc.get_pattern(
            structure,
            two_theta_range=(
                min_two_theta * 180 / np.pi,
                max_two_theta * 180 / np.pi,
            ),
        )

        intensities = np.asarray(pattern.y, dtype=float)
        two_theta = np.asarray(pattern.x, dtype=float)/180*np.pi
        d_vals = np.asarray(pattern.d_hkls, dtype=float)

        if global_intensity_norm is not None:
            intensities = intensities / global_intensity_norm * 100.0

        # ---- build reflection list ----
        rows = []

        max_intensity = intensities.max() if intensities.size else 0.0
        cutoff = intensity_cutoff_fraction * max_intensity

        for i, peak_group in enumerate(pattern.hkls):
            I = intensities[i]
            if I < cutoff:
                continue

            for refl in peak_group:
                raw_hkl = refl.get("hkl", (0,0,0))

                if len(raw_hkl) == 4:
                    hkl = _hkil_to_hkl(raw_hkl)
                else:
                    hkl = raw_hkl

                hkl = tuple(int(round(x)) for x in hkl)
                mult = int(refl.get("multiplicity", 1))

                rows.append(
                    (
                        hkl,
                        mult,
                        d_vals[i],
                        two_theta[i],
                        I,
                    )
                )

        if not rows:
            raise ValueError(f"No reflections above cutoff for {name}")

        reflections = np.array(
            rows,
            dtype=[
                ("hkl", int, (3,)),
                ("multiplicity", int),
                ("d_spacing", float),
                ("two_theta", float),
                ("intensity", float),
            ],
        )

        material = cls(
            name=name,
            structure=structure,
            space_group_symbol=sg_symbol,
            space_group_number=sg_number,
            crystal_system=crystal_system,
            lattice_params=lattice_params,
            reflections=reflections,
        )


        material.compute_h_vectors()

        point_group_map = {
            "triclinic": point_groups.trivial,
            "monoclinic": point_groups.cyclic_2,
            "orthorhombic": point_groups.orthorhombic,
            "tetragonal": point_groups.tetragonal,
            "trigonal": point_groups.trigonal,
            "hexagonal": point_groups.hexagonal,
            "cubic": point_groups.cubic,
        }
        material.attach_point_group(point_group_map)
        return material
    

    @classmethod
    def from_lattice_parameters(
        cls,
        *,
        lattice_matrix: np.ndarray,
        symmetry_group: Literal[
            "triclinic",
            "monoclinic",
            "orthorhombic",
            "tetragonal",
            "trigonal",
            "hexagonal",
            "cubic",
        ],
        wavelength_kev: float,
        min_two_theta: float,
        max_two_theta: float,
        global_intensity_norm: float | None = None,
        # ---- selection options ----
        hkl_list: Sequence[tuple[int, int, int]] | None = None,
        # ---- lattice input options ----
        lattice_matrix_kind: Literal["direct", "reciprocal"] = "direct",
        # if you pass reciprocal: set whether it is crystallographic (no 2π) or physics (with 2π)
        reciprocal_has_2pi: bool = False,
        # optional name override
        name: str = "custom_lattice",
    ):
        """
        Build a material from a lattice matrix and a crystal system (7 options).

        Parameters
        ----------
        lattice_matrix
            If lattice_matrix_kind == "direct": 3x3 direct lattice matrix (Å),
            with rows as a,b,c vectors in Cartesian coords (ASE/Pymatgen-friendly).
            If lattice_matrix_kind == "reciprocal": 3x3 reciprocal lattice matrix (1/Å).
            By default we assume crystallographic reciprocal (no 2π). If your reciprocal
            includes 2π factors, set reciprocal_has_2pi=True.

        symmetry_group
            One of: triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic.
            Used to attach a point group and to compute multiplicities via point-group equivalents.

        wavelength_kev
            X-ray energy in keV. Converted to wavelength in Å via λ[Å] = 12.398 / E[keV].

        min_two_theta, max_two_theta
            Radians. Used only if hkl_list is None.

        hkl_list
            If provided, these HKLs are used directly (no 2θ filtering, except invalid Bragg angles
            will be skipped).

        global_intensity_norm
            If not None: intensity = intensity / global_intensity_norm * 100.
            (Even though intensities are dummy=1.0 here, kept for API compatibility.)

        Notes
        -----
        - Intensities are set to 1.0 (or normalized if global_intensity_norm is given).
        - d-spacings and 2θ are computed from the lattice (no structure factors).
        - Multiplicity is computed as the number of unique point-group equivalents of (h,k,l),
        including Friedel pairs (±h,±k,±l) if inversion is not already present in your ops.
        """

        # -------------------------
        # helpers
        # -------------------------
        def _as_3x3(a: np.ndarray) -> np.ndarray:
            a = np.asarray(a, dtype=float)
            if a.shape != (3, 3):
                raise ValueError(f"lattice_matrix must be shape (3,3), got {a.shape}")
            return a

        def _unique_hkls(hkls: Iterable[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
            seen = set()
            out = []
            for hkl in hkls:
                if hkl not in seen:
                    seen.add(hkl)
                    out.append(hkl)
            return out

        def _get_point_group_ops(pg_obj) -> list[np.ndarray]:
            """
            Accepts pg_obj as:
            - tuple/list of scipy Rotation objects (your case)
            - tuple/list of 3x3 matrices
            - objects with .as_matrix(), .rotation_matrix, or .matrix
            Returns: list of (3,3) integer rotation matrices.
            """
            # Your point_groups are tuples already; but keep it generic:
            try:
                candidates = list(pg_obj)
            except TypeError as e:
                raise TypeError(
                    "Point group object is not iterable. Expected tuple/list of operations."
                ) from e

            ops: list[np.ndarray] = []
            for op in candidates:
                M = None

                # scipy Rotation
                if hasattr(op, "as_matrix"):
                    M = np.asarray(op.as_matrix(), dtype=float)

                # already a matrix-like
                elif isinstance(op, (list, tuple, np.ndarray)):
                    M = np.asarray(op, dtype=float)

                # other possible containers
                else:
                    for mat_attr in ("rotation_matrix", "matrix", "rot", "R"):
                        if hasattr(op, mat_attr):
                            M = np.asarray(getattr(op, mat_attr), dtype=float)
                            break

                if M is None or M.shape != (3, 3):
                    raise ValueError(f"Point group op is not a 3x3 matrix-like object: {op}")

                # convert to exact-ish integer rotation matrices (your definitions are exact)
                M_int = np.rint(M).astype(int)

                # sanity: enforce orthogonality (optional but helpful)
                # if not np.allclose(M_int @ M_int.T, np.eye(3), atol=0):
                #     raise ValueError(f"Rotation matrix not orthogonal after rounding:\n{M}\n->\n{M_int}")

                ops.append(M_int)

            return ops


        def _equiv_hkls(hkl: tuple[int, int, int], ops: list[np.ndarray]) -> list[tuple[int, int, int]]:
            v = np.asarray(hkl, dtype=int).reshape(3, 1)
            eq = []
            for R in ops:
                w = (R @ v).reshape(3)
                eq.append(tuple(int(x) for x in w))
            # include Friedel pair as a pragmatic default
            eq.extend([(-h, -k, -l) for (h, k, l) in eq])
            return _unique_hkls(eq)

        def _d_and_two_theta_from_hkl(
            hkl: tuple[int, int, int],
            recip_cryst: np.ndarray,  # rows are a*, b*, c* (1/Å), crystallographic (no 2π)
            wavelength_A: float,
        ) -> tuple[float, float] | None:
            h, k, l = hkl
            # reciprocal vector g = h a* + k b* + l c*
            g = h * recip_cryst[0] + k * recip_cryst[1] + l * recip_cryst[2]
            gnorm = float(np.linalg.norm(g))
            if gnorm <= 0.0:
                return None
            d = 1.0 / gnorm

            # Bragg: 2 d sin(theta) = lambda
            s = wavelength_A / (2.0 * d)
            if s <= 0.0 or s > 1.0:
                return None
            theta = float(np.arcsin(s))
            two_theta = 2.0 * theta
            return d, two_theta

        # -------------------------
        # lattice handling
        # -------------------------
        A = _as_3x3(lattice_matrix)

        if lattice_matrix_kind == "direct":
            lattice = Lattice(A)
            recip_cryst = lattice.reciprocal_lattice_crystallographic.matrix  # (3,3), 1/Å
        elif lattice_matrix_kind == "reciprocal":
            # We want recip_cryst (no 2π). If provided has 2π, divide it out.
            recip_cryst = A / (2.0 * np.pi) if reciprocal_has_2pi else A

            # Build a direct lattice for completeness (and for compute_h_vectors etc.).
            # For crystallographic reciprocal: B = (A^{-1})^T (so A = (B^{-1})^T).
            try:
                direct = np.linalg.inv(recip_cryst).T
            except np.linalg.LinAlgError as e:
                raise ValueError("Reciprocal lattice matrix is singular / not invertible.") from e
            lattice = Lattice(direct)
        else:
            raise ValueError(f"Unknown lattice_matrix_kind: {lattice_matrix_kind!r}")

        lattice_params = dict(
            a=float(lattice.a),
            b=float(lattice.b),
            c=float(lattice.c),
            alpha=float(lattice.alpha),
            beta=float(lattice.beta),
            gamma=float(lattice.gamma),
        )

        # Minimal structure placeholder (since we don't have a basis/atomic positions here)
        structure = Structure(lattice, species=["H"], coords=[[0, 0, 0]], coords_are_cartesian=False)

        # -------------------------
        # choose HKLs
        # -------------------------
        wavelength_A = 12.398 / float(wavelength_kev)

        if hkl_list is not None:
            hkls = [tuple(map(int, hkl)) for hkl in hkl_list]
            if len(hkls) == 0:
                raise ValueError("hkl_list was provided but empty.")
        else:
            # enumerate hkls that can satisfy Bragg in the given 2θ range

            # convert 2θ range -> d range (guard against zero/invalid)
            min_tt = float(min_two_theta)
            max_tt = float(max_two_theta)
            if max_tt <= 0.0 or max_tt <= min_tt:
                raise ValueError("Require 0 < min_two_theta < max_two_theta.")

            # smallest d corresponds to largest theta
            theta_min = min_tt * 0.5
            theta_max = max_tt * 0.5

            # handle sin(0)
            sin_min = float(np.sin(theta_min))
            sin_max = float(np.sin(theta_max))
            if sin_max <= 0.0:
                raise ValueError("max_two_theta too small; sin(theta_max) <= 0.")

            d_max = wavelength_A / (2.0 * sin_min) if sin_min > 0.0 else np.inf
            d_min = wavelength_A / (2.0 * sin_max)

            g_min = 0.0 if np.isinf(d_max) else 1.0 / d_max  # = 1/d_max
            g_max = 1.0 / d_min  # = 1/d_min

            # crude bound for h,k,l: |h|*|a*| + |k|*|b*| + |l|*|c*| <= g_max
            # use the smallest reciprocal basis length to bound N conservatively
            lengths = np.linalg.norm(recip_cryst, axis=1)
            min_len = float(np.min(lengths))
            if min_len <= 0.0:
                raise ValueError("Invalid reciprocal lattice (zero-length basis vector).")

            N = int(np.ceil(g_max / min_len)) + 1
            if N < 1:
                N = 1

            hkls = []
            for h in range(-N, N + 1):
                for k in range(-N, N + 1):
                    for l in range(-N, N + 1):
                        if h == 0 and k == 0 and l == 0:
                            continue
                        res = _d_and_two_theta_from_hkl((h, k, l), recip_cryst, wavelength_A)
                        if res is None:
                            continue
                        d, tt = res
                        if min_tt <= tt <= max_tt:
                            hkls.append((h, k, l))

            # Reduce duplicates where (hkl) and (-h-k-l) both present etc.
            hkls = _unique_hkls(hkls)

            if not hkls:
                raise ValueError("No reflections found in the provided 2θ range for this lattice.")

        # -------------------------
        # multiplicities via point group
        # -------------------------
        point_group_map = {
            "triclinic": point_groups.trivial,
            "monoclinic": point_groups.cyclic_2,
            "orthorhombic": point_groups.orthorhombic,
            "tetragonal": point_groups.tetragonal,
            "trigonal": point_groups.trigonal,
            "hexagonal": point_groups.hexagonal,
            "cubic": point_groups.cubic,
        }
        if symmetry_group not in point_group_map:
            raise ValueError(f"symmetry_group must be one of {list(point_group_map)}, got {symmetry_group!r}")

        pg_obj = point_group_map[symmetry_group]
        ops = _get_point_group_ops(pg_obj)

        # -------------------------
        # build reflection rows
        # -------------------------
        rows = []
        for hkl in hkls:
            # Use "canonical" representative for storage (purely aesthetic):
            # choose the lexicographically smallest among equivalents so you don't store both ± etc.
            eq = _equiv_hkls(hkl, ops)
            hkl_canon = min(eq)

            res = _d_and_two_theta_from_hkl(hkl_canon, recip_cryst, wavelength_A)
            if res is None:
                continue
            d, tt = res

            mult = len(eq)

            I = 1.0
            if global_intensity_norm is not None:
                I = I / float(global_intensity_norm) * 100.0

            rows.append((hkl_canon, mult, d, tt, I))

        if not rows:
            raise ValueError("No valid reflections after Bragg filtering.")

        # Sort by 2θ (ascending)
        rows.sort(key=lambda r: r[3])

        reflections = np.array(
            rows,
            dtype=[
                ("hkl", int, (3,)),
                ("multiplicity", int),
                ("d_spacing", float),
                ("two_theta", float),
                ("intensity", float),
            ],
        )

        # -------------------------
        # build material object
        # -------------------------
        material = cls(
            name=name,
            structure=structure,
            space_group_symbol=None,   # unknown from inputs
            space_group_number=None,   # unknown from inputs
            crystal_system=symmetry_group,
            lattice_params=lattice_params,
            reflections=reflections,
        )

        material.compute_h_vectors()
        material.attach_point_group(point_group_map)

        return material




    # ----------------------------
    # Filtering
    # ----------------------------
    def filter_by_intensity(self, min_intensity: float):
        """Keep reflections with intensity >= min_intensity."""
        mask = self.reflections["intensity"] >= min_intensity
        self.reflections = self.reflections[mask]

    def filter_by_two_theta(self, tth_min: float, tth_max: float):
        """Keep reflections with tth_min <= two_theta <= tth_max."""
        tt = self.reflections["two_theta"]
        mask = (tt >= tth_min) & (tt <= tth_max)
        self.reflections = self.reflections[mask]

    # ----------------------------
    # Accessors (operator-friendly)
    # ----------------------------
    def hkls(self):
        return self.reflections["hkl"]

    def multiplicities(self):
        return self.reflections["multiplicity"]

    def two_theta(self):
        return self.reflections["two_theta"]

    def d_spacings(self):
        return self.reflections["d_spacing"]

    def intensities(self):
        return self.reflections["intensity"]

    # ----------------------------
    # Single-reflection inspection
    # ----------------------------
    def reflection(self, idx: int) -> dict:
        """Return a single reflection as a dict."""
        r = self.reflections[idx]
        return {
            "hkl": tuple(r["hkl"]),
            "multiplicity": int(r["multiplicity"]),
            "d_spacing": float(r["d_spacing"]),
            "two_theta": float(r["two_theta"]),
            "intensity": float(r["intensity"]),
        }

    # ----------------------------
    # Convenience
    # ----------------------------
    def __len__(self):
        return len(self.reflections)

    def summary(self):
        return {
            "name": self.name,
            "space_group": f"{self.space_group_symbol} ({self.space_group_number})",
            "crystal_system": self.crystal_system,
            "num_reflections": len(self),
        }



    def _compute_reciprocal_lattice_matrix(self):
        """
        Compute reciprocal lattice matrix B based on crystal system.
        """
        system = self.crystal_system.lower()
        lp = self.lattice_params

        if "cubic" in system:
            _, B = cubic(lp["a"])
        elif "tetragonal" in system:
            _, B = tetragonal(lp["a"], lp["c"])
        elif "orthorhombic" in system:
            _, B = orthorhombic(lp["a"], lp["b"], lp["c"])
        elif "hexagonal" in system:
            _, B = hexagonal(lp["a"], lp["c"])
        elif "trigonal" in system or "rhombohedral" in system:
            _, B = trigonal_rhombohedral(lp["a"], lp["alpha"])
        elif "monoclinic" in system:
            _, B = monoclinic(lp["a"], lp["b"], lp["c"], lp["beta"])
        elif "triclinic" in system:
            _, B = triclinic(
                lp["a"], lp["b"], lp["c"],
                lp["alpha"], lp["beta"], lp["gamma"]
            )
        else:
            raise ValueError(f"Unsupported crystal system: {system}")

        return B

    @staticmethod
    def _hkil_to_hkl(hkil):
        """
        Convert Miller–Bravais (h k i l) → Miller (h k l)
        """
        h, k, i, l = hkil
        return np.array([
            (2*h + k) / 3.0,
            (h + 2*k) / 3.0,
            l
        ])

    def compute_h_vectors(self):
        """
        Compute and store reciprocal lattice vectors and normalized versions.
        """
        B = self._compute_reciprocal_lattice_matrix()

        hkl_list = []
        h_vecs = []

        for r in self.reflections:
            raw_hkl = r["hkl"]

            # HKIL → HKL if needed
            if len(raw_hkl) == 4:
                hkl = self._hkil_to_hkl(raw_hkl)
            else:
                hkl = np.asarray(raw_hkl, dtype=float)

            # enforce integer HKL after conversion
            hkl = np.round(hkl).astype(int)

            h_vec = B @ hkl

            hkl_list.append(hkl)
            h_vecs.append(h_vec)

        self.hkl = np.asarray(hkl_list, dtype=int)
        self.h_vecs = np.asarray(h_vecs, dtype=float)

        # normalized vectors (needed by PF kernel)
        norms = np.linalg.norm(self.h_vecs, axis=1, keepdims=True)
        self.h_vecs_normed = self.h_vecs / (norms + 1e-12)


    # def compute_h_vectors(self):
    #     B = self._compute_reciprocal_lattice_matrix()

    #     hkl_list = []
    #     h_vecs = []

    #     def fcc_allowed(h,k,l):
    #         return (h&1) == (k&1) == (l&1)

    #     def canonical_friedel(h,k,l):
    #         v = np.array([h,k,l],dtype=int)
    #         for i in range(3):
    #             if v[i] != 0:
    #                 if v[i] < 0:
    #                     v = -v
    #                 break
    #         return tuple(v.tolist())

    #     seen = set()

    #     for r in self.reflections:
    #         raw_hkl = r["hkl"]

    #         if len(raw_hkl) == 4:
    #             hkl = self._hkil_to_hkl(raw_hkl)
    #         else:
    #             hkl = np.asarray(raw_hkl, dtype=float)

    #         hkl = np.round(hkl).astype(int)
    #         h,k,l = hkl

    #         # remove origin
    #         if h==0 and k==0 and l==0:
    #             continue

    #         # FCC extinction
    #         if not fcc_allowed(h,k,l):
    #             continue

    #         # Friedel merge (optional but recommended for PF)
    #         key = canonical_friedel(h,k,l)
    #         if key in seen:
    #             continue
    #         seen.add(key)

    #         h_vec = B @ np.array([h,k,l],dtype=float)

    #         hkl_list.append([h,k,l])
    #         h_vecs.append(h_vec)

    #     self.hkl = np.asarray(hkl_list, dtype=int)
    #     self.h_vecs = np.asarray(h_vecs, dtype=float)

    #     norms = np.linalg.norm(self.h_vecs, axis=1, keepdims=True)
    #     self.h_vecs_normed = self.h_vecs / (norms + 1e-12)




    def attach_point_group(self, point_group_map):
            """
            Attach point group symmetry operators based on crystal system.

            Parameters
            ----------
            point_group_map : dict
                Maps crystal system string -> tuple of scipy Rotation objects
            """
            system = self.crystal_system.lower()

            # normalize system names a bit
            if "rhombohedral" in system:
                system = "trigonal"

            if system not in point_group_map:
                raise ValueError(f"No point group defined for system '{system}'")

            rotations = point_group_map[system]

            # convert to flattened 3x3 matrices
            mats = []
            for r in rotations:
                M = r.as_matrix().astype(np.float32)
                mats.append(M.reshape(-1))

            self.point_group_matrices = np.stack(mats, axis=0)
            self.num_sym_ops = self.point_group_matrices.shape[0]