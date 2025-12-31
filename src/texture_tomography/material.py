import numpy as np
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pathlib import Path
from package.utils.lattice import (
    cubic, tetragonal, orthorhombic, hexagonal,
    trigonal_rhombohedral, monoclinic, triclinic
)
from package.utils import point_groups


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