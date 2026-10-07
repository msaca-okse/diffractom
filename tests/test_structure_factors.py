"""
Structure factors and reflection families from CIF files, against gemmi, on synthetic structures in every
crystal system (random sites, occupancies and displacements; the symmetry operations written into the CIF).

Run: python -m pytest tests/test_structure_factors.py   (needs gemmi)
"""
import numpy as np
import pytest

from diffractom import Material

gemmi = pytest.importorskip("gemmi")

# (space group, cell): standard settings of every crystal system, with and without inversion centre
CASES = [
    ("P -1", (5.1, 6.2, 7.3, 81.0, 95.0, 103.0)),
    ("P 1 21/c 1", (5.1, 6.2, 7.3, 90.0, 101.0, 90.0)),
    ("C 1 2/c 1", (9.1, 6.2, 7.3, 90.0, 107.0, 90.0)),
    ("P 1 21 1", (5.1, 6.2, 7.3, 90.0, 99.0, 90.0)),
    ("P n m a", (5.1, 6.2, 7.3, 90.0, 90.0, 90.0)),
    ("I 41/a m d:1", (5.1, 5.1, 7.3, 90.0, 90.0, 90.0)),
    ("P -4 21 m", (5.1, 5.1, 7.3, 90.0, 90.0, 90.0)),
    ("R -3 m:H", (5.1, 5.1, 13.3, 90.0, 90.0, 120.0)),
    ("P 32 2 1", (4.91, 4.91, 5.40, 90.0, 90.0, 120.0)),
    ("P 63/m m c", (3.21, 3.21, 5.21, 90.0, 90.0, 120.0)),
    ("F d -3 m:1", (11.43, 11.43, 11.43, 90.0, 90.0, 90.0)),
    ("P 21 3", (6.1, 6.1, 6.1, 90.0, 90.0, 90.0)),
]
ELEMENTS = ["O", "Si", "Fe", "Al", "Mg"]
WAVELENGTH = 0.3563


def write_cif(path, sg_name, cell, seed):
    rng = np.random.default_rng(seed)
    sg = gemmi.find_spacegroup_by_name(sg_name)
    a, b, c, al, be, ga = cell
    lines = ["data_test", f"_cell_length_a {a}", f"_cell_length_b {b}", f"_cell_length_c {c}",
             f"_cell_angle_alpha {al}", f"_cell_angle_beta {be}", f"_cell_angle_gamma {ga}",
             f"_symmetry_space_group_name_H-M '{sg.hm}'", "loop_", "_symmetry_equiv_pos_as_xyz"]
    lines += [f"'{op.triplet()}'" for op in sg.operations()]
    lines += ["loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y",
              "_atom_site_fract_z", "_atom_site_occupancy", "_atom_site_U_iso_or_equiv", "_atom_site_adp_type"]
    def general_position():
        # random positions away from special positions (gemmi merges images closer than about 0.5 Å)
        A = np.array(gemmi.UnitCell(*cell).orth.mat.tolist())
        ops = [(np.array(op.rot) / gemmi.Op.DEN, np.array(op.tran) / gemmi.Op.DEN) for op in sg.operations()]
        while True:
            p = rng.random(3)
            imgs = np.array([W @ p + t for W, t in ops])
            d = imgs[:, None, :] - imgs[None, :, :]
            d = (d + 0.5) % 1.0 - 0.5
            dist = np.linalg.norm(d @ A.T, axis=-1)
            dist[np.diag_indices(len(imgs))] = np.inf
            if dist.min() > 0.6:
                return p

    for i in range(3):
        el = ELEMENTS[rng.integers(len(ELEMENTS))]
        x, y, z = general_position()
        lines.append(f"{el}{i} {el} {x:.5f} {y:.5f} {z:.5f} {rng.uniform(0.5, 1.0):.3f} {rng.uniform(0.003, 0.02):.4f} Uiso")
    # a mixed-occupancy site (two elements on one position)
    x, y, z = general_position()
    lines.append(f"Fe9 Fe {x:.5f} {y:.5f} {z:.5f} 0.4 0.008 Uiso")
    lines.append(f"Ni9 Ni {x:.5f} {y:.5f} {z:.5f} 0.6 0.008 Uiso")
    path.write_text("\n".join(lines) + "\n")
    return sg


@pytest.mark.parametrize("sg_name,cell", CASES)
def test_structure_factors_match_gemmi(tmp_path, sg_name, cell):
    cif = tmp_path / "test.cif"
    write_cif(cif, sg_name, cell, seed=hash(sg_name) % 2 ** 32)
    mat = Material.from_cif(cif, wavelength_A=WAVELENGTH, min_two_theta=0.02, max_two_theta=0.45)
    small = gemmi.read_small_structure(str(cif))
    small.change_occupancies_to_crystallographic()
    calc = gemmi.StructureFactorCalculatorX(small.cell)
    hkl = mat.reflections["hkl"]
    F_ref = np.array([calc.calculate_sf_from_small_structure(small, [int(v) for v in h]) for h in hkl])
    F = mat.structure_factors(hkl)
    # gemmi's IT92 and the Cromer-Mann table here are the same coefficients up to rounding
    np.testing.assert_allclose(np.abs(F) ** 2, np.abs(F_ref) ** 2, rtol=2e-3, atol=1e-3 * np.abs(F_ref).max() ** 2)
    np.testing.assert_allclose(mat.reflections["sf_squared"], np.abs(F_ref) ** 2, rtol=2e-3,
                               atol=1e-3 * np.abs(F_ref).max() ** 2)


@pytest.mark.parametrize("sg_name,cell", CASES)
def test_families_and_laue_rotations(tmp_path, sg_name, cell):
    cif = tmp_path / "test.cif"
    sg = write_cif(cif, sg_name, cell, seed=1)
    mat = Material.from_cif(cif, wavelength_A=WAVELENGTH, min_two_theta=0.02, max_two_theta=0.35)
    # multiplicity = orbit of hkl under the Laue group (point group of the operations + Friedel), brute force
    rots = [np.array(op.rot) // gemmi.Op.DEN for op in sg.operations().sym_ops]
    for r in mat.reflections:
        h = np.array(r["hkl"])
        orbit = {tuple(int(v) for v in s * (np.linalg.inv(W).T @ h).round().astype(int))
                 for W in rots for s in (1, -1)}
        assert len(orbit) == r["multiplicity"]
    # the Cartesian Laue rotations are orthogonal, of the right order, and map each reflection onto its family
    R = mat.point_group_matrices.reshape(-1, 3, 3).astype(float)
    n_proper = len({tuple((W * (1 if round(np.linalg.det(W)) > 0 else -1)).ravel()) for W in rots})
    assert len(R) == n_proper
    np.testing.assert_allclose(np.einsum("gij,gkj->gik", R, R), np.broadcast_to(np.eye(3), R.shape), atol=1e-5)
    B = mat._compute_reciprocal_lattice_matrix()
    for r in mat.reflections[:10]:
        h = np.array(r["hkl"], dtype=float)
        images = np.einsum("gij,j->gi", R, B @ h)
        F_img = mat.structure_factors(np.linalg.solve(B, images.T).T.round())
        np.testing.assert_allclose(np.abs(F_img) ** 2, r["sf_squared"], rtol=1e-6, atol=1e-6)


def test_form_factors_match_gemmi_it92():
    """Every element of the Cromer-Mann table against gemmi's IT92 coefficients, f0(sin(theta)/lambda) for
    s = 0 ... 2 / Å. (xfab's table is not a usable reference: its constant term is off by -1 for many elements,
    e.g. f0_Al(0) = 12, and has the wrong sign for B, N, Cl.)"""
    from diffractom.crystallography.form_factors import _FORM_FACTOR_DATA, form_factor_array
    s_vals = np.linspace(0, 2.0, 41)
    for el in _FORM_FACTOR_DATA:
        it92 = gemmi.Element(el).it92
        ref = np.array([it92.calculate_sf(float(s) ** 2) for s in s_vals])
        np.testing.assert_allclose(form_factor_array(el, 4 * np.pi * s_vals), ref, rtol=2e-3, atol=2e-3, err_msg=el)
        assert abs(form_factor_array(el, np.array([0.0]))[0] - gemmi.Element(el).atomic_number) < 0.1, el


def test_from_sites_equals_from_cif(tmp_path):
    cif = tmp_path / "mg.cif"
    sg = gemmi.find_spacegroup_by_name("P 63/m m c")
    cif.write_text("\n".join([
        "data_mg", "_cell_length_a 3.209", "_cell_length_b 3.209", "_cell_length_c 5.211", "_cell_angle_alpha 90",
        "_cell_angle_beta 90", "_cell_angle_gamma 120", "_symmetry_space_group_name_H-M 'P 63/m m c'", "loop_",
        "_symmetry_equiv_pos_as_xyz", *[f"'{op.triplet()}'" for op in sg.operations()], "loop_", "_atom_site_label",
        "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z", "_atom_site_B_iso_or_equiv",
        f"Mg1 0.33333 0.66667 0.25 {8 * np.pi ** 2 * 0.012:.6f}"]) + "\n")
    a = Material.from_cif(cif, wavelength_A=WAVELENGTH, min_two_theta=0.02, max_two_theta=0.4)
    b = Material.from_sites([("Mg", 1 / 3, 2 / 3, 0.25, 1.0, 0.012)], a=3.209, c=5.211, gamma=120,
                            space_group="P 63/m m c", wavelength_A=WAVELENGTH, min_two_theta=0.02, max_two_theta=0.4)
    np.testing.assert_array_equal(a.reflections["hkl"], b.reflections["hkl"])
    np.testing.assert_allclose(a.reflections["sf_squared"], b.reflections["sf_squared"], rtol=1e-4)


def test_cif_without_operations_uses_space_group(tmp_path):
    cif = tmp_path / "al.cif"
    cif.write_text("\n".join([
        "data_al", "_cell_length_a 4.0495", "_cell_length_b 4.0495", "_cell_length_c 4.0495", "_cell_angle_alpha 90",
        "_cell_angle_beta 90", "_cell_angle_gamma 90", "_symmetry_space_group_name_H-M 'F m -3 m'", "loop_",
        "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y",
        "_atom_site_fract_z", "_atom_site_U_iso_or_equiv", "Al1 Al3+ 0 0 0 0.0108"]) + "\n")
    mat = Material.from_cif(cif, wavelength_A=WAVELENGTH, min_two_theta=0.02, max_two_theta=0.6)
    assert len(mat._basis_frac) == 4                                  # fcc: 4 atoms in the cell
    hkl = mat.reflections["hkl"]
    assert np.all((hkl % 2 == 0).all(axis=1) | (hkl % 2 != 0).all(axis=1))   # only unmixed hkl remain
    s = 1 / (2 * mat.reflections["d_spacing"])
    from diffractom.crystallography.form_factors import form_factor_array
    f = np.array([form_factor_array("Al", np.array([4 * np.pi * v]))[0] for v in s])
    np.testing.assert_allclose(mat.reflections["sf_squared"], (4 * f * np.exp(-8 * np.pi ** 2 * 0.0108 * s ** 2)) ** 2,
                               rtol=1e-10)


def test_unknown_element_raises(tmp_path):
    cif = tmp_path / "x.cif"
    cif.write_text("\n".join(["data_x", "_cell_length_a 4", "loop_", "_symmetry_equiv_pos_as_xyz", "'x,y,z'", "loop_",
                              "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x",
                              "_atom_site_fract_y", "_atom_site_fract_z", "Q1 Qq 0 0 0"]) + "\n")
    with pytest.raises(ValueError):
        Material.from_cif(cif, wavelength_A=WAVELENGTH, min_two_theta=0.02, max_two_theta=0.6)


def test_intensity_model_factors():
    from diffractom.crystallography.intensity import IntensityModel
    tt = np.radians([10.0, 20.0, 30.0])
    th = tt / 2
    np.testing.assert_allclose(IntensityModel().factor(tt), 1 / np.sin(th))
    np.testing.assert_allclose(IntensityModel(data="q_bin_mean").factor(tt), 1 / np.sin(th) ** 2)
    np.testing.assert_allclose(IntensityModel(data="solid_angle_sum").factor(tt), 1 / (np.sin(th) * np.cos(tt) ** 3))
    np.testing.assert_allclose(IntensityModel(lorentz="none", data="none").factor(tt), 1.0)
    with pytest.raises(ValueError):
        IntensityModel(data="mean")
