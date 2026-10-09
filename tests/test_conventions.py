"""The crystal Cartesian frame is ImageD11's (Busing & Levy: a* along x, b* in the xy-plane, c along z), so orientation
matrices are interchangeable with ImageD11's U. The built-in point groups agree with the Laue groups of the space
groups in that frame."""
import numpy as np
import pytest

from diffractom.crystallography import point_groups as pg
from diffractom.crystallography.material import _laue_rotations_cartesian, _lattice_matrix_from_params

CELLS = {
    "cubic": (4.05, 4.05, 4.05, 90, 90, 90),
    "tetragonal": (3.1, 3.1, 5.2, 90, 90, 90),
    "orthorhombic": (4.1, 5.3, 6.7, 90, 90, 90),
    "hexagonal": (4.9154, 4.9154, 5.4057, 90, 90, 120),
    "rhombohedral": (5.2, 5.2, 5.2, 70, 70, 70),
    "monoclinic": (8.56, 12.97, 7.21, 90, 116.0, 90),
    "triclinic": (8.58, 12.96, 7.22, 90.6, 115.9, 87.7),
}


def canon(ms):
    return sorted(tuple(np.round(np.asarray(m), 6).ravel()) for m in ms)


@pytest.mark.parametrize("name", CELLS)
def test_matches_imaged11_B(name):
    unitcell = pytest.importorskip("ImageD11.unitcell")
    cell = CELLS[name]
    A = _lattice_matrix_from_params(*cell)
    B = unitcell.unitcell(list(cell)).B                        # columns a*, b*, c*, ImageD11 frame
    np.testing.assert_allclose(A.T @ B, np.eye(3), atol=1e-12)


def test_frame_axes():
    A = _lattice_matrix_from_params(*CELLS["triclinic"])
    astar = np.linalg.inv(A).T[:, 0]
    assert abs(astar[1]) < 1e-12 and abs(astar[2]) < 1e-12 and astar[0] > 0          # a* along x
    assert abs(A[0, 2]) < 1e-12 and abs(A[1, 2]) < 1e-12 and A[2, 2] > 0              # c along z


def test_orthogonal_cells_unchanged():
    a, b, c = 4.1, 5.3, 6.7
    A = _lattice_matrix_from_params(a, b, c, 90.0, 90.0, 90.0)
    np.testing.assert_allclose(A, np.diag([a, b, c]), atol=1e-12)


@pytest.mark.parametrize("ops,builtin", [
    (["x,y,z", "-y,x-y,z+1/3", "-x+y,-x,z+2/3", "y,x,-z", "x-y,-y,-z+2/3", "-x,-x+y,-z+1/3"], pg.trigonal),  # P3221
    (["x,y,z", "-y,x-y,z", "-x+y,-x,z", "-x,-y,z+1/2", "y,-x+y,z+1/2", "x-y,x,z+1/2", "y,x,-z", "x-y,-y,-z",
      "-x,-x+y,-z", "-y,-x,-z+1/2", "-x+y,y,-z+1/2", "x,x-y,-z+1/2"], pg.hexagonal),                          # P6322
])
def test_builtin_point_groups_match_space_groups(ops, builtin):
    A = _lattice_matrix_from_params(*CELLS["hexagonal"])
    assert canon(_laue_rotations_cartesian(ops, A)) == canon([r.as_matrix() for r in builtin])


def test_monoclinic_laue_orthogonal():
    A = _lattice_matrix_from_params(*CELLS["monoclinic"])
    L = _laue_rotations_cartesian(["x,y,z", "-x,y,-z", "-x,-y,-z", "x,-y,z"], A)       # 2/m, unique axis b
    assert L is not None and len(L) == 2
    two = [m for m in L if not np.allclose(m, np.eye(3))][0]
    np.testing.assert_allclose(two, np.diag([-1, 1, -1]), atol=1e-9)                  # 2-fold along b = y


def test_set_lattice_updates_laue_rotations(tmp_path):
    from diffractom import Material
    cif = tmp_path / "q.cif"
    cif.write_text("""data_q
_cell_length_a 4.9
_cell_length_b 4.9
_cell_length_c 5.4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
_symmetry_space_group_name_H-M 'P 32 2 1'
loop_
_symmetry_equiv_pos_as_xyz
x,y,z
-y,x-y,z+2/3
-x+y,-x,z+1/3
y,x,-z
x-y,-y,-z+1/3
-x,-x+y,-z+2/3
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
Si Si 0.4697 0.0 0.0 1.0 0.0078
""")
    m = Material()
    m.load_cif(str(cif))
    m.set_lattice(4.95, 4.95, 5.45, 90, 90, 120)
    A = _lattice_matrix_from_params(4.95, 4.95, 5.45, 90, 90, 120)
    assert canon(m._laue_rotations) == canon(_laue_rotations_cartesian(m._sym_op_strings, A))
    assert canon(m._laue_rotations) == canon([r.as_matrix() for r in pg.trigonal])
