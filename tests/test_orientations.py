"""refine_orientations: the mean-shift read-out of the dominant orientation per voxel."""
import numpy as np
from scipy.spatial.transform import Rotation

from diffractom.analysis import refine_orientations, symmetry_quaternions
from diffractom.analysis.orientations import _qmul, _quat


def _misorientation_deg(A, B, symmetry="cubic"):
    S = symmetry_quaternions(symmetry)
    d = np.abs(np.einsum("...gc,...c->...g", _qmul(_quat(A)[..., None, :], S), _quat(B))).max(axis=-1)
    return np.degrees(2 * np.arccos(np.clip(d, 0, 1)))


def _voxel_odf(true, basis, width):
    """Coefficients of a sampled Gaussian ODF around `true` (the weights a fit of a sharp texture gives)."""
    om = np.radians(_misorientation_deg(basis, np.broadcast_to(true, basis.shape)))
    return np.exp(-0.5 * (om / width) ** 2)


def _lattice(centre, spacing_deg=4.0, n=3, shift_deg=(1.3, -1.7, 0.9)):
    """Kernel centres on a cubic lattice of rotation vectors around a node that is shifted from `centre`,
    so that `centre` lies between the kernels."""
    g = np.arange(-n, n + 1) * spacing_deg
    off = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3) - np.asarray(shift_deg)
    return (Rotation.from_matrix(centre) * Rotation.from_rotvec(np.radians(off))).as_matrix()


def test_mode_lies_between_the_kernels():
    truths = Rotation.from_rotvec([[0.3, -0.2, 0.5], [-1.0, 0.4, 0.2]]).as_matrix()
    # a coarse basis around each truth (kernels 4 deg apart), widths 3 deg
    basis = np.concatenate([_lattice(t) for t in truths])
    K = len(basis)
    coeffs = np.zeros((K, 1, 3))
    for v, t in enumerate(truths):
        coeffs[:, 0, v] = _voxel_odf(t, basis, np.radians(3.0))
        coeffs[:, 0, v] *= _misorientation_deg(basis, np.broadcast_to(t, basis.shape)) < 15
    res = refine_orientations(coeffs, basis, np.radians(3.0), symmetry="cubic", top_k=64)
    err = _misorientation_deg(res["orientations"][0, :2], truths)
    err_start = _misorientation_deg(res["start"][0, :2], truths)
    assert np.all(err < 0.5), err
    assert np.all(err < err_start), (err, err_start)
    assert np.all(np.isnan(res["orientations"][0, 2]))            # no positive coefficient: no orientation
    assert np.all(res["spread_deg"][0, :2] > 0)


def test_symmetry_equivalent_basis_gives_the_same_orientation():
    rng = np.random.default_rng(1)
    true = Rotation.from_rotvec([0.1, 0.7, -0.3]).as_matrix()
    basis = _lattice(true)
    c = _voxel_odf(true, basis, np.radians(2.5))[:, None, None]
    S = np.stack([g.as_matrix() for g in Rotation.from_quat(
        np.concatenate([symmetry_quaternions("cubic")[:, 1:], symmetry_quaternions("cubic")[:, :1]], axis=1))])
    relabelled = basis @ S[rng.integers(0, len(S), len(basis))]   # the same orientations, other representatives
    ra = refine_orientations(c, basis, np.radians(2.5), top_k=64)
    a = ra["orientations"][0, 0]
    b = refine_orientations(c, relabelled, np.radians(2.5), top_k=64)["orientations"][0, 0]
    assert _misorientation_deg(a, b) < 1e-3
    # kernels narrower than the 4 deg spacing: the sampled ODF's own mode is ~0.6 deg off the truth
    assert _misorientation_deg(a, true) < min(1.0, _misorientation_deg(ra["start"][0, 0], true))


def test_single_kernel_stays_put_and_top_k():
    basis = Rotation.from_rotvec([[0.2, 0.1, 0.0], [0.2, 0.1, 0.9]]).as_matrix()   # 52 deg apart
    c = np.array([1.0, 0.3])[:, None, None]
    res = refine_orientations(c, basis, np.radians(1.0), top_k=1)
    assert _misorientation_deg(res["orientations"][0, 0], basis[0]) < 1e-6
    assert res["shift_deg"][0, 0] < 1e-6
