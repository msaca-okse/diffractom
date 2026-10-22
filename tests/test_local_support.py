"""LocalSupport and the patch layout of the local-basis operator (CPU only)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from diffractom import LocalSupport
from diffractom.operators.local_support import build_patches


def random_support(seed=0, K=7, shape=(20, 24), n=150):
    rng = np.random.default_rng(seed)
    return LocalSupport(rng.integers(0, K, n), rng.integers(0, shape[0], n), rng.integers(0, shape[1], n), K, shape)


def test_pairs_sorted_unique_and_mask_round_trip():
    s = random_support()
    lin = (s.k * s.Ny + s.iy) * s.Nx + s.ix
    assert np.all(np.diff(lin) > 0)
    t = LocalSupport.from_mask(s.to_mask())
    assert np.array_equal(t.k, s.k) and np.array_equal(t.iy, s.iy) and np.array_equal(t.ix, s.ix)
    assert s.per_voxel().sum() == s.n_pairs


def test_out_of_range_raises():
    with pytest.raises(ValueError):
        LocalSupport([0], [5], [0], 2, (4, 4))


@pytest.mark.parametrize("split", [False, True])
def test_patches_hold_every_pair_once(split):
    s = random_support(1)
    P = build_patches(s.k, s.iy, s.ix, s.K, split=split, merge_gap=1, min_box_fill=0.5)
    assert P["mask"].sum() == s.n_pairs
    assert len(np.unique(P["pair_index"])) == s.n_pairs
    assert P["n_coef"] == int((P["h"] * P["w"]).sum())
    # every pair lies in a box of its own orientation, at the stored index
    for k, y, x, i in zip(s.k, s.iy, s.ix, P["pair_index"]):
        p = np.searchsorted(P["coff"], i, side="right") - 1
        assert P["pk"][p] == k
        assert P["y0"][p] <= y < P["y0"][p] + P["h"][p] and P["x0"][p] <= x < P["x0"][p] + P["w"][p]
        assert i == P["coff"][p] + (y - P["y0"][p]) * P["w"][p] + (x - P["x0"][p])
    # boxes of one orientation do not overlap
    for k in np.unique(P["pk"]):
        idx = np.flatnonzero(P["pk"] == k)
        for a in idx:
            for b in idx:
                if a < b:
                    assert (P["y0"][a] + P["h"][a] <= P["y0"][b] or P["y0"][b] + P["h"][b] <= P["y0"][a]
                            or P["x0"][a] + P["w"][a] <= P["x0"][b] or P["x0"][b] + P["w"][b] <= P["x0"][a])


def test_split_scattered_orientation():
    # one orientation in two far corners: one box without splitting, two with
    k = np.zeros(8, int)
    iy = np.array([0, 0, 1, 1, 30, 30, 31, 31])
    ix = np.array([0, 1, 0, 1, 30, 31, 30, 31])
    P1 = build_patches(k, iy, ix, 1, split=False)
    P2 = build_patches(k, iy, ix, 1, split=True, merge_gap=2)
    assert len(P1["pk"]) == 1 and P1["n_coef"] == 32 * 32
    assert len(P2["pk"]) == 2 and P2["n_coef"] == 8


def test_dilate_and_restrict():
    s = LocalSupport([0, 1], [5, 0], [5, 0], 2, (11, 11))
    d = s.dilate(1)
    assert d.n_pairs == 5 + 3       # a cross in the middle, three voxels at the corner
    m = np.zeros((11, 11), bool)
    m[:6, :6] = True
    assert d.restrict(m).n_pairs == 3 + 3


def test_from_candidates_symmetry_and_tolerance():
    basis = Rotation.from_euler("z", [[0], [10], [20]], degrees=True).as_matrix()
    # candidate 0.5 deg from orientation 1, written as a symmetric equivalent (90 deg about x: cubic)
    cand = (Rotation.from_euler("z", 10.5, degrees=True) * Rotation.from_euler("x", 90, degrees=True)).as_matrix()
    s = LocalSupport.from_candidates(basis, [3], [4], cand[None], "cubic", (8, 8), tol_deg=1.0, dilate=0)
    assert list(s.k) == [1] and list(s.iy) == [3] and list(s.ix) == [4]
    s = LocalSupport.from_candidates(basis, [3], [4], cand[None], "cubic", (8, 8), tol_deg=0.3, dilate=0)
    assert s.n_pairs == 0
