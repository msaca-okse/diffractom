"""adjoint_multi_cl equals separate adjoints of the row-masked data, bitwise; score_candidates equals a plain reference
(one adjoint per masked input, the score test and the top-k selection in NumPy), bitwise."""
import numpy as np
import pytest


def _context():
    cl = pytest.importorskip("pyopencl")
    try:
        ctx = cl.create_some_context(interactive=False)
    except Exception:
        pytest.skip("no OpenCL device")
    return ctx, cl.CommandQueue(ctx)


CFG = {"Nx": 14, "Ny": 12, "My": 17, "N_Omega": 16, "angle_range": [0, 360], "cor_offset": 0.7, "N_eta": 24,
       "eta_angle_range": [0, 360], "energy": 12.398 / 0.35, "k_direction_0": [0, 0, 1], "p_direction_0": [1, 0, 0],
       "detector_direction_origin": [0, -1, 0], "detector_direction_positive_90": [0, 0, -1]}


def _material():
    from diffractom import Material
    return Material.from_lattice_parameters(a=4.05, b=4.05, c=4.05, alpha=90, beta=90, gamma=90, crystal_system="cubic",
                                            wavelength_A=0.35, hkl_list=np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0]]))


def _operator(ctx, queue, rot, pf_mode):
    from diffractom import MatrixGrid, SinglePhaseForwardOperator
    return SinglePhaseForwardOperator(cfg=dict(CFG), material=_material(), grid=MatrixGrid(rot, np.full(len(rot), np.deg2rad(4.0))),
                                      max_gb=2e-5, normalized=True, pf_mode=pf_mode, projector="native", ctx=ctx, queue=queue,
                                      sparse_batch_max=16)


def _adjoint(op, a):
    """adjoint_cl on device arrays (the search's per-input adjoint)."""
    import pyopencl.array as clarray
    g = clarray.zeros(op.queue, op.coeff_shape, np.float32)
    op.adjoint_cl(clarray.to_device(op.queue, np.ascontiguousarray(a, np.float32)), g)
    return g.get()


@pytest.mark.parametrize("pf_mode", ["sparse", "generated"])
def test_adjoint_multi_equals_masked_adjoints(pf_mode):
    import pyopencl.array as clarray
    from scipy.spatial.transform import Rotation
    ctx, q = _context()
    op = _operator(ctx, q, Rotation.random(40, random_state=1).as_matrix(), pf_mode)
    assert len(op.batches) > 1
    rng = np.random.default_rng(0)
    datas = [rng.normal(size=op.data_shape).astype(np.float32) for _ in range(3)]
    fit = (np.arange(op.N_Omega) % 5) != 2
    masks = [None, fit, ~fit]
    got = np.zeros((len(datas), len(masks)) + op.coeff_shape, np.float32)
    Npix = op.Nx * op.Ny

    def consume(i, s, k0, Kb, img_k):
        blk = img_k.get().reshape(Npix, op.K_batch_max)[:, :Kb]
        got[i, s, k0:k0 + Kb] = blk.T.reshape(Kb, op.Ny, op.Nx)
    op.adjoint_multi_cl([clarray.to_device(q, d) for d in datas], consume, row_masks=masks)
    for i, d in enumerate(datas):
        for s, m in enumerate(masks):
            dm = d if m is None else d * m[:, None, None].astype(np.float32)
            assert np.array_equal(got[i, s], _adjoint(op, dm)), (pf_mode, i, s)


@pytest.mark.parametrize("nrhs,rows_per_group", [(1, 24), (2, 7), (3, 12), (4, 12)])
def test_adjoint_multi_pixels_and_groupings(nrhs, rows_per_group):
    """The pixel-list back-projection over listed angles and every grouping of the products give the masked adjoint's
    values at those pixels, bitwise."""
    import pyopencl.array as clarray
    from scipy.spatial.transform import Rotation
    ctx, q = _context()
    op = _operator(ctx, q, Rotation.random(40, random_state=4).as_matrix(), "generated")
    rng = np.random.default_rng(5)
    datas = [rng.normal(size=op.data_shape).astype(np.float32) for _ in range(3)]
    fit = (np.arange(op.N_Omega) % 5) != 2
    masks = [fit, ~fit, None]
    pix = np.array([0, 5, 17, 40, 41, 100, 150, op.Nx * op.Ny - 1], np.int32)
    got = np.zeros((len(datas), len(masks), op.K, len(pix)), np.float32)

    def consume(i, s, k0, Kb, block):
        got[i, s, k0:k0 + Kb] = block.get()[:Kb * len(pix)].reshape(Kb, len(pix))
    op.adjoint_multi_cl([clarray.to_device(q, d) for d in datas], consume, row_masks=masks, pixels=pix, nrhs=nrhs,
                        rows_per_group=rows_per_group)
    for i, d in enumerate(datas):
        for s, m in enumerate(masks):
            dm = d if m is None else d * m[:, None, None].astype(np.float32)
            assert np.array_equal(got[i, s], _adjoint(op, dm).reshape(op.K, -1)[:, pix]), (nrhs, i, s)


def test_score_candidates_equals_reference():
    import pyopencl.array as clarray
    from scipy.spatial.transform import Rotation
    from diffractom.orientation_search import score_candidates
    ctx, q = _context()
    rot = Rotation.random(70, random_state=2).as_matrix()
    rng = np.random.default_rng(3)
    shape = _operator(ctx, q, rot[:2], "generated").data_shape
    E = [rng.normal(size=shape).astype(np.float32) for _ in range(2)]
    D = [np.abs(rng.normal(size=shape)).astype(np.float32) for _ in range(2)]
    fit = (np.arange(CFG["N_Omega"]) % 5) != 2
    pix = np.array([3, 17, 40, 41, 100, 150], np.int32)
    top_k, chunk = 8, 30
    res = score_candidates(lambda R: _operator(ctx, q, R, "generated"), rot, [(clarray.to_device(q, E[j]), clarray.to_device(q, D[j])) for j in range(2)],
                           [fit, ~fit], pix, top_k=top_k, chunk=chunk)
    for j in range(2):
        i0, z0, t0 = np.full((len(pix), top_k), -1, np.int32), np.full((len(pix), top_k), -np.inf, np.float32), np.zeros((len(pix), top_k), np.float32)
        for c0 in range(0, len(rot), chunk):
            op = _operator(ctx, q, rot[c0:c0 + chunk], "generated")
            K = op.K
            adj = lambda a, m: _adjoint(op, a * m[:, None, None].astype(np.float32)).reshape(K, -1)[:, pix]
            Etr, Dtr, Ete, Dte = adj(E[j], fit), adj(D[j], fit), adj(E[j], ~fit), adj(D[j], ~fit)
            ztr = Etr / np.sqrt(np.maximum(Dtr, 1e-30)); zte = Ete / np.sqrt(np.maximum(Dte, 1e-30)); ztr[Dtr <= 0] = 0; zte[Dte <= 0] = 0
            ci = np.concatenate([i0, np.tile((np.arange(K) + c0)[:, None].T, (len(pix), 1))], axis=1)
            cz = np.concatenate([z0, ztr.T], axis=1); ct = np.concatenate([t0, zte.T], axis=1)
            o = np.argpartition(-cz, top_k - 1, axis=1)[:, :top_k]
            i0, z0, t0 = np.take_along_axis(ci, o, 1).astype(np.int32), np.take_along_axis(cz, o, 1), np.take_along_axis(ct, o, 1)
        o = np.argsort(-z0, axis=1)
        assert np.array_equal(res[j]["idx"], np.take_along_axis(i0, o, 1))
        assert np.array_equal(res[j]["z"][0], np.take_along_axis(z0, o, 1))
        assert np.array_equal(res[j]["z"][1], np.take_along_axis(t0, o, 1))


def _cubic():
    from scipy.spatial.transform import Rotation
    return Rotation.create_group("O").as_matrix()


def test_select_candidates_rules():
    """Ratio and held-out tests, sep_deg within a tile, links between neighbouring tiles (with crystal symmetry, not around
    the row ends), min_tiles."""
    from scipy.spatial.transform import Rotation
    from diffractom.orientation_search import select_candidates
    rz = lambda a: Rotation.from_euler("z", a, degrees=True).as_matrix()
    rx = lambda a: Rotation.from_euler("x", a, degrees=True).as_matrix()
    R = np.array([np.eye(3), rz(3.0), rx(30.0), rz(90.0), rz(20.0)])      # R3 is equivalent to R0 under the cubic group
    Ny, Nx, top = 4, 4, 4
    idx = np.full((Ny * Nx, top), -1, np.int32)
    z0 = np.full((Ny * Nx, top), -np.inf, np.float32)
    z1 = np.zeros((Ny * Nx, top), np.float32)

    def tile(y, x, *peaks):
        t = y * Nx + x
        for i, (k, a, b) in enumerate(peaks):
            idx[t, i], z0[t, i], z1[t, i] = k, a, b
    tile(0, 0, (0, 50, 20), (2, 30, 3), (4, 5, 20))                  # R0 passes; R2 fails held-out; R4 fails the ratio
    tile(0, 1, (1, 40, 20))                                          # R1, 3 deg from R0: linked to tile (0, 0)
    tile(2, 0, (0, 50, 20), (1, 45, 20), (2, 40, 20))                # R1 within sep_deg of R0: dropped; R2 stays
    tile(3, 0, (3, 60, 20))                                          # R3 == R0 by symmetry: linked to R0 of tile (2, 0)
    tile(1, 3, (0, 70, 20))                                          # the row end above (2, 0): not a neighbour
    real = {"idx": idx, "z": np.stack([z0, z1])}
    null = {"idx": idx * 0, "z": np.stack([np.full(z0.shape, 10, np.float32), np.full(z0.shape, 5, np.float32)])}
    pix = np.arange(Ny * Nx)
    c = select_candidates(real, null, R, pix, (Ny, Nx), _cubic(), ratio=2.0, sep_deg=8.0, link_deg=5.0, max_per_tile=3, min_tiles=2)
    assert len(c) == 2
    assert sorted(c.score.tolist()) == [90.0, 110.0]
    assert [sorted(t.tolist()) for t in c.tiles] == [sorted([2 * Nx, 3 * Nx]), [0, 1]]       # strongest first
    assert np.array_equal(c.rotations[0], R[3]) and np.array_equal(c.rotations[1], R[0])
    c1 = select_candidates(real, null, R, pix, (Ny, Nx), _cubic(), ratio=2.0, min_tiles=1)
    assert len(c1) == 4 and any(t.tolist() == [1 * Nx + 3] for t in c1.tiles) and any(t.tolist() == [2 * Nx] for t in c1.tiles)
    c2 = select_candidates(real, null, R, pix, (Ny, Nx), _cubic(), ratio=7.0, min_tiles=1)
    assert len(c2) == 1 and c2.score[0] == 70.0                      # only (1, 3) reaches 7 x the null
    assert len(select_candidates(real, null, R, pix, (Ny, Nx), _cubic(), ratio=100.0, min_tiles=1)) == 0


def test_refine_candidates_finds_the_orientation():
    """A noise-free crystal: a candidate started 2.5 deg away moves to within half a degree of the truth."""
    import pyopencl.array as clarray
    from scipy.spatial.transform import Rotation
    from diffractom import LocalSinglePhaseForwardOperator, LocalSupport, MatrixGrid, SinglePhaseForwardOperator
    from diffractom.orientation_search import Candidates, refine_candidates
    ctx, q = _context()
    cfg = dict(CFG, N_Omega=60, angle_range=[0, 180], N_eta=60)
    mat = _material()
    U = Rotation.random(random_state=4).as_matrix()
    sig = np.deg2rad(2.0)
    op = SinglePhaseForwardOperator(cfg=dict(cfg), material=mat, grid=MatrixGrid(U[None], np.array([sig])), max_gb=1e-3, normalized=True,
                                    pf_mode="sparse", projector="native", ctx=ctx, queue=q)
    x = np.zeros(op.coeff_shape, np.float32)
    x[0, 4:8, 6:10] = 1.0
    d_true = op.direct(x)
    e, d = d_true, np.ones(d_true.shape, np.float32)
    shape = (int(cfg["Ny"]), int(cfg["Nx"]))
    start = Rotation.from_rotvec(np.deg2rad([1.5, -1.5, 1.2])).as_matrix() @ U

    def make_local(R, sigs, k, iy, ix):
        return LocalSinglePhaseForwardOperator(dict(cfg), mat, MatrixGrid(R, sigs), 1e-3, support=LocalSupport(k, iy, ix, len(R), shape), normalized=True,
                                               pf_mode="generated", ctx=ctx, queue=q)
    tiles = np.array([2 * 7 + 3, 2 * 7 + 4, 3 * 7 + 3, 3 * 7 + 4])       # tiles of 2 voxels on a (6, 7) scoring image
    cand = Candidates(start[None], np.array([1.0]), [tiles], (6, 7))
    res = refine_candidates(make_local, cand, e, d, np.ones(cfg["N_Omega"], bool), 2, shape, levels=((1.0, 3.0, 2.0), (0.25, 1.0, 2.0)))
    angle = np.degrees(np.arccos(np.clip((np.trace(res.rotations[0].T @ U) - 1) / 2, -1, 1)))
    assert angle < 0.5, angle
    assert clarray is not None


def test_support_and_prune():
    from diffractom.orientation_search import Candidates, candidate_support, orientation_weights, prune_added
    c = Candidates(np.tile(np.eye(3), (2, 1, 1)), np.ones(2), [np.array([0]), np.array([5])], (3, 4))
    k, iy, ix = candidate_support(c, 2, (6, 7))
    assert sorted(zip(k.tolist(), iy.tolist(), ix.tolist())) == [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 2, 2), (1, 2, 3), (1, 3, 2), (1, 3, 3)]
    k2, iy2, ix2 = candidate_support(c, 2, (6, 7), dilate=1, mask=np.zeros((6, 7), bool))
    assert len(k2) == 0
    w = orientation_weights(np.array([[[1.0, -1.0]], [[0.0, 0.0]], [[2.0, 3.0]]]))
    assert w.tolist() == [1.0, 0.0, 5.0]
    assert orientation_weights(np.array([1.0, 2.0, -1.0]), k=np.array([0, 1, 1])).tolist() == [1.0, 2.0]
    assert prune_added(np.array([9.0, 0.0, 10.0, 0.5, 0.0]), 2, rel=0.1).tolist() == [True, True, True, False, False]
