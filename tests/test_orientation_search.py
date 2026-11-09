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
