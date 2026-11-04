"""FISTAGram: least squares in Gram form, against the operator's own forward and adjoint and against FISTAHuber."""
import numpy as np
import pytest


def _context():
    cl = pytest.importorskip("pyopencl")
    try:
        ctx = cl.create_some_context(interactive=False)
    except Exception:  # no OpenCL device
        pytest.skip("no OpenCL device")
    return ctx, cl.CommandQueue(ctx)


def _operator(ctx, queue, pf_mode="dense", max_gb=0.5, projector="native", K=30, sigma_deg=4.0):
    from diffractom import Grid, Material, SinglePhaseForwardOperator
    mat = Material.from_lattice_parameters(a=4.05, b=4.05, c=4.05, alpha=90, beta=90, gamma=90, symmetry_group="cubic",
                                           wavelength_A=0.35, hkl_list=np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0]]))
    cfg = {"Nx": 20, "Ny": 20, "My": 21, "N_Omega": 30, "angle_range": [0, 180], "cor_offset": 0.5,
           "N_eta": 36, "eta_angle_range": [0, 360], "energy": 12.398 / 0.35, "k_direction_0": [0, 0, 1],
           "p_direction_0": [1, 0, 0], "detector_direction_origin": [0, -1, 0],
           "detector_direction_positive_90": [0, 0, -1]}
    np.random.seed(5)
    grid = Grid.from_random_fundamental_zone(K, "cubic", np.deg2rad(sigma_deg))
    return SinglePhaseForwardOperator(cfg=cfg, material=mat, grid=grid, max_gb=max_gb, normalized=True, pf_mode=pf_mode,
                                      projector=projector, ctx=ctx, queue=queue)


def _data(op, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.random(op.coeff_shape) ** 4 * op.support_mask()[None]).astype(np.float32)
    b = op.direct(x)
    b = (b + 0.05 * b.std() * rng.normal(size=b.shape)).astype(np.float32)
    w = np.ones(op.N_seg, np.float32)
    w[rng.random(op.N_seg) < 0.2] = 0.0
    return b, w


def _check_gradient(op, weights, build_gb=1.0):
    import pyopencl.array as clarray
    from diffractom import FISTAGram
    b, _ = _data(op)
    rng = np.random.default_rng(1)
    y = (rng.random(op.coeff_shape) * op.support_mask()[None]).astype(np.float32)
    g = FISTAGram(op, build_gb=build_gb)
    g.prepare(b, weights)
    yd = clarray.to_device(op.queue, y)
    grad = clarray.empty_like(yd)
    f = g.gradient(yd, grad)
    W = np.ones(op.N_seg) if weights is None else np.asarray(weights, np.float64).ravel() ** 2
    r = (op.direct(y).astype(np.float64) - b) * W
    ref = op.adjoint(r.astype(np.float32))
    f_ref = 0.5 * np.sum((op.direct(y).astype(np.float64) - b) ** 2 * W)
    np.testing.assert_allclose(f, f_ref, rtol=2e-4)
    err = np.linalg.norm(grad.get() - ref) / np.linalg.norm(ref)
    assert err < 2e-4, err
    g.free()


@pytest.mark.parametrize("pf_mode", ["dense", "sparse"])
def test_gradient_equals_the_operator(pf_mode):
    ctx, queue = _context()
    op = _operator(ctx, queue, pf_mode)
    _, w = _data(op)
    _check_gradient(op, None)
    _check_gradient(op, w)                                       # 0/1 weights
    _check_gradient(op, 0.5 + np.random.default_rng(2).random(op.N_seg).astype(np.float32))  # general weights


def test_built_in_chunks():
    """G and B from several pole-figure chunks (and several operator batches) are those of one chunk."""
    ctx, queue = _context()
    op = _operator(ctx, queue, "dense", max_gb=2e-5)
    assert len(op.batches) > 1
    _, w = _data(op)
    _check_gradient(op, w, build_gb=3e-5)   # chunks of 4 orientations


def test_fft_projector():
    pytest.importorskip("pyvkfft")
    ctx, queue = _context()
    op = _operator(ctx, queue, "dense", projector="fft")
    _, w = _data(op)
    _check_gradient(op, w)


def test_iterates_equal_fista_huber():
    """With 0/1 weights and huber_delta = inf, FISTAHuber minimises the same objective with the same steps."""
    import pyopencl.array as clarray
    from diffractom import FISTAGram, FISTAHuber
    ctx, queue = _context()
    op = _operator(ctx, queue, "dense")
    b, w = _data(op)
    g = FISTAGram(op)
    g.prepare(b, w)
    L = 1.1 * g.estimate_L(niter=30)
    g = FISTAGram(op, L=L)
    x_gram = g.run(np.zeros(op.coeff_shape, np.float32), b, niter=15, weights=w)
    xh = clarray.zeros(queue, op.coeff_shape, np.float32)
    FISTAHuber(op, prox_kind="nonneg", L=L, huber_delta=1e30).run(xh, b, niter=15, weights=w)
    x_h = xh.get()
    err = np.linalg.norm(x_gram - x_h) / np.linalg.norm(x_h)
    assert err < 1e-3, err
    f = [s["f"] for s in g.iter_stats]
    assert f[-1] < f[0]


def test_rejects_unsupported_inputs():
    from diffractom import FISTAGram
    ctx, queue = _context()
    op = _operator(ctx, queue, "dense")
    b, _ = _data(op)
    with pytest.raises(ValueError):
        FISTAGram(op).prepare(b, np.ones(op.data_shape, np.float32))   # per-point weights
    with pytest.raises(ValueError):
        FISTAGram(op, prox_kind="nonneg_tv")
