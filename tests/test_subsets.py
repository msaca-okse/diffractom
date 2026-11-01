"""Omega subsets of the operator: the subset forward equals the full forward on the subset's rows (and leaves the
other rows alone), the subset adjoints of a partition sum to the full adjoint, and each subset adjoint is the
transpose of the subset forward."""
import numpy as np
import pytest


def _context():
    cl = pytest.importorskip("pyopencl")
    try:
        ctx = cl.create_some_context(interactive=False)
    except Exception:
        pytest.skip("no OpenCL device")
    return ctx, cl.CommandQueue(ctx)


def _operator(ctx, queue, pf_mode, angle_range=(0, 360)):
    from diffractom import Grid, Material, SinglePhaseForwardOperator
    mat = Material.from_lattice_parameters(a=4.05, b=4.05, c=4.05, alpha=90, beta=90, gamma=90, symmetry_group="cubic",
                                           wavelength_A=0.35, hkl_list=np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0]]))
    cfg = {"Nx": 20, "Ny": 20, "My": 21, "N_Omega": 30, "angle_range": list(angle_range), "cor_offset": 0.5,
           "N_eta": 36, "eta_angle_range": [0, 360], "energy": 12.398 / 0.35, "k_direction_0": [0, 0, 1],
           "p_direction_0": [1, 0, 0], "detector_direction_origin": [0, -1, 0],
           "detector_direction_positive_90": [0, 0, -1]}
    np.random.seed(5)
    grid = Grid.from_random_fundamental_zone(30, "cubic", np.deg2rad(4.0))
    return SinglePhaseForwardOperator(cfg=cfg, material=mat, grid=grid, max_gb=0.2, normalized=True, pf_mode=pf_mode,
                                      sparse_batch_max=8, ctx=ctx, queue=queue)


@pytest.mark.parametrize("pf_mode", ["sparse", "generated"])
@pytest.mark.parametrize("subsets", [4, "blocks"])
def test_subset_forward_and_adjoint(pf_mode, subsets):
    import pyopencl.array as clarray
    ctx, queue = _context()
    op = _operator(ctx, queue, pf_mode)
    assert op.pf_mode == pf_mode
    R = op.N_Omega
    subs = op.set_omega_subsets(4 if subsets == 4 else [np.arange(0, 7), np.arange(7, 20), np.arange(20, R)])
    rng = np.random.default_rng(6)
    x = (rng.random(op.coeff_shape) * op.support_mask()[None]).astype(np.float32)
    y = rng.random(op.data_shape).astype(np.float32)
    xd, yd = clarray.to_device(queue, x), clarray.to_device(queue, y)
    full = clarray.empty(queue, op.data_shape, np.float32)
    op.direct_cl(xd, full)
    full = full.get()
    adj = clarray.empty(queue, op.coeff_shape, np.float32)
    op.adjoint_cl(yd, adj)
    adj = adj.get()
    adj_sum = np.zeros_like(adj)
    for m, s in enumerate(subs):
        d = clarray.to_device(queue, np.full(op.data_shape, -7.0, np.float32))
        op.direct_cl(xd, d, subset=m)
        d = d.get()
        np.testing.assert_allclose(d[s], full[s], rtol=1e-5, atol=1e-6 * np.abs(full).max())
        rest = np.setdiff1d(np.arange(R), s)
        assert np.all(d[rest] == -7.0)
        a_m = clarray.empty(queue, op.coeff_shape, np.float32)
        op.adjoint_cl(yd, a_m, subset=m)
        a_m = a_m.get()
        adj_sum += a_m
        lhs = float(np.dot(d[s].ravel().astype(np.float64), y[s].ravel()))
        rhs = float(np.dot(x.ravel().astype(np.float64), a_m.ravel()))
        assert abs(lhs / rhs - 1) < 1e-5
    np.testing.assert_allclose(adj_sum, adj, rtol=1e-4, atol=1e-5 * np.abs(adj).max())


def test_subsets_need_native_sparse():
    ctx, queue = _context()
    op = _operator(ctx, queue, "dense")
    with pytest.raises(NotImplementedError):
        op.set_omega_subsets(2)


def test_spdhg_poisson_reaches_the_minimiser():
    """SPDHG over 4 angle subsets reaches the objective of a long FISTAPoisson run."""
    import pyopencl.array as clarray
    from diffractom import FISTAPoisson, SPDHGPoisson
    ctx, queue = _context()
    op = _operator(ctx, queue, "sparse")
    rng = np.random.default_rng(7)
    x_true = (rng.random(op.coeff_shape) * (rng.random(op.coeff_shape) < 0.1) * op.support_mask()[None]).astype(np.float32)
    gain = 0.05
    b = (rng.poisson(op.direct(x_true) / gain) * gain).astype(np.float32)
    ref = FISTAPoisson(op)
    ref.run(clarray.zeros(queue, op.coeff_shape, np.float32), b, niter=3000, gain=gain)
    f_ref = ref.final_stats["final_f"]
    sp = SPDHGPoisson(op, subsets=4)
    sp.run(clarray.zeros(queue, op.coeff_shape, np.float32), b, n_epochs=600, gain=gain, diagnostics_interval=100)
    assert sp.final_stats["final_f"] <= f_ref * (1 + 2e-3)
