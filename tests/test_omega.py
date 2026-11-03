"""Rotation-angle conventions (angle_range_from_frames) and the omega-subdivided operator."""
import numpy as np
import pytest

from diffractom import angle_range_from_frames


def test_angle_range_from_frames():
    om = np.arange(3001) * 0.12                       # DanMAX-like: start labels, one extra point, encoder wrap
    om_wrapped = np.where(om >= 180, om - 360, om)
    for o in (om, om_wrapped):
        rng, cen = angle_range_from_frames(o, "start", frames_per_bin=6, n_frames=3000)
        assert np.allclose(rng, [0, 360]) and np.isclose(cen[0], 0.36) and len(cen) == 500
    for label, off in (("start", 0.0), ("center", 0.25), ("end", 0.5)):
        rng, cen = angle_range_from_frames(np.arange(360) * 0.5 + off, label)
        assert np.allclose(rng, [0, 180]) and np.allclose(cen[:2], [0.25, 0.75])
    rng, cen = angle_range_from_frames(-np.arange(10) * 1.0, "start")      # decreasing rotation
    assert np.allclose(rng, [0, -10]) and np.isclose(cen[0], -0.5)
    with pytest.raises(ValueError):
        angle_range_from_frames(np.arange(10.0), "middle")
    with pytest.raises(ValueError):
        angle_range_from_frames(np.r_[0, 1, 2, 4, 5.0], "start")              # a missing frame


def _context():
    cl = pytest.importorskip("pyopencl")
    try:
        ctx = cl.create_some_context(interactive=False)
    except Exception:
        pytest.skip("no OpenCL device")
    return ctx, cl.CommandQueue(ctx)


def test_omega_subdivided_operator():
    ctx, q = _context()
    from diffractom import Grid, Material, OmegaSubdividedOperator, QuadratureRule, SinglePhaseForwardOperator
    mat = Material.from_lattice_parameters(a=4.05, b=4.05, c=4.05, alpha=90, beta=90, gamma=90, symmetry_group="cubic",
                                           wavelength_A=0.35, hkl_list=np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0]]))
    cfg = {"Nx": 20, "Ny": 20, "My": 23, "N_Omega": 24, "angle_range": [0, 180], "cor_offset": 0.5, "N_eta": 16,
           "eta_angle_range": [0, 360], "energy": 12.398 / 0.35, "k_direction_0": [0, 0, 1], "p_direction_0": [1, 0, 0],
           "detector_direction_origin": [0, -1, 0], "detector_direction_positive_90": [0, 0, -1]}
    np.random.seed(5)
    grid = Grid.from_random_fundamental_zone(20, "cubic", np.deg2rad(5.0))
    kw = dict(normalized=True, pf_mode="sparse", ctx=ctx, queue=q, verbose=False)
    op = SinglePhaseForwardOperator(cfg=cfg, material=mat, grid=grid, max_gb=0.05, quadrature=QuadratureRule(1, 4, 2, 1), **kw)
    rng = np.random.default_rng(0)
    c = (rng.random(op.coeff_shape) * op.support_mask()[None]).astype(np.float32)
    r = rng.random(op.data_shape).astype(np.float32)
    d0 = op.direct(c)
    for M in (1, 2, 4):            # the same omega quadrature points: only the projection angles differ
        so = OmegaSubdividedOperator(cfg, mat, grid, 0.05, subdivisions=M, quadrature=QuadratureRule(1, 4 // M, 2, 1), **kw)
        d, a = so.direct(c), so.adjoint(r)
        if M == 1:
            assert np.array_equal(d, d0)
        assert np.linalg.norm(d - d0) < 0.05 * np.linalg.norm(d0)
        assert abs(d.sum() / d0.sum() - 1) < 1e-4
        lhs, rhs = np.sum(d.astype(np.float64) * r), np.sum(a.astype(np.float64) * c)
        assert abs(lhs - rhs) < 1e-5 * abs(lhs)
