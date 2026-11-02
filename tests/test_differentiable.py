"""The differentiable model: the float64 reference reproduces the operator, its analytic derivatives equal central
finite differences, the GPU gradients (PFGradient) equal the reference's, and set_pf_parameters moves the operator to
the reference's model at the new parameters."""
import numpy as np
import pytest


def _context():
    cl = pytest.importorskip("pyopencl")
    try:
        ctx = cl.create_some_context(interactive=False)
    except Exception:
        pytest.skip("no OpenCL device")
    return ctx, cl.CommandQueue(ctx)


def _operator(ctx, queue, pf_mode="sparse", quadrature="auto", kernel="trapezoid"):
    from diffractom import Grid, Material, SinglePhaseForwardOperator
    mat = Material.from_lattice_parameters(a=4.05, b=4.05, c=4.05, alpha=90, beta=90, gamma=90, symmetry_group="cubic",
                                           wavelength_A=0.35, hkl_list=np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0]]))
    cfg = {"Nx": 14, "Ny": 12, "My": 17, "N_Omega": 16, "angle_range": [0, 360], "cor_offset": 0.7, "N_eta": 24,
           "eta_angle_range": [0, 360], "energy": 12.398 / 0.35, "k_direction_0": [0, 0, 1], "p_direction_0": [1, 0, 0],
           "detector_direction_origin": [0, -1, 0], "detector_direction_positive_90": [0, 0, -1]}
    np.random.seed(5)
    grid = Grid.from_random_fundamental_zone(12, "cubic", np.deg2rad(6.0))
    return SinglePhaseForwardOperator(cfg=cfg, material=mat, grid=grid, max_gb=1e-5, normalized=True, pf_mode=pf_mode,
                                      projector="native", radon_kernel=kernel, quadrature=quadrature, ctx=ctx,
                                      queue=queue)


@pytest.mark.parametrize("quadrature,kernel", [("auto", "trapezoid"), ("midpoint", "hat")])
def test_reference_forward_and_finite_differences(quadrature, kernel):
    from scipy.spatial.transform import Rotation
    from diffractom.differentiable.reference import ReferenceModel, perturb
    ctx, queue = _context()
    op = _operator(ctx, queue, "dense", quadrature, kernel)
    rng = np.random.default_rng(0)
    c = (rng.random(op.coeff_shape) * op.support_mask()[None]).astype(np.float32)
    d = op.direct(c)
    d_ref = ReferenceModel(op).forward(c, ReferenceModel(op).params0())
    assert np.abs(d_ref - d).max() < 2e-5 * np.abs(d).max()
    ref = ReferenceModel(op, cut=None)
    prm = ref.params0()
    prm["rot"] = rng.normal(0, 0.01, (op.K, 3))
    r = rng.normal(size=op.data_shape)
    g = ref.vjp(c, prm, r)
    L = lambda p_: float(np.sum(r * ref.forward(c, p_)))
    for grp, ix, h in (("rot", (3, 1), 1e-6), ("sigma", (5,), 1e-7), ("two_theta", (1,), 1e-6), ("eta_offset", None, 1e-6),
                       ("axis_tilt", (1,), 1e-6), ("cor", None, 1e-5), ("shift", (7,), 1e-5)):
        if grp == "rot":
            def moved(hh):
                out = perturb(prm, "rot", ix, 0.0)
                e = np.zeros(3)
                e[ix[1]] = hh
                out["rot"][ix[0]] = (Rotation.from_rotvec(e) * Rotation.from_rotvec(prm["rot"][ix[0]])).as_rotvec()
                return out
        else:
            moved = lambda hh, grp=grp, ix=ix: perturb(prm, grp, ix, hh)
        fd = (L(moved(h)) - L(moved(-h))) / (2 * h)
        an = float(g[grp] if ix is None else g[grp][ix])
        assert abs(an - fd) <= 1e-6 * max(abs(fd), 1e-3 * np.abs(r).sum()), (grp, an, fd)
    x = rng.random(op.coeff_shape)
    assert abs(np.sum(g["c"] * x) / np.sum(r * ref.forward(x, prm)) - 1) < 1e-10


@pytest.mark.parametrize("pf_mode", ["sparse", "generated"])
def test_gpu_gradients_and_updates(pf_mode):
    from diffractom.differentiable import PFGradient, ReferenceModel, rotate_orientations, set_pf_parameters
    ctx, queue = _context()
    op = _operator(ctx, queue, pf_mode)
    rng = np.random.default_rng(1)
    c = (rng.random(op.coeff_shape) * op.support_mask()[None]).astype(np.float32)
    r = rng.normal(size=op.data_shape).astype(np.float32)
    groups = ("rot", "sigma", "eta_offset", "axis_tilt", "two_theta")
    ref = ReferenceModel(op)
    prm = ref.params0()
    g_ref = ref.vjp(c, prm, r, groups=groups)
    g_gpu = PFGradient(op)(c, r, groups=groups)
    for k in groups:
        a, b = np.atleast_1d(g_ref[k]), np.atleast_1d(g_gpu[k])
        assert np.abs(a - b).max() < 1e-4 * np.abs(a).max(), k
    prm["rot"] = rng.normal(0, 0.02, (op.K, 3))
    prm["sigma"] = prm["sigma"] * 1.1
    prm["two_theta"] = prm["two_theta"] + 0.001
    prm["eta_offset"] = 0.01
    prm["axis_tilt"] = np.array([0.01, -0.02])
    d_ref = ref.forward(c, prm)
    set_pf_parameters(op, rotations=rotate_orientations(op.grid_rot_cpu, prm["rot"]), sigma=prm["sigma"],
                      two_theta=prm["two_theta"], eta_offset=prm["eta_offset"], axis_tilt=prm["axis_tilt"])
    assert np.abs(op.direct(c) - d_ref).max() < 2e-5 * np.abs(d_ref).max()
