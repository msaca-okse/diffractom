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


@pytest.mark.parametrize("kernel", ["trapezoid", "hat"])
def test_projector_gradients_and_updates(kernel):
    from diffractom.differentiable import ProjGradient, ReferenceModel, set_proj_parameters
    ctx, queue = _context()
    op = _operator(ctx, queue, "sparse", kernel=kernel)
    rng = np.random.default_rng(2)
    c = (rng.random(op.coeff_shape) * op.support_mask()[None]).astype(np.float32)
    r = rng.normal(size=op.data_shape).astype(np.float32)
    ref = ReferenceModel(op)
    prm = ref.params0()
    g_ref = ref.vjp(c, prm, r, groups=("shift", "cor"))
    g = ProjGradient(op)(c, r)
    assert np.abs(g["shift"] - g_ref["shift"]).max() < 1e-4 * np.abs(g_ref["shift"]).max()
    assert abs(g["cor"] - g_ref["cor"]) < 1e-4 * abs(g_ref["cor"])
    prm["cor"] = prm["cor"] + 0.3
    prm["shift"] = rng.normal(0, 0.4, op.N_Omega)
    d_ref = ref.forward(c, prm)
    set_proj_parameters(op, cor=prm["cor"], shift=prm["shift"])
    assert np.abs(op.direct(c) - d_ref).max() < 2e-5 * np.abs(d_ref).max()


def test_set_geometry_is_absolute():
    """set_geometry sets offsets from the geometry the operator was built with: moving it twice ends at the second
    setting, equal to the reference model there."""
    from diffractom.differentiable import ReferenceModel, set_geometry
    ctx, queue = _context()
    op = _operator(ctx, queue, "generated")
    rng = np.random.default_rng(3)
    c = (rng.random(op.coeff_shape) * op.support_mask()[None]).astype(np.float32)
    ref = ReferenceModel(op)
    prm = ref.params0()
    prm["eta_offset"] = 0.012
    prm["axis_tilt"] = np.array([0.01, -0.015])
    prm["two_theta"] = prm["two_theta"] + 0.0015
    prm["cor"] = prm["cor"] + 0.3
    prm["shift"] = rng.normal(0, 0.2, op.N_Omega)
    d_ref = ref.forward(c, prm)
    set_geometry(op, eta_offset=-0.02, axis_tilt=[0.03, 0.0], two_theta=prm["two_theta"] - 0.004, cor=0.0,
                 shift=np.zeros(op.N_Omega))
    set_geometry(op, eta_offset=prm["eta_offset"], axis_tilt=prm["axis_tilt"], two_theta=prm["two_theta"],
                 cor=prm["cor"], shift=prm["shift"])
    assert np.abs(op.direct(c) - d_ref).max() < 2e-5 * np.abs(d_ref).max()


def test_absorption_reference():
    """Self-absorption (AbsorptionReference): mu = 0 is the plain reference; the gradient with respect to mu equals
    central finite differences under strong absorption; the coefficient gradient is the adjoint."""
    from diffractom import Grid, Material, SinglePhaseForwardOperator
    from diffractom.differentiable.absorption import AbsorptionReference
    from diffractom.differentiable.reference import ReferenceModel, perturb
    ctx, queue = _context()
    mat = Material.from_lattice_parameters(a=4.05, b=4.05, c=4.05, alpha=90, beta=90, gamma=90, symmetry_group="cubic",
                                           wavelength_A=0.35, hkl_list=np.array([[1, 1, 1], [2, 0, 0]]))
    cfg = {"Nx": 8, "Ny": 7, "My": 11, "N_Omega": 6, "angle_range": [0, 360], "cor_offset": 0.4, "N_eta": 8,
           "eta_angle_range": [0, 360], "energy": 12.398 / 0.35, "k_direction_0": [0, 0, 1], "p_direction_0": [1, 0, 0],
           "detector_direction_origin": [0, -1, 0], "detector_direction_positive_90": [0, 0, -1]}
    np.random.seed(5)
    grid = Grid.from_random_fundamental_zone(4, "cubic", np.deg2rad(10.0))
    op = SinglePhaseForwardOperator(cfg=cfg, material=mat, grid=grid, max_gb=0.2, normalized=True, pf_mode="dense",
                                    projector="native", ctx=ctx, queue=queue)
    rng = np.random.default_rng(0)
    ref = ReferenceModel(op, cut=None)
    ab = AbsorptionReference(ref, op)
    prm = ref.params0()
    supp = op.support_mask().astype(np.float64)
    c = rng.random(op.coeff_shape) * supp[None]
    prm["mu"] = np.zeros((op.Ny, op.Nx))
    assert np.abs(ab.forward(c, prm) - ref.forward(c, prm)).max() < 1e-12 * np.abs(ref.forward(c, prm)).max()
    prm["mu"] = 0.1 * supp * (1 + 0.5 * rng.random(supp.shape))
    r = rng.normal(size=op.data_shape)
    g = ab.vjp(c, prm, r, groups=("c", "mu"))
    L = lambda p_: float(np.sum(r * ab.forward(c, p_)))
    for ix in [(3, 4), (1, 2), (5, 6)]:
        fd = (L(perturb(prm, "mu", ix, 1e-6)) - L(perturb(prm, "mu", ix, -1e-6))) / 2e-6
        assert abs(g["mu"][ix] - fd) <= 1e-6 * max(abs(fd), 1e-3 * np.abs(r).sum())
    x = rng.random(op.coeff_shape)
    assert abs(np.sum(g["c"] * x) / np.sum(r * ab.forward(x, prm)) - 1) < 1e-10


def test_window_model_lattice_gradients():
    """Integration windows as channels (WindowedReference): equal to the operator built with
    rings=windows_to_rings(...); the gradients with respect to the lattice parameters and sig_rel equal central finite
    differences."""
    from diffractom import Grid, Material, SinglePhaseForwardOperator
    from diffractom.differentiable.reference import ReferenceModel, perturb
    from diffractom.differentiable.windows import WindowedReference, windows_to_rings
    ctx, queue = _context()
    mat = Material.from_lattice_parameters(a=4.05, b=4.05, c=4.05, alpha=90, beta=90, gamma=90, symmetry_group="cubic",
                                           wavelength_A=0.35, hkl_list=np.array([[1, 1, 1], [2, 0, 0], [3, 1, 1],
                                                                                 [2, 2, 2]]))
    windows, sig = [(2.55, 2.90), (2.90, 3.30), (4.95, 5.26), (5.26, 5.60)], 0.04
    cfg = {"Nx": 8, "Ny": 7, "My": 11, "N_Omega": 6, "angle_range": [0, 360], "cor_offset": 0.4, "N_eta": 8,
           "eta_angle_range": [0, 360], "energy": 12.398 / 0.35, "k_direction_0": [0, 0, 1], "p_direction_0": [1, 0, 0],
           "detector_direction_origin": [0, -1, 0], "detector_direction_positive_90": [0, 0, -1]}
    np.random.seed(5)
    grid = Grid.from_random_fundamental_zone(4, "cubic", np.deg2rad(8.0))
    op = SinglePhaseForwardOperator(cfg=cfg, material=mat, grid=grid, max_gb=0.2, normalized=True, pf_mode="dense",
                                    projector="native", rings=windows_to_rings(mat, windows, sig), ctx=ctx, queue=queue)
    rng = np.random.default_rng(0)
    c = rng.random(op.coeff_shape) * op.support_mask()[None]
    wr = WindowedReference(ReferenceModel(op), op, windows, sig)
    d = op.direct(c.astype(np.float32))
    assert np.abs(wr.forward(c, wr.params0()) - d).max() < 2e-5 * np.abs(d).max()
    wr = WindowedReference(ReferenceModel(op, cut=None), op, windows, sig)
    prm = wr.params0()
    prm["lattice"] = prm["lattice"] + np.array([0.004, -0.003, 0.002, 0.1, -0.05, 0.08])
    r = rng.normal(size=op.data_shape)
    g = wr.vjp(c, prm, r)
    L = lambda p_: float(np.sum(r * wr.forward(c, p_)))
    for i, h in ((0, 1e-6), (2, 1e-6), (5, 1e-5)):
        fd = (L(perturb(prm, "lattice", (i,), h)) - L(perturb(prm, "lattice", (i,), -h))) / (2 * h)
        assert abs(g["lattice"][i] - fd) <= 1e-5 * max(abs(fd), 1e-3 * np.abs(r).sum())
    fd = (L(perturb(prm, "sig_rel", None, 1e-7)) - L(perturb(prm, "sig_rel", None, -1e-7))) / 2e-7
    assert abs(g["sig_rel"] - fd) <= 1e-5 * abs(fd)


def test_beam_blur():
    """BeamBlur: the width derivative equals central finite differences, its "u" output is the adjoint, and the
    intensity layer's gradients equal finite differences."""
    from diffractom.differentiable.data_model import BeamBlur, IntensityLayer
    rng = np.random.default_rng(2)
    shape = (5, 13, 6)
    u, r = rng.random(shape), rng.normal(size=shape)
    bb = BeamBlur(shape)
    w = 0.8
    g = bb.vjp(u, w, r)
    fd = (np.sum(r * bb.forward(u, w + 1e-6)) - np.sum(r * bb.forward(u, w - 1e-6))) / 2e-6
    assert abs(g["beam_width"] - fd) < 1e-6 * abs(fd)
    v = rng.random(shape)
    assert abs(np.sum(g["u"] * v) / np.sum(r * bb.forward(v, w)) - 1) < 1e-12
    lay = IntensityLayer(shape, 3)
    prm = lay.params0()
    prm = {k: v_ * (1 + 0.1 * rng.random(np.shape(v_))) if k != "background" else rng.random(np.shape(v_))
           for k, v_ in prm.items()}
    gl = lay.vjp(u, prm, r)
    for key, ix in (("angle_scale", 2), ("trans_scale", 4), ("ring_scale", 1), ("background", 3)):
        p1 = {k: np.array(v_, copy=True) for k, v_ in prm.items()}
        p2 = {k: np.array(v_, copy=True) for k, v_ in prm.items()}
        p1[key][ix] += 1e-6
        p2[key][ix] -= 1e-6
        fd = (np.sum(r * lay.forward(u, p1)) - np.sum(r * lay.forward(u, p2))) / 2e-6
        assert abs(gl[key][ix] - fd) < 1e-6 * max(abs(fd), 1e-3)


def test_windowed_operator_gpu():
    """WindowedOperator (family channels + mixing on the GPU) equals WindowedReference, forward and gradients, also
    after set_lattice."""
    from diffractom import Grid, Material, SinglePhaseForwardOperator
    from diffractom.differentiable.reference import ReferenceModel
    from diffractom.differentiable.windows import WindowedOperator, WindowedReference, windows_to_rings
    ctx, queue = _context()
    mat = Material.from_lattice_parameters(a=4.05, b=4.05, c=4.05, alpha=90, beta=90, gamma=90, symmetry_group="cubic",
                                           wavelength_A=0.35, hkl_list=np.array([[1, 1, 1], [2, 0, 0], [3, 1, 1],
                                                                                 [2, 2, 2]]))
    windows, sig = [(2.55, 2.90), (2.90, 3.30), (4.95, 5.26), (5.26, 5.60)], 0.04
    cfg = {"Nx": 10, "Ny": 9, "My": 13, "N_Omega": 8, "angle_range": [0, 360], "cor_offset": 0.4, "N_eta": 12,
           "eta_angle_range": [0, 360], "energy": 12.398 / 0.35, "k_direction_0": [0, 0, 1], "p_direction_0": [1, 0, 0],
           "detector_direction_origin": [0, -1, 0], "detector_direction_positive_90": [0, 0, -1]}
    np.random.seed(5)
    grid = Grid.from_random_fundamental_zone(6, "cubic", np.deg2rad(6.0))
    kw = dict(cfg=cfg, grid=grid, max_gb=1e-5, pf_mode="sparse", projector="native", ctx=ctx, queue=queue)
    wo = WindowedOperator(mat, windows, sig, **kw)
    opw = SinglePhaseForwardOperator(material=mat, normalized=True, rings=windows_to_rings(mat, windows, sig), **kw)
    wr = WindowedReference(ReferenceModel(opw), opw, windows, sig)
    rng = np.random.default_rng(0)
    c = (rng.random(wo.op.coeff_shape) * wo.op.support_mask()[None]).astype(np.float32)
    r = rng.normal(size=wo.data_shape).astype(np.float32)
    prm = wr.params0()
    prm["lattice"] = prm["lattice"] + np.array([0.004, -0.003, 0.002, 0.1, -0.05, 0.08])
    wo.set_lattice(prm["lattice"])
    d = wr.forward(c, prm)
    assert np.abs(wo.forward(c) - d).max() < 2e-5 * np.abs(d).max()
    g_ref, g_gpu = wr.vjp(c, prm, r), wo.vjp(c, r)
    assert np.abs(g_gpu["lattice"] - g_ref["lattice"]).max() < 1e-4 * np.abs(g_ref["lattice"]).max()
    assert abs(g_gpu["sig_rel"] - g_ref["sig_rel"]) < 1e-4 * abs(g_ref["sig_rel"])


def test_windowed_operator_unnormalized():
    """normalized=False windows (shares I_f w_f, explicit families, a window without channels): WindowedOperator ==
    the operator built with the same rings; the reference's lattice gradient == central finite differences; GPU ==
    reference after set_lattice; adjoint == transpose."""
    import pyopencl.array as clarray
    from diffractom import Grid, Material, SinglePhaseForwardOperator
    from diffractom.differentiable.reference import ReferenceModel, perturb
    from diffractom.differentiable.windows import WindowedOperator, WindowedReference, windows_to_rings
    ctx, queue = _context()
    mat = Material.from_sites([("Al", 0, 0, 0, 1.0, 0.0108)], a=4.0495, space_group="F m -3 m", wavelength_A=0.35,
                              hkl_list=np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1], [2, 2, 2]]))
    windows, sig = [(2.55, 2.90), (2.90, 3.30), (3.6, 3.7), (4.95, 5.26), (5.26, 5.60)], 0.04
    fams = [ix for ix, _ in windows_to_rings(mat, windows, sig)]
    cfg = {"Nx": 10, "Ny": 9, "My": 13, "N_Omega": 8, "angle_range": [0, 360], "cor_offset": 0.4, "N_eta": 12,
           "eta_angle_range": [0, 360], "energy": 12.398 / 0.35, "k_direction_0": [0, 0, 1], "p_direction_0": [1, 0, 0],
           "detector_direction_origin": [0, -1, 0], "detector_direction_positive_90": [0, 0, -1]}
    np.random.seed(5)
    grid = Grid.from_random_fundamental_zone(6, "cubic", np.deg2rad(6.0))
    kw = dict(cfg=cfg, grid=grid, max_gb=1e-5, pf_mode="sparse", projector="native", ctx=ctx, queue=queue,
              normalized=False)
    rings = windows_to_rings(mat, windows, sig, families=fams)
    rings[2] = (np.array([2]), np.array([0.0]))           # the empty window: a placeholder of weight 0
    opw = SinglePhaseForwardOperator(material=mat, rings=rings, **kw)
    fams[2] = np.array([], dtype=int)
    wo = WindowedOperator(mat, windows, sig, families=fams, **kw)
    rng = np.random.default_rng(0)
    c = (rng.random(wo.op.coeff_shape) * wo.op.support_mask()[None]).astype(np.float32)
    r = rng.normal(size=wo.data_shape).astype(np.float32)
    d = opw.direct(c)
    assert np.abs(wo.forward(c) - d).max() < 2e-6 * np.abs(d).max()
    keep = np.ones((1, 1, 12, 5), np.float32)
    keep[..., 2] = 0                                      # the reference needs a family per window: compare the rest
    keep = keep.reshape(1, 1, -1)
    r = r * keep
    opr = SinglePhaseForwardOperator(material=mat, rings=windows_to_rings(mat, windows, sig, families=[
        f if len(f) else np.array([1]) for f in fams]), **kw)
    wf = WindowedReference(ReferenceModel(opr, cut=None), opr, windows, sig)
    prm = wf.params0()
    prm["lattice"] = prm["lattice"] + np.array([0.004, -0.003, 0.002, 0.1, -0.05, 0.08])
    g = wf.vjp(c, prm, r)
    L = lambda p_: float(np.sum(r * wf.forward(c, p_)))
    for i, h in ((0, 1e-6), (4, 1e-5)):
        fd = (L(perturb(prm, "lattice", (i,), h)) - L(perturb(prm, "lattice", (i,), -h))) / (2 * h)
        assert abs(g["lattice"][i] - fd) <= 1e-5 * max(abs(fd), 1e-3 * np.abs(r).sum())
    fd = (L(perturb(prm, "sig_rel", None, 1e-7)) - L(perturb(prm, "sig_rel", None, -1e-7))) / 2e-7
    assert abs(g["sig_rel"] - fd) <= 1e-5 * abs(fd)
    wr = WindowedReference(ReferenceModel(opr), opr, windows, sig)
    wo.set_lattice(prm["lattice"])
    dr = wr.forward(c, prm) * keep
    assert np.abs(wo.forward(c) * keep - dr).max() < 5e-6 * np.abs(dr).max()
    g_ref, g_gpu = wr.vjp(c, prm, r), wo.vjp(c, r)
    assert np.abs(g_gpu["lattice"] - g_ref["lattice"]).max() < 1e-4 * np.abs(g_ref["lattice"]).max()
    assert abs(g_gpu["sig_rel"] - g_ref["sig_rel"]) < 1e-3 * abs(g_ref["sig_rel"])
    cg, rg = clarray.to_device(queue, c), clarray.to_device(queue, r)
    od, oa = clarray.empty(queue, wo.data_shape, np.float32), clarray.empty(queue, wo.coeff_shape, np.float32)
    a1 = float(np.sum(wo.direct_cl(cg, od).get().astype(np.float64) * r))
    a2 = float(np.sum(wo.adjoint_cl(rg, oa).get().astype(np.float64) * c))
    assert abs(a1 - a2) < 1e-5 * abs(a1)


def test_absorption_operator_gpu():
    """AbsorptionOperator (GPU) equals AbsorptionReference with the same exit classes (forward, adjoint, mu gradient);
    mu = 0 is the plain operator."""
    from diffractom.differentiable.absorption import AbsorptionOperator, AbsorptionReference, exit_classes
    from diffractom.differentiable.reference import ReferenceModel
    ctx, queue = _context()
    op = _operator(ctx, queue, "sparse")
    classes = exit_classes(op, width_deg=5.0)
    abo = AbsorptionOperator(op, classes, angle_chunk=5)
    rng = np.random.default_rng(4)
    c = (rng.random(op.coeff_shape) * op.support_mask()[None]).astype(np.float32)
    r = rng.normal(size=op.data_shape).astype(np.float32)
    d0 = op.direct(c)
    assert np.abs(abo.direct(c) - d0).max() <= 1e-6 * np.abs(d0).max()
    supp = op.support_mask().astype(np.float64)
    mu = 0.1 * supp * (1 + 0.5 * rng.random(supp.shape))
    abo.set_mu(mu)
    abr = AbsorptionReference(ReferenceModel(op), op, classes)
    prm = abr.ref.params0()
    prm["mu"] = mu
    rel = lambda a, b: np.abs(a - b).max() / np.abs(b).max()
    g = abr.vjp(c, prm, r, groups=("c", "mu"))
    assert rel(abo.direct(c), abr.forward(c, prm)) < 2e-5
    assert rel(abo.adjoint(r), g["c"]) < 2e-5
    assert rel(abo.mu_gradient(c, r), g["mu"]) < 2e-5
