"""reflection_two_theta: every reflection of a channel diffracts at its own Bragg angle. A window is the weighted sum of
its reflections, each evaluated as a one-reflection channel; all PF modes agree; the float64 reference reproduces the
operator and its derivatives equal finite differences; PFGradient equals the reference; the local operator equals the
global one; the memory of the probed directions does not grow with the number of channels."""
import numpy as np
import pytest


def _context():
    cl = pytest.importorskip("pyopencl")
    try:
        ctx = cl.create_some_context(interactive=False)
    except Exception:
        pytest.skip("no OpenCL device")
    return ctx, cl.CommandQueue(ctx)


CFG = {"Nx": 14, "Ny": 12, "My": 17, "N_Omega": 16, "angle_range": [0, 360], "cor_offset": 0.7, "N_eta": 36,
       "eta_angle_range": [0, 360], "energy": 12.398 / 0.35, "k_direction_0": [0, 0, 1], "p_direction_0": [1, 0, 0],
       "detector_direction_origin": [0, -1, 0], "detector_direction_positive_90": [0, 0, -1]}
HKL = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, -1, 0], [0, 0, 1], [1, 0, 1]])
WINDOWS = [([0, 1], [1.0, 0.6]), ([2, 3], [0.5, 1.0]), ([4], [1.0])]


def _material():
    from diffractom import Material
    return Material.from_lattice_parameters(a=8.1, b=12.8, c=7.1, alpha=94, beta=116, gamma=88, crystal_system="triclinic",
                                            wavelength_A=0.35, hkl_list=HKL)


def _grid(K=10, sig=3.0, seed=5):
    from scipy.spatial.transform import Rotation
    from diffractom import MatrixGrid
    return MatrixGrid(Rotation.random(K, random_state=seed).as_matrix(), np.full(K, np.deg2rad(sig)))


def _operator(ctx, queue, pf_mode="sparse", rings=WINDOWS, refl=True, **kw):
    from diffractom import SinglePhaseForwardOperator
    return SinglePhaseForwardOperator(cfg=dict(CFG), material=_material(), grid=_grid(), max_gb=1e-4, normalized=True, pf_mode=pf_mode,
                                      projector="native", ctx=ctx, queue=queue, rings=rings, reflection_two_theta=refl, **kw)


def _coeffs(op, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(op.coeff_shape) * op.support_mask()[None]).astype(np.float32)


@pytest.mark.parametrize("pf_mode", ["dense", "sparse", "generated"])
def test_window_is_the_sum_of_its_reflections(pf_mode):
    ctx, queue = _context()
    op = _operator(ctx, queue, pf_mode)
    c = _coeffs(op)
    d = op.direct(c).reshape(op.N_Omega, op.My, op.N_eta, op.N_peaks)
    used = [f for ix, _ in WINDOWS for f in ix]
    op1 = _operator(ctx, queue, "dense", rings=[([f], [1.0]) for f in used], refl=False)
    d1 = op1.direct(c).reshape(op.N_Omega, op.My, op.N_eta, len(used))
    m = np.asarray(_material().reflections["multiplicity"], np.float64)
    pos = 0
    for p, (ix, w) in enumerate(WINDOWS):
        share = m[ix] * np.asarray(w)
        share = share / share.sum()
        exp = sum(s * d1[..., pos + i] for i, s in enumerate(share))
        pos += len(ix)
        assert np.abs(d[..., p] - exp).max() <= 3e-5 * np.abs(exp).max(), p
    # and the channel's mean angle is a different model
    d_mean = _operator(ctx, queue, "dense", refl=False).direct(c).reshape(d.shape)
    assert np.abs(d_mean[..., 0] - d[..., 0]).max() > 0.02 * np.abs(d[..., 0]).max()


def test_modes_agree_and_adjoint():
    ctx, queue = _context()
    ops = [_operator(ctx, queue, m) for m in ("dense", "sparse", "generated")]
    c = _coeffs(ops[0])
    ds = [o.direct(c) for o in ops]
    assert np.abs(ds[0] - ds[1]).max() <= 2e-6 * np.abs(ds[0]).max()
    assert np.abs(ds[1] - ds[2]).max() == 0
    rng = np.random.default_rng(3)
    y = rng.normal(size=ops[1].data_shape).astype(np.float32)
    a = np.sum(ds[1].astype(np.float64) * y)
    b = np.sum(ops[1].adjoint(y).astype(np.float64) * c)
    assert abs(a - b) <= 2e-5 * abs(a)


def test_reference_and_finite_differences():
    from scipy.spatial.transform import Rotation
    from diffractom.differentiable.reference import ReferenceModel, perturb
    ctx, queue = _context()
    op = _operator(ctx, queue, "dense", pf_cutoff_sigma=4.0)
    c = _coeffs(op)
    d = op.direct(c)
    ref0 = ReferenceModel(op)
    assert np.abs(ref0.forward(c, ref0.params0()) - d).max() < 3e-5 * np.abs(d).max()
    ref = ReferenceModel(op, cut=None)
    rng = np.random.default_rng(0)
    prm = ref.params0()
    prm["rot"] = rng.normal(0, 0.01, (op.K, 3))
    r = rng.normal(size=op.data_shape)
    g = ref.vjp(c, prm, r)
    L = lambda p_: float(np.sum(r * ref.forward(c, p_)))
    for grp, ix, h in (("rot", (3, 1), 1e-6), ("sigma", (5,), 1e-7), ("two_theta", (1,), 1e-6), ("eta_offset", None, 1e-6), ("axis_tilt", (1,), 1e-6)):
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


@pytest.mark.parametrize("pf_mode", ["sparse", "generated"])
def test_gpu_gradients_and_updates(pf_mode):
    from diffractom.differentiable import PFGradient, ReferenceModel, rotate_orientations, set_pf_parameters
    ctx, queue = _context()
    op = _operator(ctx, queue, pf_mode)
    rng = np.random.default_rng(1)
    c = _coeffs(op, 1)
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
    assert np.abs(op.direct(c) - d_ref).max() < 3e-5 * np.abs(d_ref).max()


def test_local_operator_equals_global():
    from diffractom import LocalSinglePhaseForwardOperator, LocalSupport
    ctx, queue = _context()
    op = _operator(ctx, queue, "sparse")
    mask = op.support_mask()
    iy, ix = np.nonzero(mask)
    k = np.repeat(np.arange(op.K), len(iy))
    sup = LocalSupport(k, np.tile(iy, op.K), np.tile(ix, op.K), op.K, (op.Ny, op.Nx))
    lop = LocalSinglePhaseForwardOperator(dict(CFG), _material(), _grid(), 1e-4, support=sup, normalized=True, rings=WINDOWS,
                                          reflection_two_theta=True, ctx=ctx, queue=queue)
    c = _coeffs(op)
    d = op.direct(c)
    dl = lop.direct(lop.from_dense(c))
    assert np.abs(d - dl).max() <= 3e-6 * np.abs(d).max()


def test_memory_does_not_grow_with_channels():
    ctx, queue = _context()
    a = _operator(ctx, queue, "sparse", rings=WINDOWS)
    b = _operator(ctx, queue, "sparse", rings=[([f], [1.0]) for f in (0, 1, 2, 3, 4, 5)])
    assert a.coords_cpu.nbytes == b.coords_cpu.nbytes
    assert _operator(ctx, queue, "sparse", rings=WINDOWS, refl=False).coords_cpu.nbytes == a.N_peaks * a.coords_cpu.nbytes
