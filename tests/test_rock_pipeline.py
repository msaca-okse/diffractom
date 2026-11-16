"""The pieces of the rock-fit pipeline: OSEMPoisson, em_scales, PoissonTerm / evaluation.poisson_deviance,
LocalSupport.from_coefficients and OrientationRefiner (small problems of test_subsets)."""
import numpy as np
import pytest

from test_subsets import _context, _local_operator, _operator


def _data(op, rng, density=0.1, gain=0.05):
    x_true = (rng.random(op.coeff_shape) * (rng.random(op.coeff_shape) < density)).astype(np.float32)
    if getattr(op, "coefficient_layout", "dense") == "dense":
        x_true *= op.support_mask()[None]
    b = (rng.poisson(op.direct(x_true) / gain) * gain).astype(np.float32)
    return x_true, b


def test_local_support_from_coefficients():
    from diffractom import LocalSupport
    c = np.zeros((4, 5, 6), np.float32)
    c[0, 1, 1], c[1, 1, 1], c[2, 1, 1], c[3, 1, 1] = 1.0, 0.5, 0.05, 0.2
    c[2, 3, 4] = 2.0
    s = LocalSupport.from_coefficients(c, top=2, rel=0.0, dilate=0)
    assert sorted(zip(s.k.tolist(), s.iy.tolist(), s.ix.tolist())) == [(0, 1, 1), (1, 1, 1), (2, 3, 4)]
    s = LocalSupport.from_coefficients(c, top=4, rel=0.15, dilate=0, chunk_elems=8)   # chunks of 2 voxels
    assert sorted(zip(s.k.tolist(), s.iy.tolist(), s.ix.tolist())) == [(0, 1, 1), (1, 1, 1), (2, 3, 4), (3, 1, 1)]
    s = LocalSupport.from_coefficients(c, top=1, dilate=1)
    assert s.n_pairs == 10   # two discs of radius 1 (5 voxels each)
    s = LocalSupport.from_coefficients(c, top=4, dilate=0, min_voxel=0.6)   # voxel (1, 1): largest 1.0 < 0.6 x 2.0
    assert sorted(zip(s.k.tolist(), s.iy.tolist(), s.ix.tolist())) == [(2, 3, 4)]


def test_poisson_term_matches_host_deviance():
    import pyopencl.array as clarray
    from diffractom.evaluation import block_split, poisson_deviance
    from diffractom.optimization import PoissonTerm
    ctx, queue = _context()
    rng = np.random.default_rng(1)
    shape = (10, 7, 12)
    mu = rng.uniform(0.0, 3.0, shape).astype(np.float32)
    mu[0, 0, :3] = -0.5
    b = rng.poisson(np.maximum(mu, 0) + 0.2).astype(np.float32)
    w = (rng.random(12) > 0.2).astype(np.float32)
    g = rng.uniform(0.5, 2.0, (7, 12)).astype(np.float32)
    bg = np.full(12, 0.2, np.float32)
    fit = block_split(10, block=2, period=5, held=1)
    pt = PoissonTerm(queue, shape, b, w, g, bg, fit)
    f_gpu = pt.deviance(clarray.to_device(queue, mu))
    f_host = poisson_deviance(b, mu, gain=g, weights=w, background=bg, rows=fit)
    assert f_gpu == pytest.approx(f_host, rel=1e-5)
    n = pt.n_points()
    assert poisson_deviance(b, mu, gain=g, weights=w, background=bg, rows=fit, per_point=True) == \
        pytest.approx(f_host / n, rel=1e-12)


def test_osem_reaches_the_minimiser():
    """OSEM (4 subsets) and MLEM lower the deviance to near the FISTAPoisson minimum; final_f agrees with
    evaluation.poisson_deviance."""
    import pyopencl.array as clarray
    from diffractom import FISTAPoisson, OSEMPoisson
    from diffractom.evaluation import poisson_deviance
    ctx, queue = _context()
    op = _operator(ctx, queue, "sparse")
    rng = np.random.default_rng(7)
    _, b = _data(op, rng)
    gain = 0.05
    ref = FISTAPoisson(op)
    ref.run(clarray.zeros(queue, op.coeff_shape, np.float32), b, niter=3000, gain=gain)
    f_ref = ref.final_stats["final_f"]
    em = OSEMPoisson(op, subsets=4)
    x = em.run(np.zeros(op.coeff_shape, np.float32), b, 200, gain=gain, diagnostics_interval=50)
    f = [s["f"] for s in em.iter_stats if "f" in s]
    print("f_ref", f_ref, "OSEM", f)
    assert all(np.diff(f) <= 1e-6 * f[0])
    assert em.final_f <= f_ref * 1.03   # OSEM cycles near the minimum (2 % above it here)
    assert x.min() >= 0
    assert (x * ~op.support_mask()[None]).max() == 0
    assert em.final_f == pytest.approx(poisson_deviance(b, op.direct(x), gain=gain), rel=1e-4)
    ml = OSEMPoisson(op, subsets=None)
    ml.run(x.copy(), b, 50, gain=gain, start_floor=0, diagnostics_interval=50)
    print("MLEM from OSEM", ml.final_f)
    assert ml.final_f <= em.final_f * (1 + 1e-4)


def test_osem_ignores_held_out_angles():
    """Angles in no subset (held out) do not change the fit, and the deviance counts the fitted angles only."""
    from diffractom import OSEMPoisson
    from diffractom.evaluation import block_split, interleaved_subsets, poisson_deviance
    ctx, queue = _context()
    op = _operator(ctx, queue, "sparse")
    rng = np.random.default_rng(2)
    _, b = _data(op, rng)
    fit = block_split(op.data_shape[0], block=3, period=5, held=2)
    subs = interleaved_subsets(fit, 3)
    b2 = b.copy()
    b2[~fit] = rng.uniform(0, 10, b2[~fit].shape)
    xs = []
    for bb in (b, b2):
        em = OSEMPoisson(op, subsets=subs, seed=4)
        xs.append(em.run(np.zeros(op.coeff_shape, np.float32), bb, 20, gain=0.05, diagnostics_interval=0))
    np.testing.assert_array_equal(xs[0], xs[1])
    assert em.final_f == pytest.approx(poisson_deviance(b2, op.direct(xs[1]), gain=0.05, rows=fit), rel=1e-4)


def test_osem_local_basis():
    from diffractom import OSEMPoisson
    ctx, queue = _context()
    op = _local_operator(ctx, queue, "sparse")
    rng = np.random.default_rng(7)
    _, b = _data(op, rng, density=0.3)
    em = OSEMPoisson(op, subsets=4, lam=0.0)
    x = em.run(np.zeros(op.coeff_shape, np.float32), b, 60, gain=0.05, diagnostics_interval=20)
    f = [s["f"] for s in em.iter_stats if "f" in s]
    print("local OSEM", f)
    assert f[-1] < f[0]
    allowed = op.coefficient_mask() > 0
    assert x[~allowed].max(initial=0) == 0 and x[allowed].max() > 0


def test_em_scales_recovers_scales():
    """Two components with one scale per (component, channel), and with tied channel scales."""
    from diffractom import em_scales
    ctx, queue = _context()
    op = _operator(ctx, queue, "sparse")
    rng = np.random.default_rng(3)
    x1 = (rng.random(op.coeff_shape) * (rng.random(op.coeff_shape) < 0.1) * op.support_mask()[None]).astype(np.float32)
    x2 = (rng.random(op.coeff_shape) * (rng.random(op.coeff_shape) < 0.1) * op.support_mask()[None]).astype(np.float32)
    P1, P2 = op.direct(x1), op.direct(x2)
    n_seg = op.data_shape[2]
    ch = np.arange(n_seg) % 3                     # 3 channels (e.g. the rings of each eta bin)
    a_true = np.array([[1.5, 0.7, 1.0], [0.4, 2.0, 1.2]])
    gain = 0.002
    mu = a_true[0, ch] * P1 + a_true[1, ch] * P2 + 0.01
    b = (rng.poisson(mu / gain) * gain).astype(np.float32)
    a = em_scales(queue, [P1, P2], b, channels=ch, gain=gain, background=np.full(n_seg, 0.01, np.float32),
                  iters=300, tol=1e-7)
    print("scales", a)
    np.testing.assert_allclose(a, a_true, rtol=0.03)
    # tied: one factor per channel for both components (their ratio fixed at a0)
    t_true = np.array([1.3, 0.8, 1.0])
    mu = t_true[ch] * (P1 + 2 * P2)
    b = (rng.poisson(mu / gain) * gain).astype(np.float32)
    import pyopencl.array as clarray
    a = em_scales(queue, [clarray.to_device(queue, P1), P2], clarray.to_device(queue, b), channels=ch, ties=[0, 0],
                  a0=np.array([[1.0] * 3, [2.0] * 3]), gain=gain, iters=300, tol=1e-7)
    np.testing.assert_allclose(a[1] / a[0], 2.0, rtol=1e-6)
    np.testing.assert_allclose(a[0], t_true, rtol=0.03)


def test_orientation_refiner_recovers_orientations():
    """Orientations perturbed by 0.3 deg move back towards the truth and the deviance falls."""
    from scipy.spatial.transform import Rotation
    from diffractom import OSEMPoisson
    from diffractom.differentiable import OrientationRefiner, set_pf_parameters
    ctx, queue = _context()
    op = _operator(ctx, queue, "sparse")
    rng = np.random.default_rng(11)
    x_true, b = _data(op, rng, density=0.15, gain=0.01)
    U_true = np.asarray(op.grid_rot_cpu, np.float64).copy()
    d = rng.normal(size=(op.K, 3))
    d *= np.radians(0.3) / np.linalg.norm(d, axis=1, keepdims=True)
    set_pf_parameters(op, rotations=Rotation.from_rotvec(d).as_matrix() @ U_true)

    def mis_deg():
        R = np.asarray(op.grid_rot_cpu) @ U_true.transpose(0, 2, 1)
        return np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec(), axis=1))

    def solve(x):
        return OSEMPoisson(op, subsets=4).run(x, b, 20, gain=0.01, diagnostics_interval=0)

    x = solve(np.zeros(op.coeff_shape, np.float32))
    m0 = mis_deg()
    ref = OrientationRefiner(op, b, gain=0.01, sigma_min_deg=0.5)
    f0 = ref.loss(x)
    x = ref.run(x, solve, 4, verbose=True)
    m1 = mis_deg()
    used = x_true.reshape(op.K, -1).sum(axis=1) > 0
    print("misorientation median", np.median(m0[used]), "->", np.median(m1[used]), "f", f0, "->", ref.history[-1]["f_fit"])
    assert ref.history[-1]["f_fit"] < f0
    assert np.median(m1[used]) < 0.5 * np.median(m0[used])
