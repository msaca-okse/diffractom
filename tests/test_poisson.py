"""Poisson likelihood: the counting gain of integrated data (utils.counting) and FISTAPoisson."""
import numpy as np
import pytest

from diffractom.utils.counting import gain_from_pairs, pyfai_gain


def test_gain_from_pairs():
    rng = np.random.default_rng(0)
    lam = rng.uniform(5, 500, size=(20000, 8))
    gain = np.linspace(0.1, 2.0, 8)
    a = rng.poisson(lam) * gain
    b = rng.poisson(lam) * gain
    np.testing.assert_allclose(gain_from_pairs(a, b, axis=0), gain, rtol=0.05)


def test_pyfai_gain_matches_poisson_frames():
    """Frames with Poisson counts proportional to the solid angle, integrated with the solid-angle correction:
    var / mean of the ring-window sums over many frames equals the predicted gain."""
    pyFAI = pytest.importorskip("pyFAI")
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    det = Detector(pixel1=1e-4, pixel2=1e-4, max_shape=(160, 160))
    ai = AzimuthalIntegrator(dist=0.05, poni1=0.004, poni2=0.006, detector=det, wavelength=1e-10)
    mask = np.zeros(det.max_shape, np.int8)
    mask[70:80, :] = 1  # a detector gap
    kw = dict(npt_rad=60, npt_azim=24, unit="q_nm^-1", method=("no", "csr", "cython"))
    windows = [(6.0, 9.0), (12.0, 14.0)]
    gain, npix = pyfai_gain(ai, mask, ring_windows=windows, **kw)
    sa = ai.solidAngleArray(det.max_shape)
    rng = np.random.default_rng(1)
    res = ai.integrate2d_ng(np.ones(det.max_shape, np.float32), mask=mask, **kw)
    radial = np.asarray(res.radial)
    sel = [(radial >= lo) & (radial < hi) for lo, hi in windows]
    sums = []
    for _ in range(300):
        frame = rng.poisson(40.0 * sa).astype(np.float32)
        I = np.nan_to_num(np.asarray(ai.integrate2d_ng(frame, mask=mask, **kw).intensity))
        sums.append(np.stack([I[:, s].sum(axis=1) for s in sel], axis=1))
    sums = np.array(sums)
    ok = np.isfinite(gain) & (npix > 50)
    measured = sums.var(axis=0, ddof=1) / sums.mean(axis=0)
    assert ok.sum() > 10
    ratio = measured[ok] / gain[ok]
    assert abs(np.median(ratio) - 1) < 0.05 and np.all(np.abs(ratio - 1) < 0.35)


class _HostOperator:
    """A dense nonnegative matrix as an operator with direct_cl / adjoint_cl (via the host), for the solver tests."""

    def __init__(self, ctx, queue, A, coeff_shape, data_shape, adjoint_scale=1.0):
        self.ctx, self.queue, self.A = ctx, queue, A
        self.coeff_shape, self.data_shape = coeff_shape, data_shape
        self.adjoint_scale = adjoint_scale  # like the native projector's angle weights: adjoint = scale x transpose

    def direct_cl(self, x, out):
        out.set((self.A @ x.get().ravel()).astype(np.float32).reshape(self.data_shape))

    def adjoint_cl(self, y, out):
        out.set((self.adjoint_scale * (self.A.T @ y.get().ravel())).astype(np.float32).reshape(self.coeff_shape))


def _context():
    cl = pytest.importorskip("pyopencl")
    try:
        ctx = cl.create_some_context(interactive=False)
    except Exception:  # no OpenCL device
        pytest.skip("no OpenCL device")
    if "cl_khr_fp64" not in ctx.devices[0].extensions:
        pytest.skip("the device has no double precision")
    return ctx, cl.CommandQueue(ctx)


def _problem(seed=0):
    rng = np.random.default_rng(seed)
    coeff_shape, data_shape = (6, 2, 2), (5, 4, 12)
    n, m = int(np.prod(coeff_shape)), int(np.prod(data_shape))
    A = rng.random((m, n)) ** 4  # nonnegative, uneven rows
    x_true = rng.random(n) * (rng.random(n) < 0.4) * 50
    gain = rng.uniform(0.2, 3.0, data_shape[-1])
    g_all = np.broadcast_to(gain, data_shape).ravel()
    b = rng.poisson(A @ x_true / g_all) * g_all
    return A, coeff_shape, data_shape, gain, g_all, b


def _reference(A, b, g, eps):
    """Minimiser of the same deviance by L-BFGS-B (float64, tight tolerances)."""
    from scipy.optimize import minimize

    def fun(x):
        mu = A @ x + eps * g
        val = np.sum((mu - b + np.where(b > 0, b * np.log(np.where(b > 0, b, 1) / mu), 0)) / g)
        return val, A.T @ ((1 - b / mu) / g)
    r = minimize(fun, np.ones(A.shape[1]), jac=True, method="L-BFGS-B", bounds=[(0, None)] * A.shape[1],
                 options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-12})
    return r.x, r.fun


def test_fista_poisson_reaches_the_minimiser():
    from diffractom.optimization.fista_poisson import FISTAPoisson
    ctx, queue = _context()
    A, cs, ds, gain, g_all, b = _problem()
    op = _HostOperator(ctx, queue, A.astype(np.float32), cs, ds)
    solver = FISTAPoisson(op, support=None)
    x = solver.run(np.zeros(cs, np.float32), b.reshape(ds), niter=1500, gain=gain)
    x_ref, f_ref = _reference(A, b, g_all, solver.eps_counts)
    f = solver.final_stats["final_f"]
    assert f <= f_ref * (1 + 1e-4) + 1e-6
    np.testing.assert_allclose(x.ravel(), x_ref, atol=2e-2 * x_ref.max())


def test_fista_poisson_resume_and_scale():
    """Two runs of 300 iterations (resumed from solver.state) reach the objective of one run of 600; data and gain
    multiplied by s give s x."""
    from diffractom.optimization.fista_poisson import FISTAPoisson
    ctx, queue = _context()
    A, cs, ds, gain, g_all, b = _problem(1)
    op = _HostOperator(ctx, queue, A.astype(np.float32), cs, ds)
    one = FISTAPoisson(op, support=None)
    x_one = one.run(np.zeros(cs, np.float32), b.reshape(ds), niter=600, gain=gain)
    two = FISTAPoisson(op, support=None)
    x_two = two.run(np.zeros(cs, np.float32), b.reshape(ds), niter=300, gain=gain)
    x_two = two.run(x_two, b.reshape(ds), niter=300, gain=gain, state=two.state)
    assert two.state["it"] == 600
    assert abs(two.final_stats["final_f"] - one.final_stats["final_f"]) <= 1e-3 * one.final_stats["final_f"]
    s = 7.0
    x_s = FISTAPoisson(op, support=None).run(np.zeros(cs, np.float32), (s * b).reshape(ds), niter=600,
                                            gain=s * gain)
    np.testing.assert_allclose(x_s, s * x_one, rtol=1e-3, atol=1e-3 * s * x_one.max())


def test_fista_poisson_scaled_adjoint():
    """An adjoint that is a multiple of the transpose (the native projector's over a full turn: 2) changes nothing."""
    from diffractom.optimization.fista_poisson import FISTAPoisson
    ctx, queue = _context()
    A, cs, ds, gain, g_all, b = _problem(2)
    xs, fs = [], []
    for scale in (1.0, 2.0):
        op = _HostOperator(ctx, queue, A.astype(np.float32), cs, ds, adjoint_scale=scale)
        solver = FISTAPoisson(op, support=None)
        xs.append(solver.run(np.zeros(cs, np.float32), b.reshape(ds), niter=800, gain=gain))
        fs.append(solver.final_stats["final_f"])
        assert solver.final_stats["backtracks"] < 40
        if solver.final_stats["adjoint_scale"] is not None:
            assert abs(solver.final_stats["adjoint_scale"] - scale) < 0.05 * scale
    assert abs(fs[1] - fs[0]) <= 1e-4 * fs[0]
    np.testing.assert_allclose(xs[1], xs[0], atol=1e-2 * xs[0].max())


def test_fista_poisson_streamed_equals_gpu():
    """With the coefficients in host memory (NumPy x0, several orientation batches) the iterates are those of the
    GPU-resident solver (pyopencl x0), up to the order of the sums."""
    import pyopencl.array as clarray
    from diffractom import Grid, Material, SinglePhaseForwardOperator
    from diffractom.optimization.fista_poisson import FISTAPoisson
    ctx, queue = _context()
    mat = Material.from_lattice_parameters(a=4.05, b=4.05, c=4.05, alpha=90, beta=90, gamma=90, symmetry_group="cubic",
                                           wavelength_A=0.35, hkl_list=np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0]]))
    cfg = {"Nx": 16, "Ny": 16, "My": 17, "N_Omega": 30, "angle_range": [0, 360], "cor_offset": 0.5,
           "N_eta": 36, "eta_angle_range": [0, 360], "energy": 12.398 / 0.35, "k_direction_0": [0, 0, 1],
           "p_direction_0": [1, 0, 0], "detector_direction_origin": [0, -1, 0],
           "detector_direction_positive_90": [0, 0, -1]}
    np.random.seed(3)
    grid = Grid.from_random_fundamental_zone(40, "cubic", np.deg2rad(4.0))
    # dense PF batches of a few orientations, so that the streamed solver goes through several batches
    op = SinglePhaseForwardOperator(cfg=cfg, material=mat, grid=grid, max_gb=5e-5, normalized=True, pf_mode="dense",
                                    ctx=ctx, queue=queue)
    assert len(op.batches) >= 3
    rng = np.random.default_rng(4)
    x_true = (rng.random(op.coeff_shape) * (rng.random(op.coeff_shape) < 0.1) * op.support_mask()[None]).astype(np.float32)
    mu = op.direct(x_true)
    gain = 0.05
    b = (rng.poisson(mu / gain) * gain).astype(np.float32)
    runs = {}
    for name, x0 in (("gpu", clarray.zeros(queue, op.coeff_shape, np.float32)),
                     ("streamed", np.zeros(op.coeff_shape, np.float32))):
        solver = FISTAPoisson(op)
        x = solver.run(x0, b, niter=60, gain=gain)
        runs[name] = (x.get() if name == "gpu" else x, [s["f"] for s in solver.iter_stats], solver.final_stats)
    assert runs["streamed"][2]["streamed"]
    np.testing.assert_allclose(runs["streamed"][1], runs["gpu"][1], rtol=1e-4)
    xg, xs = runs["gpu"][0], runs["streamed"][0]
    assert np.linalg.norm(xs - xg) <= 1e-3 * np.linalg.norm(xg)


def test_pdhg_poisson_reaches_the_minimiser():
    """Chambolle-Pock reaches the L-BFGS-B minimiser for a reasonable step balance (the best of three; the automatic
    balance is calibrated for the diffraction operators, not for this small dense matrix)."""
    from diffractom.optimization.pdhg_poisson import PDHGPoisson
    ctx, queue = _context()
    A, cs, ds, gain, g_all, b = _problem()
    op = _HostOperator(ctx, queue, A.astype(np.float32), cs, ds)
    x_ref, f_ref = None, None
    best = None
    for rho in (1e-4, 1e-2, 1.0):
        solver = PDHGPoisson(op, support=None, rho=rho)
        x = solver.run(np.zeros(cs, np.float32), b.reshape(ds), niter=4000, gain=gain)
        if best is None or solver.final_stats["final_f"] < best[0]:
            best = (solver.final_stats["final_f"], x)
    x_ref, f_ref = _reference(A, b, g_all, solver.eps_counts)
    assert best[0] <= f_ref * (1 + 1e-3) + 1e-6
    np.testing.assert_allclose(best[1].ravel(), x_ref, atol=3e-2 * x_ref.max())
