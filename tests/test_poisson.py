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

    def __init__(self, ctx, queue, A, coeff_shape, data_shape):
        self.ctx, self.queue, self.A = ctx, queue, A
        self.coeff_shape, self.data_shape = coeff_shape, data_shape

    def direct_cl(self, x, out):
        out.set((self.A @ x.get().ravel()).astype(np.float32).reshape(self.data_shape))

    def adjoint_cl(self, y, out):
        out.set((self.A.T @ y.get().ravel()).astype(np.float32).reshape(self.coeff_shape))


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
