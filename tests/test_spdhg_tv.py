"""SPDHGPoisson with total variation: the objective of a long run equals that of an independent float64 reference
(Chambolle-Pock on the explicit matrix, the same TV), the reported objective equals the NumPy one at the returned x,
streamed == GPU-resident, a TV state continues the run, and the local basis with a full support gives the dense
operator's iterates."""
import numpy as np
import pytest


def _context():
    cl = pytest.importorskip("pyopencl")
    try:
        ctx = cl.create_some_context(interactive=False)
    except Exception:
        pytest.skip("no OpenCL device")
    return ctx, cl.CommandQueue(ctx)


def _cfg(n=8, n_omega=8, n_eta=12):
    return {"Nx": n, "Ny": n, "My": n + 1, "N_Omega": n_omega, "angle_range": [0, 180], "cor_offset": 0.0,
            "N_eta": n_eta, "eta_angle_range": [0, 360], "energy": 12.398 / 0.35, "k_direction_0": [0, 0, 1],
            "p_direction_0": [1, 0, 0], "detector_direction_origin": [0, -1, 0],
            "detector_direction_positive_90": [0, 0, -1]}


def _material():
    from diffractom import Material
    return Material.from_lattice_parameters(a=4.05, b=4.05, c=4.05, alpha=90, beta=90, gamma=90, symmetry_group="cubic",
                                            wavelength_A=0.35, hkl_list=np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0]]))


def _grid(K):
    from diffractom import Grid
    np.random.seed(5)
    return Grid.from_random_fundamental_zone(K, "cubic", np.deg2rad(6.0))


def _problem(ctx, queue, K=4, n=8, max_gb=0.2, **kw):
    from diffractom import SinglePhaseForwardOperator
    op = SinglePhaseForwardOperator(cfg=_cfg(n), material=_material(), grid=_grid(K), max_gb=max_gb, normalized=True,
                                    pf_mode="sparse", ctx=ctx, queue=queue, **kw)
    rng = np.random.default_rng(3)
    # piecewise-constant images (TV-friendly): two half planes per orientation
    yy, xx = np.mgrid[:n, :n]
    x_true = np.zeros(op.coeff_shape, np.float32)
    for k in range(K):
        a = rng.uniform(0, np.pi)
        x_true[k] = rng.uniform(0.5, 1.5) * ((np.cos(a) * (xx - n / 2) + np.sin(a) * (yy - n / 2)) > 0)
    x_true *= op.support_mask()[None]
    gain = 0.02
    b = (rng.poisson(op.direct(x_true) / gain) * gain).astype(np.float32)
    return op, b, gain


def _dense_matrix(op):
    """A as a NumPy float64 matrix (data rows x coefficients), column by column."""
    n = int(np.prod(op.coeff_shape))
    A = np.zeros((int(np.prod(op.data_shape)), n))
    e = np.zeros(op.coeff_shape, np.float32)
    for j in range(n):
        e.flat[j] = 1.0
        A[:, j] = op.direct(e).ravel()
        e.flat[j] = 0.0
    return A


def _tv_ops(K, n, free):
    """Forward differences (right, down) between free pixels of every image: sparse D (2 rows per pixel)."""
    from scipy.sparse import coo_matrix
    rows, cols, vals = [], [], []
    for k in range(K):
        for iy in range(n):
            for ix in range(n):
                j = (k * n + iy) * n + ix
                for c, (dy, dx) in enumerate(((0, 1), (1, 0))):
                    jy, jx = iy + dy, ix + dx
                    if jy < n and jx < n:
                        j2 = (k * n + jy) * n + jx
                        if free[j] and free[j2]:
                            r = 2 * j + c
                            rows += [r, r]
                            cols += [j, j2]
                            vals += [-1.0, 1.0]
    return coo_matrix((vals, (rows, cols)), shape=(2 * K * n * n, K * n * n)).tocsr()


def _objective(A, D, x, b, gain, tv, eps=1e-3):
    """Poisson deviance (as the solver's: sum (mu - b + b log(b / mu)) / g, >= 0) + tv x TV."""
    mu = np.maximum(A @ x, 0.0) + eps * gain
    f = float(np.sum((mu - b + np.where(b > 0, b * np.log(np.where(b > 0, b, 1.0) / mu), 0.0)) / gain))
    g = (D @ x).reshape(-1, 2)
    return f + tv * float(np.sqrt((g ** 2).sum(axis=1)).sum())


def _reference(A, D, b, gain, tv, free, eps=1e-3, n_iter=20000):
    """Diagonally preconditioned Chambolle-Pock (alpha = 1) in float64 on K = [A; D]."""
    import scipy.sparse as sp
    Ka = sp.vstack([sp.csr_matrix(A), D]).tocsr()
    absK = abs(Ka)
    sig = 1.0 / np.maximum(np.asarray(absK.sum(axis=1)).ravel(), 1e-30)
    tau = 1.0 / np.maximum(np.asarray(absK.sum(axis=0)).ravel(), 1e-30)
    tau[~free] = 0.0
    nA = A.shape[0]
    x = np.zeros(A.shape[1])
    xb = x.copy()
    y = np.zeros(Ka.shape[0])
    r = eps * gain
    sc = 1.0 / gain
    for _ in range(n_iter):
        v = y + sig * (Ka @ xb)
        # Poisson rows: prox of sigma F*, F(u) = sc (u + r - b log(u + r))
        va = v[:nA] + sig[:nA] * r
        y[:nA] = 0.5 * (va + sc - np.sqrt((va - sc) ** 2 + 4 * sig[:nA] * sc * b))
        q = v[nA:].reshape(-1, 2)
        nrm = np.maximum(1.0, np.sqrt((q ** 2).sum(axis=1)) / tv)
        y[nA:] = (q / nrm[:, None]).ravel()
        xn = np.maximum(0.0, x - tau * (Ka.T @ y))
        xb = 2 * xn - x
        x = xn
    return x


def test_spdhg_tv_matches_reference():
    from diffractom import SPDHGPoisson
    ctx, queue = _context()
    op, b, gain = _problem(ctx, queue)
    A = _dense_matrix(op)
    K, n = op.K, op.Nx
    free = (op.support_mask()[None].repeat(K, 0).ravel() > 0) & (A.sum(axis=0) > 0)
    D = _tv_ops(K, n, free)
    bb = b.ravel().astype(np.float64)
    for tv in (0.0, 2.0):
        x_ref = _reference(A, D, bb, gain, max(tv, 1e-30), free)
        f_ref = _objective(A, D, x_ref, bb, gain, tv)
        sp = SPDHGPoisson(op, subsets=4, tv=tv, seed=1)
        x = sp.run(np.zeros(op.coeff_shape, np.float32), b, 2000, gain=gain, diagnostics_interval=2000)
        f = _objective(A, D, x.ravel().astype(np.float64), bb, gain, tv)
        assert f <= f_ref * (1 + 1e-3), (tv, f, f_ref)
        # the solver's own objective (deviance + tv x TV) agrees with NumPy's at the returned x
        assert abs(sp.final_stats["final_f"] - f) <= 1e-4 * f, (sp.final_stats["final_f"], f)
        if tv > 0:
            assert np.abs(x.ravel() - x_ref).max() <= 0.05 * np.abs(x_ref).max()


def test_spdhg_tv_streamed_resident_and_resume():
    import pyopencl.array as clarray
    from diffractom import SPDHGPoisson
    ctx, queue = _context()
    from test_subsets import _operator           # several orientation batches
    op = _operator(ctx, queue, "sparse", max_gb=1e-4)
    assert len(op.batches) > 1
    rng = np.random.default_rng(9)
    x_true = (rng.random(op.coeff_shape) * (rng.random(op.coeff_shape) < 0.2) * op.support_mask()[None]).astype(np.float32)
    gain = 0.02
    b = (rng.poisson(op.direct(x_true) / gain) * gain).astype(np.float32)
    ref = SPDHGPoisson(op, subsets=3, tv=1.0, seed=2)
    x_ref = ref.run(clarray.zeros(queue, op.coeff_shape, np.float32), b, 6, gain=gain, diagnostics_interval=3).get()
    f_ref = [s["f"] for s in ref.iter_stats]
    for resident in ("none", ["x", "qx"], "all"):
        sp = SPDHGPoisson(op, subsets=3, tv=1.0, seed=2, resident=resident)
        x = sp.run(np.zeros(op.coeff_shape, np.float32), b, 6, gain=gain, diagnostics_interval=3)
        np.testing.assert_allclose(x, x_ref, rtol=1e-6, atol=1e-7 * np.abs(x_ref).max())
        np.testing.assert_allclose([s["f"] for s in sp.iter_stats if s["f"] is not None],
                                   [v for v in f_ref if v is not None], rtol=1e-6)
    two = SPDHGPoisson(op, subsets=3, tv=1.0, seed=2)
    x = two.run(np.zeros(op.coeff_shape, np.float32), b, 4, gain=gain, diagnostics_interval=4)
    x = two.run(x, b, 2, gain=gain, diagnostics_interval=2, state=two.state)
    np.testing.assert_allclose(x, x_ref, rtol=1e-5, atol=1e-6 * np.abs(x_ref).max())
    with pytest.raises(ValueError):
        SPDHGPoisson(op, subsets=3, tv=1.0).run(x, b, 1, gain=gain, state={"y": two.state["y"], "epoch": 6})


def test_spdhg_tv_local_full_support_equals_dense():
    from diffractom import LocalSinglePhaseForwardOperator, LocalSupport, SPDHGPoisson
    ctx, queue = _context()
    op, b, gain = _problem(ctx, queue, K=6)
    K, n = op.K, op.Nx
    mask = np.repeat(op.support_mask()[None], K, axis=0)
    lop = LocalSinglePhaseForwardOperator(_cfg(n), _material(), _grid(K), 0.2, support=LocalSupport.from_mask(mask),
                                          normalized=True, pf_mode="sparse", split_patches=False, ctx=ctx, queue=queue)
    xs = {}
    for name, o in (("dense", op), ("local", lop)):
        sp = SPDHGPoisson(o, subsets=3, tv=1.5, seed=4)
        x = sp.run(np.zeros(o.coeff_shape, np.float32), b, 20, gain=gain, diagnostics_interval=20)
        xs[name] = x if name == "dense" else lop.to_dense(x)
    np.testing.assert_allclose(xs["local"], xs["dense"], rtol=1e-4, atol=1e-5 * np.abs(xs["dense"]).max())
