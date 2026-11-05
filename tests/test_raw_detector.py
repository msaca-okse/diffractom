"""Raw-detector model: a synthetic detector (pixels scattered over the rings), checked against an explicit NumPy
matrix: the model equals W A built pixel by pixel, the adjoint is its transpose, the sensitivity is the sum over all
cells, cells of several pixels equal the sum of their pixels, and the list-mode solvers reach the minimum of the
full Poisson likelihood (float64 reference on the dense matrix)."""
import numpy as np
import pytest


def _context():
    cl = pytest.importorskip("pyopencl")
    try:
        ctx = cl.create_some_context(interactive=False)
    except Exception:
        pytest.skip("no OpenCL device")
    return ctx, cl.CommandQueue(ctx)


def _material():
    from diffractom import Material
    return Material.from_lattice_parameters(a=4.05, b=4.05, c=4.05, alpha=90, beta=90, gamma=90, symmetry_group="cubic",
                                            wavelength_A=0.35, hkl_list=np.array([[1, 1, 1], [2, 0, 0]]))


def _operator(ctx, queue, n_eta=72, K=5, n=8, n_omega=12):
    from diffractom import Grid, SinglePhaseForwardOperator
    cfg = {"Nx": n, "Ny": n, "My": n + 1, "N_Omega": n_omega, "angle_range": [0, 180], "cor_offset": 0.0,
           "N_eta": n_eta, "eta_angle_range": [0, 360], "energy": 12.398 / 0.35, "k_direction_0": [0, 0, 1],
           "p_direction_0": [1, 0, 0], "detector_direction_origin": [0, -1, 0],
           "detector_direction_positive_90": [0, 0, -1]}
    np.random.seed(5)
    grid = Grid.from_random_fundamental_zone(K, "cubic", np.deg2rad(5.0))
    return SinglePhaseForwardOperator(cfg=cfg, material=_material(), grid=grid, max_gb=0.05, normalized=True,
                                      pf_mode="sparse", ctx=ctx, queue=queue)


def _pixels(op, n_pix=3000, seed=1):
    from diffractom import DetectorPixels
    rng = np.random.default_rng(seed)
    tt = op.two_theta_peaks.astype(np.float64)
    ring = rng.integers(0, len(tt), n_pix)
    two_theta = tt[ring] + rng.uniform(-3e-3, 3e-3, n_pix)
    eta = rng.uniform(0, 2 * np.pi, n_pix)
    factor = rng.uniform(0.5, 1.0, n_pix)
    ext = np.full(n_pix, 1e-3)
    return DetectorPixels(two_theta, eta, factor, np.arange(n_pix), (1, n_pix), ext, ext)


def _model_parts(op, pixels, cell_of_pixel=None, scales=None):
    from diffractom import DetectorCells, RadialProfiles
    prof = RadialProfiles.gaussian(op.two_theta_peaks, 1.2e-3, cutoff=3.0)
    cells = DetectorCells(pixels, prof, op.N_eta, op.eta_angle_range, ring_scales=scales,
                          cell_of_pixel=cell_of_pixel)
    return prof, cells


def _dense_W(cells):
    """(n_cells, n_seg) weights."""
    W = np.zeros((cells.n_cells, cells.n_seg))
    for c in range(cells.n_cells):
        a, b = cells.ptr[c], cells.ptr[c + 1]
        W[c, cells.seg[a:b]] += cells.w[a:b]
    return W


def _dense_A(op):
    n = int(np.prod(op.coeff_shape))
    A = np.zeros((int(np.prod(op.data_shape)), n))
    e = np.zeros(op.coeff_shape, np.float32)
    for j in range(n):
        e.flat[j] = 1.0
        A[:, j] = op.direct(e).ravel()
        e.flat[j] = 0.0
    return A


def _full_events(op, cells, counts):
    """An EventList with every cell of every frame (counts: (n_frames, n_cells))."""
    from diffractom import EventList
    F = op.N_Omega * op.My
    fr = np.repeat(np.arange(F), cells.n_cells)
    ce = np.tile(np.arange(cells.n_cells), F)
    return EventList(fr, ce, counts.ravel(), op.N_Omega, op.My, cells.n_cells)


def test_cells_match_pixel_model():
    """The cell weights reproduce the pixel model mu_p = factor rho(2theta) M(eta) with linear eta interpolation."""
    ctx, queue = _context()
    op = _operator(ctx, queue)
    pix = _pixels(op)
    prof, cells = _model_parts(op, pix)
    rng = np.random.default_rng(0)
    Mrow = rng.random(op.N_seg)
    from diffractom.detector import eta_interpolation
    j0, j1, w0, w1 = eta_interpolation(pix.eta, op.N_eta, op.eta_angle_range)
    expect = np.zeros(pix.n_pixels)
    for r in range(op.N_peaks):
        rho = prof(r, pix.two_theta) * pix.factor
        expect += rho * (w0 * Mrow[j0 * op.N_peaks + r] + w1 * Mrow[j1 * op.N_peaks + r])
    W = _dense_W(cells)
    got = np.zeros(pix.n_pixels)
    ok = cells.pixel_cell >= 0
    got[ok] = (W @ Mrow)[cells.pixel_cell[ok]]
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-6 * expect.max())
    # seg_total: the weights summed over the cells
    np.testing.assert_allclose(cells.seg_total, W.sum(axis=0), rtol=1e-6)


def test_model_adjoint_sensitivity_and_grouping():
    from diffractom import RawDetectorModel, EventList
    ctx, queue = _context()
    op = _operator(ctx, queue)
    pix = _pixels(op)
    _, cells = _model_parts(op, pix, scales=np.array([1.0, 0.6]))
    A = _dense_A(op)
    W = _dense_W(cells)
    F = op.N_Omega * op.My
    rng = np.random.default_rng(2)
    x = (rng.random(op.coeff_shape) * op.support_mask()[None]).astype(np.float32)
    Mfull = (A @ x.ravel().astype(np.float64)).reshape(F, op.N_seg)
    mu_dense = Mfull @ W.T                                         # (F, n_cells)
    counts = rng.poisson(mu_dense / mu_dense.mean() * 0.3).astype(np.float32)
    ev = _full_events(op, cells, counts)
    for subsets, storage in ((1, "gpu"), (3, "gpu"), (1, "host"), (3, "host")):
        model = RawDetectorModel(op, cells, ev, subsets=subsets, storage=storage, chunk_events=777)
        mu = model.to_input_order(model.forward(x))
        np.testing.assert_allclose(mu, mu_dense.ravel(), rtol=2e-4, atol=1e-5 * mu_dense.max())
        r = rng.random(len(ev)).astype(np.float32)
        g = model.adjoint(r[model.order])
        g_ref = (A.T @ (W.T @ r.reshape(F, -1).T.astype(np.float64)).T.ravel()).reshape(op.coeff_shape)
        np.testing.assert_allclose(g, g_ref, rtol=2e-4, atol=1e-5 * np.abs(g_ref).max())
        s = model.sensitivity()
        s_ref = (A.T @ np.tile(W.sum(axis=0), F)).reshape(op.coeff_shape)
        np.testing.assert_allclose(s, s_ref, rtol=2e-4, atol=1e-6 * s_ref.max())
        # subset adjoints and sensitivities add up to the full ones
        if subsets > 1:
            import pyopencl.array as clarray
            acc = np.zeros(op.coeff_shape)
            sacc = np.zeros(op.coeff_shape)
            rg = clarray.to_device(queue, r[model.order]) if storage == "gpu" else r[model.order]
            out = clarray.zeros(queue, op.coeff_shape, np.float32)
            for m in range(subsets):
                model.adjoint_cl(rg, out, subset=m)
                acc += out.get()
                model.sensitivity_cl(out, subset=m)
                sacc += out.get()
            np.testing.assert_allclose(acc, g_ref, rtol=2e-4, atol=1e-5 * np.abs(g_ref).max())
            np.testing.assert_allclose(sacc, s_ref, rtol=2e-4, atol=1e-6 * s_ref.max())
    # cells of two pixels = the sum of the two pixels
    groups = np.arange(pix.n_pixels) // 2
    _, cells2 = _model_parts(op, pix, cell_of_pixel=groups, scales=np.array([1.0, 0.6]))
    model2 = RawDetectorModel(op, cells2, EventList(np.zeros(1, np.int32), np.zeros(1, np.int32), np.ones(1),
                                                    op.N_Omega, op.My, cells2.n_cells))
    W2 = _dense_W(cells2)
    # every modelled pixel's row adds to its group's row
    P = np.zeros((cells2.n_cells, cells.n_cells))
    ok = cells.pixel_cell >= 0
    P[cells2.pixel_cell[ok], cells.pixel_cell[ok]] = 1.0
    np.testing.assert_allclose(W2, P @ W, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(model2.sensitivity(), model.sensitivity(), rtol=2e-4)


def test_deviance_and_solvers_reach_the_minimum():
    """List-mode events (zero cells dropped) give the deviance of all cells; ListModeSPDHG and ListModeMLEM reach
    the minimiser of a float64 MLEM reference on the dense matrix."""
    from diffractom import ListModeMLEM, ListModeSPDHG, RawDetectorModel, EventList
    ctx, queue = _context()
    op = _operator(ctx, queue, K=3, n=6, n_omega=10)
    pix = _pixels(op, n_pix=800)
    _, cells = _model_parts(op, pix)
    A = _dense_A(op)
    W = _dense_W(cells)
    F = op.N_Omega * op.My
    B = np.kron(np.eye(F), W) @ A                                   # (F * n_cells, n_coeff)
    rng = np.random.default_rng(4)
    supp = np.broadcast_to(op.support_mask()[None], op.coeff_shape).ravel() > 0
    x_true = (rng.random(op.coeff_shape) * (rng.random(op.coeff_shape) < 0.5)).ravel() * supp
    mu_t = B @ x_true
    scale = 2.0 / mu_t[mu_t > 0].mean()
    counts = rng.poisson(mu_t * scale).astype(np.float64)
    keep = counts > 0
    idx = np.flatnonzero(keep)
    ev = EventList(idx // cells.n_cells, idx % cells.n_cells, counts[keep], op.N_Omega, op.My, cells.n_cells)
    model = RawDetectorModel(op, cells, ev)
    # deviance of all cells (NumPy, dense) vs the list-mode one
    x_test = (x_true * scale + 0.01 * supp).reshape(op.coeff_shape).astype(np.float32)
    mu = B @ x_test.ravel().astype(np.float64)
    dev_ref = float(np.sum(mu - counts + np.where(counts > 0, counts * np.log(np.where(counts > 0, counts, 1) / np.maximum(mu, 1e-300)), 0)))
    assert abs(model.deviance(x_test) - dev_ref) <= 1e-4 * dev_ref
    # float64 reference minimiser of the full likelihood (L-BFGS-B with bounds on the dense matrix)
    from scipy.optimize import minimize
    s = B.sum(axis=0)
    free = supp & (s > 0)
    Bf = B[:, free]

    def fun(v):
        m_ = Bf @ v
        val = np.sum(m_ - counts + np.where(counts > 0, counts * np.log(np.where(counts > 0, counts, 1) / np.maximum(m_, 1e-300)), 0))
        grad = Bf.T @ (1 - np.where(counts > 0, counts / np.maximum(m_, 1e-300), 0))
        return val, grad
    v0 = np.full(free.sum(), counts.sum() / s.sum())
    res = minimize(fun, v0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * free.sum(),
                   options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-12})
    dev_min = float(res.fun)
    x0 = np.zeros(op.coeff_shape, np.float32)
    sp = ListModeSPDHG(model, subsets=2, rho_factor=0.3, seed=1)
    x = sp.run(x0, 400, diagnostics_interval=400)
    f = sp.iter_stats[-1]["f"]
    assert abs(f - model.deviance(x)) <= 1e-4 * f
    assert f <= dev_min * (1 + 2e-4), (f, dev_min)
    em = ListModeMLEM(model, subsets=1)
    em.run(np.full(op.coeff_shape, counts.sum() / s.sum(), np.float32), 600, diagnostics_interval=600)
    assert em.iter_stats[-1]["f"] <= dev_min * (1 + 2e-4), (em.iter_stats[-1]["f"], dev_min)
    # host storage in small chunks: the same iterates (up to the order of the atomic additions)
    xs = {}
    for storage in ("gpu", "host"):
        mdl = RawDetectorModel(op, cells, ev, storage=storage, chunk_events=500)
        sp2 = ListModeSPDHG(mdl, subsets=3, rho_factor=0.3, seed=2)
        xs[storage] = sp2.run(x0, 30, diagnostics_interval=30)
        xs[storage + "_f"] = sp2.iter_stats[-1]["f"]
    np.testing.assert_allclose(xs["host"], xs["gpu"], rtol=1e-3, atol=1e-4 * np.abs(xs["gpu"]).max())
    assert abs(xs["host_f"] - xs["gpu_f"]) <= 1e-5 * xs["gpu_f"]


def test_pixel_events_to_cells():
    """Pixel events mapped onto cells (to_cells) equal events built on the cells directly; unit sums give the
    summed pattern."""
    from diffractom import DetectorCells, DetectorPixels, EventList, RadialProfiles
    rng = np.random.default_rng(7)
    n_pix = 2000
    rings = np.array([0.1, 0.12])
    tt = rings[rng.integers(0, 2, n_pix)] + rng.uniform(-4e-3, 4e-3, n_pix)
    ext = np.full(n_pix, 1e-3)
    # used pixels = every second pixel of a 1 x 4000 frame
    pix = DetectorPixels(tt, rng.uniform(0, 2 * np.pi, n_pix), rng.uniform(0.5, 1, n_pix), np.arange(n_pix) * 2,
                         (1, 2 * n_pix), ext, ext)
    prof = RadialProfiles.gaussian(rings, 1e-3, cutoff=3.0)
    cells = DetectorCells(pix, prof, 36, np.deg2rad([0, 360]), cell_of_pixel=np.arange(n_pix) // 3)
    bp, bc = EventList.builder(pix, 4, 3), EventList.builder(cells, 4, 3)
    for io in range(4):
        for iy in range(3):
            for _ in range(2):                                    # two raw frames per bin
                idx = rng.choice(2 * n_pix, 300, replace=False)
                v = rng.poisson(2.0, 300).astype(float)
                bp.add_sparse(io, iy, idx, v)
                bc.add_sparse(io, iy, idx, v)
    pev, cev = bp.finish(), bc.finish()
    assert pev.unit == "pixels" and cev.unit == "cells"
    got = pev.to_cells(cells)
    np.testing.assert_array_equal(got.frame, cev.frame)
    np.testing.assert_array_equal(got.cell, cev.cell)
    np.testing.assert_allclose(got.count, cev.count)
    np.testing.assert_allclose(got.exposure, 2.0)
    sums = pev.unit_sums()
    assert sums.shape == (n_pix,) and abs(sums.sum() - pev.total) < 1e-6 * pev.total


def _bg_dense(bgb):
    G = np.zeros((bgb.n_cells, bgb.L))
    for c in range(bgb.n_cells):
        a, b = bgb.ptr[c], bgb.ptr[c + 1]
        G[c, bgb.idx[a:b]] += bgb.w[a:b]
    return G


def test_background_block():
    """Forward / adjoint / sensitivity of the background block against dense matrices, and the solvers with the
    background reach the minimum of the full likelihood of texture + background."""
    from diffractom import ListModeMLEM, ListModeSPDHG, RawDetectorModel, EventList
    from diffractom.detector.background import BackgroundBasis
    from scipy.optimize import minimize
    ctx, queue = _context()
    op = _operator(ctx, queue, K=3, n=6, n_omega=10)
    pix = _pixels(op, n_pix=800)
    _, cells = _model_parts(op, pix)
    F = op.N_Omega * op.My
    knots = np.linspace(pix.two_theta.min(), pix.two_theta.max(), 4)
    groups = (np.arange(F) // op.My) // 2                      # pairs of rotation bins share the background
    bgb = BackgroundBasis(pix, cells.pixel_cell, cells.n_cells, knots, n_eta_nodes=3, frame_groups=groups)
    A = _dense_A(op)
    W = _dense_W(cells)
    G = _bg_dense(bgb)
    B = np.kron(np.eye(F), W) @ A
    Bb = np.zeros((F * cells.n_cells, bgb.n_groups * bgb.L))
    for f in range(F):
        Bb[f * cells.n_cells:(f + 1) * cells.n_cells, groups[f] * bgb.L:(groups[f] + 1) * bgb.L] = G
    rng = np.random.default_rng(6)
    supp = np.broadcast_to(op.support_mask()[None], op.coeff_shape).ravel() > 0
    x_true = (rng.random(op.coeff_shape) * (rng.random(op.coeff_shape) < 0.5)).ravel() * supp
    mu_t = B @ x_true
    b_true = rng.random(bgb.n_groups * bgb.L) * 0.5 * mu_t.mean() / max(G.sum(axis=1).mean(), 1e-30)
    scale = 3.0 / mu_t[mu_t > 0].mean()
    counts = rng.poisson((mu_t + Bb @ b_true) * scale).astype(np.float64)
    keep = counts > 0
    idx = np.flatnonzero(keep)
    ev = EventList(idx // cells.n_cells, idx % cells.n_cells, counts[keep], op.N_Omega, op.My, cells.n_cells)
    for storage in ("gpu", "host"):
        model = RawDetectorModel(op, cells, ev, subsets=2, storage=storage, chunk_events=300, background_basis=bgb)
        xb = rng.random(bgb.shape).astype(np.float32)
        xt = (x_true * scale).reshape(op.coeff_shape).astype(np.float32)
        mu = model.to_input_order(model.forward(xt, beta=xb))
        ref = (B @ xt.ravel() + Bb @ xb.ravel())[keep]
        np.testing.assert_allclose(mu, ref, rtol=2e-4, atol=1e-5 * ref.max())
        r = rng.random(len(ev)).astype(np.float32)
        gx, gb = model.adjoint(r[model.order], background=True)
        rfull = np.zeros(F * cells.n_cells)
        rfull[keep] = r
        np.testing.assert_allclose(gb.ravel(), Bb.T @ rfull, rtol=2e-4, atol=1e-5 * np.abs(Bb.T @ rfull).max())
        np.testing.assert_allclose(model.bg_sens.ravel(), Bb.sum(axis=0), rtol=1e-6)
    # reference minimiser over (x, beta)
    s = B.sum(axis=0)
    free = supp & (s > 0)
    Bf = np.hstack([B[:, free], Bb])

    def fun(v):
        m_ = Bf @ v
        val = np.sum(m_ - counts + np.where(counts > 0, counts * np.log(np.where(counts > 0, counts, 1) / np.maximum(m_, 1e-300)), 0))
        return val, Bf.T @ (1 - np.where(counts > 0, counts / np.maximum(m_, 1e-300), 0))
    v0 = np.full(Bf.shape[1], counts.sum() / Bf.sum())
    res = minimize(fun, v0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * Bf.shape[1],
                   options={"maxiter": 30000, "ftol": 1e-15, "gtol": 1e-12})
    model = RawDetectorModel(op, cells, ev, background_basis=bgb)
    sp = ListModeSPDHG(model, subsets=2, rho_factor=0.3, seed=1)
    x = sp.run(np.zeros(op.coeff_shape, np.float32), 600, diagnostics_interval=600)
    f = sp.iter_stats[-1]["f"]
    assert abs(f - model.deviance(x, beta=sp.beta)) <= 1e-4 * f
    assert f <= res.fun * (1 + 1e-3), (f, res.fun)
    em = ListModeMLEM(model, subsets=1)
    em.run(np.full(op.coeff_shape, counts.sum() / s.sum(), np.float32), 1500, diagnostics_interval=1500)
    assert em.iter_stats[-1]["f"] <= res.fun * (1 + 1e-3), (em.iter_stats[-1]["f"], res.fun)


def test_weight_derivatives():
    """Cell-weight derivatives (pseudo-Voigt parameters, ring shifts, geometry through two-theta and eta) equal
    central differences of rebuilt cells."""
    from diffractom import DetectorCells, DetectorPixels
    from diffractom.detector.profiles import PseudoVoigtProfiles
    rng = np.random.default_rng(3)
    n_pix = 4000
    rings = np.array([0.10, 0.14])
    tt = rings[rng.integers(0, 2, n_pix)] + rng.uniform(-3e-3, 3e-3, n_pix)
    eta = rng.uniform(0, 2 * np.pi, n_pix)
    ext = np.full(n_pix, 1e-3)
    pix = DetectorPixels(tt, eta, rng.uniform(0.5, 1, n_pix), np.arange(n_pix), (1, n_pix), ext, ext)
    prof = PseudoVoigtProfiles(rings, U=2e-7, V=-1e-8, W=3e-7, X=2e-4, Y=1e-4, cut=6.0)
    cof = np.arange(n_pix) // 2

    def build(pr=prof, px=pix):
        return DetectorCells(px, pr, 72, np.deg2rad([0, 360]), ring_scales=np.array([1.0, 0.7]),
                             cell_of_pixel=cof, fixed_cells=True)
    base = build()

    def dense(c):
        W = np.zeros((c.n_cells, c.n_seg))
        for k in range(c.n_cells):
            a, b = c.ptr[k], c.ptr[k + 1]
            W[k, c.seg[a:b]] += c.w64[a:b]
        return W

    def csr_to_dense(c, vals):
        W = np.zeros((c.n_cells, c.n_seg))
        for k in range(c.n_cells):
            a, b = c.ptr[k], c.ptr[k + 1]
            W[k, c.seg[a:b]] += vals[a:b]
        return W
    # compare cells away from the profile cut (pixels entering or leaving a window are not in the derivative)
    ring_of = np.argmin(np.abs(tt[:, None] - rings[None, :]), axis=1)
    off = np.abs(tt - rings[ring_of]) / np.array([prof.cut * prof.fwhm(r) for r in range(2)])[ring_of]
    inner = np.zeros(base.n_cells, bool)
    inner[cof] = True
    np.logical_and.at(inner, cof, off < 0.8)
    for name, h in (("U", 2e-9), ("W", 3e-9), ("X", 2e-6)):
        d = csr_to_dense(base, base.profile_derivative(name))[inner]
        fd = (dense(build(prof.copy(**{name: prof.p[name] + h}))) - dense(build(prof.copy(**{name: prof.p[name] - h}))))[inner] / (2 * h)
        assert np.abs(d - fd).max() <= 2e-3 * np.abs(fd).max(), name
    # geometry: shift every pixel's two-theta by a * (1 + eta) and eta by b
    a = 1e-6
    dtt = 1.0 + 0.1 * np.cos(eta)
    deta = 0.3 + 0.0 * eta
    d = csr_to_dense(base, base.geometry_derivative(dtt, deta))

    def moved(sign):
        return DetectorPixels(tt + sign * a * dtt, eta + sign * a * deta, pix.factor, pix.index, pix.shape, ext, ext)
    fd = (dense(build(px=moved(1))) - dense(build(px=moved(-1)))) / (2 * a)
    d, fd = d[inner], fd[inner]
    assert np.abs(d - fd).max() <= 2e-3 * np.abs(fd).max()


def test_detector_geometry_matches_pyfai():
    pyFAI = pytest.importorskip("pyFAI")
    from diffractom.detector.geometry import DetectorGeometry
    from pyFAI.detectors import Detector
    det = Detector(pixel1=172e-6, pixel2=172e-6, max_shape=(300, 260))
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    ai = AzimuthalIntegrator(dist=0.25, poni1=0.021, poni2=0.019, rot1=0.004, rot2=-0.003, rot3=0.02, detector=det,
                             wavelength=0.35e-10)
    g = DetectorGeometry.from_pyfai(ai, (300, 260))
    idx = np.arange(300 * 260)
    tt, eta = g.angles(idx)
    np.testing.assert_allclose(tt, ai.center_array((300, 260), unit="2th_rad").ravel(), atol=1e-12)
    chi = ai.center_array((300, 260), unit="chi_rad").ravel()
    np.testing.assert_allclose(np.angle(np.exp(1j * (eta - chi - np.pi))), 0, atol=1e-12)
    # parallax and distortion move the angles the expected way
    t2, _ = g.copy(parallax=50e-6).angles(idx)
    assert np.all(t2 <= tt + 1e-15)


def test_dense_counts_equal_event_list():
    """DenseRawDetectorModel (every cell of the frames used, counts in host memory) == RawDetectorModel with the same
    cells as events: forward, adjoint (incl. background), OSEM iterates; held-out frames excluded by `frames`."""
    from diffractom import ListModeMLEM, RawDetectorModel, EventList
    from diffractom.detector.raw_model import DenseRawDetectorModel
    from diffractom.detector.background import BackgroundBasis
    ctx, queue = _context()
    op = _operator(ctx, queue, K=3, n=6, n_omega=10)
    pix = _pixels(op, n_pix=600)
    _, cells = _model_parts(op, pix)
    F = op.N_Omega * op.My
    rng = np.random.default_rng(8)
    counts = rng.poisson(2.0, (F, cells.n_cells)).astype(np.uint16)
    frames = np.flatnonzero(np.arange(F) % 3 != 1)
    knots = np.linspace(pix.two_theta.min(), pix.two_theta.max(), 3)
    bgb = BackgroundBasis(pix, cells.pixel_cell, cells.n_cells, knots, n_eta_nodes=2, n_frames=F,
                          frame_groups=np.arange(F) % op.My)
    expo = np.ones(F, np.float32)
    expo[np.arange(F) % 3 == 1] = 0.0
    fr = np.repeat(frames, cells.n_cells)
    ce = np.tile(np.arange(cells.n_cells), len(frames))
    ct = counts[frames].ravel().astype(np.float32)
    keep = ct > 0
    ev = EventList(fr[keep], ce[keep], ct[keep], op.N_Omega, op.My, cells.n_cells, expo.reshape(op.N_Omega, op.My))
    xs = {}
    for name in ("list", "dense"):
        if name == "list":
            m = RawDetectorModel(op, cells, ev, subsets=2, background_basis=bgb)
        else:
            m = DenseRawDetectorModel(op, cells, counts, exposure=expo, frames=frames, subsets=2, chunk_frames=4,
                                      background_basis=bgb)
        em = ListModeMLEM(m, subsets=None, subset_sensitivity="exact")
        x0 = np.full(op.coeff_shape, 0.5, np.float32)
        xs[name] = (em.run(x0, 20, diagnostics_interval=20), em.beta, em.iter_stats[-1]["f"])
    np.testing.assert_allclose(xs["dense"][0], xs["list"][0], rtol=1e-3, atol=1e-5 * np.abs(xs["list"][0]).max())
    np.testing.assert_allclose(xs["dense"][1], xs["list"][1], rtol=1e-3, atol=1e-5 * np.abs(xs["list"][1]).max())
    assert abs(xs["dense"][2] - xs["list"][2]) <= 1e-4 * abs(xs["list"][2])
