"""window_parallax: integration windows cut the reflections' radial profiles and a grain's depth along the beam shifts
its rings (sample parallax). With windows that hold everything the operator is bitwise the plain one; otherwise every
rotation angle equals a plain operator whose reflection weights are the window fractions of that angle (computed here
in float64)."""
import numpy as np
import pytest
from scipy.special import erf

from test_reflection_two_theta import CFG, WINDOWS, _context, _material, _grid


def _op(ctx, queue, pf_mode="sparse", rings=WINDOWS, **kw):
    from diffractom import SinglePhaseForwardOperator
    return SinglePhaseForwardOperator(cfg=dict(CFG), material=_material(), grid=_grid(), max_gb=1e-4, normalized=True, pf_mode=pf_mode,
                                      projector="native", ctx=ctx, queue=queue, rings=rings, reflection_two_theta=True, **kw)


def _point_coeffs(op, col, row):
    c = np.zeros(op.coeff_shape, np.float32)
    c[0, row, col] = 1.0
    c[3, row, col] = 0.5
    return c


@pytest.mark.parametrize("pf_mode", ["sparse", "generated"])
def test_wide_windows_are_the_plain_operator(pf_mode):
    ctx, queue = _context()
    plain = _op(ctx, queue, pf_mode)
    wp = _op(ctx, queue, pf_mode, window_parallax=dict(windows=[[0.0, np.pi]] * len(WINDOWS), sigma=1e-3, distance=1e12))
    c = _point_coeffs(plain, 5, 7)
    assert np.array_equal(plain.direct(c), wp.direct(c))


@pytest.mark.parametrize("pf_mode,lorentz", [("sparse", 0.0), ("generated", 0.0), ("sparse", 0.5)])
def test_every_angle_is_a_plain_operator_with_its_window_fractions(pf_mode, lorentz):
    ctx, queue = _context()
    from diffractom.differentiable.absorption import image_direction
    plain = _op(ctx, queue, "sparse")
    m = np.asarray(_material().reflections["multiplicity"], np.float64)
    h = np.asarray(_material().h_vecs, np.float64)
    tth_f = 2 * np.arcsin(np.linalg.norm(h, axis=1) / (4 * np.pi) * 0.35)
    # windows cutting the reflections: each one's edge 0.15 deg from a reflection, profile sigma 0.1 deg
    win = []
    for ix, _ in WINDOWS:
        t = tth_f[ix]
        win.append([t.min() - np.radians(0.15), t.max() + np.radians(0.05)])
    win = np.array(win); sig = np.radians(0.1); D = 40.0
    col, row = 3, 9
    K = plain.K
    pos = np.tile([[col, row]], (K, 1)).astype(np.float64)
    ones = [(ix, [1.0] * len(ix)) for ix, _ in WINDOWS]                     # full intensities: the fractions replace the overlaps
    gam = np.radians(0.07)
    wp = _op(ctx, queue, pf_mode, rings=ones, window_parallax=dict(windows=win, sigma=sig, distance=D, positions=pos, covariances=np.zeros((K, 3)),
                                                              lorentz=lorentz, gamma=gam))
    c = _point_coeffs(plain, col, row)
    d = wp.direct(c).reshape(plain.N_Omega, plain.My, plain.N_eta, plain.N_peaks)
    px = plain.image_width / max(plain.Nx, plain.Ny)
    rel = (np.array([col, row]) - [(plain.Nx - 1) / 2, (plain.Ny - 1) / 2]) * px
    b = image_direction(plain, np.asarray(CFG["p_direction_0"], np.float64)[:2], np.asarray(plain.angles))
    worst = 0.0
    for r in range(plain.N_Omega):
        x = rel @ b[r]
        rings_r, corr = [], []
        for p, (ix, _) in enumerate(WINDOWS):
            t = tth_f[ix]; sh = t - (x / D) * np.sin(t) * np.cos(t)
            fr = 0.5 * (erf((win[p, 1] - sh) / (np.sqrt(2) * sig)) - erf((win[p, 0] - sh) / (np.sqrt(2) * sig)))
            fr = (1 - lorentz) * fr + lorentz * (np.arctan((win[p, 1] - sh) / gam) - np.arctan((win[p, 0] - sh) / gam)) / np.pi
            fr = np.maximum(fr, 1e-12)
            rings_r.append((list(ix), fr.tolist()))
            corr.append((m[ix] * fr).sum() / m[ix].sum())        # normalized: shares are m w / sum(m w)
        ref = _op(ctx, queue, "sparse", rings=rings_r).direct(c).reshape(d.shape)[r] * np.asarray(corr)[None, None, :]
        got = d[r]
        scale = max(np.abs(ref).max(), 1e-12)
        worst = max(worst, np.abs(got - ref).max() / scale)
    assert worst < 2e-5, worst


def test_local_operator_default_positions_are_the_support_centroids():
    ctx, queue = _context()
    from diffractom import LocalSinglePhaseForwardOperator, LocalSupport
    from diffractom.operators.local_forward_operator import spatial_moments
    g = _grid()
    K = len(g.leaf_arrays()[0])
    rng = np.random.default_rng(3)
    k = np.repeat(np.arange(K), 6); iy = rng.integers(2, 10, len(k)); ix = rng.integers(2, 12, len(k))
    key = np.unique(np.stack([k, iy, ix], 1), axis=0)
    sup = LocalSupport(key[:, 0], key[:, 1], key[:, 2], K, (CFG["Ny"], CFG["Nx"]))
    win = [[0.0, np.pi]] * len(WINDOWS)
    kw = dict(cfg=dict(CFG), material=_material(), grid=g, max_gb=1e-4, support=sup, normalized=True, ctx=ctx, queue=queue,
              rings=WINDOWS, reflection_two_theta=True, pf_mode="sparse", restrict_to_fov=False)
    op_d = LocalSinglePhaseForwardOperator(**kw, window_parallax=dict(windows=win, sigma=0.01, distance=30.0))
    pos, cov = spatial_moments(sup.k, sup.iy, sup.ix, None, K)
    op_e = LocalSinglePhaseForwardOperator(**kw, window_parallax=dict(windows=win, sigma=0.01, distance=30.0, positions=pos, covariances=cov))
    x = rng.random(op_d.n_coef).astype(np.float32)
    assert np.array_equal(op_d.direct(x), op_e.direct(x))


def _cut_windows():
    h = np.asarray(_material().h_vecs, np.float64)
    tth_f = 2 * np.arcsin(np.linalg.norm(h, axis=1) / (4 * np.pi) * 0.35)
    return np.array([[tth_f[ix].min() - np.radians(0.15), tth_f[ix].max() + np.radians(0.05)] for ix, _ in WINDOWS])


@pytest.mark.parametrize("pf_mode", ["sparse", "generated"])
def test_gradients(pf_mode):
    """PFGradient carries the window fractions: wide windows give the plain gradients bitwise; with cutting windows and
    parallax, the rotation and width gradients equal central differences of <r, A c>."""
    from diffractom.differentiable import PFGradient, rotate_orientations, set_pf_parameters
    ctx, queue = _context()
    ones = [(ix, [1.0] * len(ix)) for ix, _ in WINDOWS]
    plain = _op(ctx, queue, pf_mode, rings=ones)
    wide = _op(ctx, queue, pf_mode, rings=ones, window_parallax=dict(windows=[[0.0, np.pi]] * len(WINDOWS), sigma=1e-3, distance=1e12))
    rng = np.random.default_rng(2)
    c = (rng.random(plain.coeff_shape) * plain.support_mask()[None]).astype(np.float32)
    r = rng.normal(size=plain.data_shape).astype(np.float32)
    g0, g1 = PFGradient(plain)(c, r, groups=("rot", "sigma")), PFGradient(wide)(c, r, groups=("rot", "sigma"))
    for k in ("rot", "sigma"):
        assert np.array_equal(g0[k], g1[k]), k
    K = plain.K
    pos = rng.uniform(1, 11, (K, 2)); cov = np.tile([[4.0, 1.0, 3.0]], (K, 1))
    # a fixed quadrature: the automatic rule follows the smallest kernel width, which a width perturbation may change
    op = _op(ctx, queue, pf_mode, rings=ones, quadrature="midpoint",
             window_parallax=dict(windows=_cut_windows(), sigma=np.radians(0.1), distance=40.0, positions=pos, covariances=cov))
    g = PFGradient(op)(c, r, groups=("rot", "sigma"))
    R0, s0 = op.grid_rot_cpu.copy(), op.sigma_cpu.astype(np.float64).copy()

    def loss(rot=None, sig=None):
        set_pf_parameters(op, rotations=R0 if rot is None else rotate_orientations(R0, rot), sigma=s0 if sig is None else sig)
        return float(np.sum(op.direct(c).astype(np.float64) * r))

    for k_, ax in ((0, 0), (3, 2), (7, 1)):
        h = 2e-3; dv = np.zeros((K, 3)); dv[k_, ax] = h
        fd = (loss(rot=dv) - loss(rot=-dv)) / (2 * h)
        assert abs(g["rot"][k_, ax] - fd) <= 2e-2 * max(abs(fd), 1e-2 * np.abs(g["rot"]).max()), (k_, ax, g["rot"][k_, ax], fd)
        ds = np.zeros(K); ds[k_] = 1e-3 * s0[k_]
        fd = (loss(sig=s0 + ds) - loss(sig=s0 - ds)) / (2 * ds[k_])
        assert abs(g["sigma"][k_] - fd) <= 2e-2 * max(abs(fd), 1e-2 * np.abs(g["sigma"]).max()), (k_, g["sigma"][k_], fd)
