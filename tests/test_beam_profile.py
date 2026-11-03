"""Beam profile across the translation direction: the profile's maths, the native projector's tabulated kernel against
a float64 reference (tiled == plain, adjoint), BeamProfile(1) == the trapezoid, and the operator (subsets, FFT)."""
import numpy as np
import pytest

from diffractom import BeamProfile


def _context():
    cl = pytest.importorskip("pyopencl")
    try:
        ctx = cl.create_some_context(interactive=False)
    except Exception:
        pytest.skip("no OpenCL device")
    return ctx, cl.CommandQueue(ctx)


def test_profile_maths():
    x = np.linspace(-6, 6, 24001)
    for b in (BeamProfile(1.42, 0.2), BeamProfile(2.0, 0.0), BeamProfile(0.7, 0.5)):
        assert abs(np.trapezoid(b(x), x) - 1) < 1e-3          # (sigma = 0: a jump on the grid)
        cdf = np.concatenate([[0], np.cumsum(0.5 * (b(x)[1:] + b(x)[:-1]) * np.diff(x))])
        assert np.abs(cdf - b.cdf(x)).max() < 1e-3
        om = np.array([0.3, 1.0, 2.5])
        ft = np.trapezoid(b(x)[None] * np.exp(-1j * om[:, None] * x[None]), x, axis=1)
        assert np.abs(ft - b.transform(om)).max() < 1e-3
    d = np.linspace(-2, 2, 401)
    for A in (0.7, 1.0, 1.41):                 # BeamProfile(1): the pixel footprint averaged over one bin
        trap = np.clip((1 + A) / 2 - np.abs(d), 0, min(1, A)) / A
        assert np.abs(BeamProfile(1.0).footprint(d, A) - trap).max() < 1e-12
    b = BeamProfile(1.42, 0.2)
    t = BeamProfile.tabulated(x, 3 * b(x))     # normalised to unit area
    assert np.abs(t.cdf(x) - b.cdf(x)).max() < 1e-6
    assert np.abs(t.transform(np.array([0.5, 2.0])) - b.transform(np.array([0.5, 2.0]))).max() < 1e-6


def _reference(img_k, Nx, Ny, Ns, angles, beam, shift):
    X, Y = np.cos(angles - np.pi / 2), np.sin(angles - np.pi / 2)
    T0 = (Ns - 1) / 2 - X * (Nx - 1) / 2 - Y * (Ny - 1) / 2 - shift
    yy, xx = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing="ij")
    out = np.zeros((len(angles), Ns, img_k.shape[1]))
    for a in range(len(angles)):
        t = (X[a] * xx + Y[a] * yy + T0[a]).ravel()
        out[a] = beam.footprint(t[None, :] - np.arange(Ns)[:, None], max(abs(X[a]), abs(Y[a]))) @ img_k
    return out


def test_projector_kernel():
    ctx, q = _context()
    import pyopencl.array as clarray
    from diffractom.operators.parallel_radon import ParallelRadon
    rng = np.random.default_rng(1)
    Nx, Ny, Ns, R, K = 19, 17, 27, 13, 8
    ang = np.sort(rng.uniform(0, 2 * np.pi, R))
    x = np.linspace(-2.5, 2.0, 91)
    for beam in (BeamProfile(1.42, 0.2), BeamProfile(3.0, 0.5),
                 BeamProfile.tabulated(x, np.exp(-0.5 * ((x + 0.2) / 0.6) ** 2) * (1 + 0.3 * np.tanh(x)) + 0.01)):
        img = rng.random((Nx * Ny, K))
        sino = rng.random((R, Ns, K))
        ref = _reference(img, Nx, Ny, Ns, ang, beam, 0.37)
        outs = []
        for tiled in (False, True):
            rad = ParallelRadon(q, (Nx, Ny), ang, Ns, float(Nx), float(Ns), detector_shift=0.37, tiled=tiled,
                                beam_profile=beam)
            ig = clarray.to_device(q, img.astype(np.float32).ravel())
            sg = clarray.zeros(q, (R * Ns * K,), np.float32)
            rad.forward(ig, sg, K)
            sb = clarray.to_device(q, sino.astype(np.float32).ravel())
            ib = clarray.zeros(q, (Nx * Ny * K,), np.float32)
            rad.backward(sb, ib, K)
            f, b = sg.get().reshape(R, Ns, K), ib.get().reshape(Nx * Ny, K)
            outs.append((f, b))
            assert np.abs(f - ref).max() < 2e-5 * np.abs(ref).max()
            lhs = np.sum(f.astype(np.float64) * sino)
            rhs = np.sum(b.astype(np.float64) * img) * float(rad.scale) / rad.angle_weights[0]
            assert abs(lhs - rhs) < 1e-5 * abs(lhs)
        assert np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])


def test_operator_beam():
    ctx, q = _context()
    import pyopencl.array as clarray
    from diffractom import Grid, Material, SinglePhaseForwardOperator
    mat = Material.from_lattice_parameters(a=4.05, b=4.05, c=4.05, alpha=90, beta=90, gamma=90, symmetry_group="cubic",
                                           wavelength_A=0.35, hkl_list=np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0]]))
    cfg = {"Nx": 20, "Ny": 20, "My": 23, "N_Omega": 24, "angle_range": [0, 360], "cor_offset": 0.5, "N_eta": 16,
           "eta_angle_range": [0, 360], "energy": 12.398 / 0.35, "k_direction_0": [0, 0, 1], "p_direction_0": [1, 0, 0],
           "detector_direction_origin": [0, -1, 0], "detector_direction_positive_90": [0, 0, -1]}
    np.random.seed(5)
    grid = Grid.from_random_fundamental_zone(20, "cubic", np.deg2rad(5.0))
    make = lambda **kw: SinglePhaseForwardOperator(cfg=cfg, material=mat, grid=grid, max_gb=0.05, normalized=True,
                                                   pf_mode="sparse", ctx=ctx, queue=q, verbose=False, **kw)
    beam = BeamProfile(1.42, 0.2)
    op0, op = make(projector="native"), make(projector="native", beam_profile=beam)
    rng = np.random.default_rng(0)
    c = (rng.random(op.coeff_shape) * op.support_mask()[None]).astype(np.float32)
    d0, d = op0.direct(c), op.direct(c)
    assert 1e-3 < np.linalg.norm(d - d0) / np.linalg.norm(d0) < 0.1          # the beam changes the data a little
    assert np.linalg.norm(make(projector="native", beam_profile=BeamProfile(1.0)).direct(c) - d0) < 1e-4 * np.linalg.norm(d0)
    op.set_omega_subsets(3)
    full = d.reshape(op.N_Omega, -1)
    cg = clarray.to_device(q, c)
    for m, s in enumerate(op.omega_subsets):
        dd = clarray.zeros(q, op.data_shape, np.float32)
        op.direct_cl(cg, dd, subset=m)
        np.testing.assert_allclose(dd.get().reshape(op.N_Omega, -1)[s], full[s], rtol=1e-5, atol=1e-6 * np.abs(full).max())
    from diffractom.operators import fourier_radon
    if fourier_radon.available():
        df = make(projector="fft", beam_profile=beam).direct(c)
        assert np.linalg.norm(df - d) < 0.02 * np.linalg.norm(d)
    with pytest.raises(ValueError):
        make(projector="native", radon_kernel="hat", beam_profile=beam)
