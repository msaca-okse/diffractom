"""cfg["voxel_size"]: reconstruction pixels smaller than the translation step. A fine image made by repeating a coarse
one projects like the coarse image (pixel footprints add up), the adjoint stays the transpose, and the local operator
on the fine grid equals the global one with masked coefficients."""
import numpy as np
import pytest


def _context():
    cl = pytest.importorskip("pyopencl")
    try:
        ctx = cl.create_some_context(interactive=False)
    except Exception:
        pytest.skip("no OpenCL device")
    return ctx, cl.CommandQueue(ctx)


def _setup(n=16):
    from diffractom import Grid, Material
    mat = Material.from_lattice_parameters(a=4.05, b=4.05, c=4.05, alpha=90, beta=90, gamma=90, symmetry_group="cubic",
                                           wavelength_A=0.35, hkl_list=np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0]]))
    cfg = {"Nx": n, "Ny": n, "My": n + 1, "N_Omega": 30, "angle_range": [0, 180], "cor_offset": 0.5,
           "N_eta": 36, "eta_angle_range": [0, 360], "energy": 12.398 / 0.35, "k_direction_0": [0, 0, 1],
           "p_direction_0": [1, 0, 0], "detector_direction_origin": [0, -1, 0],
           "detector_direction_positive_90": [0, 0, -1]}
    np.random.seed(0)
    grid = Grid.from_random_fundamental_zone(10, "cubic", np.deg2rad(4.0))
    return mat, cfg, grid


@pytest.mark.parametrize("projector", ["native", "fft"])
@pytest.mark.parametrize("F", [2, 3])
def test_voxel_size(projector, F):
    from diffractom import SinglePhaseForwardOperator
    if projector == "fft":
        pytest.importorskip("pyvkfft")
    ctx, q = _context()
    mat, cfg, grid = _setup()
    kw = dict(normalized=True, projector=projector, ctx=ctx, queue=q, pf_mode="sparse")
    op1 = SinglePhaseForwardOperator(cfg=cfg, material=mat, grid=grid, max_gb=0.2, **kw)
    cf = dict(cfg, Nx=cfg["Nx"] * F, Ny=cfg["Ny"] * F, voxel_size=1.0 / F)
    opf = SinglePhaseForwardOperator(cfg=cf, material=mat, grid=grid, max_gb=0.2, **kw)
    rng = np.random.default_rng(1)
    c1 = (rng.random(op1.coeff_shape) * op1.support_mask()[None]).astype(np.float32)
    cfine = np.repeat(np.repeat(c1, F, 1), F, 2)
    d1, df = op1.direct(c1), opf.direct(cfine)
    tol = 1e-2 if projector == "native" else 2e-2
    assert np.linalg.norm(df - d1) < tol * np.linalg.norm(d1)
    y = rng.random(opf.data_shape).astype(np.float32)
    a = opf.adjoint(y)
    lhs, rhs = float(np.sum(a.astype(np.float64) * cfine)), float(np.sum(df.astype(np.float64) * y))
    assert abs(lhs / rhs - 1) < (1e-5 if projector == "native" else 2e-3)
    m1, mf = op1.support_mask(), opf.support_mask()
    assert abs(int(mf.sum()) - F * F * int(m1.sum())) <= 4 * F * cfg["Nx"]      # the same disc, finer pixels


def test_voxel_size_local():
    from diffractom import LocalSinglePhaseForwardOperator, LocalSupport, SinglePhaseForwardOperator
    ctx, q = _context()
    mat, cfg, grid = _setup()
    cf = dict(cfg, Nx=32, Ny=32, voxel_size=0.5)
    opf = SinglePhaseForwardOperator(cfg=cf, material=mat, grid=grid, max_gb=0.2, normalized=True, ctx=ctx, queue=q,
                                     pf_mode="sparse")
    rng = np.random.default_rng(2)
    mask = np.zeros(opf.coeff_shape, bool)
    for k in range(mask.shape[0]):
        y0, x0 = rng.integers(0, 16, 2)
        mask[k, y0:y0 + 12, x0:x0 + 10] = True
    mask &= opf.support_mask()[None]
    lop = LocalSinglePhaseForwardOperator(cf, mat, grid, 0.2, support=LocalSupport.from_mask(mask), normalized=True,
                                          ctx=ctx, queue=q)
    c = (rng.random(opf.coeff_shape) * mask).astype(np.float32)
    dl, dg = lop.direct(lop.from_dense(c)), opf.direct(c)
    assert np.linalg.norm(dl - dg) < 1e-5 * np.linalg.norm(dg)
    y = rng.random(opf.data_shape).astype(np.float32)
    assert np.allclose(lop.to_dense(lop.adjoint(y)), opf.adjoint(y) * mask, rtol=1e-4, atol=1e-6 * float(np.abs(opf.adjoint(y)).max()))
