"""The operator's adjoint is the transpose of its forward map, <x, A^T y> = <A x, y>, for every projector and any
angle range (also a decreasing one)."""
import numpy as np
import pytest


def _context():
    cl = pytest.importorskip("pyopencl")
    try:
        ctx = cl.create_some_context(interactive=False)
    except Exception:
        pytest.skip("no OpenCL device")
    return ctx, cl.CommandQueue(ctx)


@pytest.mark.parametrize("projector", ["native", "fft", "gratopy"])
@pytest.mark.parametrize("angle_range", [(0, 180), (0, 360), (0, 90), (180, 0)])
def test_adjoint_is_transpose(projector, angle_range):
    import pyopencl.array as clarray
    from diffractom import Grid, Material, SinglePhaseForwardOperator
    if projector == "fft":
        pytest.importorskip("pyvkfft")
    if projector == "gratopy":
        pytest.importorskip("gratopy")
    ctx, queue = _context()
    mat = Material.from_lattice_parameters(a=4.05, b=4.05, c=4.05, alpha=90, beta=90, gamma=90, symmetry_group="cubic",
                                           wavelength_A=0.35, hkl_list=np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0]]))
    cfg = {"Nx": 16, "Ny": 16, "My": 17, "N_Omega": 30, "angle_range": list(angle_range), "cor_offset": 0.5,
           "N_eta": 36, "eta_angle_range": [0, 360], "energy": 12.398 / 0.35, "k_direction_0": [0, 0, 1],
           "p_direction_0": [1, 0, 0], "detector_direction_origin": [0, -1, 0],
           "detector_direction_positive_90": [0, 0, -1]}
    np.random.seed(0)
    grid = Grid.from_random_fundamental_zone(12, "cubic", np.deg2rad(4.0))
    op = SinglePhaseForwardOperator(cfg=cfg, material=mat, grid=grid, max_gb=0.2, normalized=True,
                                    projector=projector, ctx=ctx, queue=queue)
    rng = np.random.default_rng(1)
    x = (rng.random(op.coeff_shape) * op.support_mask()[None]).astype(np.float32)
    y = rng.random(op.data_shape).astype(np.float32)
    Ax = clarray.empty(queue, op.data_shape, np.float32)
    ATy = clarray.empty(queue, op.coeff_shape, np.float32)
    op.direct_cl(clarray.to_device(queue, x), Ax)
    op.adjoint_cl(clarray.to_device(queue, y), ATy)
    lhs = float(np.dot(x.ravel().astype(np.float64), ATy.get().ravel()))
    rhs = float(np.dot(Ax.get().ravel().astype(np.float64), y.ravel()))
    assert rhs > 0
    tol = 2e-3 if projector == "fft" else 1e-5  # the FFT projector's adjoint is exact to its interpolation
    assert abs(lhs / rhs - 1) < tol
