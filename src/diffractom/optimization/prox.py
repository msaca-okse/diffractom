# prox_operators.py
import pyopencl as cl
import pyopencl.array as clarray
import numpy as np

from .launch import elementwise, pixel_orientation


# Element-wise kernels use 64-bit indices in a grid-stride loop (see launch.py).
PROX_KERNELS = r"""
__kernel void prox_nonneg(__global float *x, const ulong n) {
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0))
        if (x[i] < 0.0f) x[i] = 0.0f;
}

__kernel void prox_l1(__global float *x, float lambda, const ulong n) {
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        float v = x[i];
        float a = fabs(v) - lambda;
        x[i] = (a > 0.0f) ? copysign(a, v) : 0.0f;
    }
}

__kernel void prox_nonneg_l1(__global float *x, float lambda, const ulong n) {
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        float v = x[i] - lambda;
        x[i] = (v > 0.0f) ? v : 0.0f;
    }
}

// x (Nx, Ny, K) Fortran order: zero every channel of the pixels outside the support.
// Range (npix, <= 64): pixels on axis 0, orientations strided on axis 1.
__kernel void apply_support(__global float *x, __global const uchar *mask, const int npix, const int K) {
    const int p = get_global_id(0);
    if (p >= npix || mask[p]) return;
    for (int k = get_global_id(1); k < K; k += get_global_size(1))
        x[(size_t)k * npix + p] = 0.0f;
}
"""


class ProxKernels:
    """Compiled proximal operator OpenCL kernels (nonneg, l1, nonneg_l1)."""
    def __init__(self, ctx: cl.Context):
        self.prg = cl.Program(ctx, PROX_KERNELS).build()

        # Cache kernels ONCE
        self.k_prox_nonneg     = cl.Kernel(self.prg, "prox_nonneg")
        self.k_prox_l1         = cl.Kernel(self.prg, "prox_l1")
        self.k_prox_nonneg_l1  = cl.Kernel(self.prg, "prox_nonneg_l1")
        self.k_apply_support   = cl.Kernel(self.prg, "apply_support")


def prox_nonneg(queue, kernels: ProxKernels, x_gpu):
    """Non-negativity projection in place."""
    gws, n = elementwise(x_gpu.size)
    kernels.k_prox_nonneg(queue, gws, None, x_gpu.data, n)
    return x_gpu


def prox_l1(queue, kernels: ProxKernels, x_gpu, lam, tau):
    """Soft-thresholding (L1 proximal) in place."""
    gws, n = elementwise(x_gpu.size)
    kernels.k_prox_l1(queue, gws, None, x_gpu.data, np.float32(lam * tau), n)
    return x_gpu


def prox_nonneg_l1(queue, kernels: ProxKernels, x_gpu, lam, tau):
    """Non-negative soft-thresholding in place."""
    gws, n = elementwise(x_gpu.size)
    kernels.k_prox_nonneg_l1(queue, gws, None, x_gpu.data, np.float32(lam * tau), n)
    return x_gpu


def support_mask_to_gpu(queue, operator, support):
    """Resolve the ``support`` option of the FISTA solvers to a uint8 GPU mask (or None).

    support : "fov" (the operator's field-of-view disk), None/False (no support
    constraint), or an (Nx, Ny) boolean array.
    """
    if support is None or support is False:
        return None
    if isinstance(support, str):
        if support != "fov":
            raise ValueError(f"support must be 'fov', None or an (Nx, Ny) array, not {support!r}")
        if not hasattr(operator, "support_mask"):
            raise ValueError("the operator has no support_mask(); pass support=None or an array")
        mask = operator.support_mask()
    else:
        mask = np.asarray(support)
    if mask.shape != (operator.Nx, operator.Ny):
        raise ValueError(f"support mask shape {mask.shape} != (Nx, Ny) = {(operator.Nx, operator.Ny)}")
    return clarray.to_device(queue, np.asfortranarray(mask.astype(np.uint8)).ravel(order="F"))


def apply_support(queue, kernels: ProxKernels, x_gpu, mask_gpu):
    """Zero x (Nx, Ny, K, Fortran order) outside the support in place (projection onto the support)."""
    npix = int(mask_gpu.size)
    K = int(x_gpu.size) // npix
    kernels.k_apply_support(queue, pixel_orientation(npix, K), None, x_gpu.data, mask_gpu.data,
                            np.int32(npix), np.int32(K))
    return x_gpu
