# prox_tv.py
import pyopencl as cl
import pyopencl.array as clarray
import numpy as np

from .launch import elementwise, pixel_orientation


TV_KERNELS = r"""
// Arrays (K, Ny, Nx), C order. tv_grad and tv_div run on the range (Nx*Ny, <= 64): pixels on
// axis 0, orientations strided on axis 1; the other kernels are element-wise, with 64-bit indices in a
// grid-stride loop (see launch.py).

// ============================================================
// Forward gradient (x,y only), K independent
// ============================================================
__kernel void tv_grad(
    __global const float *u,
    __global float *ux,
    __global float *uy,
    const int Nx,
    const int Ny,
    const int K
){
    const int p = get_global_id(0);
    if (p >= Nx * Ny) return;
    const int i = p % Nx;
    const int j = p / Nx;
    const size_t npix = (size_t)Nx * Ny;

    for (int k = get_global_id(1); k < K; k += get_global_size(1)) {
        const size_t idx = p + npix * k;
        // forward differences (Neumann BC)
        ux[idx] = (i < Nx-1) ? (u[idx + 1]   - u[idx]) : 0.0f;
        uy[idx] = (j < Ny-1) ? (u[idx + Nx]  - u[idx]) : 0.0f;
    }
}


// ============================================================
// Divergence (matches CPU reference EXACTLY)
// CPU reference:
//   div_p[:-1,:] -= px[:-1,:]
//   div_p[1: ,:] += px[:-1,:]
//   div_p[:,:-1] -= py[:,:-1]
//   div_p[:,1: ] += py[:,:-1]
// ============================================================
__kernel void tv_div(
    __global const float *px,
    __global const float *py,
    __global float *div,
    const int Nx,
    const int Ny,
    const int K
){
    const int p = get_global_id(0);
    if (p >= Nx * Ny) return;
    const int i = p % Nx;
    const int j = p / Nx;
    const size_t npix = (size_t)Nx * Ny;

    for (int k = get_global_id(1); k < K; k += get_global_size(1)) {
        const size_t idx = p + npix * k;
        float v = 0.0f;

        // x direction
        if (i < Nx-1) v -= px[idx];
        if (i > 0)    v += px[idx - 1];

        // y direction
        if (j < Ny-1) v -= py[idx];
        if (j > 0)    v += py[idx - Nx];

        div[idx] = v;
    }
}


// ============================================================
// Dual update + projection  p <- proj(|p|<=weight)(p + tau*grad)
// ============================================================
__kernel void tv_dual_update(
    __global float *px,
    __global float *py,
    __global const float *ux,
    __global const float *uy,
    const float tau,
    const float weight,
    const ulong n
){
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        float pxn = px[i] + tau * ux[i];
        float pyn = py[i] + tau * uy[i];

        float nrm = sqrt(pxn*pxn + pyn*pyn);
        float denom = fmax(1.0f, nrm / weight);

        px[i] = pxn / denom;
        py[i] = pyn / denom;
    }
}


// ============================================================
// Primal update: u = f - div_p
// ============================================================
__kernel void tv_primal_update(
    __global float *u,
    __global const float *f,
    __global const float *div,
    const ulong n
){
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0))
        u[i] = f[i] - div[i];
}


// ============================================================
// Nonnegativity projection
// ============================================================
__kernel void prox_nonneg(
    __global float *x,
    const ulong n
){
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0))
        if (x[i] < 0.0f) x[i] = 0.0f;
}


__kernel void tv_norm(
    __global const float *gx,
    __global const float *gy,
    __global float *out,
    const ulong n
){
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0))
        out[i] = sqrt(gx[i]*gx[i] + gy[i]*gy[i]);
}

"""


class TVProxKernels:
    """Compiled TV proximal operator OpenCL kernels (Chambolle algorithm)."""
    def __init__(self, ctx: cl.Context):
        self.prg = cl.Program(ctx, TV_KERNELS).build()

        # cache kernels once
        self.k_grad   = cl.Kernel(self.prg, "tv_grad")
        self.k_div    = cl.Kernel(self.prg, "tv_div")
        self.k_dual   = cl.Kernel(self.prg, "tv_dual_update")
        self.k_primal = cl.Kernel(self.prg, "tv_primal_update")
        self.k_nonneg = cl.Kernel(self.prg, "prox_nonneg")
        self.k_norm = cl.Kernel(self.prg, "tv_norm")


def prox_tv_nonneg_inplace(
    queue,
    kernels: TVProxKernels,
    x_gpu,      # (K, Ny, Nx) C order, modified IN PLACE
    y_gpu,      # buffer holding f (same shape)
    ux, uy,     # gradient buffers
    px, py,     # dual buffers
    div,        # divergence buffer
    weight,     # TV weight (lambda)
    tau,        # ignored (kept for API consistency)
    n_iter: int,
    return_stats = False
):
    """
    Chambolle TV prox (skimage-equivalent), per K-slice:

        argmin_u 0.5 ||u - f||^2 + weight * TV(u)

    In-place update: x_gpu <- prox_TV(x_gpu)

    IMPORTANT:
      - weight enters ONLY in the dual projection (|p| <= weight)
      - primal is u = f - div(p)
      - tau here is ignored; algorithm uses tv_tau = 1/(2*ndim)=0.25 for 2D
    """

    K, Ny, Nx = map(int, x_gpu.shape)

    # typed scalars
    iNx = np.int32(Nx)
    iNy = np.int32(Ny)
    iK  = np.int32(K)
    gws, itotal = elementwise(x_gpu.size)          # element-wise kernels
    gws_pk = pixel_orientation(Nx * Ny, K)          # tv_grad, tv_div

    # algorithm constants
    tv_tau = np.float32(0.25)  # = 1/(2*ndim) with ndim=2
    w = np.float32(weight)

    # f <- x
    cl.enqueue_copy(queue, y_gpu.data, x_gpu.data)

    # Main loop

    for _ in range(int(n_iter)):
        # div_p = div(p)
        kernels.k_div(
            queue, gws_pk, None,
            px.data, py.data, div.data,
            iNx, iNy, iK
        )

        # u = f - div_p   (write into x_gpu)
        kernels.k_primal(
            queue, gws, None,
            x_gpu.data, y_gpu.data, div.data,
            itotal
        )

        # grad(u)
        kernels.k_grad(
            queue, gws_pk, None,
            x_gpu.data, ux.data, uy.data,
            iNx, iNy, iK
        )

        # p <- proj(|p|<=w)( p + tv_tau * grad(u) )
        kernels.k_dual(
            queue, gws, None,
            px.data, py.data,
            ux.data, uy.data,
            tv_tau, w,
            itotal
        )

        if return_stats:
            # reuse div as residual
            last_res = clarray.vdot(div, div)

    # final primal u = f - div(p)
    kernels.k_div(queue, gws_pk, None, px.data, py.data, div.data, iNx, iNy, iK)
    kernels.k_primal(queue, gws, None, x_gpu.data, y_gpu.data, div.data, itotal)

    # nonnegativity (sequential prox)
    kernels.k_nonneg(queue, gws, None, x_gpu.data, itotal)

    if return_stats:
        return x_gpu, float(last_res.get())
    else:
        return x_gpu
