# prox_tv.py
import pyopencl as cl
import pyopencl.array as clarray
import numpy as np


TV_KERNELS = r"""
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
    int gid = get_global_id(0);
    int total = Nx * Ny * K;
    if (gid >= total) return;

    int i = gid % Nx;
    int t = gid / Nx;
    int j = t % Ny;
    int k = t / Ny;

    int idx = i + Nx * (j + Ny * k);

    // forward differences (Neumann BC)
    ux[idx] = (i < Nx-1) ? (u[idx + 1]   - u[idx]) : 0.0f;
    uy[idx] = (j < Ny-1) ? (u[idx + Nx]  - u[idx]) : 0.0f;
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
    int gid = get_global_id(0);
    int total = Nx * Ny * K;
    if (gid >= total) return;

    int i = gid % Nx;
    int t = gid / Nx;
    int j = t % Ny;
    int k = t / Ny;

    int idx = i + Nx * (j + Ny * k);

    float v = 0.0f;

    // x direction
    if (i < Nx-1) v -= px[idx];
    if (i > 0)    v += px[idx - 1];

    // y direction
    if (j < Ny-1) v -= py[idx];
    if (j > 0)    v += py[idx - Nx];

    div[idx] = v;
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
    const int total
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    float pxn = px[gid] + tau * ux[gid];
    float pyn = py[gid] + tau * uy[gid];

    float nrm = sqrt(pxn*pxn + pyn*pyn);
    float denom = fmax(1.0f, nrm / weight);

    px[gid] = pxn / denom;
    py[gid] = pyn / denom;
}


// ============================================================
// Primal update: u = f - div_p
// ============================================================
__kernel void tv_primal_update(
    __global float *u,
    __global const float *f,
    __global const float *div,
    const int total
){
    int gid = get_global_id(0);
    if (gid >= total) return;
    u[gid] = f[gid] - div[gid];
}


// ============================================================
// Nonnegativity projection
// ============================================================
__kernel void prox_nonneg(
    __global float *x,
    const int total
){
    int gid = get_global_id(0);
    if (gid >= total) return;
    if (x[gid] < 0.0f) x[gid] = 0.0f;
}


__kernel void tv_norm(
    __global const float *gx,
    __global const float *gy,
    __global float *out,
    const int total
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    float v = sqrt(gx[gid]*gx[gid] + gy[gid]*gy[gid]);
    out[gid] = v;
}

"""


class TVProxKernels:
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
    x_gpu,      # (Nx, Ny, K) Fortran, modified IN PLACE
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

    Nx, Ny, K = map(int, x_gpu.shape)
    total = int(x_gpu.size)

    # typed scalars
    iNx = np.int32(Nx)
    iNy = np.int32(Ny)
    iK  = np.int32(K)
    itotal = np.int32(total)

    # algorithm constants
    tv_tau = np.float32(0.25)  # = 1/(2*ndim) with ndim=2
    w = np.float32(weight)

    # f <- x
    cl.enqueue_copy(queue, y_gpu.data, x_gpu.data)

    # Main loop
    gws = (int(total),)

    for _ in range(int(n_iter)):
        # div_p = div(p)
        kernels.k_div(
            queue, gws, None,
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
            queue, gws, None,
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
    kernels.k_div(queue, gws, None, px.data, py.data, div.data, iNx, iNy, iK)
    kernels.k_primal(queue, gws, None, x_gpu.data, y_gpu.data, div.data, itotal)

    # nonnegativity (sequential prox)
    kernels.k_nonneg(queue, gws, None, x_gpu.data, itotal)

    if return_stats:
        return x_gpu, float(last_res.get())
    else:
        return x_gpu
