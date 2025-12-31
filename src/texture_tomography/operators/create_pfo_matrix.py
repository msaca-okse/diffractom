PF_KERNEL_SRC = r"""
__kernel void pfmatrix_eval(
    __global const float *coords,      // (R, C, P, 3) flattened
    __global const float *grid_inv,    // (K, 9) flattened row-major (3x3)
    __global const float *sym_ops,     // (G, 9) flattened row-major (3x3)
    __global const float *hvecs,       // (P, 3) normalized h-vectors

    __global const float *inv_sigma2,  // (K,) 1 / sigma_k^2
    __global const float *norm_factor, // (K,) 1 / (8*pi*sigma_k^2)

    __global float *out,               // (R, K, C, P) flattened

    const int R,
    const int K,
    const int C,
    const int P,
    const int G
){
    int gid = get_global_id(0);
    int total = R * K * C * P;
    if (gid >= total) return;

    // decode gid for layout out[r,k,c,p]
    int p = gid % P;
    int tmp = gid / P;
    int c = tmp % C;
    tmp /= C;
    int k = tmp % K;
    int r = tmp / K;

    // --- load coordinate v = coords[r,c,p,:] ---
    int coord_base = ((r * C + c) * P + p) * 3;
    float vx = coords[coord_base + 0];
    float vy = coords[coord_base + 1];
    float vz = coords[coord_base + 2];

    // --- apply inverse grid rotation q = grid_inv[k] * v ---
    int Rb = k * 9;
    float qx = grid_inv[Rb + 0]*vx + grid_inv[Rb + 1]*vy + grid_inv[Rb + 2]*vz;
    float qy = grid_inv[Rb + 3]*vx + grid_inv[Rb + 4]*vy + grid_inv[Rb + 5]*vz;
    float qz = grid_inv[Rb + 6]*vx + grid_inv[Rb + 7]*vy + grid_inv[Rb + 8]*vz;

    // --- load normalized h-vector for this reflection p ---
    int hb = p * 3;
    float hx0 = hvecs[hb + 0];
    float hy0 = hvecs[hb + 1];
    float hz0 = hvecs[hb + 2];

    float w = 0.0f;
    float inv_sig2 = inv_sigma2[k];

    // loop over symmetry operations
    for (int g = 0; g < G; ++g) {
        int Sb = g * 9;

        // h_rot = sym_ops[g] * h0
        float hx = sym_ops[Sb + 0]*hx0 + sym_ops[Sb + 1]*hy0 + sym_ops[Sb + 2]*hz0;
        float hy = sym_ops[Sb + 3]*hx0 + sym_ops[Sb + 4]*hy0 + sym_ops[Sb + 5]*hz0;
        float hz = sym_ops[Sb + 6]*hx0 + sym_ops[Sb + 7]*hy0 + sym_ops[Sb + 8]*hz0;

        float dot = hx*qx + hy*qy + hz*qz;

        float arg1 = -(1.0f - dot) * inv_sig2;
        if (arg1 > -6.0f) w += exp(arg1);

        float arg2 = -(1.0f + dot) * inv_sig2;
        if (arg2 > -6.0f) w += exp(arg2);
    }

    out[gid] = w * norm_factor[k];
}
"""

PFSPARSE_KERNEL_SRC = r"""
__kernel void pfmatrix_count_sparse(
    __global const float *coords,
    __global const float *grid_inv,
    __global const float *sym_ops,
    __global const float *hvecs,

    __global volatile int *nnz_count,


    const int R,
    const int K,
    const int C,
    const int P,
    const int G,
    const float inv_sigma2,
    const float norm_factor,
    const float cutoff
){
    int gid = get_global_id(0);
    int total = R * K * C * P;
    if (gid >= total) return;

    int p = gid % P;
    int tmp = gid / P;
    int c = tmp % C;
    tmp /= C;
    int k = tmp % K;
    int r = tmp / K;

    int coord_base = ((r * C + c) * P + p) * 3;
    float vx = coords[coord_base + 0];
    float vy = coords[coord_base + 1];
    float vz = coords[coord_base + 2];

    int Rb = k * 9;
    float qx = grid_inv[Rb + 0]*vx + grid_inv[Rb + 1]*vy + grid_inv[Rb + 2]*vz;
    float qy = grid_inv[Rb + 3]*vx + grid_inv[Rb + 4]*vy + grid_inv[Rb + 5]*vz;
    float qz = grid_inv[Rb + 6]*vx + grid_inv[Rb + 7]*vy + grid_inv[Rb + 8]*vz;

    int hb = p * 3;
    float hx0 = hvecs[hb + 0];
    float hy0 = hvecs[hb + 1];
    float hz0 = hvecs[hb + 2];

    float w = 0.0f;

    for (int g = 0; g < G; ++g) {
        int Sb = g * 9;

        float hx = sym_ops[Sb + 0]*hx0 + sym_ops[Sb + 1]*hy0 + sym_ops[Sb + 2]*hz0;
        float hy = sym_ops[Sb + 3]*hx0 + sym_ops[Sb + 4]*hy0 + sym_ops[Sb + 5]*hz0;
        float hz = sym_ops[Sb + 6]*hx0 + sym_ops[Sb + 7]*hy0 + sym_ops[Sb + 8]*hz0;

        float dot = hx*qx + hy*qy + hz*qz;

        float arg1 = -(1.0f - dot) * inv_sigma2;
        if (arg1 > -6.0f) w += exp(arg1);

        float arg2 = -(1.0f + dot) * inv_sigma2;
        if (arg2 > -6.0f) w += exp(arg2);
    }

    w *= norm_factor;

    if (w >= cutoff)
        atomic_add(&nnz_count[k],1);
}


__kernel void pfmatrix_write_sparse(
    __global const float *coords,     // (R, C, P, 3)
    __global const float *grid_inv,   // (K, 9)
    __global const float *sym_ops,    // (G, 9)
    __global const float *hvecs,      // (P, 3)

    __global const int *nnz_offset,   // (K+1)
    __global int *nnz_cursor,         // (K), initialized to 0

    __global int *coo_k,              // (nnz)
    __global int *coo_r,              // (nnz)
    __global int *coo_c,              // (nnz)
    __global int *coo_p,              // (nnz)
    __global float *coo_val,          // (nnz)

    const int R,
    const int K,
    const int C,
    const int P,
    const int G,
    const float inv_sigma2,
    const float norm_factor,
    const float cutoff
){
    int gid = get_global_id(0);
    int total = R * K * C * P;
    if (gid >= total) return;

    // decode gid → (r,k,c,p) for layout out[r,k,c,p]
    int p = gid % P;
    int tmp = gid / P;
    int c = tmp % C;
    tmp /= C;
    int k = tmp % K;
    int r = tmp / K;

    // coords[r,c,p,:]
    int coord_base = ((r * C + c) * P + p) * 3;
    float vx = coords[coord_base + 0];
    float vy = coords[coord_base + 1];
    float vz = coords[coord_base + 2];

    // q = grid_inv[k] * v
    int Rb = k * 9;
    float qx = grid_inv[Rb + 0]*vx + grid_inv[Rb + 1]*vy + grid_inv[Rb + 2]*vz;
    float qy = grid_inv[Rb + 3]*vx + grid_inv[Rb + 4]*vy + grid_inv[Rb + 5]*vz;
    float qz = grid_inv[Rb + 6]*vx + grid_inv[Rb + 7]*vy + grid_inv[Rb + 8]*vz;

    // h-vector for peak p
    int hb = p * 3;
    float hx0 = hvecs[hb + 0];
    float hy0 = hvecs[hb + 1];
    float hz0 = hvecs[hb + 2];

    float w = 0.0f;

    for (int g = 0; g < G; ++g) {
        int Sb = g * 9;

        // h_rot = sym_ops[g] * h0
        float hx = sym_ops[Sb + 0]*hx0 + sym_ops[Sb + 1]*hy0 + sym_ops[Sb + 2]*hz0;
        float hy = sym_ops[Sb + 3]*hx0 + sym_ops[Sb + 4]*hy0 + sym_ops[Sb + 5]*hz0;
        float hz = sym_ops[Sb + 6]*hx0 + sym_ops[Sb + 7]*hy0 + sym_ops[Sb + 8]*hz0;

        float dot = hx*qx + hy*qy + hz*qz;

        float arg1 = -(1.0f - dot) * inv_sigma2;
        if (arg1 > -6.0f) w += exp(arg1);

        float arg2 = -(1.0f + dot) * inv_sigma2;
        if (arg2 > -6.0f) w += exp(arg2);
    }

    w *= norm_factor;
    if (w < cutoff) return;

    int pos = atomic_add(&nnz_cursor[k], 1);
;
    int idx = nnz_offset[k] + pos;

    // store 4D COO indices (and value)
    coo_k[idx]   = k;
    coo_r[idx]   = r;
    coo_c[idx]   = c;
    coo_p[idx]   = p;
    coo_val[idx] = w;
}

__kernel void sparse_pf_innerprod_4d(
    __global const int   *coo_r,      // (nnz)
    __global const int   *coo_c,      // (nnz)
    __global const int   *coo_p,      // (nnz)
    __global const float *coo_val,    // (nnz)

    __global const int   *offset,     // (K+1)

    __global const float *residual,   // (R, C, P) flattened
    __global float *out,              // (K)

    const int K,
    const int C,
    const int P
){
    int k = get_global_id(0);
    if (k >= K) return;

    int start = offset[k];
    int end   = offset[k + 1];

    float acc = 0.0f;

    for (int i = start; i < end; ++i) {
        int r = coo_r[i];
        int c = coo_c[i];
        int p = coo_p[i];

        int idx = (r * C + c) * P + p;
        acc += coo_val[i] * residual[idx];
    }

    out[k] = acc;
}

__kernel void interp_theta_4d(
    __global const float *in,
    __global float *out,

    __global const float *theta_old,  // (T)
    __global const float *theta_new,  // (P)

    const int R,
    const int X,
    const int C,
    const int T,
    const int P
){
    int gid = get_global_id(0);
    int total = R * X * C * P;
    if (gid >= total) return;

    int p = gid % P;
    int tmp = gid / P;
    int c = tmp % C;
    tmp /= C;
    int x = tmp % X;
    int r = tmp / X;

    float t = theta_new[p];

    // find interval [i, i+1] in theta_old
    int i = 0;
    while (i < T - 2 && theta_old[i + 1] < t)
        i++;

    float t0 = theta_old[i];
    float t1 = theta_old[i + 1];
    float w = (t - t0) / (t1 - t0);

    // clamp
    if (w < 0.0f) w = 0.0f;
    if (w > 1.0f) w = 1.0f;

    int idx0 = ((r * X + x) * C + c) * T + i;
    int idx1 = idx0 + 1;

    out[((r * X + x) * C + c) * P + p] =
        (1.0f - w) * in[idx0] + w * in[idx1];
}

__kernel void interp_theta_3d(
    __global const float *in,
    __global float *out,

    __global const float *theta_old,  // (T)
    __global const float *theta_new,  // (P)

    const int R,
    const int C,
    const int T,
    const int P
){
    int gid = get_global_id(0);
    int total = R * C * P;
    if (gid >= total) return;

    int p = gid % P;
    int tmp = gid / P;
    int c = tmp % C;
    int r = tmp / C;

    float t = theta_new[p];

    int i = 0;
    while (i < T - 2 && theta_old[i + 1] < t)
        i++;

    float t0 = theta_old[i];
    float t1 = theta_old[i + 1];
    float w = (t - t0) / (t1 - t0);

    if (w < 0.0f) w = 0.0f;
    if (w > 1.0f) w = 1.0f;

    int idx0 = (r * C + c) * T + i;
    int idx1 = idx0 + 1;

    out[(r * C + c) * P + p] =
        (1.0f - w) * in[idx0] + w * in[idx1];
}

__kernel void sum_over_x(
    __global const float *in,
    __global float *out,
    const int R,
    const int X,
    const int C,
    const int T
){
    int gid = get_global_id(0);
    int total = R * C * T;
    if (gid >= total) return;

    int t = gid % T;
    int tmp = gid / T;
    int c = tmp % C;
    int r = tmp / C;

    float acc = 0.0f;
    for (int x = 0; x < X; ++x) {
        int idx = ((r * X + x) * C + c) * T + t;
        acc += in[idx];
    }

    out[(r * C + c) * T + t] = acc;
}


__kernel void clip_nonnegative(
    __global float *data,
    const int n
){
    int gid = get_global_id(0);
    if (gid >= n) return;

    float v = data[gid];
    if (v < 0.0f)
        data[gid] = 0.0f;
}

"""







import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

def build_pf_program(ctx: cl.Context) -> cl.Program:
    return cl.Program(ctx, PF_KERNEL_SRC).build()

def build_pfsparse_program(ctx: cl.Context) -> cl.Program:
    return cl.Program(ctx, PFSPARSE_KERNEL_SRC).build()


def pfmatrix_eval_gpu(
    queue: cl.CommandQueue,
    pfo_kernel: cl.Kernel,
    coords_gpu: clarray.Array,
    grid_inv_gpu: clarray.Array,
    sym_ops_gpu: clarray.Array,
    hvecs_gpu: clarray.Array,
    R: int, K: int, C: int, P: int, G: int,
    sigma,                      # array-like, shape (K,)
    out_gpu: clarray.Array = None
):

    # --- sanity checks ---
    assert coords_gpu.dtype == np.float32
    assert grid_inv_gpu.dtype == np.float32
    assert sym_ops_gpu.dtype == np.float32
    assert hvecs_gpu.dtype == np.float32

    sigma = np.asarray(sigma, dtype=np.float32)
    assert sigma.shape == (K,)

    # --- output ---
    if out_gpu is None:
        out_gpu = clarray.empty(queue, (R, K, C, P), dtype=np.float32, order="C")

    # --- per-grid sigma parameters ---
    inv_sigma2 = np.float32(1.0) / (sigma * sigma)
    norm_factor = np.float32(1.0) / (8.0 * np.pi * sigma * sigma)

    inv_sigma2_gpu = clarray.to_device(queue, inv_sigma2)
    norm_factor_gpu = clarray.to_device(queue, norm_factor)

    # --- launch ---
    total = R * K * C * P
    pfo_kernel(
        queue,
        (total,),
        None,
        coords_gpu.data,
        grid_inv_gpu.data,
        sym_ops_gpu.data,
        hvecs_gpu.data,
        inv_sigma2_gpu.data,
        norm_factor_gpu.data,
        out_gpu.data,
        np.int32(R),
        np.int32(K),
        np.int32(C),
        np.int32(P),
        np.int32(G),
    )

    return out_gpu





def pfmatrix_sparseeval_gpu(
    queue: cl.CommandQueue,
    count_kernel: cl.Kernel,
    write_kernel: cl.Kernel,
    coords_gpu: clarray.Array,
    grid_inv_gpu: clarray.Array,
    sym_ops_gpu: clarray.Array,
    hvecs_gpu: clarray.Array,
    R: int, K: int, C: int, P: int, G: int,
    sigma: float,
    cutoff: float,
):
    # --- sanity checks ---
    assert coords_gpu.dtype == np.float32
    assert grid_inv_gpu.dtype == np.float32
    assert sym_ops_gpu.dtype == np.float32
    assert hvecs_gpu.dtype == np.float32

    inv_sigma2 = np.float32(1.0 / (sigma * sigma))
    norm_factor = np.float32(1.0 / (8.0 * np.pi * sigma * sigma))
    cutoff = np.float32(cutoff)


    # ------------------------------------------------------------
    # PASS 1: count nnz per coefficient k
    # ------------------------------------------------------------
    nnz_count_gpu = clarray.zeros(queue, (K,), dtype=np.int32)

    total = R * K * C * P
    count_kernel(
        queue,
        (total,),
        None,
        coords_gpu.data,
        grid_inv_gpu.data,
        sym_ops_gpu.data,
        hvecs_gpu.data,
        nnz_count_gpu.data,
        np.int32(R), np.int32(K), np.int32(C), np.int32(P), np.int32(G),
        inv_sigma2,
        norm_factor,
        cutoff
    )

    nnz_count = nnz_count_gpu.get()

    # ------------------------------------------------------------
    # PREFIX SUM (CPU, exclusive)
    # ------------------------------------------------------------
    nnz_offset = np.zeros(K + 1, dtype=np.int32)
    nnz_offset[1:] = np.cumsum(nnz_count)
    total_nnz = int(nnz_offset[-1])

    # ------------------------------------------------------------
    # Allocate sparse COO buffers (GPU)
    # ------------------------------------------------------------
    # Always return GPU arrays to match your updated plan
    if total_nnz == 0:
        empty_i32 = clarray.empty(queue, (0,), dtype=np.int32)
        empty_f32 = clarray.empty(queue, (0,), dtype=np.float32)
        return {
            "k": empty_i32,
            "r": empty_i32,
            "c": empty_i32,
            "p": empty_i32,
            "val": empty_f32,
            "offset": nnz_offset,  # CPU
        }

    coo_k_gpu = clarray.empty(queue, (total_nnz,), dtype=np.int32)
    coo_r_gpu = clarray.empty(queue, (total_nnz,), dtype=np.int32)
    coo_c_gpu = clarray.empty(queue, (total_nnz,), dtype=np.int32)
    coo_p_gpu = clarray.empty(queue, (total_nnz,), dtype=np.int32)
    coo_val_gpu = clarray.empty(queue, (total_nnz,), dtype=np.float32)

    nnz_offset_gpu = clarray.to_device(queue, nnz_offset)
    nnz_cursor_gpu = clarray.zeros(queue, (K,), dtype=np.int32)

    # ------------------------------------------------------------
    # PASS 2: write sparse COO
    # ------------------------------------------------------------
    write_kernel(
        queue,
        (total,),
        None,
        coords_gpu.data,
        grid_inv_gpu.data,
        sym_ops_gpu.data,
        hvecs_gpu.data,
        nnz_offset_gpu.data,
        nnz_cursor_gpu.data,
        coo_k_gpu.data,
        coo_r_gpu.data,
        coo_c_gpu.data,
        coo_p_gpu.data,
        coo_val_gpu.data,
        np.int32(R), np.int32(K), np.int32(C), np.int32(P), np.int32(G),
        inv_sigma2,
        norm_factor,
        cutoff
    )

    return {
        "k": coo_k_gpu,
        "r": coo_r_gpu,
        "c": coo_c_gpu,
        "p": coo_p_gpu,
        "val": coo_val_gpu,
        "offset": nnz_offset,  # CPU offsets per k
    }



def sparse_pf_innerprod_gpu(
    queue: cl.CommandQueue,
    kernel: cl.Kernel,
    sparse_pf: dict,
    residual_gpu: clarray.Array,
    R: int,
    C: int,
    P: int,
    K: int,
):
    """
    Compute <AK, r> for all K coefficients using sparse 4D PF COO data.

    Parameters
    ----------
    sparse_pf : dict
        {
            "k":   coo_k_gpu (unused here),
            "r":   coo_r_gpu,
            "c":   coo_c_gpu,
            "p":   coo_p_gpu,
            "val": coo_val_gpu,
            "offset": nnz_offset (CPU, shape K+1)
        }

    residual_gpu : clarray.Array
        shape (R, C, P), float32, C-ordered

    Returns
    -------
    out_gpu : clarray.Array
        shape (K,), float32
    """

    coo_r_gpu = sparse_pf["r"]
    coo_c_gpu = sparse_pf["c"]
    coo_p_gpu = sparse_pf["p"]
    coo_val_gpu = sparse_pf["val"]
    offset_cpu = sparse_pf["offset"]

    assert residual_gpu.dtype == np.float32
    assert coo_val_gpu.dtype == np.float32
    assert coo_r_gpu.dtype == np.int32
    assert coo_c_gpu.dtype == np.int32
    assert coo_p_gpu.dtype == np.int32
    assert offset_cpu.dtype == np.int32
    assert offset_cpu.shape[0] == K + 1

    # flatten residual to 1D (C-order already guaranteed)
    residual_flat_gpu = residual_gpu.reshape((R * C * P,))

    # copy offsets to GPU
    offset_gpu = clarray.to_device(queue, offset_cpu)

    # output
    out_gpu = clarray.empty(queue, (K,), dtype=np.float32)

    # launch kernel
    kernel(
        queue,
        (K,),
        None,
        coo_r_gpu.data,
        coo_c_gpu.data,
        coo_p_gpu.data,
        coo_val_gpu.data,
        offset_gpu.data,
        residual_flat_gpu.data,
        out_gpu.data,
        np.int32(K),
        np.int32(C),
        np.int32(P),
    )

    return out_gpu




def interp_theta_4d_gpu(
    queue: cl.CommandQueue,
    kernel: cl.Kernel,
    in_gpu: clarray.Array,
    out_gpu: clarray.Array,
    theta_old_gpu: clarray.Array,
    theta_new_gpu: clarray.Array,
    R: int, X: int, C: int, T: int, P: int,
):
    assert in_gpu.dtype == np.float32
    assert out_gpu.dtype == np.float32
    assert theta_old_gpu.dtype == np.float32
    assert theta_new_gpu.dtype == np.float32

    total = R * X * C * P

    kernel(
        queue,
        (total,),
        None,
        in_gpu.data,
        out_gpu.data,
        theta_old_gpu.data,
        theta_new_gpu.data,
        np.int32(R),
        np.int32(X),
        np.int32(C),
        np.int32(T),
        np.int32(P),
    )




def interp_theta_3d_gpu(
    queue: cl.CommandQueue,
    kernel: cl.Kernel,
    in_gpu: clarray.Array,
    out_gpu: clarray.Array,
    theta_old_gpu: clarray.Array,
    theta_new_gpu: clarray.Array,
    R: int, C: int, T: int, P: int,
):
    assert in_gpu.dtype == np.float32
    assert out_gpu.dtype == np.float32
    assert theta_old_gpu.dtype == np.float32
    assert theta_new_gpu.dtype == np.float32

    total = R * C * P

    kernel(
        queue,
        (total,),
        None,
        in_gpu.data,
        out_gpu.data,
        theta_old_gpu.data,
        theta_new_gpu.data,
        np.int32(R),
        np.int32(C),
        np.int32(T),
        np.int32(P),
    )


def sum_over_x_gpu(queue, kernel, in_gpu, out_gpu, R, X, C, T):
    total = R * C * T
    kernel(
        queue,
        (total,),
        None,
        in_gpu.data,
        out_gpu.data,
        np.int32(R),
        np.int32(X),
        np.int32(C),
        np.int32(T),
    )


def clip_nonnegative_gpu(
    queue: cl.CommandQueue,
    kernel: cl.Kernel,
    arr_gpu: clarray.Array
):
    assert arr_gpu.dtype == np.float32

    n = arr_gpu.size

    kernel(
        queue,
        (n,),
        None,
        arr_gpu.data,
        np.int32(n),
    )
