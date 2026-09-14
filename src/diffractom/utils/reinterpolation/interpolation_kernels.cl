#ifndef TS
#define TS 16
#endif


// ============================================================
// Quaternion helpers
// ============================================================

inline float4 quat_mul(
    const float4 a,
    const float4 b
){
    return (float4)(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
    );
}


inline float4 quat_inverse(const float4 q)
{
    return (float4)(
        -q.x,
        -q.y,
        -q.z,
         q.w
    );
}


// ============================================================
// 1. Transform evaluation quaternions by all symmetries
// ============================================================

__kernel void transform_eval_symmetries(
    __global const float *eval_quat,
    __global const float *sym_quat,
    __global float *transformed_eval,
    const int M,
    const int Ns,
    const int offset,
    const int chunk_size
){
    int local_gid = get_global_id(0);

    if (local_gid >= chunk_size)
        return;

    int gid = local_gid + offset;

    int s = gid % Ns;
    int m = gid / Ns;

    float4 q_eval = vload4(m, eval_quat);
    float4 q_sym  = vload4(s, sym_quat);

    float4 q = quat_mul(
        q_eval,
        quat_inverse(q_sym)
    );

    vstore4(q, gid, transformed_eval);
}


// ============================================================
// 2. Count non-zero interpolation entries
// ============================================================
//
// The complete logical problem contains M*K pairs, which can
// exceed INT32_MAX.
//
// The global pair index therefore uses ulong (64-bit).
//
// Each launch processes only a chunk:
//
//     gid = offset + local_gid
//
// ============================================================

__kernel void count_interpolation_nnz(
    __global const float *grid_quat,
    __global const float *transformed_eval,
    __global int *row_counts,
    const int M,
    const int K,
    const int Ns,
    const float sigma_interp,
    const float cutoff_sigma,
    const ulong offset,
    const int chunk_size
){
    int local_gid = get_global_id(0);

    if (local_gid >= chunk_size)
        return;

    ulong gid = offset + (ulong)local_gid;

    int k = (int)(gid % (ulong)K);
    int m = (int)(gid / (ulong)K);

    float4 q_grid = vload4(
        k,
        grid_quat
    );

    float max_abs_dot = 0.0f;

    for (int s = 0; s < Ns; ++s)
    {
        int idx = m * Ns + s;

        float4 q_eval = vload4(
            idx,
            transformed_eval
        );

        float dot =
              q_eval.x * q_grid.x
            + q_eval.y * q_grid.y
            + q_eval.z * q_grid.z
            + q_eval.w * q_grid.w;

        dot = fabs(dot);

        max_abs_dot = fmax(
            max_abs_dot,
            dot
        );
    }

    float cutoff = cutoff_sigma * sigma_interp;

    float threshold = cos(
        0.5f * cutoff
    );

    if (max_abs_dot >= threshold)
    {
        atomic_inc(
            (__global volatile unsigned int *)row_counts + m
        );
    }
}



// ============================================================
// 3. Initialize an integer array
// ============================================================

__kernel void fill_int(
    __global int *array,
    const int value,
    const int N
){
    int gid = get_global_id(0);

    if (gid >= N)
        return;

    array[gid] = value;
}


// ============================================================
// 4. Copy integer array
// ============================================================

__kernel void copy_int(
    __global const int *src,
    __global int *dst,
    const int N
){
    int gid = get_global_id(0);

    if (gid >= N)
        return;

    dst[gid] = src[gid];
}


// ============================================================
// 5. Inclusive scan step
// ============================================================

__kernel void inclusive_scan_step(
    __global const int *in,
    __global int *out,
    const int N,
    const int offset
){
    int gid = get_global_id(0);

    if (gid >= N)
        return;

    int value = in[gid];

    if (gid >= offset)
        value += in[gid - offset];

    out[gid] = value;
}


// ============================================================
// 6. Convert inclusive scan to CSR row pointer
// ============================================================

__kernel void make_row_ptr(
    __global const int *inclusive,
    __global int *row_ptr,
    const int M
){
    int gid = get_global_id(0);

    if (gid > M)
        return;

    if (gid == 0)
    {
        row_ptr[0] = 0;
    }
    else
    {
        row_ptr[gid] = inclusive[gid - 1];
    }
}


// ============================================================
// 7. Initialize CSR write counters
// ============================================================

__kernel void fill_csr_offsets(
    __global int *offsets,
    const int M
){
    int gid = get_global_id(0);

    if (gid >= M)
        return;

    offsets[gid] = 0;
}

// ============================================================
// 8. Fill CSR
// ============================================================
//
// As with counting, the global M*K pair index uses ulong.
//
// ============================================================

__kernel void fill_interpolation_csr(
    __global const float *grid_quat,
    __global const float *transformed_eval,
    __global const int *row_ptr,
    __global int *write_offsets,
    __global int *col_indices,
    __global float *weights,
    const int M,
    const int K,
    const int Ns,
    const float sigma_interp,
    const float cutoff_sigma,
    const ulong offset,
    const int chunk_size
){
    int local_gid = get_global_id(0);

    if (local_gid >= chunk_size)
        return;

    ulong gid = offset + (ulong)local_gid;

    int k = (int)(gid % (ulong)K);
    int m = (int)(gid / (ulong)K);

    float4 q_grid = vload4(
        k,
        grid_quat
    );

    float max_abs_dot = 0.0f;

    for (int s = 0; s < Ns; ++s)
    {
        int idx = m * Ns + s;

        float4 q_eval = vload4(
            idx,
            transformed_eval
        );

        float dot =
              q_eval.x * q_grid.x
            + q_eval.y * q_grid.y
            + q_eval.z * q_grid.z
            + q_eval.w * q_grid.w;

        dot = fabs(dot);

        max_abs_dot = fmax(
            max_abs_dot,
            dot
        );
    }

    max_abs_dot = clamp(
        max_abs_dot,
        0.0f,
        1.0f
    );

    float cutoff = cutoff_sigma * sigma_interp;

    float threshold = cos(
        0.5f * cutoff
    );

    if (max_abs_dot < threshold)
        return;

    float angle = 2.0f * acos(
        max_abs_dot
    );

    float weight = exp(
        -0.5f
        * (angle / sigma_interp)
        * (angle / sigma_interp)
    );

    int offset_local = atomic_inc(
        (__global volatile unsigned int *)write_offsets + m
    );

    int position = row_ptr[m] + offset_local;

    col_indices[position] = k;
    weights[position] = weight;
}


// ============================================================
// 9. Sparse interpolation
// ============================================================

__kernel void smooth_reinterpolate(
    __global const float *coefficients,
    __global const int *row_ptr,
    __global const int *col_indices,
    __global const float *weights,
    __global float *output,
    const int N,
    const int K,
    const int M
){
    int gid = get_global_id(0);

    int total = N * M;

    if (gid >= total)
        return;

    int m = gid % M;
    int n = gid / M;

    int start = row_ptr[m];
    int end   = row_ptr[m + 1];

    float sum = 0.0f;

    for (int j = start; j < end; ++j)
    {
        int k = col_indices[j];

        sum +=
            weights[j]
            * coefficients[n * K + k];
    }

    output[n * M + m] = sum;
}



// ============================================================
// S²: Count non-zero interpolation entries
// ============================================================

__kernel void count_interpolation_nnz_s2(
    __global const float *grid_poles,
    __global const float *eval_poles,
    __global int *row_counts,
    const int M,
    const int K,
    const float sigma_interp,
    const float cutoff_sigma,
    const ulong offset,
    const int chunk_size
){
    int local_gid = get_global_id(0);

    if (local_gid >= chunk_size)
        return;

    ulong gid =
        offset
        + (ulong)local_gid;

    int k = (int)(
        gid % (ulong)K
    );

    int m = (int)(
        gid / (ulong)K
    );

    float3 p_grid = vload3(
        k,
        grid_poles
    );

    float3 p_eval = vload3(
        m,
        eval_poles
    );

    float abs_dot = fabs(
          p_grid.x * p_eval.x
        + p_grid.y * p_eval.y
        + p_grid.z * p_eval.z
    );

    abs_dot = clamp(
        abs_dot,
        0.0f,
        1.0f
    );

    float cutoff =
        cutoff_sigma
        * sigma_interp;

    cutoff = fmin(
        cutoff,
        1.5707963267948966f
    );

    float threshold = cos(
        cutoff
    );

    if (abs_dot >= threshold)
    {
        atomic_inc(
            (__global volatile unsigned int *)
            row_counts + m
        );
    }
}


// ============================================================
// S²: Fill sparse interpolation CSR
// ============================================================

__kernel void fill_interpolation_csr_s2(
    __global const float *grid_poles,
    __global const float *eval_poles,
    __global const int *row_ptr,
    __global int *write_offsets,
    __global int *col_indices,
    __global float *weights,
    const int M,
    const int K,
    const float sigma_interp,
    const float cutoff_sigma,
    const ulong offset,
    const int chunk_size
){
    int local_gid = get_global_id(0);

    if (local_gid >= chunk_size)
        return;

    ulong gid =
        offset
        + (ulong)local_gid;

    int k = (int)(
        gid % (ulong)K
    );

    int m = (int)(
        gid / (ulong)K
    );

    float3 p_grid = vload3(
        k,
        grid_poles
    );

    float3 p_eval = vload3(
        m,
        eval_poles
    );

    float abs_dot = fabs(
          p_grid.x * p_eval.x
        + p_grid.y * p_eval.y
        + p_grid.z * p_eval.z
    );

    abs_dot = clamp(
        abs_dot,
        0.0f,
        1.0f
    );

    float cutoff =
        cutoff_sigma
        * sigma_interp;

    cutoff = fmin(
        cutoff,
        1.5707963267948966f
    );

    float threshold = cos(
        cutoff
    );

    if (abs_dot < threshold)
        return;

    float angle = acos(
        abs_dot
    );

    float scaled_angle =
        angle / sigma_interp;

    float weight = exp(
        -0.5f
        * scaled_angle
        * scaled_angle
    );

    int offset_local = atomic_inc(
        (__global volatile unsigned int *)
        write_offsets + m
    );

    int position =
        row_ptr[m]
        + offset_local;

    col_indices[position] = k;
    weights[position] = weight;
}