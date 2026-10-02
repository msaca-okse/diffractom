// expand_gaussian_peaks
// transpose_k_nrot_mx
// transpose_d_omega_k_f_to_c
// transpose_B_r_k_nsub_to_r_nsub_k
// transpose_r_mx_k_to_k_r_mx
// transpose_omega_d_k_c_to_d_omega_k_f
// accumulate_segments
// batched_gemm_rmn
// gather_last_axis
// gather_coeffs_k_slice
// slice_k_lastaxis_f
// scatter_k_lastaxis_f
// SLICE_COEFFS_K_BATCH
// SLICE_COEFFS_K_BATCH_F
// SLICE_GRIDINV_K_BATCH
// SCALE_PF_BY_INTENSITY_INPLACE
// scatter_k_batch_c





__kernel void expand_gaussian_peaks(
    __global const float *basis,
    __global const float *gaussian,
    __global float *out,
    int R, int K, int C, int P, int T
){
    int gid = get_global_id(0);

    int t   = gid % T;
    int tmp = gid / T;
    int c   = tmp % C;
    tmp    /= C;
    int k   = tmp % K;
    int r   = tmp / K;

    if (r >= R) return;

    float acc = 0.0f;
    int base_basis = (((r*K + k)*C + c)*P);
    int base_g     = t;

    for (int p = 0; p < P; ++p) {
        acc += basis[base_basis + p] * gaussian[p*T + t];
    }

    out[(((r*K + k)*C + c)*T + t)] = acc;
}



__kernel void transpose_k_nrot_mx(
    __global const float *inp,
    __global float *out,
    int K, int R, int Mx,
    int total   // = K*R*Mx
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    int k = gid % K;
    int tmp = gid / K;
    int x = tmp % Mx;
    int r = tmp / Mx;

    out[(r*Mx + x)*K + k] = inp[(k*R + r)*Mx + x];
}



__kernel void transpose_d_omega_k_f_to_c(
    __global const float *inp,   // (d, ω, K) Fortran
    __global float *out,         // (ω, d, K) C
    const int D,                 // number of detectors (d)
    const int O,                 // number of rotations (ω)
    const int K,                 // number of coefficients
    const int total              // = D * O * K
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    // Decompose linear index assuming output C-order (ω, d, K)
    int k = gid % K;
    int tmp = gid / K;
    int d = tmp % D;
    int o = tmp / D;

    // Fortran index: d + D*(o + O*k)
    int in_idx = d + D * (o + O * k);

    out[gid] = inp[in_idx];
}



__kernel void transpose_B_r_k_nsub_to_r_nsub_k(
    __global const float *inp,   // (R, K, Nsub)
    __global float *out,         // (R, Nsub, K)
    int R, int K, int Nsub,
    int total                    // = R*K*Nsub
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    int j = gid % Nsub;
    int tmp = gid / Nsub;
    int k = tmp % K;
    int r = tmp / K;

    // out[r,j,k] = inp[r,k,j]
    out[(r * Nsub + j) * K + k] = inp[(r * K + k) * Nsub + j];
}



__kernel void transpose_r_mx_k_to_k_r_mx(
    __global const float *inp,   // (R, Mx, K)
    __global float *out,         // (K, R, Mx)
    int R, int Mx, int K,
    int total                    // = R*Mx*K
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    int k = gid % K;
    int tmp = gid / K;
    int x = tmp % Mx;
    int r = tmp / Mx;

    out[(k*R + r)*Mx + x] = inp[(r*Mx + x)*K + k];
}



__kernel void transpose_omega_d_k_c_to_d_omega_k_f(
    __global const float *inp,   // (ω, d, K) C-order
    __global float *out,         // (d, ω, K) Fortran-order
    const int O,                 // number of omega
    const int D,                 // number of detectors
    const int K,                 // number of coefficients
    const int total              // = O * D * K
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    // Decompose gid assuming INPUT C-order (ω, d, K)
    int k = gid % K;
    int tmp = gid / K;
    int d = tmp % D;
    int o = tmp / D;

    // Input index (C-order)
    int in_idx = (o * D + d) * K + k;

    // Output index (Fortran-order)
    int out_idx = d + D * (o + O * k);

    out[out_idx] = inp[in_idx];
}






__kernel void accumulate_segments(
    __global float *out_full,
    __global const float *out_sub,
    __global const int *idx,
    int R, int Mx, int Nsub, int Nfull,
    int total // = R*Mx*Nsub
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    int s = gid % Nsub;
    int tmp = gid / Nsub;
    int x = tmp % Mx;
    int r = tmp / Mx;
    if (r >= R) return;

    int full_s = idx[s];
    // optional extra safety while debugging:
    if ((unsigned)full_s >= (unsigned)Nfull) return;

    out_full[(r*Mx + x)*Nfull + full_s] += out_sub[(r*Mx + x)*Nsub + s];
}




// Simple tiled batched GEMM: C[r,m,n] = sum_k A[r,m,k] * B[r,k,n]
//
// Layout assumptions (row-major / C-order contiguous):
//   A: (R, M, K) contiguous with fastest axis K
//   B: (R, K, N) contiguous with fastest axis N
//   C: (R, M, N) contiguous with fastest axis N
//
// Global NDRange: (N, M, R)  i.e. x=n, y=m, z=r
//
// Tune TS for your GPU (16 or 32 typical).
#ifndef TS
#define TS 16
#endif

__kernel void batched_gemm_rmn(
    __global const float *A,
    __global const float *B,
    __global float *C,
    const int R,
    const int M,
    const int K,
    const int N
){
    const int n = (int)get_global_id(0); // column in N
    const int m = (int)get_global_id(1); // row in M
    const int r = (int)get_global_id(2); // batch index

    if (r >= R || m >= M || n >= N) return;

    // Local indices within the tile
    const int ln = (int)get_local_id(0);
    const int lm = (int)get_local_id(1);

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float acc = 0.0f;

    // Base pointers for batch r
    const int A0 = (r * M) * K; // start of A[r,:,:]
    const int B0 = (r * K) * N; // start of B[r,:,:]
    const int C0 = (r * M) * N; // start of C[r,:,:]

    // Iterate over K in tiles of TS
    for (int k0 = 0; k0 < K; k0 += TS) {

        // Load A tile: Asub[lm][ln] = A[r, m, k0+ln]
        int ak = k0 + ln;
        if (ak < K) Asub[lm][ln] = A[A0 + m*K + ak];
        else        Asub[lm][ln] = 0.0f;

        // Load B tile: Bsub[lm][ln] = B[r, k0+lm, n]
        int bk = k0 + lm;
        if (bk < K) Bsub[lm][ln] = B[B0 + bk*N + n];
        else        Bsub[lm][ln] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial dot
        #pragma unroll
        for (int t = 0; t < TS; ++t) {
            acc += Asub[lm][t] * Bsub[t][ln];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[C0 + m*N + n] = acc;
}



__kernel void gather_last_axis(
    __global const float *inp,   // (R, Mx, Nfull)
    __global float *out,          // (R, Mx, Nsub)
    __global const int *idx,      // (Nsub)
    int R,
    int Mx,
    int Nsub,
    int Nfull
) {
    int gid = get_global_id(0);
    int total = R * Mx * Nsub;
    if (gid >= total) return;

    int j = gid % Nsub;
    int tmp = gid / Nsub;
    int x = tmp % Mx;
    int r = tmp / Mx;

    int src = idx[j];

    out[(r * Mx + x) * Nsub + j] =
        inp[(r * Mx + x) * Nfull + src];
}




__kernel void gather_coeffs_k_slice(
    __global const float *inp,   // (Ktot, R, Mx)
    __global float *out,         // (Ki,   R, Mx)
    int i0,
    int Ki,
    int R,
    int Mx,
    int Ktot
){
    int gid = get_global_id(0);
    int total = Ki * R * Mx;
    if (gid >= total) return;

    int x = gid % Mx;
    int tmp = gid / Mx;
    int r = tmp % R;
    int k = tmp / R;

    out[(k*R + r)*Mx + x] =
        inp[((k + i0)*R + r)*Mx + x];
}



__kernel void slice_k_lastaxis_f(
    __global const float *inp,   // (d, ω, Ktot) Fortran
    __global float *out,         // (d, ω, Ki)   Fortran
    const int D,                 // number of detectors
    const int O,                 // number of rotations
    const int Ktot,              // total K
    const int i0,                // starting K index
    const int Ki,                // number of K to extract
    const int total              // = D * O * Ki
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    // Decompose gid assuming Fortran layout of output
    int d = gid % D;
    int tmp = gid / D;
    int o = tmp % O;
    int k = tmp / O;   // k in [0, Ki)

    // Input K index
    int kin = k + i0;

    // Fortran indexing
    int out_idx = d + D * (o + O * k);
    int in_idx  = d + D * (o + O * kin);

    out[out_idx] = inp[in_idx];
}



__kernel void scatter_k_lastaxis_f(
    __global float *dst,        // (d, ω, Ktot) Fortran
    __global const float *src,  // (d, ω, Ki)   Fortran
    const int D,
    const int O,
    const int Ktot,
    const int i0,
    const int Ki,
    const int total             // = D * O * Ki
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    int d = gid % D;
    int tmp = gid / D;
    int o = tmp % O;
    int k = tmp / O;

    int dst_k = k + i0;

    // 64 bit: dst (the full coefficient array) may exceed 2^31 elements
    size_t src_idx = d + (size_t)D * (o + (size_t)O * k);
    size_t dst_idx = d + (size_t)D * (o + (size_t)O * dst_k);

    dst[dst_idx] = src[src_idx];
}



__kernel void SLICE_COEFFS_K_BATCH(
    __global const float *COEFFS_IN,   // (R, Mx, K_IN) C-order
    __global float *COEFFS_OUT,        // (R, Mx, K_OUT) C-order
    const int R,
    const int Mx,
    const int K_IN,
    const int K_OUT,
    const int K_START
){
    int gid = get_global_id(0);
    int total = R * Mx * K_OUT;
    if (gid >= total) return;

    int k = gid % K_OUT;
    int tmp = gid / K_OUT;
    int x = tmp % Mx;
    int r = tmp / Mx;

    int kin = k + K_START;

    // C-order flatten:
    // in_idx  = ((r*Mx + x)*K_IN  + kin)
    // out_idx = ((r*Mx + x)*K_OUT + k)
    COEFFS_OUT[(r * Mx + x) * K_OUT + k] =
        COEFFS_IN[(r * Mx + x) * K_IN + kin];
}

__kernel void SLICE_COEFFS_K_BATCH_F(
    __global const float *COEFFS_IN,   // (Nx, Ny, K_IN), Fortran order
    __global float *COEFFS_OUT,        // (Nx, Ny, K_OUT), Fortran order
    const int Nx,
    const int Ny,
    const int K_IN,
    const int K_OUT,
    const int K_START
){
    int gid = get_global_id(0);

    int plane_size = Nx * Ny;
    int total = plane_size * K_OUT;
    if (gid >= total) return;

    int k = gid / plane_size;
    int rem = gid % plane_size;

    int kin = k + K_START;

    // Fortran-order flattening, idx = i + Nx * (j + Ny * k); 64 bit, as COEFFS_IN may exceed 2^31 elements
    COEFFS_OUT[(size_t)rem + (size_t)plane_size * k] =
        COEFFS_IN[(size_t)rem + (size_t)plane_size * kin];
}



__kernel void SLICE_GRIDINV_K_BATCH(
    __global const float *GRIDINV_IN,  // (K_IN, 9) C-order
    __global float *GRIDINV_OUT,       // (K_OUT, 9) C-order
    const int K_IN,
    const int K_OUT,
    const int K_START
){
    int gid = get_global_id(0);
    int total = K_OUT * 9;
    if (gid >= total) return;

    int j = gid % 9;
    int k = gid / 9;

    int kin = k + K_START;

    GRIDINV_OUT[k * 9 + j] = GRIDINV_IN[kin * 9 + j];
}


__kernel void SCALE_PF_BY_INTENSITY_INPLACE(
    __global float *PF,                 // (R, Kb, C, P) C-order
    __global const float *INTENSITY,    // (P,)
    const int R,
    const int Kb,
    const int C,
    const int P
){
    int gid = get_global_id(0);
    int total = R * Kb * C * P;
    if (gid >= total) return;

    int p = gid % P;
    // r,k,c are not needed explicitly; just scale by p
    PF[gid] *= INTENSITY[p];
}




// ---------------------------------------------------------------------------
// Sparse PF matrix
//
// The dense PF batch pf[r, k, j] (layout (R, Kmax, CP), j = c*P + p) is compacted
// into two CSR structures per omega r:
//   forward rows (r, j): the orientations k with pf[r, k, j] != 0
//   adjoint rows (r, k): the segments j with pf[r, k, j] != 0
// so that neither multiplication needs atomics. The count kernels give the row
// lengths; the row pointers are their prefix sums. Column indices are 16 bit
// (the operator checks that Kmax and CP are below 65536).
// ---------------------------------------------------------------------------

__kernel void pf_count_rows_fwd(
    __global const float *pf,    // (R, Kmax, CP)
    __global int *counts,        // (R*CP,)
    const int R, const int Kmax, const int CP, const int Kb
){
    int row = get_global_id(0);
    if (row >= R * CP) return;
    int r = row / CP;
    int j = row % CP;
    int n = 0;
    for (int k = 0; k < Kb; ++k)
        if (pf[(r * Kmax + k) * CP + j] != 0.0f) n++;
    counts[row] = n;
}

__kernel void pf_fill_rows_fwd(
    __global const float *pf,       // (R, Kmax, CP)
    __global const int *row_ptr,    // (R*CP + 1,)
    __global ushort *col_k,         // (nnz,)
    __global float *val,            // (nnz,)
    const int R, const int Kmax, const int CP, const int Kb
){
    int row = get_global_id(0);
    if (row >= R * CP) return;
    int r = row / CP;
    int j = row % CP;
    int i = row_ptr[row];
    for (int k = 0; k < Kb; ++k) {
        float v = pf[(r * Kmax + k) * CP + j];
        if (v != 0.0f) { col_k[i] = (ushort)k; val[i] = v; i++; }
    }
}

__kernel void pf_count_rows_adj(
    __global const float *pf,    // (R, Kmax, CP)
    __global int *counts,        // (R*Kb,)
    const int R, const int Kmax, const int CP, const int Kb
){
    int row = get_global_id(0);
    if (row >= R * Kb) return;
    int r = row / Kb;
    int k = row % Kb;
    int base = (r * Kmax + k) * CP;
    int n = 0;
    for (int j = 0; j < CP; ++j)
        if (pf[base + j] != 0.0f) n++;
    counts[row] = n;
}

__kernel void pf_fill_rows_adj(
    __global const float *pf,       // (R, Kmax, CP)
    __global const int *row_ptr,    // (R*Kb + 1,)
    __global ushort *col_j,         // (nnz,)
    __global float *val,            // (nnz,)
    const int R, const int Kmax, const int CP, const int Kb
){
    int row = get_global_id(0);
    if (row >= R * Kb) return;
    int r = row / Kb;
    int k = row % Kb;
    int base = (r * Kmax + k) * CP;
    int i = row_ptr[row];
    for (int j = 0; j < CP; ++j) {
        float v = pf[base + j];
        if (v != 0.0f) { col_j[i] = (ushort)j; val[i] = v; i++; }
    }
}

__kernel void scatter_k_batch_c(
    __global float *dst,        // (R, Mx, Ktot) C-order
    __global const float *src,  // (R, Mx, Kb)   C-order
    const int R,
    const int Mx,
    const int Ktot,
    const int k0,
    const int Kb,
    const int total             // = R * Mx * Kb
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    int k = gid % Kb;
    int tmp = gid / Kb;
    int x = tmp % Mx;
    int r = tmp / Mx;

    int dst_k = k0 + k;

    // src index in C-order (r, x, k)
    int src_idx = (r * Mx + x) * Kb + k;

    // dst index in C-order (r, x, dst_k)
    int dst_idx = (r * Mx + x) * Ktot + dst_k;

    dst[dst_idx] = src[src_idx];
}



// ---------------------------------------------------------------------------
// Sparse PF products for the channel-fastest sinogram layout (R, My, Kstride)
// C-order, used with diffractom's own projector (ParallelRadon).
//
// A work-group handles one omega r and SPMM_TY detector positions y (a tile) for get_local_size
// rows: every matrix entry is read once per tile, not once per y. The dense operand (the tile's
// lines of the sinogram or of the data) goes through local memory a chunk at a time; the rows
// are sorted by column, so every work-item keeps its position in its row from chunk to chunk.
// Each output is summed in row order, as a plain row-by-row product would.
// Range (round_up(rows, ls), ceil(My / SPMM_TY), R); array offsets are 64 bit.
// ---------------------------------------------------------------------------
#ifndef SPMM_TY
#define SPMM_TY 24
#endif

// data[r, y, j] += sum_k pf[r, k, j] * sino[r, y, k]   (forward rows (r, j), sorted by k)
__kernel void spmm_pf_forward_c(
    __global const float *sino,         // (R, My, Kstride)
    __global const int *row_ptr,        // (R*CP + 1,)
    __global const ushort *col_k,
    __global const float *val,
    __global float *data,               // (R, My, CP)
    const int R, const int My, const int CP, const int Kstride, const int Kb,
    const int KC, __local float *lsino  // SPMM_TY * KC floats
){
    const int lid = get_local_id(0), ls = get_local_size(0);
    const int j = get_group_id(0) * ls + lid;
    const int y0 = get_group_id(1) * SPMM_TY;
    const int r = get_group_id(2);
    const int ny = min(SPMM_TY, My - y0);
    const size_t line0 = (size_t)r * My + y0;
    int i = 0, e = 0;
    if (j < CP) { i = row_ptr[r * CP + j]; e = row_ptr[r * CP + j + 1]; }
    float acc[SPMM_TY];
    #pragma unroll
    for (int t = 0; t < SPMM_TY; ++t) acc[t] = 0.0f;
    for (int k0 = 0; k0 < Kb; k0 += KC) {
        const int kn = min(KC, Kb - k0);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int t = 0; t < ny; ++t) {
            __global const float *src = sino + (line0 + t) * Kstride + k0;
            for (int kk = lid; kk < kn; kk += ls) lsino[t * KC + kk] = src[kk];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int k1 = k0 + kn;
        for (; i < e; ++i) {
            const int k = col_k[i];
            if (k >= k1) break;
            const float v = val[i];
            #pragma unroll
            for (int t = 0; t < SPMM_TY; ++t)
                if (t < ny) acc[t] += v * lsino[t * KC + (k - k0)];
        }
    }
    if (j < CP) {
        #pragma unroll
        for (int t = 0; t < SPMM_TY; ++t)
            if (t < ny) data[(line0 + t) * CP + j] += acc[t];
    }
}

// sino[r, y, k] = alpha * sum_j pf[r, k, j] * data[r, y, j]   for k < Kb   (adjoint rows (r, k), sorted by j)
__kernel void spmm_pf_adjoint_c(
    __global const float *data,         // (R, My, CP)
    __global const int *row_ptr,        // (R*Kb + 1,)
    __global const ushort *col_j,
    __global const float *val,
    __global float *sino,               // (R, My, Kstride)
    const int R, const int My, const int Kb, const int CP, const int Kstride, const float alpha,
    const int JC, __local float *ldata  // SPMM_TY * JC floats
){
    const int lid = get_local_id(0), ls = get_local_size(0);
    const int k = get_group_id(0) * ls + lid;
    const int y0 = get_group_id(1) * SPMM_TY;
    const int r = get_group_id(2);
    const int ny = min(SPMM_TY, My - y0);
    const size_t line0 = (size_t)r * My + y0;
    int i = 0, e = 0;
    if (k < Kb) { i = row_ptr[r * Kb + k]; e = row_ptr[r * Kb + k + 1]; }
    float acc[SPMM_TY];
    #pragma unroll
    for (int t = 0; t < SPMM_TY; ++t) acc[t] = 0.0f;
    for (int j0 = 0; j0 < CP; j0 += JC) {
        const int jn = min(JC, CP - j0);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int t = 0; t < ny; ++t) {
            __global const float *src = data + (line0 + t) * CP + j0;
            for (int jj = lid; jj < jn; jj += ls) ldata[t * JC + jj] = src[jj];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int j1 = j0 + jn;
        for (; i < e; ++i) {
            const int j = col_j[i];
            if (j >= j1) break;
            const float v = val[i];
            #pragma unroll
            for (int t = 0; t < SPMM_TY; ++t)
                if (t < ny) acc[t] += v * ldata[t * JC + (j - j0)];
        }
    }
    if (k < Kb) {
        #pragma unroll
        for (int t = 0; t < SPMM_TY; ++t)
            if (t < ny) sino[(line0 + t) * Kstride + k] = alpha * acc[t];
    }
}
