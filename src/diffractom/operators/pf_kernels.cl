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

    int src_idx = d + D * (o + O * k);
    int dst_idx = d + D * (o + O * dst_k);

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

    // Fortran-order flattening:
    // idx = i + Nx * (j + Ny * k)
    COEFFS_OUT[rem + plane_size * k] =
        COEFFS_IN[rem + plane_size * kin];
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

