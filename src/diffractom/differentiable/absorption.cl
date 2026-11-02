// Self-absorption on the GPU (absorption.py, AbsorptionOperator).
//
// Line integrals of mu along a direction d for every pixel, on a frame of M x M points turned to d (frame point
// (iw, iu) sits at image position c + (iu - cm) d + (iw - cm) p, p = (-dy, dx)): sample mu there (bilinear), sum
// along d (D(u) = sum_{u' >= u} - f(u) / 2), sample D back at the pixel centres (bilinear). The transposes are
// written as gathers with the same weights (no atomics). Directions are processed in sets: dirs (ND, 2).
// Pixel p = col + Nx row; arrays per direction are contiguous.
//
// Compile-time: NB, NTAP, TRAPEZOID (the projector's), SPMM_TY.

#ifndef NB
#define NB 3
#endif
#ifndef NTAP
#define NTAP 2
#endif
#ifndef SPMM_TY
#define SPMM_TY 24
#endif

inline float bw(const float a) { return fmax(0.0f, 1.0f - fabs(a)); }     // 1D bilinear weight

// frame point position (image coordinates) for direction d
inline float2 frame_pos(const float2 d, const int iu, const int iw, const float cm, const float cx, const float cy)
{
    const float u = (float)iu - cm, w = (float)iw - cm;
    return (float2)(cx + u * d.x - w * d.y, cy + u * d.y + w * d.x);
}

// frame coordinates (u, w) of pixel (col, row)
inline float2 frame_coords(const float2 d, const int col, const int row, const float cm, const float cx, const float cy)
{
    const float x = (float)col - cx, y = (float)row - cy;
    return (float2)(x * d.x + y * d.y + cm, -x * d.y + y * d.x + cm);
}

// fr[n, iw, iu] = mu at the frame point (bilinear, zero outside)          range (M * M, ND)
__kernel void frame_sample(__global const float *mu, __global const float2 *dirs, __global float *fr,
                           const int Nx, const int Ny, const int M, const int d_off)
{
    const int f = get_global_id(0), n = get_global_id(1);
    if (f >= M * M) return;
    const float cm = 0.5f * (M - 1), cx = 0.5f * (Nx - 1), cy = 0.5f * (Ny - 1);
    const float2 X = frame_pos(dirs[d_off + n], f % M, f / M, cm, cx, cy);
    const int x0 = (int)floor(X.x), y0 = (int)floor(X.y);
    float v = 0.0f;
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
            const int xi = x0 + dx, yi = y0 + dy;
            if (xi >= 0 && xi < Nx && yi >= 0 && yi < Ny) v += bw(X.x - xi) * bw(X.y - yi) * mu[yi * Nx + xi];
        }
    fr[(size_t)n * M * M + f] = v;
}

// D[n, iw, u] = sum_{u' >= u} fr[n, iw, u'] - fr[n, iw, u] / 2              range (M, ND)
__kernel void frame_cumsum(__global const float *fr, __global float *D, const int M)
{
    const int iw = get_global_id(0), n = get_global_id(1);
    if (iw >= M) return;
    const size_t base = ((size_t)n * M + iw) * M;
    float acc = 0.0f;
    for (int u = M - 1; u >= 0; --u) {
        const float v = fr[base + u];
        acc += v;
        D[base + u] = acc - 0.5f * v;
    }
}

// out[n, p] = D[n] at the pixel's frame coordinates (bilinear)              range (Npix, ND)
__kernel void frame_back(__global const float *D, __global const float2 *dirs, __global float *out,
                         const int Nx, const int Ny, const int M, const int d_off)
{
    const int p = get_global_id(0), n = get_global_id(1);
    if (p >= Nx * Ny) return;
    const float cm = 0.5f * (M - 1), cx = 0.5f * (Nx - 1), cy = 0.5f * (Ny - 1);
    const float2 U = frame_coords(dirs[d_off + n], p % Nx, p / Nx, cm, cx, cy);
    const int u0 = (int)floor(U.x), w0 = (int)floor(U.y);
    float v = 0.0f;
    for (int dw = 0; dw < 2; ++dw)
        for (int du = 0; du < 2; ++du) {
            const int ui = u0 + du, wi = w0 + dw;
            if (ui >= 0 && ui < M && wi >= 0 && wi < M) v += bw(U.x - ui) * bw(U.y - wi) * D[((size_t)n * M + wi) * M + ui];
        }
    out[(size_t)n * Nx * Ny + p] = v;
}

// transpose of frame_back: gD[n, f] = sum_p weight(p -> f) g[n, p]           range (M * M, ND)
__kernel void frame_back_T(__global const float *g, __global const float2 *dirs, __global float *gD,
                           const int Nx, const int Ny, const int M, const int d_off)
{
    const int f = get_global_id(0), n = get_global_id(1);
    if (f >= M * M) return;
    const int iu = f % M, iw = f / M;
    const float cm = 0.5f * (M - 1), cx = 0.5f * (Nx - 1), cy = 0.5f * (Ny - 1);
    const float2 d = dirs[d_off + n];
    const float2 X = frame_pos(d, iu, iw, cm, cx, cy);
    float v = 0.0f;
    for (int row = (int)ceil(X.y - 1.4143f); row <= (int)floor(X.y + 1.4143f); ++row) {
        if (row < 0 || row >= Ny) continue;
        for (int col = (int)ceil(X.x - 1.4143f); col <= (int)floor(X.x + 1.4143f); ++col) {
            if (col < 0 || col >= Nx) continue;
            const float2 U = frame_coords(d, col, row, cm, cx, cy);
            const int u0 = (int)floor(U.x), w0 = (int)floor(U.y);
            if ((iu == u0 || iu == u0 + 1) && (iw == w0 || iw == w0 + 1))
                v += bw(U.x - iu) * bw(U.y - iw) * g[(size_t)n * Nx * Ny + row * Nx + col];
        }
    }
    gD[(size_t)n * M * M + f] = v;
}

// transpose of frame_cumsum: gfr[u] = sum_{u' < u} gD[u'] + gD[u] / 2       range (M, ND)
__kernel void frame_cumsum_T(__global const float *gD, __global float *gfr, const int M)
{
    const int iw = get_global_id(0), n = get_global_id(1);
    if (iw >= M) return;
    const size_t base = ((size_t)n * M + iw) * M;
    float acc = 0.0f;
    for (int u = 0; u < M; ++u) {
        const float v = gD[base + u];
        gfr[base + u] = acc + 0.5f * v;
        acc += v;
    }
}

// transpose of frame_sample, summed over the ND directions: gmu[p] += sum_n sum_f weight(f -> p) gfr[n, f]
//                                                                             range (Npix)
__kernel void frame_sample_T(__global const float *gfr, __global const float2 *dirs, __global float *gmu,
                             const int Nx, const int Ny, const int M, const int ND, const int d_off)
{
    const int p = get_global_id(0);
    if (p >= Nx * Ny) return;
    const int col = p % Nx, row = p / Nx;
    const float cm = 0.5f * (M - 1), cx = 0.5f * (Nx - 1), cy = 0.5f * (Ny - 1);
    float v = 0.0f;
    for (int n = 0; n < ND; ++n) {
        const float2 d = dirs[d_off + n];
        const float2 U = frame_coords(d, col, row, cm, cx, cy);
        for (int iw = (int)ceil(U.y - 1.4143f); iw <= (int)floor(U.y + 1.4143f); ++iw) {
            if (iw < 0 || iw >= M) continue;
            for (int iu = (int)ceil(U.x - 1.4143f); iu <= (int)floor(U.x + 1.4143f); ++iu) {
                if (iu < 0 || iu >= M) continue;
                const float2 X = frame_pos(d, iu, iw, cm, cx, cy);
                const int x0 = (int)floor(X.x), y0 = (int)floor(X.y);
                if ((col == x0 || col == x0 + 1) && (row == y0 || row == y0 + 1))
                    v += bw(X.x - col) * bw(X.y - row) * gfr[((size_t)n * M + iw) * M + iu];
            }
        }
    }
    gmu[p] += v;
}

// W[n, a, p] = exp(-(L[a, p] + fac[n] E[a, n, p])) for the angles a0 .. a0+Ra-1 of a chunk (L: (Ra, Npix), E: (Ra, NC,
// Npix), fac: the classes' length factors)                                  range (Npix, Ra, NC)
__kernel void att_weights(__global const float *L, __global const float *E, __global const float *fac,
                          __global float *W, const int Npix, const int R, const int a0, const int Ra, const int NC)
{
    const int p = get_global_id(0), a = get_global_id(1), n = get_global_id(2);
    if (p >= Npix || a >= Ra) return;
    const float v = L[(size_t)a * Npix + p] + fac[n] * E[((size_t)a * NC + n) * Npix + p];
    W[((size_t)n * R + a0 + a) * Npix + p] = exp(-v);
}

// ------------------------------------------------------------------ weighted projections (plain kernels)
// forward: sino[a, s, k] = scale sum_p K(t - s) W[a, p] img[p, k]   (radon_forward_k with the weight)
__kernel void radon_forward_w(
    __global const float4 *img, __global float4 *sino, __constant float4 *geo, __global const float *W,
    const ulong w_off, const int Nx, const int Ny, const int Ns, const int R, const int Kstride4, const int k4_off,
    const int k4_n, const float scale)
{
    if ((int)get_global_id(0) >= k4_n) return;
    const int k = get_global_id(0) + k4_off;
    const int s0 = get_global_id(1) * NB;
    const int a = get_global_id(2);
    if (k >= Kstride4 || s0 >= Ns || a >= R) return;
    const float4 g = geo[a];
    const size_t ks = (size_t)Kstride4;
    __global const float *Wa = W + w_off + (size_t)a * Nx * Ny;
    float4 acc[NB];
    #pragma unroll
    for (int j = 0; j < NB; ++j) acc[j] = (float4)(0.0f);
    const int rows_are_y = fabs(g.x) >= fabs(g.y);
    const float A = rows_are_y ? g.x : g.y;
    const float B = rows_are_y ? g.y : g.x;
    const int Nu = rows_are_y ? Nx : Ny;
    const int Nv = rows_are_y ? Ny : Nx;
    const float inv = 1.0f / A;
#ifdef TRAPEZOID
    const float hw = 0.5f * (1.0f + fabs(A)), top = fmin(1.0f, fabs(A)), iw = 1.0f / fabs(A);
#endif
    for (int v = 0; v < Nv; ++v) {
        const float c = B * v + g.z;
#ifdef TRAPEZOID
        const float lo = fmin((s0 - hw - c) * inv, (s0 + NB - 1 + hw - c) * inv);
        const float hi = fmax((s0 - hw - c) * inv, (s0 + NB - 1 + hw - c) * inv);
#else
        const float lo = fmin((s0 - 1.0f - c) * inv, (s0 + NB - c) * inv);
        const float hi = fmax((s0 - 1.0f - c) * inv, (s0 + NB - c) * inv);
#endif
        const int u0 = max((int)floor(lo), 0);
        const int u1 = min((int)ceil(hi), Nu - 1);
        for (int u = u0; u <= u1; ++u) {
            const int p = rows_are_y ? (u + Nx * v) : (v + Nx * u);
            const float4 val = Wa[p] * img[(size_t)p * ks + k];
            const float t = A * u + c - s0;
            #pragma unroll
#ifdef TRAPEZOID
            for (int j = 0; j < NB; ++j) acc[j] += (clamp(hw - fabs(t - j), 0.0f, top) * iw) * val;
#else
            for (int j = 0; j < NB; ++j) acc[j] += fmax(1.0f - fabs(t - j), 0.0f) * val;
#endif
        }
    }
    #pragma unroll
    for (int j = 0; j < NB; ++j)
        if (s0 + j < Ns) sino[((size_t)a * Ns + s0 + j) * ks + k] = scale * acc[j];
}

// backward: out[p, k] (+)= sum_a w_a W[a, p] sum_s K(t - s) sino[a, s, k]   (radon_backward_k with the weight)
__kernel void radon_backward_w(
    __global const float4 *sino, __global float4 *out, __constant float4 *geo, __global const float *W,
    const ulong w_off, const int Nx, const int Ny, const int Ns, const int R, const int Kstride4, const int k4_off,
    const int k4_n, const int accumulate)
{
    if ((int)get_global_id(0) >= k4_n) return;
    const int k = get_global_id(0) + k4_off;
    const int x = get_global_id(1);
    const int y = get_global_id(2);
    if (k >= Kstride4 || x >= Nx || y >= Ny) return;
    const size_t ks = (size_t)Kstride4;
    const int p = y * Nx + x;
    float4 acc = (float4)(0.0f);
    for (int a = 0; a < R; ++a) {
        const float4 g = geo[a];
        const float t = g.x * x + g.y * y + g.z;
        float4 v = (float4)(0.0f);
#ifdef TRAPEZOID
        const float Aa = fmax(fabs(g.x), fabs(g.y)), hw = 0.5f * (1.0f + Aa), top = fmin(1.0f, Aa), iw = 1.0f / Aa;
        const int sf = (int)floor(t - hw) + 1;
        #pragma unroll
        for (int j = 0; j < NTAP; ++j) {
            const int sj = sf + j;
            if (sj >= 0 && sj < Ns) v += (clamp(hw - fabs(t - sj), 0.0f, top) * iw) * sino[((size_t)a * Ns + sj) * ks + k];
        }
#else
        const int sm = (int)floor(t);
        const int sp = sm + 1;
        if (sm >= 0 && sm < Ns) v += (1.0f - (t - sm)) * sino[((size_t)a * Ns + sm) * ks + k];
        if (sp >= 0 && sp < Ns) v += (1.0f - (sp - t)) * sino[((size_t)a * Ns + sp) * ks + k];
#endif
        acc += (g.w * W[w_off + (size_t)a * Nx * Ny + p]) * v;
    }
    const size_t o = (size_t)p * ks + k;
    out[o] = accumulate ? out[o] + acc : acc;
}

// Q[a, p] += scale sum_s K(t_a(p) - s) sum_k img[p, k] Z[a, s, k]   (the derivative of the data with respect to the
// attenuation weight, contracted with the residual; channels beyond the batch are zero in img)   range (Npix, R)
__kernel void contract_w(__global const float4 *img, __global const float4 *Z, __constant float4 *geo,
                         __global float *Q, const ulong q_off, const int Nx, const int Ny, const int Ns, const int R,
                         const int Kstride4, const float scale)
{
    const int p = get_global_id(0), a = get_global_id(1);
    if (p >= Nx * Ny || a >= R) return;
    const int x = p % Nx, y = p / Nx;
    const float4 g = geo[a];
    const float t = g.x * x + g.y * y + g.z;
    const size_t ks = (size_t)Kstride4;
    float acc = 0.0f;
#ifdef TRAPEZOID
    const float Aa = fmax(fabs(g.x), fabs(g.y)), hw = 0.5f * (1.0f + Aa), top = fmin(1.0f, Aa), iw = 1.0f / Aa;
    const int sf = (int)floor(t - hw) + 1;
    for (int j = 0; j < NTAP; ++j) {
        const int sj = sf + j;
        if (sj < 0 || sj >= Ns) continue;
        const float kw = clamp(hw - fabs(t - sj), 0.0f, top) * iw;
#else
    const int sm = (int)floor(t);
    for (int j = 0; j < 2; ++j) {
        const int sj = sm + j;
        if (sj < 0 || sj >= Ns) continue;
        const float kw = 1.0f - fabs(t - sj);
#endif
        if (kw <= 0.0f) continue;
        float4 dsum = (float4)(0.0f);
        for (int k = 0; k < Kstride4; ++k) dsum += img[(size_t)p * ks + k] * Z[((size_t)a * Ns + sj) * ks + k];
        acc += kw * (dsum.x + dsum.y + dsum.z + dsum.w);
    }
    Q[q_off + (size_t)a * Nx * Ny + p] += scale * acc;
}

// ------------------------------------------------------------------ PF products restricted to segments (mask)
__kernel void spmm_forward_masked(
    __global const float *sino, __global const int *row_ptr, __global const ushort *col_k, __global const float *val,
    __global float *data, __global const uchar *jmask, const int R, const int My, const int CP, const int Kstride,
    const int Kb, const int KC, __local float *lsino)
{
    const int lid = get_local_id(0), ls = get_local_size(0);
    const int j = get_group_id(0) * ls + lid;
    const int y0 = get_group_id(1) * SPMM_TY;
    const int r = get_group_id(2);
    const int ny = min(SPMM_TY, My - y0);
    const size_t line0 = (size_t)r * My + y0;
    int i = 0, e = 0;
    if (j < CP && jmask[j]) { i = row_ptr[r * CP + j]; e = row_ptr[r * CP + j + 1]; }
    const int nonempty = i < e;
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
    if (j < CP && nonempty) {
        #pragma unroll
        for (int t = 0; t < SPMM_TY; ++t)
            if (t < ny) data[(line0 + t) * CP + j] += acc[t];
    }
}

__kernel void spmm_adjoint_masked(
    __global const float *data, __global const int *row_ptr, __global const ushort *col_j, __global const float *val,
    __global float *sino, __global const uchar *jmask, const int R, const int My, const int Kb, const int CP,
    const int Kstride, const float alpha, const int JC, __local float *ldata)
{
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
            if (!jmask[j]) continue;
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

// g[n, p] = s * a[n, p] * b[n, p]                                            range (n_elements)
__kernel void scaled_product(__global const float *a, __global const float *b, __global float *g, const float s,
                             const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) g[i] = s * a[i] * b[i];
}

// gl[a, p] = sum_n gw[n, a0 + a, p];  ge[a, n, p] = fac[n] gw[n, a0 + a, p]   (the chunk's transposed-integral inputs)
//                                                                             range (Npix, Ra)
__kernel void split_gw(__global const float *gw, __global const float *fac, __global float *gl, __global float *ge,
                       const int Npix, const int R, const int a0, const int Ra, const int NC)
{
    const int p = get_global_id(0), a = get_global_id(1);
    if (p >= Npix || a >= Ra) return;
    float s = 0.0f;
    for (int n = 0; n < NC; ++n) {
        const float v = gw[((size_t)n * R + a0 + a) * Npix + p];
        s += v;
        ge[((size_t)a * NC + n) * Npix + p] = fac[n] * v;
    }
    gl[(size_t)a * Npix + p] = s;
}
