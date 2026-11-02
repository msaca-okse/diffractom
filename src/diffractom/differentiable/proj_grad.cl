// Derivatives of the projections with respect to a detector shift (docs/differentiable_model.md, section 3):
//     dS[a, s, k] = scale * sum_x K'(t_a(x) - s) img[x, k]
// with the geometry and kernel of ParallelRadon (radon_kernels.cl, radon_forward_k with K' in place of K), and the
// per-angle sums  sum_{s, k < Kb} Z[a, s, k] dS[a, s, k]  (Z = PF^T r of the batch).
//
// Compile-time: NB (bins per work item), TRAPEZOID (else the hat kernel).

#ifndef NB
#define NB 3
#endif

__kernel void radon_dforward_k(
    __global const float4 *img,          // (Npix, Kstride4)
    __global float4 *sino,               // (R, Ns, Kstride4)
    __constant float4 *geo,
    const int Nx, const int Ny, const int Ns, const int R, const int Kstride4, const int k4_n, const float scale)
{
    if ((int)get_global_id(0) >= k4_n) return;
    const int k = get_global_id(0);
    const int s0 = get_global_id(1) * NB;
    const int a = get_global_id(2);
    if (s0 >= Ns || a >= R) return;
    const float4 g = geo[a];
    const size_t ks = (size_t)Kstride4;
    float4 acc[NB];
    for (int j = 0; j < NB; ++j) acc[j] = (float4)(0.0f);
    const int rows_are_y = fabs(g.x) >= fabs(g.y);
    const float A = rows_are_y ? g.x : g.y;
    const float B = rows_are_y ? g.y : g.x;
    const int Nu = rows_are_y ? Nx : Ny;
    const int Nv = rows_are_y ? Ny : Nx;
    const size_t du = rows_are_y ? ks : (size_t)Nx * ks;
    const size_t dv = rows_are_y ? (size_t)Nx * ks : ks;
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
        __global const float4 *row = img + (size_t)v * dv + k;
        for (int u = u0; u <= u1; ++u) {
            const float4 val = row[(size_t)u * du];
            const float t = A * u + c - s0;
            for (int j = 0; j < NB; ++j) {
                const float d = t - j;
#ifdef TRAPEZOID
                const float q = hw - fabs(d);
                const float w = (q > 0.0f && q < top) ? -copysign(iw, d) : 0.0f;
#else
                const float w = fabs(d) < 1.0f ? -copysign(1.0f, d) : 0.0f;
#endif
                acc[j] += w * val;
            }
        }
    }
    for (int j = 0; j < NB; ++j)
        if (s0 + j < Ns) sino[((size_t)a * Ns + s0 + j) * ks + k] = scale * acc[j];
}

// out[a] = sum_{s < Ns, k < Kb} Z[a, s, k] dS[a, s, k]; one work-group (LS) per angle
__kernel void shift_dot(__global const float *Z, __global const float *dS, __global float *out, const int Ns,
                        const int Kstride, const int Kb, __local float *red)
{
    const int a = get_group_id(0), lid = get_local_id(0), ls = get_local_size(0);
    float s = 0.0f;
    const size_t n = (size_t)Ns * Kb;
    for (size_t t = lid; t < n; t += ls) {
        const size_t sb = t / Kb, k = t - sb * Kb;
        const size_t i = ((size_t)a * Ns + sb) * Kstride + k;
        s += Z[i] * dS[i];
    }
    red[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int h = ls / 2; h > 0; h >>= 1) {
        if (lid < h) red[lid] += red[lid + h];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) out[a] = red[0];
}
