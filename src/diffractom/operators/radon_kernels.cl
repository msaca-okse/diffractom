// Parallel-beam Radon transform, vectorised over a channel axis k (the orientations).
//
// Geometry, one entry per projection angle a (geo[a] = (X, Y, T0, w)): pixel (x, y)
// lands at detector coordinate
//     t = X*x + Y*y + T0                                    (in detector bins)
// and contributes to detector bin s with the hat weight max(0, 1 - |t - s|):
//     forward:   sino[a, s, k] = scale * sum_{x, y} hat(t - s) * img[x, y, k]
//     backward:  img[x, y, k]  = sum_a w[a] * sum_s hat(t - s) * sino[a, s, k]
// i.e. the backward is the transpose of the forward, weighted by w[a] / scale.
//
// k is the fastest axis of both img and sino. All channels share the geometry, so each
// work item handles 4 consecutive channels (float4): the hat weights are computed once
// for 4 channels, the work items of a warp follow the same control flow, and every image
// or sinogram access is contiguous. The forward projection computes NB neighbouring
// detector bins per work item, so each pixel near the strip is loaded once for all the
// bins it touches.
//
// Layouts (Kstride a multiple of 4, Kstride4 = Kstride / 4):
//     img  (Npix, Kstride), C-order, pixel index p = x + Nx*y
//     sino (R, Ns, Kstride), C-order

#ifndef NB
#define NB 3
#endif
#define TILE 32

__kernel void radon_forward_k(
    __global const float4 *img,          // (Npix, Kstride4)
    __global float4 *sino,               // (R, Ns, Kstride4)
    __constant float4 *geo,
    const int Nx, const int Ny, const int Ns, const int R, const int Kstride4,
    const float scale
){
    const int k = get_global_id(0);
    const int s0 = get_global_id(1) * NB;
    const int a = get_global_id(2);
    if (k >= Kstride4 || s0 >= Ns || a >= R) return;

    const float4 g = geo[a];
    const size_t ks = (size_t)Kstride4;
    float4 acc[NB];
    #pragma unroll
    for (int j = 0; j < NB; ++j) acc[j] = (float4)(0.0f);

    const int rows_are_y = fabs(g.x) >= fabs(g.y);
    const float A = rows_are_y ? g.x : g.y;
    const float B = rows_are_y ? g.y : g.x;
    const int Nu = rows_are_y ? Nx : Ny;
    const int Nv = rows_are_y ? Ny : Nx;
    const size_t du = rows_are_y ? ks : (size_t)Nx * ks;
    const size_t dv = rows_are_y ? (size_t)Nx * ks : ks;
    const float inv = 1.0f / A;

    for (int v = 0; v < Nv; ++v) {
        const float c = B * v + g.z;
        const float lo = fmin((s0 - 1.0f - c) * inv, (s0 + NB - c) * inv);
        const float hi = fmax((s0 - 1.0f - c) * inv, (s0 + NB - c) * inv);
        const int u0 = max((int)floor(lo), 0);
        const int u1 = min((int)ceil(hi), Nu - 1);
        __global const float4 *row = img + (size_t)v * dv + k;
        for (int u = u0; u <= u1; ++u) {
            const float4 val = row[(size_t)u * du];
            const float t = A * u + c - s0;
            #pragma unroll
            for (int j = 0; j < NB; ++j) acc[j] += fmax(1.0f - fabs(t - j), 0.0f) * val;
        }
    }
    #pragma unroll
    for (int j = 0; j < NB; ++j)
        if (s0 + j < Ns) sino[((size_t)a * Ns + s0 + j) * ks + k] = scale * acc[j];
}


// Backward projection of 4 channels per work item, into a channel-fastest
// (Npix, Kstride4) float4 array (the caller scatters it to the (K, Ny, Nx) coefficients).
__kernel void radon_backward_k(
    __global const float4 *sino,         // (R, Ns, Kstride4)
    __global float4 *out,                // (Npix, Kstride4)
    __constant float4 *geo,
    const int Nx, const int Ny, const int Ns, const int R, const int Kstride4
){
    const int k = get_global_id(0);
    const int x = get_global_id(1);
    const int y = get_global_id(2);
    if (k >= Kstride4 || x >= Nx || y >= Ny) return;

    const size_t ks = (size_t)Kstride4;
    float4 acc = (float4)(0.0f);
    for (int a = 0; a < R; ++a) {
        const float4 g = geo[a];
        const float t = g.x * x + g.y * y + g.z;
        const int sm = (int)floor(t);
        const int sp = sm + 1;
        float4 v = (float4)(0.0f);
        if (sm >= 0 && sm < Ns) v += (1.0f - (t - sm)) * sino[((size_t)a * Ns + sm) * ks + k];
        if (sp >= 0 && sp < Ns) v += (1.0f - (sp - t)) * sino[((size_t)a * Ns + sp) * ks + k];
        acc += g.w * v;
    }
    out[((size_t)y * Nx + x) * ks + k] = acc;
}


// Scatter channels k < Kb of a channel-fastest (Npix, Kstride) array into a
// (Ktot, Npix) C-order array (the coefficients (K, Ny, Nx)) at channel offset k0 (the inverse of
// gather_channels_k_fastest), tiled through local memory.
__kernel void scatter_channels_k_fastest(
    __global const float *inp,
    __global float *out,
    const int Npix, const int k0, const int Kb, const int Kstride
){
    __local float tile[TILE][TILE + 1];
    const int p0 = get_group_id(0) * TILE;
    const int q0 = get_group_id(1) * TILE;
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int ny = get_local_size(1);

    for (int j = ly; j < TILE; j += ny) {        // read: channels contiguous
        const int p = p0 + j;
        const int q = q0 + lx;
        tile[j][lx] = (p < Npix && q < Kstride) ? inp[(size_t)p * Kstride + q] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int j = ly; j < TILE; j += ny) {        // write: pixels contiguous
        const int p = p0 + lx;
        const int q = q0 + j;
        if (p < Npix && q < Kb) out[(size_t)p + (size_t)Npix * (size_t)(k0 + q)] = tile[lx][j];
    }
}


// Gather channels k0 .. k0+Kb-1 of a (Ktot, Npix) C-order array (the coefficients) into the
// channel-fastest (Npix, Kstride) layout; channels Kb .. Kstride-1 are set to zero.
// Tiled through local memory so that both the reads and the writes are contiguous.
__kernel void gather_channels_k_fastest(
    __global const float *inp,
    __global float *out,
    const int Npix, const int k0, const int Kb, const int Kstride
){
    __local float tile[TILE][TILE + 1];
    const int p0 = get_group_id(0) * TILE;
    const int q0 = get_group_id(1) * TILE;
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int ny = get_local_size(1);

    for (int j = ly; j < TILE; j += ny) {        // read: pixels contiguous
        const int p = p0 + lx;
        const int q = q0 + j;
        tile[j][lx] = (p < Npix && q < Kb) ? inp[(size_t)p + (size_t)Npix * (size_t)(k0 + q)] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int j = ly; j < TILE; j += ny) {        // write: channels contiguous
        const int p = p0 + j;
        const int q = q0 + lx;
        if (p < Npix && q < Kstride) out[(size_t)p * Kstride + q] = tile[lx][j];
    }
}
