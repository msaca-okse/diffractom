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
//
// The projections handle the channels k4_off .. k4_off + k4_n - 1 (in float4 units) of the
// arrays: a launch per slice of channels. Slices of 64-128 channels keep the work items that run
// together on the same image region, which is much faster than one launch over many channels.

#ifndef NB
#define NB 3
#endif
#define TILE 32

__kernel void radon_forward_k(
    __global const float4 *img,          // (Npix, Kstride4)
    __global float4 *sino,               // (R, Ns, Kstride4)
    __constant float4 *geo,
    const int Nx, const int Ny, const int Ns, const int R, const int Kstride4, const int k4_off, const int k4_n,
    const float scale
){
    if ((int)get_global_id(0) >= k4_n) return;
    const int k = get_global_id(0) + k4_off;
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
    const int Nx, const int Ny, const int Ns, const int R, const int Kstride4, const int k4_off, const int k4_n
){
    if ((int)get_global_id(0) >= k4_n) return;
    const int k = get_global_id(0) + k4_off;
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


// ---------------------------------------------------------------------------------------------
// Tiled projections. The kernels above read every image pixel (forward) or sinogram bin
// (backward) about twice per angle from the caches, which bounds them by cache bandwidth on large
// grids. The tiled kernels stage what a work-group shares in local memory and keep each output's
// arithmetic and summation order, so they give bitwise the same results.
// Compile-time constants: NB (bins per strip), FCQ, FSB, FAB, FVR (forward: channel quads,
// strips, angles per work-group, image rows per local tile), BCQ, BAB, BLMAX (backward: channel
// quads per work-group, angles per local tile, sinogram bins of an 8 x 8 pixel tile per angle).
// ---------------------------------------------------------------------------------------------

#ifdef FCQ   // the tiled kernels are compiled when the tiling constants are given

// Forward: a work-group handles FAB angles of one block (blocks: angles with the same row axis,
// see rows_are_y), FSB strips of NB bins and FCQ channel quads. For FVR image rows at a time, the
// pixel segment the group's angles and bins touch, W pixels from lbase[row], goes to local memory.
__kernel void radon_forward_tiled(
    __global const float4 *img,          // (Npix, Kstride4)
    __global float4 *sino,               // (R, Ns, Kstride4)
    __constant float4 *geo,
    __global const int2 *blocks,         // (first angle, number of angles) of every angle block
    const int Nx, const int Ny, const int Ns, const int R, const int Kstride4, const int k4_off, const int k4_n,
    const float scale, const int W,
    __local float4 *lt,                  // FVR * W * FCQ
    __local int *lbase                   // FVR
){
    const int q = get_local_id(0), sl = get_local_id(1), al = get_local_id(2);
    const int lid = (al * FSB + sl) * FCQ + q;
    const int nloc = FCQ * FSB * FAB;
    const int2 blk = blocks[get_group_id(2)];
    const int active_a = al < blk.y;
    const int a = blk.x + (active_a ? al : 0);
    const int kq = get_group_id(0) * FCQ + q;          // channel quad within the slice
    const int k = k4_off + kq;
    const int s0 = (get_group_id(1) * FSB + sl) * NB;
    const int S0 = get_group_id(1) * FSB * NB;         // the group's bins: S0 .. S1 - 1
    const int S1 = S0 + FSB * NB;
    const size_t ks = (size_t)Kstride4;

    const float4 gfirst = geo[blk.x];
    const int rows_are_y = fabs(gfirst.x) >= fabs(gfirst.y);   // the same for every angle of the block
    const int Nu = rows_are_y ? Nx : Ny;
    const int Nv = rows_are_y ? Ny : Nx;
    const size_t du = rows_are_y ? ks : (size_t)Nx * ks;
    const size_t dv = rows_are_y ? (size_t)Nx * ks : ks;

    const float4 g = geo[a];
    const float A = rows_are_y ? g.x : g.y;
    const float B = rows_are_y ? g.y : g.x;
    const float inv = 1.0f / A;
    const int work = active_a && s0 < Ns && kq < k4_n;

    float4 acc[NB];
    #pragma unroll
    for (int j = 0; j < NB; ++j) acc[j] = (float4)(0.0f);

    for (int v0 = 0; v0 < Nv; v0 += FVR) {
        const int nv = min(FVR, Nv - v0);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < nv) {                                 // first pixel of row v0 + lid any work-item needs
            const int v = v0 + lid;
            int lo = Nu;
            for (int b = 0; b < blk.y; ++b) {
                const float4 gb = geo[blk.x + b];
                const float Ab = rows_are_y ? gb.x : gb.y;
                const float Bb = rows_are_y ? gb.y : gb.x;
                const float ib = 1.0f / Ab;
                const float c = Bb * v + gb.z;
                const float l = fmin((S0 - 1.0f - c) * ib, (S1 - c) * ib);
                lo = min(lo, max((int)floor(l), 0));
            }
            lbase[lid] = lo - 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int kq0 = k4_off + get_group_id(0) * FCQ;
        for (int i = lid; i < nv * W * FCQ; i += nloc) {
            const int r = i / (W * FCQ), rem = i - r * (W * FCQ), w = rem / FCQ, qq = rem - w * FCQ;
            const int u = lbase[r] + w;
            const int kk = kq0 + qq;
            lt[i] = (u >= 0 && u < Nu && kk < k4_off + k4_n) ? img[(size_t)(v0 + r) * dv + (size_t)u * du + kk]
                                                             : (float4)(0.0f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (work) {
            for (int r = 0; r < nv; ++r) {
                const int v = v0 + r;
                // as in radon_forward_k
                const float c = B * v + g.z;
                const float lo = fmin((s0 - 1.0f - c) * inv, (s0 + NB - c) * inv);
                const float hi = fmax((s0 - 1.0f - c) * inv, (s0 + NB - c) * inv);
                const int u0 = max((int)floor(lo), 0);
                const int u1 = min((int)ceil(hi), Nu - 1);
                const int base = lbase[r];
                __global const float4 *row = img + (size_t)v * dv + k;
                for (int u = u0; u <= u1; ++u) {
                    const int w = u - base;
                    const float4 val = (w >= 0 && w < W) ? lt[(r * W + w) * FCQ + q] : row[(size_t)u * du];
                    const float t = A * u + c - s0;
                    #pragma unroll
                    for (int j = 0; j < NB; ++j) acc[j] += fmax(1.0f - fabs(t - j), 0.0f) * val;
                }
            }
        }
    }
    if (work) {
        #pragma unroll
        for (int j = 0; j < NB; ++j)
            if (s0 + j < Ns) sino[((size_t)a * Ns + s0 + j) * ks + k] = scale * acc[j];
    }
}


// Backward: a work-group handles an 8 x 8 pixel tile and BCQ channel quads. For BAB angles at a
// time, the sinogram bins the tile touches (BLMAX from lsmin[angle]) go to local memory.
#define BT 8
__kernel void radon_backward_tiled(
    __global const float4 *sino,         // (R, Ns, Kstride4)
    __global float4 *out,                // (Npix, Kstride4)
    __constant float4 *geo,
    const int Nx, const int Ny, const int Ns, const int R, const int Kstride4, const int k4_off, const int k4_n
){
    __local float4 ls[BAB][BLMAX][BCQ];
    __local int lsmin[BAB];
    const int q = get_local_id(0), lx = get_local_id(1), ly = get_local_id(2);
    const int lid = (ly * BT + lx) * BCQ + q;
    const int nloc = BCQ * BT * BT;
    const int kq = get_group_id(0) * BCQ + q;
    const int k = k4_off + kq;
    const int x0 = get_group_id(1) * BT, y0 = get_group_id(2) * BT;
    const int x = x0 + lx, y = y0 + ly;
    const int kq0 = k4_off + get_group_id(0) * BCQ;
    const size_t ks = (size_t)Kstride4;
    float4 acc = (float4)(0.0f);
    for (int a0 = 0; a0 < R; a0 += BAB) {
        const int na = min(BAB, R - a0);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < na) {                                 // lowest bin of the tile at angle a0 + lid
            const float4 g = geo[a0 + lid];
            const float xa = (float)x0, xb = (float)(x0 + BT - 1), ya = (float)y0, yb = (float)(y0 + BT - 1);
            const float t0 = g.x * xa + g.y * ya + g.z, t1 = g.x * xb + g.y * ya + g.z;
            const float t2 = g.x * xa + g.y * yb + g.z, t3 = g.x * xb + g.y * yb + g.z;
            lsmin[lid] = (int)floor(fmin(fmin(t0, t1), fmin(t2, t3))) - 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = lid; i < na * BLMAX * BCQ; i += nloc) {
            const int al = i / (BLMAX * BCQ), rem = i - al * (BLMAX * BCQ), l = rem / BCQ, qq = rem - l * BCQ;
            const int s = lsmin[al] + l;
            const int kk = kq0 + qq;
            ls[al][l][qq] = (s >= 0 && s < Ns && kk < k4_off + k4_n) ? sino[((size_t)(a0 + al) * Ns + s) * ks + kk]
                                                                     : (float4)(0.0f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (kq < k4_n) {
            for (int al = 0; al < na; ++al) {
                const int a = a0 + al;
                // as in radon_backward_k
                const float4 g = geo[a];
                const float t = g.x * x + g.y * y + g.z;
                const int sm = (int)floor(t);
                const int sp = sm + 1;
                const int lm = sm - lsmin[al], lp = sp - lsmin[al];
                float4 v = (float4)(0.0f);
                if (sm >= 0 && sm < Ns)
                    v += (1.0f - (t - sm)) * ((lm >= 0 && lm < BLMAX) ? ls[al][lm][q] : sino[((size_t)a * Ns + sm) * ks + k]);
                if (sp >= 0 && sp < Ns)
                    v += (1.0f - (sp - t)) * ((lp >= 0 && lp < BLMAX) ? ls[al][lp][q] : sino[((size_t)a * Ns + sp) * ks + k]);
                acc += g.w * v;
            }
        }
    }
    if (x < Nx && y < Ny && kq < k4_n) out[((size_t)y * Nx + x) * ks + k] = acc;
}

#endif  // FCQ
