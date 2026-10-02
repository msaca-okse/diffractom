// Fourier-slice parallel-beam projector (fourier_radon.py): the kernels around the FFTs.
//
// Images are (C, Ny, Nx) per channel slice (channel slowest), the oversampled grid (C, My, Mx) real
// and (C, My, Mx/2+1) complex (half spectrum), polar samples (C, R, H) complex (H = L/2 + 1
// frequencies per angle), detector lines (C, R, L) real. The operator's sinograms are
// (R, Ns, Kstride), channel fastest.

// Image slice -> oversampled grid: pixel (x, y) of channel c, divided by the deapodisation, at
// grid position ((x - cx) mod Mx, (y - cy) mod My); zero elsewhere and for c >= Cb.
__kernel void fr_pad(
    __global const float *img, const long k0,              // images k0 .. k0+Cb-1 of a (K, Ny, Nx) array
    __global float *grid,                                    // (C, My, Mx)
    __global const float *inv_dx, __global const float *inv_dy,
    const int Nx, const int Ny, const int Mx, const int My, const int cx, const int cy, const int Cb)
{
    const int gx = get_global_id(0), gy = get_global_id(1), c = get_global_id(2);
    if (gx >= Mx || gy >= My) return;
    const int x = (gx < Mx / 2 ? gx : gx - Mx) + cx;
    const int y = (gy < My / 2 ? gy : gy - My) + cy;
    float v = 0.0f;
    if (c < Cb && x >= 0 && x < Nx && y >= 0 && y < Ny)
        v = img[((size_t)(k0 + c) * Ny + y) * Nx + x] * inv_dx[x] * inv_dy[y];
    grid[((size_t)c * My + gy) * Mx + gx] = v;
}

// Polar samples from the half spectrum: spec[c, a, j] = filt[a, j] * sum over the W x W taps of
// w2[l] * w1[i] * G[k2, k1] (G[-k] = conj(G[k]) for k1 beyond the half). A work-item handles one
// sample for CC channels: the taps' positions and weights are computed once.
__kernel void fr_slices(
    __global const float2 *G,                                // (C, My, Mx/2+1)
    __global const int2 *kb,                                 // (R*H,): first tap (k1, k2) on the full grid
    __global const float *w1s, __global const float *w2s,    // (R*H, W) each
    __global const float2 *filt,                             // (R*H,)
    __global float2 *spec,                                   // (C, R*H)
    const int Mx, const int My, const int RH, const int H)
{
    const int smp = get_global_id(0), c0 = get_global_id(1) * CC;
    if (smp >= RH) return;
    const int Hx = Mx / 2 + 1;
    const size_t plane = (size_t)My * Hx;
    const int2 b = kb[smp];
    float w1[W], w2[W];
    int col[W], conj[W], rd[W], rm[W];
    #pragma unroll
    for (int i = 0; i < W; ++i) {
        w1[i] = w1s[(size_t)smp * W + i];
        w2[i] = w2s[(size_t)smp * W + i];
        int k1 = (b.x + i) % Mx; if (k1 < 0) k1 += Mx;
        conj[i] = k1 > Mx / 2;
        col[i] = conj[i] ? Mx - k1 : k1;
        int k2 = (b.y + i) % My; if (k2 < 0) k2 += My;
        rd[i] = k2;
        rm[i] = k2 == 0 ? 0 : My - k2;
    }
    float2 acc[CC];
    #pragma unroll
    for (int c = 0; c < CC; ++c) acc[c] = (float2)(0.0f);
    #pragma unroll
    for (int l = 0; l < W; ++l) {
        #pragma unroll
        for (int i = 0; i < W; ++i) {
            const float w = w2[l] * w1[i];
            const size_t idx = (size_t)(conj[i] ? rm[l] : rd[l]) * Hx + col[i];
            const float sg = conj[i] ? -1.0f : 1.0f;
            #pragma unroll
            for (int c = 0; c < CC; ++c) {
                const float2 g = G[(c0 + c) * plane + idx];
                acc[c] += w * (float2)(g.x, sg * g.y);
            }
        }
    }
    const float2 f = filt[smp];
    const int nyq = (smp % H == H - 1);   // the Nyquist frequency w = pi: its real part (the symmetric
                                          // sum over j; C2R FFTs differ in how they treat the rest)
    #pragma unroll
    for (int c = 0; c < CC; ++c) {
        float2 out = (float2)(acc[c].x * f.x - acc[c].y * f.y, acc[c].x * f.y + acc[c].y * f.x);
        if (nyq) out.y = 0.0f;
        spec[(size_t)(c0 + c) * RH + smp] = out;
    }
}

// Detector lines -> the operator's sinogram: sino[(a*Ns + s)*Kstride + kofs + c] = scale * line[c, a, s].
// A 32 x 32 tile (bins s x channels c) of one angle goes through local memory (both sides coalesced).
__kernel void fr_lines_out(
    __global const float *lines, __global float *sino,
    const int R, const int Ns, const int L, const int Kstride, const int kofs, const int Cb, const float scale)
{
    __local float t[32][33];
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int s0 = get_group_id(0) * 32, c0 = get_group_id(1) * 32, a = get_group_id(2);
    for (int r = ly; r < 32; r += get_local_size(1)) {      // read: bins contiguous
        const int c = c0 + r, s = s0 + lx;
        t[r][lx] = (c < Cb && s < Ns) ? lines[((size_t)c * R + a) * L + s] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int r = ly; r < 32; r += get_local_size(1)) {      // write: channels contiguous
        const int s = s0 + r, c = c0 + lx;
        if (s < Ns && c < Cb) sino[((size_t)a * Ns + s) * Kstride + kofs + c] = scale * t[lx][r];
    }
}

// Zero channels k0 .. Kstride-1 of a sinogram (the padding of a batch, as the native gather does).
__kernel void fr_zero_channels(__global float *sino, const int R, const int Ns, const int Kstride, const int k0)
{
    const int k = k0 + get_global_id(0), line = get_global_id(1);
    if (k >= Kstride || line >= R * Ns) return;
    sino[(size_t)line * Kstride + k] = 0.0f;
}

// The operator's sinogram -> weighted detector lines, zero-padded to L: line[c, a, s] = w_a * sino[...]
// (zero for c >= Cb), through 32 x 32 tiles in local memory.
__kernel void fr_lines_in(
    __global const float *sino, __global float *lines, __constant float *wa,
    const int R, const int Ns, const int L, const int Kstride, const int kofs, const int Cb, const int C)
{
    __local float t[32][33];
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int s0 = get_group_id(0) * 32, c0 = get_group_id(1) * 32, a = get_group_id(2);
    for (int r = ly; r < 32; r += get_local_size(1)) {      // read: channels contiguous
        const int s = s0 + r, c = c0 + lx;
        t[r][lx] = (s < Ns && c < Cb) ? wa[a] * sino[((size_t)a * Ns + s) * Kstride + kofs + c] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int r = ly; r < 32; r += get_local_size(1)) {      // write: bins contiguous
        const int c = c0 + r, s = s0 + lx;
        if (c < C && s < L) lines[((size_t)c * R + a) * L + s] = t[lx][r];
    }
}

// spec[c, smp] *= conj(filt[smp]) (the adjoint of the phase and bin response).
__kernel void fr_unfilter(__global float2 *spec, __global const float2 *filt, const int RH)
{
    const int smp = get_global_id(0), c = get_global_id(1);
    if (smp >= RH) return;
    const float2 f = filt[smp];
    const size_t i = (size_t)c * RH + smp;
    const float2 v = spec[i];
    spec[i] = (float2)(v.x * f.x + v.y * f.y, v.y * f.x - v.x * f.y);
}

// The adjoint of the interpolation, as a gather: a work-group owns a TS x TS tile of the half grid
// (rows k2, columns k1 <= Mx/2) and goes through the polar samples whose taps reach it (tile_ptr,
// tile_smp: sample index, or -(index + 1) for taps reaching it through the Hermitian mirror -k).
// Every work-item adds, for its cell q, half * w2[l] * w1[i] * spec (conjugated for the mirror) of
// every tap (l, i) of a sample that lands on q, in the order of the list: deterministic, and the
// same float32 weights as fr_slices. half = cj / 2 (cj = 1 for j = 0 and L/2, else 2).
#define SB 64   // samples staged in local memory at a time
__kernel void fr_spread(
    __global const float2 *spec,                             // (C, RH)
    __global const int *tile_ptr, __global const int *tile_smp,
    __global const int2 *kb, __global const float *w1s, __global const float *w2s,
    __global float2 *Q,                                      // (C, My, Mx/2+1)
    const int Mx, const int My, const int RH, const int H, const int n_tx, const int C)
{
    __local int2 lkb[SB];
    __local float lw1[SB][W], lw2[SB][W], lhalf[SB];
    __local int lmir[SB];
    __local float2 lv[CCS][SB];
    const int tile = get_group_id(0), cg = get_group_id(1) * CCS;
    const int lx = get_local_id(0), ly = get_local_id(1), lid = ly * TS + lx;
    const int Hx = Mx / 2 + 1;
    const int q1 = (tile % n_tx) * TS + lx, q2 = (tile / n_tx) * TS + ly;   // this work-item's cell
    float2 acc[CCS];
    #pragma unroll
    for (int c = 0; c < CCS; ++c) acc[c] = (float2)(0.0f);
    const int e0 = tile_ptr[tile], e1 = tile_ptr[tile + 1];
    for (int b0 = e0; b0 < e1; b0 += SB) {
        const int nb = min(SB, e1 - b0);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < nb) {
            const int sm = tile_smp[b0 + lid];
            const int smp = sm >= 0 ? sm : -sm - 1;
            lmir[lid] = sm < 0;
            lkb[lid] = kb[smp];
            const int j = smp % H;
            lhalf[lid] = (j == 0 || j == H - 1) ? 0.5f : 1.0f;
            for (int i = 0; i < W; ++i) { lw1[lid][i] = w1s[(size_t)smp * W + i]; lw2[lid][i] = w2s[(size_t)smp * W + i]; }
            for (int c = 0; c < CCS; ++c)
                lv[c][lid] = (cg + c < C) ? spec[(size_t)(cg + c) * RH + smp] : (float2)(0.0f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (q1 < Hx && q2 < My) {
            for (int t = 0; t < nb; ++t) {
                const int2 b = lkb[t];
                // the tap position of this cell: q itself, or -q for the mirror
                const int p1 = lmir[t] ? -q1 : q1, p2 = lmir[t] ? -q2 : q2;
                int d1 = (p1 - b.x) % Mx; if (d1 < 0) d1 += Mx;
                int d2 = (p2 - b.y) % My; if (d2 < 0) d2 += My;
                if (d1 < W && d2 < W) {
                    const float w = (lw2[t][d2] * lw1[t][d1]) * lhalf[t];
                    const float sg = lmir[t] ? -1.0f : 1.0f;
                    #pragma unroll
                    for (int c = 0; c < CCS; ++c) acc[c] += w * (float2)(lv[c][t].x, sg * lv[c][t].y);
                }
            }
        }
    }
    if (q1 < Hx && q2 < My) {
        #pragma unroll
        for (int c = 0; c < CCS; ++c)
            if (cg + c < C) Q[((size_t)(cg + c) * My + q2) * Hx + q1] = acc[c];
    }
}

// Oversampled grid -> images (the adjoint of fr_pad): target[(k0 + c), y, x] = grid / deapodisation.
__kernel void fr_crop(
    __global const float *grid, __global float *img, const long k0,
    __global const float *inv_dx, __global const float *inv_dy,
    const int Nx, const int Ny, const int Mx, const int My, const int cx, const int cy, const int Cb)
{
    const int x = get_global_id(0), y = get_global_id(1), c = get_global_id(2);
    if (x >= Nx || y >= Ny || c >= Cb) return;
    int gx = (x - cx) % Mx; if (gx < 0) gx += Mx;
    int gy = (y - cy) % My; if (gy < 0) gy += My;
    img[((size_t)(k0 + c) * Ny + y) * Nx + x] = grid[((size_t)c * My + gy) * Mx + gx] * inv_dx[x] * inv_dy[y];
}
