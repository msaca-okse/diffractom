// Kernels of the local-basis operator (local_forward_operator.py).
//
// The coefficients are patches: patch p is an h x w box of the grid (rows y0.., columns x0..) of one orientation,
// stored row-major at coff[p] in a flat array, with a mask (1 where the orientation is allowed). At angle a a patch
// projects onto a strip of W[p] detector bins starting at off[p * R + a] (a margin on both sides). The strips of a
// batch of patches are stored at soff[p], (R, W[p]) row-major.
//
// Forward:  strip = scale * Radon(patch)           (local_radon_forward)
//           data[r, y, j] += sum_k pf[r, k, j] * strip_k[r, y - off]    (local_pf_forward)
// Adjoint:  strip_p[r, y - off] = alpha * sum_j pf[r, k_p, j] * data[r, y, j]   (local_pf_adjoint)
//           patch = backprojection of the strip, zero outside the mask   (local_radon_backward)
//
// The arithmetic and summation order of each output follow the global operator's kernels (radon_forward_k with NB
// bins per strip, radon_backward_k, spmm_pf_forward_c, spmm_pf_adjoint_c), so with one patch per orientation the
// results equal those of SinglePhaseForwardOperator on coefficients that are zero outside the support.
//
// The PF products work on tiles of TY detector lines at one angle: a work-group gets the list of patches whose
// strips reach into its tile (tile_ptr, tile_patch: CSR over r * n_tiles + tile, patches in ascending order).
// The PF matrix of the batch: the forward CSR (rows r * CP + j, columns k ascending) for the forward product, the
// adjoint CSR (rows r * Kb + k, columns j ascending) for the adjoint.

#ifndef NB
#define NB 3
#endif
#ifndef NTAP
#define NTAP 2
#endif
#ifndef TY
#define TY 8
#endif


__kernel void local_radon_forward(
    __global const float *coef,
    __global const uchar *mask,
    __global float *strips,
    __global const int4 *box,            // (y0, x0, h, w) of every patch of the batch
    __global const long *coff,
    __global const int *W,
    __global const int *soff,
    __global const int *off,             // (P, R)
    __constant float4 *geo,
    const int R, const float scale
){
    // work-group: get_local_size(1) angles of one patch, get_local_size(0) bins of each at a time
    const int AB = get_local_size(1);
    const int nab = (R + AB - 1) / AB;
    const int p = get_group_id(1) / nab;
    const int a = (get_group_id(1) - p * nab) * AB + get_local_id(1);
    const int4 bx = box[p];
    const int Wp = W[p];
    __global const float *cp = coef + coff[p];
    __global const uchar *mp = mask + coff[p];
    if (a < R) {
        const int o = off[(size_t)p * R + a];
        const float4 g = geo[a];
        const int rows_are_y = fabs(g.x) >= fabs(g.y);
        const float A = rows_are_y ? g.x : g.y;
        const float B = rows_are_y ? g.y : g.x;
        const int v_lo = rows_are_y ? bx.x : bx.y;      // rows of the projection: y (or x)
        const int v_n = rows_are_y ? bx.z : bx.w;
        const int u_lo = rows_are_y ? bx.y : bx.x;      // along a row: x (or y)
        const int u_hi = u_lo + (rows_are_y ? bx.w : bx.z) - 1;
        const int du = rows_are_y ? 1 : bx.w;
        const int dv = rows_are_y ? bx.w : 1;
        const float inv = 1.0f / A;
#ifdef TRAPEZOID
        const float hw = 0.5f * (1.0f + fabs(A)), top = fmin(1.0f, fabs(A)), iw = 1.0f / fabs(A);
#endif
        const float S = B * inv;                        // |S| <= 1: the u-range moves by -S per row
        __global float *sp = strips + soff[p] + (size_t)a * Wp;
        for (int j = get_local_id(0); j < Wp; j += get_local_size(0)) {
            const int s = o + j;
            const int s0 = (s >= 0 ? s / NB : -((NB - 1 - s) / NB)) * NB;     // the strip of NB bins holding s
            const int jj = s - s0;
            // the pixels with a non-zero weight for bin s: |A u + c - s| < hw (hat: < 1), c = B v + T0. Their range
            // per row, with a margin of 1e-3 pixel for rounding, and the rows where it meets the box (+- 1 row).
            // Pixels outside it would add exactly zero; the weights are those of the NB-bin strip (t from s0).
#ifdef TRAPEZOID
            const float hb = hw;
#else
            const float hb = 1.0f;
#endif
            const float e1 = (s - hb - g.z) * inv, e2 = (s + hb - g.z) * inv;
            const float Lmin = fmin(e1, e2) - 0.01f, Lmax = fmax(e1, e2) + 0.01f;
            int va = 0, vb = v_n - 1;
            if (fabs(S) > 1e-6f) {
                const float r1 = (Lmin - u_hi) / S, r2 = (Lmax - u_lo) / S;
                va = max(va, (int)floor(fmin(r1, r2)) - 1 - v_lo);
                vb = min(vb, (int)ceil(fmax(r1, r2)) + 1 - v_lo);
            } else if (Lmax < u_lo || Lmin > u_hi) {
                vb = -1;
            }
            float acc = 0.0f;
            for (int vv = va; vv <= vb; ++vv) {
                const int v = v_lo + vv;
                const float c = B * v + g.z;
                const float lo = fmin((s - hb - c) * inv, (s + hb - c) * inv) - 1e-3f;
                const float hi = fmax((s - hb - c) * inv, (s + hb - c) * inv) + 1e-3f;
                const int u0 = max((int)floor(lo), u_lo);
                const int u1 = min((int)ceil(hi), u_hi);
                for (int u = u0; u <= u1; ++u) {
                    const int q = vv * dv + (u - u_lo) * du;
                    const float val = mp[q] ? cp[q] : 0.0f;
                    const float t = A * u + c - s0;
#ifdef TRAPEZOID
                    acc += (clamp(hw - fabs(t - jj), 0.0f, top) * iw) * val;
#else
                    acc += fmax(1.0f - fabs(t - jj), 0.0f) * val;
#endif
                }
            }
            sp[j] = scale * acc;
        }
    }
}


__kernel void local_radon_backward(
    __global const float *strips,
    __global float *coef,                // written: the patches of the batch
    __global const uchar *mask,
    __global const int2 *blk,            // (patch, first pixel) of every work-group
    __global const int4 *box,
    __global const long *coff,
    __global const int *W,
    __global const int *soff,
    __global const int *off,
    __constant float4 *geo,
    const int R, const int Ns
){
    const int2 b = blk[get_group_id(0)];
    const int p = b.x;
    const int q = b.y + get_local_id(0);
    const int4 bx = box[p];
    if (q >= bx.z * bx.w) return;
    const size_t idx = coff[p] + q;
    if (!mask[idx]) { coef[idx] = 0.0f; return; }
    const int y = bx.x + q / bx.w, x = bx.y + q % bx.w;
    const int Wp = W[p];
    __global const float *sp = strips + soff[p];
    __global const int *op = off + (size_t)p * R;
    float acc = 0.0f;
    for (int a = 0; a < R; ++a) {
        const float4 g = geo[a];
        const float t = g.x * x + g.y * y + g.z;
        __global const float *sa = sp + (size_t)a * Wp - op[a];       // sa[s]: bin s of the strip
        float v = 0.0f;
#ifdef TRAPEZOID
        const float Aa = fmax(fabs(g.x), fabs(g.y)), hw = 0.5f * (1.0f + Aa), top = fmin(1.0f, Aa), iw = 1.0f / Aa;
        const int sf = (int)floor(t - hw) + 1;
        #pragma unroll
        for (int j = 0; j < NTAP; ++j) {
            const int sj = sf + j;
            if (sj >= 0 && sj < Ns) v += (clamp(hw - fabs(t - sj), 0.0f, top) * iw) * sa[sj];
        }
#else
        const int sm = (int)floor(t);
        const int sq = sm + 1;
        if (sm >= 0 && sm < Ns) v += (1.0f - (t - sm)) * sa[sm];
        if (sq >= 0 && sq < Ns) v += (1.0f - (sq - t)) * sa[sq];
#endif
        acc += g.w * v;
    }
    coef[idx] = acc;
}


// data[r, y, j] += sum_k pf[r, k, j] * sino[r, y, k] for the TY lines of a tile, as spmm_pf_forward_c: the tile's
// sinogram lines (TY x Kb, zero for orientations without a strip there) are assembled in local memory from the
// strips of the tile's patches (several patches of one orientation added in order), then a work-item per segment
// runs over its forward-CSR row (orientations ascending). Lines without a contribution are not written.
__kernel void local_pf_forward(
    __global const float *strips,
    __global const int *tile_ptr,
    __global const int *tile_patch,
    __global const int *pk,              // orientation of the patch within the PF batch
    __global const int *W,
    __global const int *soff,
    __global const int *off,
    __global const int *row_ptr,         // forward CSR of the batch (R * CP + 1), columns k ascending
    __global const ushort *col_k,
    __global const float *val,
    __global float *data,                // (R, My, CP)
    const int R, const int My, const int CP, const int Kb, const int n_tiles,
    __local float *tab                   // TY * Kb
){
    const int lid = get_local_id(0), ls = get_local_size(0);
    const int gid = get_group_id(0);
    const int r = gid / n_tiles, tile = gid - r * n_tiles;
    const int y0 = tile * TY, ny = min(TY, My - y0);
    const int i0 = tile_ptr[gid], i1 = tile_ptr[gid + 1];
    if (i0 == i1) return;
    for (int i = lid; i < TY * Kb; i += ls) tab[i] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = i0 + lid; i < i1; i += ls) {
        const int k = pk[tile_patch[i]];
        if (i > i0 && pk[tile_patch[i - 1]] == k) continue;     // added by the orientation's first patch here
        float v[TY];
        #pragma unroll
        for (int t = 0; t < TY; ++t) v[t] = 0.0f;
        for (int ii = i; ii < i1; ++ii) {
            const int p = tile_patch[ii];
            if (pk[p] != k) break;
            const int o = off[(size_t)p * R + r], Wp = W[p];
            const int tlo = max(0, o - y0), thi = min(ny, o + Wp - y0);
            __global const float *st = strips + soff[p] + (size_t)r * Wp + (y0 - o);   // st[t]: line y0 + t
            #pragma unroll
            for (int t = 0; t < TY; ++t)
                if (t >= tlo && t < thi) v[t] += st[t];
        }
        #pragma unroll
        for (int t = 0; t < TY; ++t)
            if (t < ny) tab[t * Kb + k] = v[t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int j = lid; j < CP; j += ls) {
        int e = row_ptr[r * CP + j];
        const int e1 = row_ptr[r * CP + j + 1];
        if (e == e1) continue;
        float acc[TY];
        #pragma unroll
        for (int t = 0; t < TY; ++t) acc[t] = 0.0f;
        for (; e < e1; ++e) {
            const int k = col_k[e];
            const float v = val[e];
            #pragma unroll
            for (int t = 0; t < TY; ++t)
                if (t < ny) acc[t] += v * tab[t * Kb + k];
        }
        #pragma unroll
        for (int t = 0; t < TY; ++t)
            if (t < ny && acc[t] != 0.0f) data[((size_t)r * My + y0 + t) * CP + j] += acc[t];
    }
}


static int lower_bound_u16(__global const ushort *a, int lo, int hi, const int v)
{
    while (lo < hi) {
        const int m = (lo + hi) >> 1;
        if ((int)a[m] < v) lo = m + 1; else hi = m;
    }
    return lo;
}


// strip_p[r, y] = alpha * sum_j pf[r, k_p, j] * data[r, y, j] for the lines y of the tile: the tile block of the
// data goes through local memory a chunk of JC segments at a time; a work-item per patch, partial sums kept in the
// strip between chunks (the same order of additions as one pass).
__kernel void local_pf_adjoint(
    __global const float *data,
    __global const int *tile_ptr,
    __global const int *tile_patch,
    __global const int *pk,
    __global const int *W,
    __global const int *soff,
    __global const int *off,
    __global const int *row_ptr,
    __global const ushort *col_j,
    __global const float *val,
    __global float *strips,
    const int R, const int My, const int CP, const int Kb, const int n_tiles, const int JC, const float alpha,
    __local float *ld                    // TY * JC
){
    const int lid = get_local_id(0), ls = get_local_size(0);
    const int gid = get_group_id(0);
    const int r = gid / n_tiles, tile = gid - r * n_tiles;
    const int y0 = tile * TY, ny = min(TY, My - y0);
    const int i0 = tile_ptr[gid], i1 = tile_ptr[gid + 1];
    if (i0 == i1) return;
    for (int j0 = 0; j0 < CP; j0 += JC) {
        const int jn = min(JC, CP - j0);
        const int last = j0 + jn >= CP;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int t = 0; t < ny; ++t) {
            __global const float *src = data + ((size_t)r * My + y0 + t) * CP + j0;
            for (int jj = lid; jj < jn; jj += ls) ld[t * JC + jj] = src[jj];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = i0 + lid; i < i1; i += ls) {
            const int p = tile_patch[i];
            const int o = off[(size_t)p * R + r], Wp = W[p];
            const int tlo = max(0, o - y0), thi = min(ny, o + Wp - y0);
            __global float *st = strips + soff[p] + (size_t)r * Wp + (y0 - o);
            float acc[TY];
            #pragma unroll
            for (int t = 0; t < TY; ++t) acc[t] = (j0 > 0 && t >= tlo && t < thi) ? st[t] : 0.0f;
            const int row = r * Kb + pk[p];
            const int e1 = row_ptr[row + 1];
            for (int e = lower_bound_u16(col_j, row_ptr[row], e1, j0); e < e1; ++e) {
                const int seg = (int)col_j[e] - j0;
                if (seg >= jn) break;
                const float v = val[e];
                #pragma unroll
                for (int t = 0; t < TY; ++t)
                    if (t >= tlo && t < thi) acc[t] += v * ld[t * JC + seg];
            }
            #pragma unroll
            for (int t = 0; t < TY; ++t)
                if (t >= tlo && t < thi) st[t] = last ? alpha * acc[t] : acc[t];
        }
    }
}
