// Sparse PF matrix generated directly, one orientation batch at a time, without the dense batch.
//
// The entry pf[r, k, j] (j = c*P + p) is the one of pfmatrix_eval_poles: a sum over the pole axes
// a of ring p of terms that are non-zero only where (1 - |n.v|) * inv_sigma2 < 6, n = poles[k, a],
// v the probed direction of (omega r, eta bin c, ring p). For a pole n at the omega sub-angle of
// (r, s), with the frame vectors rotated like the probed directions,
//     n.v(eta) = A cos(eta - phi) - B,   A = cos(th) |(n.d0, n.d90)|,  B = sin(th) n.p0,
// so |n.v| > tau holds on two eta intervals (around phi for n, around phi + pi for -n). The eta
// bins they touch are the candidates; tau is relaxed slightly, so they include every non-zero.
// Every candidate is then evaluated exactly as in pfmatrix_eval_poles (same operations, same
// order: the same float32 values), the zeros are dropped, and the rows are written in the order
// of the CSR built from the dense batch. The result is identical to it.
//
// Rows (r, k) of the adjoint CSR: row = r * Kb + k. Sizes that are only known on the device
// (candidates n = cand_ptr[n_rows], non-zeros nnz = row_ptr[n_rows]) are read there, so nothing
// is read back to the host.
//
// Compile-time constants: MAXWC = ceil(C / 32) words of eta bits per ring, PPW >= max(P, MAXWC)
// work-items per row, RPG rows per work-group (local size RPG * PPW).

#define MARK(c) { int _c = (c); _Pragma("unroll") for (int _w = 0; _w < MAXWC; ++_w) if (_w == (_c >> 5)) mask[_w] |= 1u << (_c & 31); }

inline float pf_entry(__global const float *coords, __global const float *poles, __global const int *axis_start,
                      __global const float *axis_count, float inv_sig2, float norm, int r, int kg, int c, int p,
                      int S, int C, int S_eta, int P, int A)
{
    // the loop body of pfmatrix_eval_poles
    int a0 = axis_start[p], a1 = axis_start[p + 1];
    int pole_base = kg * A * 3;
    float w_total = 0.0f;
    for (int s = 0; s < S; ++s) {
        for (int se = 0; se < S_eta; ++se) {
            int coord_base = ((((r * S + s) * C + c) * S_eta + se) * P + p) * 3;
            float vx = coords[coord_base + 0];
            float vy = coords[coord_base + 1];
            float vz = coords[coord_base + 2];
            for (int a = a0; a < a1; ++a) {
                int b = pole_base + a * 3;
                float d = fabs(poles[b + 0]*vx + poles[b + 1]*vy + poles[b + 2]*vz);
                float t = 0.0f;
                float e1 = (1.0f - d) * inv_sig2;
                if (e1 < 6.0f) t += exp(-e1);
                float e2 = (1.0f + d) * inv_sig2;
                if (e2 < 6.0f) t += exp(-e2);
                w_total += axis_count[a] * t;
            }
        }
    }
    return (w_total / (float)(S * S_eta)) * norm;
}

// Candidates of rows k0 .. k0+Kb-1 (all omegas). One work-item per (row, ring) marks its ring's
// candidate eta bins in a bit mask.
// mode 0: counts[row] = candidates of the row (counts[n_rows] = 0, for the scan).
// mode 1: write them from cand_ptr[row] on, j ascending (c, then p), with their row.
__kernel void pf_gen_candidates(
    __global const float *poles,       // (Ktot, A, 3)
    __global const int *axis_start,    // (P+1,)
    __global const float *inv_sigma2,  // (Ktot,)
    __global const float *frames,      // (R*S, 9): d0, d90, p0 rotated like the probed directions
    __global const float *sin_th,      // (P,) sin and cos of the Bragg angle
    __global const float *cos_th,
    __global int *counts,              // mode 0: (n_rows + 1,)
    __global const int *cand_ptr,      // mode 1: (n_rows + 1,)
    __global ushort *cand_j,           // mode 1: (n,)
    __global int *cand_row,            // mode 1: (n,)
    const int R, const int Kb, const int k0, const int C, const int P, const int A, const int S, const int S_eta,
    const float eta0, const float dsub, const int full, const int mode)
{
    __local uint lmask[RPG][PPW][MAXWC];
    __local int lcount[RPG][PPW];
    const int n_rows = R * Kb;
    int lid = get_local_id(0);
    int lr = lid / PPW, p = lid % PPW;
    int row = get_group_id(0) * RPG + lr;
    int active = (row < n_rows) && (p < P);
    int r = row / Kb, kg = k0 + row % Kb;
    int Msub = C * S_eta;
    uint mask[MAXWC];
    #pragma unroll
    for (int w = 0; w < MAXWC; ++w) mask[w] = 0u;

    if (active) {
        // relaxed threshold: candidates are a superset of the entries with (1 - |n.v|) * inv_sig2 < 6
        float tau = 1.0f - 6.0f / inv_sigma2[kg] - 1e-4f;
        float st = sin_th[p], ct = cos_th[p];
        for (int a = axis_start[p]; a < axis_start[p + 1]; ++a) {
            int b = (kg * A + a) * 3;
            float nx = poles[b], ny = poles[b + 1], nz = poles[b + 2];
            for (int s = 0; s < S; ++s) {
                __global const float *f = frames + (r * S + s) * 9;
                float x = nx*f[0] + ny*f[1] + nz*f[2];
                float y = nx*f[3] + ny*f[4] + nz*f[5];
                float z = nx*f[6] + ny*f[7] + nz*f[8];
                float Aa = ct * sqrt(x*x + y*y), B = st * z;
                if (tau - fabs(B) >= Aa) continue;          // neither n nor -n diffracts here
                float phi = atan2(y, x);
                for (int sg = 0; sg < 2; ++sg) {            // n: n.v > tau; -n: n.v < -tau
                    float ratio = (tau + (sg ? -B : B)) / fmax(Aa, 1e-12f);
                    if (ratio >= 1.0f) continue;
                    float w = (ratio <= -1.0f) ? 3.2f : acos(ratio);
                    float center = phi + (sg ? 3.14159265358979f : 0.0f);
                    for (int sh = (full ? 0 : -1); sh <= (full ? 0 : 1); ++sh) {
                        float cc = center + sh * 6.283185307179586f;
                        // eta sub-bin m has its centre at eta0 + (m + 0.5) * dsub
                        int m_lo = (int)ceil((cc - w - eta0) / dsub - 0.5f);
                        int m_hi = (int)floor((cc + w - eta0) / dsub - 0.5f);
                        if (m_hi - m_lo >= Msub) { m_lo = 0; m_hi = Msub - 1; }
                        for (int m = m_lo; m <= m_hi; ++m) {
                            int mm = m;
                            if (full) mm = ((m % Msub) + Msub) % Msub;
                            else if (m < 0 || m >= Msub) continue;
                            MARK(mm / S_eta);
                        }
                    }
                }
            }
        }
    }

    if (mode == 0) {
        int n = 0;
        #pragma unroll
        for (int w = 0; w < MAXWC; ++w) n += popcount(mask[w]);
        lcount[lr][p] = n;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (p == 0 && row <= n_rows) {
            int tot = 0;
            for (int q = 0; q < P; ++q) tot += lcount[lr][q];
            counts[row] = (row < n_rows) ? tot : 0;
        }
        return;
    }

    if (p < P) {
        #pragma unroll
        for (int w = 0; w < MAXWC; ++w) lmask[lr][p][w] = mask[w];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // re-partition: work-item (lr2, w2) writes the candidates in word w2 of row lr2, c then p ascending
    if (lid >= RPG * MAXWC) return;
    int lr2 = lid / MAXWC, w2 = lid % MAXWC;
    int row2 = get_group_id(0) * RPG + lr2;
    if (row2 >= n_rows) return;
    int pos = cand_ptr[row2];
    for (int w = 0; w < w2; ++w)
        for (int q = 0; q < P; ++q) pos += popcount(lmask[lr2][q][w]);
    uint lm[PPW];
    uint uni = 0u;
    #pragma unroll
    for (int q = 0; q < PPW; ++q) { lm[q] = (q < P) ? lmask[lr2][q][w2] : 0u; uni |= lm[q]; }
    while (uni) {
        int bit = 31 - clz(uni & -uni);   // lowest set bit
        uni &= uni - 1u;
        int c = (w2 << 5) + bit;
        #pragma unroll
        for (int q = 0; q < PPW; ++q) {
            if ((lm[q] >> bit) & 1u) {
                cand_j[pos] = (ushort)(c * P + q);
                cand_row[pos] = row2;
                pos++;
            }
        }
    }
}

// Evaluate candidate i < n (n = cand_ptr[n_rows]): its value and flag[i] = (value != 0);
// flag[i] = 0 for n <= i < n_flags, so a scan of all n_flags entries gives the positions.
__kernel void pf_gen_evaluate(
    __global const ushort *cand_j, __global const int *cand_row, __global const int *cand_ptr,
    __global const float *coords, __global const float *poles, __global const int *axis_start,
    __global const float *axis_count, __global const float *inv_sigma2, __global const float *norm_factor,
    __global const float *intens, __global float *cand_val, __global int *flag,
    const int n_rows, const int n_flags, const int Kb, const int k0, const int C, const int P, const int A,
    const int S, const int S_eta, const int normalized)
{
    int i = get_global_id(0);
    if (i >= n_flags) return;
    if (i >= cand_ptr[n_rows]) { flag[i] = 0; return; }
    int row = cand_row[i], j = cand_j[i];
    int r = row / Kb, kg = k0 + row % Kb, c = j / P, p = j % P;
    float v = pf_entry(coords, poles, axis_start, axis_count, inv_sigma2[kg], norm_factor[kg], r, kg, c, p,
                       S, C, S_eta, P, A);
    if (!normalized) v *= intens[p];   // as SCALE_PF_BY_INTENSITY_INPLACE
    cand_val[i] = v;
    flag[i] = (v != 0.0f);
}

// Keep the non-zero candidates (pos: exclusive scan of flag): the adjoint CSR, and every entry's row.
__kernel void pf_gen_compact(
    __global const ushort *cand_j, __global const int *cand_row, __global const float *cand_val,
    __global const int *flag, __global const int *pos, __global const int *cand_ptr,
    __global int *row_ptr, __global ushort *col_j, __global float *val, __global int *row_of,
    const int n_rows)
{
    int i = get_global_id(0);
    if (i <= n_rows) row_ptr[i] = pos[cand_ptr[i]];
    if (i < cand_ptr[n_rows] && flag[i]) {
        int o = pos[i];
        col_j[o] = cand_j[i];
        val[o] = cand_val[i];
        row_of[o] = cand_row[i];
    }
}

// Forward CSR (rows r * CP + j listing k) from the adjoint CSR: count, scan, fill, sort each row by k.
__kernel void pf_gen_count_fwd(
    __global const int *row_of, __global const ushort *col_j, __global const int *row_ptr_a,
    __global volatile int *counts_f, const int n_rows, const int Kb, const int CP)
{
    int i = get_global_id(0);
    if (i >= row_ptr_a[n_rows]) return;
    atomic_inc(&counts_f[(row_of[i] / Kb) * CP + col_j[i]]);
}

__kernel void pf_gen_fill_fwd(
    __global const int *row_of, __global const ushort *col_j, __global const float *val_a,
    __global const int *row_ptr_a, __global const int *row_ptr_f, __global volatile int *cursor,
    __global ushort *col_k, __global float *val_f, const int n_rows, const int Kb, const int CP)
{
    int i = get_global_id(0);
    if (i >= row_ptr_a[n_rows]) return;
    int row = row_of[i];
    int rf = (row / Kb) * CP + col_j[i];
    int o = row_ptr_f[rf] + atomic_inc(&cursor[rf]);
    col_k[o] = (ushort)(row % Kb);
    val_f[o] = val_a[i];
}

__kernel void pf_gen_sort_rows(
    __global const int *row_ptr, __global ushort *col, __global float *val, const int n_rows)
{
    int row = get_global_id(0);
    if (row >= n_rows) return;
    int b = row_ptr[row], e = row_ptr[row + 1];
    for (int i = b + 1; i < e; ++i) {   // insertion sort: the rows are short
        ushort ck = col[i];
        float cv = val[i];
        int j = i - 1;
        while (j >= b && col[j] > ck) { col[j + 1] = col[j]; val[j + 1] = val[j]; --j; }
        col[j + 1] = ck;
        val[j + 1] = cv;
    }
}
