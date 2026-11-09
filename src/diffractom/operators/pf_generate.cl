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

// PF_CUT: the Gaussian of a pole is cut where (1 - |cos angle|) / sigma^2 reaches PF_CUT, i.e. at an
// angle of about sqrt(2 PF_CUT) sigma (default 6: 3.46 sigma, exp(-6) = 0.25 % of the peak).
#ifndef PF_CUT
#define PF_CUT 6.0f
#endif
// QUAD_WEIGHTED: the PF entries are weighted sums over the quadrature nodes (Gauss-Legendre), whose offsets in an
// eta bin lie in [QX_LO, QX_HI] (fractions of the bin); without it the nodes are equal sub-bins, averaged
#ifndef QX_LO
#define QX_LO 0.0f
#endif
#ifndef QX_HI
#define QX_HI 1.0f
#endif

#define MARK(c) { int _c = (c); _Pragma("unroll") for (int _w = 0; _w < MAXWC; ++_w) if (_w == (_c >> 5)) mask[_w] |= 1u << (_c & 31); }

// AXIS_THETA: every pole axis has its own Bragg angle: coords = u (R, S, C, S_eta, 3), w = the rotated p0 (R * S, 3),
// v = cos(theta_a) u - sin(theta_a) w
#ifdef AXIS_THETA
#define TH_IDX a
#else
#define TH_IDX p
#endif

// WINDOW_PARALLAX: the data channels are windows [lo, hi] in 2theta that cut the radial profile of a reflection; a
// grain at depth x along the beam (downstream > 0) sees its ring shifted by -(x / D) sin(2theta) cos(2theta). The weight
// of every pole axis is multiplied by the fraction of a Gaussian radial profile inside the window at the shifted angle,
// for the orientation's mean depth at this angle; the spread of its depths (variance along the beam) widens the profile:
// sd^2 = sigma_a^2 + (sin 2theta cos 2theta / D)^2 var(x). wp_pos (K): mean position, wp_cov (K): covariance (xx, xy, yy)
// in the image plane (translation bins, relative to the rotation axis); wp_b (R): the beam direction in the image plane per
// angle; wp_ax (A): (2theta, sin 2theta cos 2theta, sigma^2, gamma); wp_ch (P): (lo, hi); wp_invD = 1 / D; wp_eta: the
// Lorentzian share of a pseudo-Voigt profile (half width gamma; 0: Gaussian).
#ifdef WINDOW_PARALLAX
#define WP_DECL , __global const float2 *wp_pos, __global const float4 *wp_cov, __global const float2 *wp_b, \
                __global const float4 *wp_ax, __global const float2 *wp_ch, const float wp_invD, const float wp_eta
#define WP_PASS , wp_pos, wp_cov, wp_b, wp_ax, wp_ch, wp_invD, wp_eta
#else
#define WP_DECL
#define WP_PASS
#endif

inline float pf_entry(__global const float *coords, __global const float *poles, __global const int *axis_start,
                      __global const float *axis_count, __global const float *qw, float inv_sig2, float norm, int r,
                      int kg, int c, int p, int S, int C, int S_eta, int P, int A,
                      __global const float *w, __global const float *ax_st, __global const float *ax_ct WP_DECL)
{
    // the loop body of pfmatrix_eval_poles
    int a0 = axis_start[p], a1 = axis_start[p + 1];
    int pole_base = kg * A * 3;
    float w_total = 0.0f;
#ifdef WINDOW_PARALLAX
    const float2 wb = wp_b[r], wq = wp_pos[kg];
    const float4 wc = wp_cov[kg];
    const float wx = wq.x * wb.x + wq.y * wb.y;
    const float wvar = wc.x * wb.x * wb.x + 2.0f * wc.y * wb.x * wb.y + wc.z * wb.y * wb.y;
    const float2 wlh = wp_ch[p];
#endif
    for (int s = 0; s < S; ++s) {
        for (int se = 0; se < S_eta; ++se) {
#ifdef AXIS_THETA
            long coord_base = (((long)(r * S + s) * C + c) * S_eta + se) * 3;
            __global const float *wv = w + (long)(r * S + s) * 3;
#else
            long coord_base = ((((long)(r * S + s) * C + c) * S_eta + se) * P + p) * 3;  // 64 bit: many nodes
#endif
            float vx = coords[coord_base + 0];
            float vy = coords[coord_base + 1];
            float vz = coords[coord_base + 2];
#ifdef QUAD_WEIGHTED
            float w_node = 0.0f;
#endif
            for (int a = a0; a < a1; ++a) {
                int b = pole_base + a * 3;
#ifdef AXIS_THETA
                float d = fabs(ax_ct[a] * (poles[b + 0]*vx + poles[b + 1]*vy + poles[b + 2]*vz)
                               - ax_st[a] * (poles[b + 0]*wv[0] + poles[b + 1]*wv[1] + poles[b + 2]*wv[2]));
#else
                float d = fabs(poles[b + 0]*vx + poles[b + 1]*vy + poles[b + 2]*vz);
#endif
                float t = 0.0f;
                float e1 = (1.0f - d) * inv_sig2;
                if (e1 < PF_CUT) t += exp(-e1);
                float e2 = (1.0f + d) * inv_sig2;
                if (e2 < PF_CUT) t += exp(-e2);
#ifdef WINDOW_PARALLAX
                {
                    const float4 aw = wp_ax[a];
                    const float sh = aw.x - wx * wp_invD * aw.y;
                    const float g = aw.y * wp_invD;
                    const float is = rsqrt(2.0f * (aw.z + g * g * wvar));
                    const float zh = (wlh.y - sh) * is, zl = (wlh.x - sh) * is;      // fraction in [lo, hi]; erfc in the tails
                    float fr = (zl > 0.0f) ? 0.5f * (erfc(zl) - erfc(zh)) : (zh < 0.0f) ? 0.5f * (erfc(-zh) - erfc(-zl))
                                                                    : 0.5f * (erf(zh) - erf(zl));
                    if (wp_eta > 0.0f)                                   // pseudo-Voigt: a Lorentzian share (half width aw.w)
                        fr = (1.0f - wp_eta) * fr + wp_eta * (atan((wlh.y - sh) / aw.w) - atan((wlh.x - sh) / aw.w)) * 0.31830988618f;
                    t *= fr;
                }
#endif
#ifdef QUAD_WEIGHTED
                w_node += axis_count[a] * t;
#else
                w_total += axis_count[a] * t;
#endif
            }
#ifdef QUAD_WEIGHTED
            w_total += qw[s * S_eta + se] * w_node;
#endif
        }
    }
#ifdef QUAD_WEIGHTED
    return w_total * norm;
#else
    return (w_total / (float)(S * S_eta)) * norm;
#endif
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
    __global const float *sin_th,      // (P,) sin and cos of the Bragg angle ((A,) per axis with AXIS_THETA)
    __global const float *cos_th,
    __global int *counts,              // mode 0: (n_rows + 1,)
    __global const int *cand_ptr,      // mode 1: (n_rows + 1,)
    __global ushort *cand_j,           // mode 1: (n,)
    __global int *cand_row,            // mode 1: (n,)
    const int R, const int Kb, const int k0, const int C, const int P, const int A, const int S, const int S_eta,
    const float eta0, const float dsub, const int full, const int mode,
    __global const int *omegas, const int use_om)    // use_om: rows (s, k) of the angles omegas[s] (R of them)
{
    __local uint lmask[RPG][PPW][MAXWC];
    __local int lcount[RPG][PPW];
    __local int lpref[RPG][MAXWC * 32 + 1];
    const int n_rows = R * Kb;
    int lid = get_local_id(0);
    int lr = lid / PPW, p = lid % PPW;
    int row = get_group_id(0) * RPG + lr;
    int active = (row < n_rows) && (p < P);
    int r = row / Kb, kg = k0 + row % Kb;
    if (use_om && row < n_rows) r = omegas[r];
    int Msub = C * S_eta;
    uint mask[MAXWC];
    #pragma unroll
    for (int w = 0; w < MAXWC; ++w) mask[w] = 0u;

    if (active) {
        // relaxed threshold: candidates are a superset of the entries with (1 - |n.v|) * inv_sig2 < PF_CUT
        float tau = 1.0f - PF_CUT / inv_sigma2[kg] - 1e-4f;
        for (int a = axis_start[p]; a < axis_start[p + 1]; ++a) {
            float st = sin_th[TH_IDX], ct = cos_th[TH_IDX];
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
#ifdef QUAD_WEIGHTED
                        // weighted (Gauss-Legendre) nodes: the nodes of eta bin c lie at eta0 + (c + x) * dbin with
                        // QX_LO <= x <= QX_HI (dsub is the bin width dbin here); mark the bins whose node span meets
                        // the window (a superset of the bins with a node inside it; for one node at x = 0.5 the
                        // same bins as the midpoint rule)
                        float t1 = (cc - w - eta0) / dsub, t2 = (cc + w - eta0) / dsub;
                        int m_lo = (int)ceil(fmin(t1, t2) - QX_HI);
                        int m_hi = (int)floor(fmax(t1, t2) - QX_LO);
                        if (m_hi - m_lo >= C) { m_lo = 0; m_hi = C - 1; }
                        for (int m = m_lo; m <= m_hi; ++m) {
                            int mm = m;
                            if (full) mm = ((m % C) + C) % C;
                            else if (m < 0 || m >= C) continue;
                            MARK(mm);
                        }
#else
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
#endif
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
    // write the candidates of row lr, eta bin c ascending, then ring q ascending: every work-item of the row takes the eta
    // bins c = p, p + PPW, ...; the candidates of bin c start at the row's offset plus the candidates of all bins below c
    // (a prefix over the bins in local memory), so every candidate lands where a sequential pass would put it
    for (int c = p; c < MAXWC * 32; c += PPW) {
        int n = 0;
        for (int q = 0; q < P; ++q) n += (lmask[lr][q][c >> 5] >> (c & 31)) & 1u;
        lpref[lr][c + 1] = n;
    }
    if (p == 0) lpref[lr][0] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (p == 0)
        for (int c = 1; c <= MAXWC * 32; ++c) lpref[lr][c] += lpref[lr][c - 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row >= n_rows) return;
    const int base = cand_ptr[row];
    for (int c = p; c < MAXWC * 32; c += PPW) {
        int pos = base + lpref[lr][c];
        const int w = c >> 5, b = c & 31;
        for (int q = 0; q < P; ++q) {
            if ((lmask[lr][q][w] >> b) & 1u) {
                cand_j[pos] = (ushort)(c * P + q);
                cand_row[pos] = row;
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
    __global const float *qw, __global const float *intens, __global float *cand_val, __global int *flag,
    const int n_rows, const int n_flags, const int Kb, const int k0, const int C, const int P, const int A,
    const int S, const int S_eta, const int normalized, __global const int *omegas, const int use_om,
    __global const float *w, __global const float *ax_st, __global const float *ax_ct WP_DECL)
{
    int i = get_global_id(0);
    if (i >= n_flags) return;
    if (i >= cand_ptr[n_rows]) { flag[i] = 0; return; }
    int row = cand_row[i], j = cand_j[i];
    int r = row / Kb, kg = k0 + row % Kb, c = j / P, p = j % P;
    if (use_om) r = omegas[r];
    float v = pf_entry(coords, poles, axis_start, axis_count, qw, inv_sigma2[kg], norm_factor[kg], r, kg, c, p,
                       S, C, S_eta, P, A, w, ax_st, ax_ct WP_PASS);
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

// The row of entry i of a CSR: the last row with row_ptr[row] <= i (binary search).
inline int csr_row(__global const int *row_ptr, int n_rows, int i)
{
    int lo = 0, hi = n_rows - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (row_ptr[mid] <= i) lo = mid; else hi = mid - 1;
    }
    return lo;
}

// Sort every forward row by k: the fill wrote it unordered to col_tmp/val_tmp; every entry goes to
// its rank among the k of its row (distinct within a row). One work-item per entry.
__kernel void pf_gen_rank_rows(
    __global const int *row_ptr, __global const ushort *col_tmp, __global const float *val_tmp,
    __global ushort *col, __global float *val, const int n_rows)
{
    int i = get_global_id(0);
    if (i >= row_ptr[n_rows]) return;
    int row = csr_row(row_ptr, n_rows, i);
    int b = row_ptr[row], e = row_ptr[row + 1];
    ushort ki = col_tmp[i];
    int rank = 0;
    for (int m = b; m < e; ++m) rank += (col_tmp[m] < ki);
    col[b + rank] = ki;
    val[b + rank] = val_tmp[i];
}

// row_of[i] = the row of entry i (for transposing a stored adjoint CSR); one work-item per entry
__kernel void pf_gen_row_of(__global const int *row_ptr, __global int *row_of, const int n_rows)
{
    int i = get_global_id(0);
    if (i >= row_ptr[n_rows]) return;
    row_of[i] = csr_row(row_ptr, n_rows, i);
}


// ---------------------------------------------------------------------------------------------
// Subsets of the angles (SinglePhaseForwardOperator.set_omega_subsets): rows (s, k) of the angles omegas[s].

// counts[t] = candidates of subset row t = s * Kb + k, from the candidate offsets of all rows (cached; full_ptr);
// counts[n_rows] = 0 (for the scan)
__kernel void pf_gen_sub_counts(__global const int *full_ptr, __global const int *omegas, __global int *counts,
                                const int Kb, const int n_rows)
{
    int t = get_global_id(0);
    if (t > n_rows) return;
    if (t == n_rows) { counts[t] = 0; return; }
    int s = t / Kb, k = t - s * Kb;
    int row = omegas[s] * Kb + k;
    counts[t] = full_ptr[row + 1] - full_ptr[row];
}

// lengths[s] = non-zeros of angle omegas[s] in an adjoint CSR of all angles (rows r * Kb + k); lengths[Rs] = 0
__kernel void pf_gen_sub_lengths(__global const int *row_ptr_a, __global const int *omegas, __global int *lengths,
                                 const int Kb, const int Rs)
{
    int s = get_global_id(0);
    if (s > Rs) return;
    if (s == Rs) { lengths[s] = 0; return; }
    int r = omegas[s];
    lengths[s] = row_ptr_a[(r + 1) * Kb] - row_ptr_a[r * Kb];
}

// Entry t of the subset's part of an adjoint CSR of all angles (sub_ptr: exclusive scan of pf_gen_sub_lengths):
// its index i in the CSR, the subset angle s and the orientation k of its row. 0 if t is beyond the last entry.
inline int sub_entry(__global const int *row_ptr_a, __global const int *omegas, __global const int *sub_ptr,
                     const int Kb, const int Rs, const int t, int *i, int *s, int *k)
{
    if (t >= sub_ptr[Rs]) return 0;
    int lo = 0, hi = Rs - 1;                 // the last s with sub_ptr[s] <= t
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (sub_ptr[mid] <= t) lo = mid; else hi = mid - 1;
    }
    *s = lo;
    int r = omegas[lo];
    int ii = row_ptr_a[r * Kb] + (t - sub_ptr[lo]);
    *i = ii;
    int a = 0, b = Kb - 1;                   // the last k with row_ptr_a[r * Kb + k] <= ii
    while (a < b) {
        int mid = (a + b + 1) >> 1;
        if (row_ptr_a[r * Kb + mid] <= ii) a = mid; else b = mid - 1;
    }
    *k = a;
    return 1;
}

// Forward CSR of the subset's rows (s * CP + j, listing k) from an adjoint CSR of all angles: count, fill (then
// scan and pf_gen_rank_rows as for the whole matrix). One work-item per entry of the subset.
__kernel void pf_gen_count_fwd_sub(
    __global const int *row_ptr_a, __global const ushort *col_j, __global const int *omegas,
    __global const int *sub_ptr, __global volatile int *counts_f, const int Kb, const int Rs, const int CP)
{
    int i, s, k;
    if (!sub_entry(row_ptr_a, omegas, sub_ptr, Kb, Rs, get_global_id(0), &i, &s, &k)) return;
    atomic_inc(&counts_f[s * CP + col_j[i]]);
}

__kernel void pf_gen_fill_fwd_sub(
    __global const int *row_ptr_a, __global const ushort *col_j, __global const float *val_a,
    __global const int *omegas, __global const int *sub_ptr, __global const int *row_ptr_f,
    __global volatile int *cursor, __global ushort *col_k, __global float *val_f, const int Kb, const int Rs,
    const int CP)
{
    int i, s, k;
    if (!sub_entry(row_ptr_a, omegas, sub_ptr, Kb, Rs, get_global_id(0), &i, &s, &k)) return;
    int rf = s * CP + col_j[i];
    int o = row_ptr_f[rf] + atomic_inc(&cursor[rf]);
    col_k[o] = (ushort)k;
    val_f[o] = val_a[i];
}
