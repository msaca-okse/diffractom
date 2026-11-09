// Vector-Jacobian product of SinglePhaseForwardOperator's model with respect to the PF parameters
// (docs/differentiable_model.md, section 3): for every non-zero (r, k, j) of a batch's PF matrix (adjoint CSR, rows
// r * Kb + k listing segments j = c P + p)
//     G = sum_y res[r, y, j] S[r, y, k]                    (S: the batch's projections, (R, My, Kstride))
// times the derivative of the entry, evaluated with the loop over quadrature nodes and pole axes of pf_entry
// (pf_generate.cl), summed per row. With HESS also the Gauss-Newton block of the orientation,
//     sum_y curv[r, y, j] S[r, y, k]^2  q q^T,   q = d PF / d (rotation x, y, z, sigma).
//
// Output per row (NQ floats): rotation (3), sigma (1), [Hessian block (10, upper triangle of 4 x 4)],
// eta offset (1), axis tilt (2), two-theta of every ring (P).
//
// Compile-time: PF_CUT, QUAD_WEIGHTED (as the PF kernels), PMAX = P, HESS (0 / 1), LS (local size, a power of 2).

#ifndef PF_CUT
#define PF_CUT 6.0f
#endif
#define NH (HESS ? 10 : 0)
#define Q_ETA (4 + NH)
#define Q_TILT (Q_ETA + 1)
#define Q_TTH (Q_TILT + 2)
#define NQ (Q_TTH + PMAX)

// WINDOW_PARALLAX (as pf_generate.cl): the axis weights carry the window fraction of the orientation at this angle. It
// does not depend on the rotation or the width, so their derivatives just carry it; the two-theta derivative ignores it.
#ifdef WINDOW_PARALLAX
#define WP_DECL , __global const float2 *wp_pos, __global const float4 *wp_cov, __global const float2 *wp_b, \
                __global const float4 *wp_ax, __global const float2 *wp_ch, const float wp_invD
inline float wp_frac(const float4 aw, const float wx, const float wvar, const float2 wlh, const float invD)
{
    const float sh = aw.x - wx * invD * aw.y, g = aw.y * invD;
    const float is = rsqrt(2.0f * (aw.z + g * g * wvar));
    const float zh = (wlh.y - sh) * is, zl = (wlh.x - sh) * is;
    return (zl > 0.0f) ? 0.5f * (erfc(zl) - erfc(zh)) : (zh < 0.0f) ? 0.5f * (erfc(-zh) - erfc(-zl)) : 0.5f * (erf(zh) - erf(zl));
}
#else
#define WP_DECL
#endif

__kernel void pf_param_vjp(
    __global const int *row_ptr, __global const ushort *col_j,
    __global const float *res, __global const float *curv, __global const float *sino,
    __global const float *coords, __global const float *dv_eta, __global const float *dv_t1,
    __global const float *dv_t2, __global const float *dv_tth,
    __global const float *poles, __global const int *axis_start, __global const float *axis_count,
    __global const float *inv_sigma2, __global const float *norm_factor, __global const float *qw,
    __global const float *intens, __global float *out,
    const int My, const int Kb, const int Kstride, const int k0, const int C, const int P, const int A,
    const int S, const int S_eta, const int normalized, const int flags, __local float *lS,
    __global const float *w_rot, __global const float *dw_t1, __global const float *dw_t2,
    __global const float *ax_st, __global const float *ax_ct WP_DECL)
{
    __local float red[LS];
    const int row = get_group_id(0);
    const int lid = get_local_id(0);
    const int r = row / Kb, k = row - (row / Kb) * Kb, kg = k0 + k;
    const int CP = C * P;
    for (int y = lid; y < My; y += LS) lS[y] = sino[((size_t)r * My + y) * Kstride + k];
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef WINDOW_PARALLAX
    const float2 wpb = wp_b[r], wpq = wp_pos[kg];
    const float4 wpc = wp_cov[kg];
    const float wpx = wpq.x * wpb.x + wpq.y * wpb.y;
    const float wpv = wpc.x * wpb.x * wpb.x + 2.0f * wpc.y * wpb.x * wpb.y + wpc.z * wpb.y * wpb.y;
#endif
    const float sinv = inv_sigma2[kg];
    const float sig = rsqrt(sinv);
    const int w_eta = flags & 1, w_tilt = flags & 2, w_tth = flags & 4;
    float acc[NQ];
    for (int q = 0; q < NQ; ++q) acc[q] = 0.0f;

    for (int i = row_ptr[row] + lid; i < row_ptr[row + 1]; i += LS) {
        const int j = col_j[i];
        const int c = j / P, p = j - (j / P) * P;
        float G = 0.0f, Hs = 0.0f;
        for (int y = 0; y < My; ++y) {
            const size_t di = ((size_t)r * My + y) * CP + j;
            G += res[di] * lS[y];
#if HESS
            Hs += curv[di] * lS[y] * lS[y];
#endif
        }
        const float f = norm_factor[kg] * (normalized ? 1.0f : intens[p]);
        const int a0 = axis_start[p], a1 = axis_start[p + 1];
        float dr0 = 0.0f, dr1 = 0.0f, dr2 = 0.0f, gs = 0.0f, dgs = 0.0f, de = 0.0f, dt1 = 0.0f, dt2 = 0.0f, dtt = 0.0f;
        for (int s = 0; s < S; ++s) {
            for (int se = 0; se < S_eta; ++se) {
#ifdef QUAD_WEIGHTED
                const float w = qw[s * S_eta + se];
#else
                const float w = 1.0f / (float)(S * S_eta);
#endif
#ifdef AXIS_THETA
                const size_t cb = (((size_t)(r * S + s) * C + c) * S_eta + se) * 3;   // coords = u, no ring index
                const size_t wb = (size_t)(r * S + s) * 3;
                const float ux = coords[cb], uy = coords[cb + 1], uz = coords[cb + 2];
                const float wx = w_rot[wb], wy = w_rot[wb + 1], wz = w_rot[wb + 2];
#else
                const size_t cb = ((((size_t)(r * S + s) * C + c) * S_eta + se) * P + p) * 3;
                const float vx = coords[cb], vy = coords[cb + 1], vz = coords[cb + 2];
#endif
                for (int a = a0; a < a1; ++a) {
                    const size_t b = ((size_t)kg * A + a) * 3;
                    const float nx = poles[b], ny = poles[b + 1], nz = poles[b + 2];
#ifdef AXIS_THETA
                    const float ct = ax_ct[a], st = ax_st[a];
                    const float vx = ct * ux - st * wx, vy = ct * uy - st * wy, vz = ct * uz - st * wz;
#endif
                    const float D = nx * vx + ny * vy + nz * vz;
                    const float ad = fabs(D);
                    const float x1 = (1.0f - ad) * sinv, x2 = (1.0f + ad) * sinv;
                    const float e1 = x1 < PF_CUT ? exp(-x1) : 0.0f;
                    const float e2 = x2 < PF_CUT ? exp(-x2) : 0.0f;
#ifdef WINDOW_PARALLAX
                    const float cw = w * axis_count[a] * wp_frac(wp_ax[a], wpx, wpv, wp_ch[p], wp_invD);
#else
                    const float cw = w * axis_count[a];
#endif
                    const float gd = cw * copysign(sinv, D) * (e1 - e2);
                    dr0 += gd * (ny * vz - nz * vy);
                    dr1 += gd * (nz * vx - nx * vz);
                    dr2 += gd * (nx * vy - ny * vx);
                    gs += cw * (e1 + e2);
                    dgs += cw * (-(1.0f - ad) * e1 - (1.0f + ad) * e2);
#ifdef AXIS_THETA
                    if (w_eta) de += gd * ct * (nx * dv_eta[cb] + ny * dv_eta[cb + 1] + nz * dv_eta[cb + 2]);
                    if (w_tilt) {
                        dt1 += gd * (ct * (nx * dv_t1[cb] + ny * dv_t1[cb + 1] + nz * dv_t1[cb + 2])
                                     - st * (nx * dw_t1[wb] + ny * dw_t1[wb + 1] + nz * dw_t1[wb + 2]));
                        dt2 += gd * (ct * (nx * dv_t2[cb] + ny * dv_t2[cb + 1] + nz * dv_t2[cb + 2])
                                     - st * (nx * dw_t2[wb] + ny * dw_t2[wb + 1] + nz * dw_t2[wb + 2]));
                    }
                    if (w_tth) dtt += gd * 0.5f * (-st * (nx * ux + ny * uy + nz * uz) - ct * (nx * wx + ny * wy + nz * wz));
#else
                    if (w_eta) de += gd * (nx * dv_eta[cb] + ny * dv_eta[cb + 1] + nz * dv_eta[cb + 2]);
                    if (w_tilt) {
                        dt1 += gd * (nx * dv_t1[cb] + ny * dv_t1[cb + 1] + nz * dv_t1[cb + 2]);
                        dt2 += gd * (nx * dv_t2[cb] + ny * dv_t2[cb + 1] + nz * dv_t2[cb + 2]);
                    }
                    if (w_tth) dtt += gd * (nx * dv_tth[cb] + ny * dv_tth[cb + 1] + nz * dv_tth[cb + 2]);
#endif
                }
            }
        }
        const float q0 = f * dr0, q1 = f * dr1, q2 = f * dr2;
        const float q3 = f * (-2.0f / sig * gs - 2.0f / (sig * sig * sig) * dgs);
        acc[0] += G * q0; acc[1] += G * q1; acc[2] += G * q2; acc[3] += G * q3;
#if HESS
        acc[4] += Hs * q0 * q0; acc[5] += Hs * q0 * q1; acc[6] += Hs * q0 * q2; acc[7] += Hs * q0 * q3;
        acc[8] += Hs * q1 * q1; acc[9] += Hs * q1 * q2; acc[10] += Hs * q1 * q3;
        acc[11] += Hs * q2 * q2; acc[12] += Hs * q2 * q3; acc[13] += Hs * q3 * q3;
#endif
        acc[Q_ETA] += G * f * de;
        acc[Q_TILT] += G * f * dt1;
        acc[Q_TILT + 1] += G * f * dt2;
        for (int pp = 0; pp < PMAX; ++pp) if (pp == p) acc[Q_TTH + pp] += G * f * dtt;
    }
    // sum over the work-group, one quantity at a time
    for (int q = 0; q < NQ; ++q) {
        red[lid] = acc[q];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int h = LS / 2; h > 0; h >>= 1) {
            if (lid < h) red[lid] += red[lid + h];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) out[(size_t)row * NQ + q] = red[0];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// per orientation of the batch: the sum over the angles r of the rows (r, k)
__kernel void rows_to_orientations(__global const float *rows, __global float *out, const int R, const int Kb)
{
    const int t = get_global_id(0);
    if (t >= Kb * NQ) return;
    const int k = t / NQ, q = t - (t / NQ) * NQ;
    float s = 0.0f;
    for (int r = 0; r < R; ++r) s += rows[((size_t)r * Kb + k) * NQ + q];
    out[t] = s;
}
