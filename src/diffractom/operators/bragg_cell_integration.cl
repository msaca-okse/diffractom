// Exact cell averages of single-crystal Bragg spectra: see bragg_cell_integration.py.
// Quaternions are scalar-first (w, x, y, z); 3-vectors are the xyz part of a 4-vector.

#ifdef USE_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real;
typedef double4 real4;
#define PI_R M_PI
#define NEGLIGIBLE 1e-30
#else
typedef float real;
typedef float4 real4;
#define PI_R M_PI_F
#define NEGLIGIBLE 1e-13f
#endif

inline real4 qmul(real4 p, real4 q)
{
    return (real4)(p.s0 * q.s0 - p.s1 * q.s1 - p.s2 * q.s2 - p.s3 * q.s3,
                   p.s0 * q.s1 + p.s1 * q.s0 + p.s2 * q.s3 - p.s3 * q.s2,
                   p.s0 * q.s2 - p.s1 * q.s3 + p.s2 * q.s0 + p.s3 * q.s1,
                   p.s0 * q.s3 + p.s1 * q.s2 - p.s2 * q.s1 + p.s3 * q.s0);
}

// Rotation-angle length of the great circle cos(beta) p0 + sin(beta) p1 inside the cone {q : q.a >= 0, q.w >= 0}.
// Every half-space keeps a half circle, so the cone keeps one arc; its ends are the extreme relative angles of (w.p0, w.p1).
inline real fibre_length(real4 p0, real4 p1, real4 a, __global const real4 *W, uint nw)
{
    real Aa = dot(a, p0), Ba = dot(a, p1);
    real pmax = 0, cmax = 1, smax = 0, pmin = 0, cmin = 1, smin = 0;
    for (uint m = 0; m < nw; ++m) {
        real4 w = W[m];
        real A = dot(w, p0), B = dot(w, p1);
        if (A * A + B * B < NEGLIGIBLE)
            continue;
        real c = Aa * A + Ba * B, s = Aa * B - Ba * A;
        real den = fabs(c) + fabs(s);
        real r = den > 0 ? c / den : (real)1;
        real p = s >= 0 ? 1 - r : r - 1;                     // monotonic in atan2(s, c)
        if (p > pmax) { pmax = p; cmax = c; smax = s; }
        if (p < pmin) { pmin = p; cmin = c; smin = s; }
    }
    real L = PI_R - (pmax > 0 ? atan2(smax, cmax) : (real)0) + (pmin < 0 ? atan2(smin, cmin) : (real)0);
    return L > 0 ? 2 * L : (real)0;
}

// Fibre length of the crystallites whose crystal direction g points along n; p0 = (shortest turn nc -> n) a.
inline real lambda_at(real4 n, real4 nc, real4 a, real4 g, __global const real4 *W, uint nw)
{
    real w = 1 + nc.x * n.x + nc.y * n.y + nc.z * n.z;
    real4 t = (real4)(w, nc.y * n.z - nc.z * n.y, nc.z * n.x - nc.x * n.z, nc.x * n.y - nc.y * n.x);
    t /= sqrt(dot(t, t));
    real4 p0 = qmul(t, a);
    real4 p1 = qmul(p0, (real4)(0, g.x, g.y, g.z));
    return fibre_length(p0, p1, a, W, nw);
}

// F = int Lambda(n(t, phi), g) dphi along the part of the cone t = lam / 2d that lies in the cap of the piece.
__kernel void arc_integrals(__global const uint *item_task, __global const uint *item_j, const uint n_items,
                            __global const real *lam,
                            __global const uint *t_piece, __global const uint *t_beam, __global const uint *t_refl,
                            __global const real4 *t_nc,   // xyz: centre of the cap
                            __global const real4 *t_geo,  // cos and sin of its angle to the beam, its azimuth, the angle
                            __global const real4 *piece_a, __global const uint *piece_row0,
                            __global const uint *piece_nrow, __global const real *piece_r,
                            __global const real4 *rows,
                            __global const real4 *beam, __global const real4 *e1, __global const real4 *e2,
                            __global const real4 *g_dir, __global const real *two_d,
                            const real h_arc, __global real *F)
{
    size_t i = get_global_id(0);
    if (i >= n_items)
        return;
    uint task = item_task[i];
    uint p = t_piece[task], ib = t_beam[task], ir = t_refl[task];
    real4 nc = t_nc[task], geo = t_geo[task], a = piece_a[p], b = beam[ib], g = g_dir[ir];
    real r = piece_r[p];
    __global const real4 *W = rows + piece_row0[p];
    uint nw = piece_nrow[p];

    real t = fmin(lam[item_j[i]] / two_d[ir], (real)1);
    real s = sqrt(fmax(1 - t * t, (real)0));
    F[i] = 0;
    if (fabs(acos(t) - geo.s3) > r)
        return;
    if (s < 1e-9f) {                                         // the circle is the point b
        F[i] = 2 * PI_R * lambda_at(b, nc, a, g, W, nw);
        return;
    }
    real arc = PI_R;
    if (geo.s1 >= 1e-12f) {
        real cd = (cos(r) - t * geo.s0) / (s * geo.s1);
        arc = cd <= -1 ? PI_R : (cd >= 1 ? (real)0 : acos(cd));
    }
    int M = max(8, (int)ceil(2 * arc * s / h_arc));
    real dphi = 2 * arc / M, acc = 0;
    real4 u = e1[ib], v = e2[ib];
    for (int m = 0; m < M; ++m) {
        real sp, cp = 0, ph = geo.s2 - arc + (m + (real)0.5) * dphi;
        sp = sincos(ph, &cp);
        real4 n = t * b + s * (cp * u + sp * v);
        acc += lambda_at(n, nc, a, g, W, nw);
    }
    F[i] = acc * dphi;
}

// One ring (rho fixed) of the volume integral |Z| = 1/(8 pi^2) int Lambda dn over the cap.
__kernel void cap_ring(__global const real4 *rows, const uint nw, const real4 a, const real4 nc, const real4 g,
                       const real4 u, const real4 v, const real drho, const uint n_psi, __global real *out)
{
    size_t i = get_global_id(0);
    real sr, cr = 0, rho = (i + (real)0.5) * drho, dpsi = 2 * PI_R / n_psi, acc = 0;
    sr = sincos(rho, &cr);
    for (uint j = 0; j < n_psi; ++j) {
        real sp, cp = 0, ps = (j + (real)0.5) * dpsi;
        sp = sincos(ps, &cp);
        real4 n = cr * nc + sr * (cp * u + sp * v);
        acc += lambda_at(n, nc, a, g, rows, nw);
    }
    out[i] = acc * sr;
}
