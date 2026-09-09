/*
 * bragg_edge_kernels.cl
 *
 * OpenCL kernel set for building the Bragg-edge transmission operator matrix.
 *
 * For each (orientation-index k, tomographic-angle ω, wavelength λ) this kernel
 * accumulates the total coherent elastic cross-section contributed by all active
 * reflections while the crystallite is placed at orientation k and the neutron
 * beam arrives along direction beam_dirs[ω].
 *
 * This is a faithful GPU replication of the CPU reference implementation
 * ``build_bragg_matrix_cpu`` (see bragg_edge_tomographic_operator.py), which in
 * turn replicates the MATLAB functions xs_text_full_cubic + xs_singlecrystal_2022.
 * No cubic-crystal assumption is made: reflections are described by general
 * reciprocal-lattice vectors g = (gx, gy, gz) in Å⁻¹ (no 2π factor), exactly as
 * produced by Material.neutron_bragg_table().
 *
 * Physics reference:
 *   F. Malamud et al., "Wavelength-resolved neutron transmission analyses of
 *   textured materials", 2025.
 *
 * Performance design
 * ------------------
 * The only quantity that depends on wavelength λ is the final Ikeda-Carpenter
 * profile evaluation. Everything else for a given reflection r and a given
 * (orientation k, angle ω) pair — the edge wavelength λ0, the kinematic
 * amplitude, the Gaussian width σ, and the exponential tail length α — is
 * completely independent of λ. The naive kernel (one thread per (k, ω, λ))
 * would recompute all of this trigonometry-heavy work once per wavelength bin,
 * i.e. N_lam times more than necessary.
 *
 * Instead, work-items are grouped so that every thread in a work-group shares
 * the same (k, ω) pair (local size in dimension 0 is 1) and covers a distinct
 * slice of the wavelength axis (dimension 1). Each group cooperatively
 * precomputes {λ0, amplitude, σ, α} for *all* reflections exactly once, using
 * local (work-group-shared) memory, then every thread in the group loops over
 * the cached per-reflection arrays to accumulate its own λ value. This turns
 * the redundant trigonometric work from O(N_pairs · N_hkl · N_lam) down to
 * O(N_pairs · N_hkl) — the remaining O(N_pairs · N_hkl · N_lam) work is just a
 * handful of cheap FLOPs (erfc/exp or a Gaussian) per (reflection, λ) pair,
 * which is the theoretical minimum amount of work required.
 *
 * Kernel  bragg_edge_col
 * ──────────────────────
 *   Global size : (N_orient * N_Omega, N_lam_padded)
 *   Local  size : (1, L)   — L threads share one (k, ω) pair.
 *
 *   For each reflection r (computed once per work-group):
 *     1.  Project beam direction into crystal frame:
 *           n = R_k^{-1} · beam_dirs[ω]
 *     2.  General Bragg-edge wavelength (any crystal system):
 *           λ0 = 2 (g·n) / |g|²
 *     3.  Keep only λ0 within [lam_min, lam_max].
 *     4.  Bragg angle:      θ_B  = arcsin(λ0 / 2d)
 *         Incidence angle:  α0   = arccos(λ0 / 2d)   (= π/2 − θ_B)
 *     5.  Kinematic amplitude (macroscopic attenuation coefficient):
 *           A = λ0⁴ · F2 / (2 · V² · sin²θ_B)
 *         with λ0 in Å, F2 in barns, V (unit-cell volume) in Å³ — no unit
 *         conversion factors needed: the cm-scale prefactors in the
 *         Å→cm/barn→cm²/Å³→cm³ conversions cancel exactly (10⁸·10⁻³²·10⁻²⁴/10⁻⁴⁸ = 1),
 *         so working entirely in Å/barn units gives the identical cm⁻¹ result
 *         while avoiding float32 underflow: V_cm³ ≈ 1e-23 so V_cm³² ≈ 1e-46 sits
 *         right at the edge of the float32 subnormal range, previously causing
 *         NaN/Inf in the naive cm-based computation.
 *     6.  Edge broadening (geometric + instrumental):
 *           σ = λ0 · sqrt(tan²(α0)·σ_grid² + e0²) · edge_blur_scale
 *     7.  Exponential tail decay length:
 *           α = τ(λ0) / 10000
 *   Then, for every λ assigned to this thread:
 *     8.  Ikeda-Carpenter edge profile:
 *           if σ < 0.2·α  →  erfc-exponential form
 *           else           →  Gaussian approximation
 *     9.  Accumulate into output[k * N_Omega * N_lam + ω * N_lam + λ].
 *
 * Note on the pulse-tail function
 * --------------------------------
 * τ(λ) is hard-coded below as the RADEN/J-PARC empirical fit (see
 * diffractom.utils.instrument.raden_pulse_tail). The GPU path therefore only
 * supports that instrument model; the host wrapper enforces this.
 */

/* ------------------------------------------------------------------
 * Helper: empirical pulse-tail parameter τ(λ) — RADEN/J-PARC (tau.m)
 *     τ(x) = erf((x - 1.39341)/0.18492) · (18.94806 - 10.82914·x)
 *           + 16.6964·x
 * ------------------------------------------------------------------ */
static float tau_func(float x) {
    float erf_arg = (x - 1.39341f) / 0.18492f;
    float e = erf(erf_arg);
    return e * (18.94806f - 10.82914f * x) + 16.6964f * x;
}

/* ------------------------------------------------------------------
 * Helper: Ikeda-Carpenter Bragg-edge profile for a single reflection.
 *
 * lam       – current wavelength (Å)
 * lam0      – edge position (Å)
 * sigma     – Gaussian width (Å)
 * alpha     – exponential tail length (Å)
 * amplitude – pre-computed kinematic amplitude (0 for inactive reflections)
 * ------------------------------------------------------------------ */
static float ic_profile(float lam, float lam0,
                        float sigma, float alpha, float amplitude) {
    if (amplitude == 0.0f) return 0.0f;

    if (sigma < 0.2f * alpha) {
        /*
         * Ikeda-Carpenter erfc-exponential form.
         *
         * Numerical hazard in float32: when lam << lam0,
         *   u  = -(lam-lam0)/(sqrt2*sigma) + sigma/alpha  →  +∞
         *   erfc(u)  →  0.0f  (underflows)
         *   exp((lam0-lam)/alpha)  →  +∞  (overflows)
         *   product  →  NaN  (0 * inf)
         *
         * Guard: skip when combined exponent is tiny enough to be zero.
         */
        float dlam = lam - lam0;   /* negative when lam < lam0 */
        if (-dlam / (1.41421356f * sigma) > 9.0f) return 0.0f;

        float u = -dlam / (1.41421356f * sigma) + (sigma / alpha);
        float log_dec = -dlam / alpha;  /* positive when lam < lam0 */
        float exponent = 0.5f * (sigma / alpha) * (sigma / alpha) + log_dec;
        if (exponent > 80.0f) return 0.0f;
        float expterm = exp(exponent);
        return amplitude * erfc(u) * (1.0f / (2.0f * alpha)) * expterm;
    } else {
        /* Gaussian approximation */
        float arg = (lam - lam0) / sigma;
        return amplitude * (1.0f / (sqrt(2.0f * M_PI_F) * sigma))
               * exp(-0.5f * arg * arg);
    }
}

/* ------------------------------------------------------------------
 * Main kernel
 *
 * Arguments
 * ---------
 * rot_inv   (N_orient, 9) float  – inverse rotation matrices, row-major
 * beam_dirs (N_Omega,  3) float  – beam unit vectors in sample frame
 * hkl_table (N_hkl,   5) float  – [gx, gy, gz, F2_barns, d_Angstrom]
 *                                   g is the reciprocal-lattice vector (Å⁻¹,
 *                                   no 2π factor); works for any crystal system.
 * lam       (N_lam,)      float  – wavelength grid (Å), strictly increasing
 * out       (N_orient * N_Omega, N_lam) float  – output matrix (row-major)
 *
 * lam0_loc, amp_loc, sigma_loc, alpha_loc : __local scratch, size N_hkl each.
 * Shared by every thread in the work-group (all of which handle the same
 * (k, ω) pair), filled cooperatively before the per-λ accumulation loop.
 *
 * Scalars
 * -------
 * V_A3           – unit-cell volume in Å³ (NOT converted to cm³ — see the
 *                  amplitude-formula note above for why this matters for
 *                  float32 numerical stability)
 * sigma_grid     – angular half-width of the orientation node in radians
 * e0             – instrumental exponential resolution parameter
 * edge_blur_scale– multiplier applied to the edge broadening term
 * lam_min, lam_max – wavelength window (Å)
 * N_orient, N_Omega, N_lam, N_hkl – dimensions
 * ------------------------------------------------------------------ */
__kernel void bragg_edge_col(
    __global const float *rot_inv,     /* (N_orient, 9) */
    __global const float *beam_dirs,   /* (N_Omega,  3) */
    __global const float *hkl_table,   /* (N_hkl,    5)  gx gy gz F2 d */
    __global const float *lam,         /* (N_lam,)       Å */
    __global       float *out,         /* (N_orient * N_Omega, N_lam) */

    __local        float *lam0_loc,    /* (N_hkl,) work-group scratch */
    __local        float *amp_loc,     /* (N_hkl,) */
    __local        float *sigma_loc,   /* (N_hkl,) */
    __local        float *alpha_loc,   /* (N_hkl,) */

    const float  V_A3,
    const float  sigma_grid,
    const float  e0,
    const float  edge_blur_scale,
    const float  lam_min,
    const float  lam_max,

    const int    N_orient,
    const int    N_Omega,
    const int    N_lam,
    const int    N_hkl
) {
    int gid       = get_global_id(0);      /* (k, ω) pair index, one per work-group */
    int lam_idx   = get_global_id(1);
    int local_lam = get_local_id(1);
    int L         = get_local_size(1);

    int N_pairs = N_orient * N_Omega;
    if (gid >= N_pairs) return;

    int k_idx  = gid / N_Omega;
    int om_idx = gid % N_Omega;

    /* --- load inverse rotation matrix for orientation k (row-major 3x3) --- */
    int rb = k_idx * 9;
    float R00 = rot_inv[rb + 0], R01 = rot_inv[rb + 1], R02 = rot_inv[rb + 2];
    float R10 = rot_inv[rb + 3], R11 = rot_inv[rb + 4], R12 = rot_inv[rb + 5];
    float R20 = rot_inv[rb + 6], R21 = rot_inv[rb + 7], R22 = rot_inv[rb + 8];

    /* --- load beam direction for angle ω --- */
    int bb = om_idx * 3;
    float vx = beam_dirs[bb + 0];
    float vy = beam_dirs[bb + 1];
    float vz = beam_dirs[bb + 2];

    /* --- rotate beam into crystal frame: n = R_inv . v --- */
    float nx = R00*vx + R01*vy + R02*vz;
    float ny = R10*vx + R11*vy + R12*vz;
    float nz = R20*vx + R21*vy + R22*vz;

    float sg2 = sigma_grid * sigma_grid;
    float e02 = e0 * e0;
    float inv_2V2 = 1.0f / (2.0f * V_A3 * V_A3);

    /* --- cooperatively precompute per-reflection, λ-independent quantities ---
     * Every thread in the work-group shares the same (k, ω), so this work is
     * done exactly ONCE per group instead of once per (group, λ). Reflections
     * are strided across the L threads of the group.
     */
    for (int r = local_lam; r < N_hkl; r += L) {
        int rb2 = r * 5;
        float gx = hkl_table[rb2 + 0];
        float gy = hkl_table[rb2 + 1];
        float gz = hkl_table[rb2 + 2];
        float F2 = hkl_table[rb2 + 3];   /* barns */
        float d  = hkl_table[rb2 + 4];   /* Å     */

        float g_sq = gx*gx + gy*gy + gz*gz;
        float proj = gx*nx + gy*ny + gz*nz;
        float lam0 = 2.0f * proj / g_sq;   /* general Bragg-edge wavelength */

        float amp = 0.0f, sigma = 1.0f, alpha = 1.0f;

        if (lam0 >= lam_min && lam0 <= lam_max) {
            float ratio = lam0 / (2.0f * d);
            ratio = clamp(ratio, -1.0f, 1.0f);
            float theta_B  = asin(ratio);
            float alpha0   = acos(ratio);          /* = pi/2 - theta_B */
            float sin_tB   = sin(theta_B);
            float sin2_tB  = fmax(sin_tB * sin_tB, 1.0e-30f);

            /* Amplitude computed directly in Å / barn units (see file header
             * note): algebraically identical to the cm-based formula but
             * numerically stable in float32 (avoids squaring a ~1e-23 cm^3
             * volume, which underflows toward the subnormal float32 range). */
            float lam0_2 = lam0 * lam0;
            amp = (lam0_2 * lam0_2) * F2 * inv_2V2 / sin2_tB;

            float tan_a0 = tan(alpha0);
            sigma = lam0 * sqrt(tan_a0*tan_a0*sg2 + e02) * edge_blur_scale;
            alpha = tau_func(lam0) / 10000.0f;

            if (sigma <= 0.0f) sigma = 1.0e-6f;
            if (alpha <= 0.0f) alpha = 1.0e-6f;
        }

        lam0_loc[r]  = lam0;
        amp_loc[r]   = amp;
        sigma_loc[r] = sigma;
        alpha_loc[r] = alpha;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lam_idx >= N_lam) return;   /* padding thread for divisibility */

    float lam_val = lam[lam_idx];
    float xs = 0.0f;

    for (int r = 0; r < N_hkl; ++r) {
        xs += ic_profile(lam_val, lam0_loc[r], sigma_loc[r], alpha_loc[r], amp_loc[r]);
    }

    int row = k_idx * N_Omega + om_idx;
    out[row * N_lam + lam_idx] = xs;
}
