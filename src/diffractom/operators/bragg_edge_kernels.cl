/*
 * bragg_edge_kernels.cl
 *
 * OpenCL kernel set for building the Bragg-edge transmission operator matrix.
 *
 * For each (orientation-index k, tomographic-angle ω, wavelength λ) this kernel
 * accumulates the total coherent elastic cross-section contributed by all active
 * (h,k,l) reflections of a cubic crystal whose crystallite is placed at
 * orientation k while the neutron beam arrives along direction beam_dirs[ω].
 *
 * Physics reference:
 *   F. Malamud et al., "Wavelength-resolved neutron transmission analyses of
 *   textured materials", 2025.
 *
 * Kernel  bragg_edge_col
 * ──────────────────────
 *   Global size : (N_orient * N_Omega,)
 *   Each work-item processes ONE (k, ω) pair and writes N_lam output values.
 *
 *   For each reflection r:
 *     1.  Project beam direction into crystal frame:
 *           n = R_k^{-1} · beam_dirs[ω]
 *     2.  Bragg-edge wavelength:
 *           λ₀ = 2d · (h·n₁ + k·n₂ + l·n₃) / (h²+k²+l²)
 *     3.  Positive λ₀ in [lam_min, lam_max] only.
 *     4.  Bragg angle:  θ_B = arcsin(λ₀ / 2d)
 *         Incidence angle to plane: α₀ = π/2 − θ_B
 *     5.  Amplitude (kinematic, per unit cell volume squared):
 *           A = 1e8 · (λ₀·1e-8)^4 · F2·1e-24 / (2 · V² · sin²θ_B)
 *     6.  Gaussian width of the edge (geometric + instrumental broadening):
 *           σ = λ₀ · sqrt( tan²(α₀) · σ_grid² + e0² )
 *     7.  Exponential tail decay length:
 *           α = tau(λ₀) / 10000
 *     8.  Ikeda-Carpenter edge profile evaluated at the current λ:
 *           if σ < 0.2·α  →  erfc-exponential form
 *           else           →  Gaussian approximation
 *     9.  Accumulate into output[k * N_Omega * N_lam + ω * N_lam + λ].
 */

/* ------------------------------------------------------------------
 * Helper: empirical pulse-tail parameter τ(λ)
 *   Polynomial fit from tau.m (Malamud):
 *     τ(x) = erf((x - 1.39341)/0.18492) · (18.94806 - 10.82914·x)
 *           + 16.6964·x
 * ------------------------------------------------------------------ */
static float tau_func(float x) {
    float erf_arg = (x - 1.39341f) / 0.18492f;
    /* erf via series / built-in */
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
 * amplitude – pre-computed kinematic amplitude
 *
 * Returns the cross-section contribution (cm²/… same units as amplitude).
 * ------------------------------------------------------------------ */
static float ic_profile(float lam, float lam0,
                        float sigma, float alpha, float amplitude) {
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
         * The product erfc(u)*exp((lam0-lam)/alpha) decays as
         *   exp(-(lam0-lam)^2/(2*sigma^2) + (lam0-lam)/alpha) for large u.
         * We bail when (lam0-lam)^2/(2*sigma^2) > 80 (float32 underflows erfc).
         */
        float dlam = lam - lam0;   /* negative when lam < lam0 */
        if (-dlam / (1.41421356f * sigma) > 9.0f) return 0.0f;

        float u = -dlam / (1.41421356f * sigma) + (sigma / alpha);
        float log_dec = -(lam - lam0) / alpha;  /* positive when lam < lam0 */
        /* cap combined exponent to avoid exp() overflow */
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
 * hkl_table (N_hkl,   5) float  – [h, k, l, F2_barns, d_Angstrom]
 * lam       (N_lam,)      float  – wavelength grid (Å), strictly increasing
 * out       (N_orient * N_Omega, N_lam) float  – output matrix (row-major)
 *
 * Scalars
 * -------
 * V_cm3     – unit-cell volume in cm³  (a³ × 1e-24)
 * sigma_grid – angular half-width of the orientation node in radians
 * e0         – instrumental exponential resolution parameter (dimensionless)
 * lam_min, lam_max – wavelength window (Å)
 * N_orient, N_Omega, N_lam, N_hkl – dimensions
 * ------------------------------------------------------------------ */
__kernel void bragg_edge_col(
    __global const float *rot_inv,     /* (N_orient, 9) */
    __global const float *beam_dirs,   /* (N_Omega,  3) */
    __global const float *hkl_table,   /* (N_hkl,    5)  h k l F2 d */
    __global const float *lam,         /* (N_lam,)       Å */
    __global       float *out,         /* (N_orient * N_Omega, N_lam) */

    const float  V_cm3,
    const float  sigma_grid,
    const float  e0,
    const float  lam_min,
    const float  lam_max,

    const int    N_orient,
    const int    N_Omega,
    const int    N_lam,
    const int    N_hkl
) {
    /* Each work-item handles one (k, ω) pair and one λ */
    int gid = get_global_id(0);
    int lam_idx = get_global_id(1);

    int N_pairs = N_orient * N_Omega;
    if (gid >= N_pairs || lam_idx >= N_lam) return;

    int k_idx   = gid / N_Omega;
    int om_idx  = gid % N_Omega;

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

    /* --- rotate beam into crystal frame: n = R_inv · v --- */
    float nx = R00*vx + R01*vy + R02*vz;
    float ny = R10*vx + R11*vy + R12*vz;
    float nz = R20*vx + R21*vy + R22*vz;

    float lam_val = lam[lam_idx];

    /* --- accumulate cross-section contributions from all reflections --- */
    float xs = 0.0f;

    for (int r = 0; r < N_hkl; ++r) {
        int rb2 = r * 5;
        float h  = hkl_table[rb2 + 0];
        float k  = hkl_table[rb2 + 1];
        float l  = hkl_table[rb2 + 2];
        float F2 = hkl_table[rb2 + 3];   /* barns */
        float d  = hkl_table[rb2 + 4];   /* Å     */

        float h2k2l2 = h*h + k*k + l*l;

        /* Projected Bragg wavelength for this orientation */
        float proj = h*nx + k*ny + l*nz;
        float lam0 = 2.0f * d * proj / h2k2l2;

        /* Only forward-scattering Bragg edges within the measurement window */
        if (lam0 <= 0.0f || lam0 < lam_min || lam0 > lam_max) continue;

        /* Bragg angle and incidence angle */
        float sin_tB = lam0 / (2.0f * d);
        if (sin_tB > 1.0f) sin_tB = 1.0f;   /* clamp floating-point rounding */
        float theta_B = asin(sin_tB);
        float cos_tB  = cos(theta_B);
        float alpha0  = M_PI_2_F - theta_B;  /* angle between beam and planes */
        float tan_a0  = tan(alpha0);

        /* Kinematic amplitude  [cm = cross-section per unit cell pair distance] */
        float lam0_cm  = lam0 * 1.0e-8f;   /* Å → cm */
        float F2_cm2   = F2  * 1.0e-24f;   /* barn → cm² */
        float sin2_tB  = sin_tB * sin_tB;
        float amplitude = 1.0e8f * (lam0_cm*lam0_cm*lam0_cm*lam0_cm) * F2_cm2
                          / (2.0f * V_cm3 * V_cm3 * sin2_tB);

        /* Edge width */
        float sg2 = sigma_grid * sigma_grid;
        float e02 = e0 * e0;
        float sigma = lam0 * sqrt(tan_a0 * tan_a0 * sg2 + e02);

        /* Exponential tail decay */
        float alpha = tau_func(lam0) / 10000.0f;

        xs += ic_profile(lam_val, lam0, sigma, alpha, amplitude);
    }

    /* Output index: row = k_idx * N_Omega + om_idx, col = lam_idx */
    int row = k_idx * N_Omega + om_idx;
    out[row * N_lam + lam_idx] = xs;
}
