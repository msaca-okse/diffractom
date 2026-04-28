"""Bragg-edge tomographic operator for time-of-flight neutron imaging.

Builds the operator matrix B internally from crystallographic and
experimental parameters, then delegates forward/adjoint computation to
:class:`~diffractom.operators.MatrixTomographicOperator`.

The matrix B has shape ``(N_Omega, K, N_lam)`` where:

* ``N_Omega`` – number of tomographic angles
* ``K``       – number of orientation basis functions (N_orient, or N_orient+1
  if a powder column is appended)
* ``N_lam``   – number of wavelength bins

Physics
-------
For each orientation k and tomographic angle ω the kernel computes:

.. math::

    B[\\omega, k, \\lambda] = \\sum_{hkl} \\Phi_{hkl}(\\lambda; \\lambda_0(k, \\omega))

where :math:`\\Phi_{hkl}` is an Ikeda-Carpenter Bragg-edge profile and
:math:`\\lambda_0` is the edge wavelength for the given beam direction in
the crystal frame of orientation k.

Usage example
-------------
::

    import numpy as np
    from diffractom.crystallography.neutron_material import NeutronMaterial
    from diffractom.operators import BraggEdgeTomographicOperator
    from diffractom.utils import Grid

    material = NeutronMaterial.steel_316L()
    # Build an equispaced orientation grid with ~185 nodes for cubic symmetry
    grid = Grid.from_equispaced_fundamental_zone(
        resolution_deg=15, sigma=np.deg2rad(15))
    lam = np.linspace(0.5, 6.5, 1000)          # wavelength grid in Å
    beam_angles = np.array([0, 30, 45, 60, 90, 120, 135, 150, 180])  # degrees
    powder = np.ones_like(lam) * 0.3            # placeholder powder cross-section

    K = len(grid.active_leaf_nodes())           # number of orientation columns
    op = BraggEdgeTomographicOperator(
        material=material,
        grid=grid,
        beam_angles=beam_angles,
        lam=lam,
        # sigma_grid is read from grid by default; pass explicitly to override
        e0=1e-4,
        powder_xs=powder,
        include_powder=True,
        # --- MatrixTomographicOperator args ---
        angles=np.deg2rad(beam_angles),
        N_Omega=len(beam_angles),
        My=512,
        Nx=256,
        Ny=256,
        K=K + 1,                # +1 for powder
        N_seg=len(lam),
    )
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Sequence

import pyopencl as cl
import pyopencl.array as clarray
from scipy.spatial.transform import Rotation

from .matrix_tomographic_operator import MatrixTomographicOperator
from ..crystallography.neutron_material import NeutronMaterial
from ..utils.grid import Grid


# ---------------------------------------------------------------------------
# OpenCL program loader
# ---------------------------------------------------------------------------

def _build_bragg_program(ctx: cl.Context) -> cl.Program:
    """Compile bragg_edge_kernels.cl for the given OpenCL context."""
    cl_path = Path(__file__).with_name("bragg_edge_kernels.cl")
    src = cl_path.read_text()
    return cl.Program(ctx, src).build()


# ---------------------------------------------------------------------------
# Pure-Python reference implementation (CPU)
# Faithful replication of the MATLAB functions:
#   factor.m            → _structure_factor_sq
#   genera_indices_2022 → compute_bragg_A1
#   tau.m               → _tau
#   genera_matriz_bola_2022 + xs_singlecrystal_2022 → build_bragg_matrix_cpu
# ---------------------------------------------------------------------------

def _tau(x: np.ndarray) -> np.ndarray:
    """Empirical pulse-tail parameter τ(λ) — direct translation of tau.m.

    tau.m::
        p1=1.39341; p2=0.18492; p3=18.94806; p4=-10.82914; p5=16.6964;
        tau=erf(((x-p1)/p2)).*(p3+p4*x)+p5.*x;
    """
    from scipy.special import erf as sp_erf
    p1, p2 = 1.39341, 0.18492
    p3, p4, p5 = 18.94806, -10.82914, 16.6964
    return sp_erf((x - p1) / p2) * (p3 + p4 * x) + p5 * x


def compute_bragg_A1(
    a: float,
    atoms: np.ndarray,
    threshold: float = 0.001,
) -> np.ndarray:
    """Build the HKL reflection table — replica of genera_indices_2022(a, B).

    Enumerates all (h, k, l) with h, k, l ∈ [-10, 10], computes |F|² using
    the same formula as ``factor.m``, and returns reflections with |F|² above
    ``threshold``, sorted by |F|² descending.

    Parameters
    ----------
    a : float
        Cubic lattice parameter in Å.
    atoms : np.ndarray, shape (N_atoms, 5)
        Each row: [x, y, z, b_coh_fm, u2_A2] — fractional coordinates,
        coherent scattering length in fm, and mean-square displacement in Å².
        Matches the column layout of ``atomos_316L_alloy.txt`` (without the
        leading atom-index column).
    threshold : float
        Minimum |F|² in barns to include a reflection.  The MATLAB code uses
        ``find(AA(:,4) > 0.001)``.

    Returns
    -------
    A1 : np.ndarray, shape (N_hkl, 4)
        Columns: [h, k, l, F2_barns], sorted by F2 descending.
    """
    # ----- enumerate all (h,k,l) in [-10,10]^3 excluding (0,0,0) -----
    h_range = np.arange(-10, 11, dtype=np.float64)
    hh, kk, ll = np.meshgrid(h_range, h_range, h_range, indexing="ij")
    h = hh.ravel();  k = kk.ravel();  l = ll.ravel()
    nonzero = (h != 0) | (k != 0) | (l != 0)
    h, k, l = h[nonzero], k[nonzero], l[nonzero]

    # ----- structure factor |F|² — replica of factor.m -----
    # factor.m uses b in fm;  SF = (xre² + xim²) * 0.01 converts fm² → barns
    d   = a / np.sqrt(h**2 + k**2 + l**2)   # d-spacing in Å
    q   = 2.0 * np.pi / d                     # q in Å⁻¹
    q2  = q**2

    u2_ref = atoms[0, 4]   # u² of first atom (same as B(1,6) in factor.m)
    xre = np.zeros(len(h), dtype=np.float64)
    xim = np.zeros(len(h), dtype=np.float64)

    for i in range(len(atoms)):
        xi, yi, zi, bi, u2_i = atoms[i, 0], atoms[i, 1], atoms[i, 2], atoms[i, 3], atoms[i, 4]
        qd = 2.0 * np.pi * (h * xi + k * yi + l * zi)
        # factor.m: exp(-q2*(B(1,6)*0.5)) * (B(i,5) * exp(-q2*(B(i,6)-B(1,6))*0.5) * cos/sin(qd))
        dw = np.exp(-q2 * u2_ref * 0.5) * np.exp(-q2 * (u2_i - u2_ref) * 0.5)
        xre += dw * bi * np.cos(qd)
        xim += dw * bi * np.sin(qd)

    SF = (xre**2 + xim**2) * 0.01   # convert fm² → barns

    # ----- filter and sort (replicates the logic of genera_indices_2022) -----
    mask = SF > threshold
    h, k, l, SF = h[mask], k[mask], l[mask], SF[mask]
    order = np.argsort(-SF)

    return np.column_stack([h[order], k[order], l[order], SF[order]])


def build_bragg_matrix_cpu(
    A1: np.ndarray,
    a: float,
    rotations: Rotation,
    beam_angles_deg,
    lam: np.ndarray,
    sig: float,
    e0: float,
) -> np.ndarray:
    """CPU implementation faithful to xs_text_full_cubic + xs_singlecrystal_2022.

    Replicates the MATLAB computation that produces ``matrix_15_*deg.txt``.

    Parameters
    ----------
    A1 : np.ndarray, shape (N_hkl, 4)
        HKL table from :func:`compute_bragg_A1`: columns [h, k, l, F2_barns].
    a : float
        Cubic lattice parameter in Å.  (``a = 3.596`` for 316L.)
    rotations : scipy.spatial.transform.Rotation, length N_orient
        Crystal orientations corresponding to ``S3G_B`` in MATLAB.
        These are the **forward** orientations (NOT pre-inverted); the function
        applies the inverse internally, matching ``rot = rotation.byEuler(inv(S3G_B)...)``
        in ``xs_text_full_cubic.m``.
    beam_angles_deg : sequence of float
        Tomographic beam angles in degrees.  The beam direction for angle φ
        is ``R_z(φ) @ [0,1,0]`` (MATLAB: ``rotate(vector3d(0,1,0), axis2quat(ẑ,φ))``).
    lam : np.ndarray, shape (N_lam,)
        Wavelength grid in Å.
    sig : float
        Orientation spread in **radians**.  Corresponds to ``G_big`` in MATLAB.
    e0 : float
        Instrument resolution parameter (dimensionless, ~1e-4).
        (MATLAB: ``e0 = 0.0001``.)

    Returns
    -------
    B : np.ndarray, shape (N_Omega, N_orient, N_lam)
        ``B[i_ang, i_n, :]`` is the single-crystal cross-section spectrum
        for beam angle ``beam_angles_deg[i_ang]`` and orientation
        ``rotations[i_n]``.
    """
    from scipy.special import erfc as sp_erfc

    lam       = np.asarray(lam, dtype=np.float64)
    lam_min   = lam.min()
    lam_max   = lam.max()
    N_lam     = len(lam)
    N_orient  = len(rotations)
    N_Omega   = len(beam_angles_deg)

    V_cm3 = (a * 1e-8) ** 3          # unit cell volume in cm³  (matches genera_matriz_bola_2022)

    h,  k,  l_  = A1[:, 0], A1[:, 1], A1[:, 2]
    F2          = A1[:, 3]            # |F|² in barns
    hkl_sq      = h**2 + k**2 + l_**2

    # Pre-compute inverse rotation matrices (= R^T for SO(3))
    R_inv_all = rotations.inv().as_matrix()   # (N_orient, 3, 3)

    B = np.zeros((N_Omega, N_orient, N_lam), dtype=np.float64)

    for i_ang, ang_deg in enumerate(beam_angles_deg):
        ang_rad = np.deg2rad(ang_deg)
        # Beam direction in lab frame: rotate (0,1,0) around Z by ang_deg
        # MATLAB: vv = rotate(vector3d(0,1,0), axis2quat(vector3d(0,0,1), ang*degree))
        vv = np.array([-np.sin(ang_rad), np.cos(ang_rad), 0.0])

        for i_n in range(N_orient):
            # Beam direction in crystal frame of orientation i_n
            # MATLAB: HH = rot .* HAZ_muestra(m)  where rot = rotation.byEuler(inv(S3G_B)...)
            n_crystal = R_inv_all[i_n] @ vv          # (3,)
            a1c, a2c, a3c = n_crystal

            # ---------- genera_matriz_bola_2022 ----------
            # lam0 = 2*a * (h*a1 + k*a2 + l*a3) / (h²+k²+l²)
            lam0 = 2.0 * a * (h * a1c + k * a2c + l_ * a3c) / hkl_sq

            # Keep reflections whose Bragg wavelength falls in [lam_min, lam_max]
            mask = (lam0 >= lam_min) & (lam0 <= lam_max)
            if not np.any(mask):
                continue

            lam0_m  = lam0[mask]
            F2_m    = F2[mask]
            hkl_sq_m = hkl_sq[mask]

            d_hkl = a / np.sqrt(hkl_sq_m)              # d-spacing in Å
            ratio = np.clip(lam0_m / (2.0 * d_hkl), -1.0, 1.0)
            theta_B = np.arcsin(ratio)                  # Bragg angle
            alpha0  = np.arccos(ratio)                  # = acos(lam0/(2d)) as in MATLAB

            # Guard sin²(theta_B) = 0 (grazing)
            sin2_tB = np.maximum(np.sin(theta_B)**2, 1e-30)

            # amplitude: 1E8*(lam0*1E-8)^4 * F2*1E-24 / (V^2 * 2 * sin²(thetaB))
            amplitude = (
                1e8 * (lam0_m * 1e-8)**4 * F2_m * 1e-24
                / (V_cm3**2 * 2.0 * sin2_tB)
            )

            # ---------- xs_singlecrystal_2022 ----------
            # sigma_g = lam0 * sqrt(tan²(alpha0)*sig² + e0²)    [col 11 in MATLAB]
            # alfa    = tau(lam0) / 10000                         [col 12 in MATLAB]
            sigma_g = lam0_m * np.sqrt(np.tan(alpha0)**2 * sig**2 + e0**2)
            alfa    = _tau(lam0_m) / 10000.0

            # Two groups (same split as xs_singlecrystal_2022.m):
            #   kk  : sigma_g <  0.2 * alfa  → erfc × exponential form
            #   kk1 : sigma_g >= 0.2 * alfa  → Gaussian approximation
            kk  = sigma_g <  0.2 * alfa
            kk1 = ~kk

            # Broadcast over wavelengths: shape (M_refl, N_lam)
            if np.any(kk):
                dlam = lam[np.newaxis, :] - lam0_m[kk, np.newaxis]   # (M, N_lam)
                s  = sigma_g[kk, np.newaxis]
                al = alfa[kk, np.newaxis]
                u  = -dlam / (np.sqrt(2.0) * s) + s / al
                # exponent = (s/(sqrt(2)*al))^2 - dlam/al
                exp_arg = (s / (np.sqrt(2.0) * al))**2 - dlam / al
                # Guard overflow: when exp_arg > 709 numpy overflows float64
                exp_arg = np.clip(exp_arg, -800.0, 709.0)
                y5 = (amplitude[kk, np.newaxis]
                      * sp_erfc(u)
                      * (1.0 / (2.0 * al))
                      * np.exp(exp_arg))
                B[i_ang, i_n] += np.sum(y5, axis=0)

            if np.any(kk1):
                dlam = lam[np.newaxis, :] - lam0_m[kk1, np.newaxis]  # (M, N_lam)
                s    = sigma_g[kk1, np.newaxis]
                y6   = (amplitude[kk1, np.newaxis]
                        * (1.0 / (np.sqrt(2.0 * np.pi) * s))
                        * np.exp(-dlam**2 / (2.0 * s**2)))
                B[i_ang, i_n] += np.sum(y6, axis=0)

    return B


# ---------------------------------------------------------------------------
# OpenCL matrix builder
# ---------------------------------------------------------------------------

def build_bragg_matrix_gpu(
    A1: np.ndarray,
    a: float,
    rotations: Rotation,
    beam_angles_deg: np.ndarray,
    lam: np.ndarray,
    sig: float,
    e0: float,
    ctx: cl.Context | None = None,
    queue: cl.CommandQueue | None = None,
) -> np.ndarray:
    """GPU-accelerated Bragg-edge matrix builder.

    .. note::
        The OpenCL kernel (``bragg_edge_kernels.cl``) still uses the old
        formula and needs to be updated to match ``xs_singlecrystal_2022.m``.
        Use :func:`build_bragg_matrix_cpu` for validated results.

    Parameters
    ----------
    A1 : np.ndarray, shape (N_hkl, 4)
        HKL table from :func:`compute_bragg_A1`: columns [h, k, l, F2_barns].
    a : float
        Cubic lattice parameter in Å.
    rotations : scipy Rotation, shape (N_orient,)
    beam_angles_deg : np.ndarray, shape (N_Omega,)
        Tomographic angles in degrees.
    lam : np.ndarray, shape (N_lam,)
        Wavelength grid in Å.
    sig : float
        Orientation spread in radians (G_big).
    e0 : float
        Instrument resolution parameter.
    ctx, queue : optional
        Existing OpenCL context/command-queue.

    Returns
    -------
    B : np.ndarray, shape (N_Omega, N_orient, N_lam), dtype float32
    """
    raise NotImplementedError(
        "build_bragg_matrix_gpu needs to be updated to match xs_singlecrystal_2022.m. "
        "Use build_bragg_matrix_cpu for validated results."
    )

    if ctx is None:
        ctx = cl.create_some_context(interactive=False)
    if queue is None:
        queue = cl.CommandQueue(ctx)

    prg = _build_bragg_program(ctx)
    kernel = cl.Kernel(prg, "bragg_edge_col")

    # --- HKL table ---
    hkl_tab = A1.astype(np.float32)
    N_hkl = len(hkl_tab)

    # --- Rotation inverse matrices ---
    rot_inv_mats = rotations.inv().as_matrix().reshape(-1, 9).astype(np.float32)  # (N_orient, 9)
    N_orient = len(rotations)

    # --- Beam directions: rotate ŷ around ẑ by each angle ---
    beam_dirs = np.stack([
        np.array([
            -np.sin(np.deg2rad(ang)),
            np.cos(np.deg2rad(ang)),
            0.0,
        ], dtype=np.float32)
        for ang in beam_angles_deg
    ])  # (N_Omega, 3)
    N_Omega = len(beam_angles_deg)

    lam_f32 = lam.astype(np.float32)
    N_lam = len(lam)
    lam_min = float(lam_f32.min())
    lam_max = float(lam_f32.max())
    V_cm3 = float((a * 1e-8) ** 3)

    # --- upload to GPU ---
    d_rot_inv  = clarray.to_device(queue, rot_inv_mats)
    d_beams    = clarray.to_device(queue, beam_dirs.ravel())
    d_hkl      = clarray.to_device(queue, hkl_tab.ravel())
    d_lam      = clarray.to_device(queue, lam_f32)
    d_out      = clarray.zeros(queue, (N_orient * N_Omega * N_lam,), dtype=np.float32)

    # --- dispatch ---
    # Global size: (N_orient * N_Omega, N_lam)
    global_size = (N_orient * N_Omega, N_lam)

    kernel(
        queue,
        global_size,
        None,                           # local size: auto
        d_rot_inv.data,
        d_beams.data,
        d_hkl.data,
        d_lam.data,
        d_out.data,
        np.float32(V_cm3),
        np.float32(sig),
        np.float32(e0),
        np.float32(lam_min),
        np.float32(lam_max),
        np.int32(N_orient),
        np.int32(N_Omega),
        np.int32(N_lam),
        np.int32(N_hkl),
    )
    queue.finish()

    # Retrieve and reshape: raw shape is (N_orient * N_Omega, N_lam)
    B_flat = d_out.get().reshape(N_orient, N_Omega, N_lam)
    # Transpose to (N_Omega, N_orient, N_lam)
    B = B_flat.transpose(1, 0, 2).astype(np.float32)
    return B


# ---------------------------------------------------------------------------
# Operator class
# ---------------------------------------------------------------------------

class BraggEdgeTomographicOperator(MatrixTomographicOperator):
    """Neutron Bragg-edge tomographic forward operator.

    Builds the B matrix from crystallographic parameters and tomographic
    geometry, then delegates all forward/adjoint computation to the parent
    :class:`~diffractom.operators.MatrixTomographicOperator`.

    Parameters
    ----------
    material : NeutronMaterial
        Crystal with neutron scattering parameters.
    grid : Grid
        Orientation grid whose active leaf nodes define the K basis functions.
        Sigma (angular half-width) is read from the nodes; pass ``sigma_grid``
        explicitly to override.  Use :meth:`Grid.from_equispaced_fundamental_zone`
        for an approximately equispaced cubic grid or
        :meth:`Grid.from_euler_angles` to load MTEX-exported orientations.
    beam_angles : array_like, shape (N_Omega,) — degrees
        Tomographic sample angles.  The neutron beam direction for angle φ
        is computed as :math:`R_z(\\phi) @ \\hat{y}`.
    lam : np.ndarray, shape (N_lam,)
        Wavelength grid in Å.
    sigma_grid : float or None
        Angular half-width of the orientation grid nodes in radians.
        Corresponds to ``G_big`` (in rad) in the MATLAB code.  If *None*
        (default), the value is taken from ``grid.nodes[leaf_0].sigma``.
    e0 : float
        Instrument exponential resolution parameter (dimensionless, ~1e-4).
    powder_xs : np.ndarray or None, shape (N_lam,)
        Powder-averaged cross-section spectrum.  Required if
        ``include_powder=True``.
    include_powder : bool
        If True (default False), append ``powder_xs`` as the last column of
        B, giving ``K = N_orient + 1`` basis functions.
    h_max : int
        Maximum |Miller index| when enumerating reflections.
    threshold : float
        Minimum |F|² (barns) to include a reflection.
    use_gpu : bool
        If True (default), dispatch OpenCL kernel; if False, fall back to the
        pure-numpy CPU reference.  Note: the GPU path currently raises
        ``NotImplementedError`` until the kernel is updated to match
        ``xs_singlecrystal_2022.m``.  Set ``use_gpu=False`` for validated results.

    Additional keyword arguments
    ----------------------------
    All remaining keyword arguments are forwarded verbatim to
    :class:`~diffractom.operators.MatrixTomographicOperator`.  In particular
    you must supply: ``angles`` (radians, shape N_Omega), ``N_Omega``, ``My``,
    ``Nx``, ``Ny``.  ``K`` and ``N_seg`` are inferred automatically.

    Notes
    -----
    ``N_seg`` in the parent class corresponds to ``N_lam`` here.
    ``K`` equals ``N_orient`` (+ 1 if ``include_powder=True``).
    """

    def __init__(
        self,
        material: NeutronMaterial,
        grid: Grid,
        beam_angles: np.ndarray,
        lam: np.ndarray,
        sigma_grid: float | None = None,
        e0: float = 1e-4,
        powder_xs: np.ndarray | None = None,
        include_powder: bool = False,
        h_max: int = 10,
        threshold: float = 1e-3,
        use_gpu: bool = True,
        ctx: cl.Context | None = None,
        queue: cl.CommandQueue | None = None,
        **parent_kwargs,
    ):
        # --- extract orientations and sigma from the grid ---
        active_indices = grid.active_leaf_nodes()
        if not active_indices:
            raise ValueError("Grid has no active leaf nodes.")
        nodes = [grid.nodes[i] for i in active_indices]
        rotations = Rotation.concatenate([n.R for n in nodes])
        if sigma_grid is None:
            sigma_grid = nodes[0].sigma

        beam_angles = np.asarray(beam_angles, dtype=np.float64)
        lam = np.asarray(lam, dtype=np.float64)

        N_orient = len(rotations)
        N_Omega = len(beam_angles)
        N_lam = len(lam)

        if include_powder and powder_xs is None:
            raise ValueError(
                "powder_xs must be provided when include_powder=True"
            )
        if include_powder and len(powder_xs) != N_lam:
            raise ValueError(
                f"powder_xs length {len(powder_xs)} does not match N_lam={N_lam}"
            )

        # --- build context/queue early so we can pass them to parent ---
        if ctx is None:
            ctx = cl.create_some_context(interactive=False)
        if queue is None:
            queue = cl.CommandQueue(ctx)

        # --- compute HKL reflection table (genera_indices_2022 equivalent) ---
        A1 = compute_bragg_A1(material.a, material.atoms, threshold=threshold)

        # --- build B ---
        if use_gpu:
            B = build_bragg_matrix_gpu(
                A1             = A1,
                a              = material.a,
                rotations      = rotations,
                beam_angles_deg= beam_angles,
                lam            = lam,
                sig            = sigma_grid,
                e0             = e0,
                ctx            = ctx,
                queue          = queue,
            )
        else:
            B = build_bragg_matrix_cpu(
                A1             = A1,
                a              = material.a,
                rotations      = rotations,
                beam_angles_deg= beam_angles,
                lam            = lam,
                sig            = sigma_grid,
                e0             = e0,
            ).astype(np.float32)

        if include_powder:
            # powder_xs shape: (N_lam,)
            # tile to (N_Omega, 1, N_lam) and concatenate along axis=1
            powder_col = np.tile(
                powder_xs.astype(np.float32)[np.newaxis, np.newaxis, :],
                (N_Omega, 1, 1),
            )
            B = np.concatenate([B, powder_col], axis=1)

        K = B.shape[1]

        # --- store build metadata ---
        self.material = material
        self.grid = grid
        self.rotations = rotations
        self.beam_angles = beam_angles
        self.lam = lam
        self.sigma_grid = sigma_grid
        self.e0 = e0
        self.include_powder = include_powder

        # --- delegate to parent ---
        super().__init__(
            B=B,
            N_Omega=N_Omega,
            K=K,
            N_seg=N_lam,
            ctx=ctx,
            queue=queue,
            **parent_kwargs,
        )
