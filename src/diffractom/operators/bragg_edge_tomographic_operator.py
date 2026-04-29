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
from ..crystallography.material import Material
from ..utils.grid import Grid
from ..utils.instrument import raden_pulse_tail


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
#   genera_matriz_bola_2022 + xs_singlecrystal_2022 → build_bragg_matrix_cpu
# ---------------------------------------------------------------------------


def build_bragg_matrix_cpu(
    bragg_table: dict,
    rotations: Rotation,
    beam_angles_deg,
    lam: np.ndarray,
    sig: float,
    e0: float,
    pulse_tail_fn=raden_pulse_tail,
) -> np.ndarray:
    """CPU implementation faithful to xs_text_full_cubic + xs_singlecrystal_2022.

    Replicates the MATLAB computation that produces ``matrix_15_*deg.txt``.
    Works for any crystal system — no cubic assumption is made.

    Parameters
    ----------
    bragg_table : dict
        Output of :meth:`~diffractom.crystallography.material.Material.neutron_bragg_table`.
        Required keys: ``g_vecs`` (N_hkl, 3), ``d`` (N_hkl,), ``F2`` (N_hkl,),
        ``V`` (unit-cell volume in Å³).  The g-vectors are reciprocal lattice
        vectors **without** the 2π factor (Å⁻¹).
    rotations : scipy.spatial.transform.Rotation, length N_orient
        Crystal orientations.  The function applies the inverse internally,
        matching ``rot = rotation.byEuler(inv(S3G_B)...)`` in MATLAB.
    beam_angles_deg : sequence of float
        Tomographic beam angles in degrees.  Beam direction for angle φ:
        ``R_z(φ) @ [0,1,0]``.
    lam : np.ndarray, shape (N_lam,)
        Wavelength grid in Å.
    sig : float
        Orientation spread in **radians** (``G_big`` in MATLAB).
    e0 : float
        Instrument resolution parameter (``e0 = 0.0001`` in MATLAB).
    pulse_tail_fn : callable, optional
        Function ``f(lam) -> tau`` that returns the instrument pulse-tail
        parameter τ(λ).  ``α = τ / 10000`` is the exponential decay length
        in the peak-shape formula.  Defaults to
        :func:`~diffractom.utils.instrument.raden_pulse_tail` (RADEN/J-PARC).
        Pass a different callable to use a different beamline model.

    Returns
    -------
    B : np.ndarray, shape (N_Omega, N_orient, N_lam)
    """
    from scipy.special import erfc as sp_erfc

    lam      = np.asarray(lam, dtype=np.float64)
    lam_min  = lam.min()
    lam_max  = lam.max()
    N_lam    = len(lam)
    N_orient = len(rotations)
    N_Omega  = len(beam_angles_deg)

    g_vecs = np.asarray(bragg_table['g_vecs'], dtype=np.float64)   # (N_hkl, 3) Å⁻¹
    d_hkl  = np.asarray(bragg_table['d'],      dtype=np.float64)   # (N_hkl,)   Å
    F2     = np.asarray(bragg_table['F2'],      dtype=np.float64)   # (N_hkl,)   barns
    V_cm3  = float(bragg_table['V']) * 1e-24                        # Å³ → cm³

    # |g|² for each reflection — used for the general Bragg wavelength formula
    g_sq = np.einsum('ij,ij->i', g_vecs, g_vecs)                   # (N_hkl,) Å⁻²

    R_inv_all = rotations.inv().as_matrix()                          # (N_orient, 3, 3)

    B = np.zeros((N_Omega, N_orient, N_lam), dtype=np.float64)

    for i_ang, ang_deg in enumerate(beam_angles_deg):
        ang_rad = np.deg2rad(ang_deg)
        # Beam direction in lab frame: rotate (0,1,0) around Z by ang_deg
        # MATLAB: vv = rotate(vector3d(0,1,0), axis2quat(vector3d(0,0,1), ang*degree))
        vv = np.array([-np.sin(ang_rad), np.cos(ang_rad), 0.0])

        for i_n in range(N_orient):
            # Beam direction in crystal frame (= rot.*HAZ_muestra in MATLAB)
            n_crystal = R_inv_all[i_n] @ vv                         # (3,)

            # General Bragg wavelength: λ₀ = 2(n̂·g)/|g|²
            # Derivation: Bragg condition λ₀ = 2d·sin(θB), with
            #   sin(θB) = (n̂·g)/|g|  → λ₀ = 2(n̂·g)/|g|²
            # Reduces to the MATLAB cubic formula when g = [h,k,l]/a.
            lam0 = 2.0 * (g_vecs @ n_crystal) / g_sq               # (N_hkl,)

            mask = (lam0 >= lam_min) & (lam0 <= lam_max)
            if not np.any(mask):
                continue

            lam0_m = lam0[mask]
            d_m    = d_hkl[mask]
            F2_m   = F2[mask]

            ratio   = np.clip(lam0_m / (2.0 * d_m), -1.0, 1.0)
            theta_B = np.arcsin(ratio)           # Bragg angle
            alpha0  = np.arccos(ratio)           # complement: angle between beam and planes
            sin2_tB = np.maximum(np.sin(theta_B)**2, 1e-30)

            # amplitude: 1E8*(lam0*1E-8)^4 * F2*1E-24 / (V^2 * 2 * sin²(θB))
            amplitude = (
                1e8 * (lam0_m * 1e-8)**4 * F2_m * 1e-24
                / (V_cm3**2 * 2.0 * sin2_tB)
            )

            # Peak shape from xs_singlecrystal_2022.m:
            #   σ_g = λ₀ · √(tan²(α₀)·sig² + e₀²)   [MATLAB col 11]
            #   α   = τ(λ₀) / 10000                   [MATLAB col 12]
            sigma_g = lam0_m * np.sqrt(np.tan(alpha0)**2 * sig**2 + e0**2)
            alfa    = pulse_tail_fn(lam0_m) / 10000.0

            kk  = sigma_g <  0.2 * alfa    # erfc × exponential form
            kk1 = ~kk                      # Gaussian approximation

            if np.any(kk):
                dlam    = lam[np.newaxis, :] - lam0_m[kk, np.newaxis]
                s       = sigma_g[kk, np.newaxis]
                al      = alfa[kk, np.newaxis]
                u       = -dlam / (np.sqrt(2.0) * s) + s / al
                exp_arg = np.clip(
                    (s / (np.sqrt(2.0) * al))**2 - dlam / al, -800.0, 709.0
                )
                y5 = (
                    amplitude[kk, np.newaxis]
                    * sp_erfc(u)
                    * (1.0 / (2.0 * al))
                    * np.exp(exp_arg)
                )
                B[i_ang, i_n] += np.sum(y5, axis=0)

            if np.any(kk1):
                dlam = lam[np.newaxis, :] - lam0_m[kk1, np.newaxis]
                s    = sigma_g[kk1, np.newaxis]
                y6   = (
                    amplitude[kk1, np.newaxis]
                    * (1.0 / (np.sqrt(2.0 * np.pi) * s))
                    * np.exp(-dlam**2 / (2.0 * s**2))
                )
                B[i_ang, i_n] += np.sum(y6, axis=0)

    return B


# ---------------------------------------------------------------------------
# OpenCL matrix builder
# ---------------------------------------------------------------------------

def build_bragg_matrix_gpu(
    bragg_table: dict,
    rotations: Rotation,
    beam_angles_deg: np.ndarray,
    lam: np.ndarray,
    sig: float,
    e0: float,
    pulse_tail_fn=raden_pulse_tail,
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
    bragg_table : dict
        Output of :meth:`~diffractom.crystallography.material.Material.neutron_bragg_table`.
    rotations : scipy Rotation, shape (N_orient,)
    beam_angles_deg : np.ndarray, shape (N_Omega,)
        Tomographic angles in degrees.
    lam : np.ndarray, shape (N_lam,)
        Wavelength grid in Å.
    sig : float
        Orientation spread in radians (G_big).
    e0 : float
        Instrument resolution parameter.
    pulse_tail_fn : callable, optional
        Instrument pulse-tail function — see :func:`build_bragg_matrix_cpu`.
        Defaults to :func:`~diffractom.utils.instrument.raden_pulse_tail`.
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
    material : Material
        A :class:`~diffractom.crystallography.material.Material` with neutron
        sites set via :meth:`~diffractom.crystallography.material.Material.set_neutron_sites`
        or built with :meth:`~diffractom.crystallography.material.Material.from_neutron_sites`.
        The class handles reflection enumeration and structure-factor computation;
        no material-specific logic remains in the operator itself.
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
    pulse_tail_fn : callable, optional
        Instrument pulse-tail function ``f(lam) -> tau``.  Defaults to
        :func:`~diffractom.utils.instrument.raden_pulse_tail` (RADEN/J-PARC).
        Pass a different callable for a different beamline.
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
        material: Material,
        grid: Grid,
        beam_angles: np.ndarray,
        lam: np.ndarray,
        sigma_grid: float | None = None,
        e0: float = 1e-4,
        pulse_tail_fn=raden_pulse_tail,
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

        # --- build neutron Bragg table from material ---
        # Material handles reflection enumeration, d-spacings, and |F|²;
        # the operator is not aware of lattice parameters or atom positions.
        bragg_table = material.neutron_bragg_table(
            lam_min   = float(lam.min()),
            lam_max   = float(lam.max()),
            h_max     = h_max,
            threshold = threshold,
        )

        # --- build B ---
        if use_gpu:
            B = build_bragg_matrix_gpu(
                bragg_table    = bragg_table,
                rotations      = rotations,
                beam_angles_deg= beam_angles,
                lam            = lam,
                sig            = sigma_grid,
                e0             = e0,
                pulse_tail_fn  = pulse_tail_fn,
                ctx            = ctx,
                queue          = queue,
            )
        else:
            B = build_bragg_matrix_cpu(
                bragg_table    = bragg_table,
                rotations      = rotations,
                beam_angles_deg= beam_angles,
                lam            = lam,
                sig            = sigma_grid,
                e0             = e0,
                pulse_tail_fn  = pulse_tail_fn,
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
        self.pulse_tail_fn = pulse_tail_fn
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
