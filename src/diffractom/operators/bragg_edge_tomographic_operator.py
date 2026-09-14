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

from scipy.stats import qmc



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
    edge_blur_scale: float = 1.0,
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
    edge_blur_scale : float, optional
        Scalar multiplier applied to the symmetric edge broadening term.
        Use values below 1.0 for sharper edges and above 1.0 for broader
        edges.  Must be positive.

    Returns
    -------
    B : np.ndarray, shape (N_Omega, N_orient, N_lam)
        Macroscopic elastic coherent attenuation coefficient μ_R(λ, τ) in cm⁻¹,
        equivalent to n·σ.  The number density n = 1/V_cell is not hardcoded —
        it is derived from the unit-cell volume in *bragg_table* (which in turn
        comes from :meth:`Material.neutron_bragg_table`).
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

    edge_blur_scale = float(edge_blur_scale)
    if edge_blur_scale <= 0.0:
        raise ValueError("edge_blur_scale must be positive")

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

            # Macroscopic attenuation amplitude μ_hkl (cm⁻¹):
            #   μ_hkl = n · λ₀⁴ · |F|² / (V · 2 · sin²θB)
            # where n = 1/V, so the denominator has V² overall.
            amplitude = (
                1e8 * (lam0_m * 1e-8)**4 * F2_m * 1e-24
                / (V_cm3**2 * 2.0 * sin2_tB)
            )

            # Peak shape from xs_singlecrystal_2022.m:
            #   σ_g = λ₀ · √(tan²(α₀)·sig² + e₀²)   [MATLAB col 11]
            #   α   = τ(λ₀) / 10000                   [MATLAB col 12]

            sigma_g = (
                lam0_m * np.sqrt(np.tan(alpha0)**2 * sig**2 + e0**2)
                * edge_blur_scale
            )
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
    edge_blur_scale: float = 1.0,
    ctx: cl.Context | None = None,
    queue: cl.CommandQueue | None = None,
    local_size: int = 64,
) -> np.ndarray:
    """GPU-accelerated Bragg-edge matrix builder.

    Faithful GPU replication of :func:`build_bragg_matrix_cpu` (same physics,
    same general reciprocal-lattice-vector formulation — no cubic-crystal
    assumption). See ``bragg_edge_kernels.cl`` for the kernel implementation
    and its performance notes.

    .. note::
        The kernel hard-codes the RADEN/J-PARC pulse-tail formula
        (:func:`~diffractom.utils.instrument.raden_pulse_tail`). Passing a
        different ``pulse_tail_fn`` raises ``ValueError`` since it would
        silently be ignored on the GPU path.

    Parameters
    ----------
    bragg_table : dict
        Output of :meth:`~diffractom.crystallography.material.Material.neutron_bragg_table`.
        Required keys: ``g_vecs`` (N_hkl, 3), ``d`` (N_hkl,), ``F2`` (N_hkl,), ``V``.
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
        Must be :func:`~diffractom.utils.instrument.raden_pulse_tail` (the
        only model the kernel implements).
    edge_blur_scale : float, optional
        Scalar multiplier applied to the symmetric edge broadening term.
    ctx, queue : optional
        Existing OpenCL context/command-queue.
    local_size : int, optional
        Number of wavelength bins processed per work-group. All threads in a
        work-group share the same (orientation, angle) pair and cooperatively
        cache the per-reflection trigonometry in local memory, so this value
        mainly trades off local-memory reuse against occupancy; 64 is a
        reasonable default on most GPUs.

    Returns
    -------
    B : np.ndarray, shape (N_Omega, N_orient, N_lam), dtype float32
        Macroscopic attenuation coefficient μ_R (cm⁻¹).  See
        :func:`build_bragg_matrix_cpu` for details.
    """
    if pulse_tail_fn is not raden_pulse_tail:
        raise ValueError(
            "build_bragg_matrix_gpu only supports the RADEN pulse-tail "
            "formula (raden_pulse_tail), which is hard-coded in "
            "bragg_edge_kernels.cl. Use use_gpu=False for other instruments."
        )

    if ctx is None:
        ctx = cl.create_some_context(interactive=False)
    if queue is None:
        queue = cl.CommandQueue(ctx)

    prg = _build_bragg_program(ctx)
    kernel = cl.Kernel(prg, "bragg_edge_col")

    # --- reflection table: [gx, gy, gz, F2, d] ---
    g_vecs = np.asarray(bragg_table['g_vecs'], dtype=np.float32)   # (N_hkl, 3) Å⁻¹
    d_hkl  = np.asarray(bragg_table['d'],      dtype=np.float32)   # (N_hkl,)   Å
    F2     = np.asarray(bragg_table['F2'],     dtype=np.float32)   # (N_hkl,)   barns
    # NOTE: V is passed to the kernel in Å³ (NOT converted to cm³ here). The
    # cm-scale unit-conversion factors in the amplitude formula cancel
    # exactly (10⁸·10⁻³²·10⁻²⁴/10⁻⁴⁸ = 1), so the kernel computes the
    # amplitude directly in Å/barn units. Converting V to cm³ here and
    # squaring it in float32 (~1e-23 cm³ → ~1e-46) sits right at the edge of
    # the float32 subnormal range and previously produced NaN/Inf.
    V_A3   = float(bragg_table['V'])                                 # Å³

    N_hkl = len(d_hkl)
    hkl_tab = np.empty((N_hkl, 5), dtype=np.float32)
    hkl_tab[:, 0:3] = g_vecs
    hkl_tab[:, 3]   = F2
    hkl_tab[:, 4]   = d_hkl

    # --- Rotation inverse matrices ---
    rot_inv_mats = rotations.inv().as_matrix().reshape(-1, 9).astype(np.float32)  # (N_orient, 9)
    N_orient = len(rotations)

    # --- Beam directions: rotate ŷ around ẑ by each angle ---
    ang_rad = np.deg2rad(np.asarray(beam_angles_deg, dtype=np.float64))
    beam_dirs = np.stack(
        [-np.sin(ang_rad), np.cos(ang_rad), np.zeros_like(ang_rad)], axis=1
    ).astype(np.float32)  # (N_Omega, 3)
    N_Omega = len(beam_angles_deg)

    lam_f32 = np.asarray(lam, dtype=np.float32)
    N_lam = len(lam_f32)
    lam_min = float(lam_f32.min())
    lam_max = float(lam_f32.max())

    if N_hkl == 0:
        return np.zeros((N_Omega, N_orient, N_lam), dtype=np.float32)

    # --- per-reflection scratch arrays live in local memory, shared by every
    # thread in a work-group (one work-group per (orientation, angle) pair) ---
    local_bytes = N_hkl * np.dtype(np.float32).itemsize
    dev_local_mem = ctx.devices[0].local_mem_size
    if 4 * local_bytes > dev_local_mem:
        raise RuntimeError(
            f"Reflection table too large for GPU local memory: needs "
            f"{4 * local_bytes} bytes (4 arrays of N_hkl={N_hkl} floats) but "
            f"device only has {dev_local_mem} bytes. Reduce h_max/threshold "
            f"to shrink N_hkl, or use use_gpu=False."
        )

    # --- upload to GPU ---
    d_rot_inv = clarray.to_device(queue, rot_inv_mats)
    d_beams   = clarray.to_device(queue, beam_dirs)
    d_hkl     = clarray.to_device(queue, hkl_tab)
    d_lam     = clarray.to_device(queue, lam_f32)
    d_out     = clarray.zeros(queue, (N_orient * N_Omega * N_lam,), dtype=np.float32)

    N_pairs = N_orient * N_Omega
    L = max(1, min(local_size, N_lam))
    N_lam_padded = int(np.ceil(N_lam / L)) * L

    kernel(
        queue,
        (N_pairs, N_lam_padded),
        (1, L),
        d_rot_inv.data,
        d_beams.data,
        d_hkl.data,
        d_lam.data,
        d_out.data,
        cl.LocalMemory(local_bytes),
        cl.LocalMemory(local_bytes),
        cl.LocalMemory(local_bytes),
        cl.LocalMemory(local_bytes),
        np.float32(V_A3),
        np.float32(sig),
        np.float32(e0),
        np.float32(edge_blur_scale),
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



def build_bragg_matrix_quadrature(
    bragg_table: dict,
    quad_rots: np.ndarray,
    weights: np.ndarray,
    K: int,
    beam_angles_deg: np.ndarray,
    lam: np.ndarray,
    sig: float,
    e0: float,
    pulse_tail_fn=raden_pulse_tail,
    edge_blur_scale: float = 1.0,
    use_gpu: bool = True,
    ctx: cl.Context | None = None,
    queue: cl.CommandQueue | None = None,
    local_size: int = 64,
    quadrature_memory_gb: float = 1.0,
) -> np.ndarray:
    """
    Build a Bragg-edge matrix by integrating over SO(3) quadrature
    points around each orientation.

    The quadrature points are grouped as

        [R_0,1, ..., R_0,A,
         R_1,1, ..., R_1,A,
         ...]

    where K is the number of center orientations and A is the number
    of quadrature points per orientation.

    The Bragg matrix is evaluated in batches to avoid allocating the
    full quadrature matrix simultaneously.

    Parameters
    ----------
    bragg_table
        Neutron reflection table.

    quad_rots
        Quadrature rotations with shape (K*A, 3, 3).

    weights
        Quadrature weights with shape (A,). These should normally be
        normalized to sum to one if the desired result is an average
        over each SO(3) ball.

    K
        Number of center orientations.

    beam_angles_deg
        Tomographic angles in degrees.

    lam
        Wavelength grid in Å.

    sig
        Orientation spread used by the Bragg-edge profile itself.

    e0
        Instrument resolution parameter.

    quadrature_memory_gb
        Approximate maximum size of the returned B batch in GB.

    Returns
    -------
    B_integrated
        Array with shape (N_Omega, K, N_lam).
    """

    quad_rots = np.asarray(
        quad_rots,
        dtype=np.float32,
    )

    weights = np.asarray(
        weights,
        dtype=np.float32,
    )

    beam_angles_deg = np.asarray(
        beam_angles_deg,
        dtype=np.float64,
    )

    lam = np.asarray(
        lam,
        dtype=np.float64,
    )

    if quad_rots.ndim != 3 or quad_rots.shape[1:] != (3, 3):
        raise ValueError(
            "quad_rots must have shape (K*A, 3, 3)."
        )

    if weights.ndim != 1:
        raise ValueError(
            "weights must have shape (A,)."
        )

    if quad_rots.shape[0] % K != 0:
        raise ValueError(
            "quad_rots.shape[0] must be divisible by K."
        )

    N_quad = quad_rots.shape[0] // K

    if len(weights) != N_quad:
        raise ValueError(
            f"Expected {N_quad} quadrature weights, "
            f"got {len(weights)}."
        )

    if quadrature_memory_gb <= 0:
        raise ValueError(
            "quadrature_memory_gb must be positive."
        )

    N_Omega = len(beam_angles_deg)
    N_lam = len(lam)

    # --------------------------------------------------------------
    # Determine how many center orientations fit into one batch.
    #
    # B_batch has shape:
    #
    #     (N_Omega, K_batch * N_quad, N_lam)
    #
    # and is float32.
    # --------------------------------------------------------------

    bytes_per_orientation = (
        N_Omega
        * N_quad
        * N_lam
        * np.dtype(np.float32).itemsize
    )

    max_bytes = quadrature_memory_gb * 1024**3

    batch_K = max(
        1,
        int(max_bytes // bytes_per_orientation),
    )

    batch_K = min(
        batch_K,
        K,
    )

    print(
        f"Quadrature: K={K}, N_quad={N_quad}, "
        f"batch_K={batch_K}"
    )

    batch_bytes = (
        N_Omega
        * batch_K
        * N_quad
        * N_lam
        * 4
    )

    print(
        f"Maximum B batch size: "
        f"{batch_bytes / 1024**3:.3f} GB"
    )

    # --------------------------------------------------------------
    # Output.
    # --------------------------------------------------------------

    B_integrated = np.zeros(
        (N_Omega, K, N_lam),
        dtype=np.float32,
    )

    # --------------------------------------------------------------
    # Process batches of CENTER orientations.
    #
    # This is important: we keep all N_quad points belonging to a
    # center orientation in the same batch.
    # --------------------------------------------------------------

    for k_start in range(0, K, batch_K):

        k_end = min(
            k_start + batch_K,
            K,
        )

        K_batch = k_end - k_start

        quad_start = k_start * N_quad
        quad_end = k_end * N_quad

        quad_batch = quad_rots[
            quad_start:quad_end
        ]

        # ----------------------------------------------------------
        # Calculate B for all quadrature points in this batch.
        # ----------------------------------------------------------

        if use_gpu:

            B_quad = build_bragg_matrix_gpu(
                bragg_table=bragg_table,
                rotations=Rotation.from_matrix(
                    quad_batch
                ),
                beam_angles_deg=beam_angles_deg,
                lam=lam,
                sig=sig,
                e0=e0,
                pulse_tail_fn=pulse_tail_fn,
                edge_blur_scale=edge_blur_scale,
                ctx=ctx,
                queue=queue,
                local_size=local_size,
            )

        else:

            B_quad = build_bragg_matrix_cpu(
                bragg_table=bragg_table,
                rotations=Rotation.from_matrix(
                    quad_batch
                ),
                beam_angles_deg=beam_angles_deg,
                lam=lam,
                sig=sig,
                e0=e0,
                pulse_tail_fn=pulse_tail_fn,
                edge_blur_scale=edge_blur_scale,
            ).astype(np.float32)

        # ----------------------------------------------------------
        # Reshape:
        #
        #     (N_Omega, K_batch*N_quad, N_lam)
        #
        # ->
        #
        #     (N_Omega, K_batch, N_quad, N_lam)
        # ----------------------------------------------------------

        B_quad = B_quad.reshape(
            N_Omega,
            K_batch,
            N_quad,
            N_lam,
        )

        # ----------------------------------------------------------
        # Integrate over the quadrature dimension.
        #
        # weights:
        #     (N_quad,)
        #
        # becomes:
        #     (1, 1, N_quad, 1)
        # ----------------------------------------------------------

        B_batch = np.sum(
            B_quad
            * weights[None, None, :, None],
            axis=2,
        )

        B_integrated[
            :,
            k_start:k_end,
            :,
        ] = B_batch

        print(
            f"Quadrature batch "
            f"{k_start}:{k_end} / {K}"
        )

        del B_quad
        del B_batch

    return B_integrated


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
    edge_blur_scale : float, optional
        Scalar multiplier applied to the symmetric edge broadening term.
        Use values below 1.0 for sharper edges and above 1.0 for broader
        edges.  Must be positive.
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

    **Units**: the B matrix stores the macroscopic elastic-coherent attenuation
    coefficient μ_R(λ, τ) in cm⁻¹.  The number density n = 1/V_cell is implicit
    in the formula (V² in the denominator) but is never hardcoded: V comes from
    :meth:`Material.neutron_bragg_table` which reads it from the lattice
    parameters set on the :class:`Material` object.  To recover the microscopic
    cross-section σ = μ/n, divide by ``n = 1 / (bragg_table['V'] * 1e-24)``.
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
        edge_blur_scale: float = 1.0,
        powder_xs: np.ndarray | None = None,
        include_powder: bool = False,
        h_max: int = 10,
        threshold: float = 1e-3,
        use_gpu: bool = True,
        ctx: cl.Context | None = None,
        queue: cl.CommandQueue | None = None,
        N_quad: int | None = None,
        sigma_distance: float | None = None,
        quadrature_memory_gb: float = 1.0,
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
        #
        # If N_quad is None, retain the original behavior exactly:
        #
        #     one orientation -> one column of B
        #
        # If N_quad is provided, evaluate B at N_quad orientations
        # around every grid orientation and integrate over the local
        # SO(3) ball.

        if N_quad is None:

            # --------------------------------------------------------------
            # Original implementation.
            # --------------------------------------------------------------

            if use_gpu:

                B = build_bragg_matrix_gpu(
                    bragg_table=bragg_table,
                    rotations=rotations,
                    beam_angles_deg=beam_angles,
                    lam=lam,
                    sig=sigma_grid,
                    e0=e0,
                    pulse_tail_fn=pulse_tail_fn,
                    edge_blur_scale=edge_blur_scale,
                    ctx=ctx,
                    queue=queue,
                )

            else:

                B = build_bragg_matrix_cpu(
                    bragg_table=bragg_table,
                    rotations=rotations,
                    beam_angles_deg=beam_angles,
                    lam=lam,
                    sig=sigma_grid,
                    e0=e0,
                    pulse_tail_fn=pulse_tail_fn,
                    edge_blur_scale=edge_blur_scale,
                ).astype(np.float32)

        else:

            # --------------------------------------------------------------
            # Quadrature implementation.
            # --------------------------------------------------------------

            if N_quad <= 0:
                raise ValueError(
                    "N_quad must be positive."
                )

            if sigma_distance is None:
                raise ValueError(
                    "sigma_distance must be provided when "
                    "N_quad is specified."
                )

            if sigma_distance <= 0:
                raise ValueError(
                    "sigma_distance must be positive."
                )

            # --------------------------------------------------------------
            # Generate quadrature orientations around every grid point.
            #
            # Normalized weights are used so that the result is an average
            # over each local SO(3) ball and therefore remains on the same
            # scale as the original B matrix.
            # --------------------------------------------------------------

            grid_mats = rotations.as_matrix()

            quad_rots, weights = sample_so3_ball_quadrature(
                rotations=grid_mats,
                radius=sigma_distance,
                n_samples=N_quad,
                normalize_weights=True,
            )

            B = build_bragg_matrix_quadrature(
                bragg_table=bragg_table,
                quad_rots=quad_rots,
                weights=weights,
                K=N_orient,
                beam_angles_deg=beam_angles,
                lam=lam,
                sig=1.2*sigma_distance/N_quad**(1/3),
                e0=e0,
                pulse_tail_fn=pulse_tail_fn,
                edge_blur_scale=edge_blur_scale,
                use_gpu=use_gpu,
                ctx=ctx,
                queue=queue,
                quadrature_memory_gb=quadrature_memory_gb,
            )

            del quad_rots
            del weights

        # --- store build metadata ---
        self.material = material
        self.grid = grid
        self.rotations = rotations
        self.beam_angles = beam_angles
        self.lam = lam
        self.sigma_grid = sigma_grid
        self.e0 = e0
        self.pulse_tail_fn = pulse_tail_fn
        self.edge_blur_scale = edge_blur_scale
        self.include_powder = include_powder
        self.angles = beam_angles

        # --- delegate to parent ---
        print('N_Omega',N_Omega)
        super().__init__(
            B=B,
            angles = self.angles,
            N_Omega=N_Omega,
            K=N_orient,
            N_seg=N_lam,
            ctx=ctx,
            queue=queue,
            **parent_kwargs,
        )






def sample_so3_ball_quadrature(
    rotations: np.ndarray,
    radius: float,
    n_samples: int,
    *,
    seed: int | None = 0,
    normalize_weights: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate quadrature points inside an SO(3) ball around each
    supplied orientation.

    Parameters
    ----------
    rotations
        Center orientations with shape (K, 3, 3).

    radius
        SO(3) ball radius in radians. Must be smaller than pi.

    n_samples
        Number of quadrature points per orientation.

    seed
        Sobol scrambling seed.

    normalize_weights
        If False, weights approximate integration over the SO(3)
        ball. If True, weights sum to one and therefore give a
        local average.

    Returns
    -------
    quadrature_rotations
        Shape (K*A, 3, 3), ordered as

            [R_0,1, ..., R_0,A,
             R_1,1, ..., R_1,A,
             ...]

    weights
        Shape (A,). Same weights are used for every orientation.
    """

    rotations = np.asarray(
        rotations,
        dtype=np.float64,
    )

    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(
            "rotations must have shape (K, 3, 3)."
        )

    if radius <= 0:
        raise ValueError(
            "radius must be positive."
        )

    if radius >= np.pi:
        raise ValueError(
            "radius must be smaller than pi."
        )

    if n_samples <= 0:
        raise ValueError(
            "n_samples must be positive."
        )

    K = rotations.shape[0]
    A = n_samples

    # --------------------------------------------------------------
    # Generate Sobol points in [0, 1)^3.
    # --------------------------------------------------------------

    sampler = qmc.Sobol(
        d=3,
        scramble=True,
        seed=seed,
    )

    m = int(np.ceil(np.log2(A)))

    u = sampler.random_base2(m=m)[:A]

    # --------------------------------------------------------------
    # Uniform points in a 3D ball.
    #
    # rho = R * u^(1/3)
    # --------------------------------------------------------------

    rho = radius * u[:, 0] ** (1.0 / 3.0)

    cos_theta = 1.0 - 2.0 * u[:, 1]

    sin_theta = np.sqrt(
        np.maximum(
            0.0,
            1.0 - cos_theta**2,
        )
    )

    phi = 2.0 * np.pi * u[:, 2]

    omega = np.empty(
        (A, 3),
        dtype=np.float64,
    )

    omega[:, 0] = (
        rho
        * sin_theta
        * np.cos(phi)
    )

    omega[:, 1] = (
        rho
        * sin_theta
        * np.sin(phi)
    )

    omega[:, 2] = (
        rho
        * cos_theta
    )

    # --------------------------------------------------------------
    # SO(3) Haar Jacobian.
    #
    # J(theta) =
    #     (sin(theta/2) / (theta/2))^2
    # --------------------------------------------------------------

    theta = np.linalg.norm(
        omega,
        axis=1,
    )

    half_theta = theta / 2.0

    jacobian = np.ones(
        A,
        dtype=np.float64,
    )

    mask = half_theta > 1e-12

    jacobian[mask] = (
        np.sin(half_theta[mask])
        / half_theta[mask]
    ) ** 2

    # Uniform Euclidean-ball sampling contributes
    #
    #     4*pi*r^3 / (3*A)
    #
    # and the Haar correction contributes J(theta).

    euclidean_volume = (
        4.0
        * np.pi
        * radius**3
        / 3.0
    )

    weights = (
        euclidean_volume
        / A
        * jacobian
    )

    if normalize_weights:
        weights /= weights.sum()

    # --------------------------------------------------------------
    # Exponential map:
    #
    #     omega -> exp([omega]_x)
    # --------------------------------------------------------------

    local_rotations = Rotation.from_rotvec(
        omega
    ).as_matrix()

    # --------------------------------------------------------------
    # Apply every local rotation to every center.
    #
    # R_{k,a} = R_k @ R_a
    #
    # First obtain shape (K, A, 3, 3).
    # --------------------------------------------------------------

    quadrature_rotations = np.einsum(
        "kij,ajl->kail",
        rotations,
        local_rotations,
    )

    # Flatten K and A.
    quadrature_rotations = quadrature_rotations.reshape(
        K * A,
        3,
        3,
    )

    return (
        np.ascontiguousarray(
            quadrature_rotations,
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            weights,
            dtype=np.float32,
        ),
    )