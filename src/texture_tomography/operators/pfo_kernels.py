# pfo_kernels.py

import os
import pyopencl as cl
from package.texture_tomography.operators.create_pfo_matrix import build_pf_program


def build_pfo_program(ctx: cl.Context, *, ts: int = 16) -> cl.Program:
    """
    Build the main PFO OpenCL program from pfo_kernels.cl.

    This replaces the old giant string concatenation.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cl_path = os.path.join(here, "pfo_kernels.cl")

    with open(cl_path, "r") as f:
        src = f.read()

    return cl.Program(ctx, src).build(options=[f"-DTS={ts}"])


def bind_pfo_kernels(prg: cl.Program):
    """
    Bind all kernels exactly as expected by the operator class.
    Returns a simple namespace-like object.
    """

    class Kernels:
        pass

    k = Kernels()

    # ---- core math ----
    k.batched_gemm_kernel = prg.batched_gemm_rmn
    k.expand_kernel = prg.expand_gaussian_peaks
    k.accumulate_kernel = prg.accumulate_segments
    k.gather_kernel = prg.gather_last_axis

    # ---- transposes ----
    k.transpose_kernel = prg.transpose_k_nrot_mx
    k.btranspose_kernel = prg.transpose_B_r_k_nsub_to_r_nsub_k
    k.transpose_r_mx_k_to_k_r_mx_kernel = prg.transpose_r_mx_k_to_k_r_mx
    k.transpose_d_omega_k_f_to_c = prg.transpose_d_omega_k_f_to_c
    k.transpose_omega_d_k_c_to_d_omega_k_f = prg.transpose_omega_d_k_c_to_d_omega_k_f

    # ---- slicing / scattering ----
    k.gather_coeffs_kernel = prg.gather_coeffs_k_slice
    k.slice_k_lastaxis_f = prg.slice_k_lastaxis_f
    k.scatter_k_lastaxis_f = prg.scatter_k_lastaxis_f
    k.scatter_k_batch_c = prg.scatter_k_batch_c

    # ---- PF batching helpers ----
    k.SLICE_COEFFS_K_BATCH = prg.SLICE_COEFFS_K_BATCH
    k.SLICE_GRIDINV_K_BATCH = prg.SLICE_GRIDINV_K_BATCH
    k.SCALE_PF_BY_INTENSITY_INPLACE = prg.SCALE_PF_BY_INTENSITY_INPLACE

    return k


def build_all_opencl(ctx: cl.Context, *, ts: int = 16):
    """
    Convenience helper:
      - builds main PFO kernels
      - builds PF-matrix kernels
      - returns both
    """
    prg = build_pfo_program(ctx, ts=ts)
    kernels = bind_pfo_kernels(prg)
    pf_prg = build_pf_program(ctx)
    return prg, kernels, pf_prg
