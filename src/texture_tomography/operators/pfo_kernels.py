from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pyopencl as cl

from .create_pfo_matrix import build_pf_program

# ----------------------------
# Program build
# ----------------------------
def build_pfo_program(ctx: cl.Context, *, ts: int = 16) -> cl.Program:
    """
    Build the main PFO OpenCL program from pfo_kernels.cl.
    """
    cl_path = Path(__file__).with_name("pfo_kernels.cl")
    src = cl_path.read_text()
    return cl.Program(ctx, src).build(options=[f"-DTS={ts}"])


# ----------------------------
# Typed kernel bundle
# ----------------------------
@dataclass(frozen=True)
class Kernels:
    # ---- core math ----
    batched_gemm_kernel: cl.Kernel
    expand_kernel: cl.Kernel
    accumulate_kernel: cl.Kernel
    gather_kernel: cl.Kernel

    # ---- transposes ----
    transpose_kernel: cl.Kernel
    btranspose_kernel: cl.Kernel
    transpose_r_mx_k_to_k_r_mx_kernel: cl.Kernel
    transpose_d_omega_k_f_to_c: cl.Kernel
    transpose_omega_d_k_c_to_d_omega_k_f: cl.Kernel

    # ---- slicing / scattering ----
    gather_coeffs_kernel: cl.Kernel
    slice_k_lastaxis_f: cl.Kernel
    scatter_k_lastaxis_f: cl.Kernel
    scatter_k_batch_c: cl.Kernel

    # ---- PF batching helpers ----
    SLICE_COEFFS_K_BATCH: cl.Kernel
    SLICE_COEFFS_K_BATCH_F: cl.Kernel
    SLICE_GRIDINV_K_BATCH: cl.Kernel
    SCALE_PF_BY_INTENSITY_INPLACE: cl.Kernel


# One authoritative mapping from "public attribute" -> "program symbol"
_KERNEL_MAP: Final[tuple[tuple[str, str], ...]] = (
    # ---- core math ----
    ("batched_gemm_kernel", "batched_gemm_rmn"),
    ("expand_kernel", "expand_gaussian_peaks"),
    ("accumulate_kernel", "accumulate_segments"),
    ("gather_kernel", "gather_last_axis"),

    # ---- transposes ----
    ("transpose_kernel", "transpose_k_nrot_mx"),
    ("btranspose_kernel", "transpose_B_r_k_nsub_to_r_nsub_k"),
    ("transpose_r_mx_k_to_k_r_mx_kernel", "transpose_r_mx_k_to_k_r_mx"),
    ("transpose_d_omega_k_f_to_c", "transpose_d_omega_k_f_to_c"),
    ("transpose_omega_d_k_c_to_d_omega_k_f", "transpose_omega_d_k_c_to_d_omega_k_f"),

    # ---- slicing / scattering ----
    ("gather_coeffs_kernel", "gather_coeffs_k_slice"),
    ("slice_k_lastaxis_f", "slice_k_lastaxis_f"),
    ("scatter_k_lastaxis_f", "scatter_k_lastaxis_f"),
    ("scatter_k_batch_c", "scatter_k_batch_c"),

    # ---- PF batching helpers ----
    ("SLICE_COEFFS_K_BATCH", "SLICE_COEFFS_K_BATCH"),
    ("SLICE_COEFFS_K_BATCH_F", "SLICE_COEFFS_K_BATCH_F"),
    ("SLICE_GRIDINV_K_BATCH", "SLICE_GRIDINV_K_BATCH"),
    ("SCALE_PF_BY_INTENSITY_INPLACE", "SCALE_PF_BY_INTENSITY_INPLACE"),
)


def bind_pfo_kernels(prg: cl.Program) -> Kernels:
    """
    Bind kernels into a typed bundle.
    Fails fast if a kernel symbol is missing from the compiled program.
    """
    kwargs: dict[str, cl.Kernel] = {}
    missing: list[str] = []

    for field_name, symbol_name in _KERNEL_MAP:
        try:
            kwargs[field_name] = getattr(prg, symbol_name)
        except AttributeError:
            missing.append(symbol_name)

    if missing:
        # This is *way* easier to debug than silent None / late failures.
        raise AttributeError(
            "Missing kernel symbols in compiled OpenCL program: "
            + ", ".join(missing)
        )

    return Kernels(**kwargs)  # type: ignore[arg-type]


def build_all_opencl(ctx: cl.Context, *, ts: int = 16) -> tuple[cl.Program, Kernels, cl.Program]:
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
