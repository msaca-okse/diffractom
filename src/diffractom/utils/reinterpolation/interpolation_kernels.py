from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pyopencl as cl


def build_interpolation_program(
    ctx: cl.Context,
    *,
    ts: int = 256,
) -> cl.Program:
    """Build the S²/SO(3) interpolation OpenCL program."""
    cl_path = Path(__file__).with_name("interpolation_kernels.cl")
    src = cl_path.read_text()
    return cl.Program(ctx, src).build(
        options=[f"-DTS={ts}"]
    )


@dataclass(frozen=True)
class Kernels:
    """OpenCL kernels used for S²/SO(3) interpolation."""

    transform_eval_symmetries: cl.Kernel
    count_interpolation_nnz: cl.Kernel
    count_interpolation_nnz_s2: cl.Kernel
    fill_int: cl.Kernel
    copy_int: cl.Kernel
    inclusive_scan_step: cl.Kernel
    make_row_ptr: cl.Kernel
    fill_csr_offsets: cl.Kernel
    fill_interpolation_csr: cl.Kernel
    fill_interpolation_csr_s2: cl.Kernel
    smooth_reinterpolate: cl.Kernel


_KERNEL_MAP: Final[
    tuple[tuple[str, str], ...]
] = (
    (
        "transform_eval_symmetries",
        "transform_eval_symmetries",
    ),
    (
        "count_interpolation_nnz",
        "count_interpolation_nnz",
    ),
    (
        "count_interpolation_nnz_s2",
        "count_interpolation_nnz_s2",
    ),
    (
        "fill_int",
        "fill_int",
    ),
    (
        "copy_int",
        "copy_int",
    ),
    (
        "inclusive_scan_step",
        "inclusive_scan_step",
    ),
    (
        "make_row_ptr",
        "make_row_ptr",
    ),
    (
        "fill_csr_offsets",
        "fill_csr_offsets",
    ),
    (
        "fill_interpolation_csr",
        "fill_interpolation_csr",
    ),
    (
        "fill_interpolation_csr_s2",
        "fill_interpolation_csr_s2",
    ),
    (
        "smooth_reinterpolate",
        "smooth_reinterpolate",
    ),
)


def bind_kernels(prg: cl.Program) -> Kernels:
    """Bind compiled OpenCL kernels into a typed bundle."""
    kwargs: dict[str, cl.Kernel] = {}
    missing: list[str] = []

    for field_name, symbol_name in _KERNEL_MAP:
        try:
            kwargs[field_name] = getattr(
                prg,
                symbol_name,
            )
        except AttributeError:
            missing.append(symbol_name)

    if missing:
        raise AttributeError(
            "Missing interpolation kernel symbols: "
            + ", ".join(missing)
        )

    return Kernels(**kwargs)  # type: ignore[arg-type]


def build_all_opencl(
    ctx: cl.Context,
    *,
    ts: int = 256,
) -> tuple[cl.Program, Kernels]:
    """Build and bind all S²/SO(3) interpolation kernels."""
    prg = build_interpolation_program(
        ctx,
        ts=ts,
    )

    kernels = bind_kernels(prg)

    return prg, kernels