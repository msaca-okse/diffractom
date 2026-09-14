from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
from scipy.spatial.transform import Rotation

from .interpolation_kernels import (
    Kernels,
    build_all_opencl,
)

def _round_up(n: int, block: int) -> int:
    return ((n + block - 1) // block) * block

@dataclass
class SparseInterpolationKernelSO3:
    """
    GPU-resident sparse interpolation kernel.

    The interpolation matrix is represented in CSR format:

        f_interp[m] = sum_k G[m, k] * f[k]

    where G is stored by (row_ptr, col_indices, weights).
    """

    row_ptr: clarray.Array
    col_indices: clarray.Array
    weights: clarray.Array

    K: int
    M: int
    nnz: int

    sigma_interp: float
    cutoff_sigma: float

    kernels: Kernels
    context: cl.Context
    ts: int


def _matrices_to_quaternions(
    matrices: np.ndarray,
) -> np.ndarray:
    """Convert rotation matrices to float32 quaternions [x, y, z, w]."""

    matrices = np.asarray(matrices, dtype=np.float64)

    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(
            "Rotation matrices must have shape (N, 3, 3)."
        )

    quaternions = Rotation.from_matrix(matrices).as_quat()

    return np.ascontiguousarray(
        quaternions,
        dtype=np.float32,
    )
def build_interpolation_kernelSO3(
    queue: cl.CommandQueue,
    grid_mats: np.ndarray,
    eval_mats: np.ndarray,
    symmetry_ops: np.ndarray,
    sigma_interp: float,
    cutoff_sigma: float = 3.0,
    ts: int = 256,
    verbose: int = 0,
    chunk_size: int = 250_000,
    pair_chunk_size: int = 5_000_000,
) -> SparseInterpolationKernelSO3:

    if sigma_interp <= 0:
        raise ValueError("sigma_interp must be positive.")

    if cutoff_sigma <= 0:
        raise ValueError("cutoff_sigma must be positive.")

    if ts <= 0:
        raise ValueError("ts must be positive.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if pair_chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive.")

    def print_allocations(name, arrays):
        if verbose < 1:
            return

        print(f"\n{name}")

        total = 0

        for array_name, array in arrays:
            if array is None:
                continue

            nbytes = array.nbytes
            total += nbytes

            print(
                f"  {array_name:25s}"
                f" shape={str(array.shape):22s}"
                f" dtype={str(array.dtype):8s}"
                f" size={nbytes / 1024**2:8.2f} MB"
            )

        print(
            f"  {'TOTAL':25s}"
            f" {total / 1024**2:8.2f} MB"
        )

    if verbose >= 1:
        print("\n" + "=" * 70)
        print("Building SO(3) interpolation kernel")
        print("=" * 70)

        print(f"K               = {grid_mats.shape[0]}")
        print(f"M               = {eval_mats.shape[0]}")
        print(f"Ns              = {symmetry_ops.shape[0]}")
        print(f"sigma_interp    = {sigma_interp}")
        print(f"cutoff_sigma    = {cutoff_sigma}")
        print(f"ts              = {ts}")
        print(f"chunk_size      = {chunk_size}")
        print(f"pair_chunk_size = {pair_chunk_size}")

        device = queue.device

        print("\nOpenCL device:")
        print(f"  name           = {device.name}")
        print(f"  vendor         = {device.vendor}")
        print(f"  driver         = {device.driver_version}")
        print(
            f"  global memory  = "
            f"{device.global_mem_size / 1024**3:.2f} GB"
        )
        print(
            f"  local memory   = "
            f"{device.local_mem_size / 1024:.1f} KB"
        )
        print(
            f"  max work-group = "
            f"{device.max_work_group_size}"
        )

    grid_quat = _matrices_to_quaternions(grid_mats)
    eval_quat = _matrices_to_quaternions(eval_mats)
    sym_quat = _matrices_to_quaternions(symmetry_ops)

    K = grid_quat.shape[0]
    M = eval_quat.shape[0]
    Ns = sym_quat.shape[0]

    if K == 0:
        raise ValueError(
            "grid_mats must contain at least one rotation."
        )

    if M == 0:
        raise ValueError(
            "eval_mats must contain at least one rotation."
        )

    if Ns == 0:
        raise ValueError(
            "symmetry_ops must contain at least one symmetry operation."
        )

    int32_max = np.iinfo(np.int32).max

    if M > int32_max:
        raise ValueError(
            "Number of evaluation orientations M exceeds INT32_MAX."
        )

    if K > int32_max:
        raise ValueError(
            "Number of reconstruction orientations K exceeds INT32_MAX."
        )

    if Ns > int32_max:
        raise ValueError(
            "Number of symmetry operations Ns exceeds INT32_MAX."
        )

    print_allocations(
        "CPU geometry:",
        [
            ("grid_quat", grid_quat),
            ("eval_quat", eval_quat),
            ("sym_quat", sym_quat),
        ],
    )

    if verbose >= 1:
        print("\nCompiling OpenCL kernels...")

    _, kernels = build_all_opencl(
        queue.context,
        ts=ts,
    )

    if verbose >= 1:
        print("OpenCL kernels compiled.")

    grid_quat_gpu = clarray.to_device(
        queue,
        grid_quat,
    )

    eval_quat_gpu = clarray.to_device(
        queue,
        eval_quat,
    )

    sym_quat_gpu = clarray.to_device(
        queue,
        sym_quat,
    )

    print_allocations(
        "After uploading geometry:",
        [
            ("grid_quat_gpu", grid_quat_gpu),
            ("eval_quat_gpu", eval_quat_gpu),
            ("sym_quat_gpu", sym_quat_gpu),
        ],
    )

    transformed_eval_gpu = clarray.empty(
        queue,
        (M * Ns, 4),
        dtype=np.float32,
    )

    total_sym = M * Ns

    if verbose >= 1:
        print(
            f"\nTransforming symmetry equivalents "
            f"({total_sym:,} quaternions)..."
        )

    n_chunks = (
        total_sym + chunk_size - 1
    ) // chunk_size

    progress_step = max(1, n_chunks // 10)

    for chunk_idx, start in enumerate(
        range(0, total_sym, chunk_size),
        start=1,
    ):
        end = min(
            start + chunk_size,
            total_sym,
        )

        n = end - start

        kernels.transform_eval_symmetries(
            queue,
            (_round_up(n, ts),),
            (ts,),
            eval_quat_gpu.data,
            sym_quat_gpu.data,
            transformed_eval_gpu.data,
            np.int32(M),
            np.int32(Ns),
            np.int32(start),
            np.int32(n),
        )

        if (
            verbose >= 1
            and (
                chunk_idx % progress_step == 0
                or chunk_idx == n_chunks
            )
        ):
            print(
                f"  Transform progress: "
                f"{100 * chunk_idx / n_chunks:5.1f}%"
            )

    row_counts = clarray.zeros(
        queue,
        M,
        dtype=np.int32,
    )

    total_pairs = M * K

    if verbose >= 1:
        print(
            f"\nCounting interpolation neighbours "
            f"({total_pairs:,} grid/evaluation pairs)..."
        )

    n_pair_chunks = (
        total_pairs + pair_chunk_size - 1
    ) // pair_chunk_size

    progress_step = max(1, n_pair_chunks // 10)

    for chunk_idx, start in enumerate(
        range(0, total_pairs, pair_chunk_size),
        start=1,
    ):
        end = min(
            start + pair_chunk_size,
            total_pairs,
        )

        n = end - start

        kernels.count_interpolation_nnz(
            queue,
            (_round_up(n, ts),),
            (ts,),
            grid_quat_gpu.data,
            transformed_eval_gpu.data,
            row_counts.data,
            np.int32(M),
            np.int32(K),
            np.int32(Ns),
            np.float32(sigma_interp),
            np.float32(cutoff_sigma),
            np.uint64(start),
            np.int32(n),
        )

        if (
            verbose >= 1
            and (
                chunk_idx % progress_step == 0
                or chunk_idx == n_pair_chunks
            )
        ):
            print(
                f"  Count progress:     "
                f"{100 * chunk_idx / n_pair_chunks:5.1f}%"
            )

    queue.finish()

    scan_a = clarray.empty_like(row_counts)
    scan_b = clarray.empty_like(row_counts)

    kernels.copy_int(
        queue,
        (_round_up(M, ts),),
        (ts,),
        row_counts.data,
        scan_a.data,
        np.int32(M),
    )

    offset = 1
    src = scan_a
    dst = scan_b

    while offset < M:
        kernels.inclusive_scan_step(
            queue,
            (_round_up(M, ts),),
            (ts,),
            src.data,
            dst.data,
            np.int32(M),
            np.int32(offset),
        )

        src, dst = dst, src
        offset *= 2

    inclusive = src

    row_ptr = clarray.empty(
        queue,
        M + 1,
        dtype=np.int32,
    )

    kernels.make_row_ptr(
        queue,
        (_round_up(M + 1, ts),),
        (ts,),
        inclusive.data,
        row_ptr.data,
        np.int32(M),
    )

    nnz = int(
        row_ptr[M:M + 1].get(queue=queue)[0]
    )

    if nnz < 0:
        raise RuntimeError(
            "Computed nnz is negative. "
            "This indicates integer overflow in the CSR scan."
        )

    if verbose >= 1:
        print(
            f"\nCSR structure:"
            f"\n  nnz            = {nnz:,}"
            f"\n  avg neighbours = {nnz / M:.2f}"
        )

    col_indices = clarray.empty(
        queue,
        nnz,
        dtype=np.int32,
    )

    weights = clarray.empty(
        queue,
        nnz,
        dtype=np.float32,
    )

    write_offsets = clarray.zeros(
        queue,
        M,
        dtype=np.int32,
    )

    print_allocations(
        "After CSR allocation:",
        [
            ("grid_quat_gpu", grid_quat_gpu),
            ("eval_quat_gpu", eval_quat_gpu),
            ("sym_quat_gpu", sym_quat_gpu),
            ("transformed_eval_gpu", transformed_eval_gpu),
            ("row_counts", row_counts),
            ("scan_a", scan_a),
            ("scan_b", scan_b),
            ("row_ptr", row_ptr),
            ("col_indices", col_indices),
            ("weights", weights),
            ("write_offsets", write_offsets),
        ],
    )

    if verbose >= 1:
        print(
            f"\nFilling CSR interpolation matrix "
            f"({total_pairs:,} grid/evaluation pairs)..."
        )

    progress_step = max(1, n_pair_chunks // 10)

    for chunk_idx, start in enumerate(
        range(0, total_pairs, pair_chunk_size),
        start=1,
    ):
        end = min(
            start + pair_chunk_size,
            total_pairs,
        )

        n = end - start

        kernels.fill_interpolation_csr(
            queue,
            (_round_up(n, ts),),
            (ts,),
            grid_quat_gpu.data,
            transformed_eval_gpu.data,
            row_ptr.data,
            write_offsets.data,
            col_indices.data,
            weights.data,
            np.int32(M),
            np.int32(K),
            np.int32(Ns),
            np.float32(sigma_interp),
            np.float32(cutoff_sigma),
            np.uint64(start),
            np.int32(n),
        )

        if (
            verbose >= 1
            and (
                chunk_idx % progress_step == 0
                or chunk_idx == n_pair_chunks
            )
        ):
            print(
                f"  Fill progress:      "
                f"{100 * chunk_idx / n_pair_chunks:5.1f}%"
            )

    queue.finish()

    if verbose >= 1:
        print("\nCSR construction finished.")

    del grid_quat_gpu
    del eval_quat_gpu
    del sym_quat_gpu
    del transformed_eval_gpu
    del row_counts
    del scan_a
    del scan_b
    del inclusive
    del write_offsets

    if verbose >= 1:
        print_allocations(
            "Persistent allocations:",
            [
                ("row_ptr", row_ptr),
                ("col_indices", col_indices),
                ("weights", weights),
            ],
        )

        print("=" * 70)
        print("Interpolation kernel ready")
        print("=" * 70)

    return SparseInterpolationKernelSO3(
        row_ptr=row_ptr,
        col_indices=col_indices,
        weights=weights,
        K=K,
        M=M,
        nnz=nnz,
        sigma_interp=sigma_interp,
        cutoff_sigma=cutoff_sigma,
        kernels=kernels,
        context=queue.context,
        ts=ts,
    )

def interpolateSO3(
    coeffs_interp: clarray.Array,
    coeffs: clarray.Array,
    interpolation_kernel: SparseInterpolationKernelSO3,
    queue: cl.CommandQueue,
) -> None:
    """
    Apply a precomputed GPU interpolation kernel.

    Parameters
    ----------
    coeffs_interp
        Preallocated GPU output array.

        Its shape must be:

            coeffs.shape[:-1] + (_round_up(M, ts),)

    coeffs
        GPU coefficient array.

        Its final dimension must have size K.

        Examples:

            (K,)
            (N, K)
            (Nx, Ny, K)
            (Nz, Nx, Ny, K)

    interpolation_kernel
        SparseInterpolationKernelSO3 returned by
        build_interpolation_kernel().

    queue
        OpenCL command queue used for the interpolation.

    Returns
    -------
    None
        coeffs_interp is overwritten in-place.
    """

    if not isinstance(coeffs, clarray.Array):
        raise TypeError("coeffs must be a pyopencl.array.Array.")

    if not isinstance(coeffs_interp, clarray.Array):
        raise TypeError(
            "coeffs_interp must be a pyopencl.array.Array."
        )

    if coeffs.dtype != np.dtype(np.float32):
        raise TypeError("coeffs must have dtype float32.")

    if coeffs_interp.dtype != np.dtype(np.float32):
        raise TypeError(
            "coeffs_interp must have dtype float32."
        )

    if coeffs.ndim < 1:
        raise ValueError("coeffs must have at least one dimension.")

    if coeffs.shape[-1] != interpolation_kernel.K:
        raise ValueError(
            f"coeffs has final dimension {coeffs.shape[-1]}, "
            f"but interpolation kernel expects K="
            f"{interpolation_kernel.K}."
        )

    expected_shape = (
        *coeffs.shape[:-1],
        interpolation_kernel.M,
    )

    if coeffs_interp.shape != expected_shape:
        raise ValueError(
            f"coeffs_interp has shape {coeffs_interp.shape}, "
            f"but expected {expected_shape}."
        )

    if not coeffs.flags.c_contiguous:
        raise ValueError(
            "coeffs must be C-contiguous."
        )

    if not coeffs_interp.flags.c_contiguous:
        raise ValueError(
            "coeffs_interp must be C-contiguous."
        )

    if coeffs.context != interpolation_kernel.context:
        raise ValueError(
            "coeffs is not in the same OpenCL context as "
            "the interpolation kernel."
        )

    if coeffs_interp.context != interpolation_kernel.context:
        raise ValueError(
            "coeffs_interp is not in the same OpenCL context as "
            "the interpolation kernel."
        )

    # The arrays should normally be associated with the supplied queue.
    if coeffs.queue.int_ptr != queue.int_ptr:
        raise ValueError(
            "coeffs must be associated with the supplied queue."
        )

    if coeffs_interp.queue.int_ptr != queue.int_ptr:
        raise ValueError(
            "coeffs_interp must be associated with the supplied queue."
        )

    # ------------------------------------------------------------
    # Flatten all spatial dimensions into N.
    #
    # (K,)              -> N = 1
    # (N, K)            -> N = N
    # (Nx, Ny, K)       -> N = Nx * Ny
    # (Nz, Nx, Ny, K)   -> N = Nz * Nx * Ny
    # ------------------------------------------------------------

    N = int(np.prod(coeffs.shape[:-1], dtype=np.int64))
    M = interpolation_kernel.M
    K = interpolation_kernel.K

    total = N * M

    if total > np.iinfo(np.int32).max:
        raise ValueError(
            "N * M is too large for the current OpenCL kernel "
            "indexing (int32)."
        )

    interpolation_kernel.kernels.smooth_reinterpolate(
        queue,
        (_round_up(total, interpolation_kernel.ts),),
        (interpolation_kernel.ts,),
        coeffs.data,
        interpolation_kernel.row_ptr.data,
        interpolation_kernel.col_indices.data,
        interpolation_kernel.weights.data,
        coeffs_interp.data,
        np.int32(N),
        np.int32(K),
        np.int32(M),
    )