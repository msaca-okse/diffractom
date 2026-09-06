import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

from dataclasses import dataclass

from .interpolation_kernels import Kernels, build_all_opencl


def _round_up(n: int, block: int) -> int:
    return ((n + block - 1) // block) * block


def project_rotations_to_s2(
    rotations: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """
    Project rotations onto S² using a fixed sample-space direction.

    Parameters
    ----------
    rotations
        Rotation matrices with shape (K, 3, 3), mapping
        crystal coordinates -> sample coordinates.

    direction
        Sample-space direction with shape (3,).

    Returns
    -------
    poles
        Unit vectors on S² with shape (K, 3).

        For each rotation R,

            pole = R.T @ direction
    """

    rotations = np.asarray(
        rotations,
        dtype=np.float64,
    )

    direction = np.asarray(
        direction,
        dtype=np.float64,
    )

    if (
        rotations.ndim != 3
        or rotations.shape[1:] != (3, 3)
    ):
        raise ValueError(
            "rotations must have shape (K, 3, 3)."
        )

    if direction.shape != (3,):
        raise ValueError(
            "direction must have shape (3,)."
        )

    norm = np.linalg.norm(direction)

    if norm == 0:
        raise ValueError(
            "direction must be non-zero."
        )

    direction = direction / norm

    poles = np.einsum(
        "nji,j->ni",
        rotations,
        direction,
    )

    poles /= np.linalg.norm(
        poles,
        axis=1,
        keepdims=True,
    )

    return np.ascontiguousarray(
        poles,
        dtype=np.float32,
    )




@dataclass
class SparseInterpolationKernelS2:
    """
    GPU-resident sparse interpolation kernel on S².

    The interpolation matrix is represented in CSR format:

        f_interp[m] = sum_k G[m, k] * f[k]

    where G is stored by

        row_ptr
        col_indices
        weights

    and the S² distance is antipodal:

        d(p, q) = arccos(|p dot q|)
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


def build_interpolation_kernelS2(
    queue: cl.CommandQueue,
    grid_poles: np.ndarray,
    eval_poles: np.ndarray,
    sigma_interp: float,
    cutoff_sigma: float = 3.0,
    ts: int = 256,
    verbose: int = 0,
    pair_chunk_size: int = 5_000_000,
) -> SparseInterpolationKernelS2:
    """
    Build a GPU-resident sparse interpolation kernel on S².

    Parameters
    ----------
    queue
        OpenCL command queue.

    grid_poles
        Source poles on S², shape (K, 3).

    eval_poles
        Evaluation poles on S², shape (M, 3).

    sigma_interp
        Gaussian interpolation width in radians.

    cutoff_sigma
        Contributions farther than

            cutoff_sigma * sigma_interp

        are ignored.

    ts
        OpenCL work-group size.

    verbose
        0 = silent.
        1 = print allocation information and coarse progress.

    pair_chunk_size
        Number of (evaluation pole, source pole) pairs
        processed by each OpenCL launch.

    Returns
    -------
    SparseInterpolationKernelS2
        GPU-resident CSR interpolation kernel.
    """

    if sigma_interp <= 0:
        raise ValueError(
            "sigma_interp must be positive."
        )

    if cutoff_sigma <= 0:
        raise ValueError(
            "cutoff_sigma must be positive."
        )

    if ts <= 0:
        raise ValueError(
            "ts must be positive."
        )

    if pair_chunk_size <= 0:
        raise ValueError(
            "pair_chunk_size must be positive."
        )

    grid_poles = np.asarray(
        grid_poles,
        dtype=np.float32,
    )

    eval_poles = np.asarray(
        eval_poles,
        dtype=np.float32,
    )

    if (
        grid_poles.ndim != 2
        or grid_poles.shape[1] != 3
    ):
        raise ValueError(
            "grid_poles must have shape (K, 3)."
        )

    if (
        eval_poles.ndim != 2
        or eval_poles.shape[1] != 3
    ):
        raise ValueError(
            "eval_poles must have shape (M, 3)."
        )

    K = grid_poles.shape[0]
    M = eval_poles.shape[0]

    if K == 0:
        raise ValueError(
            "grid_poles must contain at least one pole."
        )

    if M == 0:
        raise ValueError(
            "eval_poles must contain at least one pole."
        )

    grid_norms = np.linalg.norm(
        grid_poles,
        axis=1,
        keepdims=True,
    )

    eval_norms = np.linalg.norm(
        eval_poles,
        axis=1,
        keepdims=True,
    )

    if np.any(grid_norms == 0):
        raise ValueError(
            "grid_poles contains a zero vector."
        )

    if np.any(eval_norms == 0):
        raise ValueError(
            "eval_poles contains a zero vector."
        )

    grid_poles = np.ascontiguousarray(
        grid_poles / grid_norms,
        dtype=np.float32,
    )

    eval_poles = np.ascontiguousarray(
        eval_poles / eval_norms,
        dtype=np.float32,
    )

    int32_max = np.iinfo(np.int32).max

    if K > int32_max:
        raise ValueError(
            "Number of source poles K exceeds INT32_MAX."
        )

    if M > int32_max:
        raise ValueError(
            "Number of evaluation poles M exceeds INT32_MAX."
        )

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
        print("Building S² interpolation kernel")
        print("=" * 70)

        print(f"K               = {K}")
        print(f"M               = {M}")
        print(f"sigma_interp    = {sigma_interp}")
        print(f"cutoff_sigma    = {cutoff_sigma}")
        print(f"ts              = {ts}")
        print(f"pair_chunk_size = {pair_chunk_size}")

    _, kernels = build_all_opencl(
        queue.context,
        ts=ts,
    )

    grid_poles_gpu = clarray.to_device(
        queue,
        grid_poles,
    )

    eval_poles_gpu = clarray.to_device(
        queue,
        eval_poles,
    )

    print_allocations(
        "Geometry:",
        [
            ("grid_poles_gpu", grid_poles_gpu),
            ("eval_poles_gpu", eval_poles_gpu),
        ],
    )

    row_counts = clarray.zeros(
        queue,
        M,
        dtype=np.int32,
    )

    total_pairs = M * K

    n_pair_chunks = (
        total_pairs + pair_chunk_size - 1
    ) // pair_chunk_size

    progress_step = max(
        1,
        n_pair_chunks // 10,
    )

    if verbose >= 1:
        print(
            f"\nCounting interpolation neighbours "
            f"({total_pairs:,} pairs)..."
        )

    for chunk_idx, start in enumerate(
        range(
            0,
            total_pairs,
            pair_chunk_size,
        ),
        start=1,
    ):
        end = min(
            start + pair_chunk_size,
            total_pairs,
        )

        n = end - start

        kernels.count_interpolation_nnz_s2(
            queue,
            (_round_up(n, ts),),
            (ts,),
            grid_poles_gpu.data,
            eval_poles_gpu.data,
            row_counts.data,
            np.int32(M),
            np.int32(K),
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
                f"  Count progress: "
                f"{100 * chunk_idx / n_pair_chunks:5.1f}%"
            )

    scan_a = clarray.empty_like(
        row_counts
    )

    scan_b = clarray.empty_like(
        row_counts
    )

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
        row_ptr[M:M + 1].get(
            queue=queue
        )[0]
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
        "CSR allocation:",
        [
            ("row_ptr", row_ptr),
            ("col_indices", col_indices),
            ("weights", weights),
        ],
    )

    if verbose >= 1:
        print(
            f"\nFilling CSR interpolation matrix "
            f"({total_pairs:,} pairs)..."
        )

    for chunk_idx, start in enumerate(
        range(
            0,
            total_pairs,
            pair_chunk_size,
        ),
        start=1,
    ):
        end = min(
            start + pair_chunk_size,
            total_pairs,
        )

        n = end - start

        kernels.fill_interpolation_csr_s2(
            queue,
            (_round_up(n, ts),),
            (ts,),
            grid_poles_gpu.data,
            eval_poles_gpu.data,
            row_ptr.data,
            write_offsets.data,
            col_indices.data,
            weights.data,
            np.int32(M),
            np.int32(K),
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
                f"  Fill progress:  "
                f"{100 * chunk_idx / n_pair_chunks:5.1f}%"
            )

    queue.finish()

    del grid_poles_gpu
    del eval_poles_gpu
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
        print("S² interpolation kernel ready")
        print("=" * 70)

    return SparseInterpolationKernelS2(
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


def interpolateS2(
    values_interp: clarray.Array,
    values: clarray.Array,
    interpolation_kernel: SparseInterpolationKernelS2,
    queue: cl.CommandQueue,
) -> None:
    """
    Apply a GPU-resident sparse S² interpolation kernel.

    Parameters
    ----------
    values_interp
        Preallocated GPU output array with shape

            (*values.shape[:-1], M)

        where M is the number of evaluation poles.

    values
        GPU input array with shape

            (..., K)

        where K is the number of source poles.

        The array must be C-contiguous.

    interpolation_kernel
        Sparse S² interpolation kernel.

    queue
        OpenCL command queue.
    """

    if values.dtype != np.float32:
        raise TypeError(
            "values must have dtype float32."
        )

    if values_interp.dtype != np.float32:
        raise TypeError(
            "values_interp must have dtype float32."
        )

    if values.ndim < 1:
        raise ValueError(
            "values must have at least one dimension."
        )

    if values.shape[-1] != interpolation_kernel.K:
        raise ValueError(
            f"Expected values.shape[-1] == "
            f"{interpolation_kernel.K}, "
            f"got {values.shape[-1]}."
        )

    expected_shape = (
        *values.shape[:-1],
        interpolation_kernel.M,
    )

    if values_interp.shape != expected_shape:
        raise ValueError(
            f"values_interp must have shape "
            f"{expected_shape}, "
            f"got {values_interp.shape}."
        )

    # The OpenCL kernel treats the leading dimensions as one
    # flattened dimension:
    #
    #     values.shape = (..., K)
    #                    -> (N, K)
    #
    # where N = product(values.shape[:-1]).
    #
    # Therefore the underlying data must be C-contiguous.
    if not values.flags.c_contiguous:
        raise ValueError(
            "values must be C-contiguous."
        )

    if not values_interp.flags.c_contiguous:
        raise ValueError(
            "values_interp must be C-contiguous."
        )

    N = int(
        np.prod(
            values.shape[:-1],
            dtype=np.int64,
        )
    )

    K = interpolation_kernel.K
    M = interpolation_kernel.M

    total = N * M

    if total > np.iinfo(np.int32).max:
        raise ValueError(
            "N*M exceeds INT32_MAX. "
            "smooth_reinterpolate currently uses "
            "32-bit indexing."
        )

    interpolation_kernel.kernels.smooth_reinterpolate(
        queue,
        (
            _round_up(
                total,
                interpolation_kernel.ts,
            ),
        ),
        (
            interpolation_kernel.ts,
        ),
        values.data,
        interpolation_kernel.row_ptr.data,
        interpolation_kernel.col_indices.data,
        interpolation_kernel.weights.data,
        values_interp.data,
        np.int32(N),
        np.int32(K),
        np.int32(M),
    )