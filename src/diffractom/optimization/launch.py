"""Launch sizes for the solvers' OpenCL kernels, which index with 64-bit (size_t) integers.

Element-wise kernels loop over their elements in grid-stride fashion,

    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) ...

so arrays of any size, including beyond 2^31 elements, are covered by a bounded number of work items.
Kernels that need the pixel of a (K, Ny, Nx) C-order array run on a (pixel, orientation) range
with the orientations strided, which avoids 64-bit division.
"""
import numpy as np

MAX_WORK_ITEMS = 1 << 24    # work items of an element-wise launch; enough to fill any current GPU
MAX_K_WORK_ITEMS = 64       # work items along the orientation axis of a (pixel, orientation) launch


def elementwise(n):
    """Global size and the element count (as a kernel argument) of a grid-stride loop over n elements."""
    n = int(n)
    return (max(min(n, MAX_WORK_ITEMS), 1),), np.uint64(n)


REDUCE_LOCAL = 256


def reduction(n):
    """Global and local size, element count and number of partial sums of a grid-stride kernel that
    also reduces (local size REDUCE_LOCAL, one partial sum per work-group)."""
    n = int(n)
    g = max(min(-(-n // REDUCE_LOCAL) * REDUCE_LOCAL, MAX_WORK_ITEMS), REDUCE_LOCAL)
    return (g,), (REDUCE_LOCAL,), np.uint64(n), g // REDUCE_LOCAL


def pixel_orientation(npix, K):
    """Global size of a kernel over a (K, Ny, Nx) C-order array: pixels on axis 0,
    orientations strided over axis 1."""
    return (max(int(npix), 1), max(min(int(K), MAX_K_WORK_ITEMS), 1))
