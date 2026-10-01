"""
The FISTA update fused into the adjoint, one orientation batch at a time.

A FISTA iteration needs grad = A^T r only to form x_new = prox(y - tau * grad) and
y = x_new + beta * (x_new - x_old), and both are element-wise in the orientation index.
The operator computes A^T r batch by batch (adjoint_batches_cl), so every batch of the
gradient is consumed as soon as it is computed: the solver keeps two coefficient-sized
arrays, x (the current iterate) and y, instead of four (x, y, x_old, grad). Only proximal
operators that act element-wise qualify (non-negativity, L1 and their combination, with the
support mask); the total-variation prox couples the pixels of a whole image.
"""
import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
from pyopencl.reduction import ReductionKernel

from .launch import elementwise

# Same arithmetic, in the same order, as grad_step + prox_* + apply_support + extrapolate.
FUSED_KERNEL = r"""
__kernel void fista_fused_update(
    __global const float *g,          // gradient of this batch: (Npix, Kb), Fortran order
    __global float *x,                // current iterate (x_old on entry, x_new on exit), whole array
    __global float *y,                // extrapolated point, whole array
    __global const uchar *mask,       // (Npix,) support, used if use_mask
    const int use_mask,
    const ulong off,                  // offset of the batch in x and y: Npix * k0
    const ulong npix,
    const ulong n,                    // Npix * Kb
    const float tau,
    const float beta,
    const float lt,                   // lam * tau, the L1 threshold
    const int prox                    // 0: nonneg, 1: l1, 2: nonneg_l1
){
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const size_t j = off + i;
        float v = y[j] - tau * g[i];
        if (prox == 0) {
            if (v < 0.0f) v = 0.0f;
        } else if (prox == 1) {
            const float a = fabs(v) - lt;
            v = (a > 0.0f) ? copysign(a, v) : 0.0f;
        } else {
            const float w = v - lt;
            v = (w > 0.0f) ? w : 0.0f;
        }
        if (use_mask && !mask[i % npix]) v = 0.0f;
        const float xo = x[j];
        y[j] = v + beta * (v - xo);
        x[j] = v;
    }
}

// Per-work-group partial sums of x^2 (mode 0) or |x| (mode 1) into out[0 .. num_groups-1].
// pyopencl's reductions (clarray.vdot, ReductionKernel) block the host until the queue is done,
// which would keep a batch loop from running ahead of the GPU; this plain kernel does not.
__kernel void partial_sums(
    __global const float *x,
    const ulong n,
    const int mode,
    __global float *out,
    const ulong out_offset,
    __local float *scratch
){
    float acc = 0.0f;
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const float v = x[i];
        acc += (mode == 0) ? v * v : fabs(v);
    }
    const int lid = get_local_id(0);
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) out[out_offset + get_group_id(0)] = scratch[0];
}
"""

PROX_CODES = {"nonneg": 0, "l1": 1, "nonneg_l1": 2}
SUM_GROUPS, SUM_LOCAL = 256, 256


class PartialSums:
    """Sums of x^2 or |x| over several arrays, accumulated on the GPU without blocking the host:
    add() enqueues a partial-sum kernel, total() reads all partials back once (reset() to reuse)."""

    def __init__(self, fu, capacity):
        self.fu = fu
        self.capacity = capacity
        self.buf = clarray.empty(fu.queue, (capacity * SUM_GROUPS,), np.float32)
        self.count = 0

    def reset(self):
        self.count = 0
        return self

    def add(self, x_data, n, mode):
        """Add the sum over the first n elements of the buffer x_data."""
        assert self.count < self.capacity
        f = self.fu
        f.k_sums(f.queue, (SUM_GROUPS * SUM_LOCAL,), (SUM_LOCAL,), x_data, np.uint64(n), np.int32(mode),
                 self.buf.data, np.uint64(self.count * SUM_GROUPS), cl.LocalMemory(4 * SUM_LOCAL))
        self.count += 1

    def total(self):
        if self.count == 0:
            return 0.0
        return float(self.buf.get()[:self.count * SUM_GROUPS].astype(np.float64).sum())

    def release(self):
        self.buf.base_data.release()


def fused_available(operator, prox_kind):
    """Whether the fused update can be used with this operator and proximal operator."""
    return prox_kind in PROX_CODES and hasattr(operator, "adjoint_batches_cl")


class FusedUpdate:
    """Kernels and the per-iteration step of the fused FISTA update."""

    def __init__(self, ctx, queue):
        self.queue = queue
        prg = cl.Program(ctx, FUSED_KERNEL).build()
        self.k_update = prg.fista_fused_update
        self.k_sums = prg.partial_sums
        # sum |x| without a coefficient-sized temporary (diagnostic of the L1 term, once per iteration)
        self.abs_sum_kernel = ReductionKernel(ctx, np.float32, neutral="0", reduce_expr="a+b",
                                              map_expr="fabs(x[i])", arguments="__global const float *x")

        self._dummy_mask = clarray.zeros(queue, (1,), np.uint8)

    def batch_buffer(self, operator):
        """The gradient buffer of one orientation batch."""
        return clarray.empty(self.queue, (operator.Nx * operator.Ny * operator.K_batch_max,), np.float32)

    def step(self, operator, r, x, y, g_batch, tau, beta, prox_kind, lam, support_gpu):
        """
        grad = A^T r, batch by batch, and with it x <- prox(y - tau * grad) (support applied),
        y <- x_new + beta * (x_new - x_old), in place. Returns ||grad||^2.
        """
        q = self.queue
        npix = operator.Nx * operator.Ny
        mask = support_gpu if support_gpu is not None else self._dummy_mask
        use_mask = np.int32(support_gpu is not None)
        code = np.int32(PROX_CODES[prox_kind])
        lt = np.float32(lam * tau)
        if getattr(self, "_gsq", None) is None or self._gsq.capacity < len(operator.batches):
            self._gsq = PartialSums(self, len(operator.batches))
        sq = self._gsq.reset()

        def update(k0, Kb):
            n = npix * Kb
            sq.add(g_batch.data, n, 0)
            gws, n64 = elementwise(n)
            self.k_update(q, gws, None, g_batch.data, x.data, y.data, mask.data, use_mask,
                          np.uint64(npix * k0), np.uint64(npix), n64,
                          np.float32(tau), np.float32(beta), lt, code)

        operator.adjoint_batches_cl(r, g_batch, update)
        return sq.total()

    def abs_sum(self, x):
        return float(self.abs_sum_kernel(x).get())
