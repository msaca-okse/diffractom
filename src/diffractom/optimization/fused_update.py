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
"""

PROX_CODES = {"nonneg": 0, "l1": 1, "nonneg_l1": 2}


def fused_available(operator, prox_kind):
    """Whether the fused update can be used with this operator and proximal operator."""
    return prox_kind in PROX_CODES and hasattr(operator, "adjoint_batches_cl")


class FusedUpdate:
    """Kernels and the per-iteration step of the fused FISTA update."""

    def __init__(self, ctx, queue):
        self.queue = queue
        self.k_update = cl.Program(ctx, FUSED_KERNEL).build().fista_fused_update
        # sum |x| without a coefficient-sized temporary (diagnostic of the L1 term)
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
        sq = []

        def update(k0, Kb):
            n = npix * Kb
            gb = g_batch[:n]
            sq.append(clarray.vdot(gb, gb))
            gws, n64 = elementwise(n)
            self.k_update(q, gws, None, g_batch.data, x.data, y.data, mask.data, use_mask,
                          np.uint64(npix * k0), np.uint64(npix), n64,
                          np.float32(tau), np.float32(beta), lt, code)

        operator.adjoint_batches_cl(r, g_batch, update)
        return float(sum(float(s.get()) for s in sq))

    def abs_sum(self, x):
        return float(self.abs_sum_kernel(x).get())
