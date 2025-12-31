# prox_operators.py
import pyopencl as cl
import pyopencl.array as clarray
import numpy as np


PROX_KERNELS = r"""
__kernel void prox_nonneg(__global float *x, int total) {
    int gid = get_global_id(0);
    if (gid < total && x[gid] < 0.0f) x[gid] = 0.0f;
}

__kernel void prox_l1(__global float *x, float lambda, int total) {
    int gid = get_global_id(0);
    if (gid >= total) return;
    float v = x[gid];
    float a = fabs(v) - lambda;
    x[gid] = (a > 0.0f) ? copysign(a, v) : 0.0f;
}

__kernel void prox_nonneg_l1(__global float *x, float lambda, int total) {
    int gid = get_global_id(0);
    if (gid >= total) return;
    float v = x[gid] - lambda;
    x[gid] = (v > 0.0f) ? v : 0.0f;
}
"""


class ProxKernels:
    """
    Holds compiled program and cached kernels.
    """
    def __init__(self, ctx: cl.Context):
        self.prg = cl.Program(ctx, PROX_KERNELS).build()

        # Cache kernels ONCE
        self.k_prox_nonneg     = cl.Kernel(self.prg, "prox_nonneg")
        self.k_prox_l1         = cl.Kernel(self.prg, "prox_l1")
        self.k_prox_nonneg_l1  = cl.Kernel(self.prg, "prox_nonneg_l1")


def prox_nonneg(queue, kernels: ProxKernels, x_gpu):
    total = np.int32(x_gpu.size)
    kernels.k_prox_nonneg(
        queue,
        (int(total),),
        None,
        x_gpu.data,
        total,
    )
    return x_gpu


def prox_l1(queue, kernels: ProxKernels, x_gpu, lam, tau):
    total = np.int32(x_gpu.size)
    kernels.k_prox_l1(
        queue,
        (int(total),),
        None,
        x_gpu.data,
        np.float32(lam * tau),
        total,
    )
    return x_gpu


def prox_nonneg_l1(queue, kernels: ProxKernels, x_gpu, lam, tau):
    total = np.int32(x_gpu.size)
    kernels.k_prox_nonneg_l1(
        queue,
        (int(total),),
        None,
        x_gpu.data,
        np.float32(lam * tau),
        total,
    )
    return x_gpu
