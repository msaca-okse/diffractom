FISTA_KERNELS = r"""
__kernel void residual_axpb(
    __global const float *Ax,
    __global const float *b,
    __global float *r,
    const int total
){
    int gid = get_global_id(0);
    if (gid >= total) return;
    r[gid] = Ax[gid] - b[gid];
}

// v = y - tau * grad
__kernel void grad_step(
    __global const float *y,
    __global const float *grad,
    __global float *v,
    const float tau,
    const int total
){
    int gid = get_global_id(0);
    if (gid >= total) return;
    v[gid] = y[gid] - tau * grad[gid];
}

// y = x + beta * (x - x_old)
__kernel void extrapolate(
    __global const float *x,
    __global const float *x_old,
    __global float *y,
    const float beta,
    const int total
){
    int gid = get_global_id(0);
    if (gid >= total) return;
    float xv = x[gid];
    y[gid] = xv + beta * (xv - x_old[gid]);
}

// x_old <- x (copy kernel to avoid enqueue_copy corner-cases with strides)
__kernel void copy_buf(
    __global const float *src,
    __global float *dst,
    const int total
){
    int gid = get_global_id(0);
    if (gid >= total) return;
    dst[gid] = src[gid];
}
"""

HUBER_KERNELS = r"""
__kernel void huber_clip_inplace(
    __global float *r,
    const float delta,
    const int n
){
    int gid = get_global_id(0);
    if (gid >= n) return;

    float v = r[gid];
    if (v >  delta) v =  delta;
    if (v < -delta) v = -delta;
    r[gid] = v;
}
"""

HUBER_DIAG_KERNEL = r"""
__kernel void huber_loss(
    __global const float *r,
    __global float *out,
    const float delta,
    const int n
){
    int gid = get_global_id(0);
    if (gid >= n) return;

    float v = fabs(r[gid]);
    float val;
    if (v <= delta)
        val = 0.5f * v * v;
    else
        val = delta * (v - 0.5f * delta);

    out[gid] = val;
}
"""


# fista_opencl.py
import time
import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
import pyopencl.clmath as clmath
from .prox import prox_nonneg, prox_l1, prox_nonneg_l1, ProxKernels

from .prox_tv import (
    TVProxKernels,
    prox_tv_nonneg_inplace,
)
#######################
#
#    AN EXTRA DATA ARRAY IS ALLOCATED FOR DIAGNOSTICS
#
######################


# -------------------- FISTA helper kernels --------------------


def build_fista_program(ctx: cl.Context) -> cl.Program:
    """Compile FISTA + Huber helper OpenCL kernels."""
    return cl.Program(ctx, FISTA_KERNELS + HUBER_KERNELS + HUBER_DIAG_KERNEL).build()


# -------------------- FISTA implementation --------------------

class FISTAHuberCPU:
    """
    Solve: min_x 0.5||A x - b||^2 + g(x)
    with FISTA on GPU.

    x layout: (Nx, Ny, K) Fortran (but we treat it as flat for kernels)
    Ax layout: (O, D, Nseg) C
    """

    def __init__(self, operator, prox_kind="nonneg", lam=0.0, L=None, tau=None, tv_niter=50, huber_delta=1e-2):
        """Set up FISTA-Huber solver.

        Parameters
        ----------
        operator : SinglePhaseForwardOperator or MultiPhaseForwardOperator
        prox_kind : str
            One of 'nonneg', 'l1', 'nonneg_l1', 'nonneg_tv'.
        lam : float
            Regularisation weight.
        L : float, optional
            Lipschitz constant; tau = 1/L if tau not given.
        tau : float, optional
            Step size (overrides L).
        tv_niter : int
            Inner iterations for TV proximal operator.
        huber_delta : float
            Huber loss transition threshold.
        """
        self.op = operator
        self.ctx = operator.ctx
        self.queue = operator.queue
        self.huber_delta = float(huber_delta)

        # NOTE: x/y/x_old/grad (the "coeffs" arrays) are now CPU-resident
        # F-order numpy arrays, consistent with SinglePhaseForwardOperator_cpu.
        # Only the data-space vector ops (residual, huber clip) still run on
        # the GPU, since Ax/r/out_gpu stay GPU-resident.
        self.fista_prg = build_fista_program(self.ctx)
        self.k_residual_axpb  = cl.Kernel(self.fista_prg, "residual_axpb")
        self.k_huber_clip_inplace = cl.Kernel(self.fista_prg, "huber_clip_inplace")
        self.k_huber_loss = cl.Kernel(self.fista_prg, "huber_loss")

        self.prox_kind = prox_kind
        self.lam = float(lam)

        if tau is None:
            if L is None:
                raise ValueError("Provide either L or tau")
            self.tau = 1.0 / float(L)
        else:
            self.tau = float(tau)

        self.tv_niter = int(tv_niter)



        if tau is None:
            if L is None:
                raise ValueError("Provide either L or tau")
            self.tau = 1.0 / float(L)
        else:
            self.tau = float(tau)




    def _apply_prox(self, x):
            """Dispatch to the chosen proximal operator in place (CPU array).

            NOTE: only 'nonneg' is implemented for the CPU-coeffs variant.
            """
            if self.prox_kind == "nonneg":
                np.maximum(x, 0.0, out=x)

            else:
                raise NotImplementedError(
                    f"FISTAHuber (cpu) only supports prox_kind='nonneg' (got {self.prox_kind!r}); "
                    "l1/tv variants are not implemented for CPU-resident coeffs."
                )

            return x


    def run(
        self,
        x0: np.ndarray,
        out_gpu: clarray.Array,
        niter: int,
        weights: clarray.Array | None = None,
        verbose: int = 0,
        diagnostics_interval: int = 1,
    ):
        """
        x0:       numpy.ndarray (Nx, Ny, K) float32, order='F', on CPU (host).
        out_gpu:  clarray (O, D, Nseg) float32, order='C'
        weights:  OPTIONAL clarray (O, D, Nseg) float32, order='C'
                If provided, residuals are multiplied elementwise by weights.
                weights==0 masks out (ignores) corrupted data points.
        returns x solution: numpy.ndarray (same layout as x0), on CPU (host)
        """
        q = self.queue

        # ---- basic checks ----
        if not isinstance(x0, np.ndarray):
            raise TypeError("x0 must be a numpy.ndarray (CPU-resident coeffs)")
        if not isinstance(out_gpu, clarray.Array):
            raise TypeError("out_gpu must be pyopencl.array.Array")

        if out_gpu.queue is None:
            raise ValueError("out_gpu must have a queue attached (created via clarray on a queue)")

        if x0.dtype != np.float32 or out_gpu.dtype != np.float32:
            raise TypeError("This FISTA assumes float32 arrays")

        if not x0.flags.f_contiguous:
            raise ValueError("x0 must be Fortran-order (order='F')")

        if out_gpu.queue.context.int_ptr != self.ctx.int_ptr:
            raise ValueError("out_gpu context != operator context")

        # ---- optional weights checks ----
        use_weights = weights is not None
        if use_weights:
            if not isinstance(weights, clarray.Array):
                raise TypeError("weights must be a pyopencl.array.Array or None")
            if weights.queue is None:
                raise ValueError("weights must have a queue attached")
            if weights.dtype != np.float32:
                raise TypeError("weights must be float32")
            if weights.queue.context.int_ptr != self.ctx.int_ptr:
                raise ValueError("weights context != operator context")
            if weights.shape != out_gpu.shape:
                raise ValueError(f"weights.shape {weights.shape} must match out_gpu.shape {out_gpu.shape}")

        # ---- persistent buffers ----
        # x/y/x_old/grad are CPU-resident F-order numpy arrays (the "coeffs"),
        # consistent with SinglePhaseForwardOperator_cpu.direct_cl/adjoint_cl.
        x = x0
        y = np.empty(x.shape, dtype=np.float32, order="F")
        x_old = np.empty(x.shape, dtype=np.float32, order="F")
        grad = np.empty(x.shape, dtype=np.float32, order="F")

        if self.prox_kind == "nonneg_tv":
            raise NotImplementedError(
                "prox_kind='nonneg_tv' is not implemented for CPU-resident coeffs."
            )

        # copy x0 -> y, x_old
        total_x = np.int32(x.size)
        np.copyto(y, x)
        np.copyto(x_old, x)

        Ax = None
        r = None
        t = 1.0

        # ---- diagnostics storage (always collected) ----
        self.iter_stats = []   # list of dicts, one per iteration
        self.final_stats = {}  # summary at the end

        for it in range(niter):
            # ---- Ax = A(y) ----
            if Ax is None:
                Ax = clarray.empty(q, out_gpu.shape, dtype=np.float32, order="C")
                r  = clarray.empty(q, out_gpu.shape, dtype=np.float32, order="C")

            self.op.direct_cl(y, Ax)

            total_Ax = np.int32(Ax.size)

            # ---- r = Ax - b ----
            self.k_residual_axpb(
                q, (int(total_Ax),), None,
                Ax.data, out_gpu.data, r.data,
                total_Ax
            )

            # ---- optional weighting: r <- w * r ----
            if use_weights:
                # (requires weights to be contiguous like r/out_gpu; all are order='C' here)
                # elementwise multiply in-place on r
                r *= weights

            # ---- L2 data term (diagnostic only) ----
            # f = 0.5 * || (w*(Ax-b)) ||^2   if weights is provided
            # f = 0.5 * || (Ax-b) ||^2       otherwise
            r2 = float(np.dot(r.get().ravel(), r.get().ravel()))
            fval = 0.5 * float(r2)

            # ---- huber: r <- clip(r, -delta, +delta) ----
            self.k_huber_clip_inplace(
                q, (int(total_Ax),), None,
                r.data,
                np.float32(self.huber_delta),
                total_Ax
            )

            # ---- grad = A*(r) ----
            # NOTE: with weights, this is A*( clip( w*(Ax-b) ) )
            self.op.adjoint_cl(r, grad)  # (Nx,Ny,K) Fortran, CPU-resident

            # ---- v = y - tau*grad ----
            np.subtract(y, self.tau * grad, out=x)

            # ---- prox: apply in-place on v (x), then extrapolation uses y ----
            self._apply_prox(x)  # MUST modify x in-place and return x (or ignore return)

            # ---- momentum update ----
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            beta = (t - 1.0) / t_new

            # ---- y = x + beta * (x - x_old) ----
            np.add(x, beta * (x - x_old), out=y)

            # ---- x_old <- x ----
            np.copyto(x_old, x)

            t = t_new

            # ---- regularizer diagnostics ----
            gval = 0.0
            if self.prox_kind in ("l1", "nonneg_l1") and self.lam != 0.0:
                gval = self.lam * float(clarray.sum(clmath.fabs(x)).get())

            elif self.prox_kind == "nonneg_tv" and self.lam != 0.0:
                b = self._tv_buffers
                total_x_tv = np.int32(x.size)

                self.tv_kernels.k_grad(
                    q, (int(total_x_tv),), None,
                    x.data,
                    b["gx"].data,
                    b["gy"].data,
                    np.int32(x.shape[0]),
                    np.int32(x.shape[1]),
                    np.int32(x.shape[2]),
                )

                self.tv_kernels.k_norm(
                    q, (int(total_x_tv),), None,
                    b["gx"].data,
                    b["gy"].data,
                    b["div"].data,
                    total_x_tv,
                )

                gval = self.lam * float(clarray.sum(b["div"]).get())

            obj = fval + gval
            xnorm = float(np.linalg.norm(x))
            gnorm = float(np.linalg.norm(grad))

            tv_res = None
            if self.prox_kind == "nonneg_tv":
                tv_res = getattr(self, "_last_tv_residual", None)

            # ---- store per-iteration stats ----
            self.iter_stats.append({
                "iter": it + 1,
                "f": fval,
                "g": gval,
                "obj": obj,
                "xnorm": xnorm,
                "gradnorm": gnorm,
                "beta": beta,
                "tau": self.tau,
                "tv_residual": tv_res,
                "weighted": use_weights,
            })

            # ---- conditional printing only ----
            if verbose and ((it + 1) % diagnostics_interval == 0 or it == 0 or it == niter - 1):
                extra = ""
                if tv_res is not None:
                    extra += f"  tv_res={tv_res:.3e}"
                if use_weights:
                    extra += "  (weighted)"
                print(
                    f"[iter {it+1:4d}/{niter}] "
                    f"obj={obj:.6e}  f={fval:.6e}  g={gval:.6e}  "
                    f"||x||={xnorm:.6e}  ||grad||={gnorm:.6e}  "
                    f"tau={self.tau:.3e}  beta={beta:.3e}"
                    + extra
                )

        # ---- final summary stats ----
        self.final_stats = {
            "niter": niter,
            "final_f": self.iter_stats[-1]["f"],
            "final_g": self.iter_stats[-1]["g"],
            "final_obj": self.iter_stats[-1]["obj"],
            "final_xnorm": self.iter_stats[-1]["xnorm"],
            "final_gradnorm": self.iter_stats[-1]["gradnorm"],
            "weighted": use_weights,
        }

        # ---------------- GPU cleanup ----------------
        # x/y/x_old/grad are plain CPU numpy arrays now, nothing to release.
        q.finish()

        if Ax is not None:
            Ax.base_data.release()
        if r is not None:
            r.base_data.release()

        del y, x_old, grad, Ax, r
        import gc
        gc.collect()
        q.finish()
        # --------------------------------------------

        return x