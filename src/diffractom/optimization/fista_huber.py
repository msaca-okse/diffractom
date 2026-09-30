# Element-wise kernels: 64-bit indices in a grid-stride loop (see launch.py), so the coefficient and
# data arrays may have more than 2^31 elements.
FISTA_KERNELS = r"""
__kernel void residual_axpb(
    __global const float *Ax,
    __global const float *b,
    __global float *r,
    const ulong n
){
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0))
        r[i] = Ax[i] - b[i];
}

// v = y - tau * grad
__kernel void grad_step(
    __global const float *y,
    __global const float *grad,
    __global float *v,
    const float tau,
    const ulong n
){
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0))
        v[i] = y[i] - tau * grad[i];
}

// y = x + beta * (x - x_old)
__kernel void extrapolate(
    __global const float *x,
    __global const float *x_old,
    __global float *y,
    const float beta,
    const ulong n
){
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        float xv = x[i];
        y[i] = xv + beta * (xv - x_old[i]);
    }
}

// x_old <- x (copy kernel to avoid enqueue_copy corner-cases with strides)
__kernel void copy_buf(
    __global const float *src,
    __global float *dst,
    const ulong n
){
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0))
        dst[i] = src[i];
}
"""

HUBER_KERNELS = r"""
__kernel void huber_clip_inplace(
    __global float *r,
    const float delta,
    const ulong n
){
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        float v = r[i];
        if (v >  delta) v =  delta;
        if (v < -delta) v = -delta;
        r[i] = v;
    }
}
"""

HUBER_DIAG_KERNEL = r"""
__kernel void huber_loss(
    __global const float *r,
    __global float *out,
    const float delta,
    const ulong n
){
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        float v = fabs(r[i]);
        out[i] = (v <= delta) ? 0.5f * v * v : delta * (v - 0.5f * delta);
    }
}
"""


# fista_opencl.py
import time
import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
import pyopencl.clmath as clmath
from .prox import prox_nonneg, prox_l1, prox_nonneg_l1, ProxKernels, apply_support, support_mask_to_gpu
from .launch import elementwise, pixel_orientation

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

class FISTAHuber:
    """
    Solve: min_x 0.5||A x - b||^2 + g(x)
    with FISTA on GPU.

    x layout: (Nx, Ny, K) Fortran (but we treat it as flat for kernels)
    Ax layout: (O, D, Nseg) C
    """

    def __init__(self, operator, prox_kind="nonneg", lam=0.0, L=None, tau=None, tv_niter=50, huber_delta=1e-2,
                 support="fov"):
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
        support : "fov", None or (Nx, Ny) bool array
            Support constraint, applied after the proximal operator: the coefficients
            of pixels outside the support are set to zero. "fov" (default) is the disk
            seen by the detector at every angle (operator.support_mask()), i.e. the
            assumption that the sample stays in the field of view; None disables it.
        """
        self.op = operator
        self.ctx = operator.ctx
        self.queue = operator.queue
        self.huber_delta = float(huber_delta)

        self.fista_prg = build_fista_program(self.ctx)
        self.k_copy_buf       = cl.Kernel(self.fista_prg, "copy_buf")
        self.k_residual_axpb  = cl.Kernel(self.fista_prg, "residual_axpb")
        self.k_grad_step      = cl.Kernel(self.fista_prg, "grad_step")
        self.k_extrapolate    = cl.Kernel(self.fista_prg, "extrapolate")
        self.k_huber_clip_inplace = cl.Kernel(self.fista_prg, "huber_clip_inplace")
        self.k_huber_loss = cl.Kernel(self.fista_prg, "huber_loss")

        self.prox_kernels = ProxKernels(self.ctx)
        self.support_gpu = support_mask_to_gpu(self.queue, operator, support)
        self.tv_kernels = TVProxKernels(self.ctx)

        self.prox_kind = prox_kind
        self.lam = float(lam)

        if tau is None:
            if L is None:
                raise ValueError("Provide either L or tau")
            self.tau = 1.0 / float(L)
        else:
            self.tau = float(tau)

        self.tv_niter = int(tv_niter)

        # prox dispatch
        self._prox_nonneg     = prox_nonneg
        self._prox_l1         = prox_l1
        self._prox_nonneg_l1  = prox_nonneg_l1
        self._prox_nonneg_tv  = prox_tv_nonneg_inplace



        if tau is None:
            if L is None:
                raise ValueError("Provide either L or tau")
            self.tau = 1.0 / float(L)
        else:
            self.tau = float(tau)




    def _apply_prox(self, x_gpu):
            """Dispatch to the chosen proximal operator in place."""
            if self.prox_kind == "nonneg":
                self._prox_nonneg(self.queue, self.prox_kernels, x_gpu)

            elif self.prox_kind == "l1":
                self._prox_l1(self.queue, self.prox_kernels, x_gpu, self.lam, self.tau)

            elif self.prox_kind == "nonneg_l1":
                self._prox_nonneg_l1(self.queue, self.prox_kernels, x_gpu, self.lam, self.tau)

            elif self.prox_kind == "nonneg_tv":
                b = self._tv_buffers
                x_gpu, tv_res = self._prox_nonneg_tv(
                    queue=self.queue,
                    kernels=self.tv_kernels,
                    x_gpu=x_gpu,
                    y_gpu=b["y"],
                    ux=b["gx"],
                    uy=b["gy"],
                    px=b["px"],
                    py=b["py"],
                    div=b["div"],
                    weight=self.lam,
                    tau=self.tau,
                    n_iter=self.tv_niter,
                    return_stats=True
                )
                self._last_tv_residual = tv_res

            else:
                raise ValueError(f"Unknown prox_kind: {self.prox_kind}")

            if self.support_gpu is not None:
                apply_support(self.queue, self.prox_kernels, x_gpu, self.support_gpu)

            return x_gpu


    def run(
        self,
        x0_gpu: clarray.Array,
        out_gpu: clarray.Array,
        niter: int,
        weights: clarray.Array | None = None,
        verbose: int = 0,
        diagnostics_interval: int = 1,
    ):
        """
        x0_gpu:   clarray (Nx, Ny, K) float32, order='F'
        out_gpu:  clarray (O, D, Nseg) float32, order='C'
        weights:  OPTIONAL clarray (O, D, Nseg) float32, order='C'
                If provided, residuals are multiplied elementwise by weights.
                weights==0 masks out (ignores) corrupted data points.
        returns x_gpu solution (same layout as x0_gpu)
        """
        q = self.queue

        # ---- basic checks ----
        if not isinstance(x0_gpu, clarray.Array) or not isinstance(out_gpu, clarray.Array):
            raise TypeError("x0_gpu and out_gpu must be pyopencl.array.Array")

        if x0_gpu.queue is None or out_gpu.queue is None:
            raise ValueError("Arrays must have a queue attached (created via clarray on a queue)")

        if x0_gpu.dtype != np.float32 or out_gpu.dtype != np.float32:
            raise TypeError("This FISTA assumes float32 arrays")

        if x0_gpu.queue.context.int_ptr != self.ctx.int_ptr:
            raise ValueError("x0_gpu context != operator context")
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
        x = x0_gpu
        y = clarray.empty(q, x.shape, dtype=np.float32, order="F")
        x_old = clarray.empty(q, x.shape, dtype=np.float32, order="F")
        grad = clarray.empty(q, x.shape, dtype=np.float32, order="F")

        # TV buffers: allocate once per run, reuse each iter
        if self.prox_kind == "nonneg_tv":
            self._tv_buffers = {
                "y":  clarray.empty(q, x.shape, np.float32, order="F"),
                "gx": clarray.zeros(q, x.shape, np.float32, order="F"),
                "gy": clarray.zeros(q, x.shape, np.float32, order="F"),
                "px": clarray.zeros(q, x.shape, np.float32, order="F"),
                "py": clarray.zeros(q, x.shape, np.float32, order="F"),
                "div": clarray.zeros(q, x.shape, np.float32, order="F"),
            }

        # copy x0 -> y,x_old
        if self.support_gpu is not None:  # start inside the support
            apply_support(q, self.prox_kernels, x, self.support_gpu)
        gws_x, n_x = elementwise(x.size)
        self.k_copy_buf(q, gws_x, None, x.data, y.data, n_x)
        self.k_copy_buf(q, gws_x, None, x.data, x_old.data, n_x)
        gws_Ax, n_Ax = elementwise(out_gpu.size)

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

            # ---- r = Ax - b ----
            self.k_residual_axpb(
                q, gws_Ax, None,
                Ax.data, out_gpu.data, r.data,
                n_Ax
            )

            # ---- optional weighting: r <- w * r ----
            if use_weights:
                # (requires weights to be contiguous like r/out_gpu; all are order='C' here)
                # elementwise multiply in-place on r
                r *= weights

            # ---- L2 data term (diagnostic only) ----
            # f = 0.5 * || (w*(Ax-b)) ||^2   if weights is provided
            # f = 0.5 * || (Ax-b) ||^2       otherwise
            r2 = float(clarray.vdot(r, r).get())  # on the GPU (copying r to the host dominated the iteration time)
            fval = 0.5 * float(r2)

            # ---- huber: r <- clip(r, -delta, +delta) ----
            self.k_huber_clip_inplace(
                q, gws_Ax, None,
                r.data,
                np.float32(self.huber_delta),
                n_Ax
            )

            # ---- grad = A*(r) ----
            # NOTE: with weights, this is A*( clip( w*(Ax-b) ) )
            self.op.adjoint_cl(r, grad)  # (Nx,Ny,K) Fortran

            # ---- v = y - tau*grad ----
            self.k_grad_step(
                q, gws_x, None,
                y.data, grad.data, x.data,   # the third argument is the output
                np.float32(self.tau),
                n_x
            )

            # ---- prox: apply in-place on v (x), then extrapolation uses y ----
            self._apply_prox(x)  # MUST modify x in-place and return x (or ignore return)

            # ---- momentum update ----
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            beta = (t - 1.0) / t_new

            self.k_extrapolate(
                q, gws_x, None,
                x.data, x_old.data, y.data,
                np.float32(beta),
                n_x
            )

            # ---- x_old <- x ----
            self.k_copy_buf(q, gws_x, None, x.data, x_old.data, n_x)

            t = t_new

            # ---- regularizer diagnostics ----
            gval = 0.0
            if self.prox_kind in ("l1", "nonneg_l1") and self.lam != 0.0:
                gval = self.lam * float(clarray.sum(clmath.fabs(x)).get())

            elif self.prox_kind == "nonneg_tv" and self.lam != 0.0:
                b = self._tv_buffers
                Nx, Ny, K = map(int, x.shape)

                self.tv_kernels.k_grad(
                    q, pixel_orientation(Nx * Ny, K), None,
                    x.data,
                    b["gx"].data,
                    b["gy"].data,
                    np.int32(Nx),
                    np.int32(Ny),
                    np.int32(K),
                )

                self.tv_kernels.k_norm(
                    q, gws_x, None,
                    b["gx"].data,
                    b["gy"].data,
                    b["div"].data,
                    n_x,
                )

                gval = self.lam * float(clarray.sum(b["div"]).get())

            obj = fval + gval
            xnorm = float(np.sqrt(clarray.vdot(x, x).get()))
            gnorm = float(np.sqrt(clarray.vdot(grad, grad).get()))

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
        q.finish()

        for arr in (y, x_old, grad):
            if arr is not None:
                arr.base_data.release()

        if Ax is not None:
            Ax.base_data.release()
        if r is not None:
            r.base_data.release()

        if self.prox_kind == "nonneg_tv":
            for arr in self._tv_buffers.values():
                arr.base_data.release()
            del self._tv_buffers

        del y, x_old, grad, Ax, r
        import gc
        gc.collect()
        q.finish()
        # --------------------------------------------

        return x