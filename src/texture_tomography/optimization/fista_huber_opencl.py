# fista_opencl.py
import time
import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
import pyopencl.clmath as clmath
from package.texture_tomography.optimization.prox import prox_nonneg, prox_l1, prox_nonneg_l1
from package.texture_tomography.optimization.prox import ProxKernels

from package.texture_tomography.optimization.prox_tv import (
    TVProxKernels,
    prox_tv_nonneg_inplace,
)
#######################
#
#    AN EXTRA DATA ARRAY IS ALLOCATED FOR DIAGNOSTICS
#
######################


# -------------------- FISTA helper kernels --------------------

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


def build_fista_program(ctx: cl.Context) -> cl.Program:
    return cl.Program(ctx, FISTA_KERNELS + HUBER_KERNELS + HUBER_DIAG_KERNEL).build()


# -------------------- FISTA implementation --------------------

class FISTAHuberOpenCL:
    """
    Solve: min_x 0.5||A x - b||^2 + g(x)
    with FISTA on GPU.

    x layout: (Nx, Ny, K) Fortran (but we treat it as flat for kernels)
    Ax layout: (O, D, Nseg) C
    """

    def __init__(self, operator, prox_kind="nonneg", lam=0.0, L=None, tau=None, tv_niter=50, huber_delta=1e-2):
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

            return x_gpu



    def run(
        self,
        x0_gpu: clarray.Array,
        out_gpu: clarray.Array,
        niter: int,
        verbose: int = 0,
        diagnostics_interval: int = 1,
    ):
        """
        x0_gpu: clarray (Nx, Ny, K) float32, order='F'
        out_gpu:  clarray (O, D, Nseg) float32, order='C'
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

        # ---- persistent buffers ----
        x = clarray.empty(q, x0_gpu.shape, dtype=np.float32, order="F")
        y = clarray.empty(q, x0_gpu.shape, dtype=np.float32, order="F")
        x_old = clarray.empty(q, x0_gpu.shape, dtype=np.float32, order="F")
        v = clarray.empty(q, x0_gpu.shape, dtype=np.float32, order="F")
        grad = clarray.empty(q, x0_gpu.shape, dtype=np.float32, order="F")

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

        # copy x0 -> x,y,x_old
        total_x = np.int32(x0_gpu.size)
        self.k_copy_buf(q, (int(total_x),), None, x0_gpu.data, x.data, total_x)
        self.k_copy_buf(q, (int(total_x),), None, x0_gpu.data, y.data, total_x)
        self.k_copy_buf(q, (int(total_x),), None, x0_gpu.data, x_old.data, total_x)
        q.finish()

        Ax = None
        r = None
        t = 1.0

        # timers (optional)
        t_forward = 0.0
        t_residual = 0.0
        t_adjoint = 0.0
        t_gradstep = 0.0
        t_prox = 0.0
        t_extrap = 0.0
        t_obj = 0.0

        t_total_start = time.perf_counter()
        # ---- diagnostics storage (always collected) ----
        self.iter_stats = []   # list of dicts, one per iteration
        self.final_stats = {}  # summary at the end


        for it in range(niter):
            # ---- Ax = A(y) ----
            t0 = time.perf_counter()
            if Ax is None:
                Ax = clarray.empty(q, out_gpu.shape, dtype=np.float32, order="C")
                r_raw = clarray.empty(q, out_gpu.shape, dtype=np.float32, order="C")
                r  = clarray.empty(q, out_gpu.shape, dtype=np.float32, order="C")

            self.op.direct_cl(y, Ax)
            q.finish()

            t_forward += time.perf_counter() - t0


            # ---- r = Ax - b ----
            # ---- r = Ax - b ----
            total_Ax = np.int32(Ax.size)
            self.k_residual_axpb(
                q, (int(total_Ax),), None,
                Ax.data, out_gpu.data, r.data,
                total_Ax
            )
            q.finish()

            self.k_copy_buf(q, (int(total_Ax),), None, r.data, r_raw.data, total_Ax)
            q.finish()


            # ---- huber: r <- clip(r, -delta, +delta) ----
            self.k_huber_clip_inplace(
                q, (int(total_Ax),), None,
                r.data,
                np.float32(self.huber_delta),
                total_Ax
            )
            q.finish()


            # ---- grad = A*(r) ----
            t0 = time.perf_counter()
            self.op.adjoint_cl(r, grad)  # (Nx,Ny,K) Fortran
            q.finish()
            t_adjoint += time.perf_counter() - t0


            # ---- v = y - tau*grad ----
            t0 = time.perf_counter()
            total_x = np.int32(y.size)
            self.k_grad_step(
                q, (int(total_x),), None,
                y.data, grad.data, v.data,
                np.float32(self.tau),
                total_x
            )
            q.finish()
            t_gradstep += time.perf_counter() - t0

            # ---- prox: apply in-place on v, then copy v -> x ----
            t0 = time.perf_counter()
            self._apply_prox(v)   # MUST modify v in-place and return v
            q.finish()

            self.k_copy_buf(q, (int(total_x),), None, v.data, x.data, total_x)
            q.finish()
            t_prox += time.perf_counter() - t0

            # ---- momentum update ----
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            beta = (t - 1.0) / t_new

            # ---- y = x + beta*(x - x_old) ----
            t0 = time.perf_counter()
            self.k_extrapolate(
                q, (int(total_x),), None,
                x.data, x_old.data, y.data,
                np.float32(beta),
                total_x
            )
            q.finish()
            t_extrap += time.perf_counter() - t0

            # ---- x_old <- x ----
            self.k_copy_buf(q, (int(total_x),), None, x.data, x_old.data, total_x)

            t = t_new

            # ---- diagnostics ----
            t0 = time.perf_counter()

            # ---- Huber data term ----
            tmp = clarray.empty(q, r.shape, dtype=np.float32, order="C")

            total_r = np.int32(r.size)
            self.k_huber_loss(
                q, (int(total_r),), None,
                r.data,
                tmp.data,
                np.float32(self.huber_delta),
                total_r
            )
            q.finish()

            fval = float(clarray.sum(tmp).get())
            del tmp


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
            xnorm = float(np.sqrt(clarray.vdot(x, x).get()))
            gnorm = float(np.sqrt(clarray.vdot(grad, grad).get()))

            tv_res = None
            if self.prox_kind == "nonneg_tv":
                tv_res = self._last_tv_residual

            t_obj += time.perf_counter() - t0

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
            })

            # ---- conditional printing only ----
            if verbose and ((it + 1) % diagnostics_interval == 0 or it == 0 or it == niter - 1):
                extra = ""
                if tv_res is not None:
                    extra = f"  tv_res={tv_res:.3e}"

                print(
                    f"[iter {it+1:4d}/{niter}] "
                    f"obj={obj:.6e}  f={fval:.6e}  g={gval:.6e}  "
                    f"||x||={xnorm:.6e}  ||grad||={gnorm:.6e}  "
                    f"tau={self.tau:.3e}  beta={beta:.3e}"
                    + extra
                )


        total_time = time.perf_counter() - t_total_start

        if verbose:
            print("\n=== FISTA(OpenCL) TIMING SUMMARY ===")
            print(f"forward (A) total     : {t_forward:.4f} s")
            print(f"residual total        : {t_residual:.4f} s")
            print(f"adjoint (A*) total    : {t_adjoint:.4f} s")
            print(f"grad step total       : {t_gradstep:.4f} s")
            print(f"prox total            : {t_prox:.4f} s")
            print(f"extrap total          : {t_extrap:.4f} s")
            print(f"diagnostics total     : {t_obj:.4f} s")
            print("-----------------------------------")
            print(f"TOTAL                 : {total_time:.4f} s")
            print("===================================\n")


                # ---- final summary stats ----
        self.final_stats = {
            "niter": niter,
            "final_f": self.iter_stats[-1]["f"],
            "final_g": self.iter_stats[-1]["g"],
            "final_obj": self.iter_stats[-1]["obj"],
            "final_xnorm": self.iter_stats[-1]["xnorm"],
            "final_gradnorm": self.iter_stats[-1]["gradnorm"],
            "total_time": total_time,
            "timing": {
                "forward": t_forward,
                "residual": t_residual,
                "adjoint": t_adjoint,
                "grad_step": t_gradstep,
                "prox": t_prox,
                "extrap": t_extrap,
                "diagnostics": t_obj,
            },
        }


        return x
