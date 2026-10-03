# Element-wise kernels: 64-bit indices in a grid-stride loop (see launch.py), so the coefficient and
# data arrays may have more than 2^31 elements.
FISTA_KERNELS = r"""
// Ax <- Ax - b, in place (the prediction buffer becomes the residual)
__kernel void residual_inplace(
    __global float *Ax,
    __global const float *b,
    const ulong n
){
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0))
        Ax[i] = Ax[i] - b[i];
}

// Ax <- w * (Ax - b), in place; one weight per segment (eta bin, ring), shared by all
// (omega, translation): the data are (O, D, nseg) C-order, so the segment is i % nseg
__kernel void weighted_residual_inplace(
    __global float *Ax,
    __global const float *b,
    __global const float *w,
    const ulong nseg,
    const ulong n
){
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0))
        Ax[i] = w[i % nseg] * (Ax[i] - b[i]);
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

RESIDUAL_SUMSQ_KERNEL = r"""
// The data-space steps of an iteration in one pass: r = Ax - b (or w * (Ax - b) with one weight per
// segment), the work-group's partial sum of r^2 (the diagnostic objective), and (clip) r clipped to
// [-delta, delta], written in place. r and its clipping as residual_inplace / weighted_residual_inplace
// and huber_clip_inplace.
__kernel void residual_sumsq(
    __global float *Ax, __global const float *b, __global const float *w, const ulong nseg, const int weighted,
    const int clip, const float delta, const ulong n, __global float *partial, __local float *scratch)
{
    float s = 0.0f;
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        float v = weighted ? w[i % nseg] * (Ax[i] - b[i]) : Ax[i] - b[i];
        s += v * v;
        if (clip) {
            if (v >  delta) v =  delta;
            if (v < -delta) v = -delta;
        }
        Ax[i] = v;
    }
    const int lid = get_local_id(0);
    scratch[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int h = get_local_size(0) / 2; h > 0; h >>= 1) {
        if (lid < h) scratch[lid] += scratch[lid + h];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial[get_group_id(0)] = scratch[0];
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
from .launch import elementwise, pixel_orientation, reduction
from .fused_update import FusedUpdate, fused_available
from .streaming import fits_next_forward, streamer_for
from ..utils.arrays import prepare_inputs

from .prox_tv import (
    TVProxKernels,
    prox_tv_nonneg_inplace,
)


# -------------------- FISTA helper kernels --------------------


def build_fista_program(ctx: cl.Context) -> cl.Program:
    """Compile FISTA + Huber helper OpenCL kernels."""
    return cl.Program(ctx, FISTA_KERNELS + HUBER_KERNELS + RESIDUAL_SUMSQ_KERNEL + HUBER_DIAG_KERNEL).build()


# -------------------- FISTA implementation --------------------

class FISTAHuber:
    """
    Solve: min_x 0.5||A x - b||^2 + g(x)
    with FISTA on GPU.

    x layout: (K, Ny, Nx) C order (the kernels treat it as flat)
    Ax layout: (O, D, Nseg) C
    """

    def __init__(self, operator, prox_kind="nonneg", lam=0.0, L=None, tau=None, tv_niter=50, huber_delta=1e-2,
                 support="fov", fused=True, stream_threads=None, fuse_next_forward=True):
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
        support : "fov", None or (Ny, Nx) bool array
            Support constraint, applied after the proximal operator: the coefficients
            of pixels outside the support are set to zero. "fov" (default) is the disk
            seen by the detector at every angle (operator.support_mask()), i.e. the
            assumption that the sample stays in the field of view; None disables it.
        fused : bool
            Fuse the gradient step, the prox and the momentum update into the adjoint, one
            orientation batch at a time (see fused_update.py): the solver then keeps two
            coefficient-sized arrays (x, y) instead of four (x, y, x_old, grad). Used when the
            prox is element-wise ('nonneg', 'l1', 'nonneg_l1') and the operator provides
            adjoint_batches_cl; otherwise, or with fused=False, the unfused update runs.
        fuse_next_forward : bool
            With the fused update: project each batch's new y forward right after its update in the
            adjoint pass, into a second data-sized buffer, instead of a separate forward pass in the
            next iteration (the same results; streamed, y is uploaded once less per iteration).
            Measured on a V100 (K = 6000, 120 x 120, streamed): 0.36 s per iteration instead of 0.55,
            and on the GPU 0.46 s before. Used only if the buffer fits (an estimate of the GPU memory
            in use, with a margin).
        stream_threads : int, optional
            CPU threads for the host-side copies when the coefficients are streamed (x0 given
            as a NumPy array to run()); default min(8, number of CPUs).
        """
        self.op = operator
        self.stream_threads = stream_threads
        self.fuse_next_forward = bool(fuse_next_forward)
        self.ctx = operator.ctx
        self.queue = operator.queue
        self.huber_delta = float(huber_delta)

        self.fista_prg = build_fista_program(self.ctx)
        self.k_copy_buf       = cl.Kernel(self.fista_prg, "copy_buf")
        self.k_residual_inplace = cl.Kernel(self.fista_prg, "residual_inplace")
        self.k_weighted_residual_inplace = cl.Kernel(self.fista_prg, "weighted_residual_inplace")
        self.k_grad_step      = cl.Kernel(self.fista_prg, "grad_step")
        self.k_extrapolate    = cl.Kernel(self.fista_prg, "extrapolate")
        self.k_huber_clip_inplace = cl.Kernel(self.fista_prg, "huber_clip_inplace")
        self.k_residual_sumsq = cl.Kernel(self.fista_prg, "residual_sumsq")
        self.k_huber_loss = cl.Kernel(self.fista_prg, "huber_loss")

        self.prox_kernels = ProxKernels(self.ctx)
        self.support_gpu = support_mask_to_gpu(self.queue, operator, support)
        self.tv_kernels = TVProxKernels(self.ctx)

        self.prox_kind = prox_kind
        self.lam = float(lam)
        self.fused = bool(fused) and fused_available(operator, prox_kind)
        self.fused_update = FusedUpdate(self.ctx, self.queue) if self.fused else None

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
        x0_gpu:   the starting point, (K, Ny, Nx) float32 (coeffs[k] is the image of orientation k):
                a NumPy array: x and y stay in host memory and are streamed through the GPU batch by
                batch (streaming.py), so the GPU holds no coefficient-sized array (needs the fused
                update). The iterate is updated in place if x0_gpu is C-contiguous float32,
                otherwise in a converted copy; either way run() returns it.
                Or a pyopencl array, C-contiguous float32: everything stays on the GPU.
        out_gpu:  the data b, (N_Omega, My, N_seg) float32: a NumPy array (converted to C-contiguous
                float32 and uploaded) or a C-contiguous pyopencl array.
        weights:  OPTIONAL, N_seg = N_eta * N_rings elements, e.g. shaped (N_eta, N_rings): one weight
                per segment (eta bin, ring), shared by all rotations and translations; the residual
                of a data point in segment j is multiplied by weights[j], zero weight excludes the
                segment. A NumPy array or a C-contiguous float32 pyopencl array.
        returns the solution: the NumPy array (streamed) or x0_gpu (on the GPU)
        """
        t_start = time.perf_counter()  # iteration times in iter_stats count from here (setup in the first)
        q = self.queue

        # ---- inputs: shapes, dtype, layout; NumPy data and weights are uploaded ----
        streamed, x0_gpu, out_gpu, weights, uploaded = prepare_inputs(self.op, q, x0_gpu, out_gpu, weights)
        if streamed and not self.fused:
            raise ValueError("streaming the coefficients (x0 as a NumPy array) needs the fused update: an "
                             "element-wise prox ('nonneg', 'l1', 'nonneg_l1') and fused=True")
        use_weights = weights is not None

        # ---- persistent buffers ----
        # fused: x (current iterate) and y, plus one batch of the gradient;
        # unfused: x, y, x_old and grad; streamed: x and y in host memory
        streamer = None
        if streamed:
            streamer = streamer_for(self.op, self.fused_update, self.stream_threads)
            x_host = streamer.flat(x0_gpu)  # the iterate, updated in place
            y_host = np.empty_like(x_host)
            outside = None if self.support_gpu is None else ~self.support_gpu.get().astype(bool)

            def init_batch(ib):  # start inside the support; y = x: the first iteration reads x
                xv = streamer.view(x_host, ib)
                xv.reshape(-1, outside.size)[:, outside] = 0.0
            if outside is not None:
                list(streamer.pool.map(init_batch, range(len(self.op.batches))))
            x = y = x_old = grad = None
            g_batch = self.fused_update.batch_buffer(self.op)
        elif self.fused:
            x = x0_gpu
            y = clarray.empty(q, x.shape, dtype=np.float32)
            x_old = grad = None
            g_batch = self.fused_update.batch_buffer(self.op)
        else:
            x = x0_gpu
            y = clarray.empty(q, x.shape, dtype=np.float32)
            x_old = clarray.empty(q, x.shape, dtype=np.float32)
            grad = clarray.empty(q, x.shape, dtype=np.float32)
            g_batch = None

        # TV buffers: allocate once per run, reuse each iter
        if self.prox_kind == "nonneg_tv":
            self._tv_buffers = {
                "y":  clarray.empty(q, x.shape, np.float32),
                "gx": clarray.zeros(q, x.shape, np.float32),
                "gy": clarray.zeros(q, x.shape, np.float32),
                "px": clarray.zeros(q, x.shape, np.float32),
                "py": clarray.zeros(q, x.shape, np.float32),
                "div": clarray.zeros(q, x.shape, np.float32),
            }

        # copy x0 -> y,x_old
        if not streamed:
            if self.support_gpu is not None:  # start inside the support
                apply_support(q, self.prox_kernels, x, self.support_gpu)
            gws_x, n_x = elementwise(x.size)
            self.k_copy_buf(q, gws_x, None, x.data, y.data, n_x)
            if x_old is not None:
                self.k_copy_buf(q, gws_x, None, x.data, x_old.data, n_x)
        gws_Ax, n_Ax = elementwise(out_gpu.size)
        red_g, red_l, red_n, n_partial = reduction(out_gpu.size)
        partial = clarray.empty(q, (n_partial,), np.float32)

        Ax = None
        # streamed: the next iteration's forward pass inside the adjoint pass, into a second prediction
        # buffer, if it fits (saves uploading y once per iteration; the same result)
        Ax_next = None
        if self.fused and self.fuse_next_forward and niter > 1:
            solver_bytes = 2 * out_gpu.nbytes + g_batch.nbytes + (weights.nbytes if weights is not None else 0)
            if not streamed:
                solver_bytes += x.nbytes + y.nbytes  # (x0 is on the GPU)
            if fits_next_forward(self.op, streamer, solver_bytes):
                Ax_next = clarray.empty(q, out_gpu.shape, dtype=np.float32, order="C")  # the prediction A(y), overwritten in place by the (weighted, clipped) residual
        t = 1.0

        # ---- diagnostics storage (always collected) ----
        self.iter_stats = []   # list of dicts, one per iteration
        self.final_stats = {}  # summary at the end

        for it in range(niter):
            # ---- Ax = A(y) ----
            if Ax is None:
                Ax = clarray.empty(q, out_gpu.shape, dtype=np.float32, order="C")

            if streamed:
                if Ax_next is not None and it > 0:  # A(y) of this iteration, projected in the last adjoint pass
                    Ax, Ax_next = Ax_next, Ax
                else:
                    streamer.forward(y_host if it > 0 else x_host, Ax)
            elif Ax_next is not None and it > 0:  # A(y), projected in the last adjoint pass
                Ax, Ax_next = Ax_next, Ax
            else:
                self.op.direct_cl(y, Ax)

            # ---- r = w * (Ax - b), its squared norm (f, a diagnostic), then r <- clip(r, -delta, +delta):
            #      one pass over the data, in place in Ax
            # NOTE: with weights, the gradient is A*( clip( w*(Ax-b) ) )
            r = Ax
            self.k_residual_sumsq(q, red_g, red_l, Ax.data, out_gpu.data,
                                  (weights if use_weights else out_gpu).data, np.uint64(out_gpu.shape[-1]),
                                  np.int32(use_weights), np.int32(1), np.float32(self.huber_delta), red_n,
                                  partial.data, cl.LocalMemory(4 * red_l[0]))
            r2 = float(partial.get().astype(np.float64).sum())
            fval = 0.5 * r2

            # ---- momentum coefficient ----
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            beta = (t - 1.0) / t_new

            # NOTE: with weights, the gradient is A*( clip( w*(Ax-b) ) )
            if streamed:
                # ---- as below, with x and y streamed from and to host memory ----
                gsq, xsq, xabs = streamer.adjoint_update(x_host, y_host, r, g_batch, self.tau, beta,
                                                         self.prox_kind, self.lam, self.support_gpu,
                                                         y_src=None if it > 0 else x_host,
                                                         forward_into=Ax_next if it + 1 < niter else None)
                gnorm = float(np.sqrt(gsq))
            elif self.fused:
                # ---- grad = A*(r) batch by batch, each batch consumed by
                #      x <- prox(y - tau*grad), y <- x + beta*(x - x_old) ----
                gnorm = float(np.sqrt(self.fused_update.step(
                    self.op, r, x, y, g_batch, self.tau, beta, self.prox_kind, self.lam, self.support_gpu,
                    forward_into=Ax_next if it + 1 < niter else None)))
            else:
                # ---- grad = A*(r) ----
                self.op.adjoint_cl(r, grad)  # (K, Ny, Nx)

                # ---- v = y - tau*grad ----
                self.k_grad_step(
                    q, gws_x, None,
                    y.data, grad.data, x.data,   # the third argument is the output
                    np.float32(self.tau),
                    n_x
                )

                # ---- prox: apply in-place on v (x), then extrapolation uses y ----
                self._apply_prox(x)  # MUST modify x in-place and return x (or ignore return)

                self.k_extrapolate(
                    q, gws_x, None,
                    x.data, x_old.data, y.data,
                    np.float32(beta),
                    n_x
                )

                # ---- x_old <- x ----
                self.k_copy_buf(q, gws_x, None, x.data, x_old.data, n_x)
                gnorm = float(np.sqrt(clarray.vdot(grad, grad).get()))

            t = t_new

            # ---- regularizer diagnostics ----
            gval = 0.0
            if self.prox_kind in ("l1", "nonneg_l1") and self.lam != 0.0:
                if streamed:
                    gval = self.lam * xabs
                elif self.fused:
                    gval = self.lam * self.fused_update.abs_sum(x)  # no coefficient-sized temporary
                else:
                    gval = self.lam * float(clarray.sum(clmath.fabs(x)).get())

            elif self.prox_kind == "nonneg_tv" and self.lam != 0.0:
                b = self._tv_buffers
                K, Ny, Nx = map(int, x.shape)

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
            xnorm = float(np.sqrt(xsq if streamed else clarray.vdot(x, x).get()))

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
                "time": time.perf_counter() - t_start,  # after the iteration's last host read
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
            "next_forward_fused": Ax_next is not None,
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
        if streamed:
            streamer.use_fused_update(None)  # (the Streamer stays with the operator)
            del y_host
            x = x0_gpu

        for arr in (y, x_old, grad, g_batch):
            if arr is not None:
                arr.base_data.release()

        if Ax is not None:
            Ax.base_data.release()
        for arr in (Ax_next, partial):
            if arr is not None and arr.base_data is not None:
                arr.base_data.release()
        for arr in uploaded:  # data and weights uploaded from NumPy arrays
            arr.base_data.release()

        if self.prox_kind == "nonneg_tv":
            for arr in self._tv_buffers.values():
                arr.base_data.release()
            del self._tv_buffers

        del y, x_old, grad, g_batch, Ax
        import gc
        gc.collect()
        q.finish()
        # --------------------------------------------

        return x