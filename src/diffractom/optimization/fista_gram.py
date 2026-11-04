"""Weighted least squares with FISTA in Gram form: no data-space array during the iterations.

The forward operator of SinglePhaseForwardOperator is, per rotation r, a projection followed by a pole-figure
product: (A x)_r = S_r P_r, with S_r (My x K) the projections of the K orientation images and P_r (K x N_seg) the
pole-figure matrix. The gradient of 0.5 ||w (A x - b)||^2 is therefore A^T (W (S P - b)), W = w^2, and

    S_r P_r W P_r^T - b_r W P_r^T = S_r G_r - B_r,   G_r = P_r W P_r^T (K x K),   B_r = b_r W P_r^T (My x K).

G and B are formed once. An iteration then projects the orientation images, multiplies by G_r (My K^2 per
rotation instead of 2 My K N_seg for the forward and adjoint pole-figure products) and backprojects; the data
array is never touched. The objective follows from the same quantities, 0.5 <S, S G> - <S, B> + 0.5 ||w b||^2.
This pays off when K is small compared with N_seg (a few thousand orientations against thousands of detector
segments), which is also the limit of the storage: G holds N_Omega K^2 floats (0.9 GB for K = 800 and 360
rotations, 5.6 GB for K = 2000).

The idea is that of the "projected form" of the GPU fit in textom 0.7.21 (M. Frewein, M. Allain, T. Grunewald et
al., https://gitlab.fresnel.fr/textom/textom; Frewein et al., IUCrJ 11, 809-820, 2024), where the diffractlets of
a rotation enter only through such Gram matrices. This module is an independent implementation for diffractom's
operators (no textom code is used).
"""
import time

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
from pyclblast import gemmStridedBatched

from .fista_l2 import build_fista_program
from .launch import elementwise
from .prox import ProxKernels, apply_support, prox_l1, prox_nonneg, prox_nonneg_l1, support_mask_to_gpu
from ..utils.arrays import as_host, check_device

_KERNELS = r"""
// v[i] *= w[i % period] (rows of a PF chunk (R, Kc, N_seg), or data (R, My, N_seg), by the segment weights)
__kernel void scale_periodic(__global float *v, __global const float *w, const ulong period, const ulong n)
{
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0))
        v[i] *= w[i % period];
}

// G[r, i, j] = G[r, j, i] for j < i < K (the lower triangle from the upper one), r < R
__kernel void mirror_upper(__global float *G, const int Kp, const int K, const ulong n)
{
    for (size_t t = get_global_id(0); t < n; t += get_global_size(0)) {
        const size_t r = t / ((size_t)K * K);
        const int i = (int)((t / K) % K), j = (int)(t % K);
        if (i > j) G[r * Kp * Kp + (size_t)i * Kp + j] = G[r * Kp * Kp + (size_t)j * Kp + i];
    }
}

"""


def _segment_weights(op, weights):
    """W = w^2 per segment (N_seg,) float32, or None. Per-point weights cannot enter G."""
    if weights is None:
        return None
    w = np.asarray(weights.get() if isinstance(weights, clarray.Array) else weights, np.float32)
    if w.size != op.N_seg:
        raise ValueError(f"FISTAGram needs one weight per segment ({op.N_seg} values, e.g. shaped (N_eta, N_rings)); "
                         f"got {w.shape}. Weights per data point cannot enter the Gram matrices.")
    return np.ascontiguousarray((w.ravel() ** 2).astype(np.float32))


class FISTAGram:
    """Solve min_x 0.5 ||w (A x - b)||^2 + g(x) with FISTA, using the Gram form above.

    The iterates are those of FISTAHuber with huber_delta = inf (and FISTAL2 without weights) up to rounding, when
    the weights are 0 or 1; with other weights FISTAGram uses the exact gradient A^T w^2 (A x - b) (FISTAHuber's
    weighted residual enters its gradient once, A^T w (A x - b)).

    Parameters
    ----------
    operator : SinglePhaseForwardOperator with the native or FFT projector (not gratopy), dense coefficient
        layout; any pf_mode (G and B are built from the dense pole-figure evaluation, chunk by chunk).
    prox_kind : "nonneg" (default), "l1" or "nonneg_l1"
    lam : float, the L1 weight
    L : float, optional, Lipschitz constant of A^T W A; by default 1.1 x a power iteration in Gram form
    tau : float, optional, the step (overrides L)
    support : "fov" (default), None or an (Ny, Nx) bool array, as FISTAHuber
    build_gb : float, GPU memory of each of the two pole-figure chunks used to build G and B (default 1)
    """

    def __init__(self, operator, prox_kind="nonneg", lam=0.0, L=None, tau=None, support="fov", build_gb=1.0):
        op = operator
        if getattr(op, "coefficient_layout", "dense") != "dense":
            raise ValueError("FISTAGram needs an operator with dense coefficients (K, Ny, Nx)")
        if getattr(op, "projector", None) not in ("native", "fft"):
            raise ValueError("FISTAGram supports the native and the FFT projector")
        if prox_kind not in ("nonneg", "l1", "nonneg_l1"):
            raise ValueError(f"prox_kind {prox_kind!r}: FISTAGram supports 'nonneg', 'l1' and 'nonneg_l1'")
        self.op, self.queue, self.ctx = op, op.queue, op.ctx
        self.prox_kind, self.lam = prox_kind, float(lam)
        self.build_gb = float(build_gb)
        self.K, self.R, self.My, self.CP = int(op.K), int(op.N_Omega), int(op.My), int(op.N_seg)
        self.Kp = -(-self.K // 4) * 4  # the native projector handles 4 orientations per work item
        self.alpha = np.float32(self.R / np.pi)  # the angle weight of the operator's adjoint (as in _adjoint_batch)
        self.prog = cl.Program(self.ctx, _KERNELS).build()
        self.k_scale = cl.Kernel(self.prog, "scale_periodic")
        self.k_mirror = cl.Kernel(self.prog, "mirror_upper")
        fista = build_fista_program(self.ctx)
        self.k_grad_step = cl.Kernel(fista, "grad_step")
        self.k_extrapolate = cl.Kernel(fista, "extrapolate")
        self.k_copy = cl.Kernel(fista, "copy_buf")
        self.prox_kernels = ProxKernels(self.ctx)
        self.support_gpu = support_mask_to_gpu(self.queue, op, support)
        self._L, self._tau = L, tau
        self.G = None
        self.iter_stats = []

    # ------------------------------------------------------------------ building G and B
    def _eval_pf_chunk(self, buf, k0, Kb, Kc):
        """The dense PF matrix of orientations k0 .. k0+Kb-1 into buf (R, Kc, N_eta, N_peaks), zero rows after Kb:
        the operator's own evaluation kernel (as SinglePhaseForwardOperator._eval_pf_batch) with this chunk size."""
        op = self.op
        R, C, P = self.R, int(op.N_eta), int(op.N_peaks)
        op.pfpoles_kernel(
            self.queue, (R * Kc * C * P,), None,
            op.coords_gpu.data, op.poles_gpu.data, op.axis_start_gpu.data, op.axis_count_gpu.data,
            op.inv_sigma2_gpu.data, op.norm_factor_gpu.data, op.quad_weights_gpu.data, buf.data,
            np.int32(R), np.int32(Kc), np.int32(C), np.int32(P), np.int32(op.N_axes),
            np.int32(op.N_Omega_subdivisions), np.int32(op.N_eta_subdivisions), np.int32(k0), np.int32(Kb))
        if not op.normalized:
            op._scale_pf_by_intensity_inplace(buf, op.intens_gpu)

    def prepare(self, b, weights=None):
        """Form G (R, Kp, Kp), -B (R, My, Kp) and the constant 0.5 ||w b||^2 (G, B times the adjoint's angle weight).

        b : the data (N_Omega, My, N_seg), NumPy or a C-contiguous float32 pyopencl array
        weights : None, or one weight per segment (N_seg values)
        """
        q, op = self.queue, self.op
        R, My, CP, K, Kp = self.R, self.My, self.CP, self.K, self.Kp
        t0 = time.perf_counter()
        W = _segment_weights(op, weights)
        if isinstance(b, clarray.Array):
            check_device(b, op.data_shape, "data")
            bw = b.copy()
        else:
            b_host = as_host(b, op.data_shape, "data")
            bw = clarray.to_device(q, b_host)
        W_gpu = None if W is None else clarray.to_device(q, W)
        if W_gpu is not None:
            gws, n = elementwise(bw.size)
            self.k_scale(q, gws, None, bw.data, W_gpu.data, np.uint64(CP), n)  # bw = W b
        # the constant 0.5 ||w b||^2 = 0.5 <b, W b>
        if isinstance(b, clarray.Array):
            self.c0 = 0.5 * float(clarray.dot(b, bw, dtype=np.float64).get())
        else:
            b2 = np.square(b_host.reshape(-1, CP), dtype=np.float64).sum(axis=0)
            self.c0 = 0.5 * float(b2.sum() if W is None else (b2 * W).sum())

        per_k = R * CP * 4
        Kc = max(4, int(self.build_gb * 1024**3 // per_k) // 4 * 4)
        Kc = min(Kc, Kp)
        if R * Kc * CP >= 2**31:
            raise ValueError("pole-figure chunk too large for 32-bit indexing; lower build_gb")
        bufA = clarray.empty(q, (R, Kc, int(op.N_eta), int(op.N_peaks)), np.float32)
        bufB = clarray.empty(q, (R, Kc, int(op.N_eta), int(op.N_peaks)), np.float32)
        self.G = clarray.zeros(q, (R, Kp, Kp), np.float32)
        Bm = clarray.zeros(q, (R, My, Kp), np.float32)
        chunks = [(k0, min(Kc, K - k0)) for k0 in range(0, K, Kc)]
        A2 = bufA.reshape((R * Kc, CP))
        B2 = bufB.reshape((R * Kc, CP))
        G2 = self.G.reshape((R * Kp, Kp))
        bw2 = bw.reshape((R * My, CP))
        Bm2 = Bm.reshape((R * My, Kp))
        for ia, (ka, na) in enumerate(chunks):
            self._eval_pf_chunk(bufA, ka, na, Kc)
            # B[r, :, ka:ka+na] = (W b)_r P_a^T
            gemmStridedBatched(q, My, na, CP, R, bw2, A2, Bm2, CP, CP, Kp, My * CP, Kc * CP, My * Kp,
                               alpha=float(self.alpha), beta=0.0, b_transp=True, c_offset=ka)
            if W_gpu is not None:  # bufA <- P_a W
                gws, n = elementwise(bufA.size)
                self.k_scale(q, gws, None, bufA.data, W_gpu.data, np.uint64(CP), n)
            for kb, nb in chunks[ia:]:
                self._eval_pf_chunk(bufB, kb, nb, Kc)
                # G[r, ka:ka+na, kb:kb+nb] = alpha (P_a W) P_b^T
                gemmStridedBatched(q, na, nb, CP, R, A2, B2, G2, CP, CP, Kp, Kc * CP, Kc * CP, Kp * Kp,
                                   alpha=float(self.alpha), beta=0.0, b_transp=True, c_offset=ka * Kp + kb)
        gws, n = elementwise(R * K * K)
        self.k_mirror(q, gws, None, self.G.data, np.int32(Kp), np.int32(K), n)
        self.negB = -Bm
        del bufA, bufB, bw, Bm
        # work arrays of the iterations
        self.S = clarray.empty(q, (R, My, Kp), np.float32)
        self.T = clarray.empty(q, (R, My, Kp), np.float32)
        self.img = clarray.empty(q, (op.Nx * op.Ny * Kp,), np.float32) if op.projector == "native" else None
        q.finish()
        self.prepare_seconds = time.perf_counter() - t0
        self.gram_bytes = int(self.G.nbytes)

    # ------------------------------------------------------------------ the operator in Gram form
    def _project(self, x):
        """S = the projections of the K images of x (R, My, Kp), padding channels zero."""
        op = self.op
        if op.projector == "native":
            op.radon.gather(x, self.img, 0, self.K, self.Kp)
            op.radon.forward(self.img, self.S, self.Kp)
        else:
            op.fradon.forward(x, 0, self.K, self.S, self.Kp)

    def _backproject(self, src, out):
        op = self.op
        if op.projector == "native":
            op.radon.backward(src, self.img, self.Kp)
            op.radon.scatter(self.img, out, 0, self.K, self.Kp)
        else:
            op.fradon.backward(src, self.Kp, out, 0, self.K)

    def _times_G(self, beta):
        """T = S G + beta T, per rotation."""
        R, My, Kp = self.R, self.My, self.Kp
        gemmStridedBatched(self.queue, My, Kp, Kp, R, self.S.reshape((R * My, Kp)), self.G.reshape((R * Kp, Kp)),
                           self.T.reshape((R * My, Kp)), Kp, Kp, Kp, My * Kp, Kp * Kp, My * Kp,
                           alpha=1.0, beta=float(beta))

    def gradient(self, x, grad):
        """grad = A^T W (A x - b) (C-contiguous float32 pyopencl arrays (K, Ny, Nx)); returns
        0.5 ||w (A x - b)||^2."""
        q = self.queue
        self._project(x)
        cl.enqueue_copy(q, self.T.data, self.negB.data)
        self._times_G(1.0)  # T = alpha (S G - B)
        # f = 0.5 <S, S G> - <S, B> + c0 = (0.5 <S, T> + 0.5 <S, -B>) / alpha + c0
        st = clarray.dot(self.S, self.T, dtype=np.float64)
        sb = clarray.dot(self.S, self.negB, dtype=np.float64)
        self._backproject(self.T, grad)
        return 0.5 * (float(st.get()) + float(sb.get())) / float(self.alpha) + self.c0

    def normal(self, v, out):
        """out = A^T W A v."""
        self._project(v)
        self._times_G(0.0)
        self._backproject(self.T, out)

    def estimate_L(self, niter=20, seed=0):
        """The largest eigenvalue of A^T W A by power iteration (in Gram form, no data-space work)."""
        q, op = self.queue, self.op
        rng = np.random.default_rng(seed)
        v = clarray.to_device(q, rng.random(op.coeff_shape).astype(np.float32))
        if self.support_gpu is not None:
            apply_support(q, self.prox_kernels, v, self.support_gpu)
        w = clarray.empty_like(v)
        lam = 0.0
        for _ in range(niter):
            nv = float(clarray.vdot(v, v).get()) ** 0.5
            v *= np.float32(1.0 / max(nv, 1e-30))
            self.normal(v, w)
            lam = float(clarray.vdot(v, w).get())
            v, w = w, v
        return lam

    # ------------------------------------------------------------------ FISTA
    def _prox(self, x):
        q = self.queue
        if self.prox_kind == "nonneg":
            prox_nonneg(q, self.prox_kernels, x)
        elif self.prox_kind == "l1":
            prox_l1(q, self.prox_kernels, x, self.lam, self.tau)
        else:
            prox_nonneg_l1(q, self.prox_kernels, x, self.lam, self.tau)
        if self.support_gpu is not None:
            apply_support(q, self.prox_kernels, x, self.support_gpu)

    def run(self, x0, b=None, niter=100, weights=None, verbose=0):
        """FISTA from x0 (NumPy, copied to the GPU and returned as NumPy, or a C-contiguous float32 pyopencl
        array updated in place). b / weights: the data and per-segment weights (needed on the first call;
        prepare() can also be called beforehand). iter_stats holds f (at the extrapolated point, as
        FISTAHuber), the objective and the time per iteration."""
        q, op = self.queue, self.op
        if b is not None:
            self.prepare(b, weights)
        if self.G is None:
            raise ValueError("call prepare(b, weights) or pass the data to run()")
        host = not isinstance(x0, clarray.Array)
        x = clarray.to_device(q, as_host(x0, op.coeff_shape, "coeffs")) if host else x0
        check_device(x, op.coeff_shape, "coeffs")
        if self._tau is not None:
            self.tau = float(self._tau)
        else:
            L = self._L if self._L is not None else 1.1 * self.estimate_L()
            self.L, self.tau = float(L), 1.0 / float(L)
        if self.support_gpu is not None:
            apply_support(q, self.prox_kernels, x, self.support_gpu)
        y, x_old, grad = x.copy(), x.copy(), clarray.empty_like(x)
        gws, n = elementwise(x.size)
        t = 1.0
        self.iter_stats = []
        t_start = time.perf_counter()
        for it in range(int(niter)):
            fval = self.gradient(y, grad)
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            beta = (t - 1.0) / t_new
            self.k_grad_step(q, gws, None, y.data, grad.data, x.data, np.float32(self.tau), n)
            self._prox(x)
            self.k_extrapolate(q, gws, None, x.data, x_old.data, y.data, np.float32(beta), n)
            self.k_copy(q, gws, None, x.data, x_old.data, n)
            t = t_new
            gval = 0.0
            if self.prox_kind in ("l1", "nonneg_l1") and self.lam:
                gval = self.lam * float(clarray.sum(abs(x)).get())
            self.iter_stats.append({"iter": it + 1, "f": fval, "g": gval, "obj": fval + gval, "beta": beta,
                                    "tau": self.tau, "time": time.perf_counter() - t_start})
            if verbose and ((it + 1) % max(1, int(verbose)) == 0 or it == 0 or it == niter - 1):
                print(f"[gram {it + 1:4d}/{niter}] f={fval:.6e}  obj={fval + gval:.6e}  beta={beta:.3f}")
        q.finish()
        if host:
            out = x.get()
            if isinstance(x0, np.ndarray) and x0.dtype == np.float32 and x0.flags.c_contiguous:
                x0[...] = out
                return x0
            return out
        return x

    def free(self):
        """Release the Gram matrices and work arrays."""
        for name in ("G", "negB", "S", "T", "img"):
            a = getattr(self, name, None)
            if a is not None:
                a.base_data.release()
            setattr(self, name, None)
