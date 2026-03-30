from __future__ import annotations
import numpy as np
import gc
import pyopencl as cl
import pyopencl.array as clarray
import gratopy
from pyclblast import gemmStridedBatched
from .pf_kernels import build_pfo_program


class MatrixTomographicOperator:
    """Generic matrix-tomographic forward operator on GPU.

    Computes  Ax = B @ P(x)  and its adjoint, where P is parallel-beam
    tomographic projection and B is a user-supplied matrix of shape
    (N_Omega, K, N_seg).
    """

    def __init__(
        self,
        B: np.ndarray,
        angles: np.ndarray,
        N_Omega: int,
        My: int,
        Nx: int,
        Ny: int,
        K: int,
        N_seg: int,
        cor_offset: float = 0.0,
        verbose: bool = False,
        ctx: cl.Context | None = None,
        queue: cl.CommandQueue | None = None,
    ):
        """Initialise the matrix-tomographic operator.

        Parameters
        ----------
        B : ndarray, shape (N_Omega, K, N_seg)
            User-supplied matrix applied after tomographic projection.
        angles : ndarray, shape (N_Omega,)
            Projection angles in radians.
        N_Omega : int
            Number of projection angles.
        My : int
            Number of detector pixels.
        Nx, Ny : int
            Spatial image dimensions.
        K : int
            Number of basis functions.
        N_seg : int
            Number of data points per rotation/translation step.
        cor_offset : float
            Centre-of-rotation offset (default 0).
        verbose : bool
            Print buffer allocation summary.
        ctx, queue : optional
            Existing OpenCL context/queue; created automatically if None.
        """
        assert B.shape == (N_Omega, K, N_seg), (
            f"B must have shape ({N_Omega}, {K}, {N_seg}), got {B.shape}"
        )
        assert angles.shape == (N_Omega,), (
            f"angles must have shape ({N_Omega},), got {angles.shape}"
        )

        self.N_Omega = N_Omega
        self.My = My
        self.Nx = Nx
        self.Ny = Ny
        self.K = K
        self.N_seg = N_seg
        self.cor_offset = cor_offset
        self.verbose = verbose

        # --- context / queue ---
        if ctx is not None and queue is not None:
            self.ctx = ctx
            self.queue = queue
        else:
            self.ctx = cl.create_some_context(interactive=False)
            self.queue = cl.CommandQueue(self.ctx)

        # --- build transpose kernels ---
        self.prg = build_pfo_program(self.ctx)
        self.transpose_f_to_c = cl.Kernel(self.prg, "transpose_d_omega_k_f_to_c")
        self.transpose_c_to_f = cl.Kernel(self.prg, "transpose_omega_d_k_c_to_d_omega_k_f")

        # --- angles ---
        self.angles = np.asarray(angles, dtype=np.float64)

        # --- projection settings ---
        self.PS = gratopy.ProjectionSettings(
            self.queue,
            gratopy.PARALLEL,
            (self.Nx, self.Ny, self.K),
            self.angles,
            n_detectors=self.My,
            image_width=self.Nx,
            detector_width=self.My,
            detector_shift=self.cor_offset,
        )

        # --- transfer B and B^T to GPU ---
        self.B_cpu = np.asarray(B, dtype=np.float32, order="C")
        self.B_gpu = clarray.to_device(self.queue, self.B_cpu)

        self.BT_cpu = np.ascontiguousarray(self.B_cpu.transpose(0, 2, 1))
        self.BT_gpu = clarray.to_device(self.queue, self.BT_cpu)

        # --- allocate working buffers ---
        self.allocate_buffers()

    def allocate_buffers(self):
        """Pre-allocate reusable GPU buffers for forward/adjoint computation."""
        buffers = []

        def _alloc(name, shape, dtype, order):
            arr = clarray.empty(self.queue, shape, dtype=dtype, order=order)
            buffers.append((name, arr))
            return arr

        # Sinogram F-order from gratopy: (My, N_Omega, K)
        self.sino_F = _alloc(
            "sino_F", (self.My, self.N_Omega, self.K), np.float32, "F"
        )

        # Sinogram C-order after transpose: (N_Omega, My, K)
        self.sino_C = _alloc(
            "sino_C", (self.N_Omega, self.My, self.K), np.float32, "C"
        )

        # --- memory summary ---
        print("\n=== OpenCL buffer allocation summary ===")
        total_bytes = 0

        for name, arr in buffers:
            nbytes = arr.size * arr.dtype.itemsize
            total_bytes += nbytes
            shape_str = "x".join(str(s) for s in arr.shape)
            print(f"  {name:30s}: shape=({shape_str}), {nbytes / 1024**2:8.2f} MB")

        b_bytes = self.B_gpu.size * self.B_gpu.dtype.itemsize
        bt_bytes = self.BT_gpu.size * self.BT_gpu.dtype.itemsize
        total_bytes += b_bytes + bt_bytes

        b_shape = "x".join(str(s) for s in self.B_gpu.shape)
        bt_shape = "x".join(str(s) for s in self.BT_gpu.shape)
        print(f"  {'B_gpu':30s}: shape=({b_shape}), {b_bytes / 1024**2:8.2f} MB")
        print(f"  {'BT_gpu':30s}: shape=({bt_shape}), {bt_bytes / 1024**2:8.2f} MB")

        print("---------------------------------------")
        print(f"  TOTAL GPU buffer memory: {total_bytes / 1024**2:8.2f} MB")
        print("=======================================\n")
        self.total_bytes = total_bytes

    def direct(self, coeffs):
        """Forward operator: data = B @ P(coeffs).

        Parameters
        ----------
        coeffs : clarray, shape (Nx, Ny, K), F-order

        Returns
        -------
        data : clarray, shape (N_Omega, My, N_seg), C-order
        """
        data = clarray.zeros(
            self.queue,
            (self.N_Omega, self.My, self.N_seg),
            dtype=np.float32,
            order="C",
        )
        self.direct_cl(coeffs, data)
        return data

    def direct_cl(self, coeffs, data):
        """In-place forward: data = B @ P(coeffs).

        Parameters
        ----------
        coeffs : clarray, shape (Nx, Ny, K), F-order
        data   : clarray, shape (N_Omega, My, N_seg), C-order — overwritten.
        """
        self.sino_F.fill(0.0)
        self.sino_C.fill(0.0)

        # 1) Tomographic projection: (Nx, Ny, K) F -> (My, N_Omega, K) F
        gratopy.forwardprojection(coeffs, self.PS, sino=self.sino_F)

        # 2) Transpose: (My, N_Omega, K) F -> (N_Omega, My, K) C
        total = self.N_Omega * self.My * self.K
        self.transpose_f_to_c(
            self.queue,
            (total,),
            None,
            self.sino_F.data,
            self.sino_C.data,
            np.int32(self.My),
            np.int32(self.N_Omega),
            np.int32(self.K),
            np.int32(total),
        )

        # 3) Batched GEMM: data[o] = sino_C[o] @ B[o]
        #    sino_C[o]: (My, K),  B[o]: (K, N_seg)  ->  data[o]: (My, N_seg)
        _batched_gemm(
            self.queue,
            self.sino_C,
            self.B_gpu,
            data,
            R=self.N_Omega,
            M=self.My,
            K_inner=self.K,
            N=self.N_seg,
            alpha=1.0,
            beta=0.0,
        )

    def adjoint(self, data):
        """Adjoint operator: coeffs = P^T(B^T @ data).

        Parameters
        ----------
        data : clarray, shape (N_Omega, My, N_seg), C-order

        Returns
        -------
        coeffs : clarray, shape (Nx, Ny, K), F-order
        """
        coeffs = clarray.zeros(
            self.queue,
            (self.Nx, self.Ny, self.K),
            dtype=np.float32,
            order="F",
        )
        self.adjoint_cl(data, coeffs)
        return coeffs

    def adjoint_cl(self, data, coeffs):
        """In-place adjoint: coeffs = P^T(B^T @ data).

        Parameters
        ----------
        data   : clarray, shape (N_Omega, My, N_seg), C-order
        coeffs : clarray, shape (Nx, Ny, K), F-order — overwritten.
        """
        coeffs.fill(0.0)
        self.sino_C.fill(0.0)
        self.sino_F.fill(0.0)

        # 1) Batched GEMM: sino_C[o] = data[o] @ BT[o]
        #    data[o]: (My, N_seg),  BT[o]: (N_seg, K)  ->  sino_C[o]: (My, K)
        _batched_gemm(
            self.queue,
            data,
            self.BT_gpu,
            self.sino_C,
            R=self.N_Omega,
            M=self.My,
            K_inner=self.N_seg,
            N=self.K,
            alpha=1.0,
            beta=0.0,
        )

        # 2) Transpose: (N_Omega, My, K) C -> (My, N_Omega, K) F
        total = self.N_Omega * self.My * self.K
        self.transpose_c_to_f(
            self.queue,
            (total,),
            None,
            self.sino_C.data,
            self.sino_F.data,
            np.int32(self.N_Omega),
            np.int32(self.My),
            np.int32(self.K),
            np.int32(total),
        )

        # 3) Backprojection: (My, N_Omega, K) F -> (Nx, Ny, K) F
        gratopy.backprojection(self.sino_F, self.PS, img=coeffs)

    def estimate_L_power(self, niter=20, seed=0, eps=1e-30, verbose=1):
        """Estimate the Lipschitz constant L = ||A^T A|| via power iteration.

        Parameters
        ----------
        niter : int
            Number of power iterations.
        seed : int
            RNG seed for initial vector.
        eps : float
            Small constant to avoid division by zero.
        verbose : int
            Print per-iteration diagnostics.

        Returns
        -------
        L_est : float
        """
        if niter < 1:
            raise ValueError("niter must be >= 1")

        q = self.queue
        rng = np.random.default_rng(seed)

        x = clarray.empty(q, (self.Nx, self.Ny, self.K), np.float32, order="F")
        Ax = clarray.empty(q, (self.N_Omega, self.My, self.N_seg), np.float32, order="C")
        z = clarray.empty(q, x.shape, np.float32, order="F")

        x_host = rng.standard_normal(x.shape).astype(np.float32, copy=False, order="F")
        cl.enqueue_copy(q, x.data, x_host)
        q.finish()

        xnorm = float(np.sqrt(clarray.vdot(x, x).get()) + eps)
        x *= np.float32(1.0 / xnorm)
        q.finish()

        L_est = 0.0
        for it in range(niter):
            self.direct_cl(x, Ax)
            self.adjoint_cl(Ax, z)
            q.finish()

            num = float(clarray.vdot(x, z).get())
            den = float(clarray.vdot(x, x).get()) + eps
            L_est = num / den

            znorm = float(np.sqrt(clarray.vdot(z, z).get()) + eps)
            x[:] = z * np.float32(1.0 / znorm)
            q.finish()

            if verbose:
                print(f"[power {it + 1:02d}] L_est={L_est:.6e}  ||z||={znorm:.6e}")

        # --- GPU cleanup ---
        for arr in (x, Ax, z):
            if arr.base_data is not None:
                arr.base_data.release()
        del x, Ax, z, x_host
        gc.collect()
        q.finish()

        return float(L_est)

    def free_memory(self):
        """Release OpenCL buffers and remove corresponding attributes."""
        buffer_names = ["sino_F", "sino_C", "B_gpu", "BT_gpu"]

        for name in buffer_names:
            arr = getattr(self, name, None)
            if arr is None:
                continue
            try:
                base = getattr(arr, "base_data", None)
                if base is not None:
                    base.release()
            except Exception:
                pass
            try:
                delattr(self, name)
            except Exception:
                pass

        try:
            if hasattr(self, "queue") and self.queue is not None:
                self.queue.finish()
        except Exception:
            pass

        gc.collect()

        if self.verbose:
            print("OpenCL GPU memory freed.")


def _batched_gemm(queue, A3, B3, C3, R, M, K_inner, N, alpha=1.0, beta=0.0):
    """Batched GEMM via CLBlast: C[r] = alpha * A[r] @ B[r] + beta * C[r].

    Parameters
    ----------
    A3 : clarray, (R, M, K_inner), C-order
    B3 : clarray, (R, K_inner, N), C-order
    C3 : clarray, (R, M, N), C-order
    """
    A = A3.reshape((R * M, K_inner))
    B = B3.reshape((R * K_inner, N))
    C = C3.reshape((R * M, N))

    gemmStridedBatched(
        queue,
        M, N, K_inner,
        R,
        A, B, C,
        K_inner, N, N,
        M * K_inner, K_inner * N, M * N,
        alpha=alpha,
        beta=beta,
    )
