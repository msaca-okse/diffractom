from __future__ import annotations
import numpy as np
import gc
import pyopencl as cl
import pyopencl.array as clarray
import gratopy
from pyclblast import gemmStridedBatched
from .pf_kernels import build_pfo_program
from .parallel_radon import ParallelRadon
from ..utils.arrays import as_host, check_device
from ..utils.support import fov_support_mask


def gratopy_angle_weights(queue, angles, Nx, Ny, My):
    """The angle weights gratopy uses for these parallel-beam angles (half the angular gap to the
    neighbours, modulo pi), so that the native projector's backward projection equals gratopy's."""
    ps = gratopy.ProjectionSettings(queue, gratopy.PARALLEL, (Nx, Ny), np.asarray(angles, dtype=np.float64),
                                    n_detectors=My)
    return np.asarray(ps.angle_weights, dtype=np.float64)


class MatrixTomographicOperator:
    """Generic matrix-tomographic forward operator on GPU.

    Computes  Ax = B @ P(x)  and its adjoint, where P is the parallel-beam tomographic projection and
    B is a user-supplied matrix of shape (N_Omega, K, N_seg): for every projection angle, a map from
    the K channels (basis functions) to the N_seg data values of a detector position.

    Arrays: coefficients (K, Ny, Nx), C order (coeffs[k] is the image of channel k); data
    (N_Omega, My, N_seg), C order. direct and adjoint take and return NumPy arrays, or pyopencl arrays;
    the FISTA solvers accept NumPy arrays (uploaded; the result is copied back) or pyopencl arrays.
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
        image_width_factor: float = 1.0,
        verbose: bool = False,
        ctx: cl.Context | None = None,
        queue: cl.CommandQueue | None = None,
        projector: str = "gratopy",
        angle_weights=None,
    ):
        """Initialise the matrix-tomographic operator.

        Parameters
        ----------
        B : ndarray, shape (N_Omega, K, N_seg)
            User-supplied matrix applied after tomographic projection.
        angles : ndarray, shape (N_Omega,)
            Projection angles in radians (the convention of gratopy and ParallelRadon: at angle a the
            projection integrates along (cos a, sin a) in (column, row) image coordinates).
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
            Centre-of-rotation offset in detector pixels (default 0).
        image_width_factor : float
            Width of the image in detector pixels, relative to Nx (default 1: pixel = detector pixel).
        verbose : bool
            Print the buffer allocation summary.
        ctx, queue : optional
            Existing OpenCL context/queue; created automatically if None.
        projector : "gratopy" (default) or "native"
            The tomographic projector: gratopy, or diffractom's ParallelRadon (the same
            discretisation, results equal to 2e-7; faster for large images and many angles, slower
            for small ones: 123 x 123 pixels, 24 angles, K = 9400 on a V100: 62 vs 41 ms per forward).
        angle_weights : None (default), "uniform" or an array of N_Omega weights
            Weights of the angles in the backward projection, i.e. in the adjoint (the solvers then
            minimise the angle-weighted residual). None: gratopy's (half the angular gap to the
            neighbouring angles), the same as with projector="gratopy". "uniform": pi / N_Omega each,
            so that the backward projection is the adjoint up to one constant.
        """
        B = np.asarray(B)
        if B.shape != (N_Omega, K, N_seg):
            raise ValueError(f"B must have shape ({N_Omega}, {K}, {N_seg}), got {B.shape}")
        angles = np.asarray(angles, dtype=np.float64)
        if angles.shape != (N_Omega,):
            raise ValueError(f"angles must have shape ({N_Omega},), got {angles.shape}")
        if projector not in ("native", "gratopy"):
            raise ValueError(f"projector must be 'native' or 'gratopy', not {projector!r}")

        self.N_Omega = N_Omega
        self.My = My
        self.Nx = Nx
        self.Ny = Ny
        self.K = K
        self.N_seg = N_seg
        self.cor_offset = cor_offset
        self.verbose = verbose
        self.image_width_factor = image_width_factor
        self.projector = projector
        self.angles = angles

        # --- context / queue ---
        if ctx is not None and queue is not None:
            self.ctx = ctx
            self.queue = queue
        else:
            self.ctx = cl.create_some_context(interactive=False)
            self.queue = cl.CommandQueue(self.ctx)

        if isinstance(angle_weights, str):
            if angle_weights != "uniform":
                raise ValueError("angle_weights must be None, 'uniform' or an array")
            w = np.full(N_Omega, np.pi / N_Omega)
        elif angle_weights is None:
            w = gratopy_angle_weights(self.queue, angles, Nx, Ny, My)
        else:
            w = np.broadcast_to(np.asarray(angle_weights, dtype=np.float64), (N_Omega,)).copy()
        self.angle_weights = w

        image_width = self.Nx * self.image_width_factor
        if projector == "native":
            # channels fastest, padded to a multiple of 4 (zero rows of B for the padding)
            self.Kstride = -(-K // 4) * 4
            self.radon = ParallelRadon(self.queue, (Nx, Ny), angles, My, image_width=image_width,
                                       detector_width=My, detector_shift=cor_offset, angle_weights=w)
            self.PS = None
        else:
            self.Kstride = K
            self.radon = None
            self.PS = gratopy.ProjectionSettings(
                self.queue, gratopy.PARALLEL, (self.Nx, self.Ny, self.K), angles, angle_weights=w,
                n_detectors=self.My, image_width=image_width, detector_width=self.My,
                detector_shift=self.cor_offset)
            self.prg = build_pfo_program(self.ctx)
            self.transpose_f_to_c = cl.Kernel(self.prg, "transpose_d_omega_k_f_to_c")
            self.transpose_c_to_f = cl.Kernel(self.prg, "transpose_omega_d_k_c_to_d_omega_k_f")

        # --- B and B^T on the GPU (rows K .. Kstride-1 zero) ---
        Bp = np.zeros((N_Omega, self.Kstride, N_seg), dtype=np.float32)
        Bp[:, :K] = B
        self.B_cpu = np.ascontiguousarray(B, dtype=np.float32)
        self.B_gpu = clarray.to_device(self.queue, Bp)
        self.BT_gpu = clarray.to_device(self.queue, np.ascontiguousarray(Bp.transpose(0, 2, 1)))

        self.allocate_buffers()

    def allocate_buffers(self):
        """Pre-allocate reusable GPU buffers for forward/adjoint computation."""
        buffers = []

        def _alloc(name, shape, dtype, order="C"):
            arr = clarray.empty(self.queue, shape, dtype=dtype, order=order)
            setattr(self, name, arr)
            buffers.append((name, arr))
            return arr

        # sinogram, channels fastest: (N_Omega, My, Kstride), C order
        _alloc("sino_C", (self.N_Omega, self.My, self.Kstride), np.float32)
        if self.projector == "native":
            # images, channels fastest: (Ny * Nx, Kstride)
            _alloc("img_k", (self.Nx * self.Ny * self.Kstride,), np.float32)
        else:
            # gratopy's sinogram, F order: (My, N_Omega, K)
            _alloc("sino_F", (self.My, self.N_Omega, self.K), np.float32, "F")

        total_bytes = sum(a.nbytes for _, a in buffers) + self.B_gpu.nbytes + self.BT_gpu.nbytes
        self.total_bytes = total_bytes
        if self.verbose:
            print("\n=== OpenCL buffer allocation summary ===")
            for name, arr in buffers + [("B_gpu", self.B_gpu), ("BT_gpu", self.BT_gpu)]:
                shape_str = "x".join(str(s) for s in arr.shape)
                print(f"  {name:30s}: shape=({shape_str}), {arr.nbytes / 1024**2:8.2f} MB")
            print("---------------------------------------")
            print(f"  TOTAL GPU buffer memory: {total_bytes / 1024**2:8.2f} MB")
            print("=======================================\n")

    @property
    def coeff_shape(self):
        """Shape of a coefficient array, (K, Ny, Nx): coeffs[k] is the image of channel k."""
        return (self.K, self.Ny, self.Nx)

    @property
    def data_shape(self):
        return (self.N_Omega, self.My, self.N_seg)

    def support_mask(self):
        """(Ny, Nx) bool mask of the pixels inside the field of view at every projection angle."""
        return fov_support_mask(self.Nx, self.Ny, self.My, angles=self.angles,
                                image_width=self.Nx * self.image_width_factor, detector_width=self.My,
                                detector_shift=self.cor_offset)

    def device_bytes(self):
        return int(self.total_bytes)

    def direct(self, coeffs):
        """Forward operator: data = B @ P(coeffs).

        coeffs : NumPy array (K, Ny, Nx) -> NumPy array (N_Omega, My, N_seg); or a C-contiguous
            float32 pyopencl array -> pyopencl array.
        """
        host = not isinstance(coeffs, clarray.Array)
        x = clarray.to_device(self.queue, as_host(coeffs, self.coeff_shape, "coeffs")) if host else coeffs
        data = clarray.empty(self.queue, self.data_shape, dtype=np.float32)
        self.direct_cl(x, data)
        if not host:
            return data
        out = data.get()
        for a in (x, data):
            a.base_data.release()
        return out

    def direct_cl(self, coeffs, data):
        """In-place forward: data = B @ P(coeffs).

        Parameters
        ----------
        coeffs : clarray, shape (K, Ny, Nx), C-contiguous float32
        data   : clarray, shape (N_Omega, My, N_seg), C-contiguous float32 — overwritten.
        """
        check_device(coeffs, self.coeff_shape, "coeffs")
        check_device(data, self.data_shape, "data")

        # 1) tomographic projection -> sino_C (N_Omega, My, Kstride)
        if self.projector == "native":
            self.radon.gather(coeffs, self.img_k, 0, self.K, self.Kstride)
            self.radon.forward(self.img_k, self.sino_C, self.Kstride)
        else:
            # (Nx, Ny, K) F (the same memory as (K, Ny, Nx) C) -> (My, N_Omega, K) F -> (N_Omega, My, K) C
            gratopy.forwardprojection(coeffs.transpose((2, 1, 0)), self.PS, sino=self.sino_F)
            total = self.N_Omega * self.My * self.K
            self.transpose_f_to_c(self.queue, (total,), None, self.sino_F.data, self.sino_C.data,
                                  np.int32(self.My), np.int32(self.N_Omega), np.int32(self.K), np.int32(total))

        # 2) batched GEMM: data[o] = sino_C[o] @ B[o], (My, Kstride) @ (Kstride, N_seg)
        _batched_gemm(self.queue, self.sino_C, self.B_gpu, data, R=self.N_Omega, M=self.My,
                      K_inner=self.Kstride, N=self.N_seg, alpha=1.0, beta=0.0)

    def adjoint(self, data):
        """Adjoint operator: coeffs = P^T(B^T @ data).

        data : NumPy array (N_Omega, My, N_seg) -> NumPy array (K, Ny, Nx); or a C-contiguous
            float32 pyopencl array -> pyopencl array.
        """
        host = not isinstance(data, clarray.Array)
        d = clarray.to_device(self.queue, as_host(data, self.data_shape, "data")) if host else data
        coeffs = clarray.empty(self.queue, self.coeff_shape, dtype=np.float32)
        self.adjoint_cl(d, coeffs)
        if not host:
            return coeffs
        out = coeffs.get()
        for a in (d, coeffs):
            a.base_data.release()
        return out

    def adjoint_cl(self, data, coeffs):
        """In-place adjoint: coeffs = P^T(B^T @ data).

        Parameters
        ----------
        data   : clarray, shape (N_Omega, My, N_seg), C-order
        coeffs : clarray, shape (K, Ny, Nx), C-contiguous float32 — overwritten.
        """
        check_device(data, self.data_shape, "data")
        check_device(coeffs, self.coeff_shape, "coeffs")

        # 1) batched GEMM: sino_C[o] = data[o] @ BT[o], (My, N_seg) @ (N_seg, Kstride)
        _batched_gemm(self.queue, data, self.BT_gpu, self.sino_C, R=self.N_Omega, M=self.My,
                      K_inner=self.N_seg, N=self.Kstride, alpha=1.0, beta=0.0)

        # 2) backprojection
        if self.projector == "native":
            self.radon.backward(self.sino_C, self.img_k, self.Kstride)
            self.radon.scatter(self.img_k, coeffs, 0, self.K, self.Kstride)
        else:
            total = self.N_Omega * self.My * self.K
            self.transpose_c_to_f(self.queue, (total,), None, self.sino_C.data, self.sino_F.data,
                                  np.int32(self.N_Omega), np.int32(self.My), np.int32(self.K), np.int32(total))
            # (My, N_Omega, K) F -> (Nx, Ny, K) F, the same memory as (K, Ny, Nx) C
            gratopy.backprojection(self.sino_F, self.PS, img=coeffs.transpose((2, 1, 0)))

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

        x = clarray.empty(q, self.coeff_shape, np.float32)
        Ax = clarray.empty(q, (self.N_Omega, self.My, self.N_seg), np.float32, order="C")
        z = clarray.empty(q, x.shape, np.float32)

        # drawn as (Nx, Ny, K), pixel fastest: the same start as before the (K, Ny, Nx) layout
        x_host = np.ascontiguousarray(rng.standard_normal((self.Nx, self.Ny, self.K)).astype(np.float32).transpose(2, 1, 0))
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
            z *= np.float32(1.0 / znorm)  # in place: no third coefficient-sized array
            x, z = z, x
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
        for name in ("sino_F", "sino_C", "img_k", "B_gpu", "BT_gpu"):
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
