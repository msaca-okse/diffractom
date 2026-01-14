import numpy as np
import time
import gc




import pyopencl as cl
import pyopencl.array as clarray
import gratopy
from pyclblast import gemmStridedBatched


from scipy.spatial.transform import Rotation as R
from package.odf_mumott.pole_figure_geometry import GeometryContainerM

from package.utils.coordinates import get_probed_coordinates

from package.texture_tomography.operators.create_pfo_matrix import (
    pfmatrix_eval_gpu,
    build_pf_program
)
from package.texture_tomography.operators.pfo_kernels import build_all_opencl




class PFO_SINGLE:

    def __init__(
        self,
        cfg: None,
        material: None,
        grid: None,
        max_gb: None,
        verbose: bool = False,
        normalized: bool = False,
    ):
        
        self.cfg = cfg
        self.normalized = normalized
        self.material = material
        self.grid = grid
        self.verbose = verbose
        self.N_chi = self.cfg['N_chi']
        self.N_peaks = len(self.material.reflections)
        self.N_seg = self.N_peaks * self.N_chi
        self.Nx = self.cfg['Nx']
        self.Ny = self.cfg['Ny']
        self.N_rot = self.cfg['N_rot']
        self.angle_range = np.array(self.cfg['angle_range'])/180*np.pi
        self.angles = np.linspace(self.angle_range[0], self.angle_range[1], self.N_rot, endpoint=False)
        self.pf_batch_max_gb = float(max_gb)

        # --- context / queue ---
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)

        # --- build kernels ---
        self.prg, self.k, self.pf_prg = build_all_opencl(self.ctx, ts=16)
        self.pf_prg = build_pf_program(self.ctx)
        self.pfmatrix_eval_kernel = cl.Kernel(self.pf_prg, "pfmatrix_eval")


        self.transfer_material_parameters_to_gpu()
        self.detector_coordinates()
        self.transfer_grid_parameters_to_gpu()
        self.get_pf_batches_for_material()  # list of dicts



        #        --- 2) Create ProjectionSettings using THIS queue ---
        self.PS = gratopy.ProjectionSettings(
            self.queue,
            gratopy.PARALLEL,
            (self.Nx, self.Ny, self.K_batch_max),
            self.N_rot,
            n_detectors=self.Nx,
            image_width=self.Nx,
            detector_width=self.Nx,
            detector_shift=2,
            angle_range=self.angle_range
        )
        assert self.queue.context.int_ptr == self.ctx.int_ptr


        # Allocate buffers
        self.allocate_coefficient_buffer()


    def detector_coordinates(self):
        wavelength_angstrom = 12.398 / self.cfg["wavelength"]

        j0 = np.asarray(self.cfg["j_direction_0"])
        k0 = np.asarray(self.cfg["k_direction_0"])
        p0 = np.asarray(self.cfg["p_direction_0"])
        det_o = np.asarray(self.cfg["detector_direction_origin"])
        det_p90 = np.asarray(self.cfg["detector_direction_positive_90"])

        # ---------------- rotations ----------------
        projections = {
            str(i): {
                "rotation_matrix": R.from_rotvec(
                    angle * k0 / np.linalg.norm(k0)
                ).as_matrix()
            }
            for i, angle in enumerate(self.angles)
        }

        detector_angles = np.linspace(0, 2*np.pi, self.N_chi, endpoint=False)

        geom_dict = {
            "projections": projections,
            "p_direction_0": p0,
            "j_direction_0": j0,
            "k_direction_0": k0,
            "detector_direction_origin": det_o,
            "detector_direction_positive_90": det_p90,
            "detector_angles": detector_angles,
        }


        two_theta_peaks = 2.0 * np.arcsin(
            np.linalg.norm(self.h_cpu, axis=1) / (4.0 * np.pi) * wavelength_angstrom
        ).astype(np.float32)
        print(wavelength_angstrom)
        print(two_theta_peaks)

        # ---------------- probed coordinates ----------------
        coords_list = []
        for tt in two_theta_peaks:
            geom_dict["two_theta"] = np.array([tt])
            geom = GeometryContainerM(
                dictionary=geom_dict,
                data_type="dictionary"
            ).geometry
            coords = get_probed_coordinates(geom)[:, :, 0, :]
            coords_list.append(coords)

        coords_cpu = np.stack(coords_list, axis=-1)
        coords_cpu = coords_cpu.transpose((0, 1, 3, 2))
        self.coords_cpu = np.asarray(coords_cpu, dtype=np.float32, order="C")
        self.coords_gpu = clarray.to_device(self.queue, self.coords_cpu)




    def transfer_material_parameters_to_gpu(self):
        self.pf_sym_ops_gpu_list = []
        self.pf_sym_ops_cpu_list = []
        self.N_peaks_list = []

        self.h_cpu_normed = np.asarray(self.material.h_vecs_normed, dtype=np.float32, order="C")
        self.h_cpu = np.asarray(self.material.h_vecs, dtype=np.float32, order="C")
        self.h_gpu_normed = clarray.to_device(self.queue, self.h_cpu_normed)

        self.intens_cpu = np.asarray(self.material.intensities(), dtype=np.float32, order="C")
        self.intens_gpu = clarray.to_device(self.queue, self.intens_cpu)

        self.sym_ops_cpu = np.asarray(self.material.point_group_matrices, dtype=np.float32, order="C")
        self.sym_ops_gpu = clarray.to_device(self.queue, self.sym_ops_cpu)



    def transfer_grid_parameters_to_gpu(self):
        """
        Transfer active orientation grid parameters (rotations + sigmas)
        from OrientationTree objects to GPU.
        """

            # --- extract active leaf nodes ---
        active_indices = self.grid.active_leaf_nodes()
        nodes = [self.grid.nodes[i] for i in active_indices]
        self.K = len(nodes)

            # --- inverse rotations ---
        self.grid_inv_cpu = np.stack(
            [n.R.inv().as_matrix().reshape(-1) for n in nodes],
            axis=0
            ).astype(np.float32)

        self.grid_inv_gpu = clarray.to_device(self.queue, self.grid_inv_cpu)

            # --- sigma per node ---
        self.sigma_cpu = np.array(
            [node.sigma for node in nodes],
            dtype=np.float32
            )



    def allocate_coefficient_buffer(self):
        # ---- reusable transpose buffers (per material) ----

        buffers = []

        def _alloc(name, shape, dtype, order):
            arr = clarray.empty(self.queue, shape, dtype=dtype, order=order)
            buffers.append((name, arr))
            return arr

        self.coeffs_sino_F = _alloc(
            "coeffs_sino_F",
            (self.PS.n_detectors, self.PS.n_angles, self.K_batch_max),
            np.float32,
            "F",
        )

        self.coeffs_sino_C = _alloc(
            "coeffs_sino_C",
            (self.N_rot, self.Nx, self.K_batch_max),
            np.float32,
            "C",
        )

        self._coeffs_batch_F = _alloc(
            "_coeffs_batch_F",
            (self.Nx, self.Ny, self.K_batch_max),
            np.float32,
            "F",
        )

        self._basis_batch_kmax = _alloc(
            "_basis_batch_kmax",
            (self.N_rot, self.K_batch_max, self.N_chi, self.N_peaks),
            np.float32,
            "C",
        )

        self._basis_batch_transpose_kmax = _alloc(
            "_basis_batch_transpose_kmax",
            (self.N_rot, self.N_peaks * self.N_chi, self.K_batch_max),
            np.float32,
            "C",
        )

        self._grid_inv_kmax = _alloc(
            "_grid_inv_kmax",
            (self.K_batch_max, 9),
            np.float32,
            "C",
        )

        self._inv_sigma2_kmax = _alloc(
            "_inv_sigma2_kmax",
            (self.K_batch_max,),
            np.float32,
            "C",
        )

        self._norm_factor_kmax = _alloc(
            "_norm_factor_kmax",
            (self.K_batch_max,),
            np.float32,
            "C",
        )

        # --------------------------------------------------
        # Verbose memory breakdown
        # --------------------------------------------------
        if self.verbose:
            print("\n=== OpenCL buffer allocation summary ===")

            total_bytes = 0

            for name, arr in buffers:
                nbytes = arr.size * arr.dtype.itemsize
                total_bytes += nbytes

                shape_str = "x".join(str(s) for s in arr.shape)
                size_mb = nbytes / 1024**2

                print(f"{name:30s}: shape=({shape_str}), {size_mb:8.2f} MB")

            print("---------------------------------------")
            print(f"TOTAL GPU buffer memory: {total_bytes/1024**2:8.2f} MB")
            print("=======================================\n")
            self.total_bytes = total_bytes


    def free_memory(self):


        lst = getattr(self, "coeffs_sino_C", None)

        gpu_lists = [
            "coeffs_sino_C",
            "coords_gpu",
            "grid_inv_gpu",
            "sym_ops_gpu",
            "h_gpu_normed",
            "intens_gpu",
        ]

        for name in gpu_lists:
            lst = getattr(self, name, None)
            if lst is not None:
                del lst

        for name in [
            "coeffs_sino_F",
            "_x_full_gpu",
            "B_gpu",
        ]:
            if hasattr(self, name):
                delattr(self, name)

        gc.collect()




    def get_pf_batches_for_material(self):
        """
        Compute K-batching for PF-matrix generation for ONE material.

        Returns
        -------
        batches : list of dict
            Each dict contains:
                - k_start
                - k_end
                - K_batch
        """

        if not hasattr(self, "pf_batch_max_gb"):
            raise RuntimeError(
                "pf_batch_max_gb not set. Call set_pf_batch_max_gb(...) first."
            )

        # ---- dimensions ----
        R = self.N_rot
        C = self.N_chi
        T = self.N_peaks   # masked theta count
        K = self.K

        bytes_per_float = 4

        # Memory per K after convolution:
        # (R, C, T) per K
        bytes_per_K = R * C * T * bytes_per_float

        max_bytes = self.pf_batch_max_gb * (1024 ** 3)

        # At least one K per batch
        if bytes_per_K >0.01:
            K_batch_max = max(int(max_bytes // bytes_per_K), 1)
             # Safety: never exceed available K
            K_batch_max = min(K_batch_max, K)
        else:
            K_batch_max = 1



        batches = []

        k0 = 0
        while k0 < K:
            k1 = min(k0 + K_batch_max, K)
            batches.append({
                "k_start": k0,
                "k_end": k1,
                "K_batch": k1 - k0,
            })
            k0 = k1

        self.batches = batches
        self.K_batch_max = max(b["K_batch"] for b in self.batches)
    


    def direct(self, coeffs):
        """
        Allocating convenience wrapper for the OpenCL forward operator.

        Returns
        -------
        yin_gpu : clarray
            Shape (N_rot, Nx, N_seg), C order
        """

        data = clarray.zeros(
            self.queue,
            (self.N_rot, self.Nx, self.N_seg),
            dtype=np.float32,
            order="C",
        )

        self.direct_cl(coeffs, data)
        return data



    def direct_cl(self, coeffs, data):

        data.fill(0.0)

        R  = self.N_rot
        C = int(self.N_chi)
        P = int(self.N_peaks)
        G = int(len(self.sym_ops_cpu))
        Kmax = self.K_batch_max
        Nx = self.Nx
        Ny = self.Ny
        coords_gpu   = self.coords_gpu
        sym_ops_gpu  = self.sym_ops_gpu
        h_gpu_normed        = self.h_gpu_normed
        intensity_gpu = self.intens_gpu

        for b in self.batches:
            k0 = b["k_start"]
            k1 = b["k_end"]
            Kb = b["K_batch"]

            self._coeffs_batch_F.fill(0.0)
            self.coeffs_sino_C.fill(0.0)
            self.coeffs_sino_F.fill(0.0)


            total = Nx * Ny * Kb
            self.k.SLICE_COEFFS_K_BATCH_F(
                self.queue,
                (total,),
                None,
                coeffs.data,               # COEFFS_IN
                self._coeffs_batch_F.data,       # COEFFS_OUT
                np.int32(Nx),
                np.int32(Ny),
                np.int32(self.K),
                np.int32(Kb),
                np.int32(k0),
            )

            # -------------------------------------------------
            # 2) Tomographic projection (batched in K)
            # -------------------------------------------------

            gratopy.forwardprojection(
                self._coeffs_batch_F,
                self.PS,
                sino=self.coeffs_sino_F,
            )

            # -------------------------------------------------
            # 3) Transpose sino F → C (only Kb)
            # -------------------------------------------------

            total = self.N_rot * self.Nx * Kmax
            self.k.transpose_d_omega_k_f_to_c(
                self.queue,
                (total,),
                None,
                self.coeffs_sino_F.data,
                self.coeffs_sino_C.data,
                np.int32(self.Nx),
                np.int32(self.N_rot),
                np.int32(Kmax),
                np.int32(total),
            )

            # -------------------------------------------------
            # 4) PF + GEMM for this batch
            # -------------------------------------------------


            self._norm_factor_kmax.fill(0.0) # make sure that this is set to zero!
            self._inv_sigma2_kmax.fill(0.0)
            sigma = np.asarray(self.sigma_cpu[k0:k1], dtype=np.float32)
            inv_sigma2_cpu = (1.0 / (sigma * sigma)).astype(np.float32)
            norm_factor_cpu = (1.0 / (8.0 * np.pi * sigma * sigma)).astype(np.float32)
            cl.enqueue_copy(self.queue, self._inv_sigma2_kmax.data, inv_sigma2_cpu, device_offset=0)
            cl.enqueue_copy(self.queue, self._norm_factor_kmax.data, norm_factor_cpu, device_offset=0)

            total = Kb * 9
            self.k.SLICE_GRIDINV_K_BATCH(
                self.queue,
                (total,),
                None,
                self.grid_inv_gpu.data,      # input: (K_total, 9)
                self._grid_inv_kmax.data,    # output: (K_batch_max, 9)
                np.int32(self.K),            # K_IN  (total grid size)
                np.int32(Kb),                # K_OUT (this batch size)
                np.int32(k0),                # K_START
            )

            total = R * Kmax * C * P
            self.pfmatrix_eval_kernel(
                self.queue,
                (total,),
                None,
                coords_gpu.data,
                self._grid_inv_kmax.data,
                sym_ops_gpu.data,
                h_gpu_normed.data,
                self._inv_sigma2_kmax.data,
                self._norm_factor_kmax.data,
                self._basis_batch_kmax.data,
                np.int32(R),
                np.int32(Kmax),
                np.int32(C),
                np.int32(P),
                np.int32(G),
            )
            if not self.normalized:
                self._scale_pf_by_intensity_inplace(self._basis_batch_kmax, intensity_gpu)

            # OBS: batched_gemm_clblast overwrites _data_batch, but we need to accumulate. Hence this trick
            batched_gemm_clblast(self.queue, self.coeffs_sino_C, self._basis_batch_kmax.reshape((R, Kmax, C*P)), data, R=R, M=Nx, K=Kmax, N=P*C)



    def adjoint(self, data):
        """
        Allocating convenience wrapper for the OpenCL adjoint.

        Parameters
        ----------
        y_gpu : clarray
            Shape (N_rot, Nx, N_seg), C-order

        Returns
        -------
        x_gpu : clarray
            Shape (Nx, Nx, K_sum), Fortran-order
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
        """
        In-place OpenCL adjoint operator.

        Parameters
        ----------
        data : clarray
            Shape (N_rot, Nx, N_seg), C-order
        coeffs : clarray
            Shape (Nx, Nx, K_sum), Fortran-order
            Will be overwritten
        """

        # ---------------- checks ----------------
        assert data.shape == (self.N_rot, self.Nx, self.N_seg)
        assert data.flags.c_contiguous

        assert coeffs.shape == (self.Nx, self.Ny, self.K)
        assert coeffs.flags.f_contiguous

        # ---------------- zero output ----------------
        coeffs.fill(0.0)

        Kmax = self.K_batch_max
        R  = self.N_rot
        C = int(self.N_chi)
        P = int(self.N_peaks)
        G = int(len(self.sym_ops_cpu))
        Nx = self.Nx
        Ny = self.Ny
        coords_gpu   = self.coords_gpu
        sym_ops_gpu  = self.sym_ops_gpu
        h_gpu_normed        = self.h_gpu_normed
        intensity_gpu = self.intens_gpu

        for b in self.batches:
            k0 = b["k_start"]
            k1 = b["k_end"]
            Kb = b["K_batch"]

            self._coeffs_batch_F.fill(0.0)
            self.coeffs_sino_C.fill(0.0)
            self.coeffs_sino_F.fill(0.0)


            # Create pf matrix batch
            self._norm_factor_kmax.fill(0.0) # make sure that this is set to zero!
            self._inv_sigma2_kmax.fill(0.0)
            sigma = np.asarray(self.sigma_cpu[k0:k1], dtype=np.float32)
            inv_sigma2_cpu = (1.0 / (sigma * sigma)).astype(np.float32)
            norm_factor_cpu = (1.0 / (8.0 * np.pi * sigma * sigma)).astype(np.float32)
            # Input the norm factors to a zero array, so the norm factor for the extra rows in the last batch are zero
            cl.enqueue_copy(self.queue, self._inv_sigma2_kmax.data, inv_sigma2_cpu, device_offset=0)
            cl.enqueue_copy(self.queue, self._norm_factor_kmax.data, norm_factor_cpu, device_offset=0)

            total = Kb*9
            self.k.SLICE_GRIDINV_K_BATCH(
                self.queue,
                (total,),
                None,
                self.grid_inv_gpu.data,      # input: (K_total, 9)
                self._grid_inv_kmax.data,    # output: (K_batch_max, 9)
                np.int32(self.K),            # K_IN  (total grid size)
                np.int32(Kb),                # K_OUT (this batch size)
                np.int32(k0),                # K_START
            )


            total = R*Kmax*C*P
            self.pfmatrix_eval_kernel(
                self.queue,
                (total,),
                None,
                coords_gpu.data,
                self._grid_inv_kmax.data,
                sym_ops_gpu.data,
                h_gpu_normed.data,
                self._inv_sigma2_kmax.data,
                self._norm_factor_kmax.data,
                self._basis_batch_kmax.data,
                np.int32(R),
                np.int32(Kmax),
                np.int32(C),
                np.int32(P),
                np.int32(G),
            )
            if not self.normalized:
                self._scale_pf_by_intensity_inplace(self._basis_batch_kmax, intensity_gpu)

            # Transpose the pf matrix
            total = R*Kmax*P*C
            self.k.btranspose_kernel(
                self.queue,
                (total,),
                None,
                self._basis_batch_kmax.data,
                self._basis_batch_transpose_kmax.data,
                np.int32(R),
                np.int32(Kmax),
                np.int32(P*C),
                np.int32(total),
            )

            # batched gemm overwrites self.coeffs_sino_C
            batched_gemm_adj_clblast(self.queue, data, self._basis_batch_transpose_kmax, self.coeffs_sino_C, R, Nx, P*C, Kmax, self.N_rot/np.pi)
            # Now transpose, backproject and depose the coefficients in the "coeffs" array

            # Transpose
            total = R*Nx*Kmax
            self.k.transpose_omega_d_k_c_to_d_omega_k_f(
                self.queue,
                (total,),
                None,
                self.coeffs_sino_C.data,
                self.coeffs_sino_F.data,
                np.int32(R),
                np.int32(Nx),
                np.int32(Kmax),
                np.int32(total),
            )

            # Backproject
            gratopy.backprojection(
                self.coeffs_sino_F,
                self.PS,
                img=self._coeffs_batch_F,
            )


            total = Nx * Ny * Kb

            self.k.scatter_k_lastaxis_f(
                self.queue,
                (total,),
                None,
                coeffs.data,              # dst
                self._coeffs_batch_F.data,# src
                np.int32(Nx),
                np.int32(Ny),
                np.int32(self.K),
                np.int32(k0),
                np.int32(Kb),
                np.int32(total),
            )



    def _scale_pf_by_intensity_inplace(self, PF_GPU: clarray.Array, INTENSITY_GPU: clarray.Array) -> None:
        """
        PF_GPU:        (R, Kb, C, P) C-order
        INTENSITY_GPU: (P,) float32 on GPU
        """
        R = int(PF_GPU.shape[0])
        Kb = int(PF_GPU.shape[1])
        C = int(PF_GPU.shape[2])
        P = int(PF_GPU.shape[3])

        assert INTENSITY_GPU.dtype == np.float32
        assert int(INTENSITY_GPU.size) == P

        total = R * Kb * C * P
        self.k.SCALE_PF_BY_INTENSITY_INPLACE(
            self.queue,
            (total,),
            None,
            PF_GPU.data,
            INTENSITY_GPU.data,
            np.int32(R),
            np.int32(Kb),
            np.int32(C),
            np.int32(P),
        )





def batched_gemm_clblast(queue, A3, B3, C3, R, M, K, N):
    """
    A3: (R, M, K) clarray
    B3: (R, K, N) clarray
    C3: (R, M, N) clarray
    """

    # reshape views (NO COPY)
    A = A3.reshape((R*M, K))
    B = B3.reshape((R*K, N))
    C = C3.reshape((R*M, N))

    # leading dimensions (row-major)
    a_ld = K
    b_ld = N
    c_ld = N

    # strides between batches (in elements)
    a_stride = M * K
    b_stride = K * N
    c_stride = M * N

    gemmStridedBatched(
        queue,
        M, N, K,
        R,              # batch_count
        A, B, C,        # MUST be 2D
        a_ld, b_ld, c_ld,
        a_stride, b_stride, c_stride,
        alpha=1.0,
        beta=1.0,
    )




def batched_gemm_adj_clblast(queue, Y3, BT3, X3, R, Mx, Nsub, K, alpha):
    """
    Y3:  (R, Mx,   Nsub)  clarray
    BT3: (R, Nsub, K)     clarray  (explicitly transposed B)
    X3:  (R, Mx,   K)     clarray (output)
    """

    # 2D views (NO COPY)
    A = Y3.reshape((R * Mx, Nsub))   # (R*Mx, Nsub)
    B = BT3.reshape((R * Nsub, K))   # (R*Nsub, K)
    C = X3.reshape((R * Mx, K))      # (R*Mx, K)

    # leading dimensions (row-major)
    a_ld = Nsub
    b_ld = K
    c_ld = K

    # batch strides (in elements)
    a_stride = Mx * Nsub
    b_stride = Nsub * K
    c_stride = Mx * K

    gemmStridedBatched(
        queue,
        Mx, K, Nsub,     # m, n, k
        R,               # batch_count
        A, B, C,
        a_ld, b_ld, c_ld,
        a_stride, b_stride, c_stride,
        alpha=alpha,
        beta=0.0,
        a_transp=False,
        b_transp=False,  # <<< now no ambiguity
    )



def gpu_norm(x):
    return float(clarray.sum(x*x).get() ** 0.5)



def estimate_L_power(op, niter=20, seed=0, eps=1e-30, verbose=1):
    q = op.queue
    rng = np.random.default_rng(seed)

    # x in domain, Fortran
    x = clarray.empty(q, (op.Nx, op.Ny, op.K), np.float32, order="F")
    # y in range, C
    Ax = clarray.empty(q, (op.N_rot, op.Nx, op.N_seg), np.float32, order="C")
    # z = A^*Ax in domain
    z = clarray.empty(q, x.shape, np.float32, order="F")

    # init x random
    x_host = rng.standard_normal(x.shape).astype(np.float32, copy=False, order="F")
    import pyopencl as cl
    cl.enqueue_copy(q, x.data, x_host)
    q.finish()

    # normalize x
    xnorm = float(np.sqrt(clarray.vdot(x, x).get()) + eps)
    x *= np.float32(1.0 / xnorm)
    q.finish()

    L_est = None
    for it in range(niter):
        # Ax = A x
        op.direct_cl(x, Ax)
        # z = A^* Ax
        op.adjoint_cl(Ax, z)
        q.finish()

        # Rayleigh quotient: <x, z> / <x, x> ; since x normalized, denom=1
        num = float(clarray.vdot(x, z).get())
        den = float(clarray.vdot(x, x).get()) + eps
        L_est = num / den

        # next x = z / ||z||
        znorm = float(np.sqrt(clarray.vdot(z, z).get()) + eps)
        x[:] = z * np.float32(1.0 / znorm)
        q.finish()

        if verbose:
            print(f"[power {it+1:02d}] L_est={L_est:.6e}  ||z||={znorm:.6e}")

    return L_est