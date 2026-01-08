import numpy as np
import time
import gc




import pyopencl as cl
import pyopencl.array as clarray
import gratopy
from pyclblast import gemmStridedBatched


from scipy.spatial.transform import Rotation as R
from pole_figure_geometry import GeometryContainerM

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
        self.angles = np.linspace(self.angle_range[0], self.angle_range[1], self.N_rot, endpoint=True)
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
            (self.Nx, self.Ny, self.K),
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

        self.h_cpu = np.asarray(self.material.h_vecs_normed, dtype=np.float32, order="C")
        self.h_gpu = clarray.to_device(self.queue, self.h_cpu)

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

        self.coeffs_sino_C = clarray.empty(
            self.queue,
            (self.N_rot, self.Nx, self.K),
            dtype=np.float32,
            order="C",
            )
        
        self.coeffs_sino_F = clarray.empty(
                self.queue,
                (self.PS.n_detectors, self.PS.n_angles, self.K),
                dtype=np.float32,
                order="F",
            )
        

        self._data_batch = clarray.empty(
            self.queue,
            (self.N_rot, self.Nx, self.N_chi * self.N_peaks),  # N = C*P in your simple model
            dtype=np.float32,
            order="C",
        )



    def free_memory(self):


        lst = getattr(self, "coeffs_sino_C", None)

        gpu_lists = [
            "coeffs_sino_C",
            "coords_gpu",
            "grid_inv_gpu",
            "sym_ops_gpu",
            "h_gpu",
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
        """
        In-place OpenCL forward operator.

        Parameters
        ----------
        coeffs_gpu_full : clarray
            Shape (Nx, Ny, K), Fortran order
        data : clarray
            Shape (N_rot, Nx, N_seg), C order
            Accumulated into (will be zeroed here)
        """

        t_total_start = time.perf_counter()

        t_transpose_total = 0.0
        t_forward_total   = 0.0

        # --- checks ---
        assert coeffs.shape == (self.Nx, self.Ny, self.K)
        assert coeffs.flags.f_contiguous

        assert data.shape == (self.N_rot, self.Nx, self.N_seg)
        assert data.flags.c_contiguous
        assert data.queue is self.queue

        # --- zero dataput (important!) ---
        data.fill(0.0)
        self.queue.finish()

        coeffs_sino_F = self.coeffs_sino_F
        coeffs_sino_F.fill(0.0)
        coeffs_sino_C = self.coeffs_sino_C
        coeffs_sino_C.fill(0.0)

        gratopy.forwardprojection(
            coeffs,
            self.PS,
            sino=coeffs_sino_F
        )


        # Must finish before slicing/transposing
        self.queue.finish()
        assert coeffs_sino_F.flags.f_contiguous

        # ---------------- main loop over materials ----------------
        total = self.N_rot * self.Nx * self.K

        t0 = time.perf_counter()
        self.k.transpose_d_omega_k_f_to_c(
            self.queue,
            (total,),
            None,
            coeffs_sino_F.data,
            coeffs_sino_C.data,
            np.int32(self.Nx),        # D
            np.int32(self.N_rot),     # O
            np.int32(self.K),            # K
            np.int32(total),
        )

        self.queue.finish()
        t_transpose_total += time.perf_counter() - t0

        # ---- forward kernel (accumulates into data_y) ----
        t0 = time.perf_counter()
        self.forward_gpu_opencl(data, coeffs_sino_C)
        self.queue.finish()
        t_forward_total += time.perf_counter() - t0

        t_total = time.perf_counter() - t_total_start

        if self.verbose:
            print("\n=== DIRECT_CL() OpenCL TIMING ===")
            print(f"transpose total        : {t_transpose_total:.4f} s")
            print(f"forward_gpu total      : {t_forward_total:.4f} s")
            print("--------------------------------")
            print(f"TOTAL direct_cl() time : {t_total:.4f} s")
            print("================================\n")


    def forward_gpu_opencl(self, data, coeffs_sino_C):
        """
        Batched OpenCL forward operator for one material.

        out_gpu : (R, Mx, Nseg_full) C-order, accumulated in-place
        coeffs_gpu   : (R, Mx, K_i)       C-order (already per-material coeffs)
        i_mat        : material index
        """
        queue = self.queue

        # ---------------- constants / shapes ----------------
        R = int(self.N_rot)
        Mx = int(self.Nx)
        K = self.K
        data_batch = self._data_batch

        coords_gpu   = self.coords_gpu
        grid_inv_gpu = self.grid_inv_gpu
        sigma_cpu = self.sigma_cpu
        sym_ops_gpu  = self.sym_ops_gpu
        h_gpu        = self.h_gpu

        # Sanity checks (cheap and saves pain)
        assert int(coords_gpu.shape[0]) == R
        assert int(coords_gpu.shape[1]) == int(self.N_chi)
        assert int(coords_gpu.shape[2]) == int(self.N_peaks)
        assert int(coords_gpu.shape[3]) == 3
        assert int(grid_inv_gpu.shape[0]) == K
        assert int(grid_inv_gpu.shape[1]) == 9

        C = int(self.N_chi)
        P = int(self.N_peaks)
        G = int(len(self.sym_ops_cpu))

        # Intensities should be cached on GPU per material
        intensity_gpu = self.intens_gpu  # (P,) float32 GPU
        assert int(intensity_gpu.size) == P

        # ---------------- batch loop ----------------
        for b in self.batches:
            k0 = int(b["k_start"])
            k1 = int(b["k_end"])
            Kb = int(b["K_batch"])
            assert Kb == (k1 - k0)

            # ---- 1) slice coeffs_gpu -> (R, Mx, Kb) ----
            coeffs_batch = self._slice_coeffs_k_batch(coeffs_sino_C, k0, k1)

            # ---- 2) slice grid_inv -> (Kb, 9) ----
            grid_inv_batch = self._slice_gridinv_k_batch(grid_inv_gpu, k0, k1)

            sigma_cpu_batch = sigma_cpu[k0:k1]

            # ---- 3) PF basis batch: (R, Kb, C, P) ----
            basis_batch = clarray.empty(queue, (R, Kb, C, P), dtype=np.float32, order="C")
            # IMPORTANT: call pfmatrix_eval_gpu with correct signature
            pfmatrix_eval_gpu(
                queue=queue,
                pfo_kernel=self.pfmatrix_eval_kernel,
                coords_gpu=coords_gpu,
                grid_inv_gpu=grid_inv_batch,
                sym_ops_gpu=sym_ops_gpu,
                hvecs_gpu=h_gpu,
                R=R,
                K=Kb,
                C=C,
                P=P,
                G=G,
                sigma=sigma_cpu_batch,   # or wherever you store sigma
                out_gpu=basis_batch,
            )
            # ---- 4) scale by intensities: pf_basis_batch[r,k,c,p] *= intensity[p] ----
            if not self.normalized:
                self._scale_pf_by_intensity_inplace(basis_batch, intensity_gpu)
            


            basis_batch = basis_batch.reshape((R, Kb, C*P))

            batched_gemm_clblast(queue, coeffs_batch, basis_batch, data_batch, R=R, M=Mx, K=Kb, N=P*C)
            data += data_batch


            del basis_batch, coeffs_batch, grid_inv_batch




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

        t_total_start = time.perf_counter()

        t_adjoint_total   = 0.0
        t_transpose_total = 0.0

        queue = self.queue

        # ---------------- checks ----------------
        assert data.shape == (self.N_rot, self.Nx, self.N_seg)
        assert data.flags.c_contiguous
        assert data.queue is queue

        assert coeffs.shape == (self.Nx, self.Ny, self.K)
        assert coeffs.flags.f_contiguous
        assert coeffs.queue is queue

        # ---------------- zero output ----------------
        coeffs.fill(0.0)
        queue.finish()

        # ---------------- allocate intermediate (sino-space adjoint result) ----------------
        O = self.N_rot
        D = self.Nx


        coeffs_sino_F = self.coeffs_sino_F
        coeffs_sino_F.fill(0.0)
        coeffs_sino_C = self.coeffs_sino_C
        coeffs_sino_C.fill(0.0)


        K = self.K
        total = O * D * K

        # ---- adjoint pole-figure operator ----
        t0 = time.perf_counter()
        self.adjoint_gpu_opencl(data, coeffs_sino_C)  # (ω, d, K_i), C
        queue.finish()
        t_adjoint_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        self.k.transpose_omega_d_k_c_to_d_omega_k_f(
            queue,
            (total,),
            None,
            coeffs_sino_C.data,
            coeffs_sino_F.data,
            np.int32(O),
            np.int32(D),
            np.int32(K),
            np.int32(total),
        )
        queue.finish()
        t_transpose_total += time.perf_counter() - t0

        # ---------------- X-ray backprojection ----------------
        gratopy.backprojection(
            coeffs_sino_F,
            self.PS,
            img=coeffs
        )
        queue.finish()

        t_total = time.perf_counter() - t_total_start

        if self.verbose:
            print("\n=== ADJOINT_CL() OpenCL TIMING ===")
            print(f"adjoint_gpu total     : {t_adjoint_total:.4f} s")
            print(f"transpose total       : {t_transpose_total:.4f} s")
            print("--------------------------------")
            print(f"TOTAL adjoint_cl() time: {t_total:.4f} s")
            print("================================\n")

                

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

                






    def adjoint_gpu_opencl(self, data: clarray.Array, coeffs_sino_C: clarray.Array) -> clarray.Array:
        """
        Batched OpenCL adjoint operator for one material.

        Parameters
        ----------
        data_gpu : clarray.Array
            Shape (R, Mx, Nseg_full), C-order
        i_mat : int
            Material index

        Returns
        -------
        coeffs_sino_C : clarray.Array
            Shape (R, Mx, K), C-order
        """
        queue = self.queue

        # ---------------- constants / shapes ----------------
        R = int(self.N_rot)
        Mx = int(self.Nx)
        K = int(self.K)

        # ---------------- PF GPU inputs (prepared in __init__) ----------------
        coords_gpu   = self.coords_gpu     # (R, C, P, 3)
        grid_inv_gpu = self.grid_inv_gpu   # (K_i, 9)
        sigma_cpu = self.sigma_cpu
        sym_ops_gpu  = self.sym_ops_gpu    # (G, 9)
        h_gpu        = self.h_gpu         # (P, 3)

        C = int(self.N_chi)
        P = int(self.N_peaks)
        G = int(len(self.sym_ops_cpu))

    

        # Intensities cached on GPU per material: (P,)
        intensity_gpu = self.intens_gpu
        assert int(intensity_gpu.size) == P

        # ---------------- batch loop ----------------
        for b in self.batches:
            k0 = int(b["k_start"])
            k1 = int(b["k_end"])
            Kb = int(b["K_batch"])
            assert Kb == (k1 - k0)

            # ---- 2) slice grid_inv -> (Kb, 9) ----
            grid_inv_batch = self._slice_gridinv_k_batch(grid_inv_gpu, k0, k1)
            sigma_cpu_batch = sigma_cpu[k0:k1]

            # ---- 3) PF basis batch: (R, Kb, C, P) ----
            basis_batch = clarray.empty(queue, (R, Kb, C, P), dtype=np.float32, order="C")
            basis_batch.fill(0.0)

            pfmatrix_eval_gpu(
                queue=queue,
                pfo_kernel=self.pfmatrix_eval_kernel,
                coords_gpu=coords_gpu,
                grid_inv_gpu=grid_inv_batch,
                sym_ops_gpu=sym_ops_gpu,
                hvecs_gpu=h_gpu,
                R=R,
                K=Kb,
                C=C,
                P=P,
                G=G,
                sigma=sigma_cpu_batch,   # ensure you store this on self
                out_gpu=basis_batch,
            )

            if not self.normalized:
                self._scale_pf_by_intensity_inplace(basis_batch, intensity_gpu)

            basis_batch = basis_batch.reshape((R, Kb, C*P))

            # ---- 6) transpose B for adjoint GEMM: BT = (R, Nsub, Kb) ----
            basisT_batch = clarray.empty(queue, (R, P*C, Kb), dtype=np.float32, order="C")
            total_bt = np.int32(R) * np.int32(Kb) * np.int32(P*C)

            self.k.btranspose_kernel(
                queue,
                (int(total_bt),),
                None,
                basis_batch.data,
                basisT_batch.data,
                np.int32(R),
                np.int32(Kb),
                np.int32(P*C),
                np.int32(total_bt),
            )



            x_batch = clarray.empty(queue, (R, Mx, Kb), dtype=np.float32, order="C")
            x_batch.fill(0.0)
            batched_gemm_adj_clblast(queue, data, basisT_batch, x_batch, R, Mx, P*C, Kb)
            # ---- 8) scatter x_batch into coeffs_sino_C[:, :, k0:k1] ----
            total_scatter = np.int32(R) * np.int32(Mx) * np.int32(Kb)

            self.k.scatter_k_batch_c(
                queue,
                (int(total_scatter),),
                None,
                coeffs_sino_C.data,
                x_batch.data,
                np.int32(R),
                np.int32(Mx),
                np.int32(K),
                np.int32(k0),
                np.int32(Kb),
                np.int32(total_scatter),
            )

            # cleanup batch temporaries
            del grid_inv_batch, basis_batch, basisT_batch, x_batch






    def _slice_coeffs_k_batch(self, COEFFS_GPU: clarray.Array, K_START: int, K_END: int) -> clarray.Array:
        """
        COEFFS_GPU: (R, Mx, K_IN) C-order
        returns:    (R, Mx, K_OUT) C-order contiguous
        """
        R = int(COEFFS_GPU.shape[0])
        Mx = int(COEFFS_GPU.shape[1])
        K_IN = int(COEFFS_GPU.shape[2])
        K_OUT = int(K_END - K_START)

        COEFFS_OUT = clarray.empty(self.queue, (R, Mx, K_OUT), dtype=np.float32, order="C")

        total = R * Mx * K_OUT
        self.k.SLICE_COEFFS_K_BATCH(
            self.queue,
            (total,),
            None,
            COEFFS_GPU.data,
            COEFFS_OUT.data,
            np.int32(R),
            np.int32(Mx),
            np.int32(K_IN),
            np.int32(K_OUT),
            np.int32(K_START),
        )
        return COEFFS_OUT



    def _slice_gridinv_k_batch(self, GRIDINV_GPU: clarray.Array, K_START: int, K_END: int) -> clarray.Array:
        """
        GRIDINV_GPU: (K_IN, 9) C-order
        returns:     (K_OUT, 9) C-order contiguous
        """
        K_IN = int(GRIDINV_GPU.shape[0])
        assert int(GRIDINV_GPU.shape[1]) == 9

        K_OUT = int(K_END - K_START)

        GRIDINV_OUT = clarray.empty(self.queue, (K_OUT, 9), dtype=np.float32, order="C")

        total = K_OUT * 9
        self.k.SLICE_GRIDINV_K_BATCH(
            self.queue,
            (total,),
            None,
            GRIDINV_GPU.data,
            GRIDINV_OUT.data,
            np.int32(K_IN),
            np.int32(K_OUT),
            np.int32(K_START),
        )
        return GRIDINV_OUT



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
        beta=0.0,
    )




def batched_gemm_adj_clblast(queue, Y3, BT3, X3, R, Mx, Nsub, K):
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
        alpha=1.0,
        beta=0.0,
        a_transp=False,
        b_transp=False,  # <<< now no ambiguity
    )



def gpu_norm(x):
    return float(clarray.sum(x*x).get() ** 0.5)
