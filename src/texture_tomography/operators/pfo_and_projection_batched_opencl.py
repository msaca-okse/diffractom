# odf_sh_image_operator.py
from typing import Optional, Sequence, Tuple
from numpy.typing import NDArray
import numpy as np
import h5py
import time
import os
import gc

from cil.framework import ImageGeometry, ImageData, BlockDataContainer, AcquisitionGeometry, AcquisitionData
from cil.optimisation.operators import LinearOperator


import pyopencl as cl
import pyopencl.array as clarray
import gratopy
from pyclblast import gemmStridedBatched


from scipy.spatial.transform import Rotation as R
from pole_figure_geometry import GeometryContainerM
from odftt.texture import grids, odfs, point_groups
from package.utils.lattice import (
    cubic, tetragonal, orthorhombic, hexagonal,
    trigonal_rhombohedral, monoclinic, triclinic
)
from package.utils.coordinates import get_probed_coordinates

from package.texture_tomography.operators.create_pfo_matrix import (
    pfmatrix_eval_gpu,
    build_pf_program
)
from package.texture_tomography.operators.pfo_kernels import build_all_opencl




class PFO_OPENCL_BATCHED(LinearOperator):

    def __init__(
        self,
        cfg: None,
        materials: None,
        grids: None,
        two_thetas: None,
        verbose: bool = False,
    ):
        

        # -------------------------
        # Memory planning (NO allocations)
        # -------------------------


        self.cfg = cfg
        self.materials = materials
        self.grids = grids
        self.two_thetas = np.array(two_thetas).astype(np.float32)
        self.peak_width = self.cfg['peak_width']
        self.verbose = verbose


        # Shapes from SH projection matrix
        #N_rot, K, N_chi, N_theta = map(int, B_matrix.shape)
        self.N_chi = self.cfg['N_chi']
        self.N_theta = len(two_thetas)
        self.N_seg = self.N_theta * self.N_chi
        self.Nx = self.cfg['Nx']
        self.Ny = self.cfg['Ny']
        self.N_rot = self.cfg['N_rot']
        self.kernel_sigma = self.cfg['kernel_sigma']
        self.grid_resolution_parameter = self.cfg["grid_resolution_parameter"]
        self.angle_range = np.array(self.cfg['angle_range'])/180*np.pi
        self.angles = np.linspace(self.angle_range[0], self.angle_range[1], self.N_rot, endpoint=True)
        self.N_mat = len(self.materials)


        # --- context / queue ---
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)

        # --- build kernels ---
        self.prg, self.k, self.pf_prg = build_all_opencl(self.ctx, ts=16)
        self.pf_prg = build_pf_program(self.ctx)
        self.pfmatrix_eval_kernel = cl.Kernel(self.pf_prg, "pfmatrix_eval")

        # --- expose kernels under the SAME NAMES as before ---
        # self.expand_kernel = k.expand_kernel
        # self.transpose_kernel = k.transpose_kernel
        # self.accumulate_kernel = k.accumulate_kernel
        # self.gather_kernel = k.gather_kernel
        # self.btranspose_kernel = k.btranspose_kernel
        # self.transpose_r_mx_k_to_k_r_mx_kernel = k.transpose_r_mx_k_to_k_r_mx_kernel
        # self.gather_coeffs_kernel = k.gather_coeffs_kernel
        # self.transpose_d_omega_k_f_to_c = k.transpose_d_omega_k_f_to_c
        # self.slice_k_lastaxis_f = k.slice_k_lastaxis_f
        # self.transpose_omega_d_k_c_to_d_omega_k_f = k.transpose_omega_d_k_c_to_d_omega_k_f
        # self.scatter_k_lastaxis_f = k.scatter_k_lastaxis_f
        # self.SLICE_COEFFS_K_BATCH = k.SLICE_COEFFS_K_BATCH
        # self.SLICE_GRIDINV_K_BATCH = k.SLICE_GRIDINV_K_BATCH
        # self.SCALE_PF_BY_INTENSITY_INPLACE = k.SCALE_PF_BY_INTENSITY_INPLACE
        # self.scatter_k_batch_c = k.scatter_k_batch_c






        self.transfer_material_parameters_to_gpu()
        self.detector_coordinates()
        self.transfer_grid_parameters_to_gpu()
        self.set_convolution_masks()

        self.K_sum  = int(self.K_list.sum())



        #        --- 2) Create ProjectionSettings using THIS queue ---
        self.PS = gratopy.ProjectionSettings(
            self.queue,
            gratopy.PARALLEL,
            (self.Nx, self.Ny, self.K_sum),
            self.N_rot,
            n_detectors=self.Nx,
            image_width=self.Nx,
            detector_width=self.Nx,
            detector_shift=2,
            angle_range=self.angle_range
        )
        assert self.queue.context.int_ptr == self.ctx.int_ptr



        # Allocate buffers
        self.allocate_out_buffer()
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

        self.pf_coords_cpu_list = []
        self.pf_coords_gpu_list = []

        for i_mat in range(self.N_mat):
    

            h_cpu = self.pf_h_cpu_list[i_mat]

            two_theta_peaks = 2.0 * np.arcsin(
                np.linalg.norm(h_cpu, axis=1) / (4.0 * np.pi) * wavelength_angstrom
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
            coords_cpu = np.asarray(coords_cpu, dtype=np.float32, order="C")
            coords_gpu = clarray.to_device(self.queue, coords_cpu)

            self.pf_coords_cpu_list.append(coords_cpu)
            self.pf_coords_gpu_list.append(coords_gpu)



    def transfer_material_parameters_to_gpu(self):
        self.pf_h_gpu_list = []
        self.pf_h_cpu_list = []
        self.pf_intensity_gpu_list = []
        self.pf_intensity_cpu_list = []
        self.pf_sym_ops_gpu_list = []
        self.pf_sym_ops_cpu_list = []
        self.N_peaks_list = []
        self.peak_positions_np_list = []

        for mat in self.materials:
            # ---- normalized reciprocal lattice vectors ----
            h_cpu = np.asarray(mat.h_vecs_normed, dtype=np.float32, order="C")
            h_gpu = clarray.to_device(self.queue, h_cpu)
            self.pf_h_gpu_list.append(h_gpu)
            self.pf_h_cpu_list.append(h_cpu)

            # ---- peak intensities ----
            intens_cpu = np.asarray(mat.intensities(), dtype=np.float32, order="C")
            intens_gpu = clarray.to_device(self.queue, intens_cpu)
            self.pf_intensity_gpu_list.append(intens_gpu)
            self.pf_intensity_cpu_list.append(intens_cpu)

            # ---- symmetry operators (flattened 3x3) ----
            sym_ops_cpu = np.asarray(mat.point_group_matrices, dtype=np.float32, order="C")
            sym_ops_gpu = clarray.to_device(self.queue, sym_ops_cpu)
            self.pf_sym_ops_gpu_list.append(sym_ops_gpu)
            self.pf_sym_ops_cpu_list.append(sym_ops_cpu)

            self.N_peaks_list.append(len(intens_cpu))
            peak_positions = np.asarray(mat.two_theta(), dtype=np.float32, order="C")
            self.peak_positions_np_list.append(peak_positions)


    def transfer_grid_parameters_to_gpu(self):
        """
        Transfer active orientation grid parameters (rotations + sigmas)
        from OrientationTree objects to GPU.
        """

        self.pf_grid_inv_gpu_list = []
        self.sigma_cpu_list = []
        self.K_list = []

        for i_mat in range(self.N_mat):
            tree = self.grids[i_mat]

            # --- extract active leaf nodes ---
            active_indices = tree.active_leaf_nodes()
            nodes = [tree.nodes[i] for i in active_indices]

            # --- inverse rotations ---
            grid_inv_cpu = np.stack(
                [n.R.inv().as_matrix().reshape(-1) for n in nodes],
                axis=0
            ).astype(np.float32)

            grid_inv_gpu = clarray.to_device(self.queue, grid_inv_cpu)
            self.pf_grid_inv_gpu_list.append(grid_inv_gpu)

            # --- sigma per node ---
            sigma_cpu = np.array(
                [node.sigma for node in nodes],
                dtype=np.float32
            )
            self.sigma_cpu_list.append(sigma_cpu)

            # --- bookkeeping ---
            self.K_list.append(len(nodes))

        self.K_list = np.array(self.K_list, dtype=int)

        # offsets for concatenated layouts (unchanged logic)
        self.offsets = np.zeros(len(self.K_list) + 1, dtype=int)
        self.offsets[1:] = np.cumsum(self.K_list)





    def allocate_out_buffer(self):
        # Allocate buffer for forward computation
        self._out_sub = []

        for i_mat in range(self.N_mat):
            Nsub = len(self.full_idx_list[i_mat])

            out_sub = clarray.empty(
                self.queue,
                (self.N_rot, self.Nx, Nsub),
                dtype=np.float32,
                order="C",
            )

            self._out_sub.append(out_sub)


    def allocate_coefficient_buffer(self):
            # ---- reusable transpose buffers (per material) ----
        self._coeffs_t_gpu = []

        for i_mat, Ki in enumerate(self.K_list):
            coeffs_t_gpu = clarray.empty(
                self.queue,
                (self.N_rot, self.Nx, Ki),
                dtype=np.float32,
                order="C",
            )

            self._coeffs_t_gpu.append(coeffs_t_gpu)


    def set_convolution_masks(self):
        # --- PF-matrix / peak info / masks (per material) ---
        self.theta_mask_list = []
        self.full_mask_list = []
        self.full_idx_list = []
        self.N_theta_mask_list = []

        for i_mat in range(self.N_mat):

            diff = np.abs(self.two_thetas[:, None] - self.peak_positions_np_list[i_mat][None, :])
            min_dist = np.min(diff, axis=1)  # (N_theta,)
            theta_mask = min_dist < (1.8 * self.peak_width)  # (N_theta,)

            full_mask = np.tile(theta_mask, self.N_chi)      # (N_theta*N_chi,)
            full_idx = np.nonzero(full_mask)[0]              # indices into full detector axis

            self.theta_mask_list.append(theta_mask)
            self.full_mask_list.append(full_mask)
            self.full_idx_list.append(full_idx)
            self.N_theta_mask_list.append(int(np.sum(theta_mask)))


        self.full_idx_gpu_list = []

        for idx in self.full_idx_list:
            idx_np = np.asarray(idx, dtype=np.int32)
            idx_gpu = clarray.to_device(self.queue, idx_np)
            self.full_idx_gpu_list.append(idx_gpu)



    def free_memory(self):

        gpu_lists = [
            "_coeffs_t_gpu",
            "_out_sub",
            "pf_coords_gpu_list",
            "pf_grid_inv_gpu_list",
            "pf_sym_ops_gpu_list",
            "pf_h_gpu_list",
            "pf_intensity_gpu_list",
            "full_idx_gpu_list",
        ]

        for name in gpu_lists:
            lst = getattr(self, name, None)
            if lst is not None:
                for buf in lst:
                    del buf
                lst.clear()

        for name in [
            "_coeffs_gpu_full_sino",
            "_x_full_gpu",
            "B_gpu",
        ]:
            if hasattr(self, name):
                delattr(self, name)

        gc.collect()
        





    def set_pf_batch_max_gb(self, max_gb: float):
        """
        Set maximum allowed GPU memory (in GB) for ONE convolved PF batch.

        This controls batching over K.
        """
        self.pf_batch_max_gb = float(max_gb)




    def get_pf_batches_for_material(self, i_mat: int):
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
        T = self.N_theta_mask_list[i_mat]   # masked theta count
        K_total = self.K_list[i_mat]

        bytes_per_float = 4

        # Memory per K after convolution:
        # (R, C, T) per K
        bytes_per_K = R * C * T * bytes_per_float

        max_bytes = self.pf_batch_max_gb * (1024 ** 3)

        # At least one K per batch
        if bytes_per_K >0.01:
            K_batch_max = max(int(max_bytes // bytes_per_K), 1)
             # Safety: never exceed available K
            K_batch_max = min(K_batch_max, K_total)
        else:
            K_batch_max = 1



        batches = []

        k0 = 0
        while k0 < K_total:
            k1 = min(k0 + K_batch_max, K_total)
            batches.append({
                "k_start": k0,
                "k_end": k1,
                "K_batch": k1 - k0,
            })
            k0 = k1

        return batches

    
    

    def get_c_opencl_fortran(self, coeffs_gpu, i_mat):
        """
        Slice coefficients along last axis (K) on GPU.

        coeffs_gpu: clarray, shape (d, ω, K_total), order='F'
        returns:    clarray, shape (d, ω, K_i),     order='F'
        """

        # --- K slicing info ---
        i0 = self.offsets[i_mat]
        i1 = self.offsets[i_mat + 1]
        Ki = i1 - i0

        # --- dimensions ---
        D, O, Ktot = coeffs_gpu.shape

        # --- allocate output (Fortran order!) ---
        out = clarray.empty(
            self.queue,
            (D, O, Ki),
            dtype=np.float32,
            order="F",
        )

        total = D * O * Ki

        # --- launch slicing kernel ---
        self.k.slice_k_lastaxis_f(
            self.queue,
            (total,),
            None,
            coeffs_gpu.data,
            out.data,
            np.int32(D),
            np.int32(O),
            np.int32(Ktot),
            np.int32(i0),
            np.int32(Ki),
            np.int32(total),
        )

        return out





    def convolve_matrix_from_pf_batch(
        self,
        *,
        i_mat: int,
        pf_basis_batch: clarray.Array,   # (R, Kb, C, P)
    ):
        queue = self.queue

        # ---------------- shapes ----------------
        R  = int(self.N_rot)
        C  = int(self.N_chi)
        Kb = int(pf_basis_batch.shape[1])
        P  = int(pf_basis_batch.shape[3])

        # Masked theta count
        T = int(self.N_theta_mask_list[i_mat])
        Nsub = C * T

        # ---------------- Gaussian weights (CPU → GPU) ----------------
        # peak_positions_np_list[i_mat]: (P,)
        peaks_np = np.asarray(
            self.peak_positions_np_list[i_mat],
            dtype=np.float32
        )

        theta_mask = self.theta_mask_list[i_mat]
        two_theta_masked = self.two_thetas[theta_mask]

        dt = float(self.two_thetas[1] - self.two_thetas[0])
        inv_norm = 1.0 / np.sqrt(2.0 * np.pi * (self.peak_width ** 2))

        # diff: (P, T)
        diff = peaks_np[:, None] - two_theta_masked[None, :]
        gaussian_np = (
            inv_norm
            * np.exp(-0.5 * (diff / self.peak_width) ** 2)
            * dt
        ).astype(np.float32)

        gaussian_gpu = clarray.to_device(queue, gaussian_np)

        # ---------------- output buffer ----------------
        # (R, Kb, C, T)
        out_gpu = clarray.empty(
            queue,
            (R, Kb, C, T),
            dtype=np.float32,
            order="C",
        )

        # ---------------- kernel launch ----------------
        global_size = (R * Kb * C * T,)

        self.k.expand_kernel(
            queue,
            global_size,
            None,
            pf_basis_batch.data,     # basis
            gaussian_gpu.data,       # gaussian
            out_gpu.data,            # out
            np.int32(R),
            np.int32(Kb),
            np.int32(C),
            np.int32(P),
            np.int32(T),
        )

        # ---------------- reshape to GEMM-compatible layout ----------------
        # (R, Kb, C*T) == (R, Kb, Nsub)
        return out_gpu.reshape(R, Kb, Nsub)


    def set_peak_width(self, peak_width):
        self.peak_width = peak_width

    def set_kernel_sigma(self, kernel_sigma):
        self.kernel_sigma = kernel_sigma


    def forward_kernel_width(self, coeffs_gpu_full, peak_width = None, kernel_sigma = None, coefficient_constant = None):
        """
        Allocating convenience wrapper for the OpenCL forward operator.

        Returns
        -------
        yin_gpu : clarray
            Shape (N_rot, Nx, N_seg), C order
        """
        # save the old parameters
        old_peak_width = self.peak_width
        old_kernel_sigma = self.kernel_sigma

        if peak_width is not None:
            self.set_peak_width(peak_width=peak_width)

        if kernel_sigma is not None:
            self.set_kernel_sigma(kernel_sigma=kernel_sigma)

        self.set_material_lists()
        self.allocate_out_buffer()
        self.allocate_coefficient_buffer()

        if coefficient_constant is not None:
            coeffs_gpu_full *= np.float32(coefficient_constant)


        yin_gpu = clarray.zeros(
            self.queue,
            (self.N_rot, self.Nx, self.N_chi * self.N_theta),
            dtype=np.float32,
            order="C",
        )

        self.direct_cl(coeffs_gpu_full, yin_gpu)

        if peak_width is not None:
            self.set_peak_width(peak_width=old_peak_width)

        if kernel_sigma is not None:
            self.set_kernel_sigma(kernel_sigma=old_kernel_sigma)

        return yin_gpu


    def direct(self, coeffs_gpu_full):
        """
        Allocating convenience wrapper for the OpenCL forward operator.

        Returns
        -------
        yin_gpu : clarray
            Shape (N_rot, Nx, N_seg), C order
        """

        yin_gpu = clarray.zeros(
            self.queue,
            (self.N_rot, self.Nx, self.N_chi * self.N_theta),
            dtype=np.float32,
            order="C",
        )

        self.direct_cl(coeffs_gpu_full, yin_gpu)
        return yin_gpu



    def direct_cl(self, coeffs_gpu_full, out_y):
        """
        In-place OpenCL forward operator.

        Parameters
        ----------
        coeffs_gpu_full : clarray
            Shape (Nx, Nx, K_sum), Fortran order
        out_y : clarray
            Shape (N_rot, Nx, N_seg), C order
            Accumulated into (will be zeroed here)
        """

        t_total_start = time.perf_counter()

        t_get_c_total     = 0.0
        t_transpose_total = 0.0
        t_forward_total   = 0.0

        # --- checks ---
        assert coeffs_gpu_full.shape == (self.Nx, self.Ny, self.K_sum)
        assert coeffs_gpu_full.flags.f_contiguous

        assert out_y.shape == (self.N_rot, self.Nx, self.N_chi * self.N_theta)
        assert out_y.flags.c_contiguous
        assert out_y.queue is self.queue

        # --- zero output (important!) ---
        out_y.fill(0)
        self.queue.finish()

        # ---------------- gratopy projection ----------------
        if not hasattr(self, "_coeffs_gpu_full_sino"):
            coeffs_gpu_full_sino = clarray.empty(
                self.queue,
                (self.PS.n_detectors, self.PS.n_angles, self.K_sum),
                dtype=np.float32,
                order="F",
            )

            self._coeffs_gpu_full_sino = coeffs_gpu_full_sino


        coeffs_gpu_full_sino = self._coeffs_gpu_full_sino
        coeffs_gpu_full_sino.fill(0)

        gratopy.forwardprojection(
            coeffs_gpu_full,
            self.PS,
            sino=coeffs_gpu_full_sino
        )

        # Must finish before slicing/transposing
        self.queue.finish()
        assert coeffs_gpu_full_sino.flags.f_contiguous

        # ---------------- main loop over materials ----------------
        for i_mat in range(self.N_mat):

            K_i = self.K_list[i_mat]
            total = self.N_rot * self.Nx * K_i

            # ---- slice K on GPU (Fortran → Fortran) ----
            t0 = time.perf_counter()
            coeffs_sub_gpu = self.get_c_opencl_fortran(
                coeffs_gpu_full_sino,
                i_mat
            )
            self.queue.finish()
            t_get_c_total += time.perf_counter() - t0

            # ---- transpose (d,ω,K)[F] → (ω,d,K)[C] ----
            coeffs_t_gpu = self._coeffs_t_gpu[i_mat]


            t0 = time.perf_counter()
            self.k.transpose_d_omega_k_f_to_c(
                self.queue,
                (total,),
                None,
                coeffs_sub_gpu.data,
                coeffs_t_gpu.data,
                np.int32(self.Nx),        # D
                np.int32(self.N_rot),     # O
                np.int32(K_i),            # K
                np.int32(total),
            )
            self.queue.finish()
            t_transpose_total += time.perf_counter() - t0

            del coeffs_sub_gpu

            # ---- forward kernel (accumulates into out_y) ----
            t0 = time.perf_counter()
            self.forward_gpu_opencl(out_y, coeffs_t_gpu, i_mat)
            self.queue.finish()
            t_forward_total += time.perf_counter() - t0


        t_total = time.perf_counter() - t_total_start

        if self.verbose:
            print("\n=== DIRECT_CL() OpenCL TIMING ===")
            print(f"get_c (GPU slice) total : {t_get_c_total:.4f} s")
            print(f"transpose total        : {t_transpose_total:.4f} s")
            print(f"forward_gpu total      : {t_forward_total:.4f} s")
            print("--------------------------------")
            print(f"TOTAL direct_cl() time : {t_total:.4f} s")
            print("================================\n")




    def adjoint_cl(self, y_gpu, out_x):
        """
        In-place OpenCL adjoint operator.

        Parameters
        ----------
        y_gpu : clarray
            Shape (N_rot, Nx, N_seg), C-order
        out_x : clarray
            Shape (Nx, Nx, K_sum), Fortran-order
            Will be overwritten
        """

        t_total_start = time.perf_counter()

        t_adjoint_total   = 0.0
        t_transpose_total = 0.0

        queue = self.queue

        # ---------------- checks ----------------
        assert y_gpu.shape == (self.N_rot, self.Nx, self.N_chi * self.N_theta)
        assert y_gpu.flags.c_contiguous
        assert y_gpu.queue is queue

        assert out_x.shape == (self.Nx, self.Ny, self.K_sum)
        assert out_x.flags.f_contiguous
        assert out_x.queue is queue

        # ---------------- zero output ----------------
        out_x.fill(0)
        queue.finish()

        # ---------------- allocate intermediate (sino-space adjoint result) ----------------
        O = self.N_rot
        D = self.Nx

        if not hasattr(self, "_x_full_gpu"):
            x_full_gpu = clarray.empty(
                queue,
                (D, O, self.K_sum),
                dtype=np.float32,
                order="F",
            )

            self._x_full_gpu = x_full_gpu


        x_full_gpu = self._x_full_gpu
        x_full_gpu.fill(0)


        # ---------------- loop over materials ----------------
        for i_mat in range(self.N_mat):

            K_i = self.K_list[i_mat]
            total = O * D * K_i

            # ---- adjoint pole-figure operator ----
            t0 = time.perf_counter()
            xin_gpu = self.adjoint_gpu_opencl(y_gpu, i_mat)  # (ω, d, K_i), C
            queue.finish()
            t_adjoint_total += time.perf_counter() - t0

            # ---- transpose (ω, d, K)[C] → (d, ω, K)[F] ----
            xin_gpu_t = clarray.empty(
                queue,
                (D, O, K_i),
                dtype=np.float32,
                order="F",
            )

            t0 = time.perf_counter()
            self.k.transpose_omega_d_k_c_to_d_omega_k_f(
                queue,
                (total,),
                None,
                xin_gpu.data,
                xin_gpu_t.data,
                np.int32(O),
                np.int32(D),
                np.int32(K_i),
                np.int32(total),
            )
            queue.finish()
            t_transpose_total += time.perf_counter() - t0

            del xin_gpu

            # ---- scatter into full K axis ----
            i0 = self.offsets[i_mat]
            total = D * O * K_i

            self.k.scatter_k_lastaxis_f(
                queue,
                (total,),
                None,
                x_full_gpu.data,
                xin_gpu_t.data,
                np.int32(D),
                np.int32(O),
                np.int32(self.K_sum),
                np.int32(i0),
                np.int32(K_i),
                np.int32(total),
            )
            queue.finish()

            del xin_gpu_t

        # ---------------- X-ray backprojection ----------------
        gratopy.backprojection(
            x_full_gpu,
            self.PS,
            img=out_x
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

                

    def adjoint(self, y_gpu):
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

        x_gpu = clarray.zeros(
            self.queue,
            (self.Nx, self.Ny, self.K_sum),
            dtype=np.float32,
            order="F",
        )

        self.adjoint_cl(y_gpu, x_gpu)
        return x_gpu

                






    def adjoint_gpu_opencl(self, data_gpu: clarray.Array, i_mat: int) -> clarray.Array:
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
        out_gpu : clarray.Array
            Shape (R, Mx, K_i), C-order
        """
        queue = self.queue

        # ---------------- constants / shapes ----------------
        R = int(self.N_rot)
        Mx = int(self.Nx)
        K_i = int(self.K_list[i_mat])

        Nfull = int(self.N_chi * self.N_theta)
        idx_gpu = self.full_idx_gpu_list[i_mat]
        Nsub = int(idx_gpu.size)

        # ---------------- PF GPU inputs (prepared in __init__) ----------------
        coords_gpu   = self.pf_coords_gpu_list[i_mat]     # (R, C, P, 3)
        grid_inv_gpu = self.pf_grid_inv_gpu_list[i_mat]   # (K_i, 9)
        sigma_cpu = self.sigma_cpu_list[i_mat]
        sym_ops_gpu  = self.pf_sym_ops_gpu_list[i_mat]    # (G, 9)
        h_gpu        = self.pf_h_gpu_list[i_mat]          # (P, 3)

        C = self.N_chi
        P = self.N_peaks_list[i_mat]
        G = len(self.pf_sym_ops_cpu_list[i_mat])

        # Intensities cached on GPU per material: (P,)
        intensity_gpu = self.pf_intensity_gpu_list[i_mat]
        assert int(intensity_gpu.size) == P

        # ---------------- 1) gather masked detector segments once ----------------
        data_gpu_sub = clarray.empty(queue, (R, Mx, Nsub), dtype=np.float32, order="C")

        total_gather = np.int32(R) * np.int32(Mx) * np.int32(Nsub)
        self.k.gather_kernel(
            queue,
            (int(total_gather),),
            None,
            data_gpu.data,
            data_gpu_sub.data,
            idx_gpu.data,
            np.int32(R),
            np.int32(Mx),
            np.int32(Nsub),
            np.int32(Nfull),
        )

        # ---------------- output (R, Mx, K_i) ----------------
        out_gpu = clarray.empty(queue, (R, Mx, K_i), dtype=np.float32, order="C")
        out_gpu.fill(0.0)

        # ---------------- batching plan over K ----------------
        batches = self.get_pf_batches_for_material(i_mat)  # list of dicts

        # ---------------- batch loop ----------------
        for b in batches:
            k0 = int(b["k_start"])
            k1 = int(b["k_end"])
            Kb = int(b["K_batch"])
            assert Kb == (k1 - k0)

            # ---- 2) slice grid_inv -> (Kb, 9) ----
            grid_inv_batch = self._slice_gridinv_k_batch(grid_inv_gpu, k0, k1)
            sigma_cpu_batch = sigma_cpu[k0:k1]

            # ---- 3) PF basis batch: (R, Kb, C, P) ----
            pf_basis_batch = clarray.empty(queue, (R, Kb, C, P), dtype=np.float32, order="C")

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
                out_gpu=pf_basis_batch,
            )

            # ---- 4) scale by intensities in-place ----
            # pf_basis_batch[r,k,c,p] *= intensity[p]
            self._scale_pf_by_intensity_inplace(pf_basis_batch, intensity_gpu)

            # ---- 5) convolve -> B_gpu_batch (R, Kb, Nsub) ----
            # (this is your rewritten version that accepts batches)
            B_gpu_batch = self.convolve_matrix_from_pf_batch(
                i_mat=i_mat,
                pf_basis_batch=pf_basis_batch,  # (R,Kb,C,P)
            )  # expects (R, Kb, Nsub) C-order

            # ---- 6) transpose B for adjoint GEMM: BT = (R, Nsub, Kb) ----
            BT_gpu_batch = clarray.empty(queue, (R, Nsub, Kb), dtype=np.float32, order="C")
            total_bt = np.int32(R) * np.int32(Kb) * np.int32(Nsub)

            self.k.btranspose_kernel(
                queue,
                (int(total_bt),),
                None,
                B_gpu_batch.data,
                BT_gpu_batch.data,
                np.int32(R),
                np.int32(Kb),
                np.int32(Nsub),
                np.int32(total_bt),
            )

            # ---- 7) adjoint GEMM: x_batch = data_sub * BT ----
            # data_gpu_sub:   (R, Mx,   Nsub)
            # BT_gpu_batch:   (R, Nsub, Kb)
            # x_batch:        (R, Mx,   Kb)
            x_batch = clarray.empty(queue, (R, Mx, Kb), dtype=np.float32, order="C")
            batched_gemm_adj_clblast(queue, data_gpu_sub, BT_gpu_batch, x_batch, R, Mx, Nsub, Kb)

            # ---- 8) scatter x_batch into out_gpu[:, :, k0:k1] ----
            total_scatter = np.int32(R) * np.int32(Mx) * np.int32(Kb)
            self.k.scatter_k_batch_c(
                queue,
                (int(total_scatter),),
                None,
                out_gpu.data,
                x_batch.data,
                np.int32(R),
                np.int32(Mx),
                np.int32(K_i),
                np.int32(k0),
                np.int32(Kb),
                np.int32(total_scatter),
            )

            # cleanup batch temporaries
            del grid_inv_batch, pf_basis_batch, B_gpu_batch, BT_gpu_batch, x_batch

        return out_gpu




    def forward_gpu_opencl(self, out_gpu_full, coeffs_gpu, i_mat: int):
        """
        Batched OpenCL forward operator for one material.

        out_gpu_full : (R, Mx, Nseg_full) C-order, accumulated in-place
        coeffs_gpu   : (R, Mx, K_i)       C-order (already per-material coeffs)
        i_mat        : material index
        """
        queue = self.queue

        # ---------------- constants / shapes ----------------
        R = int(self.N_rot)
        Mx = int(self.Nx)
        K_i = int(self.K_list[i_mat])

        # idx for scattering sub-segments (theta masked)
        idx_gpu = self.full_idx_gpu_list[i_mat]
        Nsub = int(idx_gpu.size)

        # Reuse output buffer (R, Mx, Nsub)
        out_sub = self._out_sub[i_mat]

        # ---------------- PF GPU inputs (prepared in __init__) ----------------
        # coords_gpu:   (R, C, P, 3) float32 C-order
        # grid_inv_gpu: (K_i, 9)     float32 C-order
        # sym_ops_gpu:  (G, 9)       float32 C-order
        # h_gpu:        (P, 3)       float32 C-order   (P == number of peaks/hkls)
        coords_gpu   = self.pf_coords_gpu_list[i_mat]
        grid_inv_gpu = self.pf_grid_inv_gpu_list[i_mat]
        sigma_cpu = self.sigma_cpu_list[i_mat]
        sym_ops_gpu  = self.pf_sym_ops_gpu_list[i_mat]
        h_gpu        = self.pf_h_gpu_list[i_mat]

        # Sanity checks (cheap and saves pain)
        assert int(coords_gpu.shape[0]) == R
        assert int(coords_gpu.shape[1]) == int(self.N_chi)
        assert int(coords_gpu.shape[2]) == int(self.N_peaks_list[i_mat])
        assert int(coords_gpu.shape[3]) == 3
        assert int(grid_inv_gpu.shape[0]) == K_i
        assert int(grid_inv_gpu.shape[1]) == 9

        C = int(self.N_chi)
        P = int(self.N_peaks_list[i_mat])
        G = int(len(self.pf_sym_ops_cpu_list[i_mat]))

        # Intensities should be cached on GPU per material
        intensity_gpu = self.pf_intensity_gpu_list[i_mat]  # (P,) float32 GPU
        assert int(intensity_gpu.size) == P

        # ---------------- batching plan over K ----------------
        batches = self.get_pf_batches_for_material(i_mat)  # list of dicts

        # ---------------- batch loop ----------------
        for b in batches:
            k0 = int(b["k_start"])
            k1 = int(b["k_end"])
            Kb = int(b["K_batch"])
            assert Kb == (k1 - k0)

            # ---- 1) slice coeffs_gpu -> (R, Mx, Kb) ----
            coeffs_batch = self._slice_coeffs_k_batch(coeffs_gpu, k0, k1)

            # ---- 2) slice grid_inv -> (Kb, 9) ----
            grid_inv_batch = self._slice_gridinv_k_batch(grid_inv_gpu, k0, k1)

            sigma_cpu_batch = sigma_cpu[k0:k1]

            # ---- 3) PF basis batch: (R, Kb, C, P) ----
            pf_basis_batch = clarray.empty(queue, (R, Kb, C, P), dtype=np.float32, order="C")

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
                out_gpu=pf_basis_batch,
            )

            # ---- 4) scale by intensities: pf_basis_batch[r,k,c,p] *= intensity[p] ----
            self._scale_pf_by_intensity_inplace(pf_basis_batch, intensity_gpu)

            # ---- 5) CONVOLUTION----
            B_gpu_batch = self.convolve_matrix_from_pf_batch(
                i_mat=i_mat,
                pf_basis_batch=pf_basis_batch,
            )

            # ---- 6) GEMM and accumulate GO AFTER convolution ----
            out_sub.fill(0.0)
            batched_gemm_clblast(queue, coeffs_batch, B_gpu_batch, out_sub, R=R, M=Mx, K=Kb, N=Nsub)


            # ---- 7) accumulate out_sub into out_gpu_full ----
            total = np.int32(R) * np.int32(Mx) * np.int32(Nsub)
            self.k.accumulate_kernel(
                queue,
                (int(total),),
                None,
                out_gpu_full.data,
                out_sub.data,
                idx_gpu.data,
                np.int32(R),
                np.int32(Mx),
                np.int32(Nsub),
                np.int32(out_gpu_full.shape[2]),
                total,
            )
            del pf_basis_batch, B_gpu_batch, coeffs_batch, grid_inv_batch



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