import numpy as np
import gc


import pyopencl as cl
import pyopencl.array as clarray
import gratopy
from pyclblast import gemmStridedBatched


from scipy.spatial.transform import Rotation as R
from .create_pfo_matrix import build_pf_program
from .pfo_kernels import build_all_opencl




class PFO_OPENCL_BATCHED:

    def __init__(
        self,
        cfg: None,
        materials: None,
        grids: None,
        two_thetas: None,
        max_gb: None,
        verbose: bool = False,
        ctx=None,
        queue=None,
    ):

        # --- context / queue ---
        if ctx is not None and queue is not None:
            self.ctx = ctx
            self.queue = queue
        else:
            self.ctx = cl.create_some_context(interactive=False)
            self.queue = cl.CommandQueue(self.ctx)

        

        self.cfg = cfg
        self.materials = materials
        self.grids = grids
        self.two_thetas = np.array(two_thetas).astype(np.float32)
        self.peak_width = self.cfg['peak_width']
        self.verbose = verbose

        self.N_chi = self.cfg['N_chi']
        self.N_theta = len(two_thetas)
        self.N_seg = self.N_theta * self.N_chi
        self.Nx = self.cfg['Nx']
        self.Ny = self.cfg['Ny']
        self.N_rot = self.cfg['N_rot']
        self.angle_range = np.array(self.cfg['angle_range'])/180*np.pi
        self.angles = np.linspace(self.angle_range[0], self.angle_range[1], self.N_rot, endpoint=False)
        self.N_mat = len(self.materials)
        self.pf_batch_max_gb = float(max_gb)



        # --- build kernels ---
        self.prg, self.k, self.pf_prg = build_all_opencl(self.ctx, ts=16)
        self.pf_prg = build_pf_program(self.ctx)
        self.pfmatrix_eval_kernel = cl.Kernel(self.pf_prg, "pfmatrix_eval")


        self.transfer_material_parameters_to_gpu()
        self.detector_coordinates()
        self.transfer_grid_parameters_to_gpu()
        self.get_pf_batches_for_material()

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


        self.allocate_coefficient_buffer()


    def detector_coordinates(self, integration_samples=1, full_circle_covered=True):
        """ Calculates and returns the probed polar and azimuthal coordinates on the unit sphere at
        each angle of projection and for each detector segment in the system's geometry.
        """
        wavelength_angstrom = 12.398 / self.cfg["wavelength"]
        self.two_theta_peaks = 2.0 * np.arcsin(
            np.linalg.norm(self.h_cpu, axis=1) / (4.0 * np.pi) * wavelength_angstrom
        ).astype(np.float32)

        coords_list = []
        for tt in self.two_theta_peaks:

            probed_directions_zero_rot = np.zeros((self.N_chi, integration_samples, 3))
            # Impose symmetry if needed.
            if not full_circle_covered:
                shift = np.pi
            else:
                shift = 0
            det_bin_middles_extended = np.linspace(0, 2*np.pi, self.N_chi, endpoint=False)
            det_bin_middles_extended = np.insert(det_bin_middles_extended, 0, det_bin_middles_extended[-1] + shift)
            det_bin_middles_extended = np.append(det_bin_middles_extended, det_bin_middles_extended[1] + shift)

            for ii in range(self.N_chi):

                # Check if the interval from the previous to the next bin goes over the -pi +pi discontinuity
                before = det_bin_middles_extended[ii]
                now = det_bin_middles_extended[ii + 1]
                after = det_bin_middles_extended[ii + 2]

                if abs(before - now + 2 * np.pi) < abs(before - now):
                    before = before + 2 * np.pi
                elif abs(before - now - 2 * np.pi) < abs(before - now):
                    before = before - 2 * np.pi

                if abs(now - after + 2 * np.pi) < abs(now - after):
                    after = after - 2 * np.pi
                elif abs(now - after - 2 * np.pi) < abs(now - after):
                    after = after + 2 * np.pi

                # Generate a linearly spaced set of angles covering the detector segment
                start = 0.5 * (before + now)
                end = 0.5 * (now + after)
                inc = (end - start) / integration_samples
                angles = np.linspace(start + inc / 2, end - inc / 2, integration_samples)

                # Make the zero-rotation-frame vectors corresponding to the given angles
                probed_directions_zero_rot[ii, :, :] = np.cos(angles[:, np.newaxis]) * \
                    np.array(self.cfg["detector_direction_origin"])[np.newaxis,:]

                probed_directions_zero_rot[ii, :, :] += np.sin(angles[:, np.newaxis]) * \
                    np.array(self.cfg["detector_direction_positive_90"])[np.newaxis,:]

            twothetahalf = tt/2

            probed_directions_zero_rot = +probed_directions_zero_rot * np.cos(twothetahalf)\
                - np.sin(twothetahalf) * np.array(self.cfg['p_direction_0'])
            probed_direction_vectors = np.zeros((self.N_rot, self.N_chi, integration_samples, 3), dtype=np.float64)
            k0 = np.asarray(self.cfg["k_direction_0"])
            Rmats = R.from_rotvec(self.angles[:, None] * k0).as_matrix()
            probed_direction_vectors[...] = \
                np.einsum('kij,mli->kmlj', Rmats, probed_directions_zero_rot)

            coords = probed_direction_vectors[:,:,0,:]
            coords_list.append(coords)

        coords_cpu = np.stack(coords_list, axis=-1)
        coords_cpu = coords_cpu.transpose((0, 1, 3, 2))
        self.coords_cpu = np.asarray(coords_cpu, dtype=np.float32, order="C")
        self.coords_gpu = clarray.to_device(self.queue, self.coords_cpu)




    def transfer_material_parameters_to_gpu(self):
        self.pf_h_gpu_normed_list = []
        self.pf_h_cpu_normed_list = []
        self.pf_h_cpu_list = []
        self.pf_intensity_gpu_list = []
        self.pf_intensity_cpu_list = []
        self.pf_sym_ops_gpu_list = []
        self.pf_sym_ops_cpu_list = []
        self.N_peaks_list = []
        self.peak_positions_np_list = []

        for mat in self.materials:
            # ---- normalized reciprocal lattice vectors ----
            h_cpu_normed = np.asarray(mat.h_vecs_normed, dtype=np.float32, order="C")
            h_cpu = np.asarray(mat.h_vecs, dtype=np.float32, order="C")
            h_gpu_normed = clarray.to_device(self.queue, h_cpu_normed)
            self.pf_h_gpu_normed_list.append(h_gpu_normed)
            self.pf_h_cpu_normed_list.append(h_cpu_normed)
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

        self.grid_inv_gpu_list = []
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
            self.grid_inv_gpu_list.append(grid_inv_gpu)

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

        self._basis_batch_list = []
        for i_mat in range(self.N_mat):
            self._basis_batch_list.append(
                _alloc(
                    f"_basis_batch[{i_mat}]",
                    (self.N_rot, self.K_batch_max, self.N_chi, self.N_peaks_list[i_mat]),
                    np.float32,
                    "C",
                    )
            )



        self._basis_batch_convolved = _alloc(
            "_basis_batch_convolved",
            (self.N_rot, self.K_batch_max, self.N_seg),
            np.float32,
            "C",
        )


        self._basis_batch_convolved_T = _alloc(
            "_basis_batch_convolvedT",
            (self.N_rot, self.N_seg, self.K_batch_max),
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
        """
        Free OpenCL buffers allocated by allocate_coefficient_buffer().
        Handles both single buffers and the per-material list buffers.
        """

        # 1) Release list buffers (_basis_batch_list)
        if hasattr(self, "_basis_batch_list"):
            try:
                for arr in getattr(self, "_basis_batch_list", []) or []:
                    try:
                        if getattr(arr, "base_data", None) is not None:
                            arr.base_data.release()
                    except Exception:
                        pass
            finally:
                try:
                    delattr(self, "_basis_batch_list")
                except Exception:
                    pass

        # 2) Release named buffers
        buffer_names = [
            "coeffs_sino_F",
            "coeffs_sino_C",
            "_coeffs_batch_F",
            "_basis_batch_convolved",
            "_basis_batch_convolved_T",
            "_grid_inv_kmax",
            "_inv_sigma2_kmax",
            "_norm_factor_kmax",
        ]

        for name in buffer_names:
            arr = getattr(self, name, None)
            if arr is None:
                continue

            try:
                if getattr(arr, "base_data", None) is not None:
                    arr.base_data.release()
            except Exception:
                pass

            try:
                delattr(self, name)
            except Exception:
                pass

        # 3) Optional bookkeeping
        if hasattr(self, "total_bytes"):
            try:
                delattr(self, "total_bytes")
            except Exception:
                pass

        # 4) Ensure queue is done, then collect
        try:
            if getattr(self, "queue", None) is not None:
                self.queue.finish()
        except Exception:
            pass

        gc.collect()

        if getattr(self, "verbose", False):
            print("OpenCL GPU memory for operator freed.")



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



        # ---- dimensions ----
        R = self.N_rot
        C = self.N_chi
        T = self.N_theta   # masked theta count
        K_largest = np.max(np.array(self.K_list))

        bytes_per_float = 4

        # Memory per K after convolution:
        # (R, C, T) per K
        bytes_per_K = R * C * T * bytes_per_float

        max_bytes = self.pf_batch_max_gb * (1024 ** 3)

        # At least one K per batch
        if bytes_per_K >0.01:
            K_batch_max = max(int(max_bytes // bytes_per_K), 1)
             # Safety: never exceed available K
            K_batch_max = min(K_batch_max, K_largest)
        else:
            K_batch_max = 1

        self.K_batch_max = K_batch_max


        self.batches_list = []

        for i_mat in range(self.N_mat):
            K_total_material = self.K_list[i_mat]
            batches = []

            k0 = 0
            while k0 < K_total_material:
                k1 = min(k0 + K_batch_max, K_total_material)
                batches.append({
                    "k_start": k0,
                    "k_end": k1,
                    "K_batch": k1 - k0,
                })
                k0 = k1

            self.batches_list.append(batches)
        

    
    

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
        basis_batch: clarray.Array,
        basis_convolved: clarray.Array,
        i_mat: int,
    ):
        queue = self.queue

        # ---------------- shapes ----------------
        R  = int(self.N_rot)
        C  = int(self.N_chi)
        Kmax = int(self.K_batch_max)
        P  = int(self.N_peaks_list[i_mat])
        T = int(self.N_theta)
        two_thetas = self.two_thetas

        # ---------------- Gaussian weights (CPU → GPU) ----------------
        # peak_positions_np_list[i_mat]: (P,)
        peaks_np = np.asarray(
            self.peak_positions_np_list[i_mat],
            dtype=np.float32
        )

        dt = float(self.two_thetas[1] - self.two_thetas[0]) # Linear spacing
        inv_norm = 1.0 / np.sqrt(2.0 * np.pi * (self.peak_width ** 2))

        # diff: (P, T)
        diff = peaks_np[:, None] - two_thetas[None, :]
        gaussian_np = (
            inv_norm
            * np.exp(-0.5 * (diff / self.peak_width) ** 2)
            * dt
        ).astype(np.float32)

        gaussian_gpu = clarray.to_device(queue, gaussian_np)

        # ---------------- output buffer ----------------
        # (R, Kb, C, T)

        # ---------------- kernel launch ----------------
        total = (R * Kmax * C * T,)

        self.k.expand_kernel(
            queue,
            total,
            None,
            basis_batch.data,     # basis
            gaussian_gpu.data,       # gaussian
            basis_convolved.data,            # out
            np.int32(R),
            np.int32(Kmax),
            np.int32(C),
            np.int32(P),
            np.int32(T),
        )



    def set_peak_width(self, peak_width):
        self.peak_width = peak_width



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
            (self.N_rot, self.Nx, self.N_chi * self.N_theta),
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
            Shape (Nx, Nx, K_sum), Fortran order
        out_y : clarray
            Shape (N_rot, Nx, N_seg), C order
            Accumulated into (will be zeroed here)
        """



        # --- zero output (important!) ---
        assert coeffs.flags.f_contiguous
        assert coeffs.shape == (self.Nx, self.Ny, self.K_sum)
        assert data.flags.c_contiguous
        assert data.shape == (self.N_rot, self.Nx, self.N_seg)
        
        data.fill(0)
        R = int(self.N_rot)
        C = int(self.N_chi)
        T = int(self.N_theta)
        Kmax = self.K_batch_max
        Nx = self.Nx
        Ny = self.Ny
        N_rot = self.N_rot

        for i_mat in range(self.N_mat):
            P = int(self.N_peaks_list[i_mat])
            G = int(len(self.pf_sym_ops_cpu_list[i_mat]))
            coords_gpu   = self.pf_coords_gpu_list[i_mat]
            sym_ops_gpu  = self.pf_sym_ops_gpu_list[i_mat]
            h_gpu_normed        = self.pf_h_gpu_normed_list[i_mat]
            intensity_gpu = self.pf_intensity_gpu_list[i_mat]
            i0 = self.offsets[i_mat] # The index for the first index of the material
            sigma_cpu = self.sigma_cpu_list[i_mat]
            grid_inv_gpu = self.grid_inv_gpu_list[i_mat]

            batches = self.batches_list[i_mat]

            for b in batches:
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
                    np.int32(self.K_sum),
                    np.int32(Kb),
                    np.int32(k0 + i0),
                )

                gratopy.forwardprojection(
                    self._coeffs_batch_F,
                    self.PS,
                    sino=self.coeffs_sino_F,
                )

                total = N_rot * Nx * Kmax
                self.k.transpose_d_omega_k_f_to_c(
                    self.queue,
                    (total,),
                    None,
                    self.coeffs_sino_F.data,
                    self.coeffs_sino_C.data,
                    np.int32(Nx),
                    np.int32(N_rot),
                    np.int32(Kmax),
                    np.int32(total),
                )


                self._norm_factor_kmax.fill(0.0) # make sure that this is set to zero!
                self._inv_sigma2_kmax.fill(0.0)
                sigma = np.asarray(sigma_cpu[k0:k1], dtype=np.float32)
                inv_sigma2_cpu = (1.0 / (sigma * sigma)).astype(np.float32)
                norm_factor_cpu = (1.0 / (8.0 * np.pi * sigma * sigma)).astype(np.float32)
                cl.enqueue_copy(self.queue, self._inv_sigma2_kmax.data, inv_sigma2_cpu, device_offset=0)
                cl.enqueue_copy(self.queue, self._norm_factor_kmax.data, norm_factor_cpu, device_offset=0)


                total = Kb * 9
                self.k.SLICE_GRIDINV_K_BATCH(
                    self.queue,
                    (total,),
                    None,
                    grid_inv_gpu.data,      # input: (K_total, 9)
                    self._grid_inv_kmax.data,    # output: (K_batch_max, 9)
                    np.int32(self.K_list[i_mat]),            # K_IN  (total grid size)
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
                    self._basis_batch_list[i_mat].data,
                    np.int32(R),
                    np.int32(Kmax),
                    np.int32(C),
                    np.int32(P),
                    np.int32(G),
                )

                self._scale_pf_by_intensity_inplace(self._basis_batch_list[i_mat], intensity_gpu)

                # Convolve
                self.convolve_matrix_from_pf_batch(
                    self._basis_batch_list[i_mat],
                    self._basis_batch_convolved,
                    i_mat=i_mat,
                )

                batched_gemm_clblast(self.queue, self.coeffs_sino_C, self._basis_batch_convolved, data, R=R, M=Nx, K=Kmax, N=T*C)


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
            (self.Nx, self.Ny, self.K_sum),
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
        y_gpu : clarray
            Shape (N_rot, Nx, N_seg), C-order
        out_x : clarray
            Shape (Nx, Nx, K_sum), Fortran-order
            Will be overwritten
        """

                # --- zero output (important!) ---
        assert coeffs.flags.f_contiguous
        assert coeffs.shape == (self.Nx, self.Ny, self.K_sum)
        assert data.flags.c_contiguous
        assert data.shape == (self.N_rot, self.Nx, self.N_seg)

        coeffs.fill(0)
        R = int(self.N_rot)
        C = int(self.N_chi)
        T = int(self.N_theta)
        Kmax = self.K_batch_max
        Nx = self.Nx
        Ny = self.Ny

        for i_mat in range(self.N_mat):
            P = int(self.N_peaks_list[i_mat])
            G = int(len(self.pf_sym_ops_cpu_list[i_mat]))
            coords_gpu   = self.pf_coords_gpu_list[i_mat]
            sym_ops_gpu  = self.pf_sym_ops_gpu_list[i_mat]
            h_gpu_normed        = self.pf_h_gpu_normed_list[i_mat]
            intensity_gpu = self.pf_intensity_gpu_list[i_mat]
            i0 = self.offsets[i_mat] # The index for the first index of the material
            sigma_cpu = self.sigma_cpu_list[i_mat]
            grid_inv_gpu = self.grid_inv_gpu_list[i_mat]

            batches = self.batches_list[i_mat]

            for b in batches:
                k0 = b["k_start"]
                k1 = b["k_end"]
                Kb = b["K_batch"]

                self._coeffs_batch_F.fill(0.0)
                self.coeffs_sino_C.fill(0.0)
                self.coeffs_sino_F.fill(0.0)


                self._norm_factor_kmax.fill(0.0) # make sure that this is set to zero!
                self._inv_sigma2_kmax.fill(0.0)
                sigma = np.asarray(sigma_cpu[k0:k1], dtype=np.float32)
                inv_sigma2_cpu = (1.0 / (sigma * sigma)).astype(np.float32)
                norm_factor_cpu = (1.0 / (8.0 * np.pi * sigma * sigma)).astype(np.float32)
                cl.enqueue_copy(self.queue, self._inv_sigma2_kmax.data, inv_sigma2_cpu, device_offset=0)
                cl.enqueue_copy(self.queue, self._norm_factor_kmax.data, norm_factor_cpu, device_offset=0)


                total = Kb * 9
                self.k.SLICE_GRIDINV_K_BATCH(
                    self.queue,
                    (total,),
                    None,
                    grid_inv_gpu.data,      # input: (K_total, 9)
                    self._grid_inv_kmax.data,    # output: (K_batch_max, 9)
                    np.int32(self.K_list[i_mat]),            # K_IN  (total grid size)
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
                    self._basis_batch_list[i_mat].data,
                    np.int32(R),
                    np.int32(Kmax),
                    np.int32(C),
                    np.int32(P),
                    np.int32(G),
                )

                self._scale_pf_by_intensity_inplace(self._basis_batch_list[i_mat], intensity_gpu)

                # Convolve
                self.convolve_matrix_from_pf_batch(
                    self._basis_batch_list[i_mat],
                    self._basis_batch_convolved,
                    i_mat=i_mat,
                )

                ## transpose the convolved matrix
                total = R*Kmax*C*T
                self.k.btranspose_kernel(
                    self.queue,
                    (total,),
                    None,
                    self._basis_batch_convolved.data,
                    self._basis_batch_convolved_T.data,
                    np.int32(R),
                    np.int32(Kmax),
                    np.int32(C*T),
                    np.int32(total),
                )

                batched_gemm_adj_clblast(self.queue, data, self._basis_batch_convolved_T, self.coeffs_sino_C, R, Nx, C*T, Kmax, R/np.pi)


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
                    np.int32(self.K_sum),
                    np.int32(i0+k0),
                    np.int32(Kb),
                    np.int32(total),
                )




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
        beta=1.0, # 1.0=accumulate, 0.0=overwrite
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
        beta=0.0, # 1.0=accumulate, 0.0=overwrite
        a_transp=False,
        b_transp=False, 
    )





def estimate_L_power(
    op,
    niter: int = 20,
    seed: int = 0,
    eps: float = 1e-30,
    verbose: int = 1,
) -> float:
    if niter < 1:
        raise ValueError("niter must be >= 1")

    q = op.queue
    rng = np.random.default_rng(seed)

    # x in domain, Fortran
    x = clarray.empty(q, (op.Nx, op.Ny, op.K_sum), np.float32, order="F")
    # y in range, C
    Ax = clarray.empty(q, (op.N_rot, op.Nx, op.N_seg), np.float32, order="C")
    # z = A^*Ax in domain
    z = clarray.empty(q, x.shape, np.float32, order="F")

    # init x random
    x_host = rng.standard_normal(x.shape).astype(np.float32, copy=False, order="F")
    assert x.data is not None
    cl.enqueue_copy(q, x.data, x_host)
    q.finish()

    # normalize x
    xnorm = float(np.sqrt(clarray.vdot(x, x).get()) + eps)
    x *= np.float32(1.0 / xnorm)
    q.finish()

    L_est: float = 0.0
    for it in range(niter):
        op.direct_cl(x, Ax)
        op.adjoint_cl(Ax, z)
        q.finish()

        num = float(clarray.vdot(x, z).get())
        den = float(clarray.vdot(x, x).get()) + eps
        L_est = num / den

        znorm = float(np.sqrt(clarray.vdot(z, z).get()) + eps)
        x[:] = z * np.float32(1.0 / znorm)
        q.finish()

        if verbose:
            print(f"[power {it+1:02d}] L_est={L_est:.6e}  ||z||={znorm:.6e}")

    # ---------- GPU cleanup ----------
    if x.base_data is not None:
        x.base_data.release()
    if Ax.base_data is not None:
        Ax.base_data.release()
    if z.base_data is not None:
        z.base_data.release()

    del x, Ax, z, x_host
    gc.collect()
    q.finish()
    # --------------------------------

    return float(L_est)
