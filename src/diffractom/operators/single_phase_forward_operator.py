from __future__ import annotations
import numpy as np
import time
import gc
from typing import Any, Mapping
import pyopencl as cl
import pyopencl.array as clarray
import gratopy
from pyclblast import gemmStridedBatched
from ..utils.grid import Grid
from ..crystallography.material import Material
from scipy.spatial.transform import Rotation as R
from .create_pfo_matrix import build_pf_program
from .pf_kernels import build_all_opencl
from .parallel_radon import ParallelRadon
from ..utils.support import fov_support_mask


class SinglePhaseForwardOperator:
    """Single-material pole-figure forward operator on GPU.

    Computes  data = PF @ Radon(coeffs)  and its adjoint, where PF is the
    pole-figure transform and Radon is the parallel-beam projection.
    All heavy computation runs on OpenCL.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        material: Material,
        grid: Grid,
        max_gb: float,
        verbose: bool = False,
        normalized: bool = False,
        ctx: cl.Context | None = None,
        queue: cl.CommandQueue | None = None,
        pf_mode: str = "auto",
        sparse_max_fill: float = 0.1,
        sparse_max_gb: float | None = None,
        projector: str = "native",
        **kwargs,
    ):
        """Initialise the single-material forward operator.

        Parameters
        ----------
        cfg : dict
            Experiment configuration (keys: Nx, Ny, My, N_Omega, N_eta, angle_range,
            wavelength, detector_direction_origin, etc.).
        material : Material
            Crystallographic material with reflections and symmetry.
        grid : Grid
            Orientation discretisation tree.
        max_gb : float
            GPU memory budget (GB) for K-batching.
        verbose : bool
            Print buffer allocation summary.
        normalized : bool
            If True, skip intensity scaling of the PF matrix.
        ctx, queue : optional
            Existing OpenCL context/queue; created automatically if None.
        pf_mode : {"auto", "sparse", "dense"}
            How the pole-figure (PF) matrix is applied.

            * ``"sparse"``: the PF matrix is evaluated once here, stored in a
              sparse (CSR) format, and applied with sparse kernels. Its entries
              are exactly zero away from the poles, so this is the same operator.
            * ``"dense"``: the PF matrix is applied with a dense batched GEMM.
              If all orientations fit in one batch (``max_gb``), it is evaluated
              once and reused; otherwise it is re-evaluated batch by batch in
              every call.
            * ``"auto"`` (default): sparse, unless the fill fraction of the first
              batch exceeds ``sparse_max_fill`` or the sparse matrix would not fit
              in ``sparse_max_gb``; dense otherwise.

            Measured on an A100 (simulated Al data, 360 omega x 180 eta x 8
            rings): sparse was 1.4-2.6x faster per FISTA iteration than dense for
            fill fractions of 0.2 % (sigma = 0.4 deg) and 5 % (sigma = 2 deg).
        sparse_max_fill : float
            Largest fill fraction for which ``pf_mode="auto"`` chooses sparse.
            The default, 10 %, is roughly where the two break even.
        sparse_max_gb : float, optional
            GPU memory budget (GB) for the sparse matrix, estimated from the first
            batch. Default (``default_sparse_budget_gb``): half of the memory left
            after reserving what a FISTA reconstruction needs besides the operator
            (5 coefficient-sized and 4 data-sized arrays), the operator's own
            buffers and a 1 GB margin. The solver's arrays take priority: a large
            sparse matrix (e.g. a dense uniform grid) is not stored, and the dense
            path re-evaluates the PF matrix batch by batch instead.
        projector : {"native", "gratopy"}
            Parallel-beam Radon transform used for the tomographic part.
            ``"native"`` (default) is diffractom's ParallelRadon, vectorised over
            the orientations; ``"gratopy"`` uses gratopy. Both use the same
            discretisation (they agree to float32 rounding); the native one is
            2-8x faster for many orientations.
        **kwargs
            Override any cfg key (e.g. ``N_Omega=50``).
        """

        # --- context / queue ---
        if ctx is not None and queue is not None:
            self.ctx = ctx
            self.queue = queue
        else:
            self.ctx = cl.create_some_context(interactive=False)
            self.queue = cl.CommandQueue(self.ctx)

        
        # Keyword arguments override cfg values (e.g. N_Omega=50 overrides cfg['N_Omega'])
        self.cfg = {**cfg, **kwargs}
        self.normalized = normalized
        self.material = material
        self.grid = grid
        self.verbose = verbose
        self.N_eta = self.cfg['N_eta']
        # reflections with the same two-theta form one ring (one data channel)
        self.ring_reflections = group_reflections_into_rings(self.material)
        self.N_peaks = len(self.ring_reflections)  # number of rings
        self.N_seg = self.N_peaks * self.N_eta
        self.Nx = self.cfg['Nx']
        self.Ny = self.cfg['Ny']
        self.My = self.cfg['My']
        self.N_Omega = self.cfg['N_Omega']
        self.cor_offset = self.cfg['cor_offset']
        self.N_Omega_subdivisions = self.cfg.get('N_Omega_subdivisions', 1)
        self.angle_range = np.array(self.cfg['angle_range'])/180*np.pi
        delta = (self.angle_range[1] - self.angle_range[0]) / self.N_Omega
        sub_delta = delta / self.N_Omega_subdivisions
        # Coarse angles centered in their intervals (for gratopy projection)
        self.angles = np.linspace(self.angle_range[0], self.angle_range[1], self.N_Omega, endpoint=False) + delta / 2
        # Fine angles centered in sub-intervals (for PF coordinate generation)
        self.angles_subdivided = np.linspace(self.angle_range[0], self.angle_range[1], self.N_Omega * self.N_Omega_subdivisions, endpoint=False) + sub_delta / 2

        # --- eta subdivisions ---
        self.N_eta_subdivisions = self.cfg.get('N_eta_subdivisions', 1)
        self.eta_angle_range = np.array(self.cfg.get('eta_angle_range', [0, 360])) / 180 * np.pi
        eta_delta = (self.eta_angle_range[1] - self.eta_angle_range[0]) / self.N_eta
        eta_sub_delta = eta_delta / self.N_eta_subdivisions
        # Coarse eta bin centres
        self.eta_angles = np.linspace(self.eta_angle_range[0], self.eta_angle_range[1], self.N_eta, endpoint=False) + eta_delta / 2
        # Fine eta sub-bin centres
        self.eta_angles_subdivided = np.linspace(self.eta_angle_range[0], self.eta_angle_range[1], self.N_eta * self.N_eta_subdivisions, endpoint=False) + eta_sub_delta / 2

        self.pf_batch_max_gb = float(max_gb)


        # --- build kernels ---
        self.prg, self.k, self.pf_prg = build_all_opencl(self.ctx, ts=16)
        self.pf_prg = build_pf_program(self.ctx)
        self.pfmatrix_eval_kernel = cl.Kernel(self.pf_prg, "pfmatrix_eval")
        self.pfpoles_kernel = cl.Kernel(self.pf_prg, "pfmatrix_eval_poles")


        self.transfer_material_parameters_to_gpu()
        self.detector_coordinates()
        self.transfer_grid_parameters_to_gpu()
        self.transfer_pole_axes_to_gpu()
        self.get_pf_batches_for_material()  # list of dicts



        # --- tomographic projector ---
        if projector not in ("native", "gratopy"):
            raise ValueError(f"projector must be 'native' or 'gratopy', not {projector!r}")
        self.projector = projector
        if projector == "native":
            # angle weight = angular width of a projection (for a 180 degree range this
            # equals gratopy's default weights, so both projectors agree)
            self.radon = ParallelRadon(
                self.queue,
                (self.Nx, self.Ny),
                self.angles,
                self.My,
                image_width=self.Nx,
                detector_width=self.My,
                detector_shift=self.cor_offset,
                angle_weights=delta,
                bins_per_item=3,
            )
            self.PS = None
        else:
            self.PS = gratopy.ProjectionSettings(
                self.queue,
                gratopy.PARALLEL,
                (self.Nx, self.Ny, self.K_batch_max),
                self.angles,
                n_detectors=self.My,
                image_width=self.Nx,
                detector_width=self.My,
                detector_shift=self.cor_offset,
            )
        assert self.queue.context.int_ptr == self.ctx.int_ptr


        # Allocate buffers
        self.allocate_coefficient_buffer()

        # --- PF matrix mode ---
        if pf_mode not in ("dense", "sparse", "auto"):
            raise ValueError(f"pf_mode must be 'dense', 'sparse' or 'auto', not {pf_mode!r}")
        self.pf_mode = pf_mode
        self._pf_cached = False  # dense mode, single batch: PF matrix already in _basis_batch_kmax
        self.sparse_batches = None
        if sparse_max_gb is None:
            sparse_max_gb = self.default_sparse_budget_gb()
        if pf_mode in ("sparse", "auto"):
            self.build_sparse_pf(auto=(pf_mode == "auto"), max_fill=sparse_max_fill, max_gb=sparse_max_gb)
        if self.verbose:
            cached = self.pf_mode == "sparse" or len(self.batches) == 1
            print(f"PF matrix: {self.pf_mode}, "
                  + ("evaluated once" if cached else f"re-evaluated in {len(self.batches)} batches per call"))


    def support_mask(self):
        """(Nx, Ny) bool mask of the pixels inside the field of view at every projection angle."""
        return fov_support_mask(self.Nx, self.Ny, self.My, angles=self.angles, image_width=self.Nx,
                                detector_width=self.My, detector_shift=self.cor_offset)

    def detector_coordinates(self):
        """Compute probed unit-sphere coordinates for all (omega, eta, peak) combinations.

        Coords shape: (N_Omega, N_Omega_subdivisions, N_eta, N_eta_subdivisions, N_peaks, 3)
        """
        wavelength_A = 12.398 / self.cfg["energy"]
        self.two_theta_peaks = 2.0 * np.arcsin(
            np.linalg.norm(self.h_cpu, axis=1) / (4.0 * np.pi) * wavelength_A
        ).astype(np.float32)

        S_eta = self.N_eta_subdivisions

        # Eta sub-bin centres: (N_eta, S_eta)
        eta_angles_2d = self.eta_angles_subdivided.reshape(self.N_eta, S_eta)

        det_dir_origin = np.array(self.cfg["detector_direction_origin"])
        det_dir_pos90 = np.array(self.cfg["detector_direction_positive_90"])
        p_direction_0 = np.array(self.cfg['p_direction_0'])
        k0 = np.asarray(self.cfg["k_direction_0"])

        N_fine_omega = self.N_Omega * self.N_Omega_subdivisions
        Rmats = R.from_rotvec(self.angles_subdivided[:, None] * k0).as_matrix()

        coords_list = []
        for tt in self.two_theta_peaks:
            twothetahalf = tt / 2.0

            # Zero-rotation-frame directions: (N_eta, S_eta, 3)
            dirs_zero = (
                np.cos(eta_angles_2d)[..., np.newaxis] * det_dir_origin[np.newaxis, np.newaxis, :]
                + np.sin(eta_angles_2d)[..., np.newaxis] * det_dir_pos90[np.newaxis, np.newaxis, :]
            )

            # Apply 2theta tilt
            dirs_zero = dirs_zero * np.cos(twothetahalf) - np.sin(twothetahalf) * p_direction_0

            # Rotate by all omega angles: (N_fine_omega, N_eta, S_eta, 3)
            probed = np.einsum('oij,esi->oesj', Rmats, dirs_zero)

            # Reshape: (N_Omega, N_Omega_sub, N_eta, S_eta, 3)
            coords = probed.reshape(self.N_Omega, self.N_Omega_subdivisions, self.N_eta, S_eta, 3)
            coords_list.append(coords)

        # Stack over peaks → (N_Omega, N_Omega_sub, N_eta, S_eta, N_peaks, 3)
        coords_cpu = np.stack(coords_list, axis=-2)
        self.coords_cpu = np.asarray(coords_cpu, dtype=np.float32, order="C")
        self.coords_gpu = clarray.to_device(self.queue, self.coords_cpu)




    def transfer_material_parameters_to_gpu(self):
        """Upload reciprocal-lattice vectors, intensities and symmetry ops to GPU.

        One entry per ring: the h-vectors are those of the first reflection of
        each ring (they only set the ring's two-theta), the intensity of a ring
        is the sum over its reflections.
        """
        first = [refl[0] for refl in self.ring_reflections]
        self.h_cpu_normed = np.asarray(self.material.h_vecs_normed, dtype=np.float32)[first].copy(order="C")
        self.h_cpu = np.asarray(self.material.h_vecs, dtype=np.float32)[first].copy(order="C")
        self.h_gpu_normed = clarray.to_device(self.queue, self.h_cpu_normed)

        intensities = np.asarray(self.material.intensities(), dtype=np.float64)
        self.intens_cpu = np.array([intensities[refl].sum() for refl in self.ring_reflections],
                                   dtype=np.float32)
        self.intens_gpu = clarray.to_device(self.queue, self.intens_cpu)

        self.sym_ops_cpu = np.asarray(self.material.point_group_matrices, dtype=np.float32, order="C")
        self.sym_ops_gpu = clarray.to_device(self.queue, self.sym_ops_cpu)



    def transfer_grid_parameters_to_gpu(self):
        """
        Transfer active orientation grid parameters (rotations + sigmas)
        from Grid objects to GPU.
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

            # --- rotations (for the poles in the sample frame) ---
        self.grid_rot_cpu = np.stack([n.R.as_matrix() for n in nodes], axis=0)

            # --- sigma per node ---
        self.sigma_cpu = np.array(
            [node.sigma for node in nodes],
            dtype=np.float32
            )



    def transfer_pole_axes_to_gpu(self):
        """
        Upload the distinct pole axes of every ring, rotated into the sample
        frame of every orientation, for the pfmatrix_eval_poles kernel.

        The symmetry images S_g h of a reflection coincide in groups: for the
        cubic point group, the 24 images span only 3-12 distinct axes (up to
        sign). The PF summand is even in the sign of the axis, so each distinct
        axis is evaluated once, weighted by the number of images on it. This
        gives the same sum as looping over all symmetry operators.

        A ring with several reflection families (e.g. (333) and (511)) sums the
        images of all of them, each family weighted by its share
        w_f = m_f / sum(m) of the ring's multiplicity (or of its intensity, if
        the PF matrix is scaled by intensity), so that every pole of the ring
        counts equally. A single-family ring has w_f = 1, i.e. the plain sum.
        """
        h = np.asarray(self.material.h_vecs_normed, dtype=np.float64)
        sym = np.asarray(self.material.point_group_matrices, dtype=np.float64).reshape(-1, 3, 3)
        multiplicity = np.asarray(self.material.reflections["multiplicity"], dtype=np.float64)
        intensities = np.asarray(self.material.intensities(), dtype=np.float64)

        axes, counts, start = [], [], [0]
        self.ring_family_weights = []
        for refl in self.ring_reflections:
            share = intensities[refl] if not self.normalized else multiplicity[refl]
            if not np.all(np.isfinite(share)) or share.sum() <= 0:
                share = multiplicity[refl]
            share = share / share.sum()
            self.ring_family_weights.append(share)

            images, weights = [], []
            for f, w in zip(refl, share):
                img = sym @ h[f]                                       # (G, 3)
                images.append(img)
                weights.append(np.full(len(img), w))
            images = np.concatenate(images)
            weights = np.concatenate(weights)
            # canonical sign: first non-zero component positive
            first = np.argmax(np.abs(images) > 1e-6, axis=1)
            images = images * np.sign(images[np.arange(len(images)), first])[:, None]
            uniq, inverse = np.unique(np.round(images, 6), axis=0, return_inverse=True)
            cnt = np.bincount(inverse.ravel(), weights=weights, minlength=len(uniq))
            axes.append(uniq / np.linalg.norm(uniq, axis=1, keepdims=True))
            counts.append(cnt)
            start.append(start[-1] + len(uniq))
        axes = np.concatenate(axes)
        self.N_axes = len(axes)
        self.axis_start_cpu = np.asarray(start, dtype=np.int32)
        self.axis_count_cpu = np.concatenate(counts).astype(np.float32)
        assert np.all(np.diff(self.axis_start_cpu) > 0)
        assert np.allclose(np.add.reduceat(self.axis_count_cpu, self.axis_start_cpu[:-1]), len(sym))

        # poles[k, a] = U_k @ axis_a, since dot(S_g h, U_k^-1 v) = dot(U_k S_g h, v)
        poles = np.einsum("kij,aj->kai", self.grid_rot_cpu, axes)
        self.poles_gpu = clarray.to_device(self.queue, np.ascontiguousarray(poles, dtype=np.float32))
        self.axis_start_gpu = clarray.to_device(self.queue, self.axis_start_cpu)
        self.axis_count_gpu = clarray.to_device(self.queue, self.axis_count_cpu)

        sigma = self.sigma_cpu
        self.inv_sigma2_gpu = clarray.to_device(self.queue, (1.0 / (sigma * sigma)).astype(np.float32))
        self.norm_factor_gpu = clarray.to_device(self.queue, (1.0 / (8.0 * np.pi * sigma * sigma)).astype(np.float32))



    def allocate_coefficient_buffer(self):
        """Pre-allocate reusable GPU buffers for forward/adjoint computation."""
        buffers = []

        def _alloc(name, shape, dtype, order):
            arr = clarray.empty(self.queue, shape, dtype=dtype, order=order)
            buffers.append((name, arr))
            return arr

        # sinogram batch, orientations fastest: the native projector's output and input,
        # and the layout of the PF products
        self.coeffs_sino_C = _alloc(
            "coeffs_sino_C",
            (self.N_Omega, self.My, self.K_batch_max),
            np.float32,
            "C",
        )

        if self.projector == "native":
            # image batch, orientations fastest: (Nx*Ny, K_batch_max)
            self._img_k = _alloc(
                "_img_k",
                (self.Nx * self.Ny * self.K_batch_max,),
                np.float32,
                "C",
            )
        else:
            self.coeffs_sino_F = _alloc(
                "coeffs_sino_F",
                (self.PS.n_detectors, self.PS.n_angles, self.K_batch_max),
                np.float32,
                "F",
            )

            self._coeffs_batch_F = _alloc(
                "_coeffs_batch_F",
                (self.Nx, self.Ny, self.K_batch_max),
                np.float32,
                "F",
            )

        self._basis_batch_kmax = _alloc(
            "_basis_batch_kmax",
            (self.N_Omega, self.K_batch_max, self.N_eta, self.N_peaks),
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
        """
        Release OpenCL buffers created by allocate_coefficient_buffer() and
        remove the corresponding attributes from this object.
        """
        buffer_names = [
            "coeffs_sino_F",
            "coeffs_sino_C",
            "_coeffs_batch_F",
            "_img_k",
            "_basis_batch_kmax",
        ]
        for sb in (getattr(self, "sparse_batches", None) or []):
            for arr in sb.values():
                if isinstance(arr, clarray.Array) and arr.base_data is not None:
                    arr.base_data.release()
        self.sparse_batches = None

        # Release buffers if they exist
        for name in buffer_names:
            arr = getattr(self, name, None)
            if arr is None:
                continue

            # pyopencl.array.Array holds the underlying cl.Buffer in .base_data
            try:
                base = getattr(arr, "base_data", None)
                if base is not None:
                    base.release()
            except Exception:
                pass

            # Remove attribute so refcount drops
            try:
                delattr(self, name)
            except Exception:
                pass

        # If you stored a total_bytes summary, clear it too
        if hasattr(self, "total_bytes"):
            try:
                delattr(self, "total_bytes")
            except Exception:
                pass

        # Make sure queued commands are done (finish before/after is fine)
        try:
            if hasattr(self, "queue") and self.queue is not None:
                self.queue.finish()
        except Exception:
            pass

        gc.collect()

        if getattr(self, "verbose", False):
            print("OpenCL GPU memory freed.")





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
        R = self.N_Omega
        C = self.N_eta
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
        # multiple of 4: the native projector handles 4 orientations per work item
        # (the padding rows of the PF matrix are zero)
        self.K_batch_max = -(-self.K_batch_max // 4) * 4

        # The per-batch buffers are indexed with 32-bit integers by some kernels (the full coefficient
        # and data arrays are not); a smaller max_gb gives smaller batches.
        Kmax = self.K_batch_max
        for name, n in [("PF batch (N_Omega, K_batch, N_eta, N_rings)", R * Kmax * C * T),
                        ("sinogram batch (N_Omega, My, K_batch)", R * self.My * Kmax),
                        ("image batch (Nx, Ny, K_batch)", self.Nx * self.Ny * Kmax)]:
            if n >= 2**31:
                raise ValueError(f"The {name} has {n} elements, more than 2^31 - 1; lower max_gb.")
    


    def direct(self, coeffs):
        """
        Allocating convenience wrapper for the OpenCL forward operator.

        Returns
        -------
        yin_gpu : clarray
            Shape (N_Omega, My, N_seg), C order
        """

        data = clarray.zeros(
            self.queue,
            (self.N_Omega, self.My, self.N_seg),
            dtype=np.float32,
            order="C",
        )

        self.direct_cl(coeffs, data)
        return data



    def _eval_pf_batch(self, k0, Kb):
        """Evaluate the dense PF matrix of orientations k0 .. k0+Kb-1 into
        _basis_batch_kmax, (R, K_batch_max, C, P); rows Kb .. K_batch_max-1 are zero."""
        R = int(self.N_Omega)
        C = int(self.N_eta)
        P = int(self.N_peaks)
        Kmax = self.K_batch_max
        self.pfpoles_kernel(
            self.queue,
            (R * Kmax * C * P,),
            None,
            self.coords_gpu.data,
            self.poles_gpu.data,
            self.axis_start_gpu.data,
            self.axis_count_gpu.data,
            self.inv_sigma2_gpu.data,
            self.norm_factor_gpu.data,
            self._basis_batch_kmax.data,
            np.int32(R),
            np.int32(Kmax),
            np.int32(C),
            np.int32(P),
            np.int32(self.N_axes),
            np.int32(self.N_Omega_subdivisions),
            np.int32(self.N_eta_subdivisions),
            np.int32(k0),
            np.int32(Kb),
        )
        if not self.normalized:
            self._scale_pf_by_intensity_inplace(self._basis_batch_kmax, self.intens_gpu)



    def _dense_pf_batch(self, k0, Kb):
        """Make sure _basis_batch_kmax holds the PF matrix of this batch (dense mode).
        With a single batch the matrix never changes, so it is evaluated only once."""
        if self._pf_cached:
            return
        self._eval_pf_batch(k0, Kb)
        self._pf_cached = len(self.batches) == 1



    def default_sparse_budget_gb(self):
        """
        Default GPU memory budget (GB) for the sparse PF matrix: half of what is
        left of the device memory after a reserve for the solver (FISTA with the
        fused update keeps 2 arrays of the coefficient size (Nx, Ny, K), reserved
        with one more as a margin, and 2 of the data size (N_Omega, My, N_seg):
        the data and the prediction/residual), the operator's
        buffers and a 1 GB margin. Conservative on purpose: the GPU may be
        shared, and the solver's arrays take priority.
        """
        coeff_bytes = 4 * self.Nx * self.Ny * self.K
        data_bytes = 4 * self.N_Omega * self.My * self.N_seg
        buffers = getattr(self, "total_bytes", None)
        if buffers is None:  # (only computed with verbose=True)
            buffers = 4 * self.K_batch_max * (self.N_Omega * self.My + self.Nx * self.Ny
                                              + self.N_Omega * self.N_eta * self.N_peaks)
        free = self.queue.device.global_mem_size - 3 * coeff_bytes - 2 * data_bytes - buffers - 1024**3
        return 0.5 * max(free, 0) / 1024**3



    def build_sparse_pf(self, auto=False, max_fill=0.1, max_gb=None):
        """
        Evaluate the PF matrix once and store it per K-batch in two CSR
        structures: forward rows (r, j) listing orientations, and adjoint rows
        (r, k) listing segments, j = c*P + p.

        The size of the whole sparse matrix is estimated from the first batch.
        With ``auto``, fall back to dense mode if the fill fraction of the first
        batch exceeds ``max_fill`` or the estimate exceeds ``max_gb``; without,
        raise MemoryError if it exceeds ``max_gb``.
        """
        t0 = time.perf_counter()
        q = self.queue
        R = int(self.N_Omega)
        CP = int(self.N_eta * self.N_peaks)
        Kmax = self.K_batch_max
        ints = lambda *a: [np.int32(v) for v in a]

        # column indices are stored as 16 bit (orientation within a batch, segment)
        if Kmax > 65535 or CP > 65535:
            reason = f"K_batch_max = {Kmax} or N_eta * N_rings = {CP} exceeds the 16-bit index range"
            if not auto:
                raise ValueError(f"Sparse PF matrix: {reason}; use pf_mode='dense' or a smaller max_gb.")
            self.pf_mode = "dense"
            if self.verbose:
                print(f"Sparse PF matrix: {reason}, using the dense PF path")
            return

        def csr(count_kernel, fill_kernel, n_rows, Kb):
            counts = clarray.empty(q, (n_rows,), np.int32)
            count_kernel(q, (n_rows,), None, self._basis_batch_kmax.data, counts.data, *ints(R, Kmax, CP, Kb))
            row_ptr = np.zeros(n_rows + 1, dtype=np.int64)
            np.cumsum(counts.get(), out=row_ptr[1:])
            nnz = int(row_ptr[-1])
            if nnz >= 2**31:
                raise ValueError("Too many non-zeros in one K-batch for int32 indices; lower max_gb.")
            row_ptr_gpu = clarray.to_device(q, row_ptr.astype(np.int32))
            col = clarray.empty(q, (max(nnz, 1),), np.uint16)
            val = clarray.empty(q, (max(nnz, 1),), np.float32)
            fill_kernel(q, (n_rows,), None, self._basis_batch_kmax.data, row_ptr_gpu.data, col.data, val.data,
                        *ints(R, Kmax, CP, Kb))
            return row_ptr_gpu, col, val, nnz

        sparse_batches = []
        nnz_total = 0
        for ib, b in enumerate(self.batches):
            k0, Kb = b["k_start"], b["K_batch"]
            self._eval_pf_batch(k0, Kb)
            row_ptr_f, col_k, val_f, nnz = csr(self.k.pf_count_rows_fwd, self.k.pf_fill_rows_fwd, R * CP, Kb)
            if ib == 0:
                fill = nnz / (R * Kb * CP)
                # two CSR copies (16-bit index + 32-bit value) plus the row pointers of all batches
                est_gb = (12 * fill * R * self.K * CP + 4 * (len(self.batches) * R * CP + R * self.K)) / 1024**3
                reason = None
                if auto and fill > max_fill:
                    reason = f"fill fraction {100 * fill:.2f} % > {100 * max_fill:.2f} %"
                elif max_gb is not None and est_gb > max_gb:
                    reason = f"estimated size {est_gb:.1f} GB > {max_gb:.1f} GB"
                    if not auto:
                        raise MemoryError(f"Sparse PF matrix: {reason}; use pf_mode='dense' or raise sparse_max_gb.")
                if reason is not None:
                    for arr in (row_ptr_f, col_k, val_f):
                        arr.base_data.release()
                    self.pf_mode = "dense"
                    if self.verbose:
                        print(f"Sparse PF matrix: {reason}, using the dense PF path")
                    return
            row_ptr_a, col_j, val_a, nnz_a = csr(self.k.pf_count_rows_adj, self.k.pf_fill_rows_adj, R * Kb, Kb)
            assert nnz_a == nnz
            sparse_batches.append(dict(row_ptr_f=row_ptr_f, col_k=col_k, val_f=val_f,
                                       row_ptr_a=row_ptr_a, col_j=col_j, val_a=val_a))
            nnz_total += nnz
        q.finish()

        self.sparse_batches = sparse_batches
        self.pf_mode = "sparse"
        self.sparse_nnz = nnz_total
        self.sparse_fill = nnz_total / (R * self.K * CP)
        # the dense batch buffer is only needed to build the sparse matrix
        self._basis_batch_kmax.base_data.release()
        self._basis_batch_kmax = None
        if self.verbose:
            nbytes = sum(a.nbytes for sb in sparse_batches for a in sb.values())
            print(f"Sparse PF matrix: {nnz_total} non-zeros (fill {100 * self.sparse_fill:.3f} %), "
                  f"{nbytes / 1024**2:.1f} MB, built in {time.perf_counter() - t0:.1f} s")



    def direct_cl(self, coeffs, data):
        """In-place forward operator: data = PF @ Radon(coeffs).

        Parameters
        ----------
        coeffs : clarray, (Nx, Ny, K), F-order
        data : clarray, (N_Omega, My, N_seg), C-order — overwritten.
        """
        data.fill(0.0)

        R = int(self.N_Omega)
        CP = int(self.N_eta * self.N_peaks)
        Kmax = self.K_batch_max
        My = self.My
        sparse = self.pf_mode == "sparse"

        for ib, b in enumerate(self.batches):
            k0 = b["k_start"]
            Kb = b["K_batch"]

            # 1) Radon transform of this batch -> coeffs_sino_C (R, My, Kmax)
            if self.projector == "native":
                self.radon.gather(coeffs, self._img_k, k0, Kb, Kmax)
                self.radon.forward(self._img_k, self.coeffs_sino_C, Kmax)
            else:
                self._radon_gratopy_forward(coeffs, k0, Kb)

            # 2) PF matrix product, accumulated into data
            if sparse:
                sb = self.sparse_batches[ib]
                self.k.spmm_pf_forward_c(
                    self.queue,
                    (CP, My, R),
                    None,
                    self.coeffs_sino_C.data,
                    sb["row_ptr_f"].data,
                    sb["col_k"].data,
                    sb["val_f"].data,
                    data.data,
                    np.int32(R),
                    np.int32(My),
                    np.int32(CP),
                    np.int32(Kmax),
                )
            else:
                self._dense_pf_batch(k0, Kb)
                batched_gemm_clblast(self.queue, self.coeffs_sino_C, self._basis_batch_kmax.reshape((R, Kmax, CP)),
                                     data, R=R, M=My, K=Kmax, N=CP)



    def _radon_gratopy_forward(self, coeffs, k0, Kb):
        """gratopy forward projection of orientations k0 .. k0+Kb-1 into coeffs_sino_C."""
        Nx, Ny, My, R, Kmax = self.Nx, self.Ny, self.My, self.N_Omega, self.K_batch_max
        self._coeffs_batch_F.fill(0.0)
        self.coeffs_sino_F.fill(0.0)
        total = Nx * Ny * Kb
        self.k.SLICE_COEFFS_K_BATCH_F(
            self.queue, (total,), None,
            coeffs.data, self._coeffs_batch_F.data,
            np.int32(Nx), np.int32(Ny), np.int32(self.K), np.int32(Kb), np.int32(k0),
        )
        gratopy.forwardprojection(self._coeffs_batch_F, self.PS, sino=self.coeffs_sino_F)
        total = R * My * Kmax
        self.k.transpose_d_omega_k_f_to_c(
            self.queue, (total,), None,
            self.coeffs_sino_F.data, self.coeffs_sino_C.data,
            np.int32(My), np.int32(R), np.int32(Kmax), np.int32(total),
        )



    def _radon_gratopy_backward(self, coeffs, k0, Kb, Ktot=None):
        """gratopy backprojection of coeffs_sino_C into orientations k0 .. k0+Kb-1 of coeffs,
        an (Nx, Ny, Ktot) Fortran-order array (Ktot: default K)."""
        Nx, Ny, My, R, Kmax = self.Nx, self.Ny, self.My, self.N_Omega, self.K_batch_max
        self._coeffs_batch_F.fill(0.0)
        self.coeffs_sino_F.fill(0.0)
        total = R * My * Kmax
        self.k.transpose_omega_d_k_c_to_d_omega_k_f(
            self.queue, (total,), None,
            self.coeffs_sino_C.data, self.coeffs_sino_F.data,
            np.int32(R), np.int32(My), np.int32(Kmax), np.int32(total),
        )
        gratopy.backprojection(self.coeffs_sino_F, self.PS, img=self._coeffs_batch_F)
        total = Nx * Ny * Kb
        self.k.scatter_k_lastaxis_f(
            self.queue, (total,), None,
            coeffs.data, self._coeffs_batch_F.data,
            np.int32(Nx), np.int32(Ny), np.int32(self.K if Ktot is None else Ktot), np.int32(k0), np.int32(Kb),
            np.int32(total),
        )



    def adjoint(self, data):
        """
        Allocating convenience wrapper for the OpenCL adjoint.

        Parameters
        ----------
        y_gpu : clarray
            Shape (N_Omega, My, N_seg), C-order

        Returns
        -------
        x_gpu : clarray
            Shape (Nx, Ny, K_sum), Fortran-order
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
            Shape (N_Omega, My, N_seg), C-order
        coeffs : clarray
            Shape (Nx, Ny, K_sum), Fortran-order
            Will be overwritten
        """

        # ---------------- checks ----------------
        assert data.shape == (self.N_Omega, self.My, self.N_seg)
        assert data.flags.c_contiguous

        assert coeffs.shape == (self.Nx, self.Ny, self.K)
        assert coeffs.flags.f_contiguous

        # every orientation is written by exactly one batch
        for ib, b in enumerate(self.batches):
            self._adjoint_batch(data, ib, coeffs, b["k_start"], self.K)

    def adjoint_batches_cl(self, data, out_batch, update):
        """
        The adjoint, one orientation batch at a time, without a coefficient-sized output.

        For every batch of orientations k0 .. k0+Kb-1, their part of A^T data is written to
        the first Nx * Ny * Kb elements of out_batch (Fortran order: pixel fastest, then the
        orientation within the batch), and then update(k0, Kb) is called, e.g. to enqueue a
        kernel that consumes it before the next batch overwrites it.

        Parameters
        ----------
        data : clarray, (N_Omega, My, N_seg), C-order
        out_batch : clarray, float32, at least Nx * Ny * K_batch_max elements
        update : callable (k0, Kb)
        """
        assert data.shape == (self.N_Omega, self.My, self.N_seg)
        assert data.flags.c_contiguous
        assert out_batch.dtype == np.float32 and out_batch.size >= self.Nx * self.Ny * self.K_batch_max
        for ib, b in enumerate(self.batches):
            self._adjoint_batch(data, ib, out_batch, 0, self.K_batch_max)
            update(b["k_start"], b["K_batch"])

    def _adjoint_batch(self, data, ib, target, k_target, K_target):
        """A^T data for orientation batch ib, written to orientations k_target .. k_target+Kb-1
        of target, an (Nx, Ny, K_target) Fortran-order array."""
        b = self.batches[ib]
        k0 = b["k_start"]
        Kb = b["K_batch"]
        Kmax = self.K_batch_max
        R = self.N_Omega
        CP = int(self.N_eta * self.N_peaks)
        My = self.My
        alpha = R / np.pi

        # 1) PF^T product -> coeffs_sino_C (R, My, Kmax)
        if self.pf_mode == "sparse":
            sb = self.sparse_batches[ib]
            self.k.spmm_pf_adjoint_c(
                self.queue,
                (Kb, My, R),
                None,
                data.data,
                sb["row_ptr_a"].data,
                sb["col_j"].data,
                sb["val_a"].data,
                self.coeffs_sino_C.data,
                np.int32(R),
                np.int32(My),
                np.int32(Kb),
                np.int32(CP),
                np.int32(Kmax),
                np.float32(alpha),
            )
        else:
            self._dense_pf_batch(k0, Kb)
            # batched gemm with the PF batch transposed on the fly; overwrites coeffs_sino_C
            batched_gemm_adj_clblast(self.queue, data, self._basis_batch_kmax.reshape((R, Kmax, CP)),
                                     self.coeffs_sino_C, R, My, CP, Kmax, alpha)

        # 2) backprojection into the target
        if self.projector == "native":
            self.radon.backward(self.coeffs_sino_C, self._img_k, Kmax)
            self.radon.scatter(self._img_k, target, k_target, Kb, Kmax)
        else:
            self._radon_gratopy_backward(target, k_target, Kb, K_target)



    def _scale_pf_by_intensity_inplace(self, PF_GPU: clarray.Array, INTENSITY_GPU: clarray.Array) -> None:
        """Element-wise multiply PF_GPU (R, Kb, C, P) by intensity per peak P."""
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





def group_reflections_into_rings(material, rtol=1e-6):
    """
    Group the reflections of a material into rings of equal two-theta, sorted
    by increasing two-theta. Returns a list with, per ring, the indices of its
    reflections, e.g. [[0], [1], ..., [9, 10], ...] when (333) and (511) share a ring.
    """
    tt = np.asarray(material.reflections["two_theta"], dtype=np.float64)
    rings = []
    for i in np.argsort(tt, kind="stable"):
        if rings and np.isclose(tt[i], tt[rings[-1][0]], rtol=rtol, atol=0.0):
            rings[-1].append(int(i))
        else:
            rings.append([int(i)])
    return rings



def batched_gemm_clblast(queue, A3, B3, C3, R, M, K, N):
    """Batched GEMM via CLBlast: C += A @ B, per-batch (beta=1 accumulate)."""

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




def batched_gemm_adj_clblast(queue, Y3, B3, X3, R, Mx, Nsub, K, alpha):
    """Batched adjoint GEMM: X = alpha * Y @ B^T, per-batch (beta=0 overwrite).

    B3 is the PF batch as stored, (R, K, Nsub); CLBlast transposes it on the fly.
    """

    # 2D views (NO COPY)
    A = Y3.reshape((R * Mx, Nsub))   # (R*Mx, Nsub)
    B = B3.reshape((R * K, Nsub))    # (R*K, Nsub), used transposed
    C = X3.reshape((R * Mx, K))      # (R*Mx, K)

    # leading dimensions (row-major)
    a_ld = Nsub
    b_ld = Nsub
    c_ld = K

    # batch strides (in elements)
    a_stride = Mx * Nsub
    b_stride = K * Nsub
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
        b_transp=True,
    )



def gpu_norm(x):
    """Compute L2 norm of a GPU array."""
    return float(clarray.sum(x*x).get() ** 0.5)


def estimate_L_power(
    op,
    niter: int = 20,
    seed: int = 0,
    eps: float = 1e-30,
    verbose: int = 1,
) -> float:
    """Estimate the Lipschitz constant L = ||A^T A|| via power iteration."""
    if niter < 1:
        raise ValueError("niter must be >= 1")

    q = op.queue
    rng = np.random.default_rng(seed)

    # x in domain, Fortran
    x = clarray.empty(q, (op.Nx, op.Ny, op.K), np.float32, order="F")
    # y in range, C
    Ax = clarray.empty(q, (op.N_Omega, op.My, op.N_seg), np.float32, order="C")
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
        # Ax = A x
        op.direct_cl(x, Ax)
        # z = A^* Ax
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
