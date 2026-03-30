from __future__ import annotations
import numpy as np
import time
import gc
from typing import Any, Mapping
import pyopencl as cl
import pyopencl.array as clarray
from pyclblast import gemmStridedBatched
from ..utils.grid import Grid
from ..crystallography.material import Material
from scipy.spatial.transform import Rotation as R
from .create_pfo_matrix import build_pf_program
from .pf_kernels import build_all_opencl


# Small inline kernel for scatter-add of 1D slices on GPU
_SCATTER_ADD_1D_SRC = """
__kernel void scatter_add_1d(
    __global float *dst,
    __global const float *src,
    const int dst_offset,
    const int count
){
    int gid = get_global_id(0);
    if (gid >= count) return;
    dst[dst_offset + gid] += src[gid];
}

__kernel void sum_columns(
    __global const float *in,
    __global float *out,
    const int R,
    const int Kmax
){
    int k = get_global_id(0);
    if (k >= Kmax) return;
    float s = 0.0f;
    for (int r = 0; r < R; r++) {
        s += in[r * Kmax + k];
    }
    out[k] = s;
}
"""


class BulkTextureForwardOperator:
    """Bulk texture forward operator on GPU.

    Computes  data = PF @ coeffs  and its adjoint, where PF is the
    pole-figure transform.  No tomographic (Radon) projection is involved.
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
        **kwargs,
    ):
        """Initialise the bulk-texture forward operator.

        Parameters
        ----------
        cfg : dict
            Experiment configuration (keys: N_Omega, N_eta, angle_range,
            wavelength, detector_direction_origin, etc.).
        material : Material
            Crystallographic material with reflections and symmetry.
        grid : Grid
            Orientation discretisation tree.
        max_gb : float
            GPU memory budget (GB) for K-batching of the pole-figure matrix.
        verbose : bool
            Print buffer allocation summary.
        normalized : bool
            If True, skip intensity scaling of the PF matrix.
        ctx, queue : optional
            Existing OpenCL context/queue; created automatically if None.
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

        # Keyword arguments override cfg values
        self.cfg = {**cfg, **kwargs}
        self.normalized = normalized
        self.material = material
        self.grid = grid
        self.verbose = verbose
        self.N_eta = self.cfg['N_eta']
        self.N_peaks = len(self.material.reflections)
        self.N_seg = self.N_peaks * self.N_eta
        self.N_Omega = self.cfg['N_Omega']
        self.N_Omega_subdivisions = self.cfg.get('N_Omega_subdivisions', 1)
        self.angle_range = np.array(self.cfg['angle_range']) / 180 * np.pi
        delta = (self.angle_range[1] - self.angle_range[0]) / self.N_Omega
        sub_delta = delta / self.N_Omega_subdivisions
        # Coarse angles centered in their intervals
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

        # Compile scatter-add helper
        self._scatter_prg = cl.Program(self.ctx, _SCATTER_ADD_1D_SRC).build()
        self._k_scatter_add_1d = cl.Kernel(self._scatter_prg, "scatter_add_1d")
        self._k_sum_columns = cl.Kernel(self._scatter_prg, "sum_columns")

        self.transfer_material_parameters_to_gpu()
        self.detector_coordinates()
        self.transfer_grid_parameters_to_gpu()
        self.get_pf_batches_for_material()

        # Allocate buffers
        self.allocate_coefficient_buffer()


    def detector_coordinates(self):
        """Compute probed unit-sphere coordinates for all (omega, eta, peak) combinations.

        Coords shape: (N_Omega, N_Omega_subdivisions, N_eta, N_eta_subdivisions, N_peaks, 3)
        """
        wavelength_angstrom = 12.398 / self.cfg["wavelength"]
        self.two_theta_peaks = 2.0 * np.arcsin(
            np.linalg.norm(self.h_cpu, axis=1) / (4.0 * np.pi) * wavelength_angstrom
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
        """Upload reciprocal-lattice vectors, intensities and symmetry ops to GPU."""
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

            # --- sigma per node ---
        self.sigma_cpu = np.array(
            [node.sigma for node in nodes],
            dtype=np.float32
            )



    def allocate_coefficient_buffer(self):
        """Pre-allocate reusable GPU buffers for forward/adjoint computation."""
        buffers = []

        def _alloc(name, shape, dtype, order):
            arr = clarray.empty(self.queue, shape, dtype=dtype, order=order)
            buffers.append((name, arr))
            return arr

        self._coeffs_batch_kmax = _alloc(
            "_coeffs_batch_kmax",
            (self.K_batch_max,),
            np.float32,
            "C",
        )

        self._basis_batch_kmax = _alloc(
            "_basis_batch_kmax",
            (self.N_Omega, self.K_batch_max, self.N_eta, self.N_peaks),
            np.float32,
            "C",
        )

        self._basis_batch_transpose_kmax = _alloc(
            "_basis_batch_transpose_kmax",
            (self.N_Omega, self.N_peaks * self.N_eta, self.K_batch_max),
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

        self._adjoint_temp_kmax = _alloc(
            "_adjoint_temp_kmax",
            (self.N_Omega, self.K_batch_max),
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
            "_coeffs_batch_kmax",
            "_basis_batch_kmax",
            "_basis_batch_transpose_kmax",
            "_grid_inv_kmax",
            "_inv_sigma2_kmax",
            "_norm_factor_kmax",
            "_adjoint_temp_kmax",
        ]

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
        if bytes_per_K > 0.01:
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

        Parameters
        ----------
        coeffs : clarray
            Shape (K,), float32.

        Returns
        -------
        data : clarray
            Shape (N_Omega, N_seg), C order.
        """

        data = clarray.zeros(
            self.queue,
            (self.N_Omega, self.N_seg),
            dtype=np.float32,
            order="C",
        )

        self.direct_cl(coeffs, data)
        return data



    def direct_cl(self, coeffs, data):
        """In-place forward operator: data += PF @ coeffs.

        Parameters
        ----------
        coeffs : clarray, shape (K,), float32.
        data : clarray, shape (N_Omega, N_seg), C-order — accumulated into.
        """
        data.fill(0.0)

        R  = int(self.N_Omega)
        C = int(self.N_eta)
        P = int(self.N_peaks)
        G = int(len(self.sym_ops_cpu))
        Kmax = self.K_batch_max
        NCP = C * P
        coords_gpu   = self.coords_gpu
        sym_ops_gpu  = self.sym_ops_gpu
        h_gpu_normed = self.h_gpu_normed
        intensity_gpu = self.intens_gpu

        for b in self.batches:
            k0 = b["k_start"]
            k1 = b["k_end"]
            Kb = b["K_batch"]

            # -------------------------------------------------
            # 1) Slice coefficients into zero-padded batch buffer
            # -------------------------------------------------
            self._coeffs_batch_kmax.fill(0.0)
            if Kb > 0:
                cl.enqueue_copy(
                    self.queue,
                    self._coeffs_batch_kmax.data,
                    coeffs.data,
                    dst_offset=0,
                    src_offset=k0 * 4,
                    byte_count=Kb * 4,
                )

            # -------------------------------------------------
            # 2) Upload grid_inv and sigma for this batch
            # -------------------------------------------------
            self._norm_factor_kmax.fill(0.0)
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
                self.grid_inv_gpu.data,
                self._grid_inv_kmax.data,
                np.int32(self.K),
                np.int32(Kb),
                np.int32(k0),
            )

            # -------------------------------------------------
            # 3) Evaluate PF matrix: (R, Kmax, C, P)
            # -------------------------------------------------
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
                np.int32(self.N_Omega_subdivisions),
                np.int32(self.N_eta_subdivisions),
            )
            if not self.normalized:
                self._scale_pf_by_intensity_inplace(self._basis_batch_kmax, intensity_gpu)

            # -------------------------------------------------
            # 4) GEMM: data += PF @ coeffs_batch
            #    Per omega: (1, Kmax) @ (Kmax, C*P) = (1, C*P)
            #    a_stride=0 reuses the same coefficients for all omegas.
            # -------------------------------------------------
            bulk_forward_gemm(
                self.queue,
                self._coeffs_batch_kmax,
                self._basis_batch_kmax,
                data,
                R, Kmax, NCP,
            )



    def adjoint(self, data):
        """
        Allocating convenience wrapper for the OpenCL adjoint.

        Parameters
        ----------
        data : clarray
            Shape (N_Omega, N_seg), C-order.

        Returns
        -------
        coeffs : clarray
            Shape (K,), float32.
        """

        coeffs = clarray.zeros(
            self.queue,
            (self.K,),
            dtype=np.float32,
        )

        self.adjoint_cl(data, coeffs)
        return coeffs



    def adjoint_cl(self, data, coeffs):
        """
        In-place OpenCL adjoint operator.

        Parameters
        ----------
        data : clarray
            Shape (N_Omega, N_seg), C-order.
        coeffs : clarray
            Shape (K,), float32 — will be overwritten.
        """

        # ---------------- zero output ----------------
        coeffs.fill(0.0)

        Kmax = self.K_batch_max
        R  = self.N_Omega
        C = int(self.N_eta)
        P = int(self.N_peaks)
        G = int(len(self.sym_ops_cpu))
        NCP = C * P
        coords_gpu   = self.coords_gpu
        sym_ops_gpu  = self.sym_ops_gpu
        h_gpu_normed = self.h_gpu_normed
        intensity_gpu = self.intens_gpu

        for b in self.batches:
            k0 = b["k_start"]
            k1 = b["k_end"]
            Kb = b["K_batch"]

            # -------------------------------------------------
            # 1) Upload grid_inv and sigma for this batch
            # -------------------------------------------------
            self._norm_factor_kmax.fill(0.0)
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
                self.grid_inv_gpu.data,
                self._grid_inv_kmax.data,
                np.int32(self.K),
                np.int32(Kb),
                np.int32(k0),
            )

            # -------------------------------------------------
            # 2) Evaluate PF matrix: (R, Kmax, C, P)
            # -------------------------------------------------
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
                np.int32(self.N_Omega_subdivisions),
                np.int32(self.N_eta_subdivisions),
            )
            if not self.normalized:
                self._scale_pf_by_intensity_inplace(self._basis_batch_kmax, intensity_gpu)

            # -------------------------------------------------
            # 3) Transpose PF → PF_T: (R, C*P, Kmax)
            # -------------------------------------------------
            total = R * Kmax * C * P
            self.k.btranspose_kernel(
                self.queue,
                (total,),
                None,
                self._basis_batch_kmax.data,
                self._basis_batch_transpose_kmax.data,
                np.int32(R),
                np.int32(Kmax),
                np.int32(NCP),
                np.int32(total),
            )

            # -------------------------------------------------
            # 4) Batched GEMM: adjoint_temp[r, :] = data[r,:] @ PF_T[r,:,:]
            #    Per omega: (1, NCP) @ (NCP, Kmax) → (1, Kmax)
            # -------------------------------------------------
            self._adjoint_temp_kmax.fill(0.0)
            bulk_adjoint_gemm(
                self.queue,
                data,
                self._basis_batch_transpose_kmax,
                self._adjoint_temp_kmax,
                R, NCP, Kmax,
            )

            # -------------------------------------------------
            # 5) Sum over omegas: coeffs_batch[k] = sum_r adjoint_temp[r, k]
            # -------------------------------------------------
            self._k_sum_columns(
                self.queue,
                (Kmax,),
                None,
                self._adjoint_temp_kmax.data,
                self._coeffs_batch_kmax.data,
                np.int32(R),
                np.int32(Kmax),
            )

            # -------------------------------------------------
            # 6) Scatter-add batch result into output coefficients
            # -------------------------------------------------
            self._k_scatter_add_1d(
                self.queue,
                (Kb,),
                None,
                coeffs.data,
                self._coeffs_batch_kmax.data,
                np.int32(k0),
                np.int32(Kb),
            )



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




def bulk_forward_gemm(queue, coeffs_batch, PF, data, R, Kmax, NCP):
    """Batched GEMM for bulk forward: data += PF @ coeffs_batch.

    Per omega r: data[r, :] += coeffs_batch^T @ PF[r, :, :]
    Uses a_stride=0 to reuse the same coefficients for all omegas.

    Parameters
    ----------
    coeffs_batch : clarray (Kmax,)
    PF : clarray (R, Kmax, C, P) or (R, Kmax, NCP)
    data : clarray (R, C, P) or (R, NCP) — accumulated into.
    """
    gemmStridedBatched(
        queue,
        1, NCP, Kmax,                             # m, n, k
        R,                                          # batch_count
        coeffs_batch.reshape((1, Kmax)),            # A: (1, Kmax), same for all batches
        PF.reshape((R * Kmax, NCP)),                # B: (R*Kmax, NCP)
        data.reshape((R, NCP)),                     # C: (R, NCP)
        Kmax, NCP, NCP,                             # a_ld, b_ld, c_ld
        0, Kmax * NCP, NCP,                         # a_stride=0, b_stride, c_stride
        alpha=1.0,
        beta=1.0,                                   # accumulate across K-batches
    )


def bulk_adjoint_gemm(queue, data, PF_T, adjoint_temp, R, NCP, Kmax):
    """Batched GEMM for bulk adjoint: adjoint_temp[r,:] = data[r,:] @ PF_T[r,:,:].

    Per omega r:  (1, NCP) @ (NCP, Kmax) → (1, Kmax)

    Parameters
    ----------
    data : clarray (R, NCP)
    PF_T : clarray (R, NCP, Kmax)
    adjoint_temp : clarray (R, Kmax) — overwritten.
    """
    gemmStridedBatched(
        queue,
        1, Kmax, NCP,                               # m, n, k
        R,                                           # batch_count
        data.reshape((R, NCP)),                      # A: per batch (1, NCP)
        PF_T.reshape((R * NCP, Kmax)),               # B: per batch (NCP, Kmax)
        adjoint_temp.reshape((R, Kmax)),             # C: per batch (1, Kmax)
        NCP, Kmax, Kmax,                             # a_ld, b_ld, c_ld
        NCP, NCP * Kmax, Kmax,                       # a_stride, b_stride, c_stride
        alpha=1.0,
        beta=0.0,
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

    # x in domain (1D coefficient vector)
    x = clarray.empty(q, (op.K,), np.float32)
    # y in range
    Ax = clarray.empty(q, (op.N_Omega, op.N_seg), np.float32, order="C")
    # z = A^*Ax in domain
    z = clarray.empty(q, x.shape, np.float32)

    # init x random
    x_host = rng.standard_normal(x.shape).astype(np.float32)
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
