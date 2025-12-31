import numpy as np

from package.texture_tomography.operators.memory_tracker import MemoryCounter


class OperatorMemoryModel:
    """
    Deterministic memory model for PFO_OPENCL_BATCHED.

    This class mirrors the operator's execution flow but
    only counts memory. No OpenCL, no allocations.
    """

    def __init__(self, op, mem):
        """
        Parameters
        ----------
        op : PFO_OPENCL_BATCHED
            A fully initialized operator instance.
            Used only for shapes and configuration.
        """
        self.op = op
        self.mem = mem

        # shorthand
        self.Nx = op.Nx
        self.N_rot = op.N_rot
        self.N_chi = op.N_chi
        self.N_theta = op.N_theta
        self.N_seg = op.N_seg
        self.K_list = op.K_list
        self.K_sum = op.K_sum
        self.N_mat = op.N_mat

        # initialize persistent allocations
        self._init_persistent_buffers()

    # ------------------------------------------------------------
    # persistent allocations (exist for lifetime of operator)
    # ------------------------------------------------------------
    def _init_persistent_buffers(self):
        """
        Mirrors allocations done in __init__.
        """
        mem = self.mem

        # out_sub buffers (per material)
        for i_mat in range(self.N_mat):
            Nsub = len(self.op.full_idx_list[i_mat])
            mem.alloc(
                name=f"_out_sub[{i_mat}]",
                shape=(self.N_rot, self.Nx, Nsub),
                dtype=np.float32,
            )

        # coeffs_t buffers (per material)
        for i_mat, Ki in enumerate(self.K_list):
            mem.alloc(
                name=f"_coeffs_t_gpu[{i_mat}]",
                shape=(self.N_rot, self.Nx, Ki),
                dtype=np.float32,
            )

        # PF GPU inputs (persistent, per material)
        for i_mat in range(self.N_mat):
            dims = self.op.pf_dims_list[i_mat]
            R, C, P, K, G = (
                dims["R"],
                dims["C"],
                dims["P"],
                dims["K"],
                dims["G"],
            )

            mem.alloc(f"pf_coords[{i_mat}]", (R, C, P, 3))
            mem.alloc(f"pf_grid_inv[{i_mat}]", (K, 9))
            mem.alloc(f"pf_sym_ops[{i_mat}]", (G, 9))
            mem.alloc(f"pf_h[{i_mat}]", (P, 3))
            mem.alloc(f"pf_intensity[{i_mat}]", (P,))

        # index buffers
        for i_mat, idx in enumerate(self.op.full_idx_list):
            mem.alloc(
                name=f"full_idx[{i_mat}]",
                shape=(len(idx),),
                dtype=np.int32,
            )



    def free_memory(self):
        """
        Mirror of PFO_OPENCL_BATCHED.free_memory(),
        but operating purely on the MemoryCounter.
        """

        # --- lists of GPU buffers ---
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

        for base_name in gpu_lists:
            # We don't know list lengths dynamically here,
            # so we free all allocations that start with base_name[
            to_free = [
                name for name in self.mem._allocations
                if name.startswith(f"{base_name}[")
            ]

            for name in to_free:
                self.mem.free(name)

        # --- standalone buffers ---
        for name in [
            "_coeffs_gpu_full_sino",
            "_x_full_gpu",
            "B_gpu",
        ]:
            if name in self.mem._allocations:
                self.mem.free(name)





    def set_pf_batch_max_gb(self, max_gb: float):
        """
        Set maximum allowed GPU memory (in GB) for ONE convolved PF batch.

        This controls batching over K.
        """
        self.pf_batch_max_gb = float(max_gb)




    def get_pf_batches_for_material(self, i_mat: int):
        if not hasattr(self, "pf_batch_max_gb"):
            raise RuntimeError(
                "pf_batch_max_gb not set in memory model"
            )

        R = self.op.N_rot
        C = self.op.N_chi
        T = self.op.N_theta_mask_list[i_mat]
        K_total = self.op.K_list[i_mat]

        bytes_per_float = 4
        bytes_per_K = R * C * T * bytes_per_float

        max_bytes = self.pf_batch_max_gb * (1024 ** 3)

        if bytes_per_K > 0:
            K_batch_max = max(int(max_bytes // bytes_per_K), 1)
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




    def model_get_c_opencl_fortran(self, i_mat: int):
        """
        Memory model for get_c_opencl_fortran().
        Returns the allocation name so caller can free it.
        """

        D = self.op.PS.n_detectors   # == Nx
        O = self.op.PS.n_angles     # == N_rot
        i0 = self.op.offsets[i_mat]
        i1 = self.op.offsets[i_mat + 1]
        Ki = i1 - i0

        name = f"coeffs_sub_gpu[i_mat={i_mat}]"

        self.mem.alloc(
            name=name,
            shape=(D, O, Ki),
            dtype=np.float32
        )

        return name
    



    def model_direct(self):
        """
        Memory model for direct().

        Accounts for:
        - allocation of output yin_gpu
        - all allocations inside direct_cl
        """

        mem = self.mem
        op = self.op

        # ---------------- output allocation ----------------
        mem.alloc(
            name="yin_gpu",
            shape=(op.N_rot, op.Nx, op.N_chi * op.N_theta),
            dtype=np.float32,
        )

        # ---------------- call into direct_cl ----------------
        self.model_direct_cl()

        # NOTE:
        # yin_gpu is returned to the caller, so we do NOT free it here





    def model_direct_cl(self):
        """
        Memory model for direct_cl (forward operator).
        """

        mem = self.mem
        op = self.op

        # ---------------- persistent buffers ----------------

        # coeffs_gpu_full_sino (created once, reused)
        if not hasattr(self, "_has_coeffs_gpu_full_sino"):
            mem.alloc(
                name="_coeffs_gpu_full_sino",
                shape=(
                    op.PS.n_detectors,
                    op.PS.n_angles,
                    op.K_sum,
                ),
                dtype=np.float32,
            )
            self._has_coeffs_gpu_full_sino = True

        # ---------------- loop over materials ----------------
        for i_mat in range(op.N_mat):

            mem.enter_scope(label=f"direct_cl.material{i_mat}")

            # ---- get_c_opencl_fortran ----
            self.model_get_c_opencl_fortran(i_mat)

            # ---- forward_gpu_opencl ----
            self.model_forward_gpu_opencl(i_mat)

            mem.exit_scope()




    def model_forward_gpu_opencl(self, i_mat: int):
        """
        Memory model for forward_gpu_opencl (per material).
        """

        mem = self.mem
        op = self.op

        # ---------------- constants / shapes ----------------
        R = op.N_rot
        Mx = op.Nx
        K_i = op.K_list[i_mat]

        idx_size = op.full_idx_list[i_mat].size
        Nsub = int(idx_size)

        dims = op.pf_dims_list[i_mat]
        C = int(dims["C"])
        P = int(dims["P"])

        # ---------------- batching plan ----------------
        batches = op.get_pf_batches_for_material(i_mat)

        # ---------------- batch loop ----------------
        for bi, b in enumerate(batches):
            Kb = int(b["K_batch"])

            mem.enter_scope(label=f"forward_gpu_opencl.mat{i_mat}.batch{bi}")

            # ---- 1) coeffs_batch: (R, Mx, Kb) ----
            mem.alloc(
                name=f"coeffs_batch[m{i_mat},b{bi}]",
                shape=(R, Mx, Kb),
                dtype=np.float32,
            )

            # ---- 2) grid_inv_batch: (Kb, 9) ----
            mem.alloc(
                name=f"grid_inv_batch[m{i_mat},b{bi}]",
                shape=(Kb, 9),
                dtype=np.float32,
            )

            # ---- 3) PF basis batch: (R, Kb, C, P) ----
            mem.alloc(
                name=f"pf_basis_batch[m{i_mat},b{bi}]",
                shape=(R, Kb, C, P),
                dtype=np.float32,
            )

            # ---- 4) convolution (gaussian + conv output) ----
            self.model_convolve_matrix_from_pf_batch(
                i_mat=i_mat,
                Kb=Kb,
            )

            # ---- 5) out_sub already exists (persistent) ----
            # no alloc here

            # ---- 6) GEMM result written into out_sub (no alloc) ----

            # ---- 7) accumulate into out_gpu_full (no alloc) ----

            # ---- free batch temporaries ----
            mem.exit_scope()




    def model_convolve_matrix_from_pf_batch(self, i_mat: int, Kb: int):
        """
        Memory model for convolve_matrix_from_pf_batch.
        """

        mem = self.mem
        op = self.op

        # ---- shapes ----
        R = op.N_rot
        C = op.N_chi
        T = op.N_theta_mask_list[i_mat]   # masked theta count

        dims = op.pf_dims_list[i_mat]
        P = int(dims["P"])

        # ---- Gaussian weights GPU buffer ----
        mem.alloc(
            name=f"gaussian_gpu[m{i_mat},Kb{Kb}]",
            shape=(P, T),
            dtype=np.float32,
        )

        # ---- convolution output buffer ----
        mem.alloc(
            name=f"conv_out_gpu[m{i_mat},Kb{Kb}]",
            shape=(R, Kb, C, T),
            dtype=np.float32,
        )

        # reshape -> no allocation




    def model_adjoint(self):
        """
        Memory model for adjoint().
        """

        mem = self.mem
        op = self.op

        # ---- output allocation: x_gpu ----
        mem.alloc(
            name="x_gpu",
            shape=(op.Nx, op.Nx, op.K_sum),
            dtype=np.float32,
        )

        # ---- call adjoint_cl ----
        self.model_adjoint_cl()

        # x_gpu is returned → NOT freed here





    def model_adjoint_cl(self):
        """
        Memory model for adjoint_cl().

        IMPORTANT:
        - Do not manually mem.free() things that are in the scope stack.
        Let exit_scope() do it, or allocate them outside the scope.
        """

        mem = self.mem
        op = self.op

        O = op.N_rot
        D = op.Nx

        # ---- persistent x_full_gpu ----
        if not hasattr(self, "_x_full_gpu_modeled"):
            mem.alloc(
                name="_x_full_gpu",
                shape=(D, O, op.K_sum),
                dtype=np.float32,
            )
            self._x_full_gpu_modeled = True

        # ---- loop over materials ----
        for i_mat in range(op.N_mat):
            K_i = op.K_list[i_mat]

            mem.enter_scope(label=f"adjoint_cl.mat{i_mat}")

            # allocates xin_gpu[m{i_mat}] (returned buffer)
            self.model_adjoint_gpu_opencl(i_mat)

            # transpose buffer xin_gpu_t (temporary)
            mem.alloc(
                name=f"xin_gpu_t[m{i_mat}]",
                shape=(D, O, K_i),
                dtype=np.float32,
            )

            # del xin_gpu; del xin_gpu_t happen in the real code before next material
            # We model this by ending the scope (which frees both safely).
            mem.exit_scope()






    def model_adjoint_gpu_opencl(self, i_mat: int):
        """
        Memory model for adjoint_gpu_opencl (per material).

        IMPORTANT:
        - Allocations meant to "return" to caller (xin_gpu) are NOT freed here.
        - Temporaries (data_gpu_sub and batch temps) live in an inner scope.
        """

        mem = self.mem
        op = self.op

        # ---------------- constants ----------------
        R = op.N_rot
        Mx = op.Nx
        K_i = op.K_list[i_mat]

        Nsub = int(op.full_idx_list[i_mat].size)

        dims = op.pf_dims_list[i_mat]
        C = int(dims["C"])
        P = int(dims["P"])

        # ---------------- out_gpu (returned to caller) ----------------
        # In the real code: out_gpu = clarray.empty(...); returned to adjoint_cl
        mem.alloc(
            name=f"xin_gpu[m{i_mat}]",
            shape=(R, Mx, K_i),
            dtype=np.float32,
        )

        # ---------------- temporaries for this material ----------------
        mem.enter_scope(label=f"adjoint_gpu_opencl.mat{i_mat}.temps")

        # data_gpu_sub exists for the whole function in real code
        mem.alloc(
            name=f"data_gpu_sub[m{i_mat}]",
            shape=(R, Mx, Nsub),
            dtype=np.float32,
        )

        # ---------------- batching ----------------
        batches = op.get_pf_batches_for_material(i_mat)

        for bi, b in enumerate(batches):
            Kb = int(b["K_batch"])

            mem.enter_scope(label=f"adjoint_gpu_opencl.mat{i_mat}.batch{bi}")

            mem.alloc(
                name=f"grid_inv_batch[m{i_mat},b{bi}]",
                shape=(Kb, 9),
                dtype=np.float32,
            )

            mem.alloc(
                name=f"pf_basis_batch[m{i_mat},b{bi}]",
                shape=(R, Kb, C, P),
                dtype=np.float32,
            )

            mem.alloc(
                name=f"B_gpu_batch[m{i_mat},b{bi}]",
                shape=(R, Kb, Nsub),
                dtype=np.float32,
            )

            mem.alloc(
                name=f"BT_gpu_batch[m{i_mat},b{bi}]",
                shape=(R, Nsub, Kb),
                dtype=np.float32,
            )

            mem.alloc(
                name=f"x_batch[m{i_mat},b{bi}]",
                shape=(R, Mx, Kb),
                dtype=np.float32,
            )

            mem.exit_scope()  # frees batch temporaries

        mem.exit_scope()  # frees data_gpu_sub and any other material temporaries
