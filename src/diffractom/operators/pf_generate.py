"""
The sparse PF matrix of one orientation batch, generated directly on the GPU (pf_generate.cl).

The CSR structures are the ones SinglePhaseForwardOperator.build_sparse_pf used to build from
the dense PF batch, bitwise: forward rows (r, j) listing orientations k, adjoint rows (r, k)
listing segments j = c*P + p. Instead of evaluating all R * Kb * C * P entries, only the entries
near the Bragg condition of some pole are evaluated, so the cost scales with the number of
non-zeros. Used to build the stored sparse matrix, and with pf_mode="generated" to generate every
batch again in every forward and adjoint call, when the stored matrix would not fit.
"""
from pathlib import Path

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
import pyopencl.tools as cltools
from pyopencl.scan import ExclusiveScanKernel
from scipy.spatial.transform import Rotation

# the candidate test assumes the second term of the PF kernel, exp(-(1 + |n.v|) / sigma^2), is
# cut off (it is for sigma below 23 degrees)
MAX_SIGMA = np.deg2rad(20.0)

_LOCAL = 256  # local size of the one-dimensional kernels


def _ceil(n, m=_LOCAL):
    return -(-int(n) // m) * m


class SparsePFGenerator:
    """Generates the sparse PF matrix of an operator's orientation batches.

    generate(ib) enqueues everything on the operator's queue and reads nothing back, so it can
    run in the middle of a streamed FISTA pass; the scratch buffers it returns are overwritten by
    the next call. Call allocate() (sizes from the candidate counts of all batches) first.
    """

    def __init__(self, op):
        self.op = op
        q = self.q = op.queue
        R, C, P = op.N_Omega, op.N_eta, op.N_peaks
        self.CP = op.N_seg
        maxwc = -(-C // 32)
        self.ppw = max(1 << (P - 1).bit_length(), maxwc)
        self.rpg = max(1, 128 // self.ppw)
        src = Path(__file__).with_name("pf_generate.cl").read_text()
        prg = cl.Program(op.ctx, src).build(options=[f"-DMAXWC={maxwc}", f"-DPPW={self.ppw}", f"-DRPG={self.rpg}"])
        self.k = {n: cl.Kernel(prg, n) for n in ("pf_gen_candidates", "pf_gen_evaluate", "pf_gen_compact",
                                                  "pf_gen_count_fwd", "pf_gen_fill_fwd", "pf_gen_sort_rows")}
        # the scan's temporary buffers come from a pool (a fresh allocation every call costs ms)
        self.pool = cltools.MemoryPool(cltools.ImmediateAllocator(q))
        self.scan = ExclusiveScanKernel(op.ctx, np.int32, "a+b", "0")

        # frame vectors rotated like the probed directions in detector_coordinates (R^T d)
        cfg = op.cfg
        d0, d90, p0, kd = (np.asarray(cfg[k], dtype=np.float64) for k in
                           ("detector_direction_origin", "detector_direction_positive_90", "p_direction_0", "k_direction_0"))
        Rm = Rotation.from_rotvec(op.angles_subdivided[:, None] * kd).as_matrix()
        frames = np.concatenate([np.einsum("oij,i->oj", Rm, v) for v in (d0, d90, p0)], axis=1)
        theta = op.two_theta_peaks.astype(np.float64) / 2.0
        dev = lambda a: clarray.to_device(q, np.ascontiguousarray(a, dtype=np.float32))
        self.frames, self.sin_th, self.cos_th = dev(frames), dev(np.sin(theta)), dev(np.cos(theta))
        e0, e1 = (float(v) for v in op.eta_angle_range)
        self.eta0 = np.float32(e0)
        self.dsub = np.float32((e1 - e0) / (C * op.N_eta_subdivisions))
        self.full = np.int32(abs((e1 - e0) - 2 * np.pi) < 1e-6)
        self.n_rows_max = R * op.K_batch_max
        self.cnt = clarray.empty(q, (self.n_rows_max + 1,), np.int32)
        self.cand_ptr = clarray.empty(q, (self.n_rows_max + 1,), np.int32)
        self.n_cap = None

    @staticmethod
    def usable(op):
        return float(np.max(op.sigma_cpu)) < MAX_SIGMA

    def _ints(self, *v):
        return [np.int32(x) for x in v]

    def _candidates(self, k0, Kb, mode, cand_j=None, cand_row=None):
        op = self.op
        n_rows = op.N_Omega * Kb
        groups = -(-(n_rows + 1) // self.rpg)
        dummy = self.cnt.data
        self.k["pf_gen_candidates"](
            self.q, (groups * self.rpg * self.ppw,), (self.rpg * self.ppw,),
            op.poles_gpu.data, op.axis_start_gpu.data, op.inv_sigma2_gpu.data, self.frames.data,
            self.sin_th.data, self.cos_th.data, self.cnt.data, self.cand_ptr.data,
            cand_j.data if cand_j is not None else dummy, cand_row.data if cand_row is not None else dummy,
            *self._ints(op.N_Omega, Kb, k0, op.N_eta, op.N_peaks, op.N_axes, op.N_Omega_subdivisions,
                        op.N_eta_subdivisions),
            self.eta0, self.dsub, self.full, np.int32(mode))

    def _count(self, ib):
        """Candidates of batch ib in cand_ptr (exclusive scan of the row counts)."""
        b = self.op.batches[ib]
        n_rows = self.op.N_Omega * b["K_batch"]
        self._candidates(b["k_start"], b["K_batch"], 0)
        self.scan(self.cnt[:n_rows + 1], self.cand_ptr[:n_rows + 1], allocator=self.pool)
        return n_rows

    def count_candidates(self):
        """Number of candidates of every batch (reads them back; for sizing)."""
        out = []
        for ib in range(len(self.op.batches)):
            n_rows = self._count(ib)
            out.append(int(self.cand_ptr[n_rows:n_rows + 1].get()[0]))
        return out

    def allocate(self, n_cap):
        """Scratch buffers for batches of up to n_cap candidates (and as many non-zeros)."""
        if n_cap >= 2**31 - 1:
            raise ValueError("Too many PF candidates in one K-batch for int32 indices; lower max_gb.")
        q, n = self.q, max(int(n_cap), 1)
        self.release_scratch()
        self.n_cap = n
        e = lambda size, dt: clarray.empty(q, (size,), dt)
        self.cand_j, self.cand_row, self.cand_val = e(n, np.uint16), e(n, np.int32), e(n, np.float32)
        self.flag, self.pos = e(n + 1, np.int32), e(n + 1, np.int32)
        self.row_ptr_a, self.col_j, self.val_a, self.row_of = (e(self.n_rows_max + 1, np.int32), e(n, np.uint16),
                                                              e(n, np.float32), e(n, np.int32))
        nf = self.op.N_Omega * self.CP
        self.cnt_f, self.row_ptr_f, self.cursor = e(nf + 1, np.int32), e(nf + 1, np.int32), e(nf, np.int32)
        self.col_k, self.val_f = e(n, np.uint16), e(n, np.float32)

    def nbytes(self):
        return sum(a.nbytes for a in self._scratch())

    def _scratch(self):
        names = ("cand_j", "cand_row", "cand_val", "flag", "pos", "row_ptr_a", "col_j", "val_a", "row_of",
                 "cnt_f", "row_ptr_f", "cursor", "col_k", "val_f")
        return [getattr(self, n) for n in names if getattr(self, n, None) is not None]

    def generate(self, ib, forward=True):
        """Enqueue the sparse PF matrix of batch ib: a dict with the adjoint CSR (row_ptr_a, col_j,
        val_a) and, if forward, the forward CSR (row_ptr_f, col_k, val_f); views of the scratch
        buffers. The number of candidates must not exceed the capacity given to allocate()."""
        op, q, k = self.op, self.q, self.k
        b = op.batches[ib]
        k0, Kb = b["k_start"], b["K_batch"]
        R, C, P, CP = op.N_Omega, op.N_eta, op.N_peaks, self.CP
        n_rows = self._count(ib)
        self._candidates(k0, Kb, 1, self.cand_j, self.cand_row)
        n_flags = self.n_cap + 1
        k["pf_gen_evaluate"](q, (_ceil(n_flags),), (_LOCAL,),
                             self.cand_j.data, self.cand_row.data, self.cand_ptr.data, op.coords_gpu.data,
                             op.poles_gpu.data, op.axis_start_gpu.data, op.axis_count_gpu.data,
                             op.inv_sigma2_gpu.data, op.norm_factor_gpu.data, op.intens_gpu.data,
                             self.cand_val.data, self.flag.data,
                             *self._ints(n_rows, n_flags, Kb, k0, C, P, op.N_axes, op.N_Omega_subdivisions,
                                         op.N_eta_subdivisions, op.normalized))
        self.scan(self.flag, self.pos, allocator=self.pool)
        k["pf_gen_compact"](q, (_ceil(max(self.n_cap, n_rows + 1)),), (_LOCAL,),
                            self.cand_j.data, self.cand_row.data, self.cand_val.data, self.flag.data,
                            self.pos.data, self.cand_ptr.data, self.row_ptr_a.data, self.col_j.data,
                            self.val_a.data, self.row_of.data, np.int32(n_rows))
        out = dict(row_ptr_a=self.row_ptr_a[:n_rows + 1], col_j=self.col_j, val_a=self.val_a)
        if forward:
            nf = R * CP
            self.cnt_f.fill(0)
            k["pf_gen_count_fwd"](q, (_ceil(self.n_cap),), (_LOCAL,), self.row_of.data, self.col_j.data,
                                  self.row_ptr_a.data, self.cnt_f.data, *self._ints(n_rows, Kb, CP))
            self.scan(self.cnt_f, self.row_ptr_f, allocator=self.pool)
            self.cursor.fill(0)
            k["pf_gen_fill_fwd"](q, (_ceil(self.n_cap),), (_LOCAL,), self.row_of.data, self.col_j.data,
                                 self.val_a.data, self.row_ptr_a.data, self.row_ptr_f.data, self.cursor.data,
                                 self.col_k.data, self.val_f.data, *self._ints(n_rows, Kb, CP))
            k["pf_gen_sort_rows"](q, (_ceil(nf),), (_LOCAL,), self.row_ptr_f.data, self.col_k.data,
                                  self.val_f.data, np.int32(nf))
            out.update(row_ptr_f=self.row_ptr_f, col_k=self.col_k, val_f=self.val_f)
        return out

    def release_scratch(self):
        for a in self._scratch():
            if a.base_data is not None:
                a.base_data.release()
        for n in ("cand_j", "cand_row", "cand_val", "flag", "pos", "row_ptr_a", "col_j", "val_a", "row_of",
                  "cnt_f", "row_ptr_f", "cursor", "col_k", "val_f"):
            setattr(self, n, None)
        self.n_cap = None

    def release(self):
        self.release_scratch()
        for n in ("cnt", "cand_ptr", "frames", "sin_th", "cos_th"):
            a = getattr(self, n)
            if a is not None and a.base_data is not None:
                a.base_data.release()
            setattr(self, n, None)
        self.pool.free_held()
