"""
The sparse PF matrix of one orientation batch, generated directly on the GPU (pf_generate.cl).

The CSR structures are the ones SinglePhaseForwardOperator used to build from the dense PF
batch, bitwise: forward rows (r, j) listing orientations k, adjoint rows (r, k) listing segments
j = c*P + p. Instead of evaluating all R * Kb * C * P entries, only the entries near the Bragg
condition of some pole are evaluated, so the cost scales with the number of non-zeros.

Used to build the stored sparse matrix, to generate the batches that are not stored in every
forward and adjoint call, and to transpose a stored adjoint CSR into the forward one.
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

# scratch bytes per candidate (cand_j, cand_row, cand_val, flag, pos) and per non-zero (adjoint
# CSR col_j, val_a, row_of; forward CSR col_k, val_f and its unordered fill)
CAND_BYTES = 2 + 4 + 4 + 4 + 4
NNZ_BYTES = 2 + 4 + 4 + 2 + 4 + 2 + 4


def _ceil(n, m=_LOCAL):
    return -(-int(n) // m) * m


class SparsePFGenerator:
    """Generates (and transposes) the sparse PF matrix of orientation batches of an operator.

    generate() and transpose() enqueue everything on the operator's queue and read nothing back,
    so they can run in the middle of a streamed FISTA pass; the scratch buffers they return are
    overwritten by the next call. allocate() sizes the scratch buffers first.
    """

    def __init__(self, op, k_batch_max):
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
                                                  "pf_gen_count_fwd", "pf_gen_fill_fwd", "pf_gen_rank_rows",
                                                  "pf_gen_row_of")}
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
        self.n_rows_max = R * int(k_batch_max)
        self.cnt = clarray.empty(q, (self.n_rows_max + 1,), np.int32)
        self.cand_ptr = clarray.empty(q, (self.n_rows_max + 1,), np.int32)
        self.n_cap = self.nnz_cap = 0
        self._names = []

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

    def _count(self, k0, Kb):
        """Candidates of orientations k0 .. k0+Kb-1 in cand_ptr (exclusive scan of the row counts)."""
        n_rows = self.op.N_Omega * Kb
        self._candidates(k0, Kb, 0)
        self.scan(self.cnt[:n_rows + 1], self.cand_ptr[:n_rows + 1], allocator=self.pool)
        return n_rows

    def count_candidates(self, batches):
        """Number of candidates of every batch (reads them back; for sizing)."""
        out = []
        for b in batches:
            n_rows = self._count(b["k_start"], b["K_batch"])
            out.append(int(self.cand_ptr[n_rows:n_rows + 1].get()[0]))
        return out

    def allocate(self, n_cap=0, nnz_cap=0):
        """Scratch buffers: for generating batches of up to n_cap candidates (0: none), and for
        the forward CSR of batches of up to nnz_cap non-zeros (at least n_cap)."""
        nnz_cap = max(int(nnz_cap), int(n_cap))
        if max(n_cap, nnz_cap) >= 2**31 - 1:
            raise ValueError("Too many PF entries in one K-batch for int32 indices; lower max_gb.")
        self.release_scratch()
        q = self.q
        e = lambda size, dt: clarray.empty(q, (max(int(size), 1),), dt)
        bufs = {}
        if n_cap > 0:
            bufs.update(cand_j=e(n_cap, np.uint16), cand_row=e(n_cap, np.int32), cand_val=e(n_cap, np.float32),
                        flag=e(n_cap + 1, np.int32), pos=e(n_cap + 1, np.int32),
                        row_ptr_a=e(self.n_rows_max + 1, np.int32), col_j=e(n_cap, np.uint16),
                        val_a=e(n_cap, np.float32))
        if nnz_cap > 0:
            nf = self.op.N_Omega * self.CP
            bufs.update(row_of=e(nnz_cap, np.int32), cnt_f=e(nf + 1, np.int32), row_ptr_f=e(nf + 1, np.int32),
                        cursor=e(nf, np.int32), col_k=e(nnz_cap, np.uint16), val_f=e(nnz_cap, np.float32),
                        col_tmp=e(nnz_cap, np.uint16), val_tmp=e(nnz_cap, np.float32))
        for n, a in bufs.items():
            setattr(self, n, a)
        self._names = list(bufs)
        self.n_cap, self.nnz_cap = int(n_cap), nnz_cap

    def nbytes(self):
        return sum(getattr(self, n).nbytes for n in self._names)

    def generate(self, k0, Kb, forward=True):
        """Enqueue the sparse PF matrix of orientations k0 .. k0+Kb-1: a dict with the adjoint
        CSR (row_ptr_a, col_j, val_a) and, if forward, the forward CSR (row_ptr_f, col_k, val_f);
        views of the scratch buffers. Its candidates must not exceed allocate()'s n_cap."""
        op, q, k = self.op, self.q, self.k
        C, P = op.N_eta, op.N_peaks
        n_rows = self._count(k0, Kb)
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
            out.update(self._transpose(out, Kb, self.n_cap))
        return out

    def transpose(self, sb, Kb):
        """The forward CSR (row_ptr_f, col_k, val_f; scratch views) of a stored adjoint CSR of Kb
        orientations (row_ptr_a, col_j, val_a) with at most allocate()'s nnz_cap non-zeros."""
        n_rows = self.op.N_Omega * Kb
        self.k["pf_gen_row_of"](self.q, (_ceil(self.nnz_cap),), (_LOCAL,), sb["row_ptr_a"].data, self.row_of.data,
                                np.int32(n_rows))
        return self._transpose(sb, Kb, self.nnz_cap)

    def _transpose(self, sb, Kb, cap):
        """Forward CSR from the adjoint CSR sb, with row_of filled: count, scan, fill, rank by k."""
        q, k, CP = self.q, self.k, self.CP
        n_rows = self.op.N_Omega * Kb
        nf = self.op.N_Omega * CP
        self.cnt_f.fill(0)
        k["pf_gen_count_fwd"](q, (_ceil(cap),), (_LOCAL,), self.row_of.data, sb["col_j"].data,
                              sb["row_ptr_a"].data, self.cnt_f.data, *self._ints(n_rows, Kb, CP))
        self.scan(self.cnt_f, self.row_ptr_f, allocator=self.pool)
        self.cursor.fill(0)
        k["pf_gen_fill_fwd"](q, (_ceil(cap),), (_LOCAL,), self.row_of.data, sb["col_j"].data,
                             sb["val_a"].data, sb["row_ptr_a"].data, self.row_ptr_f.data, self.cursor.data,
                             self.col_tmp.data, self.val_tmp.data, *self._ints(n_rows, Kb, CP))
        k["pf_gen_rank_rows"](q, (_ceil(cap),), (_LOCAL,), self.row_ptr_f.data, self.col_tmp.data,
                              self.val_tmp.data, self.col_k.data, self.val_f.data, np.int32(nf))
        return dict(row_ptr_f=self.row_ptr_f, col_k=self.col_k, val_f=self.val_f)

    def release_scratch(self):
        for n in self._names:
            a = getattr(self, n)
            if a is not None and a.base_data is not None:
                a.base_data.release()
            setattr(self, n, None)
        self._names = []
        self.n_cap = self.nnz_cap = 0

    def release(self):
        self.release_scratch()
        for n in ("cnt", "cand_ptr", "frames", "sin_th", "cos_th"):
            a = getattr(self, n)
            if a is not None and a.base_data is not None:
                a.base_data.release()
            setattr(self, n, None)
        self.pool.free_held()
