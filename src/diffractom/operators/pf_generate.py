"""
The sparse PF matrix of one orientation batch, generated directly on the GPU (pf_generate.cl).

The CSR structures are the ones SinglePhaseForwardOperator used to build from the dense PF
batch, bitwise: forward rows (r, j) listing orientations k, adjoint rows (r, k) listing segments
j = c*P + p. Instead of evaluating all R * Kb * C * P entries, only the entries near the Bragg
condition of some pole are evaluated, so the cost scales with the number of non-zeros.

Used to build the stored sparse matrix, to generate the batches that are not stored in every
forward and adjoint call, and to transpose a stored adjoint CSR into the forward one.
"""
import weakref
from pathlib import Path

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
import pyopencl.tools as cltools
from pyopencl.scan import ExclusiveScanKernel
from scipy.spatial.transform import Rotation

from .quadrature import nodes_weights

# the candidate test assumes the second term of the PF kernel, exp(-(1 + |n.v|) / sigma^2), is
# cut off, i.e. 1 / sigma^2 >= the threshold (6 by default: sigma below 23 degrees); with a margin
MAX_SIGMA = np.deg2rad(20.0)  # for the default threshold

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
        self.op = weakref.proxy(op)  # weak: the operator keeps its generator (op.pf_gen), see Streamer
        q = self.q = op.queue
        R, C, P = op.N_Omega, op.N_eta, op.N_peaks
        self.CP = op.N_seg
        maxwc = -(-C // 32)
        self.ppw = max(1 << (P - 1).bit_length(), maxwc)
        self.rpg = max(1, 128 // self.ppw)
        src = Path(__file__).with_name("pf_generate.cl").read_text()
        quad = list(op.quad_options)
        if op.quad_weights_cpu is not None:     # the span of the nodes in an eta bin, for the candidate search
            x, _ = nodes_weights(op.quadrature.p_eta, op.quadrature.k_eta)
            quad += [f"-DQX_LO={float(x.min())!r}f", f"-DQX_HI={float(x.max())!r}f"]
        prg = cl.Program(op.ctx, src).build(options=[f"-DMAXWC={maxwc}", f"-DPPW={self.ppw}", f"-DRPG={self.rpg}",
                                                     *op.pf_cut_options, *quad])
        self.k = {n: cl.Kernel(prg, n) for n in ("pf_gen_candidates", "pf_gen_evaluate", "pf_gen_compact",
                                                  "pf_gen_count_fwd", "pf_gen_fill_fwd", "pf_gen_rank_rows",
                                                  "pf_gen_row_of", "pf_gen_sub_counts", "pf_gen_sub_lengths",
                                                  "pf_gen_count_fwd_sub", "pf_gen_fill_fwd_sub")}
        # the scan's temporary buffers come from a pool (a fresh allocation every call costs ms)
        self.pool = cltools.MemoryPool(cltools.ImmediateAllocator(q))
        self.scan = ExclusiveScanKernel(op.ctx, np.int32, "a+b", "0")

        # frame vectors rotated like the probed directions in detector_coordinates (R^T d)
        cfg = op.cfg
        d0, d90, p0, kd = (np.asarray(cfg[k], dtype=np.float64) for k in
                           ("detector_direction_origin", "detector_direction_positive_90", "p_direction_0", "k_direction_0"))
        Rm = Rotation.from_rotvec(op.angles_subdivided[:, None] * kd).as_matrix()
        frames = np.concatenate([np.einsum("oij,i->oj", Rm, v) for v in (d0, d90, p0)], axis=1)
        # per ring, or per pole axis if the operator gives every reflection its own Bragg angle
        theta = (op.axis_two_theta_cpu if op.reflection_two_theta else op.two_theta_peaks).astype(np.float64) / 2.0
        dev = lambda a: clarray.to_device(q, np.ascontiguousarray(a, dtype=np.float32))
        self.frames, self.sin_th, self.cos_th = dev(frames), dev(np.sin(theta)), dev(np.cos(theta))
        e0, e1 = (float(v) for v in op.eta_angle_range)
        self.eta0 = np.float32(e0)
        # eta sub-bin width of the midpoint rule; for weighted (Gauss) rules the candidate search works with
        # whole bins (pf_generate.cl, QUAD_WEIGHTED)
        self.dsub = np.float32((e1 - e0) / (C * (1 if op.quad_weights_cpu is not None else op.N_eta_subdivisions)))
        self.full = np.int32(abs((e1 - e0) - 2 * np.pi) < 1e-6)
        self.n_rows_max = R * int(k_batch_max)
        self.cnt = clarray.empty(q, (self.n_rows_max + 1,), np.int32)
        self.cand_ptr = clarray.empty(q, (self.n_rows_max + 1,), np.int32)
        self.sub_len = clarray.empty(q, (R + 1,), np.int32)    # angle subsets: entries per angle, their offsets
        self.sub_ptr = clarray.empty(q, (R + 1,), np.int32)
        self.n_cap = self.nnz_cap = 0
        self._names = []
        self.ptr_cache = {}   # k0 -> (Kb, candidate offsets of the batch's rows), see cache_counts

    @staticmethod
    def usable(op):
        cut = 6.0 if op.pf_cutoff_sigma is None else float(op.pf_cutoff_sigma) ** 2 / 2
        return float(np.max(op.sigma_cpu)) < MAX_SIGMA * np.sqrt(6.0 / cut)

    def _ints(self, *v):
        return [np.int32(x) for x in v]

    def _candidates(self, k0, Kb, mode, cand_j=None, cand_row=None, cand_ptr=None, sub=None):
        """sub: None (all angles) or the angle indices (int32 device array) whose rows are generated (compact:
        row s * Kb + k for angle sub[s])."""
        op = self.op
        R_rows = op.N_Omega if sub is None else sub.size
        n_rows = R_rows * Kb
        groups = -(-(n_rows + 1) // self.rpg)
        dummy = self.cnt.data
        cand_ptr = self.cand_ptr if cand_ptr is None else cand_ptr
        self.k["pf_gen_candidates"](
            self.q, (groups * self.rpg * self.ppw,), (self.rpg * self.ppw,),
            op.poles_gpu.data, op.axis_start_gpu.data, op.inv_sigma2_gpu.data, self.frames.data,
            self.sin_th.data, self.cos_th.data, self.cnt.data, cand_ptr.data,
            cand_j.data if cand_j is not None else dummy, cand_row.data if cand_row is not None else dummy,
            *self._ints(R_rows, Kb, k0, op.N_eta, op.N_peaks, op.N_axes, op.N_Omega_subdivisions,
                        op.N_eta_subdivisions),
            self.eta0, self.dsub, self.full, np.int32(mode), (sub if sub is not None else self.cnt).data,
            np.int32(sub is not None))

    def _count(self, k0, Kb, sub=None):
        """Candidates of orientations k0 .. k0+Kb-1 in cand_ptr (exclusive scan of the row counts); sub: the rows
        of these angles only (see _candidates)."""
        n_rows = (self.op.N_Omega if sub is None else sub.size) * Kb
        self._candidates(k0, Kb, 0, sub=sub)
        self.scan(self.cnt[:n_rows + 1], self.cand_ptr[:n_rows + 1], allocator=self.pool)
        return n_rows

    def cache_counts(self, batches):
        """Keep the candidate offsets of these batches (4 bytes per row (omega bin, orientation)), computed once:
        generate() then skips the counting pass, about a third of the generation time. Their size: cache_bytes."""
        for b in batches:
            k0, Kb = int(b["k_start"]), int(b["K_batch"])
            if k0 in self.ptr_cache:
                continue
            n_rows = self._count(k0, Kb)
            self.ptr_cache[k0] = (Kb, self.cand_ptr[:n_rows + 1].copy())
        self.q.finish()

    def cache_bytes(self, batches=None):
        """Bytes of the cached candidate offsets (of the given batches if they were cached)."""
        if batches is not None:
            return sum(4 * (self.op.N_Omega * int(b["K_batch"]) + 1) for b in batches)
        return sum(a.nbytes for _, a in self.ptr_cache.values())

    def release_cache(self):
        for _, a in self.ptr_cache.values():
            if a.base_data is not None:
                a.base_data.release()
        self.ptr_cache = {}

    def count_candidates(self, batches, sub=None, cache=False):
        """Number of candidates of every batch (for sizing; read back once for all batches); sub: of the rows of these
        angles. cache: also keep the candidate offsets of the batches (as cache_counts, from the same counting pass)."""
        if not batches:
            return []
        totals = clarray.empty(self.q, (len(batches),), np.int32)
        for ib, b in enumerate(batches):
            n_rows, ptr = self._sub_offsets(b["k_start"], b["K_batch"], sub)
            cl.enqueue_copy(self.q, totals.data, ptr.data, byte_count=4, src_offset=4 * n_rows, dst_offset=4 * ib)
            k0 = int(b["k_start"])
            if cache and sub is None and k0 not in self.ptr_cache:
                self.ptr_cache[k0] = (int(b["K_batch"]), ptr[:n_rows + 1].copy())
        out = [int(v) for v in totals.get()]
        totals.base_data.release()
        return out

    def _sub_offsets(self, k0, Kb, sub=None):
        """(n_rows, candidate offsets of the rows) of a batch: cached, derived from the cached offsets of all rows
        (an angle subset: gather and scan, no counting pass), or counted into cand_ptr."""
        cached = self.ptr_cache.get(int(k0))
        if cached is not None and cached[0] == int(Kb):
            if sub is None:
                return self.op.N_Omega * Kb, cached[1]     # counted once (cache_counts)
            n_rows = sub.size * Kb
            self.k["pf_gen_sub_counts"](self.q, (_ceil(n_rows + 1),), (_LOCAL,), cached[1].data, sub.data,
                                        self.cnt.data, np.int32(Kb), np.int32(n_rows))
            self.scan(self.cnt[:n_rows + 1], self.cand_ptr[:n_rows + 1], allocator=self.pool)
            return n_rows, self.cand_ptr
        return self._count(k0, Kb, sub), self.cand_ptr

    def count_subset_entries(self, sb, Kb, sub):
        """Non-zeros of the rows of the angles sub in a stored adjoint CSR of all angles (reads back; sizing)."""
        ptr = sb["row_ptr_a"].get()
        r = sub.get().astype(np.int64)
        return int((ptr[(r + 1) * Kb] - ptr[r * Kb]).sum())

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
        return sum(getattr(self, n).nbytes for n in self._names) + self.cache_bytes()

    def generate(self, k0, Kb, forward=True, sub=None, cap=None):
        """Enqueue the sparse PF matrix of orientations k0 .. k0+Kb-1: a dict with the adjoint
        CSR (row_ptr_a, col_j, val_a) and, if forward, the forward CSR (row_ptr_f, col_k, val_f);
        views of the scratch buffers. Its candidates must not exceed allocate()'s n_cap.

        sub: only the rows of these angles (int32 device array), compact: adjoint rows s * Kb + k and forward rows
        s * CP + j for angle sub[s]; their entries are those of the same rows of the whole matrix. cap: an upper
        bound of the candidates of these rows (count_candidates; default n_cap), the size of the per-candidate
        passes."""
        op, q, k = self.op, self.q, self.k
        C, P = op.N_eta, op.N_peaks
        n_rows, cand_ptr = self._sub_offsets(k0, Kb, sub)
        self._candidates(k0, Kb, 1, self.cand_j, self.cand_row, cand_ptr, sub)
        cap = self.n_cap if cap is None else min(int(cap), self.n_cap)
        n_flags = cap + 1
        k["pf_gen_evaluate"](q, (_ceil(n_flags),), (_LOCAL,),
                             self.cand_j.data, self.cand_row.data, cand_ptr.data, op.coords_gpu.data,
                             op.poles_gpu.data, op.axis_start_gpu.data, op.axis_count_gpu.data,
                             op.inv_sigma2_gpu.data, op.norm_factor_gpu.data, op.quad_weights_gpu.data,
                             op.intens_gpu.data,
                             self.cand_val.data, self.flag.data,
                             *self._ints(n_rows, n_flags, Kb, k0, C, P, op.N_axes, op.N_Omega_subdivisions,
                                         op.N_eta_subdivisions, op.normalized),
                             (sub if sub is not None else self.cnt).data, np.int32(sub is not None),
                             op.w_gpu_or_dummy.data, op.axis_sin_gpu.data, op.axis_cos_gpu.data)
        self.scan(self.flag[:n_flags], self.pos[:n_flags], allocator=self.pool)
        k["pf_gen_compact"](q, (_ceil(max(cap, n_rows + 1)),), (_LOCAL,),
                            self.cand_j.data, self.cand_row.data, self.cand_val.data, self.flag.data,
                            self.pos.data, cand_ptr.data, self.row_ptr_a.data, self.col_j.data,
                            self.val_a.data, self.row_of.data, np.int32(n_rows))
        out = dict(row_ptr_a=self.row_ptr_a[:n_rows + 1], col_j=self.col_j, val_a=self.val_a)
        if forward:
            out.update(self._transpose(out, Kb, cap, n_rows // Kb))
        return out

    def transpose(self, sb, Kb, sub=None, cap=None):
        """The forward CSR (row_ptr_f, col_k, val_f; scratch views) of a stored adjoint CSR of Kb
        orientations (row_ptr_a, col_j, val_a) with at most allocate()'s nnz_cap non-zeros.

        sub: the forward rows of these angles only (int32 device array; compact rows s * CP + j for angle sub[s]),
        built from their entries alone; cap: an upper bound of those entries (count_subset_entries)."""
        q, k, CP = self.q, self.k, self.CP
        if sub is None:
            n_rows = self.op.N_Omega * Kb
            self.k["pf_gen_row_of"](self.q, (_ceil(self.nnz_cap),), (_LOCAL,), sb["row_ptr_a"].data,
                                    self.row_of.data, np.int32(n_rows))
            return self._transpose(sb, Kb, self.nnz_cap)
        Rs = int(sub.size)
        cap = self.nnz_cap if cap is None else min(int(cap), self.nnz_cap)
        nf = Rs * CP
        k["pf_gen_sub_lengths"](q, (_ceil(Rs + 1),), (_LOCAL,), sb["row_ptr_a"].data, sub.data, self.sub_len.data,
                                np.int32(Kb), np.int32(Rs))
        self.scan(self.sub_len[:Rs + 1], self.sub_ptr[:Rs + 1], allocator=self.pool)
        cnt_f, row_ptr_f = self.cnt_f[:nf + 1], self.row_ptr_f[:nf + 1]
        cnt_f.fill(0)
        g = (_ceil(max(cap, 1)),)
        k["pf_gen_count_fwd_sub"](q, g, (_LOCAL,), sb["row_ptr_a"].data, sb["col_j"].data, sub.data,
                                  self.sub_ptr.data, cnt_f.data, *self._ints(Kb, Rs, CP))
        self.scan(cnt_f, row_ptr_f, allocator=self.pool)
        self.cursor[:nf].fill(0)
        k["pf_gen_fill_fwd_sub"](q, g, (_LOCAL,), sb["row_ptr_a"].data, sb["col_j"].data, sb["val_a"].data,
                                 sub.data, self.sub_ptr.data, row_ptr_f.data, self.cursor.data, self.col_tmp.data,
                                 self.val_tmp.data, *self._ints(Kb, Rs, CP))
        k["pf_gen_rank_rows"](q, g, (_LOCAL,), row_ptr_f.data, self.col_tmp.data, self.val_tmp.data,
                              self.col_k.data, self.val_f.data, np.int32(nf))
        return dict(row_ptr_f=row_ptr_f, col_k=self.col_k, val_f=self.val_f)

    def _transpose(self, sb, Kb, cap, R_rows=None):
        """Forward CSR from the adjoint CSR sb, with row_of filled: count, scan, fill, rank by k. R_rows: the
        number of angle rows of sb (default all; compact subset rows from generate)."""
        q, k, CP = self.q, self.k, self.CP
        R_rows = self.op.N_Omega if R_rows is None else int(R_rows)
        n_rows = R_rows * Kb
        nf = R_rows * CP
        cnt_f, row_ptr_f = self.cnt_f[:nf + 1], self.row_ptr_f[:nf + 1]
        cnt_f.fill(0)
        k["pf_gen_count_fwd"](q, (_ceil(cap),), (_LOCAL,), self.row_of.data, sb["col_j"].data,
                              sb["row_ptr_a"].data, cnt_f.data, *self._ints(n_rows, Kb, CP))
        self.scan(cnt_f, row_ptr_f, allocator=self.pool)
        self.cursor[:nf].fill(0)
        k["pf_gen_fill_fwd"](q, (_ceil(cap),), (_LOCAL,), self.row_of.data, sb["col_j"].data,
                             sb["val_a"].data, sb["row_ptr_a"].data, row_ptr_f.data, self.cursor.data,
                             self.col_tmp.data, self.val_tmp.data, *self._ints(n_rows, Kb, CP))
        k["pf_gen_rank_rows"](q, (_ceil(cap),), (_LOCAL,), row_ptr_f.data, self.col_tmp.data,
                              self.val_tmp.data, self.col_k.data, self.val_f.data, np.int32(nf))
        return dict(row_ptr_f=row_ptr_f, col_k=self.col_k, val_f=self.val_f)

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
        self.release_cache()
        for n in ("cnt", "cand_ptr", "sub_len", "sub_ptr", "frames", "sin_th", "cos_th"):
            a = getattr(self, n)
            if a is not None and a.base_data is not None:
                a.base_data.release()
            setattr(self, n, None)
        self.pool.free_held()
