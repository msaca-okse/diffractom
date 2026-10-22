"""
Single-phase texture-tomography operator with a local basis: orientation k has coefficients only where a
LocalSupport allows it (e.g. near the voxels where point-by-point indexing found it), instead of an image on the
whole grid.

The coefficients are a flat float32 vector of rectangular patches (local_support.build_patches): the operator
projects every patch onto a strip of the detector a few bins wider than the patch, and applies the PF matrix
only where strips are. Memory for the coefficients and the projection work scale with the patches' area, the PF
products with the strips' width, instead of K x Ny x Nx and K x My. The data, the PF matrix (stored or
generated, sparse) and the geometry are those of SinglePhaseForwardOperator, which this class extends.

Same operator as SinglePhaseForwardOperator applied to coefficients that are zero outside the support (native
projector): with one patch per orientation (split_patches=False) the results agree bitwise; with several
patches per orientation (the default when an orientation's voxels are scattered) to float rounding.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

from ..utils.arrays import as_host, check_device
from ..utils.grid import MatrixGrid
from ..utils.support import fov_support_mask
from .local_support import LocalSupport, build_patches
from .single_phase_forward_operator import SinglePhaseForwardOperator

TY = 8          # detector lines per tile of the PF products (local_kernels.cl)
LS_PF = 256     # work-group size of the forward PF product (a work-item per segment)
LS_PA = 256     # work-group size of the adjoint PF product (a work-item per patch; V100: 8.8 ms vs 20.5 with 64)
LS_RADON = 32   # strip projection: work-items per angle (bins of one patch)
RADON_AB = 4    # strip projection: angles per work-group, in parallel
LS_BACK = 64    # pixels per work-group of the backprojection


def _grid_arrays(grid):
    """(K, 3, 3) rotation matrices and (K,) kernel widths of the grid's active leaves, in the operator's order."""
    if hasattr(grid, "leaf_arrays"):
        mats, sigmas = grid.leaf_arrays()
        return np.asarray(mats, dtype=np.float64), np.asarray(sigmas, dtype=np.float64)
    nodes = [grid.nodes[i] for i in grid.active_leaf_nodes()]
    mats = np.stack([n.R.as_matrix() for n in nodes]).astype(np.float64)
    sigmas = np.array([n.sigma for n in nodes], dtype=np.float64)
    return mats, sigmas


class LocalSinglePhaseForwardOperator(SinglePhaseForwardOperator):
    """
    data = PF @ Radon(coeffs) with a local basis.

    Coefficients: a flat float32 vector of length n_coef (coeff_shape = (n_coef,)); to_dense / from_dense convert
    from and to (K, Ny, Nx) arrays in the grid's orientation order, to_coo gives (k, iy, ix, value) per allowed pair.
    NumPy and pyopencl arrays as with SinglePhaseForwardOperator (NumPy coefficients are uploaded whole: they are
    small). FISTAHuber and FISTAL2 run on it with the element-wise proxes (no TV, no streaming, no fused update: the
    four coefficient-sized arrays are small); the support constraint is part of the operator.

    Parameters (besides those of SinglePhaseForwardOperator)
    ----------
    support : LocalSupport for the grid's K orientations on the (Ny, Nx) grid. Restricted to the field of view
        unless restrict_to_fov=False. Orientations without any voxel are left out of the PF matrix.
    split_patches, merge_gap, min_box_fill : how the pairs are grouped into patches (build_patches): an
        orientation whose voxels fill less than min_box_fill of their bounding box is split into clusters
        (voxels within merge_gap of each other stay together). split_patches=False: one box per orientation.
    pf_chunk : segments per chunk of the PF products in local memory (default: as many as 32 KB hold for TY lines).

    Only the native projector (radon_kernel "trapezoid" or "hat") and the sparse PF modes ("auto", "sparse",
    "generated"; auto never switches to the dense path here) are supported.
    """

    coefficient_layout = "local"

    def __init__(self, cfg, material, grid, max_gb, support: LocalSupport, verbose=False, split_patches=True,
                 merge_gap=2, min_box_fill=0.5, restrict_to_fov=True, pf_chunk=None, **kwargs):
        t0 = time.perf_counter()
        projector = kwargs.pop("projector", "native")
        if projector not in ("native", "auto"):
            raise ValueError("LocalSinglePhaseForwardOperator has its own projector (the native discretisation); "
                             f"projector={projector!r} is not supported")
        if kwargs.get("pf_mode", "auto") == "dense":
            raise ValueError("LocalSinglePhaseForwardOperator needs a sparse PF mode ('auto', 'sparse', 'generated')")
        kwargs.setdefault("sparse_max_fill", 1.0)            # never the dense path
        kwargs.setdefault("sparse_max_fill_partial", 1.0)
        c = {**cfg, **kwargs}
        Nx, Ny, My = int(c["Nx"]), int(c["Ny"]), int(c["My"])

        mats, sigmas = _grid_arrays(grid)
        if not isinstance(support, LocalSupport):
            raise TypeError("support must be a LocalSupport")
        if support.K != len(mats) or (support.Ny, support.Nx) != (Ny, Nx):
            raise ValueError(f"support is for K = {support.K} on {support.Ny} x {support.Nx}; the grid has "
                             f"{len(mats)} orientations, the geometry {Ny} x {Nx} voxels")
        if restrict_to_fov:
            n_or = int(c["N_Omega"])
            rng = np.array(c["angle_range"], dtype=np.float64) / 180 * np.pi
            delta = (rng[1] - rng[0]) / n_or
            angles = np.linspace(rng[0], rng[1], n_or, endpoint=False) + delta / 2
            fov = fov_support_mask(Nx, Ny, My, angles=angles, image_width=Nx, detector_width=My,
                                   detector_shift=c["cor_offset"])
            support = support.restrict(fov)
        if support.n_pairs == 0:
            raise ValueError("the support is empty (inside the field of view)")
        self.support = support
        self.K_basis = len(mats)
        self.orientation_index = support.orientations_used()     # operator orientation -> grid orientation
        k_int = np.searchsorted(self.orientation_index, support.k)
        self._patches = build_patches(k_int, support.iy, support.ix, len(self.orientation_index),
                                      split=split_patches, merge_gap=merge_gap, min_box_fill=min_box_fill)
        self.n_coef = self._patches["n_coef"]
        self.n_patches = len(self._patches["pk"])
        self._pf_chunk = pf_chunk
        t_patch = time.perf_counter() - t0

        inner = MatrixGrid(mats[self.orientation_index], sigmas[self.orientation_index])
        super().__init__(cfg, material, inner, max_gb, verbose=verbose, projector="native", **kwargs)
        if self.pf_mode == "dense":
            self.free_memory()
            raise ValueError("LocalSinglePhaseForwardOperator: the PF matrix could not be set up sparse "
                             "(16-bit index range or widths of 20 degrees and more)")
        self._setup_local()
        if verbose:
            P = self._patches
            print(f"Local basis: {support.n_pairs} pairs ({100 * support.fill:.2f} % of K x Ny x Nx), "
                  f"{len(self.orientation_index)} of {self.K_basis} orientations, {self.n_patches} patches, "
                  f"{self.n_coef} coefficients ({100 * self.n_coef / (self.K_basis * Nx * Ny):.2f} %), "
                  f"strips {self._strip_bins / (self.N_Omega * self.My * len(self.orientation_index)) * 100:.1f} % "
                  f"of the sinograms; patches {t_patch:.1f} s, set up {time.perf_counter() - t0:.1f} s")

    # ------------------------------------------------------------------ base-class hooks

    def allocate_coefficient_buffer(self, dense=True):
        """No image or sinogram batch buffers (the strips are allocated in _setup_local); the dense PF batch only
        while the sparse matrix is built from it."""
        self.coeffs_sino_C = None
        self._img_k = None
        self._basis_batch_kmax = None
        if dense:
            self._basis_batch_kmax = clarray.empty(self.queue, (self.N_Omega, self.K_batch_max, self.N_eta,
                                                                self.N_peaks), np.float32)

    def default_sparse_budget_gb(self):
        """As SinglePhaseForwardOperator's, with the solver's coefficient arrays (4, unfused FISTA) of n_coef."""
        coeff_bytes = 4 * self.n_coef
        data_bytes = 4 * self.N_Omega * self.My * self.N_seg
        from ..utils.device_memory import free_device_bytes
        available = self.queue.device.global_mem_size
        free_now = free_device_bytes(self.queue.device)
        if free_now is not None:
            available = min(available, free_now)
        strips = 4 * self.N_Omega * 64 * self.K_batch_max          # a guess (the strips are not known yet)
        free = available - 4 * coeff_bytes - 2 * data_bytes - strips - 1024**3
        return 0.5 * max(free, 0) / 1024**3

    def adjoint_batches_cl(self, *args, **kwargs):
        raise NotImplementedError("the local-basis operator has no batched adjoint (no fused FISTA update)")

    @property
    def coeff_shape(self):
        """(n_coef,): the flat vector of the patches' coefficients."""
        return (self.n_coef,)

    # ------------------------------------------------------------------ set-up

    def _setup_local(self):
        q = self.queue
        P = self._patches
        R, My = int(self.N_Omega), int(self.My)
        geo = self.radon.geo_gpu.get().astype(np.float64)
        X, Y, T0 = geo[:, 0], geo[:, 1], geo[:, 2]
        trap = self.radon.kernel == "trapezoid"
        hw = 0.5 * (1.0 + np.maximum(np.abs(X), np.abs(Y))) if trap else np.zeros(R)
        ntap = int(self.radon.n_taps)
        n_tiles = -(-My // TY)
        self._n_tiles = n_tiles

        options = [f"-DNB={self.radon.bins_per_item}", f"-DTY={TY}"]
        if trap:
            options += ["-DTRAPEZOID", f"-DNTAP={ntap}"]
        src = Path(__file__).with_name("local_kernels.cl").read_text()
        self._lprg = cl.Program(self.ctx, src).build(options=options)
        self._k_rf = self._lprg.local_radon_forward
        self._k_rb = self._lprg.local_radon_backward
        self._k_pff = self._lprg.local_pf_forward
        self._k_pfa = self._lprg.local_pf_adjoint
        CP = int(self.N_seg)
        if 4 * TY * self.K_batch_max > self.queue.device.local_mem_size - 1024:
            raise ValueError(f"orientation batches of {self.K_batch_max} need {4 * TY * self.K_batch_max} bytes of local "
                             f"memory (the device has {self.queue.device.local_mem_size}); lower sparse_batch_max")
        if self._pf_chunk is None:
            budget = min(32 * 1024, self.queue.device.local_mem_size - 2048)
            self._jc = int(min(CP, max(32, budget // (4 * TY))))
        else:
            self._jc = int(min(CP, self._pf_chunk))

        bounds = np.searchsorted(P["pk"], [b["k_start"] for b in self.batches] + [self.K])
        self.local_batches = []
        max_strip = 0
        strip_bins = 0
        for ib, b in enumerate(self.batches):
            p0, p1 = int(bounds[ib]), int(bounds[ib + 1])
            if p1 == p0:
                self.local_batches.append(None)
                continue
            y0, x0, h, w = (P[n][p0:p1] for n in ("y0", "x0", "h", "w"))
            nb = p1 - p0
            # detector coordinate of the box corners at every angle
            ta, tb = np.outer(x0, X), np.outer(x0 + w - 1, X)
            ua, ub = np.outer(y0, Y), np.outer(y0 + h - 1, Y)
            tmin = np.minimum(ta, tb) + np.minimum(ua, ub) + T0
            tmax = np.maximum(ta, tb) + np.maximum(ua, ub) + T0
            if trap:
                off = np.floor(tmin - hw) - 1
                last = np.floor(tmax + hw) + ntap + 1
                lo, hi = np.floor(tmin - hw), np.floor(tmax + hw) + 1      # lines that can be non-zero (+ margin)
            else:
                off = np.floor(tmin) - 1
                last = np.floor(tmax) + 3
                lo, hi = np.floor(tmin) - 1, np.floor(tmax) + 2
            Wp = (last - off + 1).max(axis=1).astype(np.int64)
            soff = np.r_[0, np.cumsum(R * Wp)[:-1]].astype(np.int64)
            n_strip = int(R * Wp.sum())
            if n_strip >= 2**31:
                raise ValueError("too many strip bins in one batch for 32-bit indices; lower max_gb")
            strip_bins += int(Wp.sum()) * R
            # tiles of TY lines that every (patch, angle) reaches, in the order (angle, tile, patch)
            valid = (hi >= 0) & (lo <= My - 1)
            t_lo = (np.clip(lo, 0, My - 1) // TY).astype(np.int64)
            t_hi = (np.clip(hi, 0, My - 1) // TY).astype(np.int64)
            cnt = np.where(valid, t_hi - t_lo + 1, 0).ravel()
            n_ent = int(cnt.sum())
            first = np.r_[0, np.cumsum(cnt)[:-1]]
            pid = np.repeat(np.repeat(np.arange(nb), R), cnt)
            aid = np.repeat(np.tile(np.arange(R), nb), cnt)
            tid = np.repeat(t_lo.ravel(), cnt) + (np.arange(n_ent) - np.repeat(first, cnt))
            key = aid * n_tiles + tid
            order = np.argsort(key, kind="stable")                         # patches stay ascending per tile
            tile_patch = pid[order].astype(np.int32)
            tile_ptr = np.r_[0, np.cumsum(np.bincount(key, minlength=R * n_tiles))].astype(np.int32)
            # backprojection work-groups: (patch, first pixel)
            nblk = -(-(h * w) // LS_BACK)
            blk = np.stack([np.repeat(np.arange(nb), nblk),
                            np.concatenate([np.arange(n) * LS_BACK for n in nblk])], axis=1).astype(np.int32)
            box = np.stack([y0, x0, h, w], axis=1).astype(np.int32)
            dev = lambda a: clarray.to_device(q, np.ascontiguousarray(a))
            self.local_batches.append(dict(
                n=nb, n_strip=n_strip, box=dev(box), coff=dev(P["coff"][p0:p1].astype(np.int64)),
                W=dev(Wp.astype(np.int32)), soff=dev(soff.astype(np.int32)), off=dev(off.astype(np.int32)),
                pk=dev((P["pk"][p0:p1] - b["k_start"]).astype(np.int32)), tile_ptr=dev(tile_ptr),
                tile_patch=dev(tile_patch if n_ent else np.zeros(1, np.int32)), blk=dev(blk), n_blk=len(blk)))
            max_strip = max(max_strip, n_strip)
        self._strip_bins = strip_bins
        self._strips = clarray.empty(q, (max(max_strip, 1),), np.float32)
        self._mask = clarray.to_device(q, P["mask"])
        self._alpha = np.float32(R / np.pi)       # as SinglePhaseForwardOperator._adjoint_batch

    # ------------------------------------------------------------------ kernels

    def _radon_forward(self, coeffs, lb):
        R = int(self.N_Omega)
        nab = -(-R // RADON_AB)
        self._k_rf(self.queue, (LS_RADON, RADON_AB * lb["n"] * nab), (LS_RADON, RADON_AB), coeffs.data,
                   self._mask.data, self._strips.data, lb["box"].data, lb["coff"].data, lb["W"].data,
                   lb["soff"].data, lb["off"].data, self.radon.geo_gpu.data, np.int32(R), self.radon.scale)

    def _radon_backward(self, coeffs, lb):
        self._k_rb(self.queue, (lb["n_blk"] * LS_BACK,), (LS_BACK,), self._strips.data, coeffs.data,
                   self._mask.data, lb["blk"].data, lb["box"].data, lb["coff"].data, lb["W"].data,
                   lb["soff"].data, lb["off"].data, self.radon.geo_gpu.data, np.int32(self.N_Omega),
                   np.int32(self.My))

    def _pf_args(self, lb):
        return (lb["tile_ptr"].data, lb["tile_patch"].data, lb["pk"].data, lb["W"].data, lb["soff"].data,
                lb["off"].data)

    def _pf_forward(self, lb, sb, Kb, data):
        R = int(self.N_Omega)
        self._k_pff(self.queue, (LS_PF * R * self._n_tiles,), (LS_PF,), self._strips.data, *self._pf_args(lb),
                    sb["row_ptr_f"].data, sb["col_k"].data, sb["val_f"].data, data.data, np.int32(R),
                    np.int32(self.My), np.int32(self.N_seg), np.int32(Kb), np.int32(self._n_tiles),
                    cl.LocalMemory(4 * TY * Kb))

    def _pf_adjoint(self, lb, sb, Kb, data):
        R = int(self.N_Omega)
        self._k_pfa(self.queue, (LS_PA * R * self._n_tiles,), (LS_PA,), data.data, *self._pf_args(lb),
                    sb["row_ptr_a"].data, sb["col_j"].data, sb["val_a"].data, self._strips.data, np.int32(R),
                    np.int32(self.My), np.int32(self.N_seg), np.int32(Kb), np.int32(self._n_tiles), np.int32(self._jc),
                    self._alpha, cl.LocalMemory(4 * TY * self._jc))

    # ------------------------------------------------------------------ operator

    def direct_cl(self, coeffs, data):
        """data = PF @ Radon(coeffs) in place. coeffs: (n_coef,) float32; data: (N_Omega, My, N_seg), overwritten."""
        check_device(coeffs, self.coeff_shape, "coeffs")
        check_device(data, self.data_shape, "data")
        data.fill(0.0)
        for ib, lb in enumerate(self.local_batches):
            if lb is None:
                continue
            self._radon_forward(coeffs, lb)
            self._pf_forward(lb, self._pf_batch(ib, forward=True), self.batches[ib]["K_batch"], data)

    def adjoint_cl(self, data, coeffs):
        """coeffs = A^T data in place (zero outside the support)."""
        check_device(data, self.data_shape, "data")
        check_device(coeffs, self.coeff_shape, "coeffs")
        for ib, lb in enumerate(self.local_batches):
            if lb is None:
                continue
            # bins the PF product does not write are read with weight 0 by the backprojection: keep them finite
            cl.enqueue_fill_buffer(self.queue, self._strips.data, np.float32(0), 0, 4 * lb["n_strip"])
            self._pf_adjoint(lb, self._pf_batch(ib, forward=False), self.batches[ib]["K_batch"], data)
            self._radon_backward(coeffs, lb)

    def direct(self, coeffs):
        """The forward operator, allocating its output: NumPy (n_coef,) -> NumPy (N_Omega, My, N_seg), or pyopencl
        -> pyopencl."""
        if isinstance(coeffs, clarray.Array):
            data = clarray.empty(self.queue, self.data_shape, np.float32)
            self.direct_cl(coeffs, data)
            return data
        x = clarray.to_device(self.queue, as_host(coeffs, self.coeff_shape, "coeffs"))
        data = clarray.empty(self.queue, self.data_shape, np.float32)
        self.direct_cl(x, data)
        out = data.get()
        x.base_data.release()
        data.base_data.release()
        return out

    def adjoint(self, data):
        """The adjoint, allocating its output: NumPy (N_Omega, My, N_seg) -> NumPy (n_coef,), or pyopencl ->
        pyopencl."""
        if isinstance(data, clarray.Array):
            coeffs = clarray.empty(self.queue, self.coeff_shape, np.float32)
            self.adjoint_cl(data, coeffs)
            return coeffs
        d = clarray.to_device(self.queue, as_host(data, self.data_shape, "data"))
        coeffs = clarray.empty(self.queue, self.coeff_shape, np.float32)
        self.adjoint_cl(d, coeffs)
        out = coeffs.get()
        d.base_data.release()
        coeffs.base_data.release()
        return out

    # ------------------------------------------------------------------ coefficients

    def _pairs(self):
        s = self.support
        return s.k, s.iy, s.ix, self._patches["pair_index"]

    def to_dense(self, x):
        """(K, Ny, Nx) float32 array in the grid's orientation order (zero outside the support)."""
        x = x.get() if isinstance(x, clarray.Array) else np.asarray(x)
        k, iy, ix, pi = self._pairs()
        out = np.zeros((self.K_basis, self.Ny, self.Nx), dtype=np.float32)
        out[k, iy, ix] = x[pi]
        return out

    def from_dense(self, coeffs):
        """The flat coefficients of a (K, Ny, Nx) array (its values outside the support are dropped)."""
        coeffs = np.asarray(coeffs)
        if coeffs.shape != (self.K_basis, self.Ny, self.Nx):
            raise ValueError(f"coeffs must have shape {(self.K_basis, self.Ny, self.Nx)}, not {coeffs.shape}")
        k, iy, ix, pi = self._pairs()
        x = np.zeros(self.n_coef, dtype=np.float32)
        x[pi] = coeffs[k, iy, ix]
        return x

    def to_coo(self, x):
        """(k, iy, ix, value) of every allowed pair (k in the grid's orientation order)."""
        x = x.get() if isinstance(x, clarray.Array) else np.asarray(x)
        k, iy, ix, pi = self._pairs()
        return k.copy(), iy.copy(), ix.copy(), x[pi].astype(np.float32)

    def coefficient_mask(self, mask=None):
        """(n_coef,) float32: 1 for the allowed coefficients (and, given an (Ny, Nx) mask, only where it is True)."""
        m = self._patches["mask"].astype(np.float32)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            k, iy, ix, pi = self._pairs()
            m[pi[~mask[iy, ix]]] = 0.0
        return m

    # ------------------------------------------------------------------ memory

    def device_bytes(self):
        n = super().device_bytes()
        for lb in getattr(self, "local_batches", None) or []:
            if lb is not None:
                n += sum(v.nbytes for v in lb.values() if isinstance(v, clarray.Array))
        for name in ("_strips", "_mask"):
            a = getattr(self, name, None)
            if a is not None:
                n += a.nbytes
        return n

    def free_memory(self):
        for lb in getattr(self, "local_batches", None) or []:
            if lb is not None:
                for v in lb.values():
                    if isinstance(v, clarray.Array) and v.base_data is not None:
                        v.base_data.release()
        self.local_batches = None
        for name in ("_strips", "_mask"):
            a = getattr(self, name, None)
            if a is not None and a.base_data is not None:
                a.base_data.release()
            setattr(self, name, None)
        super().free_memory()
