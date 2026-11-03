"""
GPU vector-Jacobian products of SinglePhaseForwardOperator with respect to the PF parameters: the rotation and the
kernel width of every basis orientation, the eta offset, the rotation-axis tilt and the Bragg angles (definitions as
in reference.ReferenceModel; docs/differentiable_model.md). Optionally the Gauss-Newton block of every orientation
(rotation and width, 4 x 4) for a given curvature of the loss.
"""
from pathlib import Path

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

from .reference import ReferenceModel
from ..utils.arrays import as_host, check_device

PF_GROUPS = ("rot", "sigma", "eta_offset", "axis_tilt", "two_theta")


class PFGradient:
    """grad = PFGradient(op)(c, r): the gradients of <r, A(c)> with respect to the PF parameters, at the operator's
    current parameters (the orientations of its grid, its sigmas, its geometry).

    op: a SinglePhaseForwardOperator with pf_mode "sparse" or "generated" (the derivative is evaluated on the
    non-zeros of its PF matrix). hessian: also return "hess" (K, 4, 4) = J_k^T diag(curv) J_k for the rotation and
    sigma of orientation k (the exact diagonal blocks of the Gauss-Newton matrix).
    """

    def __init__(self, op, hessian=False, local_size=64):
        if op.pf_mode not in ("sparse", "generated"):
            raise NotImplementedError(f"PFGradient needs pf_mode 'sparse' or 'generated' (this operator: {op.pf_mode!r})")
        self.local = getattr(op, "coefficient_layout", "dense") == "local"
        if getattr(op, "coefficient_layout", "dense") not in ("dense", "local"):
            raise NotImplementedError("PFGradient needs the (K, Ny, Nx) or the local coefficient layout")
        self.op, self.q = op, op.queue
        self.hessian, self.ls = bool(hessian), int(local_size)
        self.P = int(op.N_peaks)
        nh = 10 if self.hessian else 0
        self.q_eta, self.q_tilt, self.q_tth = 4 + nh, 5 + nh, 7 + nh
        self.NQ = self.q_tth + self.P
        src = Path(__file__).with_name("pf_grad.cl").read_text()
        opts = list(op.pf_cut_options) + list(op.quad_options) + [f"-DPMAX={self.P}", f"-DHESS={int(self.hessian)}",
                                                                   f"-DLS={self.ls}"]
        prg = cl.Program(op.ctx, src).build(options=opts)
        self.k_vjp = cl.Kernel(prg, "pf_param_vjp")
        self.k_sum = cl.Kernel(prg, "rows_to_orientations")
        if self.local:
            self.k_scatter = cl.Kernel(cl.Program(op.ctx, _STRIP_SCATTER).build(), "strips_to_sino")
        self.dummy = clarray.zeros(self.q, (1,), np.float32)
        self.dv = {}
        self._alloc()

    def _alloc(self):
        op = self.op
        self.kmax, self.version = op.K_batch_max, getattr(op, "_pf_version", 0)
        self.rows = clarray.empty(self.q, (int(op.N_Omega) * op.K_batch_max * self.NQ,), np.float32)
        if self.local:           # the batch sinogram (R, My, Kmax) the gradient kernel reads, from the patches' strips
            self.sino = clarray.empty(self.q, (int(op.N_Omega) * int(op.My) * op.K_batch_max,), np.float32)
        self.per_k = clarray.empty(self.q, (op.K_batch_max * self.NQ,), np.float32)

    def refresh(self):
        """Forget the derivatives of the probed directions (after the geometry of the operator changed)."""
        for a in self.dv.values():
            if isinstance(a, clarray.Array):
                a.base_data.release()
        self.dv = {}

    def _dv(self, groups):
        need = [g for g in ("eta_offset", "axis_tilt", "two_theta") if g in groups and g not in self.dv]
        if need:
            ref = ReferenceModel(self.op)
            d = ref.probed_derivatives(ref.params0(), need)
            dev = lambda a: clarray.to_device(self.q, np.ascontiguousarray(a, dtype=np.float32))
            for key, val in d.items():
                self.dv[key] = dev(val)
            if "axis_tilt" in need:
                self.dv["axis_tilt"] = True
        get = lambda key: self.dv[key].data if key in self.dv else self.dummy.data
        return get("eta_offset"), get(("axis_tilt", 0)), get(("axis_tilt", 1)), get("two_theta")

    def _project(self, c, ib):
        """The projections of batch ib's coefficients into op.coeffs_sino_C (R, My, K_batch_max) (local layout: into
        self.sino, the patches' strips added at their detector offsets)."""
        op = self.op
        if self.local:
            lb = op.local_batches[ib]
            self.sino.fill(np.float32(0.0))
            if lb is None:
                return self.sino
            op._radon_forward(c, lb)
            R = int(op.N_Omega)
            n = lb["n_strip"]
            self.k_scatter(self.q, (-(-n // 256) * 256,), (256,), op._strips.data, self.sino.data, lb["W"].data,
                           lb["soff"].data, lb["off"].data, lb["pk"].data, np.int32(lb["n"]), np.int32(R),
                           np.int32(op.My), np.int32(op.K_batch_max), np.int32(n))
            return self.sino
        b = op.batches[ib]
        k0, Kb, Kmax = b["k_start"], b["K_batch"], op.K_batch_max
        if op.projector == "fft":
            op.fradon.forward(c, k0, Kb, op.coeffs_sino_C, Kmax)
            return op.coeffs_sino_C
        elif op.projector == "native":
            op.radon.gather(c, op._img_k, k0, Kb, Kmax)
            op.radon.forward(op._img_k, op.coeffs_sino_C, Kmax)
        else:
            op._radon_gratopy_forward(c, k0, Kb)
        return op.coeffs_sino_C

    def __call__(self, c, r, groups=PF_GROUPS, curv=None):
        """c: coefficients (K, Ny, Nx), r: dL/d data (N_Omega, My, N_seg) (NumPy arrays are uploaded); curv: the
        loss's curvature per data point (data shape) for the Gauss-Newton blocks (hessian=True; default 1).
        Returns a dict with the requested groups ("rot" (K, 3), "sigma" (K,), "eta_offset", "axis_tilt" (2,),
        "two_theta" (P,)) and "hess" (K, 4, 4) with hessian=True."""
        op, q = self.op, self.q
        if op.K_batch_max != self.kmax or getattr(op, "_pf_version", 0) != self.version:   # rebuilt operator
            self.refresh()
            self._alloc()
        up = []
        if not isinstance(c, clarray.Array):
            c = clarray.to_device(q, as_host(c, op.coeff_shape, "c"))
            up.append(c)
        else:
            check_device(c, op.coeff_shape, "c")
        if not isinstance(r, clarray.Array):
            r = clarray.to_device(q, as_host(r, op.data_shape, "r"))
            up.append(r)
        else:
            check_device(r, op.data_shape, "r")
        if self.hessian:
            if curv is None:
                curv = clarray.empty(q, op.data_shape, np.float32)
                curv.fill(np.float32(1.0))
                up.append(curv)
            elif not isinstance(curv, clarray.Array):
                curv = clarray.to_device(q, as_host(curv, op.data_shape, "curv"))
                up.append(curv)
        dv_eta, dv_t1, dv_t2, dv_tth = self._dv(groups)
        flags = (1 if "eta_offset" in groups else 0) | (2 if "axis_tilt" in groups else 0) | \
            (4 if "two_theta" in groups else 0)
        R, My, Kmax = int(op.N_Omega), int(op.My), op.K_batch_max
        per_k = np.zeros((op.K, self.NQ), np.float64)
        for ib, b in enumerate(op.batches):
            k0, Kb = b["k_start"], b["K_batch"]
            sino = self._project(c, ib)
            sb = op._pf_batch(ib, forward=False)
            self.k_vjp(q, (R * Kb * self.ls,), (self.ls,), sb["row_ptr_a"].data, sb["col_j"].data, r.data,
                       (curv if self.hessian else self.dummy).data, sino.data, op.coords_gpu.data,
                       dv_eta, dv_t1, dv_t2, dv_tth, op.poles_gpu.data, op.axis_start_gpu.data,
                       op.axis_count_gpu.data, op.inv_sigma2_gpu.data, op.norm_factor_gpu.data,
                       op.quad_weights_gpu.data, op.intens_gpu.data, self.rows.data, np.int32(My), np.int32(Kb),
                       np.int32(Kmax), np.int32(k0), np.int32(op.N_eta), np.int32(self.P), np.int32(op.N_axes),
                       np.int32(op.N_Omega_subdivisions), np.int32(op.N_eta_subdivisions), np.int32(op.normalized),
                       np.int32(flags), cl.LocalMemory(4 * My))
            n = Kb * self.NQ
            self.k_sum(q, (-(-n // 64) * 64,), (64,), self.rows.data, self.per_k.data, np.int32(R), np.int32(Kb))
            per_k[k0:k0 + Kb] = self.per_k[:n].get().reshape(Kb, self.NQ)
        del up
        out = {}
        if "rot" in groups:
            out["rot"] = per_k[:, 0:3].copy()
        if "sigma" in groups:
            out["sigma"] = per_k[:, 3].copy()
        if "eta_offset" in groups:
            out["eta_offset"] = float(per_k[:, self.q_eta].sum())
        if "axis_tilt" in groups:
            out["axis_tilt"] = per_k[:, self.q_tilt:self.q_tilt + 2].sum(axis=0)
        if "two_theta" in groups:
            out["two_theta"] = per_k[:, self.q_tth:self.q_tth + self.P].sum(axis=0)
        if self.hessian:
            iu = np.triu_indices(4)
            H = np.zeros((op.K, 4, 4))
            H[:, iu[0], iu[1]] = per_k[:, 4:14]
            H[:, iu[1], iu[0]] = per_k[:, 4:14]
            out["hess"] = H
        return out


_STRIP_SCATTER = r"""
inline void atomic_add_f(volatile __global float *p, const float v) {
    union { unsigned int u; float f; } old, nw;
    do { old.f = *p; nw.f = old.f + v; } while (atomic_cmpxchg((volatile __global unsigned int *)p, old.u, nw.u) != old.u);
}
// sino[(a * My + s) * Kmax + pk[p]] += strips[soff[p] + a * W[p] + j],  s = off[p * R + a] + j (inside the detector)
__kernel void strips_to_sino(__global const float *strips, __global float *sino, __global const int *W,
                             __global const int *soff, __global const int *off, __global const int *pk,
                             const int P, const int R, const int My, const int Kmax, const int n)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    int lo = 0, hi = P - 1;                       // the patch holding strip bin i (soff ascending)
    while (lo < hi) { const int mid = (lo + hi + 1) / 2; if (soff[mid] <= i) lo = mid; else hi = mid - 1; }
    const int p = lo, rem = i - soff[p], a = rem / W[p], j = rem - a * W[p];
    const int s = off[p * R + a] + j;
    if (s >= 0 && s < My) atomic_add_f(sino + ((size_t)a * My + s) * Kmax + pk[p], strips[i]);
}
"""


class ProjGradient:
    """grad = ProjGradient(op)(c, r): the gradients of <r, A(c)> with respect to a detector shift of every projection
    ("shift", (N_Omega,), in detector bins: t_a(x) <- t_a(x) + shift_a) and the centre of rotation ("cor", in the
    operator's cor_offset units). Native projector, pf_mode "sparse" or "generated"."""

    def __init__(self, op, local_size=256):
        if op.projector != "native":
            raise NotImplementedError("ProjGradient needs the native projector")
        if op.pf_mode not in ("sparse", "generated"):
            raise NotImplementedError("ProjGradient needs pf_mode 'sparse' or 'generated'")
        self.op, self.q, self.ls = op, op.queue, int(local_size)
        opts = ["-DNB=3"] + (["-DTRAPEZOID"] if op.radon.kernel == "trapezoid" else [])
        prg = cl.Program(op.ctx, Path(__file__).with_name("proj_grad.cl").read_text()).build(options=opts)
        self.k_dfwd = cl.Kernel(prg, "radon_dforward_k")
        self.k_dot = cl.Kernel(prg, "shift_dot")
        self.part = clarray.empty(self.q, (int(op.N_Omega),), np.float32)
        self.kmax, self.dS = None, None

    def __call__(self, c, r):
        from ..operators.single_phase_forward_operator import SPMM_TY
        op, q = self.op, self.q
        up = []
        if not isinstance(c, clarray.Array):
            c = clarray.to_device(q, as_host(c, op.coeff_shape, "c"))
            up.append(c)
        if not isinstance(r, clarray.Array):
            r = clarray.to_device(q, as_host(r, op.data_shape, "r"))
            up.append(r)
        R, My, CP, Kmax = int(op.N_Omega), int(op.My), int(op.N_seg), op.K_batch_max
        if self.kmax != Kmax:
            self.kmax, self.dS = Kmax, clarray.empty(q, (R * My * Kmax,), np.float32)
        ls, chunk = op._spmm_setup()
        jc = min(CP, chunk)
        n_strips = -(-My // 3)
        out = np.zeros(R)
        for ib, b in enumerate(op.batches):
            k0, Kb = b["k_start"], b["K_batch"]
            sb = op._pf_batch(ib, forward=False)
            op.k.spmm_pf_adjoint_c(q, (-(-Kb // ls) * ls, -(-My // SPMM_TY), R), (ls, 1, 1), r.data,
                                   sb["row_ptr_a"].data, sb["col_j"].data, sb["val_a"].data, op.coeffs_sino_C.data,
                                   np.int32(R), np.int32(My), np.int32(Kb), np.int32(CP), np.int32(Kmax),
                                   np.float32(1.0), np.int32(jc), cl.LocalMemory(4 * SPMM_TY * jc))   # Z = PF^T r
            op.radon.gather(c, op._img_k, k0, Kb, Kmax)
            k4 = Kmax // 4
            self.k_dfwd(q, (-(-k4 // 32) * 32, -(-n_strips // 4) * 4, R), (32, 4, 1), op._img_k.data, self.dS.data,
                        op.radon.geo_gpu.data, np.int32(op.Nx), np.int32(op.Ny), np.int32(My), np.int32(R),
                        np.int32(k4), np.int32(k4), op.radon.scale)
            self.k_dot(q, (R * self.ls,), (self.ls,), op.coeffs_sino_C.data, self.dS.data, self.part.data,
                       np.int32(My), np.int32(Kmax), np.int32(Kb), cl.LocalMemory(4 * self.ls))
            out += self.part.get()
        del up
        return {"shift": out, "cor": -float(out.sum())}     # t = ... - cor / delta_s, delta_s = 1 bin
