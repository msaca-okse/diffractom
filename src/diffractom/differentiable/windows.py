"""
Radial integration windows as data channels (the multi-phase rock pipelines), with the derivatives of the model with
respect to the lattice parameters and the relative peak width (float64 reference; docs/differentiable_model.md).

A window [lo, hi] (in q, 1 / Angstrom) holds every reflection family f whose Gaussian profile (width sig_rel q_f)
reaches into it, with the fraction inside

    w_f = (erf(u_hi) - erf(u_lo)) / 2,   u = (edge - q_f) / (sqrt(2) sig_rel q_f),   q_f = 2 pi sqrt(h^T G* h)

(G* the reciprocal metric of the lattice). As in SinglePhaseForwardOperator(rings=...), the window's pole figure
mixes its families' pole figures, evaluated at the window's two-theta (the w-weighted mean of the families'), with the
shares m_f w_f / sum(m w) (normalized=True, m the multiplicity) or I_f w_f (normalized=False, I_f the integrated
intensity of the intensity model, held at the built lattice). The lattice moves q_f, hence the shares
and the window's two-theta; the directions of the poles in the crystal frame are kept (exact for cubic phases; for
lower symmetry they also turn with the lattice, which is not included).
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import erf

LATTICE = ("a", "b", "c", "alpha", "beta", "gamma")


def metric(lat):
    """Direct metric tensor G (Angstrom^2) and its derivatives (6, 3, 3) with respect to a, b, c (Angstrom) and alpha,
    beta, gamma (degrees)."""
    a, b, c = lat[:3]
    al, be, ga = np.deg2rad(lat[3:])
    ca, cb, cg = np.cos(al), np.cos(be), np.cos(ga)
    sa, sb, sg = np.sin(al), np.sin(be), np.sin(ga)
    G = np.array([[a * a, a * b * cg, a * c * cb], [a * b * cg, b * b, b * c * ca], [a * c * cb, b * c * ca, c * c]])
    dG = np.zeros((6, 3, 3))
    dG[0] = [[2 * a, b * cg, c * cb], [b * cg, 0, 0], [c * cb, 0, 0]]
    dG[1] = [[0, a * cg, 0], [a * cg, 2 * b, c * ca], [0, c * ca, 0]]
    dG[2] = [[0, 0, a * cb], [0, 0, b * ca], [a * cb, b * ca, 2 * c]]
    r = np.pi / 180.0
    dG[3] = np.array([[0, 0, 0], [0, 0, -b * c * sa], [0, -b * c * sa, 0]]) * r
    dG[4] = np.array([[0, 0, -a * c * sb], [0, 0, 0], [-a * c * sb, 0, 0]]) * r
    dG[5] = np.array([[0, -a * b * sg, 0], [-a * b * sg, 0, 0], [0, 0, 0]]) * r
    return G, dG


def q_and_derivative(hkl, lat):
    """q (n,) in 1 / Angstrom and dq / d(lattice) (n, 6)."""
    G, dG = metric(np.asarray(lat, np.float64))
    Gs = np.linalg.inv(G)
    h = np.asarray(hkl, np.float64)
    q2 = np.einsum("ni,ij,nj->n", h, Gs, h)
    q = 2 * np.pi * np.sqrt(q2)
    dGs = -np.einsum("ij,pjk,kl->pil", Gs, dG, Gs)
    dq = (2 * np.pi) ** 2 * np.einsum("ni,pij,nj->np", h, dGs, h) / (2 * q[:, None])
    return q, dq


def overlap(q, lo, hi, sig_rel):
    """w, dw/dq, dw/dsig_rel of profiles at q in the window [lo, hi]."""
    q = np.asarray(q, np.float64)
    s = np.sqrt(2.0) * sig_rel * q
    uh, ul = (hi - q) / s, (lo - q) / s
    w = 0.5 * (erf(uh) - erf(ul))
    gh, gl = np.exp(-uh * uh) / np.sqrt(np.pi), np.exp(-ul * ul) / np.sqrt(np.pi)
    dw_dq = gh * (-hi / (np.sqrt(2.0) * sig_rel * q * q)) - gl * (-lo / (np.sqrt(2.0) * sig_rel * q * q))
    dw_ds = gh * (-uh / sig_rel) - gl * (-ul / sig_rel)
    return w, dw_dq, dw_ds


def windows_to_rings(material, windows, sig_rel, min_overlap=1e-3, families=None, lattice=None):
    """rings for SinglePhaseForwardOperator: per window (lo, hi), the families with an overlap above min_overlap (or
    the given families: a list of index arrays per window) and their overlaps, at the material's lattice parameters
    (or at lattice)."""
    lat = [material.lattice_params[k] for k in LATTICE] if lattice is None else lattice
    q, _ = q_and_derivative(material.reflections["hkl"], lat)
    rings = []
    for p, (lo, hi) in enumerate(windows):
        w, _, _ = overlap(q, lo, hi, sig_rel)
        ix = np.nonzero(w > min_overlap)[0] if families is None else np.asarray(families[p], dtype=np.int64)
        rings.append((ix, w[ix]))
    return rings


def _window_parts(q, dq, windows, families, sig, lam, amp, normalized):
    """Per window: overlaps and their derivatives, the mixing shares s_f, the window's two-theta (overlap-weighted
    mean of its families') and the pieces of the chain rule. amp: per reflection, m (normalized) or the integrated
    intensity I (normalized=False); s_f = amp_f w_f / sum(amp w) or amp_f w_f."""
    out = []
    for (lo, hi), fam in zip(windows, families):
        w, dw_dq, dw_ds = overlap(q[fam], lo, hi, sig)
        tth = 2 * np.arcsin(lam * q[fam] / (4 * np.pi))
        dtth_dq = lam / (2 * np.pi * np.cos(tth / 2))
        W = w.sum()
        wt = w / W if W > 0 else np.full(len(w), 1.0 / max(len(w), 1))   # no overlap: plain mean (the operator's)
        a = amp[fam]
        S = (a * w).sum()
        share = (a * w / S if S > 0 else np.zeros(len(w))) if normalized else a * w
        out.append(dict(fam=fam, w=w, dw_dq=dw_dq, dw_ds=dw_ds, dq=dq[fam], share=share, a=a, S=S, tth=tth,
                        dtth_dq=dtth_dq, tthw=float((wt * tth).sum()), wt=wt, W=W))
    return out


def _share_term(pt, T, normalized):
    """dL/dw_f through the shares, from T_f = <G, amp_f PF_f>-type products: normalized: T_f = <G, PF_f>,
    dL/dw = m_f / S (T_f - sum(share T)); normalized=False: T_f = <G, I_f PF_f> = dL/dw_f."""
    if normalized:
        return pt["a"] / pt["S"] * (T - (pt["share"] * T).sum())
    return T


def _lattice_chain(parts, dL_dw, dL_dtthw):
    """Gradients with respect to the lattice parameters and sig_rel, from per window dL/dw_f (through the shares)
    and dL/d(window two-theta)."""
    g_lat, g_sig = np.zeros(6), 0.0
    for pt, dw, dt in zip(parts, dL_dw, dL_dtthw):
        dtthw_dw = (pt["tth"] - pt["tthw"]) / pt["W"] if pt["W"] > 0 else np.zeros(len(pt["w"]))
        dL_dw_tot = dw + dt * dtthw_dw
        dL_dq = dL_dw_tot * pt["dw_dq"] + dt * pt["wt"] * pt["dtth_dq"]
        g_lat += dL_dq @ pt["dq"]
        g_sig += float((dL_dw_tot * pt["dw_ds"]).sum())
    return g_lat, g_sig


def _amplitudes(material, lam, normalized, intensity_model):
    if normalized:
        return np.asarray(material.reflections["multiplicity"], np.float64)
    from ..crystallography.intensity import reflection_intensities
    return np.asarray(reflection_intensities(material, lam, intensity_model), np.float64)


class WindowedReference:
    """ReferenceModel of an operator built with rings=windows_to_rings(material, windows, sig_rel[, families=...]),
    with the extra parameters "lattice" (6,: a, b, c in Angstrom, alpha, beta, gamma in degrees) and "sig_rel".
    normalized=True: shares m_f w_f / sum(m w); normalized=False: the window's pole figure is sum_f I_f w_f PF_f with
    the integrated intensities I_f of the operator's intensity model, held at the built lattice (the lattice moves
    the overlaps and the two-theta, not |F|^2 or the Lorentz factor)."""

    def __init__(self, ref, op, windows, sig_rel):
        self.ref, self.windows, self.sig0 = ref, [tuple(map(float, w)) for w in windows], float(sig_rel)
        mat = op.material
        self.normalized = bool(op.normalized)
        self.lat0 = np.array([mat.lattice_params[k] for k in LATTICE], np.float64)
        self.hkl = np.asarray(mat.reflections["hkl"], np.float64)
        self.lam = 12.398 / float(op.cfg["energy"])
        self.amp = _amplitudes(mat, self.lam, self.normalized, op.intensity_model)
        self.families = [np.asarray(ix) for ix in op.ring_reflections]
        h = np.asarray(mat.h_vecs_normed, np.float64)
        sym = np.asarray(mat.point_group_matrices, np.float64).reshape(-1, 3, 3)
        self.axes = {}                                   # per family: distinct axes and their image counts
        for f in np.unique(np.concatenate(self.families)):
            img = sym @ h[f]
            first = np.argmax(np.abs(img) > 1e-6, axis=1)
            img = img * np.sign(img[np.arange(len(img)), first])[:, None]
            uniq, inv = np.unique(np.round(img, 6), axis=0, return_inverse=True)
            self.axes[int(f)] = (uniq / np.linalg.norm(uniq, axis=1, keepdims=True),
                                 np.bincount(inv.ravel(), minlength=len(uniq)).astype(np.float64))

    def params0(self):
        prm = self.ref.params0()
        prm["lattice"], prm["sig_rel"] = self.lat0.copy(), self.sig0
        return prm

    def _windows(self, prm):
        q, dq = q_and_derivative(self.hkl, prm["lattice"])
        return _window_parts(q, dq, self.windows, self.families, prm["sig_rel"], self.lam, self.amp, self.normalized)

    def _family_pf(self, prm, v, f):
        """PF of family f (R, K, C) at a window's probed directions v (R, S, C, Se, 3), with the per-node factors of
        its derivative: (pf, poles, dg/dD, counts, normalisation)."""
        ref = self.ref
        axes, cnt = self.axes[int(f)]
        n = np.einsum("kij,aj->kai", Rotation.from_rotvec(prm["rot"]).as_matrix() @ ref.U0, axes)
        sig = np.asarray(prm["sigma"], np.float64)
        norm = 1.0 / (8.0 * np.pi * sig ** 2)
        d = np.einsum("rscex,kax->rsceka", v, n)
        g, dgd, _ = ref._g(d, (1.0 / sig ** 2)[:, None])
        pf = norm[None, :, None] * np.einsum("rsceka,a,se->rkc", g, cnt, ref.qw)
        return pf, n, dgd, cnt, norm

    def pf(self, prm):
        ref = self.ref
        win = self._windows(prm)
        p2 = dict(prm)
        p2["two_theta"] = np.array([wi["tthw"] for wi in win])
        v = ref.probed(p2)
        out = np.zeros((ref.R, ref.K, ref.C, ref.P))
        for j, wi in enumerate(win):
            for f, sh in zip(wi["fam"], wi["share"]):
                out[..., j] += sh * self._family_pf(prm, v[..., j, :], f)[0]
        return out

    def forward(self, c, prm):
        ref = self.ref
        R, K, CP = ref.R, ref.K, ref.C * ref.P
        PF = self.pf(prm).reshape(R, K, CP)
        Pm, _ = ref.projection_matrices(prm)
        S = np.einsum("asp,kp->ask", Pm, np.asarray(c, np.float64).reshape(K, -1))
        return np.einsum("akj,ask->asj", PF, S)

    def vjp(self, c, prm, r):
        """Gradients of <r, d> with respect to "lattice" (6,) and "sig_rel"."""
        ref = self.ref
        R, K, C, P, My = ref.R, ref.K, ref.C, ref.P, ref.My
        r = np.asarray(r, np.float64).reshape(R, My, C * P)
        Pm, _ = ref.projection_matrices(prm)
        S = np.einsum("asp,kp->ask", Pm, np.asarray(c, np.float64).reshape(K, -1))
        G = np.einsum("asj,ask->akj", r, S).reshape(R, K, C, P)
        win = self._windows(prm)
        p2 = dict(prm)
        p2["two_theta"] = np.array([wi["tthw"] for wi in win])
        v = ref.probed(p2)
        dv = ref.probed_derivatives(p2, ("two_theta",))["two_theta"]
        dL_dw, dL_dtthw = [], []
        for j, wi in enumerate(win):
            Gj = G[..., j]
            T, dT = [], []                   # <G, PF_f>, <G, d PF_f / d(2 theta)>
            for f in wi["fam"]:
                pf, n, dgd, cnt, norm = self._family_pf(prm, v[..., j, :], f)
                dd = np.einsum("rscex,kax->rsceka", dv[..., j, :], n)
                dpf = norm[None, :, None] * np.einsum("rsceka,rsceka,a,se->rkc", dgd, dd, cnt, ref.qw)
                T.append(float((Gj * pf).sum()))
                dT.append(float((Gj * dpf).sum()))
            T, dT = np.array(T), np.array(dT)
            dL_dw.append(_share_term(wi, T if self.normalized else wi["a"] * T, self.normalized))
            dL_dtthw.append(float((wi["share"] * dT).sum()))
        g_lat, g_sig = _lattice_chain(win, dL_dw, dL_dtthw)
        return {"lattice": g_lat, "sig_rel": g_sig}


_MIX_SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// window data d (nb, P) from family-channel data u (nb, F): d[b, p] = sum_{ch in window p} share[ch] u[b, ch]
__kernel void mix(__global const float *u, __global float *d, __global const int *wstart, __global const int *wch,
                  __global const float *share, const int F, const int P, const long n, const int accumulate)
{
    const long i = get_global_id(0);
    if (i >= n) return;
    const long b = i / P;
    const int p = (int)(i - b * P);
    float acc = 0.0f;
    for (int k = wstart[p]; k < wstart[p + 1]; k++) { const int ch = wch[k]; acc += share[ch] * u[b * F + ch]; }
    d[i] = accumulate ? d[i] + acc : acc;
}
// its transpose: u[b, ch] = share[ch] r[b, window(ch)]
__kernel void unmix(__global const float *r, __global float *u, __global const int *cwin,
                    __global const float *share, const int F, const int P, const long n)
{
    const long i = get_global_id(0);
    if (i >= n) return;
    const long b = i / F;
    const int ch = (int)(i - b * F);
    u[i] = share[ch] * r[b * P + cwin[ch]];
}
// T[ch] = sum_b r[b, window(ch)] u[b, ch] (one work-group per channel)
__kernel void channel_dot(__global const float *r, __global const float *u, __global const int *cwin,
                          __global double *T, const int F, const int P, const long nb, __local double *loc)
{
    const int ch = get_group_id(0), l = get_local_id(0), ls = get_local_size(0);
    const int p = cwin[ch];
    double acc = 0.0;
    for (long b = l; b < nb; b += ls) acc += (double)r[b * P + p] * (double)u[b * F + ch];
    loc[l] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = ls / 2; s > 0; s >>= 1) {
        if (l < s) loc[l] += loc[l + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0) T[ch] = loc[0];
}
"""


class WindowedOperator:
    """GPU version of WindowedReference: an operator whose data channels are single reflection families, each
    evaluated at its window's two-theta (rings=[([f], [1])], ring_two_theta), and a mixing layer that adds the
    families of a window with their shares:

        d[a, s, c, p] = sum_{f in window p} share_f(lattice, sig_rel) u[a, s, c, f],   u = family_op(c),

    share_f = m_f w_f / sum(m w) (normalized=True) or w_f (normalized=False: the family channel carries the
    integrated intensity I_f of the intensity model, held at the built lattice). The mixing runs on the GPU.

    windows: (lo, hi) in q (1 / Angstrom); families: optional list of reflection-index arrays per window (default:
    every family with an overlap above min_overlap at the material's lattice); work: optional float32 clarray of at
    least family_size elements, shared between operators (default: one of its own).
    direct_cl / adjoint_cl (GPU arrays) and forward / vjp (NumPy) work on window data (N_Omega, My, N_eta * N_windows);
    lattice_vjp gives the gradients with respect to the lattice parameters and sig_rel (the shares analytically, the
    channels' two-theta with PFGradient of the family operator). set_lattice(lattice, sig_rel) moves the channels'
    two-theta (rebuilds the family operator's PF matrix). The remaining keyword arguments (cfg, grid, max_gb,
    pf_mode, intensity_model, ...) go to the family operator."""

    def __init__(self, material, windows, sig_rel, min_overlap=1e-3, families=None, work=None, **op_kwargs):
        import pyopencl as cl
        import pyopencl.array as clarray
        from ..operators.single_phase_forward_operator import SinglePhaseForwardOperator
        self.material, self.windows, self.sig = material, [tuple(map(float, w)) for w in windows], float(sig_rel)
        self.normalized = bool(op_kwargs.get("normalized", True))
        self.lat = np.array([material.lattice_params[k] for k in LATTICE], np.float64)
        self.hkl = np.asarray(material.reflections["hkl"], np.float64)
        self.lam = 12.398 / float(op_kwargs["cfg"]["energy"])
        rings = windows_to_rings(material, self.windows, self.sig, min_overlap, families=families)
        self.families = [np.asarray(ix) for ix, _ in rings]
        self.channel_window = np.concatenate([[p] * len(f) for p, f in enumerate(self.families)]).astype(np.int32)
        self.channel_family = np.concatenate(self.families).astype(int)
        self.P, self.F = len(self.windows), len(self.channel_family)
        self.amp = _amplitudes(material, self.lam, self.normalized, op_kwargs.get("intensity_model"))
        info = self._info(self.lat, self.sig)
        op_kwargs = dict(op_kwargs, material=material, normalized=self.normalized,
                         rings=[([f], [1.0]) for f in self.channel_family], ring_two_theta=info["tth_ch"])
        self.op = SinglePhaseForwardOperator(**op_kwargs)
        self.q = self.op.queue
        self.C = int(self.op.N_eta)
        self.data_shape = (int(self.op.N_Omega), int(self.op.My), self.C * self.P)
        self.coeff_shape = self.op.coeff_shape
        self.nb = int(self.op.N_Omega) * int(self.op.My) * self.C
        self.family_size = self.nb * self.F
        if work is not None and work.size < self.family_size:
            raise ValueError(f"work: {work.size} elements, {self.family_size} needed")
        self.work = work if work is not None else clarray.empty(self.q, (self.family_size,), np.float32)
        prg = cl.Program(self.op.ctx, _MIX_SRC).build()
        self.k_mix, self.k_unmix, self.k_dot = prg.mix, prg.unmix, prg.channel_dot
        order = np.argsort(self.channel_window, kind="stable")
        wstart = np.searchsorted(self.channel_window[order], np.arange(self.P + 1)).astype(np.int32)
        dev = lambda a, t: clarray.to_device(self.q, np.ascontiguousarray(a, dtype=t))
        self.g_wstart, self.g_wch, self.g_cwin = dev(wstart, np.int32), dev(order, np.int32), \
            dev(self.channel_window, np.int32)
        self.T_gpu = clarray.empty(self.q, (self.F,), np.float64)
        self.info = None
        self._set_info(info)
        self._pg = None

    def _info(self, lat, sig):
        """Shares (with the family channel's own amplitude taken out for normalized=False), window two-theta and the
        chain-rule pieces at (lat, sig)."""
        q, dq = q_and_derivative(self.hkl, lat)
        parts = _window_parts(q, dq, self.windows, self.families, sig, self.lam, self.amp, self.normalized)
        share = np.zeros(self.F)
        tthw = np.array([pt["tthw"] for pt in parts])
        for p, pt in enumerate(parts):
            ch = np.nonzero(self.channel_window == p)[0]
            pt["ch"] = ch
            share[ch] = pt["share"] if self.normalized else pt["w"]      # I_f is inside the family channel
        return {"share": share, "tthw": tthw, "tth_ch": tthw[self.channel_window], "parts": parts}

    def _set_info(self, info):
        import pyopencl.array as clarray
        self.info = info
        self.g_share = clarray.to_device(self.q, info["share"].astype(np.float32))

    def set_lattice(self, lattice=None, sig_rel=None):
        from .update import set_pf_parameters
        if lattice is not None:
            self.lat = np.asarray(lattice, np.float64).copy()
        if sig_rel is not None:
            self.sig = float(sig_rel)
        self._set_info(self._info(self.lat, self.sig))
        set_pf_parameters(self.op, two_theta=self.info["tth_ch"])

    def _launch(self, kernel, n, *args):
        ls = 256
        kernel(self.q, (-(-n // ls) * ls,), (ls,), *args)

    def mix_cl(self, u, d, accumulate=False):
        """Window data d (GPU, data_shape) from family-channel data u (GPU, family_size)."""
        n = self.nb * self.P
        self._launch(self.k_mix, n, u.data, d.data, self.g_wstart.data, self.g_wch.data, self.g_share.data,
                     np.int32(self.F), np.int32(self.P), np.int64(n), np.int32(bool(accumulate)))

    def unmix_cl(self, r, u):
        """The transpose of mix_cl."""
        n = self.family_size
        self._launch(self.k_unmix, n, r.data, u.data, self.g_cwin.data, self.g_share.data, np.int32(self.F),
                     np.int32(self.P), np.int64(n))

    def direct_cl(self, c, out, accumulate=False):
        """out (GPU, data_shape) = (or +=) the window prediction of the coefficients c (GPU, coeff_shape)."""
        u = self._fam_view()
        self.op.direct_cl(c, u)
        self.mix_cl(u, out, accumulate)
        return out

    def adjoint_cl(self, r, out):
        """out (GPU, coeff_shape) = the adjoint applied to the window residual r (GPU, data_shape)."""
        u = self._fam_view()
        self.unmix_cl(r, u)
        self.op.adjoint_cl(u, out)
        return out

    def _fam_view(self):
        return self.work[:self.family_size].reshape(self.op.data_shape)

    def forward(self, c):
        import pyopencl.array as clarray
        cg = clarray.to_device(self.q, np.ascontiguousarray(c, dtype=np.float32))
        out = clarray.empty(self.q, self.data_shape, np.float32)
        return self.direct_cl(cg, out).get()

    def lattice_vjp(self, c, r, with_c=False):
        """Gradients of <r, d> with respect to "lattice" (6,) and "sig_rel" (and "c" with with_c). c, r: GPU arrays
        (or NumPy, uploaded)."""
        import pyopencl as cl
        import pyopencl.array as clarray
        from .gradients import PFGradient
        if not isinstance(c, clarray.Array):
            c = clarray.to_device(self.q, np.ascontiguousarray(c, dtype=np.float32))
        if not isinstance(r, clarray.Array):
            r = clarray.to_device(self.q, np.ascontiguousarray(r, dtype=np.float32).reshape(self.data_shape))
        u = self._fam_view()
        self.op.direct_cl(c, u)
        ls = 256
        self.k_dot(self.q, (self.F * ls,), (ls,), r.data, u.data, self.g_cwin.data, self.T_gpu.data,
                   np.int32(self.F), np.int32(self.P), np.int64(self.nb), cl.LocalMemory(8 * ls))
        T = self.T_gpu.get()                                                   # <r_window, u_f> per channel
        self.unmix_cl(r, u)                                                    # family residual (shares applied)
        if self._pg is None:
            self._pg = PFGradient(self.op)
        g_tth = self._pg(c, u, groups=("two_theta",))["two_theta"]             # per channel, shares included
        parts = self.info["parts"]
        dL_dw = [_share_term(pt, T[pt["ch"]], self.normalized) for pt in parts]   # T includes I_f if not normalized
        dL_dtthw = [float(g_tth[pt["ch"]].sum()) for pt in parts]
        g_lat, g_sig = _lattice_chain(parts, dL_dw, dL_dtthw)
        out = {"lattice": g_lat, "sig_rel": g_sig}
        if with_c:
            out["c"] = clarray.empty(self.q, self.coeff_shape, np.float32)
            self.op.adjoint_cl(u, out["c"])
        return out

    def vjp(self, c, r, groups=("lattice", "sig_rel", "c")):
        """NumPy version: gradients of <r, d> with respect to the lattice parameters, sig_rel and the coefficients."""
        g = self.lattice_vjp(c, r, with_c="c" in groups)
        out = {k: g[k] for k in ("lattice", "sig_rel") if k in groups}
        if "c" in groups:
            out["c"] = g["c"].get()
        return out
