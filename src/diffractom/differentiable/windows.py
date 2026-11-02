"""
Radial integration windows as data channels (the multi-phase rock pipelines), with the derivatives of the model with
respect to the lattice parameters and the relative peak width (float64 reference; docs/differentiable_model.md).

A window [lo, hi] (in q, 1 / Angstrom) holds every reflection family f whose Gaussian profile (width sig_rel q_f)
reaches into it, with the fraction inside

    w_f = (erf(u_hi) - erf(u_lo)) / 2,   u = (edge - q_f) / (sqrt(2) sig_rel q_f),   q_f = 2 pi sqrt(h^T G* h)

(G* the reciprocal metric of the lattice). As in SinglePhaseForwardOperator(rings=...) with normalized=True, the
window's pole figure mixes its families' pole figures with the shares m_f w_f / sum(m w) (m the multiplicity),
evaluated at the window's two-theta, the w-weighted mean of the families'. The lattice moves q_f, hence the shares
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


def windows_to_rings(material, windows, sig_rel, min_overlap=1e-3):
    """rings for SinglePhaseForwardOperator: per window (lo, hi), the families with an overlap above min_overlap and
    those overlaps (at the material's lattice parameters)."""
    lat = [material.lattice_params[k] for k in LATTICE]
    q, _ = q_and_derivative(material.reflections["hkl"], lat)
    rings = []
    for lo, hi in windows:
        w, _, _ = overlap(q, lo, hi, sig_rel)
        ix = np.nonzero(w > min_overlap)[0]
        rings.append((ix, w[ix]))
    return rings


class WindowedReference:
    """ReferenceModel of an operator built with rings=windows_to_rings(material, windows, sig_rel), with the extra
    parameters "lattice" (6,: a, b, c in Angstrom, alpha, beta, gamma in degrees) and "sig_rel". normalized=True."""

    def __init__(self, ref, op, windows, sig_rel):
        if not op.normalized:
            raise NotImplementedError("WindowedReference: normalized=True only")
        self.ref, self.windows, self.sig0 = ref, [tuple(map(float, w)) for w in windows], float(sig_rel)
        mat = op.material
        self.lat0 = np.array([mat.lattice_params[k] for k in LATTICE], np.float64)
        self.hkl = np.asarray(mat.reflections["hkl"], np.float64)
        self.mult = np.asarray(mat.reflections["multiplicity"], np.float64)
        self.lam = 12.398 / float(op.cfg["energy"])
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
        """Per window: families, overlaps w and their derivatives, shares, the window's two-theta and its
        derivatives d/dw, d/dq (chain parts)."""
        q, dq = q_and_derivative(self.hkl, prm["lattice"])
        out = []
        for (lo, hi), fam in zip(self.windows, self.families):
            w, dw_dq, dw_ds = overlap(q[fam], lo, hi, prm["sig_rel"])
            tth = 2 * np.arcsin(self.lam * q[fam] / (4 * np.pi))
            dtth_dq = self.lam / (2 * np.pi * np.cos(tth / 2))
            W = w.sum()
            m = self.mult[fam]
            S = (m * w).sum()
            out.append(dict(fam=fam, w=w, dw_dq=dw_dq, dw_ds=dw_ds, dq=dq[fam], share=m * w / S, m=m, S=S,
                            tth=tth, dtth_dq=dtth_dq, tthw=float((w * tth).sum() / W), W=W))
        return out

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
        g_lat, g_sig = np.zeros(6), 0.0
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
            T_win = float((wi["share"] * T).sum())
            dL_dw = wi["m"] / wi["S"] * (T - T_win)                    # through the shares
            dL_dtthw = float((wi["share"] * dT).sum())                  # through the window's two-theta
            dtthw_dw = (wi["tth"] - wi["tthw"]) / wi["W"]
            dtthw_dtth = wi["w"] / wi["W"]
            dL_dw_tot = dL_dw + dL_dtthw * dtthw_dw
            dL_dq = dL_dw_tot * wi["dw_dq"] + dL_dtthw * dtthw_dtth * wi["dtth_dq"]
            g_lat += dL_dq @ wi["dq"]
            g_sig += float((dL_dw_tot * wi["dw_ds"]).sum())
        return {"lattice": g_lat, "sig_rel": g_sig}


class WindowedOperator:
    """GPU version of WindowedReference: an operator whose data channels are single reflection families, each
    evaluated at its window's two-theta (rings=[([f], [1])], ring_two_theta), and a mixing layer that adds the
    families of a window with their shares m_f w_f / sum(m w):

        d[a, s, c, p] = sum_{f in window p} share_f(lattice, sig_rel) u[a, s, c, f],   u = family_op(c).

    forward(c) and vjp(c, r) work on window data (N_Omega, My, N_eta * N_windows); the gradients with respect to the
    lattice parameters and sig_rel combine the shares (analytic) and the channels' two-theta (PFGradient of the family
    operator). set_lattice(lattice, sig_rel) moves the channels' two-theta (rebuilds the family operator's PF matrix).
    The remaining keyword arguments (cfg, grid, max_gb, pf_mode, ...) go to the family operator."""

    def __init__(self, material, windows, sig_rel, min_overlap=1e-3, **op_kwargs):
        from ..operators.single_phase_forward_operator import SinglePhaseForwardOperator
        self.material, self.windows, self.sig = material, [tuple(map(float, w)) for w in windows], float(sig_rel)
        self.lat = np.array([material.lattice_params[k] for k in LATTICE], np.float64)
        self.hkl = np.asarray(material.reflections["hkl"], np.float64)
        self.mult = np.asarray(material.reflections["multiplicity"], np.float64)
        self.lam = 12.398 / float(op_kwargs["cfg"]["energy"])
        rings = windows_to_rings(material, self.windows, self.sig, min_overlap)
        self.families = [np.asarray(ix) for ix, _ in rings]
        self.channel_window = np.concatenate([[p] * len(f) for p, f in enumerate(self.families)]).astype(int)
        self.channel_family = np.concatenate(self.families).astype(int)
        self.P, self.F = len(self.windows), len(self.channel_family)
        info = self._info(self.lat, self.sig)
        if op_kwargs.get("normalized", True) is not True:
            raise NotImplementedError("WindowedOperator: normalized=True only")
        op_kwargs = dict(op_kwargs, material=material, normalized=True,
                         rings=[([f], [1.0]) for f in self.channel_family], ring_two_theta=info["tth_ch"])
        self.op = SinglePhaseForwardOperator(**op_kwargs)
        self.C = int(self.op.N_eta)
        self.data_shape = (int(self.op.N_Omega), int(self.op.My), self.C * self.P)
        self.info = info
        self._pg = None

    def _info(self, lat, sig):
        """Shares, window two-theta and their derivatives at (lat, sig)."""
        q, dq = q_and_derivative(self.hkl, lat)
        share = np.zeros(self.F)
        tthw = np.zeros(self.P)
        parts = []
        for p, ((lo, hi), fam) in enumerate(zip(self.windows, self.families)):
            ch = np.nonzero(self.channel_window == p)[0]
            w, dw_dq, dw_ds = overlap(q[fam], lo, hi, sig)
            tth = 2 * np.arcsin(self.lam * q[fam] / (4 * np.pi))
            m = self.mult[fam]
            S, W = (m * w).sum(), w.sum()
            share[ch] = m * w / S
            tthw[p] = (w * tth).sum() / W
            parts.append(dict(ch=ch, w=w, dw_dq=dw_dq, dw_ds=dw_ds, dq=dq[fam], m=m, S=S, W=W, tth=tth,
                              dtth_dq=self.lam / (2 * np.pi * np.cos(tth / 2)), tthw=tthw[p]))
        return {"share": share, "tthw": tthw, "tth_ch": tthw[self.channel_window], "parts": parts}

    def set_lattice(self, lattice=None, sig_rel=None):
        from .update import set_pf_parameters
        if lattice is not None:
            self.lat = np.asarray(lattice, np.float64).copy()
        if sig_rel is not None:
            self.sig = float(sig_rel)
        self.info = self._info(self.lat, self.sig)
        set_pf_parameters(self.op, two_theta=self.info["tth_ch"])

    def mix(self, u):
        """Window data (N_Omega, My, C * P) from family data u (N_Omega, My, C * F) (NumPy)."""
        R, My = self.data_shape[:2]
        u4 = np.asarray(u, np.float64).reshape(R, My, self.C, self.F)
        d = np.zeros((R, My, self.C, self.P))
        np.add.at(d, (slice(None), slice(None), slice(None), self.channel_window), u4 * self.info["share"])
        return d.reshape(self.data_shape)

    def unmix(self, r):
        """The transpose of mix: family residual from window residual."""
        R, My = self.data_shape[:2]
        r4 = np.asarray(r, np.float64).reshape(R, My, self.C, self.P)
        return (r4[..., self.channel_window] * self.info["share"]).reshape(R, My, self.C * self.F)

    def forward(self, c):
        return self.mix(self.op.direct(c))

    def vjp(self, c, r, groups=("lattice", "sig_rel", "c")):
        """Gradients of <r, d> with respect to the lattice parameters, sig_rel and the coefficients."""
        from .gradients import PFGradient
        R, My = self.data_shape[:2]
        u4 = np.asarray(self.op.direct(c), np.float64).reshape(R, My, self.C, self.F)
        r4 = np.asarray(r, np.float64).reshape(R, My, self.C, self.P)
        r_ch = self.unmix(r).astype(np.float32)
        out = {}
        if "c" in groups:
            out["c"] = self.op.adjoint(r_ch)
        if "lattice" in groups or "sig_rel" in groups:
            if self._pg is None:
                self._pg = PFGradient(self.op)
            g_tth = self._pg(c, r_ch, groups=("two_theta",))["two_theta"]       # per family channel
            T = np.einsum("ascf,ascf->f", r4[..., self.channel_window], u4)      # <r_window, u_f>
            g_lat, g_sig = np.zeros(6), 0.0
            for pt in self.info["parts"]:
                ch = pt["ch"]
                share = self.info["share"][ch]
                dL_dw = pt["m"] / pt["S"] * (T[ch] - (share * T[ch]).sum())
                dL_dtthw = float(g_tth[ch].sum())
                dL_dw_tot = dL_dw + dL_dtthw * (pt["tth"] - pt["tthw"]) / pt["W"]
                dL_dq = dL_dw_tot * pt["dw_dq"] + dL_dtthw * pt["w"] / pt["W"] * pt["dtth_dq"]
                g_lat += dL_dq @ pt["dq"]
                g_sig += float((dL_dw_tot * pt["dw_ds"]).sum())
            out["lattice"], out["sig_rel"] = g_lat, g_sig
        return out
