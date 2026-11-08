"""
Float64 NumPy reference of SinglePhaseForwardOperator's model (native projector) as a function of its continuous
parameters, with the analytic derivatives of the data (vector-Jacobian products). For small problems: the
specification the GPU implementation is tested against, and the place to try new parameters.

Model (docs/differentiable_model.md): data d[a, s, j], projection angle a, detector bin s, segment j = c P + p
(eta bin c, ring p),

    d[a, s, j]    = sum_k PF[a, k, j] S_k[a, s],          S_k[a, s] = scale sum_x K(t_a(x) - s) c_k(x)
    PF[a, k, j]   = I_p N(sigma_k) sum_nodes w_n sum_{axes h of ring p} m_h g(n_kh . v_n; sigma_k)
    g(d; sigma)   = exp(-(1 - |d|) / sigma^2) + exp(-(1 + |d|) / sigma^2)     (each term cut at PF_CUT)
    N(sigma)      = 1 / (8 pi sigma^2)
    n_kh          = U_k h                                  (pole of axis h for orientation k, sample frame)
    v_n           = R(omega_n)^T u(eta_n, theta_p),        u = cos(theta) (cos(eta) d0 + sin(eta) d90) - sin(theta) p0
    t_a(x)        = X_a x + Y_a y + T0_a - cor / delta_s + shift_a       (detector coordinate of pixel x, in bins)

With op.reflection_two_theta every pole axis h has its own theta_h: v_n = cos(theta_h) U_n - sin(theta_h) W_n, where
U_n = R(omega_n)^T (cos(eta_n) d0 + sin(eta_n) d90) and W_n = R(omega_n)^T p0; prm["two_theta"] (P,) is then the channels'
mean Bragg angle and every axis of channel p moves by the change of its mean.

Parameters (ReferenceModel.params0): c (K, Ny, Nx) coefficients; rot (K, 3) rotation of every orientation, a left
increment U_k <- exp([rot_k]x) U_k (the derivative is taken at rot = 0); sigma (K,) kernel widths; two_theta (P,)
Bragg angles 2 theta_p; eta_offset (detector azimuth zero, eta <- eta + eta_offset); axis_tilt (2,) rotation of the
rotation axis about two directions e1, e2 perpendicular to it (k0 <- exp([t1 e1 + t2 e2]x) k0); cor (centre of
rotation, the operator's cor_offset); shift (R,) detector shift of every projection (bins: wobble / run-out).
"""
import numpy as np
from scipy.spatial.transform import Rotation

GROUPS = ("c", "rot", "sigma", "two_theta", "eta_offset", "axis_tilt", "cor", "shift")


def _perp_basis(k):
    k = k / np.linalg.norm(k)
    a = np.array([1.0, 0.0, 0.0]) if abs(k[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(k, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(k, e1)


class ReferenceModel:
    """The model of a SinglePhaseForwardOperator (projector 'native'), in float64, for small problems.

    cut: the cut of the PF kernel terms ((1 -+ |d|) / sigma^2 < cut); "operator" (default) as the operator's
    kernels, None for no cut (smooth everywhere, for finite-difference checks).
    """

    def __init__(self, op, cut="operator"):
        if op.projector != "native":
            raise ValueError("the reference model has the native projector only")
        cfg = op.cfg
        self.R, self.My, self.C, self.P = int(op.N_Omega), int(op.My), int(op.N_eta), int(op.N_peaks)
        self.Nx, self.Ny, self.K = int(op.Nx), int(op.Ny), int(op.K)
        self.S, self.Se = int(op.N_Omega_subdivisions), int(op.N_eta_subdivisions)
        self.om_nodes = np.asarray(op.angles_subdivided, np.float64).reshape(self.R, self.S)
        self.eta_nodes = np.asarray(op.eta_angles_subdivided, np.float64).reshape(self.C, self.Se)
        if op.quad_weights_cpu is None:
            self.qw = np.full((self.S, self.Se), 1.0 / (self.S * self.Se))
        else:
            self.qw = np.asarray(op.quad_weights_cpu, np.float64).reshape(self.S, self.Se)
        vec = lambda k: np.asarray(cfg[k], np.float64)
        self.k0, self.p0 = vec("k_direction_0"), vec("p_direction_0")
        self.d0, self.d90 = vec("detector_direction_origin"), vec("detector_direction_positive_90")
        self.e_tilt = _perp_basis(self.k0)
        self.U0 = np.asarray(op.grid_rot_cpu, np.float64)
        self.sigma0 = np.asarray(op.sigma_cpu, np.float64)
        self.axes = np.asarray(op.pole_axes_cpu, np.float64)
        self.counts = np.asarray(op.axis_count_cpu, np.float64)
        self.start = np.asarray(op.axis_start_cpu)
        self.intens = np.asarray(op.intens_cpu, np.float64)
        self.two_theta0 = np.asarray(op.two_theta_peaks, np.float64)
        self.axis_mode = bool(getattr(op, "reflection_two_theta", False))
        if self.axis_mode:
            self.axis_tth0 = np.asarray(op.axis_two_theta_cpu, np.float64)
            self.axis_ring = np.repeat(np.arange(self.P), np.diff(self.start))
        if cut == "operator":
            cut = 6.0 if op.pf_cutoff_sigma is None else float(op.pf_cutoff_sigma) ** 2 / 2
        self.cut = cut
        # projector (ParallelRadon of the operator)
        ang = np.asarray(op.angles, np.float64)
        n_max = max(self.Nx, self.Ny)
        self.delta_x = self.Nx / n_max
        self.delta_s = 1.0                          # detector_width = My = number of bins
        rho = self.delta_s / self.delta_x
        self.X = np.cos(ang - 0.5 * np.pi) / rho
        self.Y = np.sin(ang - 0.5 * np.pi) / rho
        self.T0 = (self.My - 1) / 2.0 - self.X * (self.Nx - 1) / 2.0 - self.Y * (self.Ny - 1) / 2.0
        self.scale = self.delta_x ** 2 / self.delta_s
        self.kernel = op.radon_kernel
        self.cor0 = float(op.cor_offset)
        yy, xx = np.meshgrid(np.arange(self.Ny), np.arange(self.Nx), indexing="ij")
        self.px, self.py = xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)   # pixel p = x + Nx y

    # ------------------------------------------------------------------ parameters
    def params0(self):
        return {"rot": np.zeros((self.K, 3)), "sigma": self.sigma0.copy(), "two_theta": self.two_theta0.copy(),
                "eta_offset": 0.0, "axis_tilt": np.zeros(2), "cor": self.cor0, "shift": np.zeros(self.R)}

    # ------------------------------------------------------------------ geometry
    def _axis(self, prm):
        t = prm["axis_tilt"]
        q = Rotation.from_rotvec(t[0] * self.e_tilt[0] + t[1] * self.e_tilt[1]).as_matrix()
        return q @ self.k0

    def _u(self, prm):
        """u (C, Se, P, 3) and its derivatives du/d eta_offset, du/d two_theta_p (per ring)."""
        eta = self.eta_nodes + prm["eta_offset"]
        e = np.cos(eta)[..., None] * self.d0 + np.sin(eta)[..., None] * self.d90           # (C, Se, 3)
        de = -np.sin(eta)[..., None] * self.d0 + np.cos(eta)[..., None] * self.d90
        th = 0.5 * np.asarray(prm["two_theta"], np.float64)
        ct, st = np.cos(th)[None, None, :, None], np.sin(th)[None, None, :, None]
        u = ct * e[:, :, None, :] - st * self.p0
        du_eta = ct * de[:, :, None, :]
        du_tth = 0.5 * (-st * e[:, :, None, :] - ct * self.p0)                          # d u_p / d (2 theta_p)
        return u, du_eta, du_tth

    def _rot_om(self, prm):
        k0 = self._axis(prm)
        return Rotation.from_rotvec(self.om_nodes.reshape(-1, 1) * k0).as_matrix().reshape(self.R, self.S, 3, 3)

    def probed(self, prm):
        """v (R, S, C, Se, P, 3) = R(omega)^T u (the operator's coords)."""
        if self.axis_mode:
            raise ValueError("reflection_two_theta: every pole axis has its own probed direction (use probed_parts)")
        u, _, _ = self._u(prm)
        return np.einsum("rsij,cepi->rscepj", self._rot_om(prm), u)

    def probed_derivatives(self, prm, groups=("eta_offset", "axis_tilt", "two_theta")):
        """Derivatives of the probed directions, each shaped like probed(): "eta_offset", ("axis_tilt", 0),
        ("axis_tilt", 1) and "two_theta" (ring p's slice holds d v / d (2 theta_p))."""
        if self.axis_mode:
            raise ValueError("reflection_two_theta: use probed_parts")
        Rot = self._rot_om(prm)
        u, du_eta, du_tth = self._u(prm)
        out = {}
        if "eta_offset" in groups:
            out["eta_offset"] = np.einsum("rsij,cepi->rscepj", Rot, du_eta)
        if "two_theta" in groups:
            out["two_theta"] = np.einsum("rsij,cepi->rscepj", Rot, du_tth)
        if "axis_tilt" in groups:
            v = np.einsum("rsij,cepi->rscepj", Rot, u)
            for i, e in enumerate(self.e_tilt):
                eu = np.cross(e[None, None, None, :], u)
                out[("axis_tilt", i)] = np.cross(e, v) - np.einsum("rsij,cepi->rscepj", Rot, eu)
        return out

    def _axis_theta(self, prm):
        """theta (A,) of every pole axis (reflection_two_theta)."""
        return 0.5 * (self.axis_tth0 + (np.asarray(prm["two_theta"], np.float64) - self.two_theta0)[self.axis_ring])

    def probed_parts(self, prm, groups=()):
        """reflection_two_theta: "u" (R, S, C, Se, 3) and "w" (R, S, 3) with v = cos(theta) u - sin(theta) w, and their
        derivatives: "eta_offset" (du), ("axis_tilt", i) (du), ("axis_tilt_w", i) (dw)."""
        Rot = self._rot_om(prm)
        eta = self.eta_nodes + prm["eta_offset"]
        e = np.cos(eta)[..., None] * self.d0 + np.sin(eta)[..., None] * self.d90
        u = np.einsum("rsij,cei->rscej", Rot, e)
        w = np.einsum("rsij,i->rsj", Rot, self.p0)
        out = {"u": u, "w": w}
        if "eta_offset" in groups:
            de = -np.sin(eta)[..., None] * self.d0 + np.cos(eta)[..., None] * self.d90
            out["eta_offset"] = np.einsum("rsij,cei->rscej", Rot, de)
        if "axis_tilt" in groups:
            for i, a in enumerate(self.e_tilt):
                out[("axis_tilt", i)] = np.cross(a, u) - np.einsum("rsij,cei->rscej", Rot, np.cross(a[None, None, :], e))
                out[("axis_tilt_w", i)] = np.cross(a, w) - np.einsum("rsij,i->rsj", Rot, np.cross(a, self.p0))
        return out

    def _axis_dots(self, n, u, w, ct, st):
        """n.v (R, S, C, Se, K, A_p) for poles n (K, A_p, 3)."""
        du = np.einsum("rscex,kax->rsceka", u, n)
        dw = np.einsum("rsx,kax->rska", w, n)[:, :, None, None]
        return ct * du - st * dw

    def poles(self, prm):
        rot = Rotation.from_rotvec(prm["rot"]).as_matrix() @ self.U0                   # (K, 3, 3)
        return np.einsum("kij,aj->kai", rot, self.axes)

    # ------------------------------------------------------------------ pole figures
    def _g(self, d, s):
        """g, dg/d(n.v) and dg/ds for signed dot products d and s = 1 / sigma^2 (broadcast)."""
        ad = np.abs(d)
        x1, x2 = (1.0 - ad) * s, (1.0 + ad) * s
        e1, e2 = np.exp(-x1), np.exp(-x2)
        if self.cut is not None:
            e1, e2 = np.where(x1 < self.cut, e1, 0.0), np.where(x2 < self.cut, e2, 0.0)
        return e1 + e2, np.sign(d) * s * (e1 - e2), -(1.0 - ad) * e1 - (1.0 + ad) * e2

    def pf(self, prm, derivs=False):
        """PF (R, K, C, P); with derivs also the per-node factors needed for the derivatives (see _pf_grads)."""
        axm = self.axis_mode
        if axm:
            th = self._axis_theta(prm)
            parts = self.probed_parts(prm)
        else:
            v = self.probed(prm)
        n = self.poles(prm)
        sig = np.asarray(prm["sigma"], np.float64)
        s = 1.0 / sig ** 2
        norm = 1.0 / (8.0 * np.pi * sig ** 2)
        out = np.zeros((self.R, self.K, self.C, self.P))
        for p in range(self.P):
            a0, a1 = self.start[p], self.start[p + 1]
            if axm:
                d = self._axis_dots(n[:, a0:a1], parts["u"], parts["w"], np.cos(th[a0:a1]), np.sin(th[a0:a1]))
            else:
                d = np.einsum("rscex,kax->rsceka", v[..., p, :], n[:, a0:a1])
            g, _, _ = self._g(d, s[:, None])
            node = np.einsum("rsceka,a->rscek", g, self.counts[a0:a1])
            out[..., p] = self.intens[p] * norm[None, :, None] * np.einsum("rscek,se->rkc", node, self.qw)
        return out

    # ------------------------------------------------------------------ projector
    def _K(self, d, A):
        if self.kernel == "hat":
            k = np.maximum(1.0 - np.abs(d), 0.0)
            dk = np.where(np.abs(d) < 1.0, -np.sign(d), 0.0)
            return k, dk
        hw, top = 0.5 * (1.0 + A), min(1.0, A)
        q = hw - np.abs(d)
        k = np.clip(q, 0.0, top) / A
        dk = np.where((q > 0.0) & (q < top), -np.sign(d) / A, 0.0)
        return k, dk

    def projection_matrices(self, prm):
        """P (R, My, Npix) and dP/dt (R, My, Npix): S_k[a] = P[a] @ c_k, and the derivative for a detector shift."""
        npix = self.Nx * self.Ny
        Pm = np.zeros((self.R, self.My, npix))
        dPm = np.zeros((self.R, self.My, npix))
        shift = np.asarray(prm["shift"], np.float64)
        for a in range(self.R):
            A = max(abs(self.X[a]), abs(self.Y[a]))
            t = self.X[a] * self.px + self.Y[a] * self.py + self.T0[a] - prm["cor"] / self.delta_s + shift[a]
            for s in range(self.My):
                k, dk = self._K(t - s, A)
                Pm[a, s] = self.scale * k
                dPm[a, s] = self.scale * dk
        return Pm, dPm

    # ------------------------------------------------------------------ forward and derivatives
    def forward(self, c, prm):
        """d (R, My, C * P) for coefficients c (K, Ny, Nx)."""
        PF = self.pf(prm).reshape(self.R, self.K, self.C * self.P)
        Pm, _ = self.projection_matrices(prm)
        S = np.einsum("asp,kp->ask", Pm, np.asarray(c, np.float64).reshape(self.K, -1))
        return np.einsum("akj,ask->asj", PF, S)

    def vjp(self, c, prm, r, groups=GROUPS):
        """Gradients of <r, d(c, prm)> with respect to the parameter groups (dict of arrays shaped as prm, plus
        'c' shaped as c): the vector-Jacobian product of the model at r (R, My, C * P)."""
        R, K, C, P, My = self.R, self.K, self.C, self.P, self.My
        r = np.asarray(r, np.float64).reshape(R, My, C * P)
        cf = np.asarray(c, np.float64).reshape(K, -1)
        Pm, dPm = self.projection_matrices(prm)
        S = np.einsum("asp,kp->ask", Pm, cf)
        PF = self.pf(prm)
        out = {}
        Z = np.einsum("akj,asj->ask", PF.reshape(R, K, C * P), r)          # PF^T r, before the backprojection
        if "c" in groups:
            out["c"] = np.einsum("asp,ask->kp", Pm, Z).reshape(np.shape(c))
        if "shift" in groups or "cor" in groups:
            gs = np.einsum("asp,kp,ask->a", dPm, cf, Z)                       # d/d shift_a
            if "shift" in groups:
                out["shift"] = gs
            if "cor" in groups:
                out["cor"] = -gs.sum() / self.delta_s
        pf_groups = [g for g in groups if g in ("rot", "sigma", "two_theta", "eta_offset", "axis_tilt")]
        if pf_groups:
            G = np.einsum("asj,ask->akj", r, S).reshape(R, K, C, P)            # dL/dPF
            out.update(self._pf_grads(prm, G, pf_groups))
        return out

    def _pf_grads(self, prm, G, groups):
        """sum over (a, k, c, p) of G * dPF/dparameter for the PF parameter groups."""
        if self.axis_mode:
            return self._pf_grads_axis(prm, G, groups)
        Rot = self._rot_om(prm)
        u, du_eta, du_tth = self._u(prm)
        v = np.einsum("rsij,cepi->rscepj", Rot, u)
        n = self.poles(prm)
        sig = np.asarray(prm["sigma"], np.float64)
        s = 1.0 / sig ** 2
        norm = 1.0 / (8.0 * np.pi * sig ** 2)
        out = {"rot": np.zeros((self.K, 3)), "sigma": np.zeros(self.K), "two_theta": np.zeros(self.P),
               "eta_offset": 0.0, "axis_tilt": np.zeros(2)}
        dv = self.probed_derivatives(prm, groups)
        for p in range(self.P):
            a0, a1 = self.start[p], self.start[p + 1]
            vp, np_ = v[..., p, :], n[:, a0:a1]
            d = np.einsum("rscex,kax->rsceka", vp, np_)
            g, dgd, dgs = self._g(d, s[:, None])
            wc = self.counts[a0:a1]
            # weight of every node term in dL/d(.): G * I_p * N(sigma_k) * w_node * m_h
            W = (G[..., p].transpose(0, 2, 1)[:, None, :, None, :] * self.qw[None, :, None, :, None]   # (R,S,C,Se,K)
                 * (self.intens[p] * norm)[None, None, None, None, :])
            Wd = W[..., None] * wc * dgd                                                  # (R, S, C, Se, K, A)
            if "rot" in groups:
                cr = np.cross(np_[None, None, None, None], vp[:, :, :, :, None, None, :])  # n x v
                out["rot"] += np.einsum("rsceka,rscekax->kx", Wd, cr)
            if "sigma" in groups:
                node = np.einsum("rsceka,a->rscek", g, wc)
                dnode = np.einsum("rsceka,a->rscek", dgs, wc)
                out["sigma"] += np.einsum("rscek,rscek->k", W, -2.0 / sig * node - 2.0 / sig ** 3 * dnode)
            for key, dvk in dv.items():
                dd = np.einsum("rscex,kax->rsceka", dvk[..., p, :], np_)
                val = np.einsum("rsceka,rsceka->", Wd, dd)
                if key == "two_theta":
                    out["two_theta"][p] += val
                elif key == "eta_offset":
                    out["eta_offset"] += val
                else:
                    out["axis_tilt"][key[1]] += val
        return {k: out[k] for k in groups}


    def _pf_grads_axis(self, prm, G, groups):
        """_pf_grads with one Bragg angle per pole axis."""
        th = self._axis_theta(prm)
        parts = self.probed_parts(prm, groups)
        u, w = parts["u"], parts["w"]
        n = self.poles(prm)
        sig = np.asarray(prm["sigma"], np.float64)
        s = 1.0 / sig ** 2
        norm = 1.0 / (8.0 * np.pi * sig ** 2)
        out = {"rot": np.zeros((self.K, 3)), "sigma": np.zeros(self.K), "two_theta": np.zeros(self.P),
               "eta_offset": 0.0, "axis_tilt": np.zeros(2)}
        for p in range(self.P):
            a0, a1 = self.start[p], self.start[p + 1]
            np_, ct, st = n[:, a0:a1], np.cos(th[a0:a1]), np.sin(th[a0:a1])
            d = self._axis_dots(np_, u, w, ct, st)
            g, dgd, dgs = self._g(d, s[None, None, None, None, :, None])
            wc = self.counts[a0:a1]
            W = (G[..., p].transpose(0, 2, 1)[:, None, :, None, :] * self.qw[None, :, None, :, None]
                 * (self.intens[p] * norm)[None, None, None, None, :])
            Wd = W[..., None] * wc * dgd
            dotu = np.einsum("rscex,kax->rsceka", u, np_)
            dotw = np.einsum("rsx,kax->rska", w, np_)[:, :, None, None]
            if "rot" in groups:
                vv = ct[..., None] * u[:, :, :, :, None, None, :] - st[..., None] * w[:, :, None, None, None, None, :]
                cr = np.cross(np_[None, None, None, None], vv)                   # n x v, (R, S, C, Se, K, A, 3)
                out["rot"] += np.einsum("rsceka,rscekax->kx", Wd, cr)
            if "sigma" in groups:
                node = np.einsum("rsceka,a->rscek", g, wc)
                dnode = np.einsum("rsceka,a->rscek", dgs, wc)
                out["sigma"] += np.einsum("rscek,rscek->k", W, -2.0 / sig * node - 2.0 / sig ** 3 * dnode)
            if "two_theta" in groups:
                out["two_theta"][p] += np.sum(Wd * 0.5 * (-st * dotu - ct * dotw))
            if "eta_offset" in groups:
                out["eta_offset"] += np.sum(Wd * ct * np.einsum("rscex,kax->rsceka", parts["eta_offset"], np_))
            if "axis_tilt" in groups:
                for i in range(2):
                    dU = np.einsum("rscex,kax->rsceka", parts[("axis_tilt", i)], np_)
                    dW = np.einsum("rsx,kax->rska", parts[("axis_tilt_w", i)], np_)[:, :, None, None]
                    out["axis_tilt"][i] += np.sum(Wd * (ct * dU - st * dW))
        return {k: out[k] for k in groups}


def perturb(prm, group, index, h):
    """A copy of prm with prm[group][index] (or the scalar prm[group]) moved by h."""
    out = {k: (np.array(v, dtype=np.float64, copy=True) if np.ndim(v) else float(v)) for k, v in prm.items()}
    if np.ndim(out[group]) == 0:
        out[group] = out[group] + h
    else:
        out[group][index] += h
    return out
