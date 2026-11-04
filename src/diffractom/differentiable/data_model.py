"""
The intensity layer on top of the operator's prediction u = A(c) (docs/differentiable_model.md, section 2):

    d[a, s, j] = f_a * t_s * r_p * u[a, s, j] + b_j          (j = c P + p)

with a scale f_a of every projection (beam current, sample-wide absorption), t_s of every translation (beam position,
flux drift), r_p of every ring (structure factors, Lorentz-polarisation, integration) and an additive background b_j of
every segment. NumPy or pyopencl arrays (the same code runs on both: element-wise products and sums over axes).
"""
import numpy as np
import pyopencl.array as clarray

GROUPS = ("angle_scale", "trans_scale", "ring_scale", "background")


def _np(a):
    return a.get() if isinstance(a, clarray.Array) else np.asarray(a)


class IntensityLayer:
    """d = f_a t_s r_p u + b_j for data shaped (N_Omega, My, N_eta * N_rings)."""

    def __init__(self, data_shape, n_rings):
        self.R, self.My, self.N_seg = (int(v) for v in data_shape)
        self.P = int(n_rings)
        self.C = self.N_seg // self.P
        assert self.C * self.P == self.N_seg

    def params0(self):
        return {"angle_scale": np.ones(self.R), "trans_scale": np.ones(self.My), "ring_scale": np.ones(self.P),
                "background": np.zeros(self.N_seg)}

    def _factor(self, prm):
        """f_a t_s r_p as a (R, My, C, P) broadcastable array (float64 NumPy)."""
        f = np.asarray(prm["angle_scale"], np.float64)[:, None, None, None]
        t = np.asarray(prm["trans_scale"], np.float64)[None, :, None, None]
        r = np.asarray(prm["ring_scale"], np.float64)[None, None, None, :]
        return f * t * r

    def forward(self, u, prm):
        u4 = _np(u).astype(np.float64).reshape(self.R, self.My, self.C, self.P)
        d = self._factor(prm) * u4 + np.asarray(prm["background"], np.float64).reshape(1, 1, self.C, self.P)
        return d.reshape(self.R, self.My, self.N_seg)

    def vjp(self, u, prm, r, groups=GROUPS):
        """Gradients of <r, d> with respect to the layer's parameters, and "u" = dL/du (= r f t r_p) for the
        operator's gradients (PFGradient, ProjGradient, the adjoint)."""
        u4 = _np(u).astype(np.float64).reshape(self.R, self.My, self.C, self.P)
        r4 = _np(r).astype(np.float64).reshape(self.R, self.My, self.C, self.P)
        f = np.asarray(prm["angle_scale"], np.float64)
        t = np.asarray(prm["trans_scale"], np.float64)
        rp = np.asarray(prm["ring_scale"], np.float64)
        ru = r4 * u4
        out = {}
        if "angle_scale" in groups:
            out["angle_scale"] = np.einsum("ascp,s,p->a", ru, t, rp)
        if "trans_scale" in groups:
            out["trans_scale"] = np.einsum("ascp,a,p->s", ru, f, rp)
        if "ring_scale" in groups:
            out["ring_scale"] = np.einsum("ascp,a,s->p", ru, f, t)
        if "background" in groups:
            out["background"] = r4.sum(axis=(0, 1)).reshape(self.N_seg)
        out["u"] = (r4 * self._factor(prm)).reshape(self.R, self.My, self.N_seg)
        return out

    def fit_linear(self, u, b, w, prm, groups=GROUPS, sweeps=3):
        """Weighted least squares of the layer's parameters for a fixed prediction u: block coordinate descent, each
        block in closed form (sum_w g (b - bg) / sum_w g^2 per index, g the prediction without that factor). Returns
        new parameters (the gauge: the ring scales absorb the overall level, the mean angle and translation scales are
        kept at 1)."""
        u4 = _np(u).astype(np.float64).reshape(self.R, self.My, self.C, self.P)
        b4 = _np(b).astype(np.float64).reshape(self.R, self.My, self.C, self.P)
        w4 = np.broadcast_to(_np(w).astype(np.float64).reshape(-1, self.C, self.P) if np.size(w) != self.N_seg else
                             _np(w).astype(np.float64).reshape(1, 1, self.C, self.P),
                             (self.R, self.My, self.C, self.P)) if np.size(w) != b4.size else \
            _np(w).astype(np.float64).reshape(b4.shape)
        p = {k: np.array(v, np.float64, copy=True) for k, v in prm.items()}
        for _ in range(sweeps):
            for g_ in groups:
                F = self._factor(p)
                bg = p["background"].reshape(1, 1, self.C, self.P)
                if g_ == "background":
                    p["background"] = ((w4 * (b4 - F * u4)).sum((0, 1)) /
                                       np.maximum(w4.sum((0, 1)), 1e-300)).reshape(self.N_seg)
                    continue
                key, ax = {"angle_scale": ("angle_scale", (1, 2, 3)), "trans_scale": ("trans_scale", (0, 2, 3)),
                           "ring_scale": ("ring_scale", (0, 1, 2))}[g_]
                cur = p[key]
                shape = [1, 1, 1, 1]
                shape[{"angle_scale": 0, "trans_scale": 1, "ring_scale": 3}[g_]] = -1
                g = F * u4 / cur.reshape(shape)
                num = (w4 * g * (b4 - bg)).sum(ax)
                den = (w4 * g * g).sum(ax)
                p[key] = np.where(den > 0, num / np.maximum(den, 1e-300), cur)
            for key in ("angle_scale", "trans_scale"):          # gauge: mean 1, the level goes to the rings
                if key in groups:
                    m = p[key].mean()
                    p[key] /= m
                    p["ring_scale"] *= m
        return p


class BeamBlur:
    """The pencil beam's profile across the translation direction as a Gaussian blur of the prediction along the
    detector (translation) axis, beyond the bin footprint of the projector:

        d[a, s, j] = sum_D B(D) u[a, s - D, j],   B(D) = exp(-D^2 / (2 width^2)) / sum_D' exp(-D'^2 / (2 width^2))

    (D = -H .. H, H = ceil(4 width) + 1; bins outside the detector count as zero). It commutes with the PF product, so
    it acts on the whole prediction u = A(c). width in detector bins (> 0)."""

    def __init__(self, data_shape):
        self.R, self.My, self.N_seg = (int(v) for v in data_shape)

    def kernel(self, width):
        w = float(width)
        H = int(np.ceil(4 * w)) + 1
        D = np.arange(-H, H + 1, dtype=np.float64)
        e = np.exp(-D * D / (2 * w * w))
        B = e / e.sum()
        dB = B * (D * D / w ** 3 - (B * D * D).sum() / w ** 3)           # d B / d width
        return D.astype(int), B, dB

    def _conv(self, u, D, k):
        out = np.zeros_like(u)
        for d_, kk in zip(D, k):
            if d_ >= 0:
                out[:, d_:] += kk * u[:, :self.My - d_]
            else:
                out[:, :d_] += kk * u[:, -d_:]
        return out

    def forward(self, u, width):
        u = _np(u).astype(np.float64).reshape(self.R, self.My, self.N_seg)
        D, B, _ = self.kernel(width)
        return self._conv(u, D, B)

    def vjp(self, u, width, r):
        """{"beam_width": d<r, d>/d width, "u": B^T r (for the operator's gradients)}."""
        u = _np(u).astype(np.float64).reshape(self.R, self.My, self.N_seg)
        r = _np(r).astype(np.float64).reshape(self.R, self.My, self.N_seg)
        D, B, dB = self.kernel(width)
        return {"beam_width": float((r * self._conv(u, D, dB)).sum()), "u": self._conv(r, -D, B)}
