"""
Self-absorption in the forward model: a float64 NumPy reference (small problems) with the derivative with respect to
the attenuation map (docs/differentiable_model.md, absorption).

A voxel x seen at projection angle a contributes to segment j (eta bin c, ring p) attenuated by

    A[a, x, j] = exp(-L_a(x) - E_aj(x)),   L_a(x) = int_0^inf mu(x - t b_a) dt,   E_aj(x) = |k_par|^-1 int_0^inf mu(x + t e_aj) dt

(b_a: the beam in the image plane; e_aj: the in-plane direction of the diffracted beam k_out = cos(2 theta) p0 +
sin(2 theta) (cos(eta) d0 + sin(eta) d90) turned into the sample frame, |k_par| its in-plane length: a sample that is
uniform along the rotation axis, as in the NWA 7034 exit-path model). So

    d[a, s, j] = sum_k PF[a, k, j] scale sum_x K(t_a(x) - s) c_k(x) A[a, x, j].

mu is given per pixel (units: 1 / pixel length), interpolated bilinearly between pixel centres, zero outside the grid;
the line integrals are sums over points 0.5 h, 1.5 h, ... (h = step pixels) along the ray, so A is exactly the
exponential of a linear map of mu, and its derivative is exact for this discretisation.

Image directions: diffractom's frame puts sample x = -(col - c), y = row - c with the rotation axis along z, so a
sample-frame direction (vx, vy) is (-vx, vy) in (col, row): the beam R(omega)^T p0 runs along -(cos omega, sin omega)
in the image, consistent with the projector's rays. (Which side eta = 0 lies on is a property of the setup; check it on
data, e.g. with left/right intensity ratios, before using absorption on measurements.)
"""
import numpy as np
from scipy.spatial.transform import Rotation


def path_matrix(shape, d, step=0.25):
    """(Npix, Npix) M with (M mu)(x) = int_0^inf mu(x + t d) dt for every pixel centre x (pixel p = col + Nx row),
    mu bilinear between pixel centres and zero outside; d = (dcol, drow), a unit vector."""
    Ny, Nx = shape
    npix = Nx * Ny
    rows, cols = np.divmod(np.arange(npix), Nx)
    n_t = int(np.ceil(np.hypot(Nx, Ny) / step)) + 2
    t = (np.arange(n_t) + 0.5) * step
    M = np.zeros((npix, npix))
    for p in range(npix):
        xs, ys = cols[p] + t * d[0], rows[p] + t * d[1]
        x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
        fx, fy = xs - x0, ys - y0
        for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy),
                            (1, 1, fx * fy)):
            xi, yi = x0 + dx, y0 + dy
            ok = (xi >= 0) & (xi < Nx) & (yi >= 0) & (yi < Ny)
            np.add.at(M[p], yi[ok] * Nx + xi[ok], step * wgt[ok])
    return M


class AbsorptionReference:
    """The ReferenceModel's data with self-absorption (module docstring); parameter "mu" (Ny, Nx) in addition to the
    ReferenceModel's (prm["mu"] = zeros reproduces it exactly). The attenuation geometry uses the bin-centre angles
    and azimuths and the initial geometry (no derivative of A with respect to the geometry)."""

    def __init__(self, ref, op, step=0.25):
        self.ref = ref
        k0 = ref.k0 / np.linalg.norm(ref.k0)
        if not np.allclose(np.abs(k0), [0, 0, 1]):
            raise NotImplementedError("AbsorptionReference assumes the rotation axis along z")
        R, C, P = ref.R, ref.C, ref.P
        shape = (ref.Ny, ref.Nx)
        ang = np.asarray(op.angles, np.float64)
        Rm = Rotation.from_rotvec(ang[:, None] * ref.k0).as_matrix()                      # (R, 3, 3)
        b = np.einsum("aij,i->aj", Rm, ref.p0)                                             # R^T p0
        b_img = np.stack([-b[:, 0], b[:, 1]], 1)
        b_img /= np.linalg.norm(b_img, axis=1, keepdims=True)
        eta = np.asarray(op.eta_angles, np.float64)
        tth = np.asarray(ref.two_theta0, np.float64)
        e = np.cos(eta)[:, None, None] * ref.d0 + np.sin(eta)[:, None, None] * ref.d90      # (C, 1, 3)
        kout = np.cos(tth)[None, :, None] * ref.p0 + np.sin(tth)[None, :, None] * e         # (C, P, 3)
        ks = np.einsum("aij,cpi->acpj", Rm, kout)                                          # (R, C, P, 3)
        kpar = np.hypot(ks[..., 0], ks[..., 1])
        e_img = np.stack([-ks[..., 0], ks[..., 1]], -1) / kpar[..., None]
        self.Lin = np.stack([path_matrix(shape, -b_img[a], step) for a in range(R)])       # upstream of the voxel
        self.E = np.zeros((R, C * P) + self.Lin.shape[1:])
        for a in range(R):
            for c in range(C):
                for p in range(P):
                    self.E[a, c * P + p] = path_matrix(shape, e_img[a, c, p], step) / kpar[a, c, p]

    def attenuation(self, mu):
        """A (R, Npix, C P)."""
        m = np.asarray(mu, np.float64).ravel()
        tot = (self.Lin @ m)[:, :, None] + np.einsum("ajxy,y->axj", self.E, m)
        return np.exp(-tot)

    def forward(self, c, prm):
        ref = self.ref
        R, K, CP = ref.R, ref.K, ref.C * ref.P
        PF = ref.pf(prm).reshape(R, K, CP)
        Pm, _ = ref.projection_matrices(prm)
        A = self.attenuation(prm["mu"])
        cf = np.asarray(c, np.float64).reshape(K, -1)
        S = np.einsum("asx,kx,axj->askj", Pm, cf, A)                                      # attenuated projections
        return np.einsum("akj,askj->asj", PF, S)

    def vjp(self, c, prm, r, groups=("c", "mu", "rot", "sigma", "two_theta", "eta_offset", "axis_tilt")):
        ref = self.ref
        R, K, C, P, My = ref.R, ref.K, ref.C, ref.P, ref.My
        CP = C * P
        r = np.asarray(r, np.float64).reshape(R, My, CP)
        cf = np.asarray(c, np.float64).reshape(K, -1)
        PF = ref.pf(prm).reshape(R, K, CP)
        Pm, _ = ref.projection_matrices(prm)
        A = self.attenuation(prm["mu"])
        out = {}
        Z = np.einsum("akj,asj->askj", PF, r)                                              # PF^T r per segment
        BZ = np.einsum("asx,askj->akxj", Pm, Z)                                            # backprojected
        if "c" in groups:
            out["c"] = np.einsum("akxj,axj->kx", BZ, A).reshape(np.shape(c))
        if "mu" in groups:
            W = np.einsum("kx,akxj->axj", cf, BZ) * A                                     # dL / d(-tot)
            g = -np.einsum("axy,ax->y", self.Lin, W.sum(axis=2)) - np.einsum("ajxy,axj->y", self.E, W)
            out["mu"] = g.reshape(np.shape(prm["mu"]))
        pf_groups = [g_ for g_ in groups if g_ in ("rot", "sigma", "two_theta", "eta_offset", "axis_tilt")]
        if pf_groups:
            S = np.einsum("asx,kx,axj->askj", Pm, cf, A)
            G = np.einsum("asj,askj->akj", r, S).reshape(R, K, C, P)
            out.update(ref._pf_grads(prm, G, pf_groups))
        return out
