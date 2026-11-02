"""
Self-absorption in the forward model: a float64 NumPy reference (small problems) with the derivative with respect to
the attenuation map (docs/differentiable_model.md, absorption).

A voxel x seen at projection angle a contributes to segment j (eta bin c, ring p) attenuated by

    A[a, x, j] = exp(-L_a(x) - E_aj(x)),   L_a(x) = int_0^inf mu(x - t b_a) dt,   E_aj(x) = |k_par|^-1 int_0^inf mu(x + t e_aj) dt

(b_a: the beam in the image plane; e_aj: the in-plane direction of the diffracted beam k_out = cos(2 theta) p0 +
sin(2 theta) (cos(eta) d0 + sin(eta) d90) turned into the sample frame, |k_par| its in-plane length: a sample that is
uniform along the rotation axis, as in the NWA 7034 exit-path model). So

    d[a, s, j] = sum_k PF[a, k, j] scale sum_x K(t_a(x) - s) c_k(x) A[a, x, j].

mu is given per pixel (units: 1 / pixel length), bilinear between pixel centres and zero outside the grid. A line
integral along a direction d is computed on a grid of M x M points turned to d (M = ceil(sqrt(2) max(Nx, Ny)) + 3,
unit spacing, centred on the image): mu sampled there (bilinear), a cumulative sum along d (D(u) = sum_{u' >= u} -
f(u) / 2), sampled back at the pixel centres (bilinear) - the discretisation of the GPU version (AbsorptionOperator),
so A is exactly the exponential of a linear map of mu and its derivative is exact. Segments may be grouped into
classes of exit direction (exit_classes): the members of a class share one direction and length factor.

Image directions: diffractom's frame puts sample x = -(col - c), y = row - c with the rotation axis along z, so a
sample-frame direction (vx, vy) is (-vx, vy) in (col, row): the beam R(omega)^T p0 runs along -(cos omega, sin omega)
in the image, consistent with the projector's rays. (Which side eta = 0 lies on is a property of the setup; check it on
data, e.g. with left/right intensity ratios, before using absorption on measurements.)
"""
import numpy as np


def frame_size(shape):
    return int(np.ceil(np.sqrt(2.0) * max(shape))) + 3


def _bilinear_matrix(xs, ys, nx, ny):
    """Sparse (n, nx * ny) matrix of bilinear interpolation at points (xs, ys) of a grid (index x + nx y), zero
    outside."""
    from scipy.sparse import coo_matrix
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = xs - x0, ys - y0
    rows, cols, vals = [], [], []
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        rows.append(np.nonzero(ok)[0])
        cols.append(yi[ok] * nx + xi[ok])
        vals.append(w[ok])
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(xs), nx * ny)).tocsr()


def frame_matrix(shape, d):
    """(Npix, Npix) dense matrix of the line integral int_0^inf mu(x + t d) dt (frame discretisation, module
    docstring) for a unit direction d = (dcol, drow); pixel p = col + Nx row."""
    from scipy.sparse import identity, kron, csr_matrix
    Ny, Nx = shape
    M = frame_size(shape)
    cm, cx, cy = (M - 1) / 2.0, (Nx - 1) / 2.0, (Ny - 1) / 2.0
    dx, dy = float(d[0]), float(d[1])
    px, py = -dy, dx
    iw, iu = np.mgrid[0:M, 0:M].astype(np.float64)
    xs = cx + (iu - cm) * dx + (iw - cm) * px
    ys = cy + (iu - cm) * dy + (iw - cm) * py
    S1 = _bilinear_matrix(xs.ravel(), ys.ravel(), Nx, Ny)                    # (M^2, Npix): mu on the frame
    tri = csr_matrix(np.triu(np.ones((M, M))) - 0.5 * np.eye(M))             # D(u) = sum_{u' >= u} - f(u) / 2
    C = kron(identity(M, format="csr"), tri, format="csr")                    # per frame row w
    rows, cols = np.divmod(np.arange(Nx * Ny), Nx)
    u = (cols - cx) * dx + (rows - cy) * dy + cm
    w = (cols - cx) * px + (rows - cy) * py + cm
    S2 = _bilinear_matrix(u, w, M, M)                                         # frame (x = u, y = w) at the pixels
    return np.asarray((S2 @ C @ S1).todense())


def exit_directions(op):
    """Per segment j = c P + p: the in-plane part (x, y in the lab frame, rotation axis z) of the diffracted beam's
    direction, unit (CP, 2), and 1 / its length (CP,)."""
    vec = lambda k: np.asarray(op.cfg[k], np.float64)
    p0, d0, d90 = vec("p_direction_0"), vec("detector_direction_origin"), vec("detector_direction_positive_90")
    eta = np.asarray(op.eta_angles, np.float64)
    tth = np.asarray(op.two_theta_peaks, np.float64)
    e = np.cos(eta)[:, None, None] * d0 + np.sin(eta)[:, None, None] * d90
    k = (np.cos(tth)[None, :, None] * p0 + np.sin(tth)[None, :, None] * e).reshape(-1, 3)
    kpar = np.hypot(k[:, 0], k[:, 1])
    return k[:, :2] / kpar[:, None], 1.0 / kpar


def exit_classes(op, width_deg=2.0):
    """Segments grouped by the in-plane angle of their exit direction (bins of width_deg): (assign (CP,), unit lab
    directions (n, 2), length factors (n,)), each class's direction and factor the mean of its members'."""
    dirs, fac = exit_directions(op)
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])
    phi0 = np.arctan2(*np.asarray(op.cfg["p_direction_0"], np.float64)[[1, 0]])
    rel = np.angle(np.exp(1j * (phi - phi0)))                                # relative to the beam
    b = np.floor(rel / np.deg2rad(width_deg) + 0.5).astype(int)
    keys, assign = np.unique(b, return_inverse=True)
    n = len(keys)
    cdir = np.array([dirs[assign == i].mean(axis=0) for i in range(n)])
    cdir /= np.linalg.norm(cdir, axis=1, keepdims=True)
    cfac = np.array([fac[assign == i].mean() for i in range(n)])
    return assign.astype(np.int32), cdir, cfac


def image_direction(op, lab_xy, omega):
    """A lab-frame in-plane direction at rotation omega, in image coordinates (col, row): turned into the sample frame
    (R(omega)^T about the rotation axis z), then (x, y) -> (-x, y)."""
    c, s = np.cos(omega), np.sin(omega)
    x, y = lab_xy[..., 0], lab_xy[..., 1]
    xs, ys = c * x + s * y, -s * x + c * y                                     # R(omega)^T (x, y)
    d = np.stack([-xs, ys], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


class AbsorptionReference:
    """The ReferenceModel's data with self-absorption (module docstring); parameter "mu" (Ny, Nx) in addition to the
    ReferenceModel's (prm["mu"] = zeros reproduces it exactly). The attenuation geometry uses the bin-centre angles
    and azimuths and the initial geometry (no derivative of A with respect to the geometry)."""

    def __init__(self, ref, op, classes=None):
        """classes: None (every segment its own exact exit direction) or exit_classes(op, ...)."""
        self.ref = ref
        k0 = ref.k0 / np.linalg.norm(ref.k0)
        if not np.allclose(np.abs(k0), [0, 0, 1]):
            raise NotImplementedError("AbsorptionReference assumes the rotation axis along z")
        R, CP = ref.R, ref.C * ref.P
        shape = (ref.Ny, ref.Nx)
        ang = np.asarray(op.angles, np.float64)
        p0 = np.asarray(op.cfg["p_direction_0"], np.float64)[:2]
        if classes is None:
            dirs, fac = exit_directions(op)
            classes = (np.arange(CP, dtype=np.int32), dirs, fac)
        self.assign, cdir, cfac = classes
        n = len(cfac)
        self.Lin = np.stack([frame_matrix(shape, -image_direction(op, p0, ang[a])) for a in range(R)])
        self.E = np.stack([np.stack([cfac[i] * frame_matrix(shape, image_direction(op, cdir[i], ang[a]))
                                     for i in range(n)]) for a in range(R)])  # (R, n, Npix, Npix)

    def attenuation(self, mu):
        """A (R, Npix, C P)."""
        m = np.asarray(mu, np.float64).ravel()
        tot = (self.Lin @ m)[:, :, None] + np.einsum("anxy,y->axn", self.E, m)[:, :, self.assign]
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
            Wn = np.zeros(W.shape[:2] + (self.E.shape[1],))
            np.add.at(Wn, (slice(None), slice(None), self.assign), W)                    # per class
            g = -np.einsum("axy,ax->y", self.Lin, W.sum(axis=2)) - np.einsum("anxy,axn->y", self.E, Wn)
            out["mu"] = g.reshape(np.shape(prm["mu"]))
        pf_groups = [g_ for g_ in groups if g_ in ("rot", "sigma", "two_theta", "eta_offset", "axis_tilt")]
        if pf_groups:
            S = np.einsum("asx,kx,axj->askj", Pm, cf, A)
            G = np.einsum("asj,askj->akj", r, S).reshape(R, K, C, P)
            out.update(ref._pf_grads(prm, G, pf_groups))
        return out


class AbsorptionOperator:
    """SinglePhaseForwardOperator's model with self-absorption on the GPU (module docstring; the discretisation of
    AbsorptionReference with the same classes): direct_cl / adjoint_cl (the transpose) on pyopencl arrays,
    mu_gradient(c, r) = d<r, A_mu(c)>/d mu. set_mu(mu) computes the attenuation weights W (classes x angles x pixels).

    Cost: per batch and class one weighted projection and one PF product masked to the class's segments (n_classes
    times the operator's projections; the masked PF products read all non-zeros). Native projector (plain kernels),
    pf_mode "sparse" or "generated", rotation axis z."""

    def __init__(self, op, classes=None, angle_chunk=32):
        import pyopencl as cl
        import pyopencl.array as clarray
        from pathlib import Path
        from ..operators.single_phase_forward_operator import SPMM_TY
        if op.projector != "native" or op.pf_mode not in ("sparse", "generated"):
            raise NotImplementedError("AbsorptionOperator needs the native projector and pf_mode 'sparse' or 'generated'")
        k0 = np.asarray(op.cfg["k_direction_0"], np.float64)
        if not np.allclose(np.abs(k0 / np.linalg.norm(k0)), [0, 0, 1]):
            raise NotImplementedError("AbsorptionOperator assumes the rotation axis along z")
        self.cl, self.ca = cl, clarray
        self.op, self.q = op, op.queue
        self.classes = classes if classes is not None else exit_classes(op)
        assign, cdir, cfac = self.classes
        self.NC = len(cfac)
        R, CP = int(op.N_Omega), int(op.N_seg)
        self.R, self.Npix, self.CP = R, op.Nx * op.Ny, CP
        self.M = frame_size((op.Ny, op.Nx))
        self.chunk = int(angle_chunk)
        ang = np.asarray(op.angles, np.float64)
        p0 = np.asarray(op.cfg["p_direction_0"], np.float64)[:2]
        Ld = -image_direction(op, p0, ang)                                            # (R, 2)
        Ed = image_direction(op, cdir[None, :, :], ang[:, None])                      # (R, NC, 2)
        dev = lambda a, dt=np.float32: clarray.to_device(self.q, np.ascontiguousarray(a, dtype=dt))
        self.Ld, self.Ed = dev(Ld), dev(Ed.reshape(-1, 2))
        self.fac = dev(cfac)
        self.masks = [dev((np.asarray(assign) == n).astype(np.uint8), np.uint8) for n in range(self.NC)]
        rad = op.radon
        opts = [f"-DNB={rad.bins_per_item}", f"-DNTAP={rad.n_taps}", f"-DSPMM_TY={SPMM_TY}"]
        if rad.kernel == "trapezoid":
            opts.append("-DTRAPEZOID")
        prg = cl.Program(op.ctx, Path(__file__).with_name("absorption.cl").read_text()).build(options=opts)
        self.k = {n: cl.Kernel(prg, n) for n in ("frame_sample", "frame_cumsum", "frame_back", "frame_back_T",
                                                  "frame_cumsum_T", "frame_sample_T", "att_weights", "radon_forward_w",
                                                  "radon_backward_w", "contract_w", "spmm_forward_masked",
                                                  "spmm_adjoint_masked", "split_gw", "scaled_product")}
        nd = self.chunk * (1 + self.NC)
        self.fr = clarray.empty(self.q, (nd * self.M * self.M,), np.float32)
        self.D = clarray.empty(self.q, (nd * self.M * self.M,), np.float32)
        self.Lb = clarray.empty(self.q, (self.chunk * self.Npix,), np.float32)
        self.Eb = clarray.empty(self.q, (self.chunk * self.NC * self.Npix,), np.float32)
        self.W = clarray.empty(self.q, (self.NC * R * self.Npix,), np.float32)
        self.set_mu(np.zeros((op.Ny, op.Nx), np.float32))

    # ------------------------------------------------------------------ line integrals
    def _integrate(self, mu, dirs, d_off, n, out):
        """out[0:n] (n, Npix) = line integrals of mu along dirs[d_off:d_off+n] (device arrays)."""
        op, q, M, k = self.op, self.q, self.M, self.k
        g = lambda m: (-(-m // 64) * 64, n)
        k["frame_sample"](q, g(M * M), (64, 1), mu.data, dirs.data, self.fr.data, np.int32(op.Nx), np.int32(op.Ny),
                          np.int32(M), np.int32(d_off))
        k["frame_cumsum"](q, g(M), (64, 1), self.fr.data, self.D.data, np.int32(M))
        k["frame_back"](q, g(self.Npix), (64, 1), self.D.data, dirs.data, out.data, np.int32(op.Nx), np.int32(op.Ny),
                        np.int32(M), np.int32(d_off))

    def _integrate_T(self, g_in, dirs, d_off, n, gmu):
        """gmu += sum over the directions d_off .. d_off+n-1 of the transposed line integrals of g_in (n, Npix)."""
        op, q, M, k = self.op, self.q, self.M, self.k
        g = lambda m: (-(-m // 64) * 64, n)
        k["frame_back_T"](q, g(M * M), (64, 1), g_in.data, dirs.data, self.D.data, np.int32(op.Nx), np.int32(op.Ny),
                          np.int32(M), np.int32(d_off))
        k["frame_cumsum_T"](q, g(M), (64, 1), self.D.data, self.fr.data, np.int32(M))
        k["frame_sample_T"](q, (-(-self.Npix // 64) * 64,), (64,), self.fr.data, dirs.data, gmu.data,
                            np.int32(op.Nx), np.int32(op.Ny), np.int32(M), np.int32(n), np.int32(d_off))

    def set_mu(self, mu):
        """The attenuation map mu (Ny, Nx), 1 / pixel; computes the weights W."""
        ca = self.ca
        self.mu = ca.to_device(self.q, np.ascontiguousarray(mu, dtype=np.float32).reshape(-1))
        for a0 in range(0, self.R, self.chunk):
            Ra = min(self.chunk, self.R - a0)
            self._integrate(self.mu, self.Ld, a0, Ra, self.Lb)
            self._integrate(self.mu, self.Ed, a0 * self.NC, Ra * self.NC, self.Eb)
            self.k["att_weights"](self.q, (-(-self.Npix // 64) * 64, Ra, self.NC), (64, 1, 1), self.Lb.data,
                                  self.Eb.data, self.fac.data, self.W.data, np.int32(self.Npix), np.int32(self.R),
                                  np.int32(a0), np.int32(Ra), np.int32(self.NC))

    def _woff(self, n):
        return np.uint64(n * self.R * self.Npix)

    # ------------------------------------------------------------------ projections
    def _radon_fwd(self, img, sino, n, Kmax):
        op, rad = self.op, self.op.radon
        n_strips = -(-op.My // rad.bins_per_item)
        for k4, n4 in rad._slices(Kmax, 128):
            self.k["radon_forward_w"](self.q, (-(-n4 // 32) * 32, -(-n_strips // 4) * 4, self.R), (32, 4, 1),
                                      img.data, sino.data, rad.geo_gpu.data, self.W.data, self._woff(n),
                                      np.int32(op.Nx), np.int32(op.Ny),
                                      np.int32(op.My), np.int32(self.R), np.int32(Kmax // 4), np.int32(k4),
                                      np.int32(n4), rad.scale)

    def _radon_bwd(self, sino, img, n, Kmax, accumulate):
        op, rad = self.op, self.op.radon
        for k4, n4 in rad._slices(Kmax, 128):
            self.k["radon_backward_w"](self.q, (-(-n4 // 32) * 32, -(-op.Nx // 4) * 4, op.Ny), (32, 4, 1), sino.data,
                                       img.data, rad.geo_gpu.data, self.W.data, self._woff(n), np.int32(op.Nx),
                                       np.int32(op.Ny),
                                       np.int32(op.My), np.int32(self.R), np.int32(Kmax // 4), np.int32(k4),
                                       np.int32(n4), np.int32(accumulate))

    def _spmm_fwd(self, ib, sino, data, mask):
        from ..operators.single_phase_forward_operator import SPMM_TY
        op = self.op
        Kb, Kmax, CP, My = op.batches[ib]["K_batch"], op.K_batch_max, self.CP, int(op.My)
        sb = op._pf_batch(ib, forward=True)
        ls, chunk = op._spmm_setup()
        kc = min(Kb, chunk)
        self.k["spmm_forward_masked"](self.q, (-(-CP // ls) * ls, -(-My // SPMM_TY), self.R), (ls, 1, 1), sino.data,
                                      sb["row_ptr_f"].data, sb["col_k"].data, sb["val_f"].data, data.data, mask.data,
                                      np.int32(self.R), np.int32(My), np.int32(CP), np.int32(Kmax), np.int32(Kb),
                                      np.int32(kc), self.cl.LocalMemory(4 * SPMM_TY * kc))

    def _spmm_adj(self, ib, data, sino, mask, alpha):
        from ..operators.single_phase_forward_operator import SPMM_TY
        op = self.op
        Kb, Kmax, CP, My = op.batches[ib]["K_batch"], op.K_batch_max, self.CP, int(op.My)
        sb = op._pf_batch(ib, forward=False)
        ls, chunk = op._spmm_setup()
        jc = min(CP, chunk)
        self.k["spmm_adjoint_masked"](self.q, (-(-Kb // ls) * ls, -(-My // SPMM_TY), self.R), (ls, 1, 1), data.data,
                                      sb["row_ptr_a"].data, sb["col_j"].data, sb["val_a"].data, sino.data, mask.data,
                                      np.int32(self.R), np.int32(My), np.int32(Kb), np.int32(CP), np.int32(Kmax),
                                      np.float32(alpha), np.int32(jc), self.cl.LocalMemory(4 * SPMM_TY * jc))

    # ------------------------------------------------------------------ operator
    def direct_cl(self, c, data):
        op = self.op
        data.fill(0.0)
        for ib, b in enumerate(op.batches):
            op.radon.gather(c, op._img_k, b["k_start"], b["K_batch"], op.K_batch_max)
            for n in range(self.NC):
                self._radon_fwd(op._img_k, op.coeffs_sino_C, n, op.K_batch_max)
                self._spmm_fwd(ib, op.coeffs_sino_C, data, self.masks[n])

    def adjoint_cl(self, data, c):
        """c = the transpose of direct_cl applied to data."""
        op = self.op
        for ib, b in enumerate(op.batches):
            for n in range(self.NC):
                self._spmm_adj(ib, data, op.coeffs_sino_C, self.masks[n], self.R / np.pi)
                self._radon_bwd(op.coeffs_sino_C, op._img_k, n, op.K_batch_max, n > 0)
            op.radon.scatter(op._img_k, c, b["k_start"], b["K_batch"], op.K_batch_max)

    def direct(self, c):
        ca, op = self.ca, self.op
        cd = ca.to_device(self.q, np.ascontiguousarray(c, dtype=np.float32))
        d = ca.empty(self.q, op.data_shape, np.float32)
        self.direct_cl(cd, d)
        return d.get()

    def adjoint(self, r):
        ca, op = self.ca, self.op
        rd = ca.to_device(self.q, np.ascontiguousarray(r, dtype=np.float32))
        c = ca.empty(self.q, op.coeff_shape, np.float32)
        self.adjoint_cl(rd, c)
        return c.get()

    def mu_gradient(self, c, r):
        """d<r, A_mu(c)>/d mu (Ny, Nx), NumPy."""
        ca, op, q = self.ca, self.op, self.q
        cd = c if isinstance(c, ca.Array) else ca.to_device(q, np.ascontiguousarray(c, dtype=np.float32))
        rd = r if isinstance(r, ca.Array) else ca.to_device(q, np.ascontiguousarray(r, dtype=np.float32))
        Qall = ca.zeros(q, (self.NC * self.R * self.Npix,), np.float32)
        Kmax = op.K_batch_max
        for ib, b in enumerate(op.batches):
            op.radon.gather(cd, op._img_k, b["k_start"], b["K_batch"], Kmax)
            for n in range(self.NC):
                self._spmm_adj(ib, rd, op.coeffs_sino_C, self.masks[n], 1.0)
                self.k["contract_w"](q, (-(-self.Npix // 64) * 64, self.R), (64, 1), op._img_k.data,
                                     op.coeffs_sino_C.data, op.radon.geo_gpu.data, Qall.data, self._woff(n),
                                     np.int32(op.Nx),
                                     np.int32(op.Ny), np.int32(op.My), np.int32(self.R), np.int32(Kmax // 4),
                                     op.radon.scale)
        n_all = self.NC * self.R * self.Npix
        self.k["scaled_product"](q, (min(n_all, 1 << 22),), None, self.W.data, Qall.data, Qall.data,
                                 np.float32(-1.0), np.uint64(n_all))                # dL/d(L + f E) = -W Q
        gmu = ca.zeros(q, (self.Npix,), np.float32)
        for a0 in range(0, self.R, self.chunk):
            Ra = min(self.chunk, self.R - a0)
            self.k["split_gw"](q, (-(-self.Npix // 64) * 64, Ra), (64, 1), Qall.data, self.fac.data, self.Lb.data,
                               self.Eb.data, np.int32(self.Npix), np.int32(self.R), np.int32(a0), np.int32(Ra),
                               np.int32(self.NC))
            self._integrate_T(self.Lb, self.Ld, a0, Ra, gmu)
            self._integrate_T(self.Eb, self.Ed, a0 * self.NC, Ra * self.NC, gmu)
        return gmu.get().reshape(op.Ny, op.Nx)
