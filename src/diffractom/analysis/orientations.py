"""Orientation per voxel from the kernel coefficients: the mode of the voxel's ODF, found by mean shift.

A reconstruction gives per voxel the weights c_k of Gaussian kernels centred on the orientations U_k of the basis,
an ODF f(q) = sum_k c_k exp(-omega(q, U_k)^2 / (2 sigma_k^2)) with omega the misorientation angle. Its strongest
kernel (or the weighted medoid of the strongest kernels) is restricted to the basis orientations; on a coarse
basis the true orientation usually lies between them. refine_orientations climbs from the strongest kernel to the
nearest maximum of f by mean shift (Fukunaga & Hostetler, IEEE Trans. Inf. Theory 21, 32-40, 1975; Cheng, IEEE
Trans. PAMI 17, 790-799, 1995): with each kernel taken at its symmetry-equivalent closest to the current estimate q,

    q <- normalise( sum_k c_k / sigma_k^2 exp(-omega_k^2 / (2 sigma_k^2)) q_k ),

the weighted chordal mean of unit quaternions (sign-aligned with q), which moves q uphill on f; at a fixed point
the gradient of f vanishes to first order in omega.

Reading the orientations out of the ODF by mean shift instead of taking a basis orientation follows textom 0.7.21
(calculate_preferred_orientations; M. Frewein, M. Allain, T. Grunewald et al.,
https://gitlab.fresnel.fr/textom/textom; Frewein et al., IUCrJ 11, 809-820, 2024). This is an independent
implementation (no textom code is used); the kernels here are those of diffractom's basis (the width sigma_k of
each orientation).
"""
import numpy as np
from scipy.spatial.transform import Rotation

from ..utils.grid import Grid


def _quat(mats):
    """Unit quaternions (w, x, y, z) of rotation matrices (..., 3, 3)."""
    q = Rotation.from_matrix(np.asarray(mats, np.float64).reshape(-1, 3, 3)).as_quat()  # (x, y, z, w)
    return np.concatenate([q[:, 3:], q[:, :3]], axis=1).reshape(np.shape(mats)[:-2] + (4,))


def _mats(q):
    q = np.asarray(q, np.float64)
    return Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], axis=-1).reshape(-1, 4)).as_matrix().reshape(
        q.shape[:-1] + (3, 3))


def _qmul(a, b):
    """Quaternion product a b (w, x, y, z), broadcasting."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def symmetry_quaternions(symmetry):
    """The proper rotations of a crystal symmetry ("cubic", "hexagonal", ..., as Grid) as quaternions (G, 4).
    They act on the right of the orientation, U S (crystal frame)."""
    mats = np.stack([g.as_matrix() for g in Grid._point_group(symmetry)])
    return _quat(mats)


def refine_orientations(coeffs, grid_mats, sigmas, symmetry="cubic", support=None, top_k=24, max_iter=50,
                        tol_deg=1e-3, max_elements=5e7):
    """Per voxel, the orientation of the ODF maximum next to the voxel's strongest kernel.

    Parameters
    ----------
    coeffs : (K, Ny, Nx) array, the reconstruction (coeffs[k] = image of orientation k)
    grid_mats : (K, 3, 3), the basis orientations U_k (as used by the operator)
    sigmas : float or (K,), the kernel widths in radians (the operator's grid sigmas)
    symmetry : crystal symmetry name (as Grid.from_random_fundamental_zone)
    support : optional (Ny, Nx) bool, voxels to evaluate (default: every voxel with a positive coefficient)
    top_k : the strongest kernels of a voxel that enter its ODF (kernels further down add little near the mode)
    max_iter, tol_deg : mean-shift iterations and the step (deg) below which a voxel has converged
    max_elements : bound on K x (voxels per chunk) when the top kernels are selected

    Returns
    -------
    dict with
      orientations : (Ny, Nx, 3, 3), NaN where the voxel has no positive coefficient (or is outside support)
      start : (Ny, Nx, 3, 3), the strongest kernel's orientation (the starting point)
      shift_deg : (Ny, Nx), misorientation between start and result
      spread_deg : (Ny, Nx), coefficient-weighted rms misorientation to the result of the top kernels within
                   3 sigma of it (an orientation spread of the voxel's dominant component)
      iterations : the number of mean-shift iterations of the slowest chunk
    """
    C = np.asarray(coeffs)
    K, Ny, Nx = C.shape
    Q = _quat(grid_mats)                                               # (K, 4)
    sig = np.broadcast_to(np.asarray(sigmas, np.float64), (K,)).copy()
    S = symmetry_quaternions(symmetry)                                 # (G, 4)
    flat = C.reshape(K, Ny * Nx)
    vox = np.arange(Ny * Nx)
    if support is not None:
        vox = vox[np.asarray(support, bool).ravel()]
    top_k = int(min(top_k, K))
    out_q = np.full((Ny * Nx, 4), np.nan)
    start_q = np.full((Ny * Nx, 4), np.nan)
    spread = np.full(Ny * Nx, np.nan)
    chunk = max(1, int(max_elements // K))
    tol = np.cos(np.deg2rad(tol_deg) / 2)
    n_iter_max = 0
    for c0 in range(0, vox.size, chunk):
        v = vox[c0:c0 + chunk]
        cv = np.asarray(flat[:, v], np.float64)                         # (K, V)
        if top_k < K:
            idx = np.argpartition(-cv, top_k - 1, axis=0)[:top_k].T     # (V, N)
        else:
            idx = np.broadcast_to(np.arange(K), (v.size, K)).copy()
        w0 = np.take_along_axis(cv.T, idx, axis=1)                      # (V, N)
        w0 = np.where(w0 > 0, w0, 0.0)
        ok = w0.max(axis=1) > 0
        v, idx, w0 = v[ok], idx[ok], w0[ok]
        if v.size == 0:
            continue
        s = sig[idx]                                                     # (V, N)
        E = _qmul(Q[idx][:, :, None, :], S[None, None, :, :])            # (V, N, G, 4) equivalents U_k S_g
        q = Q[idx[np.arange(v.size), np.argmax(w0, axis=1)]]             # (V, 4) strongest kernel
        start_q[v] = q
        active = np.ones(v.size, bool)
        for it in range(int(max_iter)):
            d = np.einsum("vngc,vc->vng", E[active], q[active])          # (Va, N, G)
            g = np.argmax(np.abs(d), axis=2)                              # nearest equivalent per kernel
            e = np.take_along_axis(E[active], g[:, :, None, None], axis=2)[:, :, 0]   # (Va, N, 4)
            dg = np.take_along_axis(d, g[:, :, None], axis=2)[:, :, 0]
            e = e * np.sign(dg)[:, :, None]
            om = 2.0 * np.arccos(np.clip(np.abs(dg), 0.0, 1.0))
            sa = s[active]
            w = w0[active] * np.exp(-0.5 * (om / sa) ** 2) / sa ** 2
            qn = np.einsum("vn,vnc->vc", w, e)
            nrm = np.linalg.norm(qn, axis=1)
            good = nrm > 0
            qn[good] /= nrm[good, None]
            qn[~good] = q[active][~good]
            moved = np.abs(np.einsum("vc,vc->v", qn, q[active])) < tol
            q[active] = qn
            n_iter_max = max(n_iter_max, it + 1)
            act_idx = np.nonzero(active)[0]
            active[act_idx[~moved]] = False
            if not active.any():
                break
        out_q[v] = q
        # spread of the dominant component: top kernels within 3 sigma of the mode
        d = np.abs(np.einsum("vngc,vc->vng", E, q)).max(axis=2)
        om = 2.0 * np.arccos(np.clip(d, 0.0, 1.0))
        near = om < 3.0 * s
        wn = np.where(near, w0, 0.0)
        tot = wn.sum(axis=1)
        spread[v] = np.degrees(np.sqrt(np.where(tot > 0, (wn * om ** 2).sum(axis=1) / np.maximum(tot, 1e-300), np.nan)))
    found = np.isfinite(out_q[:, 0])
    ori = np.full((Ny * Nx, 3, 3), np.nan)
    st = np.full((Ny * Nx, 3, 3), np.nan)
    ori[found] = _mats(out_q[found])
    st[found] = _mats(start_q[found])
    cosang = np.full(Ny * Nx, np.nan)
    if found.any():
        dd = np.abs(np.einsum("vgc,vc->vg", _qmul(start_q[found][:, None, :], S[None]), out_q[found])).max(axis=1)
        cosang[found] = np.clip(dd, 0, 1)
    return {"orientations": ori.reshape(Ny, Nx, 3, 3), "start": st.reshape(Ny, Nx, 3, 3),
            "shift_deg": np.degrees(2 * np.arccos(cosang)).reshape(Ny, Nx), "spread_deg": spread.reshape(Ny, Nx),
            "iterations": n_iter_max}
