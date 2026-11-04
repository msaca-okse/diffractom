"""A smooth, non-negative background in the raw-detector model, fitted with the texture instead of being subtracted.

    b(frame, cell) = sum_l beta[group(frame), l] g(cell, l),   g(cell, l) = sum_{p in cell} f_p B_j(2theta_p) H_k(eta_p)

B_j are hat functions on two-theta knots (piecewise linear in two-theta, constant beyond the end knots), H_k periodic
hat functions on n_eta_nodes azimuths (1: constant in eta), l = j * n_eta_nodes + k, f_p the pixel factor (solid
angle x polarisation, or solid angle only). With beta >= 0 every non-negative piecewise-bilinear background on these
knots is representable, and nothing sharper: a background cannot imitate a Bragg peak if the knots are spaced well
beyond the peak width. MIDAS's XRD-CT notes measure the background as the dominant error of ring intensities (it
varies with radius and azimuth, and estimates from ring-free radii are biased under the peaks); here the likelihood
of all pixels, between the rings and under them, decides it together with the texture.

frame_groups: (n_frames,) group of every frame (frames of one group share beta), e.g. one group per rotation bin
pooling the translations, or None: one group per frame.
"""
import numpy as np


def _hat_weights(x, knots):
    """Indices and weights of the two hat functions at x on increasing knots (clamped beyond the ends)."""
    x = np.clip(np.asarray(x, np.float64), knots[0], knots[-1])
    j = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, len(knots) - 2)
    t = (x - knots[j]) / (knots[j + 1] - knots[j])
    return j, j + 1, 1.0 - t, t


class BackgroundBasis:
    def __init__(self, pixels, pixel_cell, n_cells, tt_knots, n_eta_nodes=1, frame_groups=None, n_frames=None,
                 factor=None):
        self.tt_knots = np.asarray(tt_knots, np.float64)
        if self.tt_knots.ndim != 1 or len(self.tt_knots) < 2 or np.any(np.diff(self.tt_knots) <= 0):
            raise ValueError("tt_knots: >= 2 increasing two-theta values")
        self.n_tt = len(self.tt_knots)
        self.n_eta_nodes = int(n_eta_nodes)
        self.L = self.n_tt * self.n_eta_nodes
        pc = np.asarray(pixel_cell, np.int64)
        use = np.flatnonzero(pc >= 0)
        f = pixels.factor[use] if factor is None else np.asarray(factor, np.float64)[use]
        j0, j1, a0, a1 = _hat_weights(pixels.two_theta[use], self.tt_knots)
        if self.n_eta_nodes == 1:
            k0 = k1 = np.zeros(len(use), np.int64)
            b0, b1 = np.ones(len(use)), np.zeros(len(use))
        else:
            u = np.mod(pixels.eta[use], 2 * np.pi) / (2 * np.pi) * self.n_eta_nodes
            k0 = np.floor(u).astype(np.int64) % self.n_eta_nodes
            k1 = (k0 + 1) % self.n_eta_nodes
            b1 = u - np.floor(u)
            b0 = 1.0 - b1
        cells, ls, ws = [], [], []
        for jj, aa in ((j0, a0), (j1, a1)):
            for kk, bb in ((k0, b0), (k1, b1)):
                w = f * aa * bb
                nz = w > 0
                cells.append(pc[use][nz])
                ls.append((jj * self.n_eta_nodes + kk)[nz])
                ws.append(w[nz])
        cells, ls, ws = np.concatenate(cells), np.concatenate(ls), np.concatenate(ws)
        key = cells * self.L + ls
        uk, inv = np.unique(key, return_inverse=True)
        w = np.bincount(inv, weights=ws)
        cell = uk // self.L
        self.idx = (uk % self.L).astype(np.int32)
        self.w = w.astype(np.float32)
        self.n_cells = int(n_cells)
        self.ptr = np.searchsorted(cell, np.arange(self.n_cells + 1)).astype(np.int64)
        self.col_total = np.bincount(self.idx, weights=w, minlength=self.L)
        if frame_groups is None:
            if n_frames is None:
                raise ValueError("n_frames is needed without frame_groups")
            frame_groups = np.arange(int(n_frames))
        self.group_of_frame = np.asarray(frame_groups, np.int32)
        self.n_groups = int(self.group_of_frame.max()) + 1

    @property
    def shape(self):
        return (self.n_groups, self.L)

    def sensitivity(self, exposure, frames=None):
        """(n_groups, L): sum over all cells of the frames (all, or a boolean / index selection) of exposure x g."""
        e = np.asarray(exposure, np.float64).ravel()
        sel = np.ones(len(e), bool) if frames is None else np.zeros(len(e), bool)
        if frames is not None:
            sel[frames] = True
        per_group = np.bincount(self.group_of_frame[sel], weights=e[sel], minlength=self.n_groups)
        return per_group[:, None] * self.col_total[None, :]

    def values(self, beta, frame, cell):
        """NumPy: the background of events (frame, cell) for coefficients beta (n_groups, L)."""
        beta = np.asarray(beta, np.float64)
        out = np.zeros(len(frame))
        g = self.group_of_frame[frame]
        a, b = self.ptr[cell], self.ptr[cell + 1]
        n = (b - a).max() if len(a) else 0
        for t in range(int(n)):
            ok = a + t < b
            pos = (a + t)[ok]
            out[ok] += self.w[pos] * beta[g[ok], self.idx[pos]]
        return out
