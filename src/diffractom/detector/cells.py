"""Detector cells and their weights onto the operator's data channels.

A cell is a set of detector pixels whose counts are summed (default: every pixel its own cell). Summing raw counts
keeps the Poisson likelihood exact: the sum of independent Poisson counts is Poisson with the summed mean, and the
model of a cell is the sum of its pixels' models. Only information is lost, where the model varies within a cell.

The model of pixel p (detector/raw_model.py):
    mu_p = factor_p sum_r s_r rho_r(2theta_p) M_r(eta_p)
with M_r(eta) the operator's prediction of ring r (a data channel) linearly interpolated between the centres of its
eta bins (the operator is built with eta bins finer than the pixels' azimuthal extent, e.g. 0.25 deg), s_r a scale
per ring and rho_r the radial profile (profiles.py). So a cell is a sparse row of weights on the data segments
seg = eta_bin * N_rings + ring of one frame:
    mu_cell = sum_seg w_cell,seg M[frame, seg],   w_cell,seg = sum_{p in cell} factor_p s_r rho_r(2theta_p) h_j(eta_p)
(h_j: the linear interpolation weight of eta bin j). Pixels outside every ring's profile window are not modelled
(their counts are not used).
"""
import numpy as np


def eta_interpolation(eta, n_eta, eta_angle_range):
    """Linear interpolation between eta-bin centres: (j0, j1, w0, w1) per value; periodic if the range is 2 pi,
    otherwise clamped at the end bins and invalid (j0 = -1) outside the range."""
    a0, a1 = float(eta_angle_range[0]), float(eta_angle_range[1])
    d = (a1 - a0) / n_eta
    periodic = abs((a1 - a0) - 2 * np.pi) < 1e-9
    e = np.asarray(eta, dtype=np.float64)
    if periodic:
        e = a0 + np.mod(e - a0, 2 * np.pi)
    inside = (e >= a0) & (e <= a1)
    u = (e - a0) / d - 0.5
    j0 = np.floor(u).astype(np.int64)
    f = u - j0
    j1 = j0 + 1
    if periodic:
        j0 %= n_eta
        j1 %= n_eta
    else:
        lo, hi = j0 < 0, j1 > n_eta - 1
        j0 = np.clip(j0, 0, n_eta - 1)
        j1 = np.clip(j1, 0, n_eta - 1)
        f = np.where(lo | hi, 0.0, f)
        j0 = np.where(inside, j0, -1)
    return j0, j1, 1.0 - f, f


class DetectorCells:
    """Cells of pixels and their weights on the (eta bin, ring) segments of the operator's data.

    pixels : DetectorPixels
    profiles : RadialProfiles (one per ring of the operator, in its order)
    n_eta, eta_angle_range : the operator's eta bins (op.N_eta, op.eta_angle_range in radians)
    ring_scales : (R,) s_r (default 1)
    cell_of_pixel : (P,) cell number of every used pixel (>= 0; -1: not used), or None: every modelled pixel its own
        cell
    fixed_cells : with cell_of_pixel: keep its numbering (n_cells = max + 1) even where a cell has no pixel inside a
        ring's window (its row is empty), so that events stay valid when profiles or geometry change (refinement)

    Attributes: n_cells, n_rings, n_seg; pixel_cell (P,) cell of every used pixel (-1: not modelled);
    ptr (n_cells + 1,), seg (nnz,) int32, w (nnz,) float32 (CSR, sorted by cell, then segment); seg_total (n_seg,)
    = the sum of all cells' weights (a frame's total model is sum_seg seg_total[seg] M[frame, seg]).
    """

    def __init__(self, pixels, profiles, n_eta, eta_angle_range, ring_scales=None, cell_of_pixel=None,
                 fixed_cells=False):
        self.pixels = pixels
        self.profiles = profiles
        self.n_eta = int(n_eta)
        self.eta_angle_range = np.asarray(eta_angle_range, dtype=np.float64)
        self.n_rings = profiles.n_rings
        self.n_seg = self.n_eta * self.n_rings
        P = pixels.n_pixels
        # radial weights: (pixel, ring) pairs with rho > 0
        pix, ring, val = [], [], []
        for r in range(self.n_rings):
            lo, hi = profiles.window(r)
            sel = np.flatnonzero((pixels.two_theta >= lo) & (pixels.two_theta <= hi))
            v = profiles(r, pixels.two_theta[sel]) * pixels.factor[sel]
            keep = v > 0
            pix.append(sel[keep])
            ring.append(np.full(keep.sum(), r, dtype=np.int64))
            val.append(v[keep])
        pix, ring, val = np.concatenate(pix), np.concatenate(ring), np.concatenate(val)
        j0, j1, w0, w1 = eta_interpolation(pixels.eta[pix], self.n_eta, self.eta_angle_range)
        ok = j0 >= 0
        pix, ring, val, j0, j1, w0, w1 = (a[ok] for a in (pix, ring, val, j0, j1, w0, w1))
        # d w1 / d eta = 1 / bin (0 where the interpolation is clamped)
        dbin = (float(self.eta_angle_range[1]) - float(self.eta_angle_range[0])) / self.n_eta
        dw1 = np.where(j1 != j0, 1.0 / dbin, 0.0)
        modelled = np.zeros(P, dtype=bool)
        modelled[pix] = True
        if cell_of_pixel is None:
            pixel_cell = np.full(P, -1, dtype=np.int64)
            pixel_cell[modelled] = np.arange(modelled.sum())
        else:
            pixel_cell = np.asarray(cell_of_pixel, dtype=np.int64).copy()
            if pixel_cell.shape != (P,):
                raise ValueError("cell_of_pixel needs one cell number per used pixel")
            if not fixed_cells:
                pixel_cell[~modelled] = -1
                # renumber the cells that hold modelled pixels 0 .. C-1
                used, inv = np.unique(pixel_cell[modelled], return_inverse=True)
                pixel_cell[modelled] = inv
        self.pixel_cell = pixel_cell
        if fixed_cells and cell_of_pixel is not None:
            self.n_cells = int(pixel_cell.max()) + 1
        else:
            self.n_cells = int(pixel_cell.max()) + 1 if modelled.any() else 0
        keep = pixel_cell[pix] >= 0
        pix, ring, val, j0, j1, w0, w1, dw1 = (a[keep] for a in (pix, ring, val, j0, j1, w0, w1, dw1))
        # entries (cell, seg, value) for both interpolation nodes; the ring scale enters later (scaled weights)
        cell = pixel_cell[pix]
        self._cell = np.concatenate((cell, cell))
        self._seg = np.concatenate((j0 * self.n_rings + ring, j1 * self.n_rings + ring))
        self._ring = np.concatenate((ring, ring))
        self._val = np.concatenate((val * w0, val * w1))
        # per entry: the pixel, the radial factor (factor x rho) and the eta weight with its eta derivative
        self._pixe = np.concatenate((pix, pix))
        self._vbase = np.concatenate((val, val))
        self._wnode = np.concatenate((w0, w1))
        self._dwdeta = np.concatenate((-dw1, dw1))
        nz = (self._val > 0) | (self._dwdeta != 0)
        for name in ("_cell", "_seg", "_ring", "_val", "_pixe", "_vbase", "_wnode", "_dwdeta"):
            setattr(self, name, getattr(self, name)[nz])
        self.set_ring_scales(np.ones(self.n_rings) if ring_scales is None else ring_scales)

    def set_ring_scales(self, ring_scales):
        """Rebuild the CSR weights with new ring scales s_r."""
        s = np.asarray(ring_scales, dtype=np.float64)
        if s.shape != (self.n_rings,) or np.any(s < 0):
            raise ValueError("one ring scale >= 0 per ring")
        self.ring_scales = s.copy()
        key = self._cell * self.n_seg + self._seg
        uk, inv = np.unique(key, return_inverse=True)
        self._inv = inv
        w = np.bincount(inv, weights=self._val * s[self._ring])
        cell = uk // self.n_seg
        self.seg = (uk % self.n_seg).astype(np.int32)
        self.w64 = w
        self.w = w.astype(np.float32)
        self.ptr = np.searchsorted(cell, np.arange(self.n_cells + 1)).astype(np.int64)
        self.seg_total = np.bincount(self.seg, weights=w, minlength=self.n_seg)
        # the same weights without the ring scale, per ring (for scale updates: the part of a cell in ring r)
        self.ring_of_seg = (np.arange(self.n_seg) % self.n_rings).astype(np.int32)

    def entry_derivative(self, dvbase=None, deta=None):
        """Derivative of the CSR weights (aligned with self.seg / self.w, float64) for per-entry derivatives of
        the radial factor (dvbase: d(factor x rho) / d parameter) and of the pixel's eta (deta)."""
        d = np.zeros(len(self._cell))
        if dvbase is not None:
            d += dvbase * self._wnode
        if deta is not None:
            d += self._vbase * self._dwdeta * deta
        d *= self.ring_scales[self._ring]
        return np.bincount(self._inv, weights=d, minlength=len(self.w))

    def profile_derivative(self, name, ring=None):
        """CSR weight derivative for a profile parameter (profiles.gradients names: U, V, W, X, Y; "shift" of
        ring `ring`). Needs profiles with a gradients method (PseudoVoigtProfiles)."""
        tt = self.pixels.two_theta[self._pixe]
        fac = self.pixels.factor[self._pixe]
        dv = np.zeros(len(self._cell))
        rings = range(self.n_rings) if ring is None else [ring]
        for r in rings:
            sel = self._ring == r
            if not sel.any():
                continue
            g = self.profiles.gradients(r, tt[sel], names=(name,))[name]
            dv[sel] = fac[sel] * g
        return self.entry_derivative(dvbase=dv)

    def geometry_derivative(self, dtt_pix, deta_pix):
        """CSR weight derivative for a geometry parameter, from the per-used-pixel derivatives of two-theta and
        eta (DetectorGeometry.jacobian on pixels.index). The pixel factor's derivative is neglected."""
        tt = self.pixels.two_theta[self._pixe]
        fac = self.pixels.factor[self._pixe]
        dv = np.zeros(len(self._cell))
        for r in range(self.n_rings):
            sel = self._ring == r
            if not sel.any():
                continue
            if hasattr(self.profiles, "gradients"):
                g = self.profiles.gradients(r, tt[sel], names=("two_theta",))["two_theta"]
            else:
                h = 1e-6
                g = (self.profiles(r, tt[sel] + h) - self.profiles(r, tt[sel] - h)) / (2 * h)
            dv[sel] = fac[sel] * g * dtt_pix[self._pixe[sel]]
        return self.entry_derivative(dvbase=dv, deta=deta_pix[self._pixe])

    def frame_cell_lookup(self):
        """(prod(frame shape),) int64: cell of every frame pixel, -1 where not modelled."""
        lut = np.full(int(np.prod(self.pixels.shape)), -1, dtype=np.int64)
        lut[self.pixels.index] = self.pixel_cell
        return lut

    @property
    def nnz(self):
        return len(self.w)
