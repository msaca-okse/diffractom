"""Radial (two-theta) profiles of the rings: how the intensity of a ring is spread across the ring.

A ring's intensity at azimuth eta is spread over two-theta by a profile rho_r (instrument resolution: beam divergence
and energy spread, detector point spread, the sample's extent along the beam, lattice-spacing variations). The
raw-detector model predicts a pixel as  factor_p sum_r s_r rho_r(2theta_p) M_r(eta_p)  (detector/raw_model.py), with
rho_r normalised to a unit integral over two-theta (radians). Profiles are tabulated (linear interpolation, zero
outside the ring's window): RadialProfiles.gaussian, or RadialProfiles.from_powder from the azimuthally averaged
pattern of the data themselves (an empirical instrument function, the counts are not changed).
"""
import numpy as np


class RadialProfiles:
    """Tabulated, normalised profiles rho_r(2theta) (radians), one per ring.

    grids : list of (n_r,) increasing two-theta values (radians); values : list of (n_r,) profile values >= 0.
    The profile is the linear interpolation of the table and zero outside [grid[0], grid[-1]] (the ring's window).
    The tables are normalised to a unit integral (trapezoid rule) on construction.
    """

    def __init__(self, grids, values):
        if len(grids) != len(values):
            raise ValueError("one grid and one table of values per ring")
        self.grids, self.values = [], []
        for g, v in zip(grids, values):
            g = np.asarray(g, dtype=np.float64)
            v = np.clip(np.asarray(v, dtype=np.float64), 0.0, None)
            if g.ndim != 1 or g.shape != v.shape or len(g) < 2 or np.any(np.diff(g) <= 0):
                raise ValueError("a profile needs an increasing grid of >= 2 points and as many values")
            area = np.trapezoid(v, g)
            if not area > 0:
                raise ValueError("a profile must have a positive integral")
            self.grids.append(g)
            self.values.append(v / area)

    @property
    def n_rings(self):
        return len(self.grids)

    def window(self, r):
        return float(self.grids[r][0]), float(self.grids[r][-1])

    def __call__(self, r, two_theta):
        """rho_r at the given two-theta values (radians), 0 outside the window."""
        g, v = self.grids[r], self.values[r]
        tt = np.asarray(two_theta, dtype=np.float64)
        return np.interp(tt, g, v, left=0.0, right=0.0)

    def centre(self, r):
        """Profile mean (radians)."""
        g, v = self.grids[r], self.values[r]
        return float(np.trapezoid(v * g, g))

    def width(self, r):
        """Profile standard deviation (radians)."""
        g, v = self.grids[r], self.values[r]
        c = self.centre(r)
        return float(np.sqrt(np.trapezoid(v * (g - c) ** 2, g)))

    @classmethod
    def gaussian(cls, two_theta_rings, widths, cutoff=4.0, n=161):
        """Gaussians of standard deviation widths (radians, scalar or per ring) at the rings' two-theta, cut at
        cutoff widths."""
        tt = np.atleast_1d(np.asarray(two_theta_rings, dtype=np.float64))
        w = np.broadcast_to(np.asarray(widths, dtype=np.float64), tt.shape)
        grids = [np.linspace(c - cutoff * s, c + cutoff * s, n) for c, s in zip(tt, w)]
        return cls(grids, [np.exp(-0.5 * ((g - c) / s) ** 2) for g, c, s in zip(grids, tt, w)])

    @classmethod
    def from_powder(cls, two_theta_pixels, summed_counts, factor, two_theta_rings, half_widths, bin_width,
                    background=None, edge_fraction=0.15, smooth_bins=0):
        """Empirical profiles from the azimuthally averaged pattern of the data.

        two_theta_pixels, factor : (P,) per used pixel (DetectorPixels.two_theta, .factor)
        summed_counts : (P,) counts of every pixel summed over the frames used
        two_theta_rings : (R,) ring positions (radians); half_widths : scalar or (R,) half window widths (radians)
        bin_width : histogram bin in two-theta (radians); a fraction of the pixel's two-theta extent is enough,
            since the pixels of a ring sample every offset
        Windows of neighbouring rings that overlap are cut at the midpoint between the two rings.
        background : None (the data have none) or "edges": a straight line through the mean of the outer
            edge_fraction of the window on each side is subtracted
        smooth_bins : running mean over this many bins (0: none)

        The pattern is the mean of counts / factor over the pixels of each bin (the azimuthal average of the
        intensity per unit relative solid angle).
        """
        tt = np.asarray(two_theta_pixels, dtype=np.float64)
        cts = np.asarray(summed_counts, dtype=np.float64)
        fac = np.asarray(factor, dtype=np.float64)
        rings = np.atleast_1d(np.asarray(two_theta_rings, dtype=np.float64))
        hw = np.broadcast_to(np.asarray(half_widths, dtype=np.float64), rings.shape)
        lo, hi = rings - hw, rings + hw
        order = np.argsort(rings)
        for a, b in zip(order[:-1], order[1:]):          # neighbouring windows that overlap end at the midpoint
            if lo[b] < hi[a]:
                mid = 0.5 * (rings[a] + rings[b])
                hi[a], lo[b] = mid, mid
        grids, values = [], []
        for r in range(len(rings)):
            nb = max(int(np.ceil((hi[r] - lo[r]) / bin_width)), 4)
            edges = np.linspace(lo[r], hi[r], nb + 1)
            sel = (tt >= lo[r]) & (tt < hi[r])
            b = np.minimum(((tt[sel] - lo[r]) / (edges[1] - edges[0])).astype(np.int64), nb - 1)
            num = np.bincount(b, weights=cts[sel], minlength=nb)
            den = np.bincount(b, weights=fac[sel], minlength=nb)
            centres = 0.5 * (edges[1:] + edges[:-1])
            good = den > 0
            prof = np.interp(centres, centres[good], num[good] / den[good])
            if smooth_bins and smooth_bins > 1:
                k = np.ones(int(smooth_bins)) / int(smooth_bins)
                prof = np.convolve(prof, k, mode="same")
            if background == "edges":
                ne = max(int(round(edge_fraction * nb)), 1)
                x0, x1 = centres[:ne].mean(), centres[-ne:].mean()
                y0, y1 = prof[:ne].mean(), prof[-ne:].mean()
                prof = prof - (y0 + (y1 - y0) * (centres - x0) / (x1 - x0))
            elif background is not None:
                raise ValueError("background must be None or 'edges'")
            prof = np.clip(prof, 0.0, None)
            # the table spans the window (edge points at the window limits)
            g = np.concatenate(([lo[r]], centres, [hi[r]]))
            v = np.concatenate(([prof[0]], prof, [prof[-1]]))
            grids.append(g)
            values.append(v)
        return cls(grids, values)
