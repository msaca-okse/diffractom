"""Single-crystal Bragg spectra averaged exactly over Voronoi cells of orientation space, on the GPU.

A crystallite U of a reflection G (crystal direction g, spacing d) in a beam b scatters at the single wavelength
lambda = 2 d t, t = b . (U g). Its spectrum is a delta function. Averaging over a cell Z of orientation space:

    (1/|Z|) int_Z delta(lambda - 2 d b.Ug) dV = 1/(2 d) 1/(8 pi^2 |Z|) int_0^2pi Lambda(n(t, phi), g) dphi,    t = lambda / 2d,

where n(t, phi) runs over the cone of half-angle arccos(t) about b and Lambda(n, g) is the rotation angle of the fibre
{U : U g = n} inside Z. A cell of the misorientation Voronoi mesh is, in quaternions, a convex cone (the half-spaces
q.(a - b) >= 0 and q.(a + b) >= 0 for the other centres b), and a fibre is a great circle, so Lambda follows in closed
form from two arctangents. Only the integral over the azimuth phi is numerical (midpoint rule, arc step h_arc).

The wavelength broadening (lattice-spacing spread, pulse tail) does not depend on U, so it is applied afterwards to the
cell spectrum: wavelength_kernel(), exact_cell_matrix(). No orientation width enters.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyopencl as cl
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ..utils.instrument import raden_pulse_tail
from .bragg_edge_tomographic_operator import emg_profile


def _wxyz(rotations: Rotation) -> np.ndarray:
    return np.ascontiguousarray(rotations.as_quat()[..., [3, 0, 1, 2]])


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R(q) v for scalar-first quaternions."""
    w, x = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(x, v)
    return v + w * t + np.cross(x, t)


def beam_directions(angles_deg) -> np.ndarray:
    """Beam directions R_z(angle) (0, 1, 0), as the Bragg-edge operators."""
    a = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    return np.stack([-np.sin(a), np.cos(a), np.zeros_like(a)], axis=-1)


class VoronoiCells:
    """Voronoi cells of `centres` (crystal -> sample) in the misorientation metric.

    symmetry: the proper rotations of the crystal (right-multiplied). partner: an optional sample rotation; then a cell
    is the region around a centre together with its image under the partner (two convex pieces).
    """

    def __init__(self, centres: Rotation, symmetry: Rotation, partner: Rotation | None = None):
        refs = [centres] if partner is None else [centres, partner * centres]
        self.n_cells = len(centres)
        self.n_symmetry = len(symmetry)
        self.n_pieces = len(refs)
        self.images = np.concatenate([_wxyz(r * symmetry[o]) for r in refs for o in range(self.n_symmetry)])
        self._identity = int(np.argmin(symmetry.magnitude()))
        self._centres = [_wxyz(r * symmetry[self._identity]) for r in refs]

    def _own(self, k: int, p: int) -> int:
        return (p * self.n_symmetry + self._identity) * self.n_cells + k

    def constraints(self, k: int, p: int, r_max: float):
        """Centre a of piece p of cell k and the normalised half-space normals of all centres within 2 r_max of it."""
        a = self._centres[p][k]
        ang = 2 * np.arccos(np.clip(np.abs(self.images @ a), 0, 1))
        sel = np.flatnonzero(ang <= 2 * r_max)
        sel = sel[sel != self._own(k, p)]
        b = self.images[sel] * np.sign(self.images[sel] @ a)[:, None]
        rows = np.concatenate([a[None] - b, a[None] + b])
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        return a, rows

    def piece_radii(self, n_samples: int = 20_000_000, margin_deg: float = 3.0, seed: int = 100) -> np.ndarray:
        """(n_cells, n_pieces): a bound on the rotation angle from the centre within every piece, radians (largest
        angle among random rotations + a margin; the sum of the cell volumes checks it)."""
        img = np.concatenate([self.images, -self.images])
        tree = cKDTree(img)
        rmax = np.zeros(len(self.images))
        for s in range(int(np.ceil(n_samples / 2_000_000))):
            q = _wxyz(Rotation.random(2_000_000, random_state=seed + s))
            _, j = tree.query(q, workers=-1)
            j %= len(self.images)
            np.maximum.at(rmax, j, 2 * np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", q, self.images[j])), 0, 1)))
        r = np.zeros((self.n_cells, self.n_pieces))
        for p in range(self.n_pieces):
            r[:, p] = rmax[[self._own(k, p) for k in range(self.n_cells)]] + np.deg2rad(margin_deg)
        return r


def _amplitude(lam, d, F2, V_A3):
    """Amplitude of the delta function of a reflection, as build_bragg_matrix_cpu (cm^-1 per unit volume fraction, A)."""
    V_cm3 = V_A3 * 1e-24
    return 1e8 * (lam * 1e-8) ** 4 * F2 * 1e-24 / (V_cm3 ** 2 * 2.0 * (lam / (2 * d)) ** 2)


class CellIntegrator:
    """Cell averages of the delta spectra on a fine wavelength grid (midpoints of cells of equal width)."""

    def __init__(self, table: dict, beams, lam_fine, ctx: cl.Context, queue: cl.CommandQueue, h_arc_deg: float = 0.1,
                 n_rho: int = 300, n_psi: int = 720, precision: str = "single", local_size: int = 64):
        if precision not in ("single", "double"):
            raise ValueError("precision must be 'single' or 'double'")
        self.ctx, self.queue, self.local = ctx, queue, int(local_size)
        self.n_rho, self.n_psi, self.h_arc = n_rho, n_psi, np.deg2rad(h_arc_deg)
        self.lam = np.asarray(lam_fine, dtype=np.float64)
        self.h_lam = float(self.lam[1] - self.lam[0])
        g = np.asarray(table["g_vecs"], dtype=np.float64)
        self.g = g / np.linalg.norm(g, axis=1, keepdims=True)
        self.d = np.asarray(table["d"], dtype=np.float64)
        self.F2 = np.asarray(table["F2"], dtype=np.float64)
        self.V = float(table["V"])
        self.families, self.family = np.unique(np.round(self.d, 6), return_inverse=True)
        self.j_edge = np.searchsorted(self.lam - self.h_lam / 2, 2 * self.d, side="left")
        b = np.asarray(beams, dtype=np.float64)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        up = np.where(np.abs(b[:, 2:]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
        e1 = np.cross(b, up)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        self.beams, self.e1, self.e2 = b, e1, np.cross(b, e1)
        self.real = np.float32 if precision == "single" else np.float64
        src = Path(__file__).with_suffix(".cl").read_text()
        self.program = cl.Program(ctx, src).build(options=["-DUSE_DOUBLE"] if precision == "double" else [])
        self.k_arc = self.program.arc_integrals
        self.k_ring = self.program.cap_ring
        self._bufs: dict[str, cl.Buffer] = {}
        self._static = {"lam": self._upload("lam", self.lam.astype(self.real)),
                        "two_d": self._upload("two_d", (2 * self.d).astype(self.real)),
                        "g": self._upload("g", self._v4(self.g)), "b": self._upload("b", self._v4(self.beams)),
                        "e1": self._upload("e1", self._v4(self.e1)), "e2": self._upload("e2", self._v4(self.e2))}

    def _v4(self, v: np.ndarray) -> np.ndarray:
        return np.concatenate([v, np.zeros((*v.shape[:-1], 1))], axis=-1).astype(self.real)

    def _upload(self, name: str, arr: np.ndarray) -> cl.Buffer:
        arr = np.ascontiguousarray(arr)
        b = self._bufs.get(name)
        if b is None or b.size < arr.nbytes:
            b = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=max(int(arr.nbytes * 1.3), 64))
            self._bufs[name] = b
        cl.enqueue_copy(self.queue, b, arr)
        return b

    def _reserve(self, name: str, nbytes: int) -> cl.Buffer:
        b = self._bufs.get(name)
        if b is None or b.size < nbytes:
            b = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=max(int(nbytes * 1.3), 64))
            self._bufs[name] = b
        return b

    def _tasks(self, p: int, a: np.ndarray, r: float):
        """The (beam, reflection) pairs whose cones can meet the piece, and their wavelength ranges."""
        nc = _rotate(a, self.g)
        cb = self.beams @ nc.T
        thc = np.arccos(np.clip(cb, -1, 1))
        lo = 2 * self.d * np.cos(np.minimum(thc + r, np.pi)) - self.h_lam
        hi = 2 * self.d * np.cos(np.maximum(thc - r, 0.0)) + self.h_lam
        j0 = np.searchsorted(self.lam, lo, side="left")
        j1 = np.minimum(np.searchsorted(self.lam, hi, side="right"), self.j_edge)
        ib, ir = np.nonzero(j1 > j0)
        sc = np.sqrt(np.maximum(1 - cb[ib, ir] ** 2, 0.0))
        phic = np.arctan2(np.einsum("ij,ij->i", self.e2[ib], nc[ir]), np.einsum("ij,ij->i", self.e1[ib], nc[ir]))
        geo = np.stack([cb[ib, ir], sc, phic, thc[ib, ir]], axis=1)
        return np.full(len(ib), p, dtype=np.uint32), ib.astype(np.uint32), ir.astype(np.uint32), nc[ir], geo, j0[ib, ir], j1[ib, ir]

    def _volume(self, a: np.ndarray, rows: cl.Buffer, nrows: int, r: float) -> float:
        g0 = self.g[0]
        nc = _rotate(a, g0)
        u = np.array([-nc[1], nc[0], 0.0]) if abs(nc[2]) < 0.9 else np.array([0.0, -nc[2], nc[1]])
        u /= np.linalg.norm(u)
        v = np.cross(nc, u)
        drho = r / self.n_rho
        out = self._reserve("ring", self.n_rho * self.real().nbytes)
        args = [rows, np.uint32(nrows), a.astype(self.real), self._v4(nc), self._v4(g0), self._v4(u), self._v4(v),
                self.real(drho), np.uint32(self.n_psi), out]
        for i, x in enumerate(args):
            self.k_ring.set_arg(i, x)
        cl.enqueue_nd_range_kernel(self.queue, self.k_ring, (self.n_rho,), None)
        res = np.empty(self.n_rho, dtype=self.real)
        cl.enqueue_copy(self.queue, res, out)
        return float(res.astype(np.float64).sum()) * drho * (2 * np.pi / self.n_psi) / (8 * np.pi ** 2)

    def cell_density(self, cells: VoronoiCells, k: int, radii) -> tuple[np.ndarray, float]:
        """S (n_family, n_beam, n_fine): the sum over the reflections of one spacing of amplitude x (cell average of the
        delta function), and the cell volume (SO(3) = 1)."""
        pieces = [cells.constraints(k, p, float(radii[p])) for p in range(cells.n_pieces)]
        nrow = np.array([len(w) for _, w in pieces], dtype=np.uint32)
        row0 = np.concatenate([[0], np.cumsum(nrow)[:-1]]).astype(np.uint32)
        rows = np.concatenate([w for _, w in pieces]).astype(self.real)
        centres = np.stack([a for a, _ in pieces])
        r = np.asarray(radii, dtype=np.float64)[:cells.n_pieces]
        b_rows = self._upload("rows", rows)
        vol = 0.0
        for p in range(cells.n_pieces):
            # rows of this piece only: a sub-buffer would need alignment, so the volume kernel gets them on their own
            b_p = self._upload("rows_p", rows[row0[p]:row0[p] + nrow[p]])
            vol += self._volume(centres[p], b_p, int(nrow[p]), float(r[p]))
        parts = [self._tasks(p, centres[p], float(r[p])) for p in range(cells.n_pieces)]
        t_piece, t_beam, t_refl, nc, geo, j0, j1 = (np.concatenate(x) for x in zip(*parts))
        counts = j1 - j0
        total = int(counts.sum())
        S = np.zeros((len(self.families), len(self.beams), len(self.lam)))
        if total == 0:
            return S, vol
        first = np.concatenate([[0], np.cumsum(counts)[:-1]])
        item_task = np.repeat(np.arange(len(counts), dtype=np.uint32), counts)
        item_j = (np.arange(total) - np.repeat(first, counts) + np.repeat(j0, counts)).astype(np.uint32)
        padded = int(np.ceil(total / self.local)) * self.local
        F_buf = self._reserve("F", padded * self.real().nbytes)
        s = self._static
        args = [self._upload("item_task", item_task), self._upload("item_j", item_j), np.uint32(total), s["lam"],
                self._upload("t_piece", t_piece), self._upload("t_beam", t_beam), self._upload("t_refl", t_refl),
                self._upload("t_nc", self._v4(nc)), self._upload("t_geo", geo.astype(self.real)),
                self._upload("piece_a", centres.astype(self.real)), self._upload("piece_row0", row0),
                self._upload("piece_nrow", nrow), self._upload("piece_r", r.astype(self.real)), b_rows,
                s["b"], s["e1"], s["e2"], s["g"], s["two_d"], self.real(self.h_arc), F_buf]
        for i, x in enumerate(args):
            self.k_arc.set_arg(i, x)
        cl.enqueue_nd_range_kernel(self.queue, self.k_arc, (padded,), (self.local,))
        F = np.empty(total, dtype=self.real)
        cl.enqueue_copy(self.queue, F, F_buf)
        ir = t_refl[item_task]
        lf, dj = self.lam[item_j], self.d[ir]
        edge = np.clip((2 * dj - (lf - self.h_lam / 2)) / self.h_lam, 0, 1)
        weight = _amplitude(lf, dj, self.F2[ir], self.V) * edge / (2 * dj * 8 * np.pi ** 2)
        index = (self.family[ir] * len(self.beams) + t_beam[item_task].astype(np.int64)) * len(self.lam) + item_j
        S += np.bincount(index, weights=F.astype(np.float64) * weight, minlength=S.size).reshape(S.shape)
        return S / vol, vol


def wavelength_kernel(lam_out, lam_fine, e0: float, pulse_tail_fn=raden_pulse_tail, pulse_tail_scale: float = 1.0):
    """Sparse K[out, fine] = h EMG(lam_out - lam_fine; sigma = lam_fine e0, tail tau(lam_fine) / 1e4)."""
    lam_out, lam_fine = np.asarray(lam_out, np.float64), np.asarray(lam_fine, np.float64)
    h = lam_fine[1] - lam_fine[0]
    tail = pulse_tail_scale * pulse_tail_fn(lam_fine) / 1e4
    width = lam_fine * e0
    rows, cols, vals = [], [], []
    for j0 in range(0, len(lam_fine), 2000):
        js = np.arange(j0, min(j0 + 2000, len(lam_fine)))
        lo = np.searchsorted(lam_out, lam_fine[js] - 8 * width[js]).min()
        hi = np.searchsorted(lam_out, lam_fine[js] + 14 * tail[js] + 8 * width[js]).max()
        if hi <= lo:
            continue
        o = np.arange(lo, hi)
        kv = emg_profile(lam_out[o][:, None] - lam_fine[js][None, :], width[js][None, :], tail[js][None, :]) * h
        kv[~np.isfinite(kv)] = 0
        ii, jj = np.nonzero(kv > 1e-12 * kv.max())
        rows.append(o[ii]); cols.append(js[jj]); vals.append(kv[ii, jj])
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(lam_out), len(lam_fine)))


def exact_cell_matrix(table: dict, cells: VoronoiCells, beams, lam, e0: float, ctx: cl.Context, queue: cl.CommandQueue,
                      *, h_lam: float = 0.25e-3, h_arc_deg: float = 0.1, radii=None, pulse_tail_fn=raden_pulse_tail,
                      pulse_tail_scale: float = 1.0, precision: str = "single", progress=None):
    """Spectra of the cells: B (n_beam, n_cells, n_lam) in cm^-1 per unit volume fraction, and the cell volumes.

    h_lam: fine wavelength step (resolve the kernel: lam e0); h_arc_deg: arc step along the cone. Errors of the defaults
    relative to much finer steps: 5e-5.
    """
    lam = np.asarray(lam, dtype=np.float64)
    lam_fine = np.arange(lam.min() - 0.08, lam.max() + 0.02, h_lam) + h_lam / 2      # peaks just outside reach the range
    if radii is None:
        radii = cells.piece_radii()
    integrator = CellIntegrator(table, beams, lam_fine, ctx, queue, h_arc_deg=h_arc_deg, precision=precision)
    kernel = wavelength_kernel(lam, lam_fine, e0, pulse_tail_fn, pulse_tail_scale)
    B = np.zeros((len(integrator.beams), cells.n_cells, len(lam)), dtype=np.float32)
    volumes = np.zeros(cells.n_cells)
    for k in range(cells.n_cells):
        S, volumes[k] = integrator.cell_density(cells, k, radii[k])
        B[:, k] = (kernel @ S.sum(0).T).T
        if progress is not None:
            progress(k)
    return B, volumes
