"""
Where each orientation of a basis may be non-zero: a local basis.

A texture-tomography basis is usually global: every orientation k has an image on the whole grid, (K, Ny, Nx)
coefficients. When it is known beforehand where an orientation can occur (e.g. from point-by-point indexing:
the candidate grains of every voxel), orientation k only needs coefficients near those voxels. LocalSupport
holds the allowed (orientation, voxel) pairs; LocalSinglePhaseForwardOperator stores and projects only them.

The pairs are grouped into rectangular patches (one bounding box per orientation, or per cluster of its
voxels when they are scattered): the operator keeps the coefficients of a patch as a small dense image, with
the voxels outside the support held at zero.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, label
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def _disk(radius):
    """Offsets (dy, dx) of a disc of the given radius in voxels (radius 0: the voxel itself)."""
    r = int(np.ceil(radius))
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    keep = dy ** 2 + dx ** 2 <= radius ** 2 + 0.5
    return dy[keep], dx[keep]


class LocalSupport:
    """
    A set of (orientation k, voxel (iy, ix)) pairs for a basis of K orientations on an Ny x Nx grid
    (iy: row, ix: column, as in the (K, Ny, Nx) coefficient arrays). The pairs are kept sorted and unique
    (k, then iy, then ix).
    """

    def __init__(self, k, iy, ix, K, shape):
        self.K = int(K)
        self.Ny, self.Nx = (int(n) for n in shape)
        k = np.asarray(k, dtype=np.int64).ravel()
        iy = np.asarray(iy, dtype=np.int64).ravel()
        ix = np.asarray(ix, dtype=np.int64).ravel()
        if not (len(k) == len(iy) == len(ix)):
            raise ValueError("k, iy and ix must have the same length")
        ok = (k >= 0) & (k < self.K) & (iy >= 0) & (iy < self.Ny) & (ix >= 0) & (ix < self.Nx)
        if not ok.all():
            raise ValueError(f"{int((~ok).sum())} pairs outside the basis (K = {self.K}) or the grid "
                             f"({self.Ny} x {self.Nx})")
        lin = np.unique((k * self.Ny + iy) * self.Nx + ix)
        self.k, rem = np.divmod(lin, self.Ny * self.Nx)
        self.iy, self.ix = np.divmod(rem, self.Nx)

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_mask(cls, mask):
        """From a boolean array (K, Ny, Nx): orientation k is allowed where mask[k] is True."""
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 3:
            raise ValueError("mask must have shape (K, Ny, Nx)")
        k, iy, ix = np.nonzero(mask)
        return cls(k, iy, ix, mask.shape[0], mask.shape[1:])

    @classmethod
    def everywhere(cls, K, shape):
        """Every orientation in every voxel (the global basis; for tests)."""
        Ny, Nx = shape
        k, iy, ix = np.meshgrid(np.arange(K), np.arange(Ny), np.arange(Nx), indexing="ij")
        return cls(k, iy, ix, K, shape)

    @classmethod
    def from_candidates(cls, basis, iy, ix, orientations, symmetry, shape, tol_deg=1.0, dilate=2):
        """
        Orientation k of the basis is allowed in voxel (iy, ix) if a candidate orientation found there (e.g. by
        point-by-point indexing) lies within tol_deg of it, crystal symmetry included; then each orientation's
        voxels are grown by `dilate` voxels (a disc). Tested on simulated Al (pbp candidates, best 1-3 per voxel):
        tol 1 deg and dilate 2 give the maps of the global basis with 2-5 % of its coefficients; without growing,
        voxels at grain boundaries lose their orientation.

        basis : (K, 3, 3) rotation matrices (crystal -> sample), the operator's grid order
        iy, ix : (N,) voxel of every candidate
        orientations : (N, 3, 3) the candidates' rotation matrices, same convention
        symmetry : crystal system name ("cubic", ...), a Material or a scipy Rotation of the point group
        shape : (Ny, Nx)
        """
        from ..odf_basis import chord, equivalents, symmetry_rotations
        sym = symmetry_rotations(symmetry)
        basis = np.asarray(basis, dtype=np.float64)
        K = len(basis)
        tree = cKDTree(equivalents(Rotation.from_matrix(basis).as_quat(), sym))
        q = Rotation.from_matrix(np.asarray(orientations, dtype=np.float64)).as_quat()
        hits = tree.query_ball_point(q, chord(tol_deg))
        n = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        ks = np.concatenate([np.asarray(h, dtype=np.int64) for h in hits]) % K if n.sum() else np.zeros(0, np.int64)
        sup = cls(ks, np.repeat(np.asarray(iy), n), np.repeat(np.asarray(ix), n), K, shape)
        return sup.dilate(dilate) if dilate > 0 else sup

    # ------------------------------------------------------------------ operations

    def dilate(self, radius):
        """Every orientation's voxels grown by a disc of the given radius (in voxels)."""
        dy, dx = _disk(radius)
        k = np.repeat(self.k, len(dy))
        iy = (self.iy[:, None] + dy[None]).ravel()
        ix = (self.ix[:, None] + dx[None]).ravel()
        ok = (iy >= 0) & (iy < self.Ny) & (ix >= 0) & (ix < self.Nx)
        return LocalSupport(k[ok], iy[ok], ix[ok], self.K, (self.Ny, self.Nx))

    def restrict(self, mask):
        """Only the voxels where the (Ny, Nx) mask is True (e.g. the field of view or the sample outline)."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.Ny, self.Nx):
            raise ValueError(f"mask shape {mask.shape} != {(self.Ny, self.Nx)}")
        keep = mask[self.iy, self.ix]
        return LocalSupport(self.k[keep], self.iy[keep], self.ix[keep], self.K, (self.Ny, self.Nx))

    def union(self, other):
        if (other.K, other.Ny, other.Nx) != (self.K, self.Ny, self.Nx):
            raise ValueError("supports of different bases or grids")
        return LocalSupport(np.r_[self.k, other.k], np.r_[self.iy, other.iy], np.r_[self.ix, other.ix], self.K,
                            (self.Ny, self.Nx))

    def to_mask(self):
        """Boolean (K, Ny, Nx) array (for small problems)."""
        m = np.zeros((self.K, self.Ny, self.Nx), dtype=bool)
        m[self.k, self.iy, self.ix] = True
        return m

    @property
    def n_pairs(self):
        return len(self.k)

    @property
    def fill(self):
        """Share of the (K, Ny, Nx) coefficients that are allowed."""
        return self.n_pairs / float(self.K * self.Ny * self.Nx)

    def orientations_used(self):
        return np.unique(self.k)

    def per_voxel(self):
        """(Ny, Nx) number of allowed orientations per voxel."""
        return np.bincount(self.iy * self.Nx + self.ix, minlength=self.Ny * self.Nx).reshape(self.Ny, self.Nx)

    def __repr__(self):
        return (f"LocalSupport({self.n_pairs} pairs, {len(self.orientations_used())} of {self.K} orientations, "
                f"{self.Ny} x {self.Nx} voxels, fill {100 * self.fill:.2f} %)")


# ---------------------------------------------------------------------- patches

def _merge_boxes(boxes):
    """Merge overlapping boxes [y0, y1, x0, x1] (inclusive) until none overlap."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed and len(boxes) > 1:
        changed = False
        out = []
        while boxes:
            b = boxes.pop()
            i = 0
            while i < len(boxes):
                c = boxes[i]
                if b[0] <= c[1] and c[0] <= b[1] and b[2] <= c[3] and c[2] <= b[3]:
                    b = [min(b[0], c[0]), max(b[1], c[1]), min(b[2], c[2]), max(b[3], c[3])]
                    boxes.pop(i)
                    changed = True
                else:
                    i += 1
            out.append(b)
        boxes = out
    return boxes


def build_patches(k, iy, ix, K, split=True, merge_gap=2, min_box_fill=0.5):
    """
    Group sorted, unique (k, iy, ix) pairs into rectangular patches.

    One patch per orientation (its bounding box), unless split and its voxels fill less than min_box_fill of
    the box: then its voxels are clustered (connected after growing by merge_gap voxels), every cluster gets its
    bounding box, and overlapping boxes are merged. Patches are ordered by orientation, then position.

    Returns a dict: pk (orientation of every patch), y0, x0, h, w, coff (offset of the patch's coefficients in the
    flat vector, row-major within the box), n_coef, mask (uint8 (n_coef,): 1 where a pair is), pair_index (index of
    every pair's coefficient, in the pairs' order).
    """
    k = np.asarray(k, dtype=np.int64)
    iy = np.asarray(iy, dtype=np.int64)
    ix = np.asarray(ix, dtype=np.int64)
    n = len(k)
    if n == 0:
        raise ValueError("empty support")
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    ends = np.r_[starts[1:], n]
    ymin = np.minimum.reduceat(iy, starts)
    ymax = np.maximum.reduceat(iy, starts)
    xmin = np.minimum.reduceat(ix, starts)
    xmax = np.maximum.reduceat(ix, starts)
    count = ends - starts
    area = (ymax - ymin + 1) * (xmax - xmin + 1)
    sparse = split & (count < min_box_fill * area)

    pk, py0, px0, ph, pw = [], [], [], [], []
    pair_patch = np.empty(n, dtype=np.int64)
    struct = np.ones((3, 3), dtype=bool)
    grow = np.ones((2 * merge_gap + 1, 2 * merge_gap + 1), dtype=bool) if merge_gap > 0 else None
    for g in range(len(starts)):
        s, e = starts[g], ends[g]
        if not sparse[g]:
            pair_patch[s:e] = len(pk)
            pk.append(k[s]); py0.append(ymin[g]); px0.append(xmin[g])
            ph.append(ymax[g] - ymin[g] + 1); pw.append(xmax[g] - xmin[g] + 1)
            continue
        y = iy[s:e] - ymin[g]
        x = ix[s:e] - xmin[g]
        m = np.zeros((ymax[g] - ymin[g] + 1, xmax[g] - xmin[g] + 1), dtype=bool)
        m[y, x] = True
        lab, nl = label(binary_dilation(m, grow) if grow is not None else m, structure=struct)
        lv = lab[y, x]
        boxes = []
        for c in range(1, nl + 1):
            sel = lv == c
            if sel.any():
                boxes.append([y[sel].min(), y[sel].max(), x[sel].min(), x[sel].max()])
        boxes = sorted(_merge_boxes(boxes))
        for b in boxes:
            inside = (y >= b[0]) & (y <= b[1]) & (x >= b[2]) & (x <= b[3])
            pair_patch[s:e][inside] = len(pk)
            pk.append(k[s]); py0.append(ymin[g] + b[0]); px0.append(xmin[g] + b[2])
            ph.append(b[1] - b[0] + 1); pw.append(b[3] - b[2] + 1)

    pk = np.asarray(pk, dtype=np.int64)
    py0 = np.asarray(py0, dtype=np.int64)
    px0 = np.asarray(px0, dtype=np.int64)
    ph = np.asarray(ph, dtype=np.int64)
    pw = np.asarray(pw, dtype=np.int64)
    sizes = ph * pw
    coff = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)
    n_coef = int(sizes.sum())
    pair_index = coff[pair_patch] + (iy - py0[pair_patch]) * pw[pair_patch] + (ix - px0[pair_patch])
    mask = np.zeros(n_coef, dtype=np.uint8)
    mask[pair_index] = 1
    assert int(mask.sum()) == n, "patches overlap"
    return dict(pk=pk, y0=py0, x0=px0, h=ph, w=pw, coff=coff, n_coef=n_coef, mask=mask, pair_index=pair_index)
