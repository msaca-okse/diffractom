"""A basis local in space and in orientation, for refinement: every basis function is one orientation (and kernel
width) allowed only in a region of the slice (LocalSupport + LocalSinglePhaseForwardOperator). Giving every region
its own copy of an orientation lets the refinement turn them independently, so the reconstructed orientation can
vary across a grain (intragranular gradients) with few parameters: a coefficient per allowed voxel plus four
numbers (rotation, width) per region.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..operators.local_support import LocalSupport


def split_regions(mats, sigmas, support: LocalSupport, mode: str = "components", tile: int = 8, min_pairs: int = 1):
    """Copies of the orientations, one per region of their support.

    mats (K, 3, 3), sigmas (K,): the basis; support: LocalSupport of those K orientations.
    mode "components": each 8-connected component of an orientation's voxels is a region; "tiles": the voxels are
    grouped by tile x tile blocks of the grid (a piecewise-constant orientation field of that resolution);
    "components+tiles": components cut by the tiles.
    Regions with fewer than min_pairs voxels are dropped. Returns (mats, sigmas, support, parent) with parent[k'] the
    original orientation of the new orientation k'."""
    if mode not in ("components", "tiles", "components+tiles"):
        raise ValueError(f"mode: 'components', 'tiles' or 'components+tiles', not {mode!r}")
    Ny, Nx = support.Ny, support.Nx
    k, iy, ix = support.k, support.iy, support.ix
    order = np.argsort(k, kind="stable")
    k, iy, ix = k[order], iy[order], ix[order]
    bounds = np.searchsorted(k, np.arange(support.K + 1))
    new_k, new_iy, new_ix, parent = [], [], [], []
    nk = 0
    for kk in range(support.K):
        a, b = bounds[kk], bounds[kk + 1]
        if a == b:
            continue
        y, x = iy[a:b], ix[a:b]
        if mode == "tiles":
            lab = (y // tile) * (-(-Nx // tile)) + x // tile
        else:
            y0, x0 = y.min(), x.min()
            m = np.zeros((y.max() - y0 + 1, x.max() - x0 + 1), bool)
            m[y - y0, x - x0] = True
            L, _ = ndimage.label(m, structure=np.ones((3, 3)))
            lab = L[y - y0, x - x0].astype(np.int64)
            if mode == "components+tiles":
                lab = lab * (Ny * Nx) + (y // tile) * (-(-Nx // tile)) + x // tile
        uniq, inv = np.unique(lab, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(uniq))
        keep = cnt >= min_pairs
        remap = np.full(len(uniq), -1, np.int64)
        remap[keep] = nk + np.arange(int(keep.sum()))
        sel = remap[inv] >= 0
        new_k.append(remap[inv][sel])
        new_iy.append(y[sel])
        new_ix.append(x[sel])
        parent += [kk] * int(keep.sum())
        nk += int(keep.sum())
    parent = np.asarray(parent, np.int64)
    sup = LocalSupport(np.concatenate(new_k), np.concatenate(new_iy), np.concatenate(new_ix), nk, (Ny, Nx))
    return np.asarray(mats)[parent], np.asarray(sigmas)[parent], sup, parent
