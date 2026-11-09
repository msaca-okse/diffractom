"""
Residual orientation search: which orientations, missing from a reconstruction's basis, the data still ask for, and where.

A fit with Poisson counts b, prediction mu (background included), detector gain g and weights w leaves the score
(gradient of the Poisson log-likelihood with respect to the prediction) and the Fisher information

    e = w / g (b / mu - 1),        d = w / (g mu).

For a candidate orientation k and an image pixel p (a tile of the sample, at the resolution of the scoring operator),
the score test of adding a coefficient c_kp is

    z_kp = (A^T e)_kp / sqrt((A^T d)_kp),

the matched filter of the candidate's predicted pattern in the residual, normalised by its expected spread. A real
missing crystal gives large z on the angles that were fitted and on held-out angles. A null for the size of z comes
from the same residual with its azimuths scrambled (``scramble_eta``): orientation-free, but with the same counting
statistics and model errors.

``score_candidates`` scores a list of candidate orientations (e.g. a uniform grid over the fundamental zone) in chunks,
for several (e, d) pairs at once (e.g. real and scrambled) and several row sets (fitted / held-out angles), and keeps
per pixel the ``top_k`` candidates with the largest z on the first row set. Each chunk's operator makes its PF matrix
once for all inputs and back-projects only the scored pixels (``SinglePhaseForwardOperator.adjoint_multi_cl``); the
result equals separate adjoints of the row-masked inputs bitwise.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

from .utils.arrays import check_device

__all__ = ["poisson_score_inputs", "scramble_eta", "score_candidates", "Candidates", "select_candidates", "refine_candidates",
           "candidate_support", "orientation_weights", "prune_added"]

_SCORE = r"""
// z = E / sqrt(max(D, 1e-30)), 0 where D <= 0: the host's float32 expression (correctly rounded division and square
// root, NaN propagated as numpy.maximum does), so the scores equal the host's bitwise
__kernel void score_z(__global const float *E, __global const float *D, __global float *Z, const int n)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    const float d = D[i];
    const float m = (d != d) ? d : fmax(d, 1e-30f);
    float z = E[i] / sqrt(m);
    if (d <= 0.0f) z = 0.0f;
    Z[i] = z;
}
"""


def poisson_score_inputs(b, mu, gain, weights, floor=0.25):
    """e = w / g (b / mu - 1) and d = w / (g mu) (float32, the data's shape), with mu >= floor x g (counts in the
    gain's units; cells with weight 0 give 0). b, mu: data arrays; gain, weights: broadcastable to them."""
    b = np.asarray(b, np.float32)
    g = np.asarray(gain, np.float32)
    w = np.asarray(weights, np.float32)
    m = np.maximum(np.maximum(np.asarray(mu, np.float32), 0), floor * g)
    e = np.where(w > 0, w / g * (b / m - 1), 0).astype(np.float32)
    d = np.where(w > 0, w / (g * m), 0).astype(np.float32)
    return e, d


def scramble_eta(a, n_eta, n_channels, block=12, seed=1, margin=10, mask=None):
    """The null of the search: a (N_Omega, My, n_eta * n_channels) array (segment = eta * n_channels + channel) with
    the azimuth of every channel turned by a random number of bins (margin .. n_eta - margin) per block of ``block``
    rotation angles. Breaks the link between orientation and pattern, keeps the counting statistics. mask: optional
    array multiplied in (e.g. the segment weights > 0)."""
    from concurrent.futures import ThreadPoolExecutor
    No = a.shape[0]
    shifts = np.random.default_rng(seed).integers(margin, n_eta - margin, size=(No // block + 1, n_channels))
    out = np.empty_like(a)

    def run(b_):                      # every block of rows is its own permutation copy (blocks in threads: same values)
        s = shifts[b_ // block]
        perm = np.empty(n_eta * n_channels, np.int64)
        for r in range(n_channels):
            perm[np.arange(n_eta) * n_channels + r] = ((np.arange(n_eta) + s[r]) % n_eta) * n_channels + r
        out[b_:b_ + block] = a[b_:b_ + block][:, :, perm]
        if mask is not None:
            out[b_:b_ + block] *= mask
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(run, range(0, No, block)))
    return out


def score_candidates(make_operator, rotations, inputs, row_masks, pixels, top_k=64, chunk=30000, verbose=False,
                     reuse_operator=True, nrhs=4, rows_per_group=16, gpu_scores=True):
    """Score test of candidate orientations per pixel; keep the best ``top_k`` per pixel.

    Parameters
    ----------
    make_operator : callable(rotations (n, 3, 3)) -> SinglePhaseForwardOperator (pf_mode "sparse" or "generated",
        native projector) for a chunk of candidates; all chunks must share the image grid and data shape.
    rotations : (K, 3, 3) candidate orientations (crystal to sample).
    inputs : list of (E, D) pairs of device arrays in the operator's data space (any data-space factors of the model,
        such as attenuation per ray or a beam profile, already applied as their adjoint would): e.g. the real residual
        and a scrambled one (``poisson_score_inputs``, ``scramble_eta``).
    row_masks : list of (N_Omega,) bool arrays: the angle rows of each score (e.g. fitted, held-out). Candidates are
        ranked by the first.
    pixels : (npix,) flat pixel indices (x + Nx * y) of the operator's image to score.
    top_k, chunk : candidates kept per pixel; at most this many candidates per operator.
    reuse_operator : build the operator once and move it to the next chunk's orientations (set_pf_parameters). A last
        chunk at most 10 % shorter is padded with copies of its last orientation (their scores are dropped); a shorter one
        gets its own operator. The chunks are always those of ``chunk``, so exactly tied scores are ordered the same way. Needs candidates that
        share the operator's kernel widths; False calls make_operator for every chunk. Same scores either way.
    nrhs, rows_per_group : how adjoint_multi_cl splits the products (speed only; the scores do not change).
    gpu_scores : compute z on the GPU (correctly rounded division and square root: the host's values bitwise) and copy
        only z to the host (half the transfers); False computes it on the host.

    Returns
    -------
    list (one per input pair) of dicts: "idx" (npix, top_k) int32 candidate indices (-1: none), "z" (n_rows, npix,
    top_k) float32 (z[0] ranks, descending).

    The selection (score test, top-k per pixel of each chunk merged with the kept ones) runs on the host in a worker
    thread while the GPU scores the next chunk; the result is the same as computing every chunk in turn.
    """
    from concurrent.futures import ThreadPoolExecutor
    rotations = np.asarray(rotations, np.float64)
    K_all = len(rotations)
    pixels = np.asarray(pixels, np.int32)
    npix, nr, ni = len(pixels), len(row_masks), len(inputs)
    top = [(np.full((npix, top_k), -1, np.int32), [np.full((npix, top_k), -np.inf, np.float32)] +
            [np.zeros((npix, top_k), np.float32) for _ in range(nr - 1)]) for _ in range(ni)]
    t0 = time.perf_counter()

    def merge_rows(host, K, c0, j, p0, p1):
        """The selection of input pair j for pixels p0..p1 (every pixel's own computation, as for all pixels at once)."""
        z = []
        for s in range(nr):
            if gpu_scores:
                z.append(host[j][s][:K, p0:p1])
                continue
            e_, d_ = host[2 * j][s][:K, p0:p1], host[2 * j + 1][s][:K, p0:p1]
            zz = e_ / np.sqrt(np.maximum(d_, 1e-30))
            zz[d_ <= 0] = 0
            z.append(zz)
        i0, z0 = top[j]
        cand_i = np.concatenate([i0[p0:p1], np.tile((np.arange(K) + c0)[:, None].T, (p1 - p0, 1))], axis=1)
        cand_z = [np.concatenate([z0[s][p0:p1], z[s].T], axis=1) for s in range(nr)]
        o_ = np.argpartition(-cand_z[0], top_k - 1, axis=1)[:, :top_k]
        return (np.take_along_axis(cand_i, o_, 1).astype(np.int32),
                [np.take_along_axis(cand_z[s], o_, 1).astype(np.float32) for s in range(nr)])

    try:
        n_thr = max(1, min(8, len(__import__("os").sched_getaffinity(0))))
    except AttributeError:
        n_thr = 4
    blocks_p = [(p, min(p + -(-npix // n_thr), npix)) for p in range(0, npix, -(-npix // n_thr))]

    def merge(host, K, c0):
        with ThreadPoolExecutor(n_thr) as pool:
            for j in range(ni):
                parts = list(pool.map(lambda pb: merge_rows(host, K, c0, j, *pb), blocks_p))
                top[j] = (np.concatenate([p_[0] for p_ in parts]), [np.concatenate([p_[1][s] for p_ in parts]) for s in range(nr)])

    op = None
    pending = None
    score_prg = {}
    with ThreadPoolExecutor(1) as worker:
        for c0 in range(0, K_all, chunk):
            Rc = rotations[c0:c0 + chunk]
            K = len(Rc)
            if op is None or not reuse_operator or op.K - K > op.K // 10:
                # a fresh operator (the first, or a last chunk much shorter than the others: padding it would waste
                # more than building one)
                op = None
                op = make_operator(Rc)
            else:
                from .differentiable.update import set_pf_parameters
                Rp = Rc if K == op.K else np.concatenate([Rc, np.repeat(Rc[-1:], op.K - K, axis=0)])
                set_pf_parameters(op, rotations=Rp)
            q = op.queue
            datas = [a for pair in inputs for a in pair]
            for a in datas:
                check_device(a, op.data_shape, "input")
            if pending is not None:          # at most two chunks' host blocks at a time
                pending.result()
            if gpu_scores:
                host = [[np.empty((op.K, npix), np.float32) for _ in range(nr)] for _ in range(ni)]
                if score_prg.get("ctx") is not op.ctx:
                    prg_ = cl.Program(op.ctx, _SCORE).build(options=["-cl-fp32-correctly-rounded-divide-sqrt"])
                    score_prg.update(ctx=op.ctx, k=cl.Kernel(prg_, "score_z"),
                                     e=[clarray.empty(q, (op.K_batch_max * npix,), np.float32) for _ in range(nr)],
                                     z=clarray.empty(q, (op.K_batch_max * npix,), np.float32))
                if score_prg["e"][0].size < op.K_batch_max * npix:
                    score_prg["e"] = [clarray.empty(q, (op.K_batch_max * npix,), np.float32) for _ in range(nr)]
                    score_prg["z"] = clarray.empty(q, (op.K_batch_max * npix,), np.float32)

                def consume(i, s, k0, Kb, block, host=host):
                    n = Kb * npix
                    if i % 2 == 0:       # E: keep it until its D arrives (another group may reuse the block)
                        cl.enqueue_copy(q, score_prg["e"][s].data, block.data, byte_count=4 * n)
                        return
                    score_prg["k"](q, (-(-n // 256) * 256,), (256,), score_prg["e"][s].data, block.data,
                                   score_prg["z"].data, np.int32(n))
                    cl.enqueue_copy(q, host[i // 2][s][k0:k0 + Kb], score_prg["z"].data, is_blocking=True)
            else:
                host = [[np.empty((op.K, npix), np.float32) for _ in range(nr)] for _ in datas]

                def consume(i, s, k0, Kb, block, host=host):
                    cl.enqueue_copy(q, host[i][s][k0:k0 + Kb], block.data, is_blocking=True)

            op.adjoint_multi_cl(datas, consume, row_masks=row_masks, pixels=pixels, nrhs=nrhs,
                                rows_per_group=rows_per_group)
            pending = worker.submit(merge, host, K, c0)
            if not reuse_operator:
                del op
                op = None
            if verbose:
                print(f"  candidates {c0}..{c0 + K} scored ({time.perf_counter() - t0:.0f} s)", flush=True)
        if pending is not None:
            pending.result()
    out = []
    for i_, z_ in top:
        o_ = np.argsort(-z_[0], axis=1)
        out.append({"idx": np.take_along_axis(i_, o_, 1), "z": np.stack([np.take_along_axis(a, o_, 1) for a in z_])})
    return out


# ---------------------------------------------------------------------------------------------------------------------
# From the per-pixel top lists to candidates of the basis
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Candidates:
    """Orientations proposed by the search: rotations (n, 3, 3), score (n,) (sum of the fitted z of their tile peaks),
    tiles (list of n arrays: flat indices x + Nx y in the scoring image, shape ``tile_shape`` = (Ny, Nx))."""
    rotations: np.ndarray
    score: np.ndarray
    tiles: list
    tile_shape: tuple

    def __len__(self):
        return len(self.rotations)

    def subset(self, keep):
        keep = np.asarray(keep)
        idx = np.nonzero(keep)[0] if keep.dtype == bool else keep
        return Candidates(self.rotations[idx], self.score[idx], [self.tiles[i] for i in idx], self.tile_shape)


def _proper_rotations(symmetry):
    """(G, 4) quaternions (x, y, z, w) of the proper rotations of a point group given as matrices (G, 3, 3) (improper ones
    are dropped) or a scipy Rotation."""
    if isinstance(symmetry, Rotation):
        return symmetry.as_quat()
    m = np.asarray(symmetry, np.float64).reshape(-1, 3, 3)
    return Rotation.from_matrix(m[np.linalg.det(m) > 0]).as_quat()


def _equivalents(q, S):
    """(n, G, 4): q (n, 4) times every symmetry rotation on the crystal side (quaternions x, y, z, w)."""
    x1, y1, z1, w1 = np.moveaxis(q[:, None], -1, 0)
    x2, y2, z2, w2 = np.moveaxis(S[None], -1, 0)
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], axis=-1)


def _close(qa, Qb, cos_half):
    """(len(qa), len(Qb)) bool: misorientation below the angle whose cos(angle / 2) is cos_half. Qb: equivalents (m, G, 4)."""
    return np.abs(np.einsum("ac,bgc->abg", qa, Qb)).max(axis=2) > cos_half


def select_candidates(real, null, rotations, pixels, tile_shape, symmetry, ratio=2.0, sep_deg=8.0, link_deg=5.0, max_per_tile=3,
                      min_tiles=2):
    """Candidates from the per-pixel top lists of ``score_candidates``.

    A peak of a tile counts if its fitted z (row set 0) is at least ``ratio`` x the best z of the null in that tile and
    its z on the other row set (held-out angles) reaches the null's best there; a tile keeps at most ``max_per_tile`` peaks,
    ``sep_deg`` apart. Peaks of neighbouring tiles (8-neighbourhood) within ``link_deg`` form one candidate, with the
    orientation of its strongest peak; candidates spanning fewer than ``min_tiles`` tiles are dropped.

    A strict selection (ratio 2, min_tiles 2) keeps the false candidates few. Because added orientations cost little in the
    fit, a liberal one (e.g. ratio 1.5, min_tiles 1) with the joint fit deciding finds more crystals.

    real, null : dicts of score_candidates ("idx" (npix, top_k), "z" (n_rows, npix, top_k)); rotations: the candidate list
    they index; pixels: (npix,) flat indices of the scoring image (shape tile_shape = (Ny, Nx)); symmetry: proper rotations
    of the crystal (matrices or scipy Rotation). Returns Candidates, the largest score first.
    """
    pixels = np.asarray(pixels, np.int64)
    idx, z0, z1 = real["idx"], real["z"][0], real["z"][1]
    null0, null1 = null["z"][0].max(axis=1), null["z"][1].max(axis=1)
    ok = (idx >= 0) & (z0 >= ratio * null0[:, None]) & (z1 >= null1[:, None])
    Ny, Nx = tile_shape
    empty = Candidates(np.zeros((0, 3, 3)), np.zeros(0), [], tuple(tile_shape))
    if not ok.any():
        return empty
    uniq, inv = np.unique(idx[ok], return_inverse=True)
    q = Rotation.from_matrix(np.asarray(rotations, np.float64)[uniq]).as_quat()
    Q = _equivalents(q, _proper_rotations(symmetry))
    cos_sep, cos_link = np.cos(np.deg2rad(sep_deg) / 2), np.cos(np.deg2rad(link_deg) / 2)
    uidx = np.full(idx.shape, -1, np.int64)
    uidx[ok] = inv
    # peaks of every tile: greedy in rank order, sep_deg apart
    per_tile = {}
    for t in np.nonzero(ok.any(axis=1))[0]:
        rk = np.nonzero(ok[t])[0]
        u = uidx[t, rk]
        near = _close(q[u], Q[u], cos_sep)
        got = []
        for i in range(len(rk)):
            if not near[i, got].any():
                got.append(i)
                if len(got) >= max_per_tile:
                    break
        per_tile[int(pixels[t])] = (u[got], z0[t, rk[got]], len(per_tile))
    # peaks of neighbouring tiles within link_deg are linked
    off = np.cumsum([0] + [len(v[0]) for v in per_tile.values()])
    start = {p: off[v[2]] for p, v in per_tile.items()}
    ei, ej = [], []
    for p, (u, _, _) in per_tile.items():
        y, x = p // Nx, p % Nx
        for dy, dx in ((0, 0), (0, 1), (1, -1), (1, 0), (1, 1)):
            y2, x2 = y + dy, x + dx
            if not (0 <= y2 < Ny and 0 <= x2 < Nx):
                continue
            p2 = int(y2) * Nx + int(x2)
            if p2 not in per_tile:
                continue
            u2 = per_tile[p2][0]
            a, b = np.nonzero(_close(q[u], Q[u2], cos_link))
            ei.append(start[p] + a)
            ej.append(start[p2] + b)
    n = int(off[-1])
    ei, ej = np.concatenate(ei), np.concatenate(ej)
    _, lab = connected_components(coo_matrix((np.ones(len(ei), bool), (ei, ej)), shape=(n, n)), directed=False)
    us = np.concatenate([v[0] for v in per_tile.values()])
    zs = np.concatenate([v[1] for v in per_tile.values()]).astype(np.float64)
    ts = np.repeat(list(per_tile.keys()), [len(v[0]) for v in per_tile.values()])
    order = np.argsort(lab, kind="stable")
    bounds = np.r_[0, np.nonzero(np.diff(lab[order]))[0] + 1, n]
    rot, score, tiles = [], [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        g = order[a:b]
        tl = np.unique(ts[g])
        if len(tl) >= min_tiles:
            rot.append(uniq[us[g[np.argmax(zs[g])]]])
            score.append(zs[g].sum())
            tiles.append(tl)
    if not rot:
        return empty
    o = np.argsort(-np.array(score), kind="stable")
    return Candidates(np.asarray(rotations, np.float64)[np.array(rot)[o]], np.array(score)[o], [tiles[i] for i in o], tuple(tile_shape))


def _tile_voxels(tile, tile_shape, tile_size, shape):
    """(iy, ix) of the image voxels of tiles (flat indices of the scoring image); tile_size voxels per tile edge."""
    ty, tx = tile // tile_shape[1], tile % tile_shape[1]
    a, b = np.meshgrid(np.arange(tile_size), np.arange(tile_size), indexing="ij")
    iy = (ty[:, None] * tile_size + a.ravel()[None]).ravel()
    ix = (tx[:, None] * tile_size + b.ravel()[None]).ravel()
    ok = (iy < shape[0]) & (ix < shape[1])
    return iy[ok], ix[ok]


DEFAULT_LEVELS = ((1.25, 6.0, 1.0), (0.4, 1.6, 0.4))


def refine_candidates(make_local_operator, candidates, e, d, row_mask, tile_size, shape, levels=DEFAULT_LEVELS, verbose=False):
    """Move every candidate to the best orientation of finer lattices around it.

    At each level (spacing, radius, sigma in degrees) the rotation vectors of a cubic lattice (spacing) within radius are
    applied to the candidate on the crystal side, and the score test z = sum E / sqrt(sum D) over the candidate's voxels
    (fitted angles) picks the best. The score cannot choose a kernel width (it grows as the kernel narrows, because it is
    normalised by A^T d, not (A^2)^T d): start the added orientations narrow and let the fit's Gauss-Newton steps set the width.

    make_local_operator(rotations, sigmas (radians), k, iy, ix) -> a LocalSinglePhaseForwardOperator (possibly wrapped) with
    these orientations on the pairs (k, iy, ix). e, d: (N_Omega, My, N_seg) NumPy arrays in that operator's data space.
    row_mask: (N_Omega,) bool, the fitted angles. tile_size: voxels per tile edge of the scoring image; shape (Ny, Nx) of the
    image. Returns new Candidates (same tiles, refined rotations).
    """
    if len(candidates) == 0:
        return candidates
    n = len(candidates)
    R = np.asarray(candidates.rotations, np.float64).copy()
    mask = np.asarray(row_mask, np.float32)[:, None, None]
    em, dm = (np.ascontiguousarray(a * mask, np.float32) for a in (e, d))
    vox = [_tile_voxels(t, candidates.tile_shape, tile_size, shape) for t in candidates.tiles]

    def scores(rots, sigs, rep):
        """z per entry of rots; rep (len(rots),): the candidate of each entry."""
        ks, iys, ixs = [], [], []
        order = np.argsort(rep, kind="stable")
        start = np.searchsorted(rep[order], np.arange(n + 1))
        for j in range(n):
            cj = order[start[j]:start[j + 1]]
            iy, ix = vox[j]
            ks.append(np.repeat(cj, len(iy)))
            iys.append(np.tile(iy, len(cj)))
            ixs.append(np.tile(ix, len(cj)))
        k, iy, ix = (np.concatenate(a) for a in (ks, iys, ixs))
        op = make_local_operator(rots, sigs, k, iy, ix)
        q = op.queue

        def adj(arr):
            a = clarray.to_device(q, arr)
            g = clarray.zeros(q, op.coeff_shape, np.float32)
            op.adjoint_cl(a, g)
            kk, _, _, v = op.to_coo(g)
            a.base_data.release()
            g.base_data.release()
            return np.bincount(kk, weights=v.astype(np.float64), minlength=len(rots))
        E, D = adj(em), adj(dm)
        del op
        return E / np.sqrt(np.maximum(D, 1e-30))

    def best_per(rep, z):
        order = np.lexsort((-z, rep))
        return order[np.searchsorted(rep[order], np.arange(n))]

    for spacing, radius, sig in levels:
        m = int(np.ceil(radius / spacing))
        gv = np.stack(np.meshgrid(*[np.arange(-m, m + 1) * spacing] * 3, indexing="ij"), -1).reshape(-1, 3)
        gv = gv[np.linalg.norm(gv, axis=1) <= radius + 1e-9]
        dR = Rotation.from_rotvec(np.deg2rad(gv)).as_matrix()
        rots = (R[:, None] @ dR[None]).reshape(-1, 3, 3)
        rep = np.repeat(np.arange(n), len(gv))
        z = scores(rots, np.full(len(rots), np.deg2rad(sig)), rep)
        R = rots[best_per(rep, z)]
        if verbose:
            print(f"  level {spacing} deg (sigma {sig}): {len(rots)} orientations", flush=True)
    return Candidates(R, candidates.score, candidates.tiles, candidates.tile_shape)


def candidate_support(candidates, tile_size, shape, dilate=0, mask=None):
    """(k, iy, ix) of the voxels of every candidate (k = its index): its tiles, grown by ``dilate`` voxels and cut to the
    (Ny, Nx) bool ``mask`` if given, as a LocalSupport needs them."""
    from scipy import ndimage as ndi
    ks, iys, ixs = [], [], []
    for j, t in enumerate(candidates.tiles):
        iy, ix = _tile_voxels(t, candidates.tile_shape, tile_size, shape)
        im = np.zeros(shape, bool)
        im[iy, ix] = True
        if dilate:
            im = ndi.binary_dilation(im, iterations=int(dilate))
        if mask is not None:
            im &= mask
        y, x = np.nonzero(im)
        ks.append(np.full(len(y), j, np.int64))
        iys.append(y)
        ixs.append(x)
    return tuple(np.concatenate(a) if a else np.zeros(0, np.int64) for a in (ks, iys, ixs))


def orientation_weights(coefficients, k=None):
    """Total coefficient of every orientation: from a dense (K, Ny, Nx) array, or, with k, from the values and orientation
    indices of a local basis (to_coo)."""
    c = np.asarray(coefficients, np.float64)
    if k is not None:
        return np.bincount(np.asarray(k), weights=np.maximum(c, 0))
    return np.maximum(c, 0).reshape(c.shape[0], -1).sum(axis=1)


def prune_added(weights, n_old, rel=0.1):
    """Bool (len(weights),): keep the first n_old orientations and the added ones with a total weight of at least ``rel`` x
    the median of the added ones that received any: after the joint fit, orientations it left (almost) empty go."""
    w = np.asarray(weights, np.float64)
    keep = np.ones(len(w), bool)
    add = w[n_old:]
    if (add > 0).any():
        keep[n_old:] = add >= rel * np.median(add[add > 0])
    else:
        keep[n_old:] = False
    return keep
