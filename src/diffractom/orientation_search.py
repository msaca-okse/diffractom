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
once for all inputs (``SinglePhaseForwardOperator.adjoint_multi_cl``); the result equals separate adjoints of the
row-masked inputs bitwise.
"""
from __future__ import annotations

import time

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

from .utils.arrays import check_device

__all__ = ["poisson_score_inputs", "scramble_eta", "score_candidates"]

_GATHER = r"""
__kernel void gather_pixels(__global const float *img, __global const int *pix, __global float *out,
                            const int npix, const int Kb, const int Kstride)
{   // out[k, s] = img[pix[s], k] for k < Kb  (img: (Npix, Kstride) pixel-major, out: (Kb, npix))
    const int s = get_global_id(0), k = get_global_id(1);
    if (s >= npix || k >= Kb) return;
    out[(long)k * npix + s] = img[(long)pix[s] * Kstride + k];
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
    No = a.shape[0]
    shifts = np.random.default_rng(seed).integers(margin, n_eta - margin, size=(No // block + 1, n_channels))
    out = np.empty_like(a)
    for b_ in range(0, No, block):
        s = shifts[b_ // block]
        perm = np.empty(n_eta * n_channels, np.int64)
        for r in range(n_channels):
            perm[np.arange(n_eta) * n_channels + r] = ((np.arange(n_eta) + s[r]) % n_eta) * n_channels + r
        out[b_:b_ + block] = a[b_:b_ + block][:, :, perm]
    return out if mask is None else out * mask


def score_candidates(make_operator, rotations, inputs, row_masks, pixels, top_k=64, chunk=30000, verbose=False):
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
    top_k, chunk : candidates kept per pixel; candidates per operator.

    Returns
    -------
    list (one per input pair) of dicts: "idx" (npix, top_k) int32 candidate indices (-1: none), "z" (n_rows, npix,
    top_k) float32 (z[0] ranks, descending).
    """
    rotations = np.asarray(rotations, np.float64)
    K_all = len(rotations)
    pixels = np.asarray(pixels, np.int32)
    npix, nr, ni = len(pixels), len(row_masks), len(inputs)
    top = [(np.full((npix, top_k), -1, np.int32), [np.full((npix, top_k), -np.inf, np.float32)] +
            [np.zeros((npix, top_k), np.float32) for _ in range(nr - 1)]) for _ in range(ni)]
    t0 = time.perf_counter()
    prg = None
    for c0 in range(0, K_all, chunk):
        Rc = rotations[c0:c0 + chunk]
        K = len(Rc)
        op = make_operator(Rc)
        q = op.queue
        if prg is None or prg.context != op.ctx:
            prg = cl.Program(op.ctx, _GATHER).build()
            k_gather = cl.Kernel(prg, "gather_pixels")
            pix_d = clarray.to_device(q, pixels)
        datas = [a for pair in inputs for a in pair]
        for a in datas:
            check_device(a, op.data_shape, "input")
        # host blocks (K, npix) per (data array, row mask), as one adjoint per masked input would give
        host = [[np.empty((K, npix), np.float32) for _ in range(nr)] for _ in datas]
        buf = clarray.empty(q, (op.K_batch_max * npix,), np.float32)

        def consume(i, s, k0, Kb, img_k):
            k_gather(q, (-(-npix // 64) * 64, Kb), (64, 1), img_k.data, pix_d.data, buf.data,
                              np.int32(npix), np.int32(Kb), np.int32(op.K_batch_max))
            cl.enqueue_copy(q, host[i][s][k0:k0 + Kb], buf.data, is_blocking=True)

        op.adjoint_multi_cl(datas, consume, row_masks=row_masks)
        buf.base_data.release()
        for j in range(ni):
            E_, D_ = host[2 * j], host[2 * j + 1]
            z = []
            for s in range(nr):
                zz = E_[s] / np.sqrt(np.maximum(D_[s], 1e-30))
                zz[D_[s] <= 0] = 0
                z.append(zz)
            i0, z0 = top[j]
            cand_i = np.concatenate([i0, np.tile((np.arange(K) + c0)[:, None].T, (npix, 1))], axis=1)
            cand_z = [np.concatenate([z0[s], z[s].T], axis=1) for s in range(nr)]
            o_ = np.argpartition(-cand_z[0], top_k - 1, axis=1)[:, :top_k]
            top[j] = (np.take_along_axis(cand_i, o_, 1).astype(np.int32),
                      [np.take_along_axis(cand_z[s], o_, 1).astype(np.float32) for s in range(nr)])
        del op, host
        if verbose:
            print(f"  candidates {c0}..{c0 + K} scored ({time.perf_counter() - t0:.0f} s)", flush=True)
    out = []
    for i_, z_ in top:
        o_ = np.argsort(-z_[0], axis=1)
        out.append({"idx": np.take_along_axis(i_, o_, 1), "z": np.stack([np.take_along_axis(a, o_, 1) for a in z_])})
    return out
