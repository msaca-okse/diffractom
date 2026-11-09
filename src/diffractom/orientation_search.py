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

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

from .utils.arrays import check_device

__all__ = ["poisson_score_inputs", "scramble_eta", "score_candidates"]

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
