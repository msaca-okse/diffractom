"""
FISTA with the coefficient arrays in host memory, streamed through the GPU batch by batch.

With the fused update (fused_update.py), a FISTA iteration touches the coefficients only in
two passes over the operator's orientation batches: the forward projection A(y) reads y, and
the adjoint pass reads and writes x and y. Both work on one batch at a time, so x and y can
stay in host memory: the GPU holds the data, the prediction/residual and batch-sized staging
buffers, independent of the number of orientations K.

Transfers: x and y are ordinary (pageable) host arrays. A batch goes through a small page-locked
(pinned) staging buffer, filled or emptied by a multi-threaded CPU copy, and is moved by DMA on
its own command queue (one for uploads, one for downloads, so both directions run at once). The
next batch is uploaded and the previous one downloaded while the current batch is computed.
(Large page-locked buffers are not an option: NVIDIA's OpenCL backs ALLOC_HOST_PTR buffers with
device memory of the same size, and migrates USE_HOST_PTR buffers to the device on first use.)
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

from .fused_update import PROX_CODES, PartialSums
from .launch import elementwise


class _Pinned:
    """A page-locked host buffer of n float32 (mapped as a NumPy array)."""

    def __init__(self, ctx, queue, n):
        self.buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, n * 4)
        self.arr, _ = cl.enqueue_map_buffer(queue, self.buf, cl.map_flags.READ | cl.map_flags.WRITE, 0, (n,),
                                            np.float32)
        self.dma = None  # last DMA event involving this buffer

    def release(self):
        self.arr = None
        self.buf.release()


class Streamer:
    """Batch-wise transfers of host coefficient arrays (K, Ny, Nx), C order, and the two
    coefficient passes of a fused FISTA iteration."""

    def __init__(self, op, fused_update=None, n_threads=None):
        self.op = op
        self.q = op.queue
        self.tq_up = cl.CommandQueue(op.ctx, op.queue.device)
        self.tq_dn = cl.CommandQueue(op.ctx, op.queue.device)
        self.fu = fused_update
        self.npix = op.Nx * op.Ny
        self.shape = (op.K, op.Ny, op.Nx)
        n = self.npix * op.K_batch_max
        dev = lambda: clarray.empty(self.q, (n,), np.float32)
        self.xs, self.ys = [dev(), dev()], [dev(), dev()]  # device staging slots
        pin = lambda: _Pinned(op.ctx, self.q, n)
        self.pin_up = {"x": [pin(), pin()], "y": [pin(), pin()]}
        self.pin_dn = {"x": [pin(), pin()], "y": [pin(), pin()]}
        self.n_threads = n_threads or min(8, os.cpu_count() or 1)
        self.pool = ThreadPoolExecutor(self.n_threads)  # CPU copies
        self.io_up = ThreadPoolExecutor(1)  # transfers of the adjoint pass, in order
        self.io_dn = ThreadPoolExecutor(1)
        self.k_batch_max = op.K_batch_max
        self.released = False
        self.fu, self.sums = None, []
        self.use_fused_update(fused_update)

    def use_fused_update(self, fused_update):
        """Set the fused update whose kernels compute the norms of the adjoint pass."""
        if fused_update is self.fu:
            return
        for p in self.sums:
            p.release()
        self.fu, self.sums = fused_update, []
        if fused_update is not None:  # norms of the adjoint pass, without blocking the host
            nb = len(self.op.batches)
            self.sums = [PartialSums(fused_update, nb) for _ in range(3)]

    # ---------------------------------------------------------------- helpers
    def flat(self, x):
        x = np.asarray(x)
        if x.shape != self.shape or x.dtype != np.float32 or not x.flags.c_contiguous:
            raise ValueError(f"expected a C-contiguous float32 array of shape {self.shape}")
        return x.reshape(-1)  # a view

    def view(self, flat, ib):
        b = self.op.batches[ib]
        return flat[self.npix * b["k_start"]:self.npix * (b["k_start"] + b["K_batch"])]

    def _pcopy(self, dst, src):
        """dst[:] = src with the thread pool (NumPy releases the GIL for large copies)."""
        parts = max(1, min(self.pool._max_workers, src.size // (1 << 20)))
        edges = np.linspace(0, src.size, parts + 1).astype(np.int64)
        list(self.pool.map(lambda i: np.copyto(dst[edges[i]:edges[i + 1]], src[edges[i]:edges[i + 1]]), range(parts)))

    def _upload(self, host_view, pin, stage, wait=None):
        """host -> pinned (CPU copy) -> device staging (DMA on the upload queue)."""
        if pin.dma is not None:
            pin.dma.wait()  # the previous upload from this pinned buffer has finished
        n = host_view.size
        self._pcopy(pin.arr[:n], host_view)
        pin.dma = cl.enqueue_copy(self.tq_up, stage.data, pin.arr[:n], is_blocking=False, wait_for=wait)
        return pin.dma

    # ---------------------------------------------------------------- passes
    def forward(self, src_flat, data):
        """data = A(src) for a host coefficient array (flattened (K, Ny, Nx))."""
        op, q = self.op, self.q
        nb = len(op.batches)
        data.fill(0.0)
        up, done = [None] * nb, [None, None]
        up[0] = self._upload(self.view(src_flat, 0), self.pin_up["y"][0], self.ys[0],
                             [cl.enqueue_marker(q)])  # after all earlier work on the staging slots
        for ib in range(nb):
            s = ib % 2
            cl.enqueue_barrier(q, wait_for=[up[ib]])
            op._direct_batch(self.ys[s], 0, op.K_batch_max, ib, data)
            done[s] = cl.enqueue_marker(q)
            if ib + 1 < nb:  # while batch ib is projected; slot 1-s is free once batch ib-1 is done
                t = 1 - s
                up[ib + 1] = self._upload(self.view(src_flat, ib + 1), self.pin_up["y"][t], self.ys[t],
                                          [done[t]] if done[t] is not None else None)

    def adjoint_update(self, x_flat, y_flat, r, g_batch, tau, beta, prox_kind, lam, support_gpu, y_src=None,
                       forward_into=None):
        """grad = A^T r batch by batch; x <- prox(y - tau*grad), y <- x + beta*(x - x_old), with x
        and y streamed from and to the host arrays (y read from y_src if given, e.g. x in the first
        iteration, when y = x). Returns ||grad||^2, ||x||^2 and sum|x| (the latter only for L1
        proxes).

        forward_into (a data-sized array): also forward_into = A(y) of the new y, every batch
        projected while it is on the device (the forward pass of the next iteration, without
        uploading y again; the same sum over the batches, in the same order, as forward()).

        The transfers run in two worker threads (uploads, downloads): NVIDIA's OpenCL blocks the
        calling thread in a device-to-host copy until it has run, so the thread that enqueues
        the computation must not issue them, or computation and transfers would alternate."""
        op, q, fu = self.op, self.q, self.fu
        nb = len(op.batches)
        mask = support_gpu if support_gpu is not None else fu._dummy_mask
        use_mask = np.int32(support_gpu is not None)
        code = np.int32(PROX_CODES[prox_kind])
        want_l1 = prox_kind in ("l1", "nonneg_l1") and lam != 0.0
        y_in = y_flat if y_src is None else y_src
        up, dn = [None] * nb, [None] * nb
        gsq, xsq, xabs = (p.reset() for p in self.sums)
        if forward_into is not None:
            forward_into.fill(0.0)
        start = cl.enqueue_marker(q)  # the forward pass, which used the staging slots, is done
        q.flush()  # (other threads wait for markers: they must have been submitted)

        def upload(ib):  # worker: host -> pinned -> device slot ib % 2
            s = ib % 2
            if ib >= 2:
                dn[ib - 2].result()  # slot s has been downloaded
            else:
                start.wait()
            for name, flat, stage in (("x", x_flat, self.xs[s]), ("y", y_in, self.ys[s])):
                pin = self.pin_up[name][s]
                n = self.npix * op.batches[ib]["K_batch"]
                self._pcopy(pin.arr[:n], self.view(flat, ib))
                cl.enqueue_copy(self.tq_up, stage.data, pin.arr[:n], is_blocking=True)

        def download(ib, ev):  # worker: device slot ib % 2 -> pinned -> host
            s = ib % 2
            ev.wait()
            for name, flat, stage in (("x", x_flat, self.xs[s]), ("y", y_flat, self.ys[s])):
                pin = self.pin_dn[name][s]
                n = self.npix * op.batches[ib]["K_batch"]
                cl.enqueue_copy(self.tq_dn, pin.arr[:n], stage.data, is_blocking=True)
                self._pcopy(self.view(flat, ib), pin.arr[:n])

        up[0] = self.io_up.submit(upload, 0)
        for ib, b in enumerate(op.batches):
            s, n = ib % 2, self.npix * b["K_batch"]
            op._adjoint_batch(r, ib, g_batch, 0, op.K_batch_max)
            gsq.add(g_batch.data, n, 0)
            if ib + 1 < nb:
                up[ib + 1] = self.io_up.submit(upload, ib + 1)
            up[ib].result()  # x and y of this batch are on the device
            gws, n64 = elementwise(n)
            fu.k_update(q, gws, None, g_batch.data, self.xs[s].data, self.ys[s].data, mask.data, use_mask,
                        np.uint64(0), np.uint64(self.npix), n64, np.float32(tau), np.float32(beta),
                        np.float32(lam * tau), code)
            xsq.add(self.xs[s].data, n, 0)
            if want_l1:
                xabs.add(self.xs[s].data, n, 1)
            if forward_into is not None:  # (before the marker: slot s is reused after its download)
                op._direct_batch(self.ys[s], 0, op.K_batch_max, ib, forward_into)
            ev = cl.enqueue_marker(q)
            q.flush()
            dn[ib] = self.io_dn.submit(download, ib, ev)
        for f in dn:
            f.result()
        return gsq.total(), xsq.total(), (xabs.total() if want_l1 else 0.0)

    def adjoint_to_host(self, r, z_flat):
        """z = A^T r, batch by batch into a host array (flattened (K, Ny, Nx)); the downloads
        run in the download thread (see adjoint_update)."""
        op, q = self.op, self.q
        nb = len(op.batches)
        dn = [None] * nb

        def download(ib, ev):
            s, n = ib % 2, self.npix * op.batches[ib]["K_batch"]
            ev.wait()
            pin = self.pin_dn["x"][s]
            cl.enqueue_copy(self.tq_dn, pin.arr[:n], self.xs[s].data, is_blocking=True)
            self._pcopy(self.view(z_flat, ib), pin.arr[:n])

        for ib in range(nb):
            s = ib % 2
            if ib >= 2:
                dn[ib - 2].result()  # slot s has been downloaded
            op._adjoint_batch(r, ib, self.xs[s], 0, op.K_batch_max)
            ev = cl.enqueue_marker(q)
            q.flush()
            dn[ib] = self.io_dn.submit(download, ib, ev)
        for f in dn:
            f.result()

    def finish(self):
        self.q.finish()
        self.tq_up.finish()
        self.tq_dn.finish()

    def release(self):
        if self.released:
            return
        self.released = True
        self.finish()
        self.pool.shutdown()
        self.io_up.shutdown()
        self.io_dn.shutdown()
        for p in self.pin_up["x"] + self.pin_up["y"] + self.pin_dn["x"] + self.pin_dn["y"]:
            p.release()
        for a in self.xs + self.ys:
            a.base_data.release()
        for p in self.sums:
            p.release()
        self.sums = []


    def nbytes(self):
        """GPU memory of the staging slots and of the pinned host buffers (NVIDIA's OpenCL backs an
        ALLOC_HOST_PTR buffer with device memory of the same size)."""
        pins = self.pin_up["x"] + self.pin_up["y"] + self.pin_dn["x"] + self.pin_dn["y"]
        return sum(a.nbytes for a in self.xs + self.ys) + sum(p.buf.size for p in pins)


def fits_next_forward(op, streamer, extra_bytes, margin_gb=1.5):
    """Whether one more data-sized array fits on the GPU (for adjoint_update's forward_into), given the
    operator's buffers, the streamer's and extra_bytes of the solver's (data, prediction, gradient
    batch, ...): an estimate, with a margin."""
    data_bytes = 4 * int(np.prod(op.data_shape))
    used = op.device_bytes() + streamer.nbytes() + extra_bytes
    return used + data_bytes + margin_gb * 1024**3 <= op.queue.device.global_mem_size


def streamer_for(op, fused_update=None, n_threads=None):
    """The operator's Streamer: created on first use and kept (released by op.release_streamer()
    or op.free_memory()), so that its pinned host buffers and device staging slots are allocated
    once, not in every call (37 s per call at 1200 x 1200 pixels, K = 20000). The staging slots
    hold 4 batches of images on the GPU meanwhile."""
    st = getattr(op, "_streamer", None)
    if st is not None and (st.released or st.k_batch_max != op.K_batch_max
                           or (n_threads is not None and n_threads != st.n_threads)):
        st.release()
        st = None
    if st is None:
        st = Streamer(op, None, n_threads)
        op._streamer = st
    st.use_fused_update(fused_update)
    return st


def _chunks(n, chunk):
    return [(i, min(i + chunk, n)) for i in range(0, n, chunk)]


def _dot(a, b, chunk=1 << 24, pool=None):
    """float64 dot product of two float32 arrays, in chunks (no full-size float64 temporary). The
    chunks' dot products are summed in order, with or without a thread pool."""
    f = lambda c: np.dot(a[c[0]:c[1]].astype(np.float64), b[c[0]:c[1]])
    cs = _chunks(a.size, chunk)
    return float(sum(pool.map(f, cs) if pool is not None else map(f, cs)))


def _scale(x, factor, pool, chunk=1 << 24):
    """x *= factor (float32), in chunks on the thread pool."""
    def f(c):
        x[c[0]:c[1]] *= factor
    list(pool.map(f, _chunks(x.size, chunk)))


def _random_start(x3, rng, pool, seed, parallel):
    """The random start of estimate_L_power, drawn as (Nx, Ny, K) in C order, into x3 (K, Ny, Nx).

    parallel=False: the same numbers as estimate_L_power (a single random stream, drawn one x
    column at a time on this thread; converting and writing them, the slower part, runs on the
    pool, a block of columns at a time). parallel=True: column ix from its own stream
    (default_rng([seed, ix])), all on the pool: much faster, but different numbers, so a slightly
    different estimate of L."""
    K, Ny, Nx = x3.shape
    if parallel:
        def col(ix):
            x3[:, :, ix] = np.random.default_rng([seed, ix]).standard_normal((Ny, K)).astype(np.float32).T
        list(pool.map(col, range(Nx)))
        return
    B = max(1, min(Nx, (256 << 20) // (4 * Ny * K)))  # columns per block (~256 MB in float32)
    kc = _chunks(K, max(1, -(-K // (4 * pool._max_workers))))

    def write(ix0, blk):  # x3[:, :, ix0 + b] = blk[b].T, K-ranges in parallel
        def part(c):
            x3[c[0]:c[1], :, ix0:ix0 + len(blk)] = blk[:, :, c[0]:c[1]].transpose(2, 1, 0)
        list(pool.map(part, kc))

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(1) as writer:  # writes block b while block b + 1 is drawn
        pending = None
        for ix0 in range(0, Nx, B):
            blk = np.empty((min(B, Nx - ix0), Ny, K), np.float32)
            for b in range(len(blk)):
                blk[b] = rng.standard_normal((Ny, K))   # float64 -> float32, as .astype(np.float32)
            if pending is not None:
                pending.result()
            pending = writer.submit(write, ix0, blk)
        pending.result()


def estimate_L_power_streamed(op, niter=20, seed=0, eps=1e-30, verbose=1, parallel_start=True):
    """estimate_L_power with the two coefficient-sized vectors in host memory, streamed batch by
    batch (for problems whose coefficient arrays do not fit on the GPU).

    The host work (the random start, dot products, scaling) runs on the operator's thread pool;
    the result is the same as with one thread. By default (parallel_start=True) the random start is
    drawn in parallel, a column from its own stream (default_rng([seed, ix])): a few seconds.
    parallel_start=False draws the same numbers as estimate_L_power (the GPU version), from one
    stream, which cannot be parallelised: about 7 minutes for 1200 x 1200 pixels and K = 20000.
    The two starts give slightly different estimates (and so slightly different FISTA steps)."""
    st = streamer_for(op)
    pool = st.pool
    rng = np.random.default_rng(seed)
    x = np.empty(op.Nx * op.Ny * op.K, np.float32)
    z = np.empty_like(x)
    _random_start(x.reshape(op.K, op.Ny, op.Nx), rng, pool, seed, parallel_start)
    _scale(x, np.float32(1.0 / (np.sqrt(_dot(x, x, pool=pool)) + eps)), pool)
    Ax = clarray.empty(op.queue, (op.N_Omega, op.My, op.N_seg), np.float32, order="C")
    L_est = 0.0
    for it in range(niter):
        st.forward(x, Ax)
        st.adjoint_to_host(Ax, z)
        st.finish()
        L_est = _dot(x, z, pool=pool) / (_dot(x, x, pool=pool) + eps)
        znorm = float(np.sqrt(_dot(z, z, pool=pool))) + eps
        _scale(z, np.float32(1.0 / znorm), pool)
        x, z = z, x
        if verbose:
            print(f"[power {it+1:02d}] L_est={L_est:.6e}  ||z||={znorm:.6e}")
    Ax.base_data.release()
    return L_est
