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
        self.pool = ThreadPoolExecutor(n_threads or min(8, os.cpu_count() or 1))  # CPU copies
        self.io_up = ThreadPoolExecutor(1)  # transfers of the adjoint pass, in order
        if fused_update is not None:  # norms of the adjoint pass, without blocking the host
            nb = len(op.batches)
            self.sums = [PartialSums(fused_update, nb) for _ in range(3)]
        self.io_dn = ThreadPoolExecutor(1)

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

    def adjoint_update(self, x_flat, y_flat, r, g_batch, tau, beta, prox_kind, lam, support_gpu):
        """grad = A^T r batch by batch; x <- prox(y - tau*grad), y <- x + beta*(x - x_old), with x
        and y streamed from and to the host arrays. Returns ||grad||^2, ||x||^2 and sum|x| (the
        latter only for L1 proxes).

        The transfers run in two worker threads (uploads, downloads): NVIDIA's OpenCL blocks the
        calling thread in a device-to-host copy until it has run, so the thread that enqueues
        the computation must not issue them, or computation and transfers would alternate."""
        op, q, fu = self.op, self.q, self.fu
        nb = len(op.batches)
        mask = support_gpu if support_gpu is not None else fu._dummy_mask
        use_mask = np.int32(support_gpu is not None)
        code = np.int32(PROX_CODES[prox_kind])
        want_l1 = prox_kind in ("l1", "nonneg_l1") and lam != 0.0
        up, dn = [None] * nb, [None] * nb
        gsq, xsq, xabs = (p.reset() for p in self.sums)
        start = cl.enqueue_marker(q)  # the forward pass, which used the staging slots, is done
        q.flush()  # (other threads wait for markers: they must have been submitted)

        def upload(ib):  # worker: host -> pinned -> device slot ib % 2
            s = ib % 2
            if ib >= 2:
                dn[ib - 2].result()  # slot s has been downloaded
            else:
                start.wait()
            for name, flat, stage in (("x", x_flat, self.xs[s]), ("y", y_flat, self.ys[s])):
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
        self.finish()
        self.pool.shutdown()
        self.io_up.shutdown()
        self.io_dn.shutdown()
        for p in self.pin_up["x"] + self.pin_up["y"] + self.pin_dn["x"] + self.pin_dn["y"]:
            p.release()
        for a in self.xs + self.ys:
            a.base_data.release()
        for p in getattr(self, "sums", []):
            p.release()


def _dot(a, b, chunk=1 << 24):
    """float64 dot product of two float32 arrays, in chunks (no full-size float64 temporary)."""
    return float(sum(np.dot(a[i:i + chunk].astype(np.float64), b[i:i + chunk]) for i in range(0, a.size, chunk)))


def estimate_L_power_streamed(op, niter=20, seed=0, eps=1e-30, verbose=1):
    """estimate_L_power with the two coefficient-sized vectors in host memory, streamed batch by
    batch (for problems whose coefficient arrays do not fit on the GPU)."""
    st = Streamer(op)
    rng = np.random.default_rng(seed)
    x = np.empty(op.Nx * op.Ny * op.K, np.float32)
    z = np.empty_like(x)
    for i in range(0, x.size, 1 << 24):
        x[i:i + (1 << 24)] = rng.standard_normal(min(1 << 24, x.size - i)).astype(np.float32)
    x *= np.float32(1.0 / (np.sqrt(_dot(x, x)) + eps))
    Ax = clarray.empty(op.queue, (op.N_Omega, op.My, op.N_seg), np.float32, order="C")
    L_est = 0.0
    for it in range(niter):
        st.forward(x, Ax)
        st.adjoint_to_host(Ax, z)
        st.finish()
        L_est = _dot(x, z) / (_dot(x, x) + eps)
        znorm = float(np.sqrt(_dot(z, z))) + eps
        z *= np.float32(1.0 / znorm)
        x, z = z, x
        if verbose:
            print(f"[power {it+1:02d}] L_est={L_est:.6e}  ||z||={znorm:.6e}")
    Ax.base_data.release()
    st.release()
    return L_est
