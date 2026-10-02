"""Parallel-beam Radon transform on the GPU, vectorised over many channels (orientations)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray


def _round_up(n: int, m: int) -> int:
    return (n + m - 1) // m * m


class ParallelRadon:
    """
    Parallel-beam Radon transform of a stack of images that share one geometry.

    Built for the texture-tomography operators, where every orientation of the
    basis is a channel. The channel axis is the fastest axis of the images and
    sinograms, and each work item handles 4 channels, so the ray geometry is
    shared and all memory accesses are contiguous.

    Discretisation (pixel-centred image, detector bins of width delta_s):
    pixel (x, y) projects at angle theta_a to detector coordinate
    ``t = X_a x + Y_a y + T0_a`` (in bins) and contributes to bin s with the hat
    weight ``max(0, 1 - |t - s|)``. The forward projection is scaled by
    ``delta_x**2 / delta_s``; the backward projection is the transpose weighted by
    the angle weights, i.e. the adjoint up to ``w_a delta_s / delta_x**2``.
    This is the discretisation of gratopy's parallel-beam transform
    (K. Bredies and R. Huber), with the same parameters.

    Channel-fastest arrays ("_k" below) are flat float32 arrays: images
    (Nx*Ny, Kstride) with pixel index x + Nx*y, sinograms (R, n_detectors, Kstride),
    both C-order. Kstride must be a multiple of 4.

    Parameters
    ----------
    queue : pyopencl.CommandQueue
    img_shape : (Nx, Ny)
    angles : (R,) array, radians
    n_detectors : int
    image_width, detector_width : float
        Physical widths of the image and the detector line.
    detector_shift : float
        Physical shift of the detector along the detector line (centre-of-rotation offset).
    angle_weights : (R,) array or float, optional
        Weights of the backward projection, e.g. the angular width of each projection.
        Default: pi / R, i.e. R evenly spaced angles covering 180 degrees.
    bins_per_item : int
        Detector bins computed per work item in the forward projection.
    tiled : bool
        Use the tiled kernels (default): they stage what a work-group shares in local memory (the
        image rows of a block of angles and bins in the forward projection, the sinogram bins of a
        pixel tile in the backward one) instead of reading it from the caches about twice per
        angle. Bitwise the same results as the plain kernels (tiled=False).
    channels_per_launch : int, optional
        The projections run in slices of this many channels (a multiple of 4). Default: all at
        once with the tiled kernels; with the plain kernels 64 for images of 160 x 160 pixels or
        more, 128 otherwise (measured on an A40, 360 angles: for 400 x 400 pixels one launch over
        256-1024 channels took 2.6x as long as slices of 64; for 99 x 99 up to 3x as long as
        slices of 128).
    """

    def __init__(self, queue, img_shape, angles, n_detectors, image_width, detector_width,
                 detector_shift=0.0, angle_weights=None, bins_per_item=3, channels_per_launch=None,
                 tiled=True):
        self.queue = queue
        self.Nx, self.Ny = (int(n) for n in img_shape)
        self.Ns = int(n_detectors)
        angles = np.asarray(angles, dtype=np.float64)
        self.R = len(angles)
        self.angles = angles

        if angle_weights is None:
            angle_weights = np.pi / self.R
        self.angle_weights = np.broadcast_to(np.asarray(angle_weights, dtype=np.float64), (self.R,)).copy()

        n_max = max(self.Nx, self.Ny)
        delta_x = image_width / float(n_max)
        delta_s = float(detector_width) / self.Ns
        rho = delta_s / delta_x                         # detector bin width in pixels
        X = np.cos(angles - 0.5 * np.pi) / rho
        Y = np.sin(angles - 0.5 * np.pi) / rho
        T0 = (self.Ns - 1) / 2.0 - X * (self.Nx - 1) / 2.0 - Y * (self.Ny - 1) / 2.0 - detector_shift / delta_s
        self.scale = np.float32(delta_x * delta_x / delta_s)

        geo = np.stack([X, Y, T0, self.angle_weights], axis=1).astype(np.float32)
        self.geo_gpu = clarray.to_device(queue, np.ascontiguousarray(geo))

        self.bins_per_item = int(bins_per_item)
        self.tiled = bool(tiled)
        if channels_per_launch is None:
            if self.tiled:
                channels_per_launch = 1 << 30
            else:
                channels_per_launch = 64 if self.Nx * self.Ny >= 160 * 160 else 128
        if channels_per_launch % 4 or channels_per_launch <= 0:
            raise ValueError("channels_per_launch must be a positive multiple of 4")
        self.channels_per_launch = int(channels_per_launch)
        src = Path(__file__).with_name("radon_kernels.cl").read_text()
        options = [f"-DNB={self.bins_per_item}"]
        if self.tiled:
            if self.FCQ * self.FSB * self.FAB > queue.device.max_work_group_size:
                self.FAB = max(1, queue.device.max_work_group_size // (self.FCQ * self.FSB))
            options += self._tiling(geo, queue.device)
        self.prg = cl.Program(queue.context, src).build(options=options)
        self._fwd = self.prg.radon_forward_k
        self._bwd = self.prg.radon_backward_k
        if self.tiled:
            self._fwd_t = self.prg.radon_forward_tiled
            self._bwd_t = self.prg.radon_backward_tiled
        self._gather = self.prg.gather_channels_k_fastest
        self._scatter = self.prg.scatter_channels_k_fastest

    # work-group shapes of the tiled kernels: forward (channel quads, strips, angles), backward
    # (channel quads, 8 x 8 pixels). Forward measured on an A40 (360 angles, us per channel at
    # 120 / 400 / 800 pixels): (4, 8, 16) 10.4 / 133 / 639, (8, 8, 8) 10.7 / 141 / 645,
    # (4, 8, 8) 10.9 / 164 / 787, (2, 16, 8) 14.6 / 241 / 922; plain kernel 14.2 / 228 / 2156.
    FCQ, FSB, FAB = 4, 8, 16
    BCQ, BAB, BT = 4, 16, 8

    def _tiling(self, geo, device):
        """Compile-time constants of the tiled kernels, from the geometry: the angle blocks of the
        forward projection (consecutive angles with the same row axis), the width W of the image
        segment a forward work-group needs per row (computed for every block, bin group and row,
        plus a margin; a pixel outside is read from global memory, so this only affects speed),
        the rows per local tile, and the sinogram bins of a backward pixel tile."""
        X, Y, T0 = (geo[:, i].astype(np.float32) for i in range(3))
        rows_are_y = np.abs(X) >= np.abs(Y)
        blocks = []
        a = 0
        while a < self.R:
            b = a
            while b < self.R and b - a < self.FAB and rows_are_y[b] == rows_are_y[a]:
                b += 1
            blocks.append((a, b - a))
            a = b
        self.blocks_gpu = clarray.to_device(self.queue, np.asarray(blocks, dtype=np.int32))
        self.n_blocks = len(blocks)

        NB = self.bins_per_item
        W = 1
        S0 = np.arange(0, self.Ns, self.FSB * NB, dtype=np.float64)[:, None]   # first bin of each group
        S1 = S0 + self.FSB * NB
        for a0, na in blocks:
            ry = rows_are_y[a0]
            A = np.where(ry, X, Y)[a0:a0 + na].astype(np.float64)
            B = np.where(ry, Y, X)[a0:a0 + na].astype(np.float64)
            Nu, Nv = (self.Nx, self.Ny) if ry else (self.Ny, self.Nx)
            v = np.arange(Nv, dtype=np.float64)
            c = B[:, None] * v[None, :] + T0[a0:a0 + na, None]                  # (angles, rows)
            e1 = (S0[:, :, None, None] - 1 - c[None]) / A[None, :, None]         # (groups, 1, angles, rows)
            e2 = (S1[:, :, None, None] - c[None]) / A[None, :, None]
            lo = np.clip(np.floor(np.minimum(e1, e2)), 0, Nu - 1).min(axis=(1, 2))   # (groups, rows)
            hi = np.clip(np.ceil(np.maximum(e1, e2)), 0, Nu - 1).max(axis=(1, 2))
            W = max(W, int((hi - lo).max()) + 4)
        budget = min(40 * 1024, device.local_mem_size - 2048)
        per_row = W * self.FCQ * 16
        self.FVR = max(1, min(8, budget // per_row))
        if self.FVR * per_row > budget:                  # very wide segments: fewer pixels in local memory
            W = budget // (self.FCQ * 16)
        self.W = int(W)
        span = np.max((self.BT - 1) * (np.abs(X.astype(np.float64)) + np.abs(Y.astype(np.float64))))
        self.BLMAX = int(np.ceil(span)) + 4
        per_angle = self.BLMAX * self.BCQ * 16
        self.BAB = max(1, min(self.BAB, budget // per_angle))   # fewer angles per tile for narrow bins
        if self.BAB * per_angle > budget:                      # (bins beyond BLMAX: from global memory)
            self.BLMAX = budget // (self.BCQ * 16)
        return [f"-DFCQ={self.FCQ}", f"-DFSB={self.FSB}", f"-DFAB={self.FAB}", f"-DFVR={self.FVR}",
                f"-DBCQ={self.BCQ}", f"-DBAB={self.BAB}", f"-DBLMAX={self.BLMAX}"]

    def gather(self, coeffs, img_k, k0, Kb, Kstride):
        """img_k[p, k] = coeffs[k0 + k, p] for k < Kb (p: flat pixel index), 0 for Kb <= k < Kstride.

        coeffs : (Ktot, Ny, Nx), C order: coeffs[k] is an image, pixel p = x + Nx * y.
        """
        Npix = self.Nx * self.Ny
        gsize = (_round_up(Npix, 32), _round_up(Kstride, 32) // 32 * 8)
        self._gather(self.queue, gsize, (32, 8), coeffs.data, img_k.data,
                     np.int32(Npix), np.int32(k0), np.int32(Kb), np.int32(Kstride))

    def scatter(self, img_k, coeffs, k0, Kb, Kstride):
        """coeffs[k0 + k, p] = img_k[p, k] for k < Kb (the inverse of gather)."""
        Npix = self.Nx * self.Ny
        gsize = (_round_up(Npix, 32), _round_up(Kstride, 32) // 32 * 8)
        self._scatter(self.queue, gsize, (32, 8), img_k.data, coeffs.data,
                      np.int32(Npix), np.int32(k0), np.int32(Kb), np.int32(Kstride))

    def forward(self, img_k, sino_k, Kstride):
        """sino_k (R, Ns, Kstride) = forward projection of img_k (Nx*Ny, Kstride)."""
        assert Kstride % 4 == 0, "Kstride must be a multiple of 4"
        n_strips = -(-self.Ns // self.bins_per_item)
        if self.tiled:
            for k4, n4 in self._slices(Kstride):
                gsize = (_round_up(n4, self.FCQ), _round_up(n_strips, self.FSB), self.n_blocks * self.FAB)
                self._fwd_t(self.queue, gsize, (self.FCQ, self.FSB, self.FAB), img_k.data, sino_k.data,
                            self.geo_gpu.data, self.blocks_gpu.data, np.int32(self.Nx), np.int32(self.Ny),
                            np.int32(self.Ns), np.int32(self.R), np.int32(Kstride // 4), np.int32(k4),
                            np.int32(n4), self.scale, np.int32(self.W),
                            cl.LocalMemory(self.FVR * self.W * self.FCQ * 16), cl.LocalMemory(4 * self.FVR))
            return
        for k4, n4 in self._slices(Kstride):
            gsize = (_round_up(n4, 32), _round_up(n_strips, 4), self.R)
            self._fwd(self.queue, gsize, (32, 4, 1), img_k.data, sino_k.data, self.geo_gpu.data,
                      np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Ns), np.int32(self.R),
                      np.int32(Kstride // 4), np.int32(k4), np.int32(n4), self.scale)

    def backward(self, sino_k, img_k, Kstride):
        """img_k (Nx*Ny, Kstride) = backward projection of sino_k (R, Ns, Kstride)."""
        assert Kstride % 4 == 0, "Kstride must be a multiple of 4"
        if self.tiled:
            for k4, n4 in self._slices(Kstride):
                gsize = (_round_up(n4, self.BCQ), _round_up(self.Nx, self.BT), _round_up(self.Ny, self.BT))
                self._bwd_t(self.queue, gsize, (self.BCQ, self.BT, self.BT), sino_k.data, img_k.data,
                            self.geo_gpu.data, np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Ns),
                            np.int32(self.R), np.int32(Kstride // 4), np.int32(k4), np.int32(n4))
            return
        for k4, n4 in self._slices(Kstride):
            gsize = (_round_up(n4, 32), _round_up(self.Nx, 4), self.Ny)
            self._bwd(self.queue, gsize, (32, 4, 1), sino_k.data, img_k.data, self.geo_gpu.data,
                      np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Ns), np.int32(self.R),
                      np.int32(Kstride // 4), np.int32(k4), np.int32(n4))

    def _slices(self, Kstride):
        """(first, count) of the channel slices, in float4 units."""
        K4, c4 = Kstride // 4, self.channels_per_launch // 4
        return [(k4, min(c4, K4 - k4)) for k4 in range(0, K4, c4)]
