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
    """

    def __init__(self, queue, img_shape, angles, n_detectors, image_width, detector_width,
                 detector_shift=0.0, angle_weights=None, bins_per_item=3):
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
        src = Path(__file__).with_name("radon_kernels.cl").read_text()
        self.prg = cl.Program(queue.context, src).build(options=[f"-DNB={self.bins_per_item}"])
        self._fwd = self.prg.radon_forward_k
        self._bwd = self.prg.radon_backward_k
        self._gather = self.prg.gather_channels_k_fastest
        self._scatter = self.prg.scatter_channels_k_fastest

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
        gsize = (_round_up(Kstride // 4, 32), _round_up(n_strips, 4), self.R)
        self._fwd(self.queue, gsize, (32, 4, 1), img_k.data, sino_k.data, self.geo_gpu.data,
                  np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Ns), np.int32(self.R),
                  np.int32(Kstride // 4), self.scale)

    def backward(self, sino_k, img_k, Kstride):
        """img_k (Nx*Ny, Kstride) = backward projection of sino_k (R, Ns, Kstride)."""
        assert Kstride % 4 == 0, "Kstride must be a multiple of 4"
        gsize = (_round_up(Kstride // 4, 32), _round_up(self.Nx, 4), self.Ny)
        self._bwd(self.queue, gsize, (32, 4, 1), sino_k.data, img_k.data, self.geo_gpu.data,
                  np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Ns), np.int32(self.R),
                  np.int32(Kstride // 4))
