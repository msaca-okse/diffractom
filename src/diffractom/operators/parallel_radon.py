"""Parallel-beam Radon transform on the GPU, vectorised over many channels (orientations)."""
from __future__ import annotations

import time
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
    ``t = X_a x + Y_a y + T0_a`` (in bins) and contributes to bin s with the weight
    ``K(t - s)`` of the kernel (see ``kernel``). The forward projection is scaled by
    ``delta_x**2 / delta_s``; the backward projection is the transpose weighted by
    the angle weights, i.e. the adjoint up to ``w_a delta_s / delta_x**2``.
    With ``kernel="hat"`` this is the discretisation of gratopy's parallel-beam transform
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
    kernel : {"trapezoid", "hat"}
        How a pixel is spread over the detector bins.
        "trapezoid" (default): the footprint of the pixel, a box as wide as the spacing A of neighbouring pixel
        centres on the detector (A = max(|cos|, |sin|) delta_x / delta_s bins), averaged over the bin
        (distance-driven projection): ``K(d) = clip((1 + A) / 2 - |d|, 0, min(1, A)) / A``. Every bin gets the same
        total weight from a row of pixels at every angle. Pixels may be smaller or larger than the detector bins:
        for A <= 1 a pixel touches the two nearest bins, for larger pixels ceil(1 + A) bins.
        "hat": linear interpolation between the two bins nearest to the projected pixel centre, the discretisation
        above and gratopy's (diffractom's default until 0.2.0; identical to the trapezoid at 0 and 90 degrees when
        delta_x = delta_s). Its total weight per bin is constant only for A = 1 / integer: for delta_x = delta_s the
        projection of a smooth image is modulated from bin to bin by 6.7 % (rms) at exactly 45 degrees (a few
        percent within +-1.5 degrees of it for compact features), 1.3 % at 26.6 and 63.4, 0.9 % at 33.7 and 56.3
        degrees, and a reconstruction gets a diagonal cross-hatch pattern. It needs delta_x <= delta_s (larger
        pixels would leave bins empty).
    beam_profile : BeamProfile, optional
        The profile of the beam across the detector coordinate (the translation direction), in detector bins, unit
        area (see ``operators.beam_profile``). Given, the weight is the pixel footprint (as for the trapezoid)
        convolved with the profile, tabulated per angle (``K_a(d)``, linear interpolation on 256 points per bin)
        instead of averaged over one bin; ``BeamProfile(width=1)`` is the trapezoid. Needs kernel="trapezoid".
        A wider profile makes a pixel touch more bins (forward and backward cost grow with it).
    tiled : "auto", bool or dict
        The tiled kernels stage what a work-group shares in local memory (the image rows of a block
        of angles and bins in the forward projection, the sinogram bins of a pixel tile in the
        backward one) instead of reading it from the caches about twice per angle; the plain kernels
        do not. Both give bitwise the same results; which is faster depends on the GPU, the grid and
        the angles. "auto" (default): the forward and the backward projection each use the faster of
        the tiled kernel and the plain one in slices of 128 or 64 channels, timed once here on a
        scratch stack (see ``modes``). True / False: the tiled / plain kernels for both. A dict
        ``modes`` of another instance: its choice (e.g. for subsets of the same angles).
        Measured (500 angles over 360 degrees, 512 channels, ms): forward tiled / plain-128 on an
        A100 at 64, 115, 200, 400 pixels 5.9/2.1, 9.8/6.3, 27.7/19.0, 124/138, on a V100 6.0/2.9,
        17.5/9.3, 50/52, 225/268; backward tiled / plain on the A100 at 115 and 400 pixels
        7.9/5.5, 87/129. For every 8th angle (a subset of 63) the plain forward is 1.8-3.4x faster
        than the tiled one at 400 pixels (the tiled kernel's angle blocks then span 90 degrees).
        The tiled kernels were tuned on an A40 (with the hat kernel), where they were the faster
        ones at 120 pixels too.
    channels_per_launch : int, optional
        The projections run in slices of this many channels (a multiple of 4). Default: all at
        once with the tiled kernels; with the plain kernels (tiled=False) 64 for images of 160 x 160
        pixels or more, 128 otherwise (measured on an A40, 360 angles: for 400 x 400 pixels one
        launch over 256-1024 channels took 2.6x as long as slices of 64; for 99 x 99 up to 3x as
        long as slices of 128). Given, it applies to both kernels (and "auto" only chooses between
        them).
    """

    def __init__(self, queue, img_shape, angles, n_detectors, image_width, detector_width,
                 detector_shift=0.0, angle_weights=None, bins_per_item=3, channels_per_launch=None,
                 tiled="auto", kernel="trapezoid", beam_profile=None):
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

        if kernel not in ("hat", "trapezoid"):
            raise ValueError(f"kernel must be 'hat' or 'trapezoid', not {kernel!r}")
        a_max = float(np.maximum(np.abs(X), np.abs(Y)).max())       # largest spacing of pixel centres, in bins
        if kernel == "hat" and a_max > 1.0 + 1e-6:
            raise ValueError("kernel='hat' needs pixels no larger than the detector bins (delta_x <= delta_s); "
                             "use kernel='trapezoid'")
        self.kernel = kernel
        self.n_taps = max(2, int(np.ceil(1.0 + a_max - 1e-6)))      # bins a pixel touches (trapezoid)
        from .beam_profile import as_beam_profile
        self.beam_profile = as_beam_profile(beam_profile)
        self.phw = 0.5 * (1.0 + a_max) if kernel == "trapezoid" else 1.0   # half support of the weight (bins)
        if self.beam_profile is not None:
            if kernel != "trapezoid":
                raise ValueError("beam_profile needs kernel='trapezoid' (the profile replaces the bin average)")
            self.kernel = "profile"
            tab, self.phw, self.pnt = self.beam_profile.tables(np.maximum(np.abs(X), np.abs(Y)))
            self.n_taps = int(np.ceil(2.0 * self.phw)) + 1
            self.ktab_gpu = clarray.to_device(queue, tab)
        else:
            self.ktab_gpu = clarray.zeros(queue, (1,), np.float32)
        geo = np.stack([X, Y, T0, self.angle_weights], axis=1).astype(np.float32)
        self.geo_gpu = clarray.to_device(queue, np.ascontiguousarray(geo))

        self.bins_per_item = int(bins_per_item)
        if channels_per_launch is not None and (channels_per_launch % 4 or channels_per_launch <= 0):
            raise ValueError("channels_per_launch must be a positive multiple of 4")
        plain_cpl = channels_per_launch or (64 if self.Nx * self.Ny >= 160 * 160 else 128)
        if isinstance(tiled, dict):
            self.modes = {d: (bool(tiled[d][0]), int(tiled[d][1])) for d in ("forward", "backward")}
        elif tiled == "auto":
            self.modes = None   # timed below
        elif tiled in (True, False):
            m = (True, channels_per_launch or 1 << 30) if tiled else (False, plain_cpl)
            self.modes = {"forward": m, "backward": m}
        else:
            raise ValueError(f"tiled must be 'auto', True, False or a modes dict, not {tiled!r}")
        self._use_tiled = self.modes is None or any(m[0] for m in self.modes.values())
        src = Path(__file__).with_name("radon_kernels.cl").read_text()
        options = [f"-DNB={self.bins_per_item}"]
        if self.kernel == "trapezoid":
            options += ["-DTRAPEZOID", f"-DNTAP={self.n_taps}"]
        elif self.kernel == "profile":
            options += ["-DPROFILE", f"-DNTAP={self.n_taps}", f"-DPNT={self.pnt}", f"-DPHW={self.phw:.9e}f",
                        f"-DPINV={(self.pnt - 1) / (2.0 * self.phw):.9e}f"]
        if self._use_tiled:
            if self.FCQ * self.FSB * self.FAB > queue.device.max_work_group_size:
                self.FAB = max(1, queue.device.max_work_group_size // (self.FCQ * self.FSB))
            options += self._tiling(geo, queue.device, a_max)
        self.prg = cl.Program(queue.context, src).build(options=options)
        self._fwd = self.prg.radon_forward_k
        self._bwd = self.prg.radon_backward_k
        if self._use_tiled:
            self._fwd_t = self.prg.radon_forward_tiled
            self._bwd_t = self.prg.radon_backward_tiled
        self._gather = self.prg.gather_channels_k_fastest
        self._scatter = self.prg.scatter_channels_k_fastest
        if self.modes is None:
            cand = [(True, channels_per_launch or 1 << 30)]
            cand += [(False, channels_per_launch)] if channels_per_launch else [(False, 128), (False, 64)]
            self.modes = self._tune(cand)
        self.tune_times = getattr(self, "tune_times", None)

    # work-group shapes of the tiled kernels: forward (channel quads, strips, angles), backward
    # (channel quads, 8 x 8 pixels). Forward measured on an A40 (360 angles, us per channel at
    # 120 / 400 / 800 pixels): (4, 8, 16) 10.4 / 133 / 639, (8, 8, 8) 10.7 / 141 / 645,
    # (4, 8, 8) 10.9 / 164 / 787, (2, 16, 8) 14.6 / 241 / 922; plain kernel 14.2 / 228 / 2156.
    FCQ, FSB, FAB = 4, 8, 16
    BCQ, BAB, BT = 4, 16, 8

    def _tiling(self, geo, device, a_max=1.0):
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
            hw = (0.5 * (1.0 + np.abs(A)) if self.kernel == "trapezoid" else
                  np.full_like(A, self.phw))[None, :, None]                         # half width
            e1 = (S0[:, :, None, None] - hw - c[None]) / A[None, :, None]        # (groups, 1, angles, rows)
            e2 = (S1[:, :, None, None] - 1 + hw - c[None]) / A[None, :, None]
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
        hw_max = self.phw
        self.BLMAX = int(np.ceil(span + 2.0 * hw_max)) + 2
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

    TUNE_BYTES = 256 * 2**20   # scratch of the timing in __init__ (tiled="auto")

    def _tune(self, cand):
        """The fastest of the candidate (tiled, channels per launch) modes of the forward and the backward
        projection, timed (best of 3) on a scratch stack of up to 256 channels."""
        q = self.queue
        npix = self.Nx * self.Ny
        C = 4 * max(1, min(64, self.TUNE_BYTES // (16 * (npix + self.R * self.Ns))))
        while True:
            try:
                img = clarray.empty(q, (npix * C,), np.float32)
                sino = clarray.empty(q, (self.R * self.Ns * C,), np.float32)
                break
            except cl.MemoryError:
                if C == 4:            # no room even for 4 channels: the tiled kernels, untimed
                    self.tune_times = {}
                    return {"forward": cand[0], "backward": cand[0]}
                C = max(4, C // 8 * 4)
        img.fill(np.float32(1.0))
        sino.fill(np.float32(1.0))
        out, times = {}, {}
        for d, f in (("forward", lambda: self.forward(img, sino, C)), ("backward", lambda: self.backward(sino, img, C))):
            best, tb = None, np.inf
            for c in cand:
                self.modes = {d: c, "forward" if d == "backward" else "backward": c}
                try:
                    f()
                    q.finish()
                    tc = np.inf
                    for _ in range(3):
                        t0 = time.perf_counter()
                        f()
                        q.finish()
                        tc = min(tc, time.perf_counter() - t0)
                except cl.Error:     # e.g. OUT_OF_RESOURCES for the work-group of a tiled kernel
                    tc = np.inf
                times[(d,) + c] = tc
                if tc < tb:
                    best, tb = c, tc
            out[d] = best if best is not None else (False, 128)
        img.base_data.release()
        sino.base_data.release()
        self.tune_times = times
        return out

    def forward(self, img_k, sino_k, Kstride):
        """sino_k (R, Ns, Kstride) = forward projection of img_k (Nx*Ny, Kstride)."""
        assert Kstride % 4 == 0, "Kstride must be a multiple of 4"
        n_strips = -(-self.Ns // self.bins_per_item)
        tiled, cpl = self.modes["forward"]
        if tiled:
            for k4, n4 in self._slices(Kstride, cpl):
                gsize = (_round_up(n4, self.FCQ), _round_up(n_strips, self.FSB), self.n_blocks * self.FAB)
                self._fwd_t(self.queue, gsize, (self.FCQ, self.FSB, self.FAB), img_k.data, sino_k.data,
                            self.geo_gpu.data, self.blocks_gpu.data, np.int32(self.Nx), np.int32(self.Ny),
                            np.int32(self.Ns), np.int32(self.R), np.int32(Kstride // 4), np.int32(k4),
                            np.int32(n4), self.scale, np.int32(self.W),
                            cl.LocalMemory(self.FVR * self.W * self.FCQ * 16), cl.LocalMemory(4 * self.FVR),
                            self.ktab_gpu.data)
            return
        for k4, n4 in self._slices(Kstride, cpl):
            gsize = (_round_up(n4, 32), _round_up(n_strips, 4), self.R)
            self._fwd(self.queue, gsize, (32, 4, 1), img_k.data, sino_k.data, self.geo_gpu.data,
                      np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Ns), np.int32(self.R),
                      np.int32(Kstride // 4), np.int32(k4), np.int32(n4), self.scale, self.ktab_gpu.data)

    def backward(self, sino_k, img_k, Kstride):
        """img_k (Nx*Ny, Kstride) = backward projection of sino_k (R, Ns, Kstride)."""
        assert Kstride % 4 == 0, "Kstride must be a multiple of 4"
        tiled, cpl = self.modes["backward"]
        if tiled:
            for k4, n4 in self._slices(Kstride, cpl):
                gsize = (_round_up(n4, self.BCQ), _round_up(self.Nx, self.BT), _round_up(self.Ny, self.BT))
                self._bwd_t(self.queue, gsize, (self.BCQ, self.BT, self.BT), sino_k.data, img_k.data,
                            self.geo_gpu.data, np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Ns),
                            np.int32(self.R), np.int32(Kstride // 4), np.int32(k4), np.int32(n4), self.ktab_gpu.data)
            return
        for k4, n4 in self._slices(Kstride, cpl):
            gsize = (_round_up(n4, 32), _round_up(self.Nx, 4), self.Ny)
            self._bwd(self.queue, gsize, (32, 4, 1), sino_k.data, img_k.data, self.geo_gpu.data,
                      np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Ns), np.int32(self.R),
                      np.int32(Kstride // 4), np.int32(k4), np.int32(n4), self.ktab_gpu.data)

    def _slices(self, Kstride, cpl):
        """(first, count) of the channel slices of cpl channels, in float4 units."""
        K4, c4 = Kstride // 4, min(cpl, Kstride) // 4
        return [(k4, min(c4, K4 - k4)) for k4 in range(0, K4, c4)]
