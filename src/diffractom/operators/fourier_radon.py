"""
Fourier-slice parallel-beam projector on the GPU (FFTs: VkFFT via pyvkfft; fourier_radon.cl).

The same geometry and conventions as ParallelRadon (detector coordinate t = X x + Y y + T0 in bins,
forward scaled by delta_x^2 / delta_s, backward = transpose weighted by the angle weights), but the
projection is computed through the Fourier slice theorem instead of summing pixels along every
angle. For angle a and detector bin s,

    P f [a, s] = 1/L sum_{j = -L/2+1}^{L/2} fhat(w_j X_a, w_j Y_a) e^{-i w_j T_a} h(w_j) e^{i w_j s},

with w_j = 2 pi j / L, fhat the discrete-time Fourier transform of the pixel samples, h(w) =
sinc^2(w / 2) the response of a detector bin (the hat weight of ParallelRadon) and L the padded
detector length. fhat is evaluated on the polar samples by a type-2 non-uniform FFT: pixels divided
by the kernel's transform, zero-padded to an oversampled grid (sigma = 1.5), a 2-D FFT, interpolation
with the "exponential of semicircle" kernel of W = 7 taps (as in FINUFFT): about 1e-5 relative
error against the exact transform. The backward projection is the exact transpose (the same
float32 interpolation weights, spread as a gather over grid points: deterministic).

How it differs from ParallelRadon: ParallelRadon spreads every pixel sample with a hat onto the
detector bins, which aliases (at many angles the projected pixel centres line up on a grid that
beats with the bins). Against exact line integrals of smooth images, ParallelRadon errs by about
0.5-1.5 %, this projector by about 1e-5; the two differ by about 1 % on reconstructed coefficient
maps. Cost: about N^2 log N per image instead of R N^2. Measured on an A40 (360 angles, forward and
backward projection, us per orientation): 400 x 400 pixels 125 vs 217 for ParallelRadon, 800 x 800
353 vs 951, 1200 x 1200 738 vs 2275.

Images are read from and written to (K, Ny, Nx) coefficient arrays directly (one image per channel,
the layout of the FFTs); sinograms are the operator's (R, Ns, Kstride), channel fastest. Channels go
through the FFTs in slices of C (the scratch buffers: about 8 Mx My + 12 R L bytes per channel).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

try:  # optional dependency
    from pyvkfft.opencl import VkFFTApp
except Exception:  # noqa: BLE001
    VkFFTApp = None


def available():
    """Whether the FFT library (pyvkfft) can be used."""
    return VkFFTApp is not None


def _smooth(n, even=True):
    """The smallest m >= n with prime factors 2, 3, 5, 7 (fast FFT sizes), even if asked."""
    m = int(np.ceil(n))
    while True:
        k = m
        for p in (2, 3, 5, 7):
            while k % p == 0:
                k //= p
        if k == 1 and (not even or m % 2 == 0):
            return m
        m += 1


def _es(z, beta):
    out = np.zeros_like(z)
    inside = np.abs(z) < 1
    out[inside] = np.exp(beta * (np.sqrt(1 - z[inside] ** 2) - 1))
    return out


def _round_up(n, m):
    return (n + m - 1) // m * m


class FourierRadon:
    """Fourier-slice parallel-beam Radon transform of image batches (see the module docstring).

    Parameters as ParallelRadon (img_shape (Nx, Ny), angles, n_detectors, image_width,
    detector_width, detector_shift, angle_weights), and:
    W : int
        Interpolation kernel width (taps per dimension).
    sigma : float
        Oversampling of the Fourier grid. The default, sigma = 1.5 with W = 7, gives about 1e-5
        relative accuracy (as sigma = 2 with W = 6) and was faster on an A40 (forward / backward
        projection, us per orientation, 400 / 800 / 1200 pixels: 54 / 169 / 374 and 71 / 184 / 364,
        against 68 / 256 / 556 and 79 / 237 / 508 with sigma = 2, W = 6).
    L_factor : float
        Padded detector length relative to the span of the projections (and the detector).
    max_gb : float
        GPU memory for the scratch buffers of one channel slice.
    """

    def __init__(self, queue, img_shape, angles, n_detectors, image_width, detector_width,
                 detector_shift=0.0, angle_weights=None, W=7, sigma=1.5, L_factor=1.25, max_gb=1.5):
        if VkFFTApp is None:
            raise ImportError("projector='fft' needs pyvkfft (pip install pyvkfft)")
        self.queue = q = queue
        self.Nx, self.Ny = (int(n) for n in img_shape)
        self.Ns = int(n_detectors)
        angles = np.asarray(angles, dtype=np.float64)
        self.R = R = len(angles)
        if angle_weights is None:
            angle_weights = np.pi / R
        self.angle_weights = np.broadcast_to(np.asarray(angle_weights, dtype=np.float64), (R,)).copy()

        # ---- geometry: as ParallelRadon
        n_max = max(self.Nx, self.Ny)
        delta_x = image_width / float(n_max)
        delta_s = float(detector_width) / self.Ns
        rho = delta_s / delta_x
        X = np.cos(angles - 0.5 * np.pi) / rho
        Y = np.sin(angles - 0.5 * np.pi) / rho
        T0 = (self.Ns - 1) / 2.0 - X * (self.Nx - 1) / 2.0 - Y * (self.Ny - 1) / 2.0 - detector_shift / delta_s
        self.scale = np.float32(delta_x * delta_x / delta_s)
        self.cx, self.cy = self.Nx // 2, self.Ny // 2
        T = T0 + X * self.cx + Y * self.cy                    # phase reference: pixel (cx, cy)

        # ---- sizes
        corners = np.array([[0, 0], [self.Nx - 1, 0], [0, self.Ny - 1], [self.Nx - 1, self.Ny - 1]], float)
        t = corners[:, :1] * X[None, :] + corners[:, 1:] * Y[None, :] + T0[None, :]
        span = max(t.max(), self.Ns - 1) - min(t.min(), 0.0) + 4
        self.L = L = _smooth(L_factor * span)
        self.H = H = L // 2 + 1
        self.W = W = int(W)
        beta = 0.97 * np.pi * (1 - 1 / (2 * sigma)) * W        # (FINUFFT's choice for the ES kernel)
        self.Mx, self.My = Mx, My = _smooth(sigma * self.Nx), _smooth(sigma * self.Ny)
        self.Hx = Hx = Mx // 2 + 1
        RH = R * H

        # ---- deapodisation: psi(x) = int phi(kappa) cos(2 pi kappa x / M) dkappa
        kap = np.linspace(-W / 2, W / 2, 4001)
        phi = _es(2 * kap / W, beta)
        dk = kap[1] - kap[0]
        psi = lambda x, M: (phi[None, :] * np.cos(2 * np.pi * kap[None, :] * x[:, None] / M)).sum(1) * dk
        dev = lambda a: clarray.to_device(q, np.ascontiguousarray(a))
        self.inv_dx = dev((1.0 / psi(np.arange(self.Nx) - self.cx, Mx)).astype(np.float32))
        self.inv_dy = dev((1.0 / psi(np.arange(self.Ny) - self.cy, My)).astype(np.float32))

        # ---- polar samples (angle a, frequency j >= 0): phase and bin response, taps, weights
        om = 2 * np.pi * np.arange(H) / L
        filt = (np.sinc(om / (2 * np.pi)) ** 2)[None, :] * np.exp(-1j * om[None, :] * T[:, None]) / L
        filt32 = filt.astype(np.complex64).ravel()
        self.filt = dev(filt32)
        ku = (Mx * om[None, :] * X[:, None] / (2 * np.pi)).ravel()          # (RH,) grid coordinates
        kv = (My * om[None, :] * Y[:, None] / (2 * np.pi)).ravel()
        k1b = (np.floor(ku - W / 2) + 1).astype(np.int64)
        k2b = (np.floor(kv - W / 2) + 1).astype(np.int64)
        w1 = _es(2 * (ku[:, None] - (k1b[:, None] + np.arange(W))) / W, beta).astype(np.float32)   # (RH, W)
        w2 = _es(2 * (kv[:, None] - (k2b[:, None] + np.arange(W))) / W, beta).astype(np.float32)
        # the tap weights as fr_slices computes them (float32 products), so that the adjoint uses the same
        wt = w2[:, :, None] * w1[:, None, :]                                                       # (RH, W, W): [l][i]
        self.kb = dev(np.stack([k1b, k2b], 1).astype(np.int32))
        self.w1s, self.w2s = dev(w1), dev(w2)

        # ---- the adjoint of the interpolation: for every TS x TS tile of the half grid (rows k2, columns
        # k1 <= Mx/2), the samples with a tap on it, directly (k) or through the Hermitian mirror (-k):
        # the spread is the Hermitian part of the transposed interpolation, which the C2R FFT turns
        # into Re(inverse DFT); fr_spread gathers it tile by tile
        self.TS = TS = 16
        self.n_tx, n_ty = -(-Hx // TS), -(-My // TS)
        k1 = (k1b[:, None] + np.arange(W)).repeat(W, 0).reshape(RH, W, W)       # [smp, l, i] -> k1b + i
        k2 = (k2b[:, None] + np.arange(W)).repeat(W, 1).reshape(RH, W, W)       # [smp, l, i] -> k2b + l
        smp = np.broadcast_to(np.arange(RH)[:, None, None], (RH, W, W))
        keys = []
        for sgn, flag in ((1, 0), (-1, 1)):
            c1, c2 = (sgn * k1) % Mx, (sgn * k2) % My
            ok = c1 <= Mx // 2
            tile = (c2[ok] // TS) * self.n_tx + c1[ok] // TS
            keys.append((tile * 2 + flag) * RH + smp[ok])
        keys = np.unique(np.concatenate(keys))                                   # sorted: tile, flag, sample
        tile, flag, smps = keys // (2 * RH), (keys // RH) % 2, keys % RH
        self.n_tiles = self.n_tx * n_ty
        tile_ptr = np.zeros(self.n_tiles + 1, np.int64)
        np.cumsum(np.bincount(tile, minlength=self.n_tiles), out=tile_ptr[1:])
        self.tile_ptr = dev(tile_ptr.astype(np.int32))
        self.tile_smp = dev(np.where(flag == 1, -smps - 1, smps).astype(np.int32))
        self.wa = dev(self.angle_weights.astype(np.float32))

        # ---- channel slice and scratch buffers
        per_channel = 4 * Mx * My + 8 * My * Hx + 8 * RH + 4 * R * L
        # channels per work-item of the interpolations: 1 (A40, us per orientation for the forward one at
        # 400 / 800 / 1200 pixels: CC = 1 31 / 103 / 213, CC = 2 40 / 169 / 472, CC = 8 152 / 632 / 1163:
        # what runs at once must stay within a few channel planes of the grid for the L2 cache)
        self.CC = CC = 1
        self.CCS = 8                                       # channels per work-group of the spreading
        self.C = C = int(max(1, min(256 // CC, max_gb * 1024**3 // (CC * per_channel)))) * CC
        self.grid = clarray.empty(q, (C, My, Mx), np.float32)
        self.G = clarray.empty(q, (C, My, Hx), np.complex64)
        self.spec = clarray.empty(q, (C * R, H), np.complex64)
        self.lines = clarray.empty(q, (C * R, L), np.float32)
        self._plans = {}                                   # FFT plans (and buffer views) per slice size

        src = Path(__file__).with_name("fourier_radon.cl").read_text()
        prg = cl.Program(q.context, src).build(options=[f"-DW={W}", f"-DCC={CC}", f"-DCCS={self.CCS}", f"-DTS={TS}"])
        self.k = {n: cl.Kernel(prg, n) for n in ("fr_pad", "fr_slices", "fr_lines_out", "fr_zero_channels",
                                                  "fr_lines_in", "fr_unfilter", "fr_spread", "fr_crop")}
        self.RH = RH

    def _slices(self, Kb):
        """Equal slices of at most C channels: (first channel, channels, slice size)."""
        n = -(-Kb // self.C)
        cs = -(-Kb // n)
        return [(c0, min(cs, Kb - c0), cs) for c0 in range(0, Kb, cs)]

    def _plan(self, cs):
        """FFT plans and buffer views for slices of cs channels (cached)."""
        if cs not in self._plans:
            R, q = self.R, self.queue
            grid, G, spec, lines = self.grid[:cs], self.G[:cs], self.spec[:cs * R], self.lines[:cs * R]
            app2 = VkFFTApp(grid.shape, np.float32, queue=q, ndim=2, inplace=False, norm=0, r2c=True)
            app1 = VkFFTApp(lines.shape, np.float32, queue=q, ndim=1, inplace=False, norm=0, r2c=True)
            self._plans[cs] = (app2, app1, grid, G, spec, lines)
        return self._plans[cs]

    def nbytes(self):
        bufs = (self.grid, self.G, self.spec, self.lines, self.kb, self.w1s, self.w2s, self.tile_ptr, self.tile_smp,
                self.filt)
        return sum(a.nbytes for a in bufs)

    def forward(self, coeffs, k0, Kb, sino, Kstride):
        """sino (R, Ns, Kstride), channels 0 .. Kb-1 = forward projections of images k0 .. k0+Kb-1 of
        coeffs (a (K, Ny, Nx) C-order array); channels Kb .. Kstride-1 = 0."""
        q, k = self.queue, self.k
        R, L, Ns = self.R, self.L, self.Ns
        for c0, cb, C in self._slices(Kb):
            app2, app1, grid, G, spec, lines = self._plan(C)
            k["fr_pad"](q, (_round_up(self.Mx, 64), _round_up(self.My, 4), C), (64, 4, 1), coeffs.data, np.int64(k0 + c0),
                        grid.data, self.inv_dx.data, self.inv_dy.data, np.int32(self.Nx), np.int32(self.Ny),
                        np.int32(self.Mx), np.int32(self.My), np.int32(self.cx), np.int32(self.cy), np.int32(cb))
            app2.fft(grid, G)
            k["fr_slices"](q, (_round_up(self.RH, 64), C // self.CC), (64, 1), G.data, self.kb.data,
                           self.w1s.data, self.w2s.data, self.filt.data, spec.data, np.int32(self.Mx),
                           np.int32(self.My), np.int32(self.RH), np.int32(self.H))
            app1.ifft(spec, lines)
            k["fr_lines_out"](q, (_round_up(Ns, 32), _round_up(cb, 32) // 4, R), (32, 8, 1), lines.data, sino.data,
                              np.int32(R), np.int32(Ns), np.int32(L), np.int32(Kstride), np.int32(c0), np.int32(cb),
                              self.scale)
        if Kb < Kstride:
            k["fr_zero_channels"](q, (_round_up(Kstride - Kb, 32), R * Ns), (32, 1), sino.data, np.int32(R),
                                  np.int32(Ns), np.int32(Kstride), np.int32(Kb))

    def backward(self, sino, Kstride, coeffs, k0, Kb):
        """Images k0 .. k0+Kb-1 of coeffs (a (K, Ny, Nx) C-order array) = backward projections of
        channels 0 .. Kb-1 of sino (R, Ns, Kstride): the transpose of forward, weighted by the angle
        weights (as ParallelRadon.backward)."""
        q, k = self.queue, self.k
        R, L, Ns, TS = self.R, self.L, self.Ns, self.TS
        for c0, cb, C in self._slices(Kb):
            app2, app1, grid, G, spec, lines = self._plan(C)
            k["fr_lines_in"](q, (_round_up(L, 32), _round_up(C, 32) // 4, R), (32, 8, 1), sino.data, lines.data,
                             self.wa.data, np.int32(R), np.int32(Ns), np.int32(L), np.int32(Kstride), np.int32(c0),
                             np.int32(cb), np.int32(C))
            app1.fft(lines, spec)
            k["fr_unfilter"](q, (_round_up(self.RH, 64), C), (64, 1), spec.data, self.filt.data, np.int32(self.RH))
            k["fr_spread"](q, (self.n_tiles * TS, -(-C // self.CCS) * TS), (TS, TS), spec.data, self.tile_ptr.data,
                           self.tile_smp.data, self.kb.data, self.w1s.data, self.w2s.data, G.data,
                           np.int32(self.Mx), np.int32(self.My), np.int32(self.RH), np.int32(self.H),
                           np.int32(self.n_tx), np.int32(C))
            app2.ifft(G, grid)
            k["fr_crop"](q, (_round_up(self.Nx, 64), _round_up(self.Ny, 4), cb), (64, 4, 1), grid.data, coeffs.data,
                         np.int64(k0 + c0), self.inv_dx.data, self.inv_dy.data, np.int32(self.Nx), np.int32(self.Ny),
                         np.int32(self.Mx), np.int32(self.My), np.int32(self.cx), np.int32(self.cy), np.int32(cb))

    def release(self):
        self._plans = {}
        for a in (self.grid, self.G, self.spec, self.lines, self.kb, self.w1s, self.w2s, self.tile_ptr, self.tile_smp,
                  self.filt, self.inv_dx, self.inv_dy, self.wa):
            if a.base_data is not None:
                a.base_data.release()
