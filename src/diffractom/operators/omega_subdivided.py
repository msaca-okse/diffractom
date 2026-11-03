"""The forward model integrated over each omega bin's rotation interval, projection included.

SinglePhaseForwardOperator integrates the diffraction part (the pole-figure factor) over each bin's interval but
projects the sample once, at the bin centre. While a frame is recorded the sample turns by the bin width, so the
projection moves too: a voxel at distance r from the axis sweeps r * d (d the bin width in radians) across the
detector, and a spot recorded early in the bin comes from the sample turned by up to d / 2 from the centre.

OmegaSubdividedOperator models this exactly in the limit of many sub-bins: an operator with M sub-bins per data bin
(each with its own projection angle and its own share of the omega quadrature) whose M predictions are averaged
into the data bin (the operators return bin means). M = 1 is the ordinary operator. It costs M times the work and
memory of the ordinary operator; it has direct_cl / adjoint_cl (and NumPy direct / adjoint), so the solvers use it
with the unfused update (FISTAHuber(..., fused=False)) and estimate_L_power work.
"""
from __future__ import annotations

import numpy as np
import pyopencl.array as clarray
from pyopencl.elementwise import ElementwiseKernel

from .single_phase_forward_operator import SinglePhaseForwardOperator


class OmegaSubdividedOperator:
    """SinglePhaseForwardOperator(cfg with N_Omega * subdivisions bins, ...) with its omega sub-bins averaged into the
    cfg's N_Omega data bins. The keyword arguments go to the inner operator (``self.fine``)."""

    def __init__(self, cfg, material, grid, max_gb, subdivisions: int = 2, **kwargs):
        M = int(subdivisions)
        if M < 1:
            raise ValueError("subdivisions >= 1")
        cfg_f = dict(cfg)
        cfg_f["N_Omega"] = int(cfg["N_Omega"]) * M
        self.fine = SinglePhaseForwardOperator(cfg_f, material, grid, max_gb, **kwargs)
        f = self.fine
        self.M, self.cfg = M, dict(cfg)
        self.ctx, self.queue = f.ctx, f.queue
        self.K, self.Nx, self.Ny, self.My, self.N_seg = f.K, f.Nx, f.Ny, f.My, f.N_seg
        self.N_Omega = int(cfg["N_Omega"])
        self.coeff_shape = f.coeff_shape
        self.data_shape = (self.N_Omega, self.My, self.N_seg)
        self.angles = f.angles.reshape(self.N_Omega, M).mean(axis=1)
        self.grid, self.material = f.grid, f.material
        self._work = clarray.empty(self.queue, f.data_shape, np.float32)
        n_row = self.My * self.N_seg
        self._mean = ElementwiseKernel(
            self.ctx, "const float *fine, float *out, const int M, const long n_row",
            "const long a = i / n_row, j = i - a * n_row; float s = 0.0f; "
            "for (int m = 0; m < M; ++m) s += fine[(a * M + m) * n_row + j]; out[i] = s / M;", "omega_mean")
        self._spread = ElementwiseKernel(
            self.ctx, "const float *coarse, float *fine, const int M, const long n_row",
            "const long af = i / n_row, j = i - af * n_row; fine[i] = coarse[(af / M) * n_row + j] / M;",
            "omega_spread")
        self._n_row = np.int64(n_row)

    def support_mask(self):
        return self.fine.support_mask()

    def direct_cl(self, coeffs, data):
        self.fine.direct_cl(coeffs, self._work)
        self._mean(self._work, data, np.int32(self.M), self._n_row, range=slice(0, int(np.prod(self.data_shape))))
        return data

    def adjoint_cl(self, data, coeffs):
        self._spread(data, self._work, np.int32(self.M), self._n_row, range=slice(0, self._work.size))
        self.fine.adjoint_cl(self._work, coeffs)
        return coeffs

    def direct(self, coeffs):
        c = clarray.to_device(self.queue, np.ascontiguousarray(coeffs, np.float32))
        out = clarray.empty(self.queue, self.data_shape, np.float32)
        return self.direct_cl(c, out).get()

    def adjoint(self, data):
        d = clarray.to_device(self.queue, np.ascontiguousarray(data, np.float32).reshape(self.data_shape))
        out = clarray.empty(self.queue, self.coeff_shape, np.float32)
        return self.adjoint_cl(d, out).get()
