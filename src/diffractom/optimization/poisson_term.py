"""The Poisson data term on the GPU for the solvers and refiners that work with a prediction pred = A x (PoissonTerm):
the deviance on the fitted angles, the EM ratio, and the gradient and Fisher curvature with respect to the prediction.

    f = sum_i fit_i w_i / g_i [ mu_i - b_i + b_i log(b_i / mu_i) ],   mu_i = max(pred_i, 0) + bg_i + eps_counts g_i

(as FISTAPoisson's; fit_i: the angle of point i is fitted). w, g, bg: per segment, per (translation, segment) or
data-shaped (FISTAPoisson's conventions; None: w = g = 1, bg = 0).
"""
import numpy as np
import pyopencl.array as clarray
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel

from .fista_poisson import _periodic
from ..utils.arrays import as_host, check_device

_PRE = r"""
#define PER(a, p, i) ((p) ? (a)[(i) % (p)] : 0.0f)
"""
_COMMON = ("const float *b, const float *w, unsigned long pw, const float *g, unsigned long pg, const float *bg, "
           "unsigned long pbg, const unsigned char *fit, unsigned long row, float eps")


class PoissonTerm:
    def __init__(self, queue, data_shape, b, weights=None, gain=None, background=None, fit_rows=None,
                 eps_counts=1e-3):
        """fit_rows: bool (N_Omega,) or indices of the fitted angles (None: all)."""
        self.queue = q = queue
        ctx = q.context
        self.dshape = dshape = tuple(int(d) for d in data_shape)
        self.b = check_device(b, dshape, "b") if isinstance(b, clarray.Array) else \
            clarray.to_device(q, as_host(b, dshape, "b"))
        self._up = []
        one = np.ones(1, np.float32)
        self.w, self.pw = _periodic(q, weights if weights is not None else one, dshape, "weights", self._up)
        self.g, self.pg = _periodic(q, gain if gain is not None else one, dshape, "gain", self._up)
        self.bg, self.pbg = _periodic(q, background, dshape, "background", self._up)
        if self.bg is None:
            self.bg = clarray.zeros(q, (1,), np.float32)
        self.row = int(np.prod(dshape[1:]))
        fit = np.ones(dshape[0], np.uint8)
        if fit_rows is not None:
            r = np.asarray(fit_rows)
            fit[:] = 0
            fit[np.flatnonzero(r) if r.dtype == bool else r.astype(np.int64)] = 1
        self.fit_host = fit.astype(bool)
        self.fit = clarray.to_device(q, fit)
        self.eps = np.float32(eps_counts)
        head = ("const unsigned long a = i / row; const float wi = fit[a] ? PER(w, pw, i) : 0.0f; "
                "const float gi = PER(g, pg, i); const float mu = fmax(pred[i], 0.0f) + PER(bg, pbg, i) + eps * gi; ")
        self._ratio = ElementwiseKernel(ctx, "float *out, const float *pred, " + _COMMON,
                                        head + "out[i] = wi / gi * b[i] / mu", "pt_ratio", preamble=_PRE)
        self._wg = ElementwiseKernel(ctx, "float *out, " + _COMMON,
                                     "const unsigned long a = i / row; "
                                     "out[i] = fit[a] ? PER(w, pw, i) / PER(g, pg, i) : 0.0f", "pt_wg",
                                     preamble=_PRE)
        self._gc = ElementwiseKernel(ctx, "float *r, float *c, const float *pred, float floor, " + _COMMON,
                                     head + "r[i] = wi / gi * (1.0f - b[i] / mu); "
                                            "c[i] = wi / (gi * fmax(mu, floor * gi))", "pt_grad_curv",
                                     preamble=_PRE)
        self._dev = ReductionKernel(
            ctx, np.float64, neutral="0", reduce_expr="a+b",
            map_expr="pt_dev(i, pred, b, w, pw, g, pg, bg, pbg, fit, row, eps)",
            arguments="const float *pred, " + _COMMON,
            preamble=_PRE + r"""
double pt_dev(long i, __global const float *pred, __global const float *b, __global const float *w,
              unsigned long pw, __global const float *g, unsigned long pg, __global const float *bg,
              unsigned long pbg, __global const unsigned char *fit, unsigned long row, float eps)
{
    if (!fit[i / row]) return 0.0;
    const float wi = PER(w, pw, i);
    if (wi == 0.0f) return 0.0;
    const float gi = PER(g, pg, i);
    const double mu = (double)fmax(pred[i], 0.0f) + PER(bg, pbg, i) + eps * gi;
    const double bi = b[i];
    return wi / gi * (mu - bi + (bi > 0.0 ? bi * log(bi / mu) : 0.0));
}""")

    def _args(self):
        return (self.b, self.w, np.uint64(self.pw), self.g, np.uint64(self.pg), self.bg, np.uint64(self.pbg),
                self.fit, np.uint64(self.row), self.eps)

    def deviance(self, pred):
        """The deviance of the prediction pred (= A x, data-shaped GPU array) on the fitted angles."""
        return float(self._dev(pred, *self._args(), queue=self.queue).get())

    def n_points(self):
        """The number of counted data points (fitted angle, weight != 0)."""
        w = self.w.get()
        wf = np.resize(w, int(np.prod(self.dshape))).reshape(self.dshape[0], -1)[self.fit_host]
        return int(np.count_nonzero(wf))

    def em_ratio(self, pred, out):
        """out = w b / (g mu) on the fitted angles (0 elsewhere): A^T of it is EM's back-projection."""
        self._ratio(out, pred, *self._args())

    def weight_over_gain(self, out):
        """out = w / g on the fitted angles (0 elsewhere): A^T of it is EM's sensitivity."""
        self._wg(out, *self._args())

    def grad_curv(self, pred, r, c, floor_counts=0.25):
        """r = d f / d pred = w/g (1 - b/mu) and c = w / (g max(mu, floor_counts g)) (the Fisher information, mu at
        least floor_counts photons) on the fitted angles, 0 elsewhere."""
        self._gc(r, c, pred, np.float32(floor_counts), *self._args())
