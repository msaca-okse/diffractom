"""Shape, dtype and layout checks for the arrays passed to the operators and solvers.

Conventions (all C-contiguous float32):
  coefficients  (K, Ny, Nx): coeffs[k] is the image of orientation k, rows y, columns x
  data          (N_Omega, My, N_seg) with N_seg = N_eta * N_rings, segment j = eta_bin * N_rings + ring
  weights       N_seg elements, e.g. shaped (N_eta, N_rings), or the data shape (one per data point)
  support       (Ny, Nx) boolean
"""
import numpy as np
import pyopencl.array as clarray


def check_device(a, shape, name):
    """A pyopencl array of the given shape, float32 and C-contiguous; raises otherwise."""
    if not isinstance(a, clarray.Array):
        raise TypeError(f"{name} must be a pyopencl array (or a NumPy array where accepted), not {type(a).__name__}")
    if a.dtype != np.float32:
        raise TypeError(f"{name} must be float32, not {a.dtype}")
    if tuple(a.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, not {tuple(a.shape)}")
    if not a.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous (e.g. clarray.to_device(queue, np.ascontiguousarray(a)))")
    return a


def as_host(a, shape, name):
    """A C-contiguous float32 NumPy array of the given shape: a itself if it already is one,
    otherwise a converted copy."""
    out = np.ascontiguousarray(a, dtype=np.float32)
    if out.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, not {out.shape}")
    return out


def to_device(queue, a, shape, name):
    """A device array from a NumPy array (converted to C-contiguous float32 and uploaded) or a
    pyopencl array (checked)."""
    if isinstance(a, clarray.Array):
        return check_device(a, shape, name)
    return clarray.to_device(queue, as_host(a, shape, name))


def supports_streaming(op):
    """Whether the solvers can stream the coefficients of this operator from host memory (batched
    operators with the fused adjoint update, e.g. SinglePhaseForwardOperator)."""
    return hasattr(op, "adjoint_batches_cl") and hasattr(op, "batches")


def prepare_inputs(op, queue, x0, b, weights):
    """
    The inputs of a solver's run(): returns (streamed, x0, b, weights, uploaded, host_x0, w_period).

    x0: a NumPy array or a pyopencl array (checked). A NumPy array is streamed (converted to
    C-contiguous float32 if needed, else used as is, so it is updated in place) if the operator
    supports streaming; otherwise it is uploaded, the solver runs on the GPU and copies the result
    back into host_x0 (the converted array) at the end. b and weights: NumPy arrays are converted and
    uploaded (listed in uploaded, for release after the run), pyopencl arrays are checked. Shapes
    are those of the operator's coeff_shape and data_shape where it has them.

    weights: one per segment (N_seg elements, shared by all rotations and translations) or one per
    data point (the data shape); w_period is the number of weights, so that data point i has the
    weight weights[i % w_period].
    """
    uploaded = []
    host_x0 = None
    cshape = getattr(op, "coeff_shape", None) or tuple(np.shape(x0))
    streamed = not isinstance(x0, clarray.Array) and supports_streaming(op)
    if streamed:
        x0 = as_host(x0, cshape, "x0")
    elif not isinstance(x0, clarray.Array):
        host_x0 = as_host(x0, cshape, "x0")
        x0 = clarray.to_device(queue, host_x0)
    else:
        check_device(x0, cshape, "x0")
    dshape = getattr(op, "data_shape", None) or tuple(b.shape)
    if not isinstance(b, clarray.Array):
        b = clarray.to_device(queue, as_host(b, dshape, "b"))
        uploaded.append(b)
    else:
        check_device(b, dshape, "b")
    w_period = dshape[-1]
    if weights is not None:
        n_seg, n_all = dshape[-1], int(np.prod(dshape))
        if not isinstance(weights, clarray.Array):
            w = np.ascontiguousarray(weights, dtype=np.float32)
            if w.size not in (n_seg, n_all):
                raise ValueError(f"weights must have N_seg = {n_seg} elements (one per segment) or the data "
                                 f"shape {tuple(dshape)} (one per data point), not shape {w.shape}")
            weights = clarray.to_device(queue, w.ravel())
            uploaded.append(weights)
        elif weights.dtype != np.float32 or weights.size not in (n_seg, n_all) or not weights.flags.c_contiguous:
            raise ValueError(f"weights must be C-contiguous float32 with N_seg = {n_seg} elements (one per "
                             f"segment) or {n_all} (one per data point); got shape {weights.shape}, {weights.dtype}")
        w_period = int(weights.size)
    return streamed, x0, b, weights, uploaded, host_x0, w_period
