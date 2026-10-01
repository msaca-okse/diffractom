"""Shape, dtype and layout checks for the arrays passed to the operators and solvers.

Conventions (all C-contiguous float32):
  coefficients  (K, Ny, Nx): coeffs[k] is the image of orientation k, rows y, columns x
  data          (N_Omega, My, N_seg) with N_seg = N_eta * N_rings, segment j = eta_bin * N_rings + ring
  weights       N_seg elements, e.g. shaped (N_eta, N_rings)
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


def prepare_inputs(op, queue, x0, b, weights):
    """
    The inputs of a solver's run(): returns (streamed, x0, b, weights, uploaded).

    x0: a NumPy array (streamed; converted to C-contiguous float32 if needed, else used as is,
    so it is updated in place) or a pyopencl array (checked). b and weights: NumPy arrays are
    converted and uploaded (listed in uploaded, for release after the run), pyopencl arrays are
    checked. Shapes are those of the operator's coeff_shape and data_shape where it has them.
    """
    uploaded = []
    streamed = not isinstance(x0, clarray.Array)
    cshape = getattr(op, "coeff_shape", None) or tuple(np.shape(x0))
    if streamed:
        x0 = as_host(x0, cshape, "x0")
    else:
        check_device(x0, cshape, "x0")
    dshape = getattr(op, "data_shape", None) or tuple(b.shape)
    if not isinstance(b, clarray.Array):
        b = clarray.to_device(queue, as_host(b, dshape, "b"))
        uploaded.append(b)
    else:
        check_device(b, dshape, "b")
    if weights is not None:
        n_seg = dshape[-1]
        if not isinstance(weights, clarray.Array):
            w = np.ascontiguousarray(weights, dtype=np.float32)
            if w.size != n_seg:
                raise ValueError(f"weights must have N_eta * N_rings = {n_seg} elements (one per segment), "
                                 f"not shape {w.shape}")
            weights = clarray.to_device(queue, w.ravel())
            uploaded.append(weights)
        elif weights.dtype != np.float32 or weights.size != n_seg or not weights.flags.c_contiguous:
            raise ValueError(f"weights must be C-contiguous float32 with N_eta * N_rings = {n_seg} elements, "
                             f"one per segment; got shape {weights.shape}, {weights.dtype}")
    return streamed, x0, b, weights, uploaded
