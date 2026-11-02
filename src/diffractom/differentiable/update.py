"""Changing the PF parameters of a SinglePhaseForwardOperator in place (orientations, kernel widths, Bragg angles,
eta offset, rotation-axis tilt; definitions as in reference.ReferenceModel)."""
import numpy as np
import pyopencl.array as clarray
from scipy.spatial.transform import Rotation

from .reference import _perp_basis


def rotate_orientations(rotations, increments):
    """U_k <- exp([increment_k]x) U_k (the left increment the gradients refer to). rotations (K, 3, 3)."""
    return Rotation.from_rotvec(np.asarray(increments, np.float64)).as_matrix() @ np.asarray(rotations, np.float64)


def set_pf_parameters(op, rotations=None, sigma=None, two_theta=None, eta_offset=0.0, axis_tilt=None):
    """Give the operator new PF parameters and rebuild what depends on them (poles, kernel widths, probed
    directions, the sparse PF matrix as it was set up: stored or generated, same budget).

    rotations: (K, 3, 3) orientations (crystal to sample) of the basis; sigma: (K,) widths (radians); two_theta: (P,)
    Bragg angles of the rings (radians); eta_offset: added to the azimuths (radians); axis_tilt: (2,) rotation of the
    rotation axis about the two directions perpendicular to it (ReferenceModel.e_tilt), applied to the current axis.
    The quadrature rule is kept (it was chosen for the widths at construction). Coefficient batches may change
    (the sparse batch size depends on the fill): coefficient arrays keep their (K, Ny, Nx) shape.
    """
    _initial_geometry(op)
    pf_changed = geo_changed = False
    if rotations is not None:
        rot = np.asarray(rotations, np.float64).reshape(op.K, 3, 3)
        op.grid_rot_cpu = rot
        op.grid_inv_cpu = np.ascontiguousarray(rot.transpose(0, 2, 1).reshape(op.K, 9), dtype=np.float32)
        op.grid_inv_gpu = clarray.to_device(op.queue, op.grid_inv_cpu)
        pf_changed = True
    if sigma is not None:
        op.sigma_cpu = np.asarray(sigma, np.float32).reshape(op.K).copy()
        pf_changed = True
    if two_theta is not None:
        op._ring_two_theta = np.asarray(two_theta, np.float32).reshape(op.N_peaks).copy()
        geo_changed = True
    if eta_offset:
        d = float(eta_offset)
        op.eta_angle_range = np.asarray(op.eta_angle_range, np.float64) + d
        op.eta_angles = np.asarray(op.eta_angles) + d
        op.eta_angles_subdivided = np.asarray(op.eta_angles_subdivided) + d
        geo_changed = True
    if axis_tilt is not None and np.any(np.asarray(axis_tilt) != 0):
        k0 = np.asarray(op.cfg["k_direction_0"], np.float64)
        e1, e2 = _perp_basis(k0)
        q = Rotation.from_rotvec(axis_tilt[0] * e1 + axis_tilt[1] * e2).as_matrix()
        op.cfg = dict(op.cfg)                      # do not change the caller's dict
        op.cfg["k_direction_0"] = list(q @ k0)
        geo_changed = True
    if not (pf_changed or geo_changed):
        return op
    _rebuild(op, pf_changed, geo_changed)
    return op


def _initial_geometry(op):
    """The geometry the operator was built with (recorded on the first call): rotation axis, azimuth grids, Bragg
    angles, centre of rotation."""
    if getattr(op, "_geo_initial", None) is None:
        op._geo_initial = {"k0": np.asarray(op.cfg["k_direction_0"], np.float64).copy(),
                           "eta": tuple(np.array(a, dtype=np.float64) for a in
                                        (op.eta_angle_range, op.eta_angles, op.eta_angles_subdivided)),
                           "two_theta": np.asarray(op.two_theta_peaks, np.float64).copy(),
                           "cor": float(op.cor_offset)}
    return op._geo_initial


def set_geometry(op, eta_offset=None, axis_tilt=None, two_theta=None, cor=None, shift=None):
    """Set the operator's geometry ABSOLUTELY, relative to the geometry it was built with (for optimisers):
    eta_offset (radians) added to the initial azimuths; axis_tilt (2,): the initial rotation axis turned about its
    two perpendicular directions (ReferenceModel.e_tilt of the initial axis); two_theta (P,): Bragg angles (radians);
    cor: centre of rotation (cor_offset units); shift (N_Omega,): per-angle detector shifts (bins). None keeps a
    parameter as it is. Rebuilds the probed directions, the PF matrix and the projector as needed."""
    g0 = _initial_geometry(op)
    geo_changed = False
    if eta_offset is not None:
        d = float(eta_offset)
        op.eta_angle_range, op.eta_angles, op.eta_angles_subdivided = (a + d for a in g0["eta"])
        geo_changed = True
    if axis_tilt is not None:
        e1, e2 = _perp_basis(g0["k0"])
        q = Rotation.from_rotvec(axis_tilt[0] * e1 + axis_tilt[1] * e2).as_matrix()
        op.cfg = dict(op.cfg)
        op.cfg["k_direction_0"] = list(q @ g0["k0"])
        geo_changed = True
    if two_theta is not None:
        op._ring_two_theta = np.asarray(two_theta, np.float32).reshape(op.N_peaks).copy()
        geo_changed = True
    if geo_changed:
        _rebuild(op, False, True)
    if cor is not None or shift is not None:
        set_proj_parameters(op, cor=cor, shift=shift)
    return op


def _rebuild(op, pf_changed, geo_changed):
    """Rebuild what depends on the PF parameters: poles and widths, probed directions, the sparse PF matrix."""
    if pf_changed:
        op.transfer_pole_axes_to_gpu()             # poles, 1 / sigma^2, normalisation
    if geo_changed:
        if getattr(op, "coords_gpu", None) is not None:
            op.coords_gpu.base_data.release()
        op.detector_coordinates()
    # everything built from the PF parameters
    op.release_streamer()
    op.release_omega_subsets()
    for sb in (op.sparse_batches or []):
        for a in (sb or {}).values():
            if isinstance(a, clarray.Array) and a.base_data is not None:
                a.base_data.release()
    op.sparse_batches = None
    if op.pf_gen is not None:
        op.pf_gen.release()
        op.pf_gen = None
    op._pf_cached = False
    if op._pf_build["mode"] != "dense":
        op.build_sparse_pf(**op._pf_build)
    op._pf_version = getattr(op, "_pf_version", 0) + 1


def set_proj_parameters(op, cor=None, shift=None):
    """Give the operator (native projector) a new centre of rotation (cor, the cfg's cor_offset units) and / or a
    detector shift of every projection (shift (N_Omega,), in detector bins, added to the detector coordinate: run-out
    or drift of the rotation stage), and rebuild its projector (the PF matrix is not affected)."""
    from ..operators.parallel_radon import ParallelRadon
    if op.projector != "native":
        raise NotImplementedError("set_proj_parameters needs the native projector")
    _initial_geometry(op)
    if cor is not None:
        op.cor_offset = float(cor)
    if shift is not None:
        op.angle_shift = np.asarray(shift, np.float64).reshape(int(op.N_Omega)).copy()
    old = op.radon
    op.radon = ParallelRadon(op.queue, (op.Nx, op.Ny), op.angles, op.My, image_width=op.Nx, detector_width=op.My,
                             detector_shift=op.detector_shifts(), angle_weights=old.angle_weights, bins_per_item=3,
                             kernel=op.radon_kernel, tiled=old.modes)
    op.release_omega_subsets()
    op._pf_version = getattr(op, "_pf_version", 0) + 1
    return op
