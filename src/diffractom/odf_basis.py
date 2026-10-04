"""
Orientation basis from the sample's ODF, without peak indexing.

Summed over the translations, the texture-tomography data become a bulk measurement, data = PF @ w,
where w_k is the volume of the sample with orientation k (the ODF): the operator is the texture
operator on a single pixel (1 x 1 grid, one translation), so all its PF machinery (sparse or
generated matrix, batching) applies, and the problem is small in data and coefficients. The ODF
is reconstructed coarse to fine: a uniform grid of the fundamental zone at spacing h0, then
refinements at half the spacing around the orientations with weight. A TT basis is then the
support of the fine ODF (thinned to a minimum distance) or a sample drawn from it.

Orientations are rotation matrices U (crystal -> sample frame) as Grid uses them, or their unit
quaternions (x, y, z, w); the crystal symmetry acts from the right (U and U S are the same).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pyopencl.array as clarray
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .crystallography import point_groups
from .operators.single_phase_forward_operator import SinglePhaseForwardOperator
from .optimization.fista_huber import FISTAHuber
from .optimization.streaming import estimate_L_power_streamed
from .utils.grid import Grid

POINT_GROUPS = {
    "triclinic": point_groups.trivial,
    "monoclinic": point_groups.cyclic_2,
    "orthorhombic": point_groups.orthorhombic,
    "tetragonal": point_groups.tetragonal,
    "trigonal": point_groups.trigonal,
    "hexagonal": point_groups.hexagonal,
    "cubic": point_groups.cubic,
}


# --------------------------------------------------------------------------- symmetry and distances

def symmetry_rotations(symmetry) -> Rotation:
    """The proper rotations of the point group: from a crystal system name ("cubic", ...), a Material
    (its point_group_matrices) or a Rotation."""
    if isinstance(symmetry, Rotation):
        return symmetry
    if isinstance(symmetry, str):
        return Rotation.concatenate(POINT_GROUPS[symmetry])
    mats = getattr(symmetry, "point_group_matrices", None)
    if mats is None:
        raise ValueError("symmetry: a crystal system name, a Material with a point group or a Rotation")
    return Rotation.from_matrix(np.asarray(mats, dtype=np.float64).reshape(-1, 3, 3))


def chord(theta_deg):
    """Quaternion distance |q1 - q2| of two rotations theta apart: 2 sin(theta / 4)."""
    return 2.0 * np.sin(np.deg2rad(theta_deg) / 4.0)


def chord_to_deg(d):
    return np.degrees(4.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0)))


def equivalents(q, sym: Rotation):
    """(2 G N, 4): the symmetric equivalents of q (N, 4), both signs; row j belongs to q[j % N]."""
    R = Rotation.from_quat(q)
    out = np.concatenate([(R * s).as_quat() for s in sym])
    return np.concatenate([out, -out])


def to_fundamental_zone(q, sym: Rotation):
    """The equivalent of each orientation with the smallest rotation angle (largest |w|), w >= 0."""
    R = Rotation.from_quat(q)
    eq = np.stack([(R * s).as_quat() for s in sym], axis=1)  # (N, G, 4)
    out = eq[np.arange(len(q)), np.argmax(np.abs(eq[..., 3]), axis=1)]
    return out * np.where(out[:, 3:4] < 0, -1.0, 1.0)


def thin(q, min_distance_deg, sym: Rotation, order=None):
    """Greedy thinning with symmetry: visit q in `order` (default: as given) and keep an orientation
    unless it is within min_distance of one already kept. Returns the kept indices."""
    N = len(q)
    order = np.arange(N) if order is None else np.asarray(order)
    tree = cKDTree(equivalents(q, sym))
    used = np.zeros(N, bool)
    keep = []
    r = chord(min_distance_deg)
    for i in order:
        if used[i]:
            continue
        keep.append(i)
        used[np.asarray(tree.query_ball_point(q[i], r), dtype=np.int64) % N] = True
    return np.asarray(keep, dtype=np.int64)


def nearest_misorientation_deg(q_query, q_set, sym: Rotation):
    """Misorientation (deg) from every q_query to the nearest orientation of q_set."""
    d, _ = cKDTree(equivalents(q_set, sym)).query(q_query)
    return chord_to_deg(d)


def uniform_fundamental_zone(spacing_deg, symmetry, method="cubochoric", seed=0):
    """Near-uniform orientations of the fundamental zone, as quaternions (N, 4).
    method "cubochoric" (orix.sampling.get_sample_fundamental; its `resolution` = spacing_deg) or
    "random" (uniform random rotations, mapped into the zone, thinned to 0.7 spacing: no orix needed)."""
    sym = symmetry_rotations(symmetry)
    if method == "cubochoric":
        from orix.sampling import get_sample_fundamental
        r = get_sample_fundamental(resolution=spacing_deg, point_group=_orix_group(len(sym)), method="cubochoric")
        return np.ascontiguousarray(r.data[:, [1, 2, 3, 0]])
    if method == "random":
        n = int(3 * 8 * np.pi ** 2 / np.deg2rad(spacing_deg) ** 3 / len(sym))
        q = to_fundamental_zone(Rotation.random(n, random_state=seed).as_quat(), sym)
        return q[thin(q, 0.7 * spacing_deg, sym)]
    raise ValueError(f"unknown method {method!r}")


def _orix_group(order):
    """orix's proper point group with this many rotations (1, 2, 222, 32, 422, 622, 432)."""
    from orix.quaternion import symmetry as S
    groups = {1: S.C1, 2: S.C2, 4: S.D2, 6: S.D3, 8: S.D4, 12: S.D6, 24: S.O}
    if order not in groups:
        raise ValueError(f"no cubochoric sampling for a point group of order {order}; use method='random'")
    return groups[order]


# --------------------------------------------------------------------------- the bulk problem

def bulk_data(data):
    """The data summed over translations: (N_Omega, My, N_seg) -> (N_Omega, 1, N_seg)."""
    return np.ascontiguousarray(np.asarray(data, dtype=np.float32).sum(axis=1, keepdims=True))


def bulk_operator(cfg, material, orientations, sigma_deg, max_gb=4.0, sparse_batch_max=65536, **kwargs):
    """The texture operator on one pixel and one translation (bulk data = PF @ w) for the given
    orientations (rotation matrices (K, 3, 3) or quaternions (K, 4)); kwargs go to
    SinglePhaseForwardOperator (normalized=..., ctx=..., queue=...). With one pixel a batch of
    orientations is tiny, so the sparse batches are made much larger than for TT (sparse_batch_max)."""
    o = np.asarray(orientations)
    mats = Rotation.from_quat(o).as_matrix() if o.shape[-1] == 4 else o
    grid = Grid.from_rotation_matrices(mats, np.deg2rad(sigma_deg))
    return SinglePhaseForwardOperator(cfg={**cfg, "Nx": 1, "Ny": 1, "My": 1}, material=material, grid=grid,
                                      max_gb=max_gb, sparse_batch_max=sparse_batch_max, **kwargs)


@dataclass
class ODFLevel:
    """One level of the coarse-to-fine ODF: orientations (quaternions), their weights w (the bulk
    coefficients: volume per basis function), the grid spacing and kernel width (deg)."""
    q: np.ndarray
    w: np.ndarray
    spacing_deg: float
    sigma_deg: float
    residual: float  # || w (A x - b) || / || w b || of the bulk fit
    seconds: float
    objective: list = field(default_factory=list)

    @property
    def matrices(self):
        return Rotation.from_quat(self.q).as_matrix()

    def support(self, rel=1e-2):
        """Indices of the orientations with weight above rel * max, by decreasing weight."""
        s = np.flatnonzero(self.w > rel * self.w.max())
        return s[np.argsort(-self.w[s], kind="stable")]


def _children(q, spacing_deg, n_sub, sym):
    """A local n_sub^3 lattice of rotation-vector offsets at half the spacing around each orientation
    (n_sub = 4: offsets {-3/4, -1/4, 1/4, 3/4} spacing, the parent's cell and half of each neighbour;
    n_sub = 2: {-1/4, 1/4} spacing), mapped into the fundamental zone. Row p * n_sub^3 + o: parent p."""
    h = np.deg2rad(spacing_deg)
    t = (np.arange(n_sub) - (n_sub - 1) / 2) * h / 2
    off = Rotation.from_rotvec(np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3))
    R = Rotation.from_quat(q)
    kids = np.stack([(R * o).as_quat() for o in off], axis=1).reshape(-1, 4)
    return to_fundamental_zone(kids, sym)


def reconstruct_odf(cfg, material, data, weights=None, spacing_deg=4.0, levels=5, sigma_factor=0.6,
                    niter=200, niter_first=50, keep_rel=1e-3, n_sub=4, grid_method="cubochoric",
                    huber_delta=1e30, max_gb=4.0, normalized=True, ctx=None, queue=None, verbose=1):
    """
    Coarse-to-fine ODF from the data summed over translations.

    Level 0: a uniform grid of the fundamental zone at `spacing_deg` (uniform_fundamental_zone); every
    further level halves the spacing around the orientations whose weight exceeds keep_rel * max
    (children of the strongest parents first, thinned to 0.7 x the new spacing). Each level is a
    nonnegative FISTA fit (least squares with the default huber_delta) of the bulk data with kernels
    of width sigma_factor x spacing; the first level runs niter_first iterations (it only has to
    locate the support), the others niter.

    data: (N_Omega, My, N_seg) as for SinglePhaseForwardOperator (or already summed, My = 1);
    weights: one per segment, as FISTAHuber.run. Returns a list of ODFLevel, coarse to fine.
    """
    sym = symmetry_rotations(material)
    b = bulk_data(data)
    wseg = None if weights is None else np.asarray(weights, np.float32).ravel()
    wb = b[:, 0] * (1.0 if wseg is None else wseg)
    bnorm = float(np.linalg.norm(wb))
    q = uniform_fundamental_zone(spacing_deg, sym, method=grid_method)
    h = float(spacing_deg)
    out = []
    for lev in range(levels):
        sigma = sigma_factor * h
        t0 = time.perf_counter()
        op = bulk_operator(cfg, material, q, sigma, max_gb=max_gb, normalized=normalized, ctx=ctx, queue=queue)
        ctx, queue = op.ctx, op.queue
        L = 1.1 * estimate_L_power_streamed(op, niter=6, seed=0, verbose=0)
        solver = FISTAHuber(op, prox_kind="nonneg", L=L, huber_delta=huber_delta)
        # the coefficients (K floats) and the bulk data are small: both stay on the GPU
        x = clarray.zeros(queue, op.coeff_shape, np.float32)
        solver.run(x, clarray.to_device(queue, b), niter=niter_first if lev == 0 else niter,
                   weights=None if wseg is None else clarray.to_device(queue, wseg))
        x = x.get()
        r = op.direct(x)[:, 0] - b[:, 0]
        if wseg is not None:
            r *= wseg
        level = ODFLevel(q=q, w=x.ravel().copy(), spacing_deg=h, sigma_deg=sigma,
                         residual=float(np.linalg.norm(r)) / bnorm, seconds=time.perf_counter() - t0,
                         objective=[s["f"] for s in solver.iter_stats])
        op.free_memory()
        out.append(level)
        if verbose:
            print(f"ODF level {lev}: spacing {h:.3f} deg, sigma {sigma:.3f} deg, K {len(q)}, PF {op.pf_mode}, "
                  f"{level.seconds:.0f} s, relative residual {level.residual:.4f}", flush=True)
        if lev == levels - 1:
            break
        keep = level.support(keep_rel)  # by decreasing weight
        n3 = n_sub ** 3
        kids = _children(q[keep], h, n_sub, sym)
        sel = thin(kids, 0.7 * h / 2, sym)  # rows are parent-major: strong parents' children first
        if verbose:
            print(f"  {len(keep)} orientations above {keep_rel:g} x max ({level.w[keep].sum() / level.w.sum():.4f} "
                  f"of the weight) -> {len(keep) * n3} children -> {len(sel)} after thinning", flush=True)
        q = kids[sel]
        h /= 2
    return out


# --------------------------------------------------------------------------- the TT basis

def basis_from_odf(level: ODFLevel, symmetry, rel=1e-2, min_distance_deg=0.2):
    """The support of an ODF level (weight above rel * max), thinned to min_distance in order of
    decreasing weight: rotation matrices (K, 3, 3)."""
    s = level.support(rel)
    keep = s[thin(level.q[s], min_distance_deg, symmetry_rotations(symmetry))]
    return Rotation.from_quat(level.q[keep]).as_matrix()


def sample_from_odf(level: ODFLevel, symmetry, n, min_distance_deg=0.2, alpha=1.0, seed=0):
    """n orientations drawn from the ODF level (probability ~ w^alpha; alpha < 1 favours weak
    components), each jittered by a Gaussian of the level's kernel width, thinned to min_distance:
    rotation matrices (K <= n, 3, 3)."""
    sym = symmetry_rotations(symmetry)
    rng = np.random.default_rng(seed)
    p = np.maximum(level.w, 0.0) ** alpha
    idx = rng.choice(len(level.q), n, p=p / p.sum())
    jitter = Rotation.from_rotvec(rng.normal(scale=np.deg2rad(level.sigma_deg), size=(n, 3)))
    q = to_fundamental_zone((Rotation.from_quat(level.q[idx]) * jitter).as_quat(), sym)
    return Rotation.from_quat(q[thin(q, min_distance_deg, sym)]).as_matrix()
