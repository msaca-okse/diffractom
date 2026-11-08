"""Nested cells of an orientation basis, for the hierarchical total variation of SPDHGPoisson (tv_tree=).

Why: the total variation of every orientation's image separately charges a change of orientation between neighbouring
voxels as two edges, whatever the angle: a smooth gradient inside a grain (weight moving to the next kernel, 0.5 deg
away) costs the same as a grain boundary, and it knows nothing about which kernels are close.

The hierarchical penalty charges the mass that has to move in orientation space. The basis is cut into nested cells of
radius r_0 > r_1 > ... > r_(L-1) (greedy cover, symmetry respected). A node's sum s_n(j) adds the coefficients of its
orientations in voxel j, and

    R(x) = tv sum_nodes e_n sum_j |grad s_n(j)|,

e_n = the length of the edge above the node in the tree: e = r_(L-1) / 2 for the leaves (single orientations) and
(r_(l-1) - r_l) / 2 for a cell of level l >= 1. The cells of level 0 have no parent (e = 0, unless top_edge is set),
which caps the penalty: moving a unit of mass over a voxel edge between orientations whose smallest common cell is of
level l costs about r_l, and any two orientations in different cells of level 0 cost r_0, however far apart they are.
That is the tree-Wasserstein distance between the two voxels' orientation distributions, so tv has the units of the
data term per coefficient unit and per degree.

Radii are in degrees. A single radius (r,) gives the plain total variation with weight tv r / 2."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def _chord(angle_deg):
    """Euclidean distance of two unit quaternions whose rotations differ by angle_deg."""
    return 2.0 * np.sin(np.deg2rad(angle_deg) / 4.0)


def _cover(q, eq, ids, radius):
    """Labels 0..m-1 of the orientations ids: a greedy cover by balls of the radius, every member then taken by its
    nearest ball centre. q: (K, 4) unit quaternions; eq: (K, S, 4) their symmetric copies."""
    S = eq.shape[1]
    copies = eq[ids].reshape(-1, 4)
    owner = np.tile(np.repeat(np.arange(len(ids)), S), 2)
    tree = cKDTree(np.concatenate([copies, -copies]))
    free = np.ones(len(ids), bool)
    seeds = []
    chord = _chord(radius)
    for i in range(len(ids)):
        if free[i]:
            seeds.append(i)
            free[owner[tree.query_ball_point(q[ids[i]], chord)]] = False
    seeds = np.asarray(seeds)
    sc = eq[ids[seeds]].reshape(-1, 4)
    sown = np.tile(np.repeat(np.arange(len(seeds)), S), 2)
    _, nearest = cKDTree(np.concatenate([sc, -sc])).query(q[ids])
    return sown[nearest], len(seeds)


class OrientationTree:
    """rotations: (K, 3, 3) crystal-to-sample matrices of the basis, in the grid's order; symmetry: (S, 3, 3) rotations
    with U S equivalent to U (None: no symmetry); radii_deg: the cell radii, coarsest first, strictly decreasing;
    top_edge: edge length (deg) above the cells of level 0 (0: none, the cap of the module docstring).

    Attributes: nodes (list of (K,) int arrays, per level: the orientation's cell), edges (per level), leaf_edge,
    n_nodes (per level)."""

    def __init__(self, rotations, symmetry, radii_deg, top_edge=0.0):
        R = np.asarray(rotations, np.float64)
        if R.ndim != 3 or R.shape[1:] != (3, 3):
            raise ValueError("rotations must have shape (K, 3, 3)")
        radii = [float(r) for r in radii_deg]
        if not radii or any(r <= 0 for r in radii) or any(a <= b for a, b in zip(radii, radii[1:])):
            raise ValueError("radii_deg must be positive and strictly decreasing (coarsest first)")
        S = np.eye(3)[None] if symmetry is None else np.asarray(symmetry, np.float64)
        K = len(R)
        q = Rotation.from_matrix(R).as_quat()
        eq = Rotation.from_matrix((R[:, None] @ S[None]).reshape(-1, 3, 3)).as_quat().reshape(K, len(S), 4)
        self.K = K
        self.radii = radii
        self.nodes, self.n_nodes = [], []
        parent = np.zeros(K, np.int64)
        for r in radii:
            node = np.empty(K, np.int64)
            total = 0
            for p in np.unique(parent):
                ids = np.nonzero(parent == p)[0]
                lab, m = _cover(q, eq, ids, r)
                node[ids] = lab + total
                total += m
            self.nodes.append(node)
            self.n_nodes.append(total)
            parent = node
        self.edges = [float(top_edge)] + [(a - b) / 2.0 for a, b in zip(radii, radii[1:])]
        self.leaf_edge = radii[-1] / 2.0

    def levels(self):
        """(node array, edge) of the levels that carry a penalty, coarsest first (the leaves are not among them)."""
        return [(n, e) for n, e in zip(self.nodes, self.edges) if e > 0]
