"""The hierarchical total variation (OrientationTree, SPDHGPoisson(tv_tree=)): the nested cells, one radius == the plain TV,
and the optimum of a long run against an independent float64 Chambolle-Pock on the explicit matrix."""
import numpy as np
from scipy.spatial.transform import Rotation

from test_spdhg_tv import _cfg, _context, _dense_matrix, _grid, _material, _objective, _problem, _reference


def _local_problem(ctx, queue, K=6, n=8):
    from diffractom import LocalSinglePhaseForwardOperator, LocalSupport
    op, b, gain = _problem(ctx, queue, K=K, n=n)
    mask = np.repeat(op.support_mask()[None], K, axis=0)
    lop = LocalSinglePhaseForwardOperator(_cfg(n), _material(), _grid(K), 0.2, support=LocalSupport.from_mask(mask), normalized=True,
                                          pf_mode="sparse", split_patches=False, ctx=ctx, queue=queue)
    return lop, b, gain


def test_orientation_tree_cells():
    from diffractom import OrientationTree
    rng = np.random.default_rng(0)
    base = Rotation.random(40, random_state=1)
    R = (base * Rotation.from_rotvec(np.deg2rad(rng.normal(0, 1.0, (40, 3))))).as_matrix()
    S = Rotation.from_euler("z", np.array([[0.0], [90.0], [180.0], [270.0]]), degrees=True).as_matrix()
    # a symmetric copy of the first orientation is the same orientation: same cell at every level
    R2 = np.concatenate([R, (R[0] @ S[1])[None]])
    t = OrientationTree(R2, S, (30.0, 10.0, 3.0))
    assert [len(n) for n in t.nodes] == [41] * 3
    for level in t.nodes:
        assert level[0] == level[-1]
    # nested: a cell of a level lies in one cell of the level above
    for fine, coarse in zip(t.nodes[1:], t.nodes[:-1]):
        assert all(len(np.unique(coarse[fine == c])) == 1 for c in np.unique(fine))
    assert t.edges == [0.0, 10.0, 3.5] and t.leaf_edge == 1.5
    assert t.n_nodes[0] <= t.n_nodes[1] <= t.n_nodes[2]


def test_single_radius_is_the_plain_tv():
    from diffractom import OrientationTree, SPDHGPoisson
    ctx, queue = _context()
    lop, b, gain = _local_problem(ctx, queue)
    tree = OrientationTree(_rot(lop), None, (4.0,))
    a = SPDHGPoisson(lop, subsets=3, tv=3.0, seed=2).run(np.zeros(lop.coeff_shape, np.float32), b, 15, gain=gain, diagnostics_interval=15)
    c = SPDHGPoisson(lop, subsets=3, tv=1.5, tv_tree=tree, seed=2).run(np.zeros(lop.coeff_shape, np.float32), b, 15, gain=gain, diagnostics_interval=15)
    np.testing.assert_allclose(c, a, rtol=1e-5, atol=1e-6 * np.abs(a).max())


def _rot(lop):
    return np.asarray(lop.grid_rot_cpu)


def _tree_ops(lop, tree, free):
    """Rows of the hierarchical penalty on the flat coefficients: per level and node forward differences (right, down) of the node's sum,
    2 rows per pixel, scaled by the edge; the leaves as nodes of one orientation."""
    from scipy.sparse import coo_matrix
    k, iy, ix, pi = (np.asarray(a) for a in lop._pairs())
    n = int(lop.Nx); K = int(lop.K_basis)
    levels = [(np.arange(K), tree.leaf_edge)] + tree.levels()
    rows, cols, vals, n_rows = [], [], [], 0
    for node, edge in levels:
        cell = {}                                          # (node, iy, ix) -> flat coefficients that are free there
        for k_, y, x, j in zip(k, iy, ix, pi):
            if free[j]:
                cell.setdefault((int(node[k_]), int(y), int(x)), []).append(int(j))
        for (nd, y, x), js in cell.items():
            for c, (dy, dx) in enumerate(((0, 1), (1, 0))):
                other = cell.get((nd, y + dy, x + dx))
                if other is not None:
                    r = n_rows + 2 * ((nd * n + y) * n + x) + c
                    rows += [r] * (len(js) + len(other)); cols += js + other
                    vals += [-edge] * len(js) + [edge] * len(other)
        n_rows += 2 * (int(node.max()) + 1) * n * n
    return coo_matrix((vals, (rows, cols)), shape=(n_rows, int(lop.n_coef))).tocsr()


def test_tree_tv_matches_reference():
    from diffractom import OrientationTree, SPDHGPoisson
    ctx, queue = _context()
    lop, b, gain = _local_problem(ctx, queue)
    tree = OrientationTree(_rot(lop), None, (120.0, 60.0))
    assert tree.levels() and tree.n_nodes[1] < lop.K_basis            # a level with cells of several orientations
    n = int(lop.n_coef)
    A = np.zeros((int(np.prod(lop.data_shape)), n))
    e = np.zeros((n,), np.float32)
    for j in range(n):
        e[j] = 1.0
        A[:, j] = lop.direct(e).ravel()
        e[j] = 0.0
    free = A.sum(axis=0) > 0
    D = _tree_ops(lop, tree, free)
    bb = b.ravel().astype(np.float64)
    tv = 0.06
    x_ref = _reference(A, D, bb, gain, tv, free, n_iter=30000)
    f_ref = _objective(A, D, x_ref, bb, gain, tv)
    sp = SPDHGPoisson(lop, subsets=4, tv=tv, tv_tree=tree, seed=1)
    x = sp.run(np.zeros(lop.coeff_shape, np.float32), b, 3000, gain=gain, diagnostics_interval=3000)
    f = _objective(A, D, x.astype(np.float64), bb, gain, tv)
    assert f <= f_ref * (1 + 1e-3), (f, f_ref)
    assert abs(sp.final_stats["final_f"] - f) <= 1e-4 * f, (sp.final_stats["final_f"], f)   # the solver's objective agrees with NumPy's
