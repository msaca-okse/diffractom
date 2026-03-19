from dataclasses import dataclass
from typing import Optional, List
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
import numpy as np

from diffractom.crystallography import point_groups

@dataclass
class OrientationNode:
    """Single node in the orientation tree, holding a rotation and metadata."""
    R: Rotation                # scipy Rotation object
    level: int                 # index into sigma_levels
    sigma: float
    parent: Optional[int]      # index into nodes list
    children: List[int]
    active: bool = True
    coeff: float = 0.0
    score: float = 0.0


class OrientationTree:
    """Hierarchical orientation grid with multi-resolution refinement."""

    def __init__(self, sigma_levels):
        """Create tree with the given sigma levels (one per refinement depth)."""
        self.nodes: list[OrientationNode] = []
        self.levels: dict[int, list[int]] = {i: [] for i in range(len(sigma_levels))}
        self.sigma_levels = list(sigma_levels)

    def leaf_nodes(self):
        """Indices of active nodes with no children."""
        return [i for i, n in enumerate(self.nodes)
                if n.active and len(n.children) == 0]

    def nodes_at_level(self, level):
        """Indices of active nodes at a given level."""
        return [i for i in self.levels[level] if self.nodes[i].active]

    # ------------------------------------------------------------

    def generate_children(
        self,
        parents,
        radius,
        stencil: int = 6,
    ):
        """
        Generate children for one or multiple parent nodes.

        Parameters
        ----------
        parents : int or list[int]
            Parent node index or list of indices.
        radius : float
            Radius in SO(3) (rotvec norm) for child generation.
        stencil : int
            One of {6, 12, 26}. Default: 6.

        Returns
        -------
        new_indices : list[int]
            Indices of newly created child nodes.
        """

        if isinstance(parents, int):
            parents = [parents]

        # --- choose stencil offsets in R^3 ---
        d = radius

        if stencil == 6:
            offsets = [
                (+d, 0, 0), (-d, 0, 0),
                (0, +d, 0), (0, -d, 0),
                (0, 0, +d), (0, 0, -d),
            ]

        elif stencil == 12:
            offsets = []
            for s1 in (+1, -1):
                for s2 in (+1, -1):
                    offsets += [
                        (s1*d, s2*d, 0),
                        (s1*d, 0, s2*d),
                        (0, s1*d, s2*d),
                    ]

        elif stencil == 26:
            offsets = []
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    for k in (-1, 0, 1):
                        if i == j == k == 0:
                            continue
                        offsets.append((i*d, j*d, k*d))
        else:
            raise ValueError("stencil must be one of {6, 12, 26}")

        offsets = np.array(offsets)

        new_indices = []

        for p_idx in parents:
            parent = self.nodes[p_idx]
            next_level = parent.level + 1

            if next_level >= len(self.sigma_levels):
                continue

            sigma_new = self.sigma_levels[next_level]

            for omega in offsets:
                R_child = parent.R * Rotation.from_rotvec(omega)

                child = OrientationNode(
                    R=R_child,
                    level=next_level,
                    sigma=sigma_new,
                    parent=p_idx,
                    children=[]
                )

                idx = len(self.nodes)
                self.nodes.append(child)
                self.levels[next_level].append(idx)
                parent.children.append(idx)
                new_indices.append(idx)

        return new_indices
    
    def active_nodes(self):
        """Indices of all active nodes."""
        return [
            i for i, n in enumerate(self.nodes)
            if n.active
        ]


    def active_leaf_nodes(self):
        """Indices of active leaf nodes (no children)."""
        return [
            i for i, n in enumerate(self.nodes)
            if n.active and len(n.children) == 0
        ]
    
    def active_nodes_at_level(self, level: int):
        """Indices of active nodes at a given level."""
        return [
            i for i in self.levels.get(level, [])
            if self.nodes[i].active
        ]
    

    def active_leaf_nodes_at_level(self, level: int):
        """Indices of active leaf nodes at a given level."""
        return [
            i for i in self.levels.get(level, [])
            if self.nodes[i].active and len(self.nodes[i].children) == 0
        ]
    
    def summary(self):
        """
        Returns a dict: level -> (n_active, n_total)
        """
        out = {}
        for lvl, indices in self.levels.items():
            total = len(indices)
            active = sum(self.nodes[i].active for i in indices)
            out[lvl] = (active, total)
        return out
    
    def print_summary(self):
        """Pretty-print level-wise active/total node counts."""
        print("OrientationTree summary:")
        for lvl in sorted(self.levels):
            active, total = self.summary()[lvl]
            print(f"  level {lvl}: {active} active / {total} total")


    def inspect_node(self, idx):
        """Return a dict with full info about node *idx*."""
        n = self.nodes[idx]
        return {
            "index": idx,
            "level": n.level,
            "active": n.active,
            "sigma": n.sigma,
            "score": n.score,
            "parent": n.parent,
            "children": n.children,
            "rotation_matrix": n.R.as_matrix(),
            "rotvec": n.R.as_rotvec(),
        }

    def rotations_at_level(self, level, active_only=True):
        """List of Rotation objects for nodes at *level*."""
        indices = self.levels.get(level, [])
        if active_only:
            indices = [i for i in indices if self.nodes[i].active]
        return [self.nodes[i].R for i in indices]

    def active_rotations(self, level, active_only=True):
        """Rotations for all active leaf nodes."""
        indices = self.active_leaf_nodes()
        return [self.nodes[i].R for i in indices]


    def scores_at_level(self, level, active_only=True):
        """Score array for nodes at *level*."""
        indices = self.levels.get(level, [])
        if active_only:
            indices = [i for i in indices if self.nodes[i].active]
        return np.array([self.nodes[i].score for i in indices])

    @classmethod
    def from_rotation_matrices(
        cls,
        R_mats: np.ndarray,
        sigma: float,
    ):
        """
        Initialize an OrientationTree from a given set of orientation matrices.

        Parameters
        ----------
        R_mats : (N, 3, 3) ndarray
            Rotation matrices.
        sigma : float
            Sigma value assigned to all orientations (level 0).

        Returns
        -------
        tree : OrientationTree
        """
        if R_mats.ndim != 3 or R_mats.shape[1:] != (3, 3):
            raise ValueError("R_mats must have shape (N, 3, 3)")

        rotations = Rotation.from_matrix(R_mats)

        tree = cls(sigma_levels=[sigma])

        for R in rotations:
            node = OrientationNode(
                R=R,
                level=0,
                sigma=sigma,
                parent=None,
                children=[],
                active=True,
            )
            idx = len(tree.nodes)
            tree.nodes.append(node)
            tree.levels[0].append(idx)

        return tree

    @classmethod
    def from_random_fundamental_zone(
        cls,
        n_orientations: int,
        symmetry: str,
        sigma: float,
    ):
        """
        Initialize an OrientationTree with randomly sampled orientations
        mapped into the fundamental zone of the given crystal symmetry.

        This creates a single-level tree (no children).

        Parameters
        ----------
        n_orientations : int
            Number of random orientations to generate.
        symmetry : str
            Crystal symmetry name.  One of ``"triclinic"``, ``"monoclinic"``,
            ``"orthorhombic"``, ``"tetragonal"``, ``"trigonal"``,
            ``"hexagonal"``, ``"cubic"``.
        sigma : float
            Sigma value assigned to all orientations (level 0).

        Returns
        -------
        tree : OrientationTree
        """
        point_group_map = {
            "triclinic": point_groups.trivial,
            "monoclinic": point_groups.cyclic_2,
            "orthorhombic": point_groups.orthorhombic,
            "tetragonal": point_groups.tetragonal,
            "trigonal": point_groups.trigonal,
            "hexagonal": point_groups.hexagonal,
            "cubic": point_groups.cubic,
        }

        if symmetry not in point_group_map:
            raise ValueError(
                f"Unknown symmetry '{symmetry}'. "
                f"Must be one of {list(point_group_map.keys())}"
            )

        pg_elements = point_group_map[symmetry]

        # Point-group symmetry matrices: (G, 3, 3)
        pg_mats = np.stack([g.as_matrix() for g in pg_elements], axis=0)

        # Random orientations: (N, 3, 3)
        R_random = Rotation.random(n_orientations).as_matrix()

        # For each orientation, apply every point-group element and keep
        # the product closest to the identity (i.e. largest trace).
        #   products[n, g] = R_random[n] @ pg_mats[g]   -> shape (N, G, 3, 3)
        products = np.einsum('nij,gjk->ngik', R_random, pg_mats)

        # Trace of each product: (N, G)
        traces = np.einsum('ngii->ng', products)

        # Index of the best symmetry element per orientation
        best_g = np.argmax(traces, axis=1)  # (N,)

        # Gather the best-projected matrices: (N, 3, 3)
        R_fz = products[np.arange(n_orientations), best_g]

        # Build scipy Rotation objects from the (N, 3, 3) stack
        rotations = Rotation.from_matrix(R_fz)

        tree = cls(sigma_levels=[sigma])

        for i, R in enumerate(rotations):
            node = OrientationNode(
                R=R,
                level=0,
                sigma=sigma,
                parent=None,
                children=[],
                active=True,
            )
            idx = len(tree.nodes)
            tree.nodes.append(node)
            tree.levels[0].append(idx)

        return tree

    def plot_stereographic(
        self,
        level: int = 0,
        direction: np.ndarray = None,
        symmetry: str = None,
        ax=None,
        **scatter_kw,
    ):
        """Plot a stereographic projection of orientations at a given level.

        For each active rotation *R* the crystal direction
        ``d = R^T @ direction`` (inverse pole figure convention) is computed,
        folded to the upper hemisphere, and projected stereographically onto
        the equatorial plane.

        When *symmetry* is given, every orientation is expanded by all
        point-group operations so that the projection shows the full
        coverage of the grid (not just the fundamental-zone representatives).

        Parameters
        ----------
        level : int
            Tree level to visualise (default 0).
        direction : (3,) array-like, optional
            Sample direction to project.  Default ``[0, 0, 1]`` (Z).
        symmetry : str, optional
            Crystal symmetry name (e.g. ``"cubic"``, ``"hexagonal"``).
            When provided, each orientation is expanded by the
            corresponding point-group operations before projection.
        ax : matplotlib Axes, optional
            If *None* a new figure is created.
        **scatter_kw
            Extra keyword arguments forwarded to ``ax.scatter``.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle as _Circle

        if direction is None:
            direction = np.array([0.0, 0.0, 1.0])
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)

        # Gather rotations
        rotations = self.rotations_at_level(level, active_only=True)
        if not rotations:
            raise ValueError(f"No active rotations at level {level}")

        R_concat = Rotation.concatenate(rotations)
        mats = R_concat.as_matrix()                          # (K, 3, 3)

        # Optionally expand by symmetry operations → (K*G, 3, 3)
        if symmetry is not None:
            sym_map = {
                "triclinic": point_groups.trivial,
                "monoclinic": point_groups.cyclic_2,
                "orthorhombic": point_groups.orthorhombic,
                "tetragonal": point_groups.tetragonal,
                "trigonal": point_groups.trigonal,
                "hexagonal": point_groups.hexagonal,
                "cubic": point_groups.cubic,
            }
            if symmetry not in sym_map:
                raise ValueError(
                    f"Unknown symmetry '{symmetry}'. "
                    f"Must be one of {list(sym_map.keys())}"
                )
            pg_ops = np.stack(
                [g.as_matrix() for g in sym_map[symmetry]], axis=0
            )  # (G, 3, 3)
            # Expand: equiv[k, g] = pg_ops[g] @ mats[k]
            expanded = np.einsum('gij,kjl->kgil', pg_ops, mats)  # (K, G, 3, 3)
            mats = expanded.reshape(-1, 3, 3)                    # (K*G, 3, 3)

        # Crystal directions: d_i = R_i^T @ direction  (IPF convention)
        dirs = mats.transpose(0, 2, 1) @ direction           # (N, 3)

        # Fold to upper hemisphere
        dirs[dirs[:, 2] < 0] *= -1

        # Stereographic projection  (project from south pole)
        denom = 1.0 + dirs[:, 2]
        denom = np.where(denom < 1e-12, 1e-12, denom)
        X = dirs[:, 0] / denom
        Y = dirs[:, 1] / denom

        # --- Plot ---
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(6, 6))

        scatter_defaults = dict(s=4, alpha=0.6, edgecolors="none")
        scatter_defaults.update(scatter_kw)
        ax.scatter(X, Y, **scatter_defaults)

        # Unit circle boundary
        ax.add_patch(_Circle((0, 0), 1.0, fill=False, color="k", lw=0.8))

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")

        if created:
            n = len(rotations)
            dir_lbl = {(1,0,0): "X", (0,1,0): "Y", (0,0,1): "Z"}.get(
                tuple(int(x) for x in direction), str(direction))
            ax.set_title(
                f"Stereographic — IPF {dir_lbl}  "
                f"(n={n}{f', sym={symmetry}' if symmetry else ''})"
            )
            plt.tight_layout()
            plt.show()

        return ax

    def prune_close_orientations(
        self,
        theta_deg: float,
        target: Optional[int] = None,
    ):
        """
        Remove nodes whose orientations are too close to an already-kept
        node.  Uses a greedy strategy with a KDTree on unit quaternions for
        fast neighbour look-ups.

        Pruned nodes are deleted from the tree and all indices (node list,
        level lists, parent/children references) are rebuilt so that the
        tree is compact afterwards.

        Parameters
        ----------
        theta_deg : float
            Minimum angular separation (in degrees) between kept
            orientations.  Pairs closer than this are pruned.
        target : int, optional
            If given, stop once this many nodes have been kept.

        Returns
        -------
        n_kept : int
            Number of nodes remaining after pruning.
        """
        if len(self.nodes) == 0:
            return 0

        # --- build quaternion array for all nodes ---
        R_all = Rotation.concatenate([n.R for n in self.nodes])
        q = R_all.as_quat()                          # (N, 4)  [x,y,z,w]
        q /= np.linalg.norm(q, axis=1, keepdims=True)

        # Antipodal equivalence: q and -q represent the same rotation,
        # so we include both copies and query against a single copy.
        q_full = np.vstack([q, -q])                   # (2N, 4)

        # Quaternion Euclidean distance for misorientation angle θ:
        #   d = 2 sin(θ / 4)
        radius = 2.0 * np.sin(np.deg2rad(theta_deg) / 4.0)

        kd = KDTree(q_full)

        N = len(q)
        used = np.zeros(2 * N, dtype=bool)
        keep_mask = np.zeros(N, dtype=bool)

        kept_count = 0
        for i in range(N):
            if used[i]:
                continue

            keep_mask[i] = True
            kept_count += 1

            # Mark all neighbours (including antipodal copies) as used
            nbrs = kd.query_ball_point(q[i], r=radius)
            used[nbrs] = True

            if target is not None and kept_count >= target:
                break

        # --- rebuild the tree keeping only the surviving nodes ---
        old_nodes = [self.nodes[i] for i in range(N) if keep_mask[i]]

        # Build old-index -> new-index map
        old_to_new = {}
        new_idx = 0
        for i in range(N):
            if keep_mask[i]:
                old_to_new[i] = new_idx
                new_idx += 1

        # Rewrite parent / children references
        for node in old_nodes:
            node.parent = old_to_new.get(node.parent) if node.parent is not None else None
            node.children = [
                old_to_new[c] for c in node.children if c in old_to_new
            ]

        self.nodes = old_nodes

        # Rebuild level index
        for lvl in self.levels:
            self.levels[lvl] = []
        for i, node in enumerate(self.nodes):
            lvl = node.level
            if lvl in self.levels:
                self.levels[lvl].append(i)

        return len(self.nodes)


def invert_grid(grid):
    """Return flattened inverse rotation matrices (K, 9) for a list of Rotations."""
    grid_inv_cpu = np.stack([rot.inv().as_matrix().reshape(-1) for rot in grid],axis=0).astype(np.float32)
    return grid_inv_cpu
