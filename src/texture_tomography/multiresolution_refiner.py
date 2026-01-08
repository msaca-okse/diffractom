from dataclasses import dataclass
from typing import Optional, List
from scipy.spatial.transform import Rotation
import numpy as np
from odftt.texture import grids, odfs, point_groups

@dataclass
class OrientationNode:
    R: Rotation                # scipy Rotation object
    level: int                 # index into sigma_levels
    sigma: float
    parent: Optional[int]      # index into nodes list
    children: List[int]
    active: bool = True
    coeff: float = 0.0
    score: float = 0.0


class OrientationTree:
    def __init__(self, sigma_levels):
        self.nodes: list[OrientationNode] = []
        self.levels: dict[int, list[int]] = {i: [] for i in range(len(sigma_levels))}
        self.sigma_levels = list(sigma_levels)

    # ------------------------------------------------------------
    # NEW: initialize from Hopf grid in fundamental zone
    # ------------------------------------------------------------
    @classmethod
    def from_hopf_fzone(
        cls,
        pg,
        grid_resolution_parameter: int,
        sigma_levels,
    ):
        
        """
        Initialize an OrientationTree from a Hopf grid
        restricted to the fundamental zone of the material.

        Parameters
        ----------
        material : Material
            Material object with crystal system information.
        grid_resolution_parameter : int
            Hopf grid resolution parameter.
        sigma_levels : list[float]
            Sigma values per refinement level. Level 0 is used here.

        Returns
        -------
        tree : OrientationTree
        """
        # --- generate Hopf grid in FZ ---
        pg_mats = pg.reshape(-1, 3, 3)
        pg_rotations = Rotation.from_matrix(pg_mats)
        rotations = grids.hopf_grid(grid_resolution_parameter, pg_rotations)

        # --- create tree ---
        tree = cls(sigma_levels=sigma_levels)

        sigma0 = sigma_levels[0]

        for R in rotations:
            node = OrientationNode(
                R=R,
                level=0,
                sigma=sigma0,
                parent=None,
                children=[],
                active=True,
            )
            idx = len(tree.nodes)
            tree.nodes.append(node)
            tree.levels[0].append(idx)

        return tree

    def leaf_nodes(self):
        return [i for i, n in enumerate(self.nodes)
                if n.active and len(n.children) == 0]

    def nodes_at_level(self, level):
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
        return [
            i for i, n in enumerate(self.nodes)
            if n.active
        ]


    def active_leaf_nodes(self):
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
        print("OrientationTree summary:")
        for lvl in sorted(self.levels):
            active, total = self.summary()[lvl]
            print(f"  level {lvl}: {active} active / {total} total")


    def inspect_node(self, idx):
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
        indices = self.levels.get(level, [])
        if active_only:
            indices = [i for i in indices if self.nodes[i].active]
        return [self.nodes[i].R for i in indices]


    def scores_at_level(self, level, active_only=True):
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







def generate_hopf_grid_fzone(material, grid_resolution_parameter, kernel_sigma):
    """
    Prepare ODF grids and GaussianRBF objects per material.

    Returns
    -------
    grid_list : list
    odf_list  : list
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


    system = material["system"].lower()

    if system not in point_group_map:
        raise ValueError(system)

    pg = point_group_map[system]
    grid = grids.hopf_grid(grid_resolution_parameter, pg)

    return grid

def invert_grid(grid):
    grid_inv_cpu = np.stack([rot.inv().as_matrix().reshape(-1) for rot in grid],axis=0).astype(np.float32)
    return grid_inv_cpu