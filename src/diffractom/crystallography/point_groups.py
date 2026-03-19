from scipy.spatial.transform import Rotation as R
import numpy as np

""" Tuples of `scipy.spatial.transform.Rotation` objects correspoinding to the different
enantiomorthic space groups.

Be carefull, especially when working with monoclinic and rhombohedral lattices, that the
rotation axes are correclty oriented. (note that monoclinic usually has the rotation along 'b'
but this document has the rotation along 'z'). The lattice matrix convention in
`odftt.crystallography.crystal_families' should always be constistent with the choices
in this document.
"""

# C1
trivial = (
    R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # I
)

# C2
cyclic_2 = (
    R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # I
    R.from_matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # 180 deg [001]
)

# D2
orthorhombic = (
    R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # I
    R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),  # 180 deg [100]
    R.from_matrix([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # 180 deg [010]
    R.from_matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # 180 deg [001]
)

# C3
cyclic_3 = (
    R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # I
    R.from_matrix([[-0.5, np.sqrt(3)/2, 0], [-np.sqrt(3)/2, -0.5, 0], [0, 0, 1]]),  # 120 deg [001]
    R.from_matrix([[-0.5, -np.sqrt(3)/2, 0], [np.sqrt(3)/2, -0.5, 0], [0, 0, 1]]),  # 240 deg [001]
)

# D3
trigonal = (
    R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # I
    R.from_matrix([[-0.5, np.sqrt(3)/2, 0], [-np.sqrt(3)/2, -0.5, 0], [0, 0, 1]]),  # 120 deg [001]
    R.from_matrix([[-0.5, -np.sqrt(3)/2, 0], [np.sqrt(3)/2, -0.5, 0], [0, 0, 1]]),  # 240 deg [001]

    R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),  # 180 deg [100]
    R.from_matrix([[-0.5, np.sqrt(3)/2, 0], [np.sqrt(3)/2, 0.5, 0], [0, 0, -1]]),  # 180 deg [1, sqrt(3), 0]
    R.from_matrix([[-0.5, -np.sqrt(3)/2, 0], [-np.sqrt(3)/2, 0.5, 0], [0, 0, -1]]),
    # 180 deg [1, -sqrt(3), 0]
)

# C4
cyclic_4 = (
    R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # I
    R.from_matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),  # 90 deg [001]
    R.from_matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # 180 deg [001]
    R.from_matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),  # 270 deg [001]
)

# D4
tetragonal = (
    R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # I
    R.from_matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),  # 90 deg [001]
    R.from_matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # 180 deg [001]
    R.from_matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),  # 270 deg [001]

    R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),  # 180 deg [100]
    R.from_matrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),  # 180 deg [110]
    R.from_matrix([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # 180 deg [010]
    R.from_matrix([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),  # 180 deg [1-10]
)

# C6
cyclic_6 = (
    R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # I
    R.from_matrix([[0.5, np.sqrt(3)/2, 0], [-np.sqrt(3)/2, 0.5, 0], [0, 0, 1]]),  # 60 deg [001]
    R.from_matrix([[-0.5, np.sqrt(3)/2, 0], [-np.sqrt(3)/2, -0.5, 0], [0, 0, 1]]),  # 120 deg [001]
    R.from_matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # 180 deg [001]
    R.from_matrix([[-0.5, -np.sqrt(3)/2, 0], [np.sqrt(3)/2, -0.5, 0], [0, 0, 1]]),  # 240 deg [001]
    R.from_matrix([[0.5, -np.sqrt(3)/2, 0], [np.sqrt(3)/2, 0.5, 0], [0, 0, 1]]),  # 300 deg [001]
)

# D6
hexagonal = (
    R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # I
    R.from_matrix([[0.5, np.sqrt(3)/2, 0], [-np.sqrt(3)/2, 0.5, 0], [0, 0, 1]]),  # 60 deg [001]
    R.from_matrix([[-0.5, np.sqrt(3)/2, 0], [-np.sqrt(3)/2, -0.5, 0], [0, 0, 1]]),  # 120 deg [001]
    R.from_matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # 180 deg [001]
    R.from_matrix([[-0.5, -np.sqrt(3)/2, 0], [np.sqrt(3)/2, -0.5, 0], [0, 0, 1]]),  # 240 deg [001]
    R.from_matrix([[0.5, -np.sqrt(3)/2, 0], [np.sqrt(3)/2, 0.5, 0], [0, 0, 1]]),  # 300 deg [001]

    R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),  # 180 deg [100]
    R.from_matrix([[0.5, np.sqrt(3)/2, 0], [np.sqrt(3)/2, -0.5, 0], [0, 0, -1]]),  # 180 deg [sqrt(3), 1, 0]
    R.from_matrix([[-0.5, np.sqrt(3)/2, 0], [np.sqrt(3)/2, 0.5, 0], [0, 0, -1]]),  # 180 deg [1, sqrt(3), 0]
    R.from_matrix([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # 180 deg [010]
    R.from_matrix([[-0.5, -np.sqrt(3)/2, 0], [-np.sqrt(3)/2, 0.5, 0], [0, 0, -1]]),
    # 180 deg [1, -sqrt(3), 0]
    R.from_matrix([[0.5, -np.sqrt(3)/2, 0], [-np.sqrt(3)/2, -0.5, 0], [0, 0, -1]]),
    # 180 deg [sqrt(3), -1, 0]
)

# T
tetrahedral = (
    R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # I

    R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),  # 180 deg [100]
    R.from_matrix([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # 180 deg [010]
    R.from_matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # 180 deg [001]

    R.from_matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),  # 120 deg [111]
    R.from_matrix([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),  # 120 deg [11-1]
    R.from_matrix([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),  # 120 deg [1-11]
    R.from_matrix([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),  # 120 deg [-111]
    R.from_matrix([[0, 0, -1], [1, 0, 0], [0, -1, 0]]),  # 120 deg [1-1-1]
    R.from_matrix([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),  # 120 deg [-11-1]
    R.from_matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),  # 120 deg [-1-11]
    R.from_matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),  # 120 deg [-1-1-1]
)

# O
cubic = (
    R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # I

    R.from_matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),  # 90 deg [100]
    R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),  # 180 deg [100]
    R.from_matrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),  # 270 deg [100]

    R.from_matrix([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),  # 90 deg [010]
    R.from_matrix([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # 180 deg [010]
    R.from_matrix([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),  # 270 deg [010]

    R.from_matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),  # 90 deg [001]
    R.from_matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # 180 deg [001]
    R.from_matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),  # 270 deg [001]

    R.from_matrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),  # 180 deg [110]
    R.from_matrix([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),  # 180 deg [101]
    R.from_matrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),  # 180 deg [011]

    R.from_matrix([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),  # 180 deg [1-10]
    R.from_matrix([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]),  # 180 deg [10-1]
    R.from_matrix([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]),  # 180 deg [01-1]

    R.from_matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),  # 120 deg [111]
    R.from_matrix([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),  # 120 deg [11-1]
    R.from_matrix([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),  # 120 deg [1-11]
    R.from_matrix([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),  # 120 deg [-111]
    R.from_matrix([[0, 0, -1], [1, 0, 0], [0, -1, 0]]),  # 120 deg [1-1-1]
    R.from_matrix([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),  # 120 deg [-11-1]
    R.from_matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),  # 120 deg [-1-11]
    R.from_matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),  # 120 deg [-1-1-1]
)

octahedral = cubic
