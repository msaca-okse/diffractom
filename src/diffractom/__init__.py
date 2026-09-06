"""diffractom — GPU-accelerated texture tomography reconstruction for diffraction data."""

from .crystallography.material import Material
from .utils.grid import Grid
from .operators.single_phase_forward_operator import SinglePhaseForwardOperator
from .operators.multi_phase_forward_operator import MultiPhaseForwardOperator
from .operators.bulk_texture_forward_operator import BulkTextureForwardOperator
from .operators.matrix_tomographic_operator import MatrixTomographicOperator
from .operators.bragg_edge_tomographic_operator import BraggEdgeTomographicOperator
from .optimization.fista_huber import FISTAHuber
from .optimization.fista_l2 import FISTAL2

from .utils.reinterpolation.reinterpolationSO3 import (
    build_interpolation_kernelSO3,
    interpolateSO3,
)

from .utils.reinterpolation.reinterpolationS2 import (
    build_interpolation_kernelS2,
    interpolateS2,
    project_rotations_to_s2,
)

__all__ = [
    "Material",
    "Grid",
    "SinglePhaseForwardOperator",
    "MultiPhaseForwardOperator",
    "FISTAHuber",
    "FISTAL2",
    "BulkTextureForwardOperator",
    "MatrixTomographicOperator",
    "BraggEdgeTomographicOperator",
    "build_interpolation_kernelSO3",
    "interpolateSO3",
    "build_interpolation_kernelS2",
    "interpolateS2",
    "project_rotations_to_s2",
]