"""diffractom — GPU-accelerated texture tomography reconstruction for diffraction data."""

from .crystallography.material import Material
from .crystallography.intensity import IntensityModel
from .utils.grid import Grid, MatrixGrid
from .utils.support import fov_support_mask
from .operators.single_phase_forward_operator import SinglePhaseForwardOperator, estimate_L_power
from .operators.local_forward_operator import LocalSinglePhaseForwardOperator
from .operators.local_support import LocalSupport
from .operators.multi_phase_forward_operator import MultiPhaseForwardOperator
from .operators.bulk_texture_forward_operator import BulkTextureForwardOperator
from .operators.matrix_tomographic_operator import MatrixTomographicOperator
from .operators.bragg_edge_tomographic_operator import BraggEdgeTomographicOperator, bragg_edge_matrix
from .optimization.fista_huber import FISTAHuber
from .optimization.fista_l2 import FISTAL2
from .optimization.fista_sum import FISTASum, estimate_L_sum
from .optimization.streaming import estimate_L_power_streamed

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
    "IntensityModel",
    "Material",
    "Grid",
    "MatrixGrid",
    "fov_support_mask",
    "SinglePhaseForwardOperator",
    "LocalSinglePhaseForwardOperator",
    "LocalSupport",
    "estimate_L_power",
    "MultiPhaseForwardOperator",
    "FISTAHuber",
    "FISTAL2",
    "FISTASum",
    "estimate_L_sum",
    "estimate_L_power_streamed",
    "BulkTextureForwardOperator",
    "MatrixTomographicOperator",
    "BraggEdgeTomographicOperator",
    "bragg_edge_matrix",
    "build_interpolation_kernelSO3",
    "interpolateSO3",
    "build_interpolation_kernelS2",
    "interpolateS2",
    "project_rotations_to_s2",
]