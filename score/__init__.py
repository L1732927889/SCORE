"""SCORE: subspace coordination for discriminative representation.

Only the components introduced by SCORE are exported here. The surrounding
class-incremental-learning pipeline follows CREATE (Chen et al., CVPR 2025);
see README.md for the provenance statement.
"""

from .calibration import ClassSubspaceCalibrator, select_class_encoder
from .losses import GuidedCalibrationLoss, SupConLoss
from .projection import CalibrationProjection, DiscriminativeProjection
from .region_metrics import (
    drift_and_recovery,
    hit_ratio,
    reference_centre,
    region_radius,
    save_stage_diagnostics,
)

__all__ = [
    "SupConLoss",
    "GuidedCalibrationLoss",
    "DiscriminativeProjection",
    "CalibrationProjection",
    "ClassSubspaceCalibrator",
    "select_class_encoder",
    "reference_centre",
    "region_radius",
    "hit_ratio",
    "drift_and_recovery",
    "save_stage_diagnostics",
]
