"""Class-subspace calibration step (Section 3.3).

Takes a per-class latent code ``u_i`` produced by a class-specific autoencoder, maps it
into the discriminative reference space through ``G^t``, and aligns it to the current
reference representation ``p_i`` with a one-way objective.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import GuidedCalibrationLoss
from .projection import CalibrationProjection


class ClassSubspaceCalibrator(nn.Module):
    """Maps class-relevant latent codes to the reference space and calibrates them.

    Args:
        latent_dim:   d_u, the autoencoder latent dimension (16 in the paper).
        hidden_dim:   shared intermediate dimension of G^t (512 in the paper).
        ref_dim:      d_p, the discriminative reference dimension (128 in the paper).
        guide_lambda: weight of L_guide in the total objective (1.0 in the paper).
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 512,
                 ref_dim: int = 128, guide_lambda: float = 1.0):
        super().__init__()
        self.projection = CalibrationProjection(latent_dim, hidden_dim, ref_dim)
        self.criterion = GuidedCalibrationLoss()
        self.guide_lambda = guide_lambda

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Return the calibrated representation s_i of Eq. (9)."""
        return self.projection(u)

    def calibration_loss(self, u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """L_guide (Eq. 13): align s_i to the stop-gradient target p_i.

        Args:
            u: [N, d_u] class-relevant latent codes of stored old-class samples.
            p: [N, d_p] their current-stage reference representations.
        """
        s = self.forward(u)
        return self.guide_lambda * self.criterion(s, p)


@torch.no_grad()
def select_class_encoder(latent_codes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Training-time selection of the class-consistent latent code.

    During training the ground-truth label picks the corresponding encoder output
    (Eq. 7); at inference the model does not rely on the label to choose a subspace.

    Args:
        latent_codes: [B, C, d_u] codes from all observed class autoencoders.
        labels:       [B] ground-truth class ids.

    Returns:
        [B, d_u] the code of the label-consistent subspace.
    """
    idx = torch.arange(labels.size(0), device=labels.device)
    return latent_codes[idx, labels]
