"""Projection heads for the discriminative reference space and the calibration branch.

Corresponds to Section 3.2 (M^t) and Section 3.3 (G^t) of the manuscript.
Both heads are two-layer MLPs whose output is L2-normalised so that inner products
equal cosine similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLPHead(nn.Module):
    """Two-layer MLP with L2-normalised output (Sec. 3.2 / 3.3)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


class DiscriminativeProjection(nn.Module):
    """M^t: R^d -> R^{d_p}, producing the reference representation p_i (Eq. 2)."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.head = _MLPHead(in_dim, hidden_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


class CalibrationProjection(nn.Module):
    """G^t: R^{d_u} -> R^{d_p}, producing the calibrated representation s_i (Eq. 9).

    Shares the intermediate hidden dimension with ``DiscriminativeProjection`` so that
    the two branches can be aligned in the same reference space.
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.head = _MLPHead(latent_dim, hidden_dim, out_dim)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.head(u)
