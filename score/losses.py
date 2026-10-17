"""SCORE losses: supervised contrastive construction (L_sup) and guided calibration (L_guide).

Corresponds to Eq. (4) and Eq. (13) of the manuscript.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss over the discriminative reference space.

    Eq. (4) of the manuscript. Anchors, positives and negatives are all drawn from
    the current-stage new-class index set I_t^n (see ``positive_mask`` below).
    """

    def __init__(self, temperature: float = 0.07, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, n_views, D] or [B, D] L2-normalised reference representations.
            labels:   [B]

        Returns:
            Scalar loss. Returns 0.0 when no anchor has a positive partner.
        """
        if features.dim() == 2:
            features = features.unsqueeze(1)

        device = features.device
        bsz = features.shape[0]

        # Similarity between every pair of reference representations.
        contrast = torch.cat(torch.unbind(features, dim=1), dim=0)      # [B*n_views, D]
        sim = torch.matmul(contrast, contrast.T) / self.temperature      # [M, M]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)             # [B, B]

        # Positive set P(i): same class, excluding the anchor itself (Eq. 3).
        logits_mask = torch.ones_like(mask) - torch.eye(bsz, device=device)
        positive_mask = mask * logits_mask
        n_pos = positive_mask.sum(dim=1)

        # Anchors with an empty positive set contribute nothing.
        valid = n_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Numerical stability.
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + self.eps)

        # Mean over positives, then mean over valid anchors.
        per_anchor = (positive_mask * log_prob).sum(dim=1) / (n_pos + self.eps)
        return -per_anchor[valid].mean()


class GuidedCalibrationLoss(nn.Module):
    """One-way cosine calibration of historical representations to current references.

    Eq. (13) of the manuscript:  L_guide = mean_i (1 - sg(p_i)^T s_i)

    ``p_i`` is a stop-gradient target, so gradients flow only through the calibration
    branch that produces ``s_i``.
    """

    def __init__(self):
        super().__init__()

    def forward(self, s: torch.Tensor, p: torch.Tensor,
                detach_target: bool = True) -> torch.Tensor:
        """
        Args:
            s: [N, d_p] calibration representations (gradient flows here).
            p: [N, d_p] current-stage reference representations (target).
            detach_target: apply the stop-gradient of Eq. (13).

        Returns:
            Scalar loss.
        """
        if s.numel() == 0:
            return torch.tensor(0.0, device=s.device, requires_grad=True)

        s = F.normalize(s, dim=1)
        p = F.normalize(p, dim=1)
        target = p.detach() if detach_target else p

        return (1.0 - (target * s).sum(dim=1)).mean()
