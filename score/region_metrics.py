"""Region-based reference characterisation and calibration diagnostics.

Corresponds to Eq. (5), Eq. (6) and the analysis of Section 4.4.6 (Figure 8).
"""

import csv
import os
from typing import Dict, List, Optional

import numpy as np
import torch

EPS = 1e-8


@torch.no_grad()
def reference_centre(p: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Within-batch reference centre r_c^t for each class (Eq. 5).

    Args:
        p:       [B, d_p] L2-normalised reference representations.
        labels:  [B] integer class ids.
        n_classes: number of classes observed so far.

    Returns:
        [n_classes, d_p] centres; rows for absent classes are zero.
    """
    d_p = p.shape[1]
    centres = torch.zeros(n_classes, d_p, device=p.device, dtype=p.dtype)
    counts = torch.zeros(n_classes, device=p.device, dtype=p.dtype)

    centres.index_add_(0, labels, p)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=p.dtype))
    return centres / counts.clamp(min=1).unsqueeze(1)


@torch.no_grad()
def region_radius(p: torch.Tensor, labels: torch.Tensor, centres: torch.Tensor,
                  quantile: float = 0.9) -> torch.Tensor:
    """Effective region radius R_c^t as the ``quantile`` of within-class distances (Eq. 6).

    Returns:
        [n_classes] radii; zero for classes with fewer than two samples.
    """
    n_classes = centres.shape[0]
    radii = torch.zeros(n_classes, device=p.device, dtype=p.dtype)
    dist = torch.norm(p - centres[labels], dim=1)          # [B]

    for c in torch.unique(labels).tolist():
        d = dist[labels == c]
        if d.numel() >= 2:
            radii[c] = torch.quantile(d, q=quantile)
    return radii


@torch.no_grad()
def hit_ratio(s: torch.Tensor, labels: torch.Tensor,
              centres: torch.Tensor, radii: torch.Tensor) -> float:
    """Fraction of calibrated samples falling inside their class region R_c^t (Fig. 8b)."""
    if s.numel() == 0:
        return 0.0
    dist = torch.norm(s - centres[labels], dim=1)
    inside = dist <= radii[labels]
    return float(inside.float().mean().item())


@torch.no_grad()
def drift_and_recovery(ref_prev: torch.Tensor, ref_curr: torch.Tensor,
                       s_curr: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Drift, guided distance and recovery per class (Section 4.4.6, Fig. 8a-b).

    Args:
        ref_prev: [C, d_p] reference centres at stage t-1.
        ref_curr: [C, d_p] reference centres at stage t.
        s_curr:   [N, d_p] calibrated old-class representations at stage t.
        labels:   [N] class ids of the calibrated samples.

    Returns:
        dict of per-class tensors: ``drift``, ``guided``, ``recovery``.
    """
    drift = torch.norm(ref_curr - ref_prev, dim=1)          # centre shift before calibration

    guided = torch.zeros_like(drift)
    for c in torch.unique(labels).tolist():
        sel = s_curr[labels == c]
        if sel.numel() == 0:
            continue
        guided[c] = torch.norm(sel.mean(dim=0) - ref_curr[c])

    recovery = (drift - guided) / (drift + EPS)
    return {"drift": drift, "guided": guided, "recovery": recovery}


def save_stage_diagnostics(path: str, stage: int, drift: torch.Tensor,
                           guided: torch.Tensor, recovery: torch.Tensor,
                           hit: float, radii: torch.Tensor) -> None:
    """Persist per-stage diagnostics as .npz plus a one-line CSV summary."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        f"{path}_task{stage}_region_metrics.npz",
        drift=drift.cpu().numpy(),
        guide=guided.cpu().numpy(),
        recovery=recovery.cpu().numpy(),
        radii=radii.cpu().numpy(),
        hit_ratio=hit,
    )

    summary = f"{path}_region_summary.csv"
    new_file = not os.path.exists(summary)
    with open(summary, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["task", "drift_mean", "guide_mean",
                        "recovery_mean", "hit_ratio_mean", "radius_mean"])
        w.writerow([stage, f"{drift.mean():.6f}", f"{guided.mean():.6f}",
                    f"{recovery.mean():.6f}", f"{hit:.6f}", f"{radii.mean():.6f}"])
