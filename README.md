# SCORE: Subspace COordination for discRiminative rEpresentation

Companion implementation for the manuscript:

> **Discriminative reference-guided class-subspace calibration for class-incremental learning**
> Yukun Ma, Xinze Li, Chao Chen, Yunxin Yang, Jinquan Ji

---

## What this repository contains

This repository provides the implementation of the **novel components** introduced by
SCORE. It is released as a component-level reference so that the two new losses and
the region-based diagnostic can be inspected and re-implemented directly.

| File | Corresponds to | Description |
|---|---|---|
| `score/losses.py` | Eq. (4), Eq. (13) | `SupConLoss` (L_sup) and `GuidedCalibrationLoss` (L_guide) |
| `score/region_metrics.py` | Eq. (5), Eq. (6), Sec. 4.4.6 | Reference centres, 0.9-quantile region radii, drift / recovery / hit-ratio |
| `score/projection.py` | Sec. 3.2, Sec. 3.3 | Discriminative projection head `M^t` and calibration head `G^t` |
| `score/calibration.py` | Eq. (9), Eq. (13) | Calibrated representation `s_i` and one-way alignment to `p_i` |

## Provenance

The class-incremental learning pipeline in which these components are embedded follows
**CREATE** (Chen et al., *Reducing Class-wise Confusion for Incremental Learning with
Disentangled Manifolds*, CVPR 2025), specifically its class-specific autoencoder
structure, the reconstruction-error class response, and the distillation over
reconstruction responses (Eqs. (10)–(12), (14) of the manuscript).

SCORE's contribution on top of that pipeline is:

- **L_sup** — supervised contrastive construction of the discriminative reference space
  (Eq. (4)), which is not part of CREATE.
- **L_guide** — one-way cosine calibration of class-relevant latent representations to
  the current-stage reference representations (Eq. (13)), which is not part of CREATE.
- **Region-based reference characterisation** (Eq. (5)–(6)) and the drift / recovery /
  hit-ratio diagnostics reported in Section 4.4.6.

The full training pipeline is not redistributed here; the files above are self-contained
and can be dropped into any PyTorch class-incremental-learning pipeline that exposes a
per-class latent code and a projected reference representation.

## Dependencies

```
torch >= 2.0
numpy
```

## Minimal usage

```python
import torch
from score.losses import SupConLoss, GuidedCalibrationLoss

supcon = SupConLoss(temperature=0.07)
guide  = GuidedCalibrationLoss()

# p: [B, d_p] L2-normalised reference representations of current-stage new-class samples
# y: [B]      their labels
l_sup = supcon(p.unsqueeze(1), y)

# s: [N_old, d_p] calibration representations of stored old-class samples
# p_old: [N_old, d_p] their current-stage reference representations
l_guide = guide(s, p_old)
```

## Notes on the reported results

The region statistics in Section 4.4.6 (Drift Mean, Guided Mean, Recovery Mean,
Hit Ratio Mean, Radius Mean) are produced by `score/region_metrics.py` and written as
per-task `.npz` archives plus a summary `.csv`, matching the values reported in
Figure 8.
