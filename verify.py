"""Self-check for the released SCORE components.

Runs the losses, projections and region diagnostics on synthetic tensors and asserts
the properties stated in the manuscript. Run with:  python verify.py
"""

import torch
import torch.nn.functional as F

from score import (
    CalibrationProjection,
    DiscriminativeProjection,
    GuidedCalibrationLoss,
    SupConLoss,
    drift_and_recovery,
    hit_ratio,
    reference_centre,
    region_radius,
)

torch.manual_seed(0)


def test_supcon_shape_and_range():
    """L_sup is a finite non-negative scalar for a well-formed batch."""
    p = F.normalize(torch.randn(16, 128), dim=1)
    y = torch.tensor([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)

    loss = SupConLoss(temperature=0.07)(p, y)
    assert loss.dim() == 0, "L_sup must be a scalar"
    assert torch.isfinite(loss), "L_sup must be finite"
    assert loss.item() >= 0.0, "L_sup must be non-negative"
    return loss.item()


def test_supcon_singleton_classes_return_zero():
    """Each class appears once, so every positive set is empty (Eq. 3) -> zero loss."""
    p = F.normalize(torch.randn(8, 128), dim=1)
    y = torch.arange(8)

    loss = SupConLoss()(p, y)
    assert abs(loss.item()) < 1e-12, "no positive pairs must yield zero loss"
    return loss.item()


def test_guide_loss_equals_one_minus_cosine():
    """Eq. (13) reduces to 1 - cos(s, p) for a single aligned pair."""
    s = F.normalize(torch.randn(1, 128), dim=1)
    p = s.clone()  # perfectly aligned

    loss = GuidedCalibrationLoss()(s, p)
    assert abs(loss.item()) < 1e-6, "aligned representations must give ~0 loss"

    p_orth = F.normalize(torch.randn(1, 128), dim=1)
    loss_orth = GuidedCalibrationLoss()(s, p_orth)
    expected = 1.0 - (s * p_orth).sum().item()
    assert abs(loss_orth.item() - expected) < 1e-6, "loss must equal 1 - cos(s, p)"
    return loss_orth.item()


def test_guide_target_is_stop_gradient():
    """p must not receive gradient (Eq. 13 uses sg(p))."""
    s = F.normalize(torch.randn(4, 128), dim=1).requires_grad_(True)
    p = F.normalize(torch.randn(4, 128), dim=1).requires_grad_(True)

    GuidedCalibrationLoss()(s, p).backward()
    assert s.grad is not None, "s must receive gradient"
    assert p.grad is None, "p must be detached (stop-gradient)"
    return True


def test_projection_heads_are_normalised():
    """Both heads output unit-norm codes on the hypersphere (Sec. 3.2 / 3.3)."""
    m = DiscriminativeProjection(in_dim=512, hidden_dim=512, out_dim=128)
    g = CalibrationProjection(latent_dim=16, hidden_dim=512, out_dim=128)

    p = m(torch.randn(8, 512))
    s = g(torch.randn(8, 16))

    for name, t in (("M^t", p), ("G^t", s)):
        norms = t.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"{name} output must be L2-normalised"
    return p.shape, s.shape


def test_region_radius_covers_ninety_percent():
    """The 0.9-quantile radius contains at least 90% of within-class samples (Eq. 6)."""
    p = F.normalize(torch.randn(200, 128), dim=1)
    y = torch.zeros(200, dtype=torch.long)

    centres = reference_centre(p, y, n_classes=1)
    radii = region_radius(p, y, centres, quantile=0.9)
    hr = hit_ratio(p, y, centres, radii)

    assert 0.85 <= hr <= 0.95, f"expected ~0.90 coverage, got {hr:.4f}"
    return hr


def test_drift_recovery_degenerate_case():
    """If the calibrated centre coincides with the current centre, recovery is ~1."""
    ref_prev = F.normalize(torch.randn(1, 128), dim=1)
    ref_curr = F.normalize(torch.randn(1, 128), dim=1)
    s = ref_curr.repeat(10, 1)
    labels = torch.zeros(10, dtype=torch.long)

    out = drift_and_recovery(ref_prev, ref_curr, s, labels)
    assert out["guided"].item() < 1e-5, "guided distance must be ~0"
    assert out["recovery"].item() > 0.999, "recovery must be ~1"
    return out["recovery"].item()


if __name__ == "__main__":
    checks = [
        ("L_sup scalar / finite / non-negative", test_supcon_shape_and_range),
        ("L_sup zero for empty positive sets", test_supcon_singleton_classes_return_zero),
        ("L_guide = 1 - cos(s, p)", test_guide_loss_equals_one_minus_cosine),
        ("L_guide stop-gradient on p", test_guide_target_is_stop_gradient),
        ("projection heads unit-norm", test_projection_heads_are_normalised),
        ("0.9-quantile radius coverage", test_region_radius_covers_ninety_percent),
        ("drift/recovery degenerate case", test_drift_recovery_degenerate_case),
    ]

    failed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")

    print()
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    raise SystemExit(1 if failed else 0)
