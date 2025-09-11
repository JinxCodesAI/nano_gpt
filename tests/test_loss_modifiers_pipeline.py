import os
import sys
import math
import torch
import torch.nn.functional as F

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loss_modifiers.pipeline import LossModifierPipeline
from loss_modifiers.target_smoothing_modifier import TargetSmoothingModifier
from loss_modifiers.mask_ratio_weight_modifier import MaskRatioWeightModifier


def manual_smoothed_ce(logits: torch.Tensor, targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    """
    Compute per-position smoothed cross-entropy matching TargetSmoothingModifier's convention:
    - true class prob: 1 - smoothing
    - other classes: smoothing / (V - 1)
    Returns tensor of shape (B, T) with per-position losses.
    """
    B, T, V = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)
    confidence = 1.0 - smoothing
    smoothing_value = smoothing / (V - 1)
    # build one-hot
    one_hot = torch.zeros(B, T, V, dtype=logits.dtype, device=logits.device)
    one_hot.scatter_(-1, targets.unsqueeze(-1), 1.0)
    q = one_hot * confidence + (1 - one_hot) * smoothing_value
    per_pos = -(q * log_probs).sum(dim=-1)
    return per_pos


def test_target_smoothing_matches_manual_expectation():
    torch.manual_seed(0)
    B, T, V = 2, 3, 5
    logits = torch.randn(B, T, V)
    targets = torch.randint(low=0, high=V, size=(B, T))

    pipe = LossModifierPipeline([
        TargetSmoothingModifier({
            'enabled': True,
            'smoothing_factor': 0.2,
            'exclude_padding': False,
        })
    ])

    base_loss = torch.tensor(0.0)  # will be replaced
    mask = torch.ones(B, T, dtype=torch.bool)
    out = pipe.modify_loss(logits, targets, base_loss, mask=mask)

    per_pos_manual = manual_smoothed_ce(logits, targets, smoothing=0.2)
    expected = per_pos_manual.mean()
    assert torch.allclose(out, expected, atol=1e-6), f"{out} vs {expected}"


def test_mask_ratio_weights_with_provided_per_position_loss():
    torch.manual_seed(0)
    B, T = 3, 4
    # Create arbitrary per-position loss and mask
    per_pos = torch.rand(B, T)
    mask = torch.tensor([[1,1,0,0],[1,1,1,0],[1,0,0,0]], dtype=torch.bool)

    # Expected per-sequence masked mean
    eps = 1e-8
    seq_means = (per_pos * mask.float()).sum(dim=1) / (mask.float().sum(dim=1) + eps)
    # Mask ratios per sequence
    ratios = mask.float().sum(dim=1) / (T + eps)
    power = 0.5
    weights = 1.0 / torch.clamp(ratios + eps, min=eps) ** power
    weights = torch.clamp(weights, min=0.1, max=10.0)
    expected = (seq_means * weights).mean()

    pipe = LossModifierPipeline([
        MaskRatioWeightModifier({
            'enabled': True,
            'power': power,
            'min_weight': 0.1,
            'max_weight': 10.0,
            'eps': eps,
        })
    ])

    # logits/targets are unused by this modifier when per_position_loss is provided
    logits = torch.zeros(B, T, 7)
    targets = torch.zeros(B, T, dtype=torch.long)
    base_loss = torch.tensor(123.45)  # will be replaced by aggregation in modifier
    out = pipe.modify_loss(logits, targets, base_loss, mask=mask, per_position_loss=per_pos)

    assert torch.allclose(out, expected, atol=1e-6), f"{out} vs {expected}"


def test_composed_smoothing_then_mask_ratio():
    torch.manual_seed(0)
    B, T, V = 2, 5, 8
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)

    smoothing = 0.1
    power = 0.5
    eps = 1e-8

    pipe = LossModifierPipeline([
        TargetSmoothingModifier({'enabled': True, 'smoothing_factor': smoothing, 'exclude_padding': False}),
        MaskRatioWeightModifier({'enabled': True, 'power': power, 'min_weight': 0.1, 'max_weight': 10.0, 'eps': eps}),
    ])

    base_loss = torch.tensor(0.0)
    out = pipe.modify_loss(logits, targets, base_loss, mask=mask)

    # Manual expectation: smoothing per-position CE, then mask ratio weighting
    per_pos = manual_smoothed_ce(logits, targets, smoothing)
    seq_means = (per_pos * mask.float()).sum(dim=1) / (mask.float().sum(dim=1) + eps)
    ratios = mask.float().sum(dim=1) / (T + eps)
    weights = 1.0 / torch.clamp(ratios + eps, min=eps) ** power
    weights = torch.clamp(weights, min=0.1, max=10.0)
    expected = (seq_means * weights).mean()

    assert torch.allclose(out, expected, atol=1e-6), f"{out} vs {expected}"
