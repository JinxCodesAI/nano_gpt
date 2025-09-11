import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loss_modifiers.pipeline import LossModifierPipeline
from loss_modifiers.entropy_modifier import EntropyModifier


def test_entropy_modifier_per_sample_weighting():
    torch.manual_seed(0)
    B, T, V = 3, 4, 6
    logits = torch.randn(B, T, V)

    # Construct targets so that sample 0 has very concentrated wrong-answer distribution,
    # and sample 1 has more diffuse wrong-answer distribution, by manipulating logits.
    targets = torch.randint(0, V, (B, T))

    # Force sample 0 to have a very peaky wrong answer distribution: push one wrong logit high
    for t in range(T):
        correct = targets[0, t].item()
        wrong = (correct + 1) % V
        logits[0, t, wrong] += 5.0  # make one wrong token dominant

    # Force sample 1 to be more uniform among wrong answers: keep logits small noise
    # sample 2 left random

    mask = torch.ones(B, T, dtype=torch.bool)

    # Build arbitrary per-position base loss to be weighted: use standard CE per-pos
    flat_logits = logits.view(-1, V)
    flat_targets = targets.view(-1)
    per_position_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none').view(B, T)

    pipe = LossModifierPipeline([
        EntropyModifier({
            'enabled': True,
            'weight': 1.0,
            'entropy_threshold': 1e-3,
            'eps': 1e-8,
        })
    ])

    base_loss = torch.tensor(0.0)
    out = pipe.modify_loss(logits, targets, base_loss, mask=mask, per_position_loss=per_position_loss)

    # Manually compute per-sample entropies over wrong answers
    probs = F.softmax(logits, dim=-1)
    target_mask = torch.zeros_like(probs, dtype=torch.bool)
    target_mask.scatter_(-1, targets.unsqueeze(-1), True)
    wrong_probs = probs.clone(); wrong_probs[target_mask] = 0.0
    sums = wrong_probs.sum(dim=-1, keepdim=True)
    wrong_probs = wrong_probs / (sums + 1e-8)
    ent = -(wrong_probs * torch.log(wrong_probs + 1e-8)).sum(dim=-1)
    sample_entropy = (ent * mask.float()).sum(dim=1) / (mask.float().sum(dim=1) + 1e-8)
    weights = 1.0 / (torch.clamp(sample_entropy, min=1e-3) + 1e-8)
    seq_means = per_position_loss.mean(dim=1)
    expected = (seq_means * weights).mean()

    # Lower entropy for sample 0 should produce higher weight than sample 1
    assert weights[0] > weights[1]
    assert torch.allclose(out, expected, atol=1e-6)
