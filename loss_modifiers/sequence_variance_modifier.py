"""
SequenceScorerVarianceModifier: scales sequence scoring loss by a clamped ratio of
batch variances: var(targets) / var(predictions). Intended only for SEQUENCE_SCORER mode.

Behavior:
- Compute X = var(predictions), Y = var(targets) across the batch (unbiased=False)
- Compute factor = clamp(Y / max(X, eps), min=1.0, max=sequence_variance_scale)
- Scale the base scalar loss: loss *= factor
- Preserves zero overhead when disabled.

Configuration keys (read via create_loss_modifier_pipeline):
- sequence_variance_enabled (bool): Enable this modifier
- sequence_variance_scale (float): Upper bound for the multiplicative factor (>1.0). Default 1.0
- sequence_variance_eps (float): Numerical stability when dividing by small X. Default 1e-8

Returned loss is a scalar; no per-position loss is produced.
"""

from .base import BaseLossModifier
import torch
import torch.nn.functional as F

# Import ModelMode safely
try:
    from model import ModelMode
except Exception:
    from enum import Enum
    class ModelMode(Enum):
        LANGUAGE_MODEL = "language_model"
        TOKEN_CLASSIFIER = "token_classifier"
        SEQUENCE_SCORER = "sequence_scorer"


class SequenceScorerVarianceModifier(BaseLossModifier):
    def __init__(self, config):
        super().__init__(config)
        self.enabled = config.get('enabled', False)
        self.scale = float(config.get('sequence_variance_scale', 2.0))  # upper cap (>1.0)
        self.alpha = float(config.get('sequence_variance_alpha', 1.5))  # growth rate
        self.eps = float(config.get('sequence_variance_eps', 1e-8))

    def supports_mode(self, mode: ModelMode) -> bool:
        return mode == ModelMode.SEQUENCE_SCORER

    def modify_loss(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor, model_mode: ModelMode = None, **kwargs):
        if not self.enabled or model_mode != ModelMode.SEQUENCE_SCORER:
            return loss

        # logits shape for sequence scorer passed from model: (B,) or (B,1)
        preds = logits.view(-1).float()
        targs = targets.view(-1).float()

        # Compute variances
        var_pred = preds.var(unbiased=False)
        var_targ = targs.var(unbiased=False)

        # Compute ratio r = Y/X with numerical stability
        denom = torch.clamp(var_pred, min=self.eps)
        r = var_targ / denom

        # Aggressive-but-saturating growth: factor = 1 for r<=1, then rise quickly and saturate
        # Use exponential saturation: factor = 1 + (scale-1) * (1 - exp(-alpha * (r-1)_+))
        excess = torch.clamp(r - 1.0, min=0.0)
        raw = 1.0 + (self.scale - 1.0) * (1.0 - torch.exp(-self.alpha * excess))
        factor = torch.clamp(raw, min=1.0, max=self.scale)

        scaled_loss = loss * factor

        # metrics for logging
        self._metrics = {
            'variance_pred': float(var_pred.detach().cpu().item()),
            'variance_target': float(var_targ.detach().cpu().item()),
            'ratio_y_over_x': float(r.detach().cpu().item()),
            'factor_applied': float(factor.detach().cpu().item()),
            'alpha': self.alpha,
            'scale_cap': self.scale,
            'original_loss': float(loss.detach().cpu().item()),
            'final_loss': float(scaled_loss.detach().cpu().item()),
            'loss_ratio': float((scaled_loss / (loss + self.eps)).detach().cpu().item()),
        }

        return scaled_loss

    def get_metrics(self):
        return self._metrics.copy()

