"""
SequenceScorerVarianceModifier: scales sequence scoring loss by the variance between
predictions and targets. Intended only for SEQUENCE_SCORER mode.

Behavior:
- Computes batch-level variance of errors (pred - target) or of targets/predictions and
  scales the base scalar loss accordingly.
- Preserves zero overhead when disabled.

Configuration keys (read via create_loss_modifier_pipeline):
- sequence_variance_enabled (bool): Enable this modifier
- sequence_variance_mode (str): One of {"error", "prediction", "target"}; default "error"
- sequence_variance_scale (float): Multiplier applied to the variance; default 1.0
- sequence_variance_eps (float): Numerical stability; default 1e-8

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
        self.mode = config.get('sequence_variance_mode', 'error')
        self.scale = float(config.get('sequence_variance_scale', 1.0))
        self.eps = float(config.get('sequence_variance_eps', 1e-8))

    def supports_mode(self, mode: ModelMode) -> bool:
        return mode == ModelMode.SEQUENCE_SCORER

    def modify_loss(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor, model_mode: ModelMode = None, **kwargs):
        if not self.enabled or model_mode != ModelMode.SEQUENCE_SCORER:
            return loss

        # logits shape for sequence scorer passed from model: (B,) or (B,1)
        preds = logits.view(-1).float()
        targs = targets.view(-1).float()

        if self.mode == 'prediction':
            var = preds.var(unbiased=False)
        elif self.mode == 'target':
            var = targs.var(unbiased=False)
        else:  # 'error' (default): variance of residuals
            var = (preds - targs).var(unbiased=False)

        # safe scalar
        var = torch.clamp(var, min=0.0)
        scaled_loss = loss * (1.0 + self.scale * var)

        # metrics for logging
        self._metrics = {
            'variance_mode': self.mode,
            'variance_value': float(var.detach().cpu().item()),
            'scale': self.scale,
            'original_loss': float(loss.detach().cpu().item()),
            'final_loss': float(scaled_loss.detach().cpu().item()),
            'loss_ratio': float((scaled_loss / (loss + self.eps)).detach().cpu().item()),
        }

        return scaled_loss

    def get_metrics(self):
        return self._metrics.copy()

