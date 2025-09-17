"""
SequenceScorerCorrelationModifier: scales sequence scoring loss by a non-linear
function of the Pearson correlation between predictions and targets.

Mapping (correlation c in [-1, 1]):
- c =  1 -> factor = 1.0 (no change)
- c =  0 -> factor = sqrt(alpha)
- c = -1 -> factor = alpha

Intuition: penalize anti-correlation most, neutral correlation moderately,
no penalty for perfect positive correlation.

Intended only for SEQUENCE_SCORER mode.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from .base import BaseLossModifier

# Import ModelMode safely
try:
    from model import ModelMode
except Exception:
    from enum import Enum
    class ModelMode(Enum):
        LANGUAGE_MODEL = "language_model"
        TOKEN_CLASSIFIER = "token_classifier"
        SEQUENCE_SCORER = "sequence_scorer"


class SequenceScorerCorrelationModifier(BaseLossModifier):
    def __init__(self, config):
        super().__init__(config)
        self.enabled = config.get('enabled', False)
        self.alpha = float(config.get('sequence_correlation_alpha', 4.0))
        if self.alpha < 1.0:
            raise ValueError("sequence_correlation_alpha must be >= 1.0")
        self.eps = float(config.get('sequence_correlation_eps', 1e-8))
        # Precompute coefficients for quadratic A*c^2 + B*c + C
        sqrt_alpha = math.sqrt(self.alpha)
        self.A = (self.alpha - 2 * sqrt_alpha + 1.0) / 2.0
        self.B = (1.0 - self.alpha) / 2.0
        self.C = sqrt_alpha

    def supports_mode(self, mode: ModelMode) -> bool:
        return mode == ModelMode.SEQUENCE_SCORER

    @staticmethod
    def _pearson_corr(preds: torch.Tensor, targs: torch.Tensor, eps: float) -> torch.Tensor:
        # Both are 1D float tensors
        x = preds - preds.mean()
        y = targs - targs.mean()
        cov = (x * y).sum()
        std_x = torch.sqrt((x * x).sum() + eps)
        std_y = torch.sqrt((y * y).sum() + eps)
        corr = cov / (std_x * std_y + eps)
        # Clamp to [-1,1] for safety
        return torch.clamp(corr, min=-1.0, max=1.0)

    def modify_loss(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor, model_mode: ModelMode = None, **kwargs):
        if not self.enabled or model_mode != ModelMode.SEQUENCE_SCORER:
            return loss

        preds = logits.view(-1).float().detach()  # detach to avoid gradients through correlation
        targs = targets.view(-1).float()

        # Pearson correlation
        corr = self._pearson_corr(preds, targs, self.eps)
        # Non-linear factor via quadratic
        factor = self.A * (corr ** 2) + self.B * corr + self.C
        # Scale the base scalar loss
        scaled_loss = loss * factor

        # Metrics for logging
        self._metrics = {
            'alpha': self.alpha,
            'A': float(self.A),
            'B': float(self.B),
            'C': float(self.C),
            'correlation': float(corr.detach().cpu().item()),
            'multiplier': float(factor.detach().cpu().item()),
            'original_loss': float(loss.detach().cpu().item()),
            'final_loss': float(scaled_loss.detach().cpu().item()),
            'loss_ratio': float((scaled_loss / (loss + self.eps)).detach().cpu().item()),
        }
        return scaled_loss

    def get_metrics(self):
        return self._metrics.copy()

