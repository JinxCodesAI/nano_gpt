"""
Generic metrics-collector loss modifier.

This modifier computes training metrics without changing the loss. It is intended
for cheap, per-forward-pass metric calculations that can be picked up by the
pipeline for logging. It supports multiple model modes and can be extended with
additional metrics in the future.

Currently implemented metrics:
- For SEQUENCE_SCORER mode: Average Absolute Relative Error (AARE)
  aare = mean(abs(pred - target) / clamp(abs(target), eps))
  Exposes both instantaneous and EMA-smoothed values.
"""

from typing import Dict, Any
import torch

from .base import BaseLossModifier
from model import ModelMode


class MetricsCollectorModifier(BaseLossModifier):
    """
    Metric-only modifier: computes metrics and returns the loss unchanged.

    Config supported keys:
    - enabled: bool
    - eps: float (default 1e-6) numerical stability for denominators
    - ema_alpha: float (default 0.1) smoothing factor for EMA of metrics
    - collect_sequence_scorer_aare: bool (default True) enable AARE in SEQUENCE_SCORER
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.eps = float(config.get('eps', 1e-6))
        self.ema_alpha = float(config.get('ema_alpha', 0.1))
        self.collect_seq_aare = bool(config.get('collect_sequence_scorer_aare', True))
        # Persistent EMA states (do not clear on reset_metrics)
        self._aare_ema = None

    def supports_mode(self, mode: ModelMode) -> bool:
        # Generic collector exists for all modes; will compute only those applicable
        return True

    @torch.no_grad()
    def _update_ema(self, name: str, value: float) -> float:
        if name == 'aare':
            if self._aare_ema is None:
                self._aare_ema = value
            else:
                self._aare_ema = (1.0 - self.ema_alpha) * self._aare_ema + self.ema_alpha * value
            return self._aare_ema
        return value

    def modify_loss(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor, model_mode: ModelMode = None, **kwargs):
        # Clear per-pass metrics
        self._metrics = {}

        # Sequence scorer metrics
        if model_mode == ModelMode.SEQUENCE_SCORER and self.collect_seq_aare:
            with torch.no_grad():
                denom = torch.clamp(torch.abs(targets), min=self.eps)
                aare_now = (torch.abs(logits - targets) / denom).mean()
                aare_now_val: float = float(aare_now.detach().cpu().item())
                aare_ema_val: float = self._update_ema('aare', aare_now_val)
                self._metrics['aare_instant'] = aare_now_val
                self._metrics['aare_ema'] = aare_ema_val

        # Do not modify loss
        return loss

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)

