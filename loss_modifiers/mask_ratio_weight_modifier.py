"""
Mask ratio weight modifier that applies loss weighting based on mask ratios.

This modifier applies a loss multiplier that is inversely proportional to the 
square root of the mask ratio. This can help balance training when dealing with
sequences that have varying amounts of valid (non-masked) tokens.

The intuition is that sequences with fewer valid tokens should receive higher
weights to compensate for having less signal in the loss calculation.
"""

from typing import Dict, Any, Optional
import torch
from .base import BaseLossModifier
from model import ModelMode


class MaskRatioWeightModifier(BaseLossModifier):
    """
    Loss modifier that weights loss inversely to the square root of mask ratio.
    
    This modifier calculates the ratio of valid (non-masked) positions in each
    sequence and applies a weight inversely proportional to the power (default 0.5) of
    this ratio. This gives an option to reward model more for solving correctly harder samples.
    
    The weighting formula is: weight = 1 / mask_ratio^power
    where mask_ratio is the fraction of valid tokens in the sequence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize mask ratio weight modifier.
        
        Config keys:
            enabled (bool): Whether to enable this modifier
            power (float): Power for the inverse square root weighting (default: 0.5)
            min_weight (float): Minimum weight to apply (prevents extreme weights) (default: 0.1)
            max_weight (float): Maximum weight to apply (prevents extreme weights) (default: 10.0)
            eps (float): Small value to prevent division by zero (default: 1e-8)
        """
        super().__init__(config)
        self.power = config.get('power', 0.5)
        self.min_weight = config.get('min_weight', 0.1)
        self.max_weight = config.get('max_weight', 10.0)
        self.eps = config.get('eps', 1e-8)
        print(f"MaskRatioWeightModifier: power={self.power}, min_weight={self.min_weight}, max_weight={self.max_weight}, eps={self.eps}")
    
    def _calculate_mask_ratios(
        self,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate mask ratios for each sequence in the batch.
        
        Args:
            mask: Boolean mask tensor (batch_size, seq_len) where True indicates valid positions
            
        Returns:
            mask_ratios: (batch_size,) - ratio of valid positions per sequence
        """
        # Calculate number of valid positions per sequence
        valid_counts = mask.float().sum(dim=1)  # (batch_size,)
        
        # Calculate total positions per sequence
        total_counts = mask.shape[1]  # seq_len
        
        # Calculate ratios (add eps to prevent division by zero)
        mask_ratios = valid_counts / (total_counts + self.eps)
        
        return mask_ratios
    
    def _calculate_weights(
        self,
        mask_ratios: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate weights based on mask ratios.
        
        Args:
            mask_ratios: (batch_size,) - ratio of valid positions per sequence
            
        Returns:
            weights: (batch_size,) - weight per sequence
        """
        # Apply inverse square root weighting: weight = 1 / sqrt(mask_ratio)^power
        # Add eps to prevent division by zero
        weights = 1.0 / ((mask_ratios + self.eps) ** self.power)
        
        # Clamp weights to prevent extreme values
        weights = torch.clamp(weights, min=self.min_weight, max=self.max_weight)
        
        return weights
    
    def supports_mode(self, mode: ModelMode) -> bool:
        return mode == ModelMode.LANGUAGE_MODEL

    def modify_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        model_mode=None,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply mask ratio based loss weighting.
        
        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size)
            targets: Target token IDs (batch_size, seq_len)
            loss: Original loss value (scalar tensor)
            mask: Optional boolean mask (batch_size, seq_len) - required for this modifier
            
        Returns:
            Modified loss tensor weighted by mask ratios
        """
        if not self.enabled:
            return loss
        
        # Check mode compatibility
        if model_mode and not self.supports_mode(model_mode):
            return loss

        # If no mask provided, infer it from targets using ignore_index (default -100)
        if mask is None:
            ignore_index = kwargs.get('ignore_index', -100)
            mask = targets != ignore_index
        
        # Calculate mask ratios
        mask_ratios = self._calculate_mask_ratios(mask)
        
        # Calculate weights
        weights = self._calculate_weights(mask_ratios)
        
        # Store metrics
        self._metrics = {
            'mean_mask_ratio': mask_ratios.mean().item(),
            'min_mask_ratio': mask_ratios.min().item(),
            'max_mask_ratio': mask_ratios.max().item(),
            'mean_weight': weights.mean().item(),
            'min_weight': weights.min().item(),
            'max_weight': weights.max().item(),
            'weight_std': weights.std().item(),
        }

        # Always apply per-sequence weighting
        batch_size, seq_len, vocab_size = logits.shape

        # Use provided per-position loss if available; otherwise compute CE without ignore handling
        per_position_loss = kwargs.get('per_position_loss', None)
        if per_position_loss is None:
            # Fallback: compute per-position CE
            flat_logits = logits.view(-1, vocab_size)
            flat_targets = targets.view(-1)
            per_position_loss = torch.nn.functional.cross_entropy(
                flat_logits, flat_targets, reduction='none'
            ).view(batch_size, seq_len)

        # Apply mask and calculate per-sequence loss
        mask_f = mask.float()
        masked_loss = per_position_loss * mask_f
        per_sequence_loss = masked_loss.sum(dim=1) / (mask_f.sum(dim=1) + self.eps)

        # Apply weights to per-sequence losses
        weighted_losses = per_sequence_loss * weights

        # Return mean of weighted losses
        final_loss = weighted_losses.mean()

        return final_loss
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get mask ratio weighting metrics from the last forward pass."""
        return self._metrics.copy()