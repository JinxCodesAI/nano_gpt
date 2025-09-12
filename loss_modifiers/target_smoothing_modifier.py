
"""
Target smoothing loss modifier that applies label smoothing to target tokens.

This modifier implements label smoothing by redistributing probability mass from
the target token to all other tokens in the vocabulary. This can help with
overfitting and improve model generalization.

Special tokens can be excluded from smoothing to preserve their semantic importance.
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F
from .base import BaseLossModifier


class TargetSmoothingModifier(BaseLossModifier):
    """
    Loss modifier that applies label smoothing to target tokens.
    
    Label smoothing redistributes probability mass from the target token to all other
    tokens in the vocabulary. Instead of using hard targets (1 for correct, 0 for others),
    it uses soft targets that assign some probability to incorrect tokens.
    
    This can help reduce overfitting and improve model generalization by preventing
    the model from becoming too confident in its predictions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize target smoothing modifier.
        
        Config keys:
            enabled (bool): Whether to enable this modifier
            smoothing_factor (float): Label smoothing factor (0.0 = no smoothing, higher values = more smoothing)
            special_token_ids (List[int]): List of special token IDs to exclude from smoothing
            exclude_padding (bool): Whether to exclude padding tokens from loss calculation (default: True)
            padding_token_id (int): Padding token ID to exclude (default: -100, matching PyTorch default)
        """
        super().__init__(config)
        self.smoothing_factor = config.get('smoothing_factor', 0.1)
        self.special_token_ids = config.get('special_token_ids', [])
        self.exclude_padding = config.get('exclude_padding', True)
        self.padding_token_id = config.get('padding_token_id', -100)
        
        # Create special token mask for efficient lookup
        self.special_token_set = set(self.special_token_ids) if self.special_token_ids else set()
        print(f"TargetSmoothingModifier: smoothing_factor={self.smoothing_factor}, special_token_ids={self.special_token_ids}, exclude_padding={self.exclude_padding}, padding_token_id={self.padding_token_id}")
    
    def _create_smoothed_targets(
        self,
        targets: torch.Tensor,
        vocab_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create smoothed target probability distributions.
        
        Args:
            targets: Target token IDs (batch_size, seq_len)
            vocab_size: Size of vocabulary
            device: Device to create tensors on
            
        Returns:
            smoothed_targets: Probability distribution targets (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = targets.shape
        
        # Create one-hot encoding of targets
        one_hot = torch.zeros(batch_size, seq_len, vocab_size, device=device)
        
        # Handle padding tokens and invalid indices
        valid_mask = (targets >= 0) & (targets < vocab_size)
        if self.exclude_padding:
            valid_mask = valid_mask & (targets != self.padding_token_id)
        
        # Only scatter for valid positions
        valid_targets = targets * valid_mask.long()  # Zero out invalid positions
        one_hot.scatter_(-1, valid_targets.unsqueeze(-1), 1.0)
        
        # Apply label smoothing
        if self.smoothing_factor > 0.0:
            # Calculate smoothing parameters
            confidence = 1.0 - self.smoothing_factor
            smoothing_value = self.smoothing_factor / (vocab_size - 1)

            # Create smoothed distribution (FIXED)
            smoothed = one_hot * confidence + smoothing_value * (1 - one_hot)
            
            # Exclude special tokens from smoothing if specified
            if self.special_token_set:
                for token_id in self.special_token_set:
                    if 0 <= token_id < vocab_size:
                        # For special tokens, keep original one-hot distribution
                        special_mask = (targets == token_id).unsqueeze(-1)
                        original_one_hot = torch.zeros_like(smoothed)
                        original_one_hot.scatter_(-1, torch.full_like(targets.unsqueeze(-1), token_id), 1.0)
                        smoothed = torch.where(special_mask, original_one_hot, smoothed)
            
            # Zero out smoothing for invalid positions
            smoothed = smoothed * valid_mask.unsqueeze(-1).float()
        else:
            smoothed = one_hot * valid_mask.unsqueeze(-1).float()
        
        return smoothed
    
    def modify_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply target smoothing to loss calculation.
        
        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size)
            targets: Target token IDs (batch_size, seq_len)
            loss: Original loss value (scalar tensor) - will be replaced with smoothed version
            
        Returns:
            Modified loss using smoothed targets
        """
        if not self.enabled:
            return loss
        
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # Create smoothed target distributions
        smoothed_targets = self._create_smoothed_targets(targets, vocab_size, device)
        
        # Calculate log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Calculate smoothed cross-entropy per position
        smoothed_ce = -(smoothed_targets * log_probs).sum(dim=-1)  # (batch_size, seq_len)

        # Create mask for valid positions (exclude padding if specified)
        if self.exclude_padding:
            valid_mask = (targets != self.padding_token_id) & (targets >= 0) & (targets < vocab_size)
        else:
            valid_mask = (targets >= 0) & (targets < vocab_size)

        # Zero out invalid positions for metrics aggregation; aggregation is handled by pipeline
        per_position_loss = smoothed_ce * valid_mask.float()

        # Store metrics (also compute original CE for reference)
        original_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            ignore_index=self.padding_token_id if self.exclude_padding else -100
        )

        # Compute masked mean for reporting only
        denom = valid_mask.float().sum().clamp_min(1.0)
        smoothed_loss_mean = per_position_loss.sum() / denom

        self._metrics = {
            'original_loss': original_loss.item(),
            'smoothed_loss': smoothed_loss_mean.item(),
            'smoothing_factor': self.smoothing_factor,
            'valid_positions': valid_mask.float().sum().item(),
            'total_positions': targets.numel(),
        }

        # Return per-position loss replacement; pipeline will aggregate
        return {'per_position_loss': smoothed_ce}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get target smoothing metrics from the last forward pass."""
        return self._metrics.copy()
