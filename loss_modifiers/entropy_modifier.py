"""
Entropy-based loss modifier that weights loss based on wrong answer distribution entropy.

This modifier calculates the entropy of wrong answer distributions per position.
High entropy (uniform wrong answers) indicates good signal-to-noise ratio,
while low entropy (concentrated wrong answers) indicates poor signal-to-noise ratio.

The modifier can be used to weight the loss based on this entropy calculation.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from .base import BaseLossModifier


class EntropyModifier(BaseLossModifier):
    """
    Loss modifier that uses entropy of wrong answer distributions to weight loss.
    
    Calculates entropy of the probability distribution over incorrect tokens for each position.
    High entropy (uniform distribution over wrong answers) indicates high signal-to-noise ratio.
    Low entropy (concentrated distribution over wrong answers) indicates low signal-to-noise ratio.
    
    The loss can be weighted based on this entropy to focus training on positions with
    better signal-to-noise ratios.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize entropy modifier.
        
        Config keys:
            enabled (bool): Whether to enable this modifier
            weight (float): Weight factor for entropy-based loss modification (default: 1.0)
            entropy_threshold (float): Threshold for filtering low-entropy positions (default: 0.0)
            eps (float): Small value to prevent log(0) in entropy calculation (default: 1e-8)
        """
        super().__init__(config)
        self.weight = config.get('weight', 1.0)
        self.entropy_threshold = config.get('entropy_threshold', 0.0)
        self.eps = config.get('eps', 1e-8)
    
    def _calculate_wrong_answer_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate entropy of wrong answer distributions per position.
        
        Args:
            logits: Model logits (batch_size, seq_len, vocab_size)
            targets: Target tokens (batch_size, seq_len)
            mask: Optional boolean mask (batch_size, seq_len) - only calculate for masked positions
            
        Returns:
            per_position_entropies: (batch_size, seq_len) - entropy per position
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
        
        # Create mask for wrong answers
        target_mask = torch.zeros_like(probs, dtype=torch.bool)
        target_mask.scatter_(-1, targets.unsqueeze(-1), True)  # Mark correct answers
        
        # Zero out probabilities of correct answers to focus on wrong answer distribution
        wrong_probs = probs.clone()
        wrong_probs[target_mask] = 0.0
        
        # Renormalize to get distribution over wrong answers only
        wrong_prob_sum = wrong_probs.sum(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        
        # Avoid division by zero - if all probability is on correct answer, entropy is 0
        valid_positions = wrong_prob_sum.squeeze(-1) > self.eps  # (batch_size, seq_len)
        wrong_probs = wrong_probs / (wrong_prob_sum + self.eps)
        
        # Calculate entropy: H = -sum(p * log(p))
        # Add epsilon to prevent log(0)
        log_probs = torch.log(wrong_probs + self.eps)
        entropy_per_token = -(wrong_probs * log_probs).sum(dim=-1)  # (batch_size, seq_len)
        
        # Set entropy to 0 for positions where there are no wrong answers with significant probability
        entropy_per_token[~valid_positions] = 0.0
        
        # Do not apply mask here; return per-position entropies unmasked.
        return entropy_per_token
    
    def modify_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply entropy-based loss modification.
        
        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size)
            targets: Target token IDs (batch_size, seq_len)
            loss: Original loss value (scalar tensor)
            mask: Optional boolean mask for valid positions
            
        Returns:
            Modified loss tensor
        """
        if not self.enabled:
            return loss
        
        # Calculate per-position entropy
        per_position_entropy = self._calculate_wrong_answer_entropy(logits, targets, mask)
        
        # Store metrics (masked mean if mask is provided)
        if mask is not None:
            mask_f = mask.float()
            valid_positions = mask_f.sum()
            mean_entropy = (per_position_entropy * mask_f).sum() / (valid_positions + self.eps)
        else:
            mean_entropy = per_position_entropy.mean()
        
        self._metrics = {
            'mean_entropy': mean_entropy.item(),
            'max_entropy': per_position_entropy.max().item(),
            'min_entropy': per_position_entropy.min().item(),
            'entropy_std': per_position_entropy.std().item(),
        }
        
        # Always apply entropy-based dynamic weighting
        # Calculate per-sample average entropy
        if mask is not None:
            mask_f = mask.float()
            sample_entropy = (per_position_entropy * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + self.eps)
        else:
            sample_entropy = per_position_entropy.mean(dim=1)

        # Weight loss by entropy (higher entropy = higher weight)
        # Apply threshold filtering if specified
        entropy_weights = torch.clamp(sample_entropy, min=self.entropy_threshold)
        batch_weight = entropy_weights.mean() * self.weight

        self._metrics['entropy_weight'] = batch_weight.item()
        return loss * batch_weight
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get entropy metrics from the last forward pass."""
        return self._metrics.copy()