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
    Loss modifier that uses normalized entropy of wrong answer distributions to weight loss.
    
    Calculates the normalized entropy of probability distribution over incorrect tokens for each position.
    The entropy is vocabulary-size independent, ranging from 0 to 1:
    - High entropy (1.0): uniform distribution over wrong answers → high signal-to-noise ratio
    - Low entropy (0.0): concentrated distribution over wrong answers → low signal-to-noise ratio
    
    The loss is weighted based on this normalized entropy, with lower entropy receiving higher weights
    to penalize samples where wrong answer probability concentrates on few tokens.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize entropy modifier.
        
        Config keys:
            enabled (bool): Whether to enable this modifier
            weight (float): Weight factor for entropy-based loss modification (default: 1.0)
            entropy_threshold (float): Minimum normalized entropy floor for weighting (default: 0.0, range: [0,1])
            eps (float): Small value to prevent log(0) in entropy calculation (default: 1e-8)
        """
        super().__init__(config)
        self.weight = config.get('weight', 1.0)
        self.entropy_threshold = config.get('entropy_threshold', 0.0)
        self.eps = config.get('eps', 1e-8)
        print(f"EntropyModifier: weight={self.weight}, entropy_threshold={self.entropy_threshold}, eps={self.eps}")
    
    def _calculate_wrong_answer_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate normalized entropy of wrong answer distributions per position.
        
        The entropy is normalized to be vocabulary-size independent, ranging from 0 to 1:
        - 0: All wrong-answer probability concentrated on a single token (low diversity)
        - 1: Wrong-answer probability uniformly distributed (high diversity)
        
        Args:
            logits: Model logits (batch_size, seq_len, vocab_size)
            targets: Target tokens (batch_size, seq_len)
            mask: Optional boolean mask (batch_size, seq_len) - only calculate for masked positions
            
        Returns:
            per_position_entropies: (batch_size, seq_len) - normalized entropy per position [0,1]
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
        raw_entropy = -(wrong_probs * log_probs).sum(dim=-1)  # (batch_size, seq_len)
        
        # Normalize entropy to make it vocabulary-size independent
        # Count number of wrong tokens with non-negligible probability
        num_wrong_tokens = (wrong_probs > self.eps).sum(dim=-1).float()  # (batch_size, seq_len)
        max_possible_entropy = torch.log(num_wrong_tokens + self.eps)  # Maximum entropy for uniform distribution
        
        # Normalized entropy: ranges from 0 (concentrated) to 1 (uniform)
        entropy_per_token = raw_entropy / (max_possible_entropy + self.eps)  # (batch_size, seq_len)
        
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
        
        # Always apply entropy-based dynamic weighting at per-sample level.
        # Compute per-sample average entropy
        if mask is not None:
            mask_f = mask.float()
            sample_entropy = (per_position_entropy * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + self.eps)
        else:
            sample_entropy = per_position_entropy.mean(dim=1)

        # Compute per-sample weights: lower entropy -> higher weight (punish concentrated wrong answers)
        # w_i = weight * 1 / (clamp(H_i, min=threshold) + eps)
        clamped_entropy = torch.clamp(sample_entropy, min=self.entropy_threshold)
        per_sample_weights = self.weight / (clamped_entropy + self.eps)

        # Obtain per-position loss to aggregate per-sample
        per_position_loss = kwargs.get('per_position_loss', None)
        if per_position_loss is None:
            # Fallback: compute per-position CE without ignore handling
            batch_size, seq_len, vocab_size = logits.shape
            flat_logits = logits.view(-1, vocab_size)
            flat_targets = targets.view(-1)
            per_position_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none').view(batch_size, seq_len)

        # Aggregate to per-sample losses using mask when available
        if mask is not None:
            mask_f = mask.float()
            per_sample_loss = (per_position_loss * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + self.eps)
        else:
            per_sample_loss = per_position_loss.mean(dim=1)

        # Apply per-sample weights and average over batch
        weighted_losses = per_sample_loss * per_sample_weights
        final_loss = weighted_losses.mean()

        # Metrics
        self._metrics['mean_entropy'] = sample_entropy.mean().item()
        self._metrics['min_entropy'] = sample_entropy.min().item()
        self._metrics['max_entropy'] = sample_entropy.max().item()
        self._metrics['entropy_weight_mean'] = per_sample_weights.mean().item()

        return final_loss
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get entropy metrics from the last forward pass."""
        return self._metrics.copy()