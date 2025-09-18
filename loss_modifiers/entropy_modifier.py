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
from model import ModelMode


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
            verbose (bool): Enable verbose logging of calculation irregularities (default: False)
        """
        super().__init__(config)
        self.weight = config.get('weight', 1.0)
        self.entropy_threshold = config.get('entropy_threshold', 0.0)
        self.eps = config.get('eps', 1e-8)
        self.verbose = config.get('verbose', False)
        self.verbose = False
        print(f"EntropyModifier: weight={self.weight}, entropy_threshold={self.entropy_threshold}, eps={self.eps}, verbose={self.verbose}")
    
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
        
        # Create mask for wrong answers - with bounds checking
        target_mask = torch.zeros_like(probs, dtype=torch.bool)
        
        # Check for out-of-bounds target indices that cause scatter errors
        # Note: ignore_index (typically -100) should be handled separately
        ignore_index = getattr(self, 'ignore_index', -100)  # Default PyTorch ignore index
        
        # Mask for positions that should be ignored (like padding)
        ignore_mask = targets == ignore_index
        
        # Mask for valid target indices in vocabulary range
        vocab_valid = (targets >= 0) & (targets < vocab_size)
        
        # Overall valid targets: either ignore_index OR valid vocab index
        valid_targets = ignore_mask | vocab_valid
        invalid_target_count = (~valid_targets).sum().item()
        
        if invalid_target_count > 0 or self.verbose:
            min_target = targets.min().item()
            max_target = targets.max().item()
            
            # Analyze target distribution for character-level debugging
            unique_targets = targets.unique()
            negative_targets = (targets < 0).sum().item()
            too_high_targets = (targets >= vocab_size).sum().item()
            ignore_targets = (targets == ignore_index).sum().item()
            
            print(f"[EntropyModifier] Target Analysis:")
            print(f"  Target range: [{min_target}, {max_target}] vs model vocab_size: {vocab_size}")
            print(f"  Invalid targets: {invalid_target_count} total")
            print(f"  - Negative targets: {negative_targets}")  
            print(f"  - Too high (>= {vocab_size}): {too_high_targets}")
            print(f"  - Ignore index ({ignore_index}): {ignore_targets}")
            print(f"  - Unique target values: {len(unique_targets)} distinct")
            
            if len(unique_targets) <= 20:  # Show actual values if reasonable
                print(f"  - Unique values: {sorted(unique_targets.tolist())}")
            
            # Check if this looks like vocab size mismatch
            if max_target < vocab_size // 10:  # Character vocab much smaller than model vocab
                print(f"  - LIKELY ISSUE: Character vocab (~{max_target}) vs large model vocab ({vocab_size})")
                print(f"  - Consider resizing model vocabulary or using correct config")
            
            # For scatter operation, replace invalid indices with valid ones (we'll mask them out later)
            # Clamp any out-of-bounds targets to vocab_size-1 (last valid index)
            safe_targets = torch.clamp(targets, min=0, max=vocab_size-1)
            # But keep ignore_index as 0 for scatter (will be masked out anyway)  
            safe_targets = torch.where(ignore_mask, torch.zeros_like(targets), safe_targets)
        else:
            safe_targets = torch.where(~ignore_mask, targets, torch.zeros_like(targets))
            
        target_mask.scatter_(-1, safe_targets.unsqueeze(-1), True)  # Mark correct answers
        
        # Zero out probabilities of correct answers to focus on wrong answer distribution
        wrong_probs = probs.clone()
        # Only zero out target probabilities for valid (non-ignored) positions AND non-mask-token targets
        valid_vocab_positions = ~ignore_mask  # Positions that are not ignore_index
        
        # For mask tokens (targets >= vocab_size), we might want different handling
        # For now, treat them like regular targets but handle bounds safely
        wrong_probs[target_mask & valid_vocab_positions.unsqueeze(-1)] = 0.0
        
        # Renormalize to get distribution over wrong answers only
        wrong_prob_sum = wrong_probs.sum(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        
        # Valid positions for entropy calculation: non-ignored AND have wrong answer probability
        valid_positions = valid_vocab_positions & (wrong_prob_sum.squeeze(-1) > self.eps)
        wrong_probs = wrong_probs / (wrong_prob_sum + self.eps)
        
        if self.verbose:
            invalid_count = (~valid_positions).sum().item()
            if invalid_count > 0:
                print(f"[EntropyModifier] {invalid_count}/{valid_positions.numel()} positions have no significant wrong answer probability")
        
        # Calculate entropy: H = -sum(p * log(p))
        # Add epsilon to prevent log(0)
        log_probs = torch.log(wrong_probs + self.eps)
        raw_entropy = -(wrong_probs * log_probs).sum(dim=-1)  # (batch_size, seq_len)
        
        # Normalize entropy to make it vocabulary-size independent
        # Count number of wrong tokens with non-negligible probability
        num_wrong_tokens = (wrong_probs > self.eps).sum(dim=-1).float()  # (batch_size, seq_len)
        
        # Safe normalization: only normalize when there are 2+ wrong tokens
        # When num_wrong_tokens <= 1, entropy should be 0 (no diversity possible)
        safe_mask = num_wrong_tokens > 1.0  # Only positions with 2+ wrong tokens can have entropy > 0
        max_possible_entropy = torch.log(torch.clamp(num_wrong_tokens, min=2.0))  # Clamp to avoid log(0) and log(1)
        
        if self.verbose:
            unsafe_count = (~safe_mask).sum().item()
            if unsafe_count > 0:
                print(f"[EntropyModifier] {unsafe_count}/{safe_mask.numel()} positions have ≤1 wrong token (entropy forced to 0)")
        
        # Normalized entropy: ranges from 0 (concentrated) to 1 (uniform)
        entropy_per_token = torch.zeros_like(raw_entropy)  # Start with zeros
        entropy_per_token[safe_mask] = raw_entropy[safe_mask] / max_possible_entropy[safe_mask]  # Only normalize valid positions
        
        # Additional safety: clamp normalized entropy to [0, 1] and handle any NaN/inf
        pre_clamp_entropy = entropy_per_token.clone()
        entropy_per_token = torch.clamp(entropy_per_token, min=0.0, max=1.0)
        non_finite_mask = ~torch.isfinite(entropy_per_token)
        entropy_per_token = torch.where(torch.isfinite(entropy_per_token), entropy_per_token, torch.zeros_like(entropy_per_token))
        
        if self.verbose:
            clamped_count = ((pre_clamp_entropy < 0) | (pre_clamp_entropy > 1)).sum().item()
            non_finite_count = non_finite_mask.sum().item()
            if clamped_count > 0:
                print(f"[EntropyModifier] {clamped_count} entropy values were clamped to [0,1] range")
            if non_finite_count > 0:
                print(f"[EntropyModifier] WARNING: {non_finite_count} non-finite entropy values detected and set to 0")
        
        # Set entropy to 0 for positions where there are no wrong answers with significant probability
        entropy_per_token[~valid_positions] = 0.0
        
        # Do not apply mask here; return per-position entropies unmasked.
        return entropy_per_token
    
    def supports_mode(self, mode: ModelMode) -> bool:
        return mode in (ModelMode.LANGUAGE_MODEL, ModelMode.TOKEN_CLASSIFIER)

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
        
        # Check mode compatibility
        if model_mode and not self.supports_mode(model_mode):
            return loss
        
        # Calculate per-position entropy
        per_position_entropy = self._calculate_wrong_answer_entropy(logits, targets, mask)
        
        # Store metrics (masked mean if mask is provided) - with safety checks
        if mask is not None:
            mask_f = mask.float()
            valid_positions = mask_f.sum()
            mean_entropy = (per_position_entropy * mask_f).sum() / (valid_positions + self.eps)
        else:
            mean_entropy = per_position_entropy.mean()
        
        # Safety checks for metrics to prevent CUDA errors
        mean_entropy_orig = mean_entropy.clone()
        mean_entropy = torch.clamp(mean_entropy, min=0.0, max=1.0)
        if not torch.isfinite(mean_entropy):
            if self.verbose:
                print(f"[EntropyModifier] WARNING: Non-finite mean_entropy detected, set to 0")
            mean_entropy = torch.tensor(0.0, device=mean_entropy.device)
        elif self.verbose and not torch.allclose(mean_entropy, mean_entropy_orig):
            print(f"[EntropyModifier] Mean entropy clamped: {mean_entropy_orig.item():.6f} -> {mean_entropy.item():.6f}")
            
        max_entropy = torch.clamp(per_position_entropy.max(), min=0.0, max=1.0)
        min_entropy = torch.clamp(per_position_entropy.min(), min=0.0, max=1.0)
        entropy_std = per_position_entropy.std()
        if not torch.isfinite(entropy_std):
            if self.verbose:
                print(f"[EntropyModifier] WARNING: Non-finite entropy_std detected, set to 0")
            entropy_std = torch.tensor(0.0, device=entropy_std.device)
        
        # Initialize metrics dict - will be completed at end of modify_loss
        self._metrics = {
            'per_position_mean_entropy': mean_entropy.item(),
            'per_position_max_entropy': max_entropy.item(),
            'per_position_min_entropy': min_entropy.item(),
            'per_position_entropy_std': entropy_std.item(),
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
        # Add safety: ensure sample_entropy is finite and in [0,1] range
        sample_entropy_orig = sample_entropy.clone()
        sample_entropy = torch.clamp(sample_entropy, min=0.0, max=1.0)
        non_finite_samples = ~torch.isfinite(sample_entropy_orig)
        sample_entropy = torch.where(torch.isfinite(sample_entropy), sample_entropy, torch.zeros_like(sample_entropy))
        
        if self.verbose:
            clamped_samples = ((sample_entropy_orig < 0) | (sample_entropy_orig > 1)).sum().item()
            non_finite_samples_count = non_finite_samples.sum().item()
            if clamped_samples > 0:
                print(f"[EntropyModifier] {clamped_samples} per-sample entropies clamped to [0,1]")
            if non_finite_samples_count > 0:
                print(f"[EntropyModifier] WARNING: {non_finite_samples_count} non-finite per-sample entropies set to 0")
        
        clamped_entropy = torch.clamp(sample_entropy, min=self.entropy_threshold)
        per_sample_weights = self.weight / (clamped_entropy + self.eps)
        
        # Safety: clamp weights to reasonable range to prevent extreme values
        weights_orig = per_sample_weights.clone()
        per_sample_weights = torch.clamp(per_sample_weights, min=self.eps, max=1000.0)
        
        # Normalize weights to preserve batch mean: sum(weights) = batch_size
        batch_size = per_sample_weights.size(0)
        weight_sum = per_sample_weights.sum()
        per_sample_weights = per_sample_weights * (batch_size / (weight_sum + self.eps))
        
        if self.verbose:
            extreme_weights = ((weights_orig < self.eps) | (weights_orig > 1000.0)).sum().item()
            if extreme_weights > 0:
                print(f"[EntropyModifier] {extreme_weights} extreme per-sample weights clamped to [{self.eps:.2e}, 1000.0]")
            min_weight, max_weight = per_sample_weights.min().item(), per_sample_weights.max().item()
            normalized_sum = per_sample_weights.sum().item()
            print(f"[EntropyModifier] Per-sample weights range: [{min_weight:.4f}, {max_weight:.4f}]")
            print(f"[EntropyModifier] Normalized weight sum: {normalized_sum:.4f} (target: {batch_size})")

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
        
        # Calculate loss metrics
        original_loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
        final_loss_val = final_loss.item() if torch.isfinite(final_loss) else float('nan')
        loss_ratio = final_loss_val / original_loss_val if original_loss_val != 0 and torch.isfinite(final_loss) else float('nan')
        
        if self.verbose:
            print(f"[EntropyModifier] Loss modification: {original_loss_val:.6f} -> {final_loss_val:.6f} (ratio: {loss_ratio:.4f})")
        
        # Safety check for final loss
        loss_fallback = False
        if not torch.isfinite(final_loss):
            if self.verbose:
                print(f"[EntropyModifier] WARNING: Non-finite final loss detected, falling back to original loss")
            else:
                print(f"WARNING: EntropyModifier produced non-finite loss, falling back to original loss")
            final_loss = loss
            loss_fallback = True

        # Complete metrics with per-sample and weight statistics
        mean_sample_entropy = sample_entropy.mean()
        min_sample_entropy = sample_entropy.min()  
        max_sample_entropy = sample_entropy.max()
        sample_entropy_std = sample_entropy.std()
        mean_weights = per_sample_weights.mean()
        min_weights = per_sample_weights.min()
        max_weights = per_sample_weights.max()
        weight_std = per_sample_weights.std()
        
        # Add comprehensive metrics (preserving the per-position ones from earlier)
        metrics_fixed = 0
        
        # Per-sample entropy stats
        if torch.isfinite(mean_sample_entropy):
            self._metrics['sample_mean_entropy'] = mean_sample_entropy.item()
        else:
            self._metrics['sample_mean_entropy'] = 0.0
            metrics_fixed += 1
            
        if torch.isfinite(min_sample_entropy):
            self._metrics['sample_min_entropy'] = min_sample_entropy.item()
        else:
            self._metrics['sample_min_entropy'] = 0.0
            metrics_fixed += 1
            
        if torch.isfinite(max_sample_entropy):
            self._metrics['sample_max_entropy'] = max_sample_entropy.item()
        else:
            self._metrics['sample_max_entropy'] = 0.0
            metrics_fixed += 1
            
        if torch.isfinite(sample_entropy_std):
            self._metrics['sample_entropy_std'] = sample_entropy_std.item()
        else:
            self._metrics['sample_entropy_std'] = 0.0
            metrics_fixed += 1
        
        # Weight statistics
        if torch.isfinite(mean_weights):
            self._metrics['weight_mean'] = mean_weights.item()
        else:
            self._metrics['weight_mean'] = 1.0
            metrics_fixed += 1
            
        if torch.isfinite(min_weights):
            self._metrics['weight_min'] = min_weights.item()
        else:
            self._metrics['weight_min'] = 1.0
            metrics_fixed += 1
            
        if torch.isfinite(max_weights):
            self._metrics['weight_max'] = max_weights.item()
        else:
            self._metrics['weight_max'] = 1.0
            metrics_fixed += 1
            
        if torch.isfinite(weight_std):
            self._metrics['weight_std'] = weight_std.item()
        else:
            self._metrics['weight_std'] = 0.0
            metrics_fixed += 1
        
        # Loss modification metrics
        self._metrics['original_loss'] = original_loss_val
        self._metrics['final_loss'] = final_loss_val if not loss_fallback else original_loss_val
        self._metrics['loss_ratio'] = loss_ratio if not loss_fallback else 1.0
        self._metrics['loss_fallback'] = loss_fallback
        
        # Batch statistics 
        self._metrics['batch_size'] = batch_size
        self._metrics['weight_sum'] = per_sample_weights.sum().item()
        
        if self.verbose and metrics_fixed > 0:
            print(f"[EntropyModifier] WARNING: {metrics_fixed} metrics were non-finite and corrected")

        return final_loss
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get entropy metrics from the last forward pass."""
        return self._metrics.copy()