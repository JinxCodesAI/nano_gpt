"""
Utility functions for diffusion training including timing and logging
"""
import time
import torch
import torch.nn.functional as F
from collections import defaultdict


class Timer:
    """Timer class for performance monitoring with context manager support"""
    
    def __init__(self):
        self.times = defaultdict(list)
    
    def time_function(self, name):
        """Context manager for timing function calls"""
        class TimerContext:
            def __init__(self, timer, name):
                self.timer = timer
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, *args):
                elapsed = time.time() - self.start_time
                self.timer.times[self.name].append(elapsed)
        
        return TimerContext(self, name)
    
    def get_average(self, name, last_n=100):
        """Get average time for last N calls"""
        if name not in self.times or not self.times[name]:
            return 0.0
        return sum(self.times[name][-last_n:]) / min(len(self.times[name]), last_n)


def apply_sticky_masking(tokens, rounds, mask_token_id, sticky_p1_p2_multiplier):
    """
    Apply sticky masking algorithm

    Args:
        tokens: Original token sequence (batch_size, seq_len)
        rounds: Number of masking rounds
        mask_token_id: ID of mask token
        sticky_p1_p2_multiplier: Multiplier for p2 = p1 * multiplier

    Returns:
        masked_tokens: Tokens with sticky masking applied
        mask: Boolean mask showing which positions were masked
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device

    # Start with no masks
    masked_tokens = tokens.clone()

    for round_idx in range(rounds):
        # Dynamically sample sticky probabilities each round
        p1 = torch.rand(1).item() / (rounds * 2)  # Sample from (0, 1/(rounds*2))
        p2 = min(1.0, p1 * sticky_p1_p2_multiplier)  # p2 = p1 * multiplier, capped at 1

        # Current mask state
        current_mask = (masked_tokens == mask_token_id)

        # For each position, check if neighbors are masked
        neighbor_masked = torch.zeros_like(current_mask, dtype=torch.bool)

        # Check left neighbor
        neighbor_masked[:, 1:] |= current_mask[:, :-1]
        # Check right neighbor
        neighbor_masked[:, :-1] |= current_mask[:, 1:]

        # Generate random values for masking decision
        rand_vals = torch.rand(batch_size, seq_len, device=device)

        # Apply p1 where neighbors not masked, p2 where neighbors masked
        mask_probs = torch.where(neighbor_masked, p2, p1)
        new_masks = rand_vals < mask_probs

        # Don't mask positions that are already masked
        new_masks = new_masks & ~current_mask

        # Apply new masks
        masked_tokens[new_masks] = mask_token_id

    # Final mask state
    final_mask = (masked_tokens == mask_token_id)
    return masked_tokens, final_mask


def analyze_clustering(mask):
    """Analyze clustering properties of mask patterns"""
    batch_size, seq_len = mask.shape
    cluster_sizes = []

    for batch_idx in range(batch_size):
        mask_seq = mask[batch_idx].cpu().numpy()

        # Find connected components (clusters)
        in_cluster = False
        current_cluster_size = 0

        for pos in range(seq_len):
            if mask_seq[pos]:  # Masked position
                if not in_cluster:
                    in_cluster = True
                    current_cluster_size = 1
                else:
                    current_cluster_size += 1
            else:  # Unmasked position
                if in_cluster:
                    cluster_sizes.append(current_cluster_size)
                    in_cluster = False
                    current_cluster_size = 0

        # Handle cluster at end of sequence
        if in_cluster:
            cluster_sizes.append(current_cluster_size)

    if cluster_sizes:
        avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes)
        max_cluster_size = max(cluster_sizes)
        num_clusters = len(cluster_sizes)
        return {
            'avg_cluster_size': avg_cluster_size,
            'max_cluster_size': max_cluster_size,
            'num_clusters_per_batch': num_clusters / batch_size
        }
    else:
        return {
            'avg_cluster_size': 0,
            'max_cluster_size': 0,
            'num_clusters_per_batch': 0
        }


def analyze_masking_patterns_with_transition(mask, iter_num, sticky_transition_start, sticky_transition_end):
    """Analyze masking patterns during independent->sticky transition"""
    mask_ratio = mask.float().mean().item()

    # Calculate current transition state
    if iter_num < sticky_transition_start:
        transition_state = "independent"
        sticky_ratio = 0.0
    elif iter_num >= sticky_transition_end:
        transition_state = "sticky"
        sticky_ratio = 1.0
    else:
        progress = (iter_num - sticky_transition_start) / (sticky_transition_end - sticky_transition_start)
        sticky_ratio = progress
        transition_state = f"transition ({sticky_ratio:.2f})"

    # Analyze clustering (more relevant during sticky phase)
    if sticky_ratio > 0.1:  # Only analyze clusters when some sticky masking present
        cluster_stats = analyze_clustering(mask)
        return {
            'mask_ratio': mask_ratio,
            'transition_state': transition_state,
            'sticky_ratio': sticky_ratio,
            **cluster_stats
        }
    else:
        return {
            'mask_ratio': mask_ratio,
            'transition_state': transition_state,
            'sticky_ratio': sticky_ratio
        }


def log_masking_stats(mask, iter_num, log_interval, sticky_transition_start=None, sticky_transition_end=None):
    """Log statistics about masking patterns"""
    if sticky_transition_start is not None and sticky_transition_end is not None:
        # Enhanced logging with transition tracking
        masking_stats = analyze_masking_patterns_with_transition(mask, iter_num, sticky_transition_start, sticky_transition_end)
        if iter_num % (log_interval * 10) == 0:  # Less frequent detailed stats
            print(f"Masking: {masking_stats}")
    else:
        # Original simple logging
        mask_ratio = mask.float().mean().item()
        batch_size, seq_len = mask.shape

        # Count consecutive masked regions
        mask_np = mask.cpu().numpy()
        consecutive_regions = []
        for batch_idx in range(batch_size):
            regions = []
            current_length = 0
            for pos in range(seq_len):
                if mask_np[batch_idx, pos]:
                    current_length += 1
                else:
                    if current_length > 0:
                        regions.append(current_length)
                        current_length = 0
            if current_length > 0:
                regions.append(current_length)
            consecutive_regions.extend(regions)

        avg_region_length = sum(consecutive_regions) / len(consecutive_regions) if consecutive_regions else 0

        if iter_num % (log_interval * 10) == 0:  # Less frequent detailed stats
            print(f"Masking stats: {mask_ratio:.3f} ratio, {avg_region_length:.1f} avg region length")


# Multi-phase training management functions
def get_current_phase(iter_num, n1_iterations, n2_iterations, n3_iterations):
    """Determine current training phase"""
    if iter_num < n1_iterations:
        return 1
    elif iter_num < n1_iterations + n2_iterations:
        return 2
    elif iter_num < n1_iterations + n2_iterations + n3_iterations:
        return 3
    else:
        return 3  # Continue phase 3 beyond n3_iterations


def in_entropy_penalty_phase(iter_num, n1_iterations, n4_iterations):
    """Check if entropy penalty should be applied (concurrent with phases 2-3)"""
    phase_2_3_start = n1_iterations
    phase_4_end = n1_iterations + n4_iterations  # N4 < N2 + N3 constraint
    return phase_2_3_start <= iter_num < phase_4_end


def get_entropy_penalty_strength(iter_num, n1_iterations, n4_iterations, entropy_multiplier_max, entropy_multiplier_min):
    """Calculate current entropy penalty strength with adaptive decay"""
    if not in_entropy_penalty_phase(iter_num, n1_iterations, n4_iterations):
        return 0.0

    # Calculate progress through entropy penalty phase
    phase_4_start = n1_iterations
    phase_4_progress = (iter_num - phase_4_start) / n4_iterations
    phase_4_progress = min(1.0, max(0.0, phase_4_progress))  # Clamp to [0, 1]

    # Linear decay from max to min
    current_strength = entropy_multiplier_max - phase_4_progress * (entropy_multiplier_max - entropy_multiplier_min)
    return current_strength


def get_phase_info(iter_num, n1_iterations, n2_iterations, n3_iterations, n4_iterations, entropy_multiplier_max, entropy_multiplier_min):
    """Get comprehensive information about current training phase"""
    current_phase = get_current_phase(iter_num, n1_iterations, n2_iterations, n3_iterations)
    has_entropy_penalty = in_entropy_penalty_phase(iter_num, n1_iterations, n4_iterations)
    entropy_strength = get_entropy_penalty_strength(iter_num, n1_iterations, n4_iterations, entropy_multiplier_max, entropy_multiplier_min)

    return {
        'phase': current_phase,
        'has_entropy_penalty': has_entropy_penalty,
        'entropy_strength': entropy_strength,
        'phase_name': f"Phase {current_phase}" + (" + Entropy Penalty" if has_entropy_penalty else "")
    }


# Soft targets system for multi-phase training
def create_soft_targets(hard_targets, vocab_size, mask, mask_token_id, phase, iter_num, n1_iterations, n2_iterations):
    """
    Create soft targets based on training phase

    Args:
        hard_targets: Original hard token targets (batch_size, seq_len)
        vocab_size: Size of vocabulary
        mask: Boolean mask indicating masked positions
        mask_token_id: ID of mask token
        phase: Current training phase (1, 2, or 3)
        iter_num: Current iteration number
        n1_iterations: Number of Phase 1 iterations
        n2_iterations: Number of Phase 2 iterations

    Returns:
        soft_targets: Soft probability distributions or hard targets
        is_soft: Boolean indicating if targets are soft
    """
    batch_size, seq_len = hard_targets.shape
    device = hard_targets.device

    if phase == 1:
        # Phase 1: Identity task
        # Unmasked positions: predict same token (hard targets)
        # Masked positions: uniform distribution (soft targets)
        soft_targets = torch.zeros(batch_size, seq_len, vocab_size, device=device)

        # For unmasked positions, use one-hot encoding of original tokens
        unmasked_positions = ~mask
        if unmasked_positions.any():
            soft_targets[unmasked_positions] = F.one_hot(hard_targets[unmasked_positions], vocab_size).float()

        # For masked positions, use uniform distribution (excluding mask token)
        masked_positions = mask
        if masked_positions.any():
            uniform_prob = 1.0 / (vocab_size - 1)  # Exclude mask token
            soft_targets[masked_positions] = uniform_prob
            # Set mask token probability to 0
            batch_indices, seq_indices = torch.where(masked_positions)
            soft_targets[batch_indices, seq_indices, mask_token_id] = 0.0

        return soft_targets, True

    elif phase == 2:
        # Phase 2: Gradual target increase
        # Calculate progress through Phase 2
        phase_2_start = n1_iterations
        phase_2_progress = (iter_num - phase_2_start) / n2_iterations
        phase_2_progress = min(1.0, max(0.0, phase_2_progress))  # Clamp to [0, 1]

        soft_targets = torch.zeros(batch_size, seq_len, vocab_size, device=device)

        # For unmasked positions, always use identity (one-hot)
        unmasked_positions = ~mask
        if unmasked_positions.any():
            soft_targets[unmasked_positions] = F.one_hot(hard_targets[unmasked_positions], vocab_size).float()

        # For masked positions, gradually increase correct token probability
        masked_positions = mask
        if masked_positions.any():
            # Start with uniform distribution (excluding mask token)
            uniform_prob = 1.0 / (vocab_size - 1)
            soft_targets[masked_positions] = uniform_prob

            # Get batch and sequence indices for masked positions
            batch_indices, seq_indices = torch.where(masked_positions)
            soft_targets[batch_indices, seq_indices, mask_token_id] = 0.0

            # Gradually increase correct token probability
            correct_tokens = hard_targets[masked_positions]
            current_correct_prob = uniform_prob + phase_2_progress * (1.0 - uniform_prob)

            # Redistribute remaining probability uniformly among other tokens
            remaining_prob = 1.0 - current_correct_prob
            other_prob = remaining_prob / (vocab_size - 2)  # Exclude correct token and mask token

            # Set all probabilities to other_prob, then override correct and mask tokens
            soft_targets[masked_positions] = other_prob
            soft_targets[batch_indices, seq_indices, mask_token_id] = 0.0
            soft_targets[batch_indices, seq_indices, correct_tokens] = current_correct_prob

        return soft_targets, True

    else:
        # Phase 3: Standard training with hard targets
        return hard_targets, False


def compute_loss_with_soft_targets(logits, targets, is_soft, mask=None):
    """
    Compute loss with support for both hard and soft targets

    Args:
        logits: Model output logits (batch_size, seq_len, vocab_size)
        targets: Either hard targets (long tensor) or soft targets (float tensor)
        is_soft: Boolean indicating if targets are soft
        mask: Optional mask for computing loss only on specific positions

    Returns:
        loss: Computed loss
    """
    if is_soft:
        # Soft targets: use KL divergence loss
        log_probs = F.log_softmax(logits, dim=-1)

        if mask is not None:
            # Apply mask to both log_probs and targets
            log_probs_masked = log_probs[mask]
            targets_masked = targets[mask]
            loss = F.kl_div(log_probs_masked, targets_masked, reduction='batchmean')
        else:
            # Compute loss on all positions
            log_probs_flat = log_probs.view(-1, log_probs.size(-1))
            targets_flat = targets.view(-1, targets.size(-1))
            loss = F.kl_div(log_probs_flat, targets_flat, reduction='batchmean')
    else:
        # Hard targets: use standard cross-entropy loss
        if mask is not None:
            # Compute loss only on masked positions
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            mask_flat = mask.view(-1)
            loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])
        else:
            # Compute loss on all positions
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

    return loss


class EntropyPenaltyModifier:
    """
    Applies penalty to loss for low-entropy (overconfident) wrong guesses
    during unmasking tasks. Implements "override and recalculate" strategy
    to be fully argmax-free and correctly handle soft targets.
    """
    def __init__(self, penalty_strength, vocab_size):
        if vocab_size is None:
            raise ValueError("EntropyPenaltyModifier requires vocab_size for max entropy calculation.")

        self.penalty_strength = penalty_strength
        self.vocab_size = vocab_size
        self.max_entropy = torch.log(torch.tensor(self.vocab_size, dtype=torch.float32))

    def apply_penalty(self, loss, logits, targets, inputs, mask_token_id, is_soft_targets=False):
        """
        Apply entropy penalty by modifying loss

        Args:
            loss: Original loss value
            logits: Model output logits (batch_size, seq_len, vocab_size)
            targets: Target tokens or probability distributions
            inputs: Input tokens (batch_size, seq_len)
            mask_token_id: ID of mask token
            is_soft_targets: Whether targets are soft probability distributions

        Returns:
            modified_loss: Loss with entropy penalty applied
            diagnostics: Dictionary with penalty diagnostics
        """
        if self.penalty_strength <= 0:
            return loss, {}

        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        # Identify unmasking task positions
        if is_soft_targets:
            # For soft labels, unmask where input is masked
            unmask_task_mask = (inputs == mask_token_id)
        else:
            # For hard labels, unmask where input is masked but target is not mask token
            unmask_task_mask = (inputs == mask_token_id) & (targets != mask_token_id)

        if not unmask_task_mask.any():
            return loss, {'avg_entropy_penalty_factor': 0.0, 'penalty_entropy_strength': self.penalty_strength}

        # Calculate entropy of incorrect predictions
        probs = F.softmax(logits, dim=-1)

        # Get target probability distribution
        if is_soft_targets:
            target_probs = targets  # Already probability distribution
        else:
            target_probs = F.one_hot(targets, num_classes=vocab_size).float()

        # Calculate incorrect distribution (model confidence in wrong answers)
        incorrect_distribution = F.relu(probs - target_probs)

        # Normalize to valid probability distribution
        epsilon = 1e-9
        incorrect_sum = incorrect_distribution.sum(dim=-1, keepdim=True)
        incorrect_distribution = incorrect_distribution / (incorrect_sum + epsilon)

        # Calculate entropy of incorrect predictions
        # Add small epsilon to prevent log(0)
        log_incorrect = torch.log(incorrect_distribution + epsilon)
        entropy_of_incorrect = -(incorrect_distribution * log_incorrect).sum(dim=-1)

        # Normalize penalty (low entropy = high penalty)
        max_entropy_device = self.max_entropy.to(device)
        normalized_entropy_penalty = (max_entropy_device - entropy_of_incorrect) / max_entropy_device
        normalized_entropy_penalty = torch.clamp(normalized_entropy_penalty, 0.0, 1.0)

        # Apply penalty multiplicatively to loss
        # Calculate average penalty for unmasking positions
        penalty_values = normalized_entropy_penalty[unmask_task_mask]
        if penalty_values.numel() > 0:
            avg_penalty = penalty_values.mean()
            penalty_multiplier = 1.0 + self.penalty_strength * avg_penalty
            modified_loss = loss * penalty_multiplier

            diagnostics = {
                'avg_entropy_penalty_factor': avg_penalty.item(),
                'penalty_entropy_strength': self.penalty_strength,
                'penalty_multiplier': penalty_multiplier.item()
            }
        else:
            modified_loss = loss
            diagnostics = {
                'avg_entropy_penalty_factor': 0.0,
                'penalty_entropy_strength': self.penalty_strength,
                'penalty_multiplier': 1.0
            }

        return modified_loss, diagnostics
