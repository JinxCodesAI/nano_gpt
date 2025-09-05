# Per-Sample Loss Processing with External Wrongness Factor

This document describes the modifications needed to implement per-sample loss processing with external wrongness factor scaling in the diffusion training system.

## Overview

The current system aggregates loss at the cross-entropy level using `reduction='mean'`. We need to:

1. **Remove aggregation from cross-entropy** - use `reduction='none'` to get per-position losses
2. **Move loss aggregation to main training loop** - aggregate after all per-sample modifications
3. **Apply entropy penalty per-sample** - calculate and apply entropy penalty before aggregation
4. **Integrate wrongness factor scaling** - scale per-sample losses using external metrics

## Current vs Target Architecture

### Current Flow (lines 532-575 in train_run2.py)
```
logits, targets, mask → cross_entropy(reduction='mean') → scalar_loss → entropy_penalty_global → final_loss
```

**Current entropy penalty application (lines 563-575)**:
- `wrong_answer_entropy = calculate_wrong_answer_entropy(logits, Y, training_ctx.extended_vocab_size)`
- Calculates entropy across ALL positions in ALL samples
- Applies single entropy multiplier to the already-aggregated scalar loss

### Target Flow
```
logits, targets, mask → cross_entropy(reduction='none') → per_position_losses →
aggregate_to_per_sample → entropy_penalty_per_sample → wrongness_factor_scaling →
final_aggregation → scalar_loss
```

**Target entropy penalty application**:
- Calculate entropy per sample (only for masked positions in each sample)
- Apply entropy multiplier per sample BEFORE final aggregation
- Apply wrongness factor per sample BEFORE final aggregation

## Implementation Plan

### 1. Modify Cross-Entropy Loss Calculation

**File**: `train_run2.py` (lines 532-561)

**Current Code**:
```python
loss = torch.nn.functional.cross_entropy(
    logits_reshaped[mask_reshaped],
    targets_reshaped[mask_reshaped],
    reduction='mean'  # <-- Remove this aggregation
)
```

**New Code**:
```python
# Get per-position losses without aggregation
position_losses = torch.nn.functional.cross_entropy(
    logits_reshaped[mask_reshaped],
    targets_reshaped[mask_reshaped],
    reduction='none'  # <-- Key change: no aggregation
)
# position_losses shape: (num_masked_positions,)

# Map masked positions back to sample indices
mask_sample_indices = torch.arange(batch_size, device=mask.device).unsqueeze(1).expand(-1, seq_len).view(-1)[mask_reshaped]

# Aggregate losses per sample
per_sample_losses = torch.zeros(batch_size, device=logits.device)
per_sample_mask_counts = torch.zeros(batch_size, device=logits.device)

per_sample_losses.scatter_add_(0, mask_sample_indices, position_losses)
per_sample_mask_counts.scatter_add_(0, mask_sample_indices, torch.ones_like(position_losses))

# Average loss per sample (only for samples with masked positions)
per_sample_losses = per_sample_losses / (per_sample_mask_counts + 1e-8)
# per_sample_losses shape: (batch_size,)
```

### 2. Modify Existing Entropy Utils for Per-Sample Support

**File**: `training_utils/entropy_utils.py` (modify existing file)

**Add new function to support per-sample entropy calculation**:
```python
def calculate_wrong_answer_entropy_per_sample(logits, targets, mask, vocab_size):
    """
    Calculate entropy of wrong answer distributions per sample (reuses existing logic).

    Args:
        logits: Model logits (batch_size, seq_len, vocab_size)
        targets: Target tokens (batch_size, seq_len)
        mask: Boolean mask (batch_size, seq_len) - only calculate for masked positions
        vocab_size: Size of vocabulary

    Returns:
        per_sample_entropies: (batch_size,) - entropy per sample
    """
    batch_size, seq_len, vocab_size_logits = logits.shape
    device = logits.device
    per_sample_entropies = torch.zeros(batch_size, device=device)

    for sample_idx in range(batch_size):
        sample_mask = mask[sample_idx]  # (seq_len,)
        if not sample_mask.any():
            continue

        # Extract masked positions for this sample
        sample_logits = logits[sample_idx:sample_idx+1, :, :]  # Keep batch dim: (1, seq_len, vocab_size)
        sample_targets = targets[sample_idx:sample_idx+1, :]   # Keep batch dim: (1, seq_len)
        sample_mask_expanded = sample_mask.unsqueeze(0)        # (1, seq_len)

        # Apply mask to get only masked positions
        masked_logits = sample_logits[:, sample_mask, :]       # (1, num_masked_positions, vocab_size)
        masked_targets = sample_targets[:, sample_mask]        # (1, num_masked_positions)

        # Reuse existing entropy calculation logic
        if masked_logits.numel() > 0:
            # Reshape to match existing function expectations
            reshaped_logits = masked_logits.view(1, -1, vocab_size_logits)
            reshaped_targets = masked_targets.view(1, -1)

            # Call existing function (it will return scalar for this single sample)
            sample_entropy = calculate_wrong_answer_entropy(reshaped_logits, reshaped_targets, vocab_size)
            per_sample_entropies[sample_idx] = sample_entropy

    return per_sample_entropies
```

### 3. Create Per-Sample Loss Processing Function

**File**: `training_utils/loss_processing.py` (new file)

```python
"""
Per-sample loss processing utilities for diffusion training.
"""

import torch
import math
from .entropy_utils import calculate_wrong_answer_entropy_per_sample, get_current_entropy_penalty

def calculate_per_sample_losses(logits, targets, mask):
    """
    Calculate cross-entropy loss per sample without aggregation.

    Args:
        logits: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len) or (batch_size, seq_len, vocab_size)
        mask: (batch_size, seq_len) - boolean mask

    Returns:
        per_sample_losses: (batch_size,) - average loss per sample
        per_sample_mask_counts: (batch_size,) - number of masked positions per sample
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Reshape tensors
    logits_reshaped = logits.view(-1, vocab_size)
    mask_reshaped = mask.view(-1)

    if targets.dim() == 3:
        # Soft targets
        targets_reshaped = targets.view(-1, vocab_size)
        position_losses = torch.nn.functional.cross_entropy(
            logits_reshaped[mask_reshaped],
            targets_reshaped[mask_reshaped],
            reduction='none'
        )
    else:
        # Hard targets
        targets_reshaped = targets.view(-1)
        position_losses = torch.nn.functional.cross_entropy(
            logits_reshaped[mask_reshaped],
            targets_reshaped[mask_reshaped],
            reduction='none'
        )

    # Map masked positions back to sample indices
    mask_sample_indices = torch.arange(batch_size, device=mask.device).unsqueeze(1).expand(-1, seq_len).view(-1)[mask_reshaped]

    # Aggregate losses per sample
    per_sample_losses = torch.zeros(batch_size, device=logits.device)
    per_sample_mask_counts = torch.zeros(batch_size, device=logits.device)

    per_sample_losses.scatter_add_(0, mask_sample_indices, position_losses)
    per_sample_mask_counts.scatter_add_(0, mask_sample_indices, torch.ones_like(position_losses))

    # Average loss per sample (replicates reduction='mean' behavior per sample)
    per_sample_losses = per_sample_losses / (per_sample_mask_counts + 1e-8)

    return per_sample_losses, per_sample_mask_counts

def apply_per_sample_modifications(per_sample_losses, logits, targets, mask, training_ctx, iter_num, wrongness_factor=None):
    """
    Apply per-sample modifications: entropy penalty and wrongness factor scaling.

    Args:
        per_sample_losses: (batch_size,) - base losses per sample
        logits: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len) or (batch_size, seq_len, vocab_size)
        mask: (batch_size, seq_len)
        training_ctx: TrainingContext
        iter_num: current iteration
        wrongness_factor: (batch_size,) - external wrongness scaling factor, default=None

    Returns:
        modified_losses: (batch_size,) - modified losses per sample
    """
    batch_size = per_sample_losses.shape[0]
    modified_losses = per_sample_losses.clone()

    # 1. Apply entropy penalty per sample (reuses existing entropy calculation)
    if training_ctx.enable_entropy_penalty:
        current_entropy_penalty = get_current_entropy_penalty(iter_num, training_ctx)

        if current_entropy_penalty > 0:
            # Calculate per-sample entropy penalties using existing function
            per_sample_entropies = calculate_wrong_answer_entropy_per_sample(
                logits, targets, mask, training_ctx.extended_vocab_size
            )

            # Calculate entropy multipliers (same logic as original)
            max_wrong_entropy = math.log(training_ctx.extended_vocab_size - 1)
            entropy_penalty_factors = (max_wrong_entropy - per_sample_entropies) / max_wrong_entropy
            entropy_multipliers = 1.0 + current_entropy_penalty * entropy_penalty_factors

            modified_losses = modified_losses * entropy_multipliers

    # 2. Apply wrongness factor scaling
    if wrongness_factor is not None:
        # Ensure wrongness_factor is the right shape and on the right device
        if wrongness_factor.shape[0] != batch_size:
            raise ValueError(f"wrongness_factor shape {wrongness_factor.shape} doesn't match batch_size {batch_size}")

        wrongness_factor = wrongness_factor.to(modified_losses.device)
        modified_losses = modified_losses * wrongness_factor

    return modified_losses
```

### 4. Modify Main Training Loop

**File**: `train_run2.py` (lines 532-575)

**CURRENT CODE TO REPLACE**:
```python
# Lines 532-561: Cross-entropy with reduction='mean'
if training_ctx.training_type == 'unmasking' and mask.any():
    logits_reshaped = logits.view(-1, logits.size(-1))
    mask_reshaped = mask.view(-1)

    if Y.dim() == 3:
        # Soft targets
        targets_reshaped = Y.view(-1, Y.size(-1))
        loss = torch.nn.functional.cross_entropy(
            logits_reshaped[mask_reshaped],
            targets_reshaped[mask_reshaped],
            reduction='mean'  # <-- REMOVE THIS AGGREGATION
        )
    else:
        # Hard targets
        targets_reshaped = Y.view(-1)
        loss = torch.nn.functional.cross_entropy(
            logits_reshaped[mask_reshaped],
            targets_reshaped[mask_reshaped],
            reduction='mean'  # <-- REMOVE THIS AGGREGATION
        )

    # Apply mask ratio weighting if enabled
    if training_ctx.weight_loss_by_mask_ratio:
        mask_ratio = mask.float().mean().item()
        if mask_ratio > 0:
            weight = (1.0 / mask_ratio) ** 0.5
            loss = loss * weight
else:
    if training_ctx.training_type == 'unmasking':
        loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

# Lines 563-575: REMOVE THIS ENTIRE SECTION - entropy penalty applied too late
with timer.time_function('loss_processing'):
    # Apply entropy penalty if enabled
    if training_ctx.enable_entropy_penalty:
        current_entropy_penalty = get_current_entropy_penalty(iter_num, training_ctx)
        if current_entropy_penalty > 0:
            wrong_answer_entropy = calculate_wrong_answer_entropy(logits, Y, training_ctx.extended_vocab_size)
            max_wrong_entropy = math.log(training_ctx.extended_vocab_size - 1)
            entropy_penalty_factor = (max_wrong_entropy - wrong_answer_entropy) / max_wrong_entropy
            entropy_multiplier = 1.0 + current_entropy_penalty * entropy_penalty_factor
            loss = loss * entropy_multiplier  # <-- WRONG: Applied to scalar loss
            update_entropy_multiplier_ema(training_ctx, entropy_multiplier)
        else:
            update_entropy_multiplier_ema(training_ctx, 1.0)
```

**NEW CODE**:
```python
# Apply masking for unmasking training (per-sample processing)
if training_ctx.training_type == 'unmasking' and mask.any():
    from training_utils.loss_processing import calculate_per_sample_losses, apply_per_sample_modifications

    # Step 1: Get per-sample losses without aggregation (replaces reduction='mean')
    per_sample_losses, per_sample_mask_counts = calculate_per_sample_losses(logits, Y, mask)

    # Step 2: Apply mask ratio weighting if enabled (per-sample)
    if training_ctx.weight_loss_by_mask_ratio:
        mask_ratios = mask.float().mean(dim=1)  # (batch_size,) - ratio per sample
        valid_ratios = mask_ratios > 0
        weights = torch.ones_like(mask_ratios)
        weights[valid_ratios] = (1.0 / mask_ratios[valid_ratios]) ** 0.5
        per_sample_losses = per_sample_losses * weights

    # Step 3: Get external wrongness factor (replace with your actual implementation)
    wrongness_factor = getattr(training_ctx, 'wrongness_factor', None)
    if wrongness_factor is None:
        wrongness_factor = torch.ones(training_ctx.batch_size, device=logits.device)

    # Step 4: Apply per-sample modifications (entropy penalty + wrongness factor)
    # This moves entropy penalty calculation BEFORE aggregation
    with timer.time_function('loss_processing'):
        modified_per_sample_losses = apply_per_sample_modifications(
            per_sample_losses, logits, Y, mask, training_ctx, iter_num, wrongness_factor
        )

    # Step 5: Final aggregation to scalar loss (replaces the original reduction='mean')
    valid_samples = per_sample_mask_counts > 0
    if valid_samples.any():
        loss = modified_per_sample_losses[valid_samples].mean()
    else:
        loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

else:
    if training_ctx.training_type == 'unmasking':
        loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
```

### 5. Add Wrongness Factor to TrainingContext

**File**: `training_utils/training_config.py`

Add to TrainingContext class:
```python
# External loss modification
wrongness_factor: torch.Tensor = None  # (batch_size,) external wrongness scaling factor
```

### 6. Integration Points

**In your external system**, you can set the wrongness factor:
```python
# Before training iteration
wrongness_factor = compute_your_wrongness_metrics(batch_data)  # Shape: (batch_size,)
training_ctx.wrongness_factor = wrongness_factor
```

## Key Reuse of Existing Functions

This approach **reuses existing entropy calculation logic** from `entropy_utils.py`:

1. **`calculate_wrong_answer_entropy`** - Core entropy calculation logic is reused
2. **`get_current_entropy_penalty`** - Existing penalty scheduling is preserved
3. **`update_entropy_multiplier_ema`** - EMA tracking can still be used (call with per-sample averages)

**No duplication** - the new `calculate_wrong_answer_entropy_per_sample` function calls the existing `calculate_wrong_answer_entropy` for each sample, preserving all the existing logic for:
- Safe normalization with epsilon
- Handling edge cases (no wrong probabilities)
- Proper entropy calculation

## Benefits of This Approach

1. **Granular Control**: Loss modification happens at the sample level
2. **External Integration**: Easy to plug in external metrics via wrongness_factor
3. **Backward Compatibility**: Maintains scalar loss for optimizer
4. **Code Reuse**: Leverages existing entropy calculation functions
5. **Modular Design**: Each modification is isolated and testable
6. **Performance**: Efficient tensor operations, minimal loops

## Testing Strategy

1. **Verify shapes**: Ensure all intermediate tensors have expected dimensions
2. **Compare aggregated results**: New system should match old system when wrongness_factor=1.0
3. **Test edge cases**: Empty masks, single-sample batches, extreme wrongness factors
4. **Performance benchmarking**: Measure overhead of per-sample processing
5. **Entropy consistency**: Verify per-sample entropy matches global entropy when averaged

## Migration Notes

- **Minimal code changes**: Most existing entropy logic is reused
- The entropy penalty calculation becomes more expensive (per-sample vs global)
- Memory usage increases slightly due to intermediate per-sample tensors
- All existing functionality is preserved when wrongness_factor defaults to 1.0
- **Backward compatibility**: When wrongness_factor=1.0, results should match original implementation
