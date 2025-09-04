# Intelligent Remasking Implementation Plan

## Overview

This document provides a step-by-step plan to implement intelligent remasking (selfmasking) functionality in both `sample.py` and `evaluate_models.py`. This will enable the `randomness_strength` parameter to work even without a separate remasking model by using the main model itself to identify surprising tokens for remasking.

## What We Want to Achieve

### Current Problem
- The `randomness_strength` parameter only works when a separate remasking model is provided
- Without a remasking model, the system falls back to pure random remasking, ignoring `randomness_strength`
- This makes model evaluation inconsistent and limits the effectiveness of the diffusion process

### Desired Solution
Implement **selfmasking** where:
1. **Unmasking Phase**: The model predicts tokens for masked positions
2. **Selfmasking Phase**: The same model evaluates the current text and identifies "surprising" tokens (low probability tokens)
3. **Intelligent Remasking**: The system selects tokens for remasking based on:
   - `randomness_strength = 1.0`: Pure random selection
   - `randomness_strength = 0.0`: Pure model-guided selection (most surprising tokens)
   - `randomness_strength = 0.4`: 40% random + 60% model-guided selection

## Why This Change is Important

### Benefits
1. **Consistent Behavior**: `randomness_strength` will work regardless of remasking model availability
2. **Better Quality**: Model-guided remasking focuses on tokens the model is uncertain about
3. **Unified Codebase**: Same intelligent remasking logic across all generation scripts
4. **Improved Evaluation**: Model comparisons will be more meaningful with consistent remasking

### Target Files
- `sample.py`: Main sampling script used for text generation
- `evaluate_models.py`: Model evaluation script that compares multiple models
- `sample_utils.py`: Core utilities that need enhancement

## Implementation Plan

### Phase 1: Enhance Core Utilities

#### Step 1.1: Update `sample_utils.py`
**File**: `sample_utils.py`
**Function**: `apply_remasking()`

**Changes Needed**:
1. Add new parameters:
   ```python
   def apply_remasking(tokens, remask_ratio, remasking_model, randomness_strength, mask_token_id, device, 
                      base_model=None, intelligent_remasking=False, verbose=False):
   ```

2. Add intelligent remasking logic when `remasking_model is None`:
   ```python
   if remasking_model is None:
       if intelligent_remasking and base_model is not None:
           # Selfmasking: Use base model to identify surprising tokens
           with torch.no_grad():
               dummy_targets = torch.zeros_like(tokens)
               logits, _ = base_model(tokens, dummy_targets)
               
               # Calculate surprise scores: 1 - P(current_token)
               probs = torch.softmax(logits, dim=-1)
               current_token_probs = probs.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
               surprise_scores = 1 - current_token_probs
               
               # Combine with randomness
               for batch_idx in range(batch_size):
                   # Get remaskable positions
                   unmasked_positions = (tokens[batch_idx] != mask_token_id)
                   unmasked_indices = torch.where(unmasked_positions)[0]
                   
                   if len(unmasked_indices) > 0:
                       # Get surprise scores for unmasked positions
                       model_scores = surprise_scores[batch_idx, unmasked_indices]
                       random_scores = torch.rand(len(unmasked_indices), device=device)
                       
                       # Combine: randomness_strength * random + (1-randomness_strength) * model
                       combined_scores = randomness_strength * random_scores + (1 - randomness_strength) * model_scores
                       
                       # Select top positions for remasking
                       num_to_remask = min(additional_masks_needed[batch_idx].item(), len(unmasked_indices))
                       _, top_indices = torch.topk(combined_scores, num_to_remask)
                       remask_indices = unmasked_indices[top_indices]
                       tokens[batch_idx, remask_indices] = mask_token_id
       else:
           # Fallback to pure random remasking
           # ... existing random remasking code ...
   ```

#### Step 1.2: Update `apply_remasking_step()`
**File**: `sample_utils.py`
**Function**: `apply_remasking_step()`

**Changes Needed**:
1. Add new parameters to function signature
2. Pass through the new parameters to `apply_remasking()`

### Phase 2: Update `sample.py`

#### Step 2.1: Add Configuration Parameters
**File**: `sample.py`
**Location**: Configuration section (around line 44)

**Add**:
```python
# Intelligent remasking parameters
intelligent_remasking = True  # Enable selfmasking when no remasking model is available
```

#### Step 2.2: Update `diffusion_generate()` Function
**File**: `sample.py`
**Function**: `diffusion_generate()`

**Changes Needed**:
1. Add `intelligent_remasking` parameter to function signature
2. Pass `base_model=model` and `intelligent_remasking=intelligent_remasking` to `apply_remasking_step()`

#### Step 2.3: Update Function Call
**File**: `sample.py`
**Location**: Main generation call (around line 518)

**Update**:
```python
generated_tokens = diffusion_generate(
    model=model,
    batch_size=num_samples,
    total_length=sequence_length,
    iterations=diffusion_iterations,
    remasking_model=remasking_model,
    mask_token_id=mask_token_id,
    randomness_strength=randomness_strength,
    decode_fn=decode,
    decode_mask_fn=decode_with_mask_char,
    verbose=use_verbose_logging,
    temperature=temperature,
    top_p=top_p,
    schedule_type=schedule_type,
    masking_ratios=masking_ratios,
    repetition_penalty=repetition_penalty,
    repetition_window=repetition_window,
    log_debug=log_debug,
    intelligent_remasking=intelligent_remasking  # Add this parameter
)
```

### Phase 3: Update `evaluate_models.py`

#### Step 3.1: Add Configuration Parameter
**File**: `evaluate_models.py`
**Location**: `EVALUATION_CONFIG` dictionary (around line 260)

**Add**:
```python
'intelligent_remasking': True,  # Enable selfmasking for consistent evaluation
```

#### Step 3.2: Update `diffusion_generate()` Function
**File**: `evaluate_models.py`
**Function**: `diffusion_generate()` (around line 121)

**Changes Needed**:
1. Add `intelligent_remasking` parameter to function signature
2. Pass the parameter through to `apply_remasking_step()`

#### Step 3.3: Update `SampleGenerator` Class
**File**: `evaluate_models.py`
**Class**: `SampleGenerator`
**Method**: `generate_samples_for_model()`

**Changes Needed**:
1. Pass `base_model=model` and `intelligent_remasking=self.config['intelligent_remasking']` to `diffusion_generate()`

## Implementation Order

1. **Start with `sample_utils.py`** - Core functionality
2. **Update `sample.py`** - Test with single model
3. **Update `evaluate_models.py`** - Test with multiple models
4. **Add comprehensive testing** - Ensure reliability
5. **Documentation updates** - Update user guides

## Expected Outcomes

After implementation:
- `randomness_strength` parameter will work consistently across all scripts
- Model evaluation will be more meaningful and consistent
- Users can control the balance between random and intelligent remasking
- The diffusion process will be more effective at iterative refinement

## Backward Compatibility

The implementation maintains full backward compatibility:
- Existing remasking models continue to work as before
- Default behavior remains unchanged when `intelligent_remasking=False`
- All existing configuration parameters are preserved

## Detailed Code Changes

### 1. Enhanced `apply_remasking()` in `sample_utils.py`

```python
def apply_remasking(tokens, remask_ratio, remasking_model, randomness_strength, mask_token_id, device,
                   base_model=None, intelligent_remasking=False, verbose=False):
    """
    Apply remasking using random, model-guided, or intelligent selfmasking selection

    Args:
        tokens: Current token sequence (batch_size, seq_len)
        remask_ratio: Fraction of tokens to remask
        remasking_model: Optional separate remasking model
        randomness_strength: Balance between random (1.0) and model-guided (0.0)
        mask_token_id: ID of mask token
        device: Device to run on
        base_model: Base model for intelligent remasking when remasking_model is None
        intelligent_remasking: Enable selfmasking using base_model
        verbose: Whether to print debug info

    Returns:
        tokens: Updated token sequence with remasked positions
    """
    # ... existing setup code ...

    if remasking_model is None:
        if intelligent_remasking and base_model is not None:
            # SELFMASKING: Use base model to identify surprising tokens
            if verbose:
                print(f"  Using intelligent selfmasking: randomness={randomness_strength:.2f}")

            with torch.no_grad():
                # Get base model predictions for current tokens
                dummy_targets = torch.zeros_like(tokens)
                logits, _ = base_model(tokens, dummy_targets)

                # Calculate surprise scores: 1 - P(current_token)
                probs = torch.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
                current_token_probs = probs.gather(2, tokens.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)
                surprise_scores = 1 - current_token_probs  # Higher = more surprising

                for batch_idx in range(batch_size):
                    num_additional = additional_masks_needed[batch_idx].item()
                    if num_additional > 0:
                        # Only consider unmasked positions for remasking
                        unmasked_positions = (tokens[batch_idx] != mask_token_id)
                        unmasked_indices = torch.where(unmasked_positions)[0]

                        if len(unmasked_indices) > 0:
                            # Get scores for unmasked positions only
                            model_scores = surprise_scores[batch_idx, unmasked_indices]
                            random_scores = torch.rand(len(unmasked_indices), device=device)

                            # Combine random and model scores
                            combined_scores = (randomness_strength * random_scores +
                                             (1 - randomness_strength) * model_scores)

                            # Select top positions by combined score
                            num_to_select = min(num_additional, len(unmasked_indices))
                            _, top_indices = torch.topk(combined_scores, num_to_select)
                            remask_indices = unmasked_indices[top_indices]
                            tokens[batch_idx, remask_indices] = mask_token_id

                            if verbose and batch_idx == 0:  # Show stats for first sample
                                avg_surprise = model_scores[top_indices].mean().item()
                                avg_random = random_scores[top_indices].mean().item()
                                print(f"    Selected tokens - avg surprise: {avg_surprise:.3f}, avg random: {avg_random:.3f}")
        else:
            # Pure random remasking (existing code)
            # ... existing random remasking implementation ...
    else:
        # Model-guided remasking with separate remasking model (existing code)
        # ... existing remasking model implementation ...

    return tokens
```

### 2. Updated `apply_remasking_step()` in `sample_utils.py`

```python
def apply_remasking_step(tokens, prediction_tokens, iteration, iterations, schedule_type, masking_ratios,
                        start_ratio, end_ratio, remasking_model, randomness_strength, mask_token_id,
                        device, base_model=None, intelligent_remasking=False, verbose=False):
    """
    Apply remasking step with scheduling and intelligent remasking support
    """
    # ... existing schedule calculation ...

    tokens = apply_remasking(tokens, remask_ratio, remasking_model, randomness_strength,
                           mask_token_id, device, base_model, intelligent_remasking, verbose)

    # ... existing debug output ...
    return tokens
```

### 3. Configuration Updates

#### `sample.py` Configuration:
```python
# Intelligent remasking parameters
intelligent_remasking = True  # Enable selfmasking when no remasking model is available
```

#### `evaluate_models.py` Configuration:
```python
EVALUATION_CONFIG = {
    # ... existing config ...
    'intelligent_remasking': True,  # Enable selfmasking for consistent evaluation
    # ... rest of config ...
}
```

## Risk Assessment

### Low Risk
- Backward compatibility is maintained
- Changes are additive, not destructive
- Existing functionality remains unchanged

### Medium Risk
- Performance impact from additional model forward passes
- GPU memory usage may increase slightly
- Need thorough testing of edge cases

### Mitigation Strategies
1. **Performance**: Cache model outputs when possible
2. **Memory**: Use gradient checkpointing if needed
3. **Testing**: Comprehensive test suite covering all scenarios

## Success Metrics

1. **Functional**: `randomness_strength` parameter works without remasking model
2. **Quality**: Generated text quality improves with intelligent remasking
3. **Consistency**: Model evaluation results are more stable and meaningful
4. **Performance**: <10% performance degradation acceptable
5. **Compatibility**: All existing scripts continue to work unchanged

## Future Enhancements

After successful implementation, consider:
1. **Adaptive Randomness**: Automatically adjust `randomness_strength` based on model confidence
2. **Position-Aware Remasking**: Consider token position importance (e.g., protect sentence boundaries)
3. **Multi-Model Consensus**: Use multiple models for remasking decisions
4. **Learning-Based Remasking**: Train a lightweight remasking head on top of base models
