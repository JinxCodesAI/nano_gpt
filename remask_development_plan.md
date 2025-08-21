# Remasking Model Development Plan

## Overview
Currently, `sample.py` uses random remasking in the diffusion generation process (lines 120-123). This plan outlines training a separate model to learn intelligent remasking patterns, which should improve the quality and coherence of generated text.

## Key Design Principles
1. **Same Architecture**: Both models use identical transformer architecture for future merging
2. **Extended Vocabulary**: Add `wrong_token_id` alongside `mask_token_id` for distinct corruption
3. **Token Prediction**: Remasking model predicts tokens (not binary), keeping architecture consistent
4. **No Insertions**: Only replacement corruption, maintain sequence length
5. **Full Loss**: Loss computed on ALL positions (error to mask correct OR not mask incorrect)

## Current State Analysis

### Current Training (Unmasking)
- **Task**: Given masked input, predict original tokens at masked positions
- **Input**: `masked_x` (some tokens replaced with `mask_token_id`)
- **Target**: `y = x.clone()` (original sequence)
- **Loss**: Only on masked positions (where `mask[i] == True`)
- **Objective**: Learn to "unmask" or "denoise" corrupted text
- **Vocabulary**: `vocab_size + 1` (original + `mask_token_id`)

### Current Sampling (Random Remasking)
```python
# sample.py lines 120-123
remask_indices = torch.randperm(total_length)[:num_to_remask]
positions_to_remask = all_positions[remask_indices]
tokens[0, positions_to_remask] = mask_token_id
```

## Proposed Remasking Training

### New Training Type: 'remasking'
- **Task**: Given corrupted sequence, predict correct tokens at ALL positions
- **Input**: Sequence with some positions replaced by `wrong_token_id`
- **Target**: Original tokens at correct positions, `wrong_token_id` at corrupted positions
- **Loss Strategy**: Two approaches to test performance
- **Objective**: Learn to identify and fix incorrect tokens
- **Vocabulary**: `vocab_size + 2` (original + `mask_token_id` + `wrong_token_id`)

### Simple Mode Implementation

#### Data Generation for Remasking Training
1. **Start with original sequence**: `x` (clean text)
2. **Select positions to corrupt**: Same random strategy as current unmasking training
3. **Create corrupted input**: Replace selected positions with random vocabulary tokens
4. **Target**: Original tokens at correct positions, `wrong_token_id` at corrupted positions
5. **No insertions**: Only replacement, maintain sequence length

```
Original:     "Hello world, this is a test"
Corrupt mask: [False, True, False, True, False, False, True]
Input (x):    "Hello fis, this bar a foo"  // fis, bar, foo are random wrong tokens
Target (y):   "Hello [WRONG], this [WRONG] a [WRONG]"  # Original sequence
Mask:         [True, True, True, True, True, True]  # For loss computation options 
```

Mask is all true because if model predicts lat say [WRONG, WRONG, WRONG, WRONG, WRONG, WRONG] we want to penalize it for masking 'Hello' 'this' and 'a'.

#### Key Differences from Unmasking Training
| Aspect | Unmasking Training | Remasking Training |
|--------|-------------------|-------------------|
| Input corruption | Replace with `mask_token_id` | Replace with `wrong_token_id` |
| Target | Original tokens | Original + `wrong_token_id` at corrupted positions |
| Loss function | Cross-entropy on masked positions | Cross-entropy (two approaches) |
| Loss scope | Only corrupted positions | **Approach A**: Only corrupted positions<br>**Approach B**: ALL positions |
| Vocabulary size | `vocab_size + 1` | `vocab_size + 2` |
| Corruption token | `mask_token_id` (vocab_size) | `wrong_token_id` (vocab_size + 1) |
| Objective | Predict masked content | Identify and fix wrong content |

## Detailed Implementation Plan

### Phase 1: Core Infrastructure

#### 1.1 Vocabulary Extension (model.py)
```python
# Current: vocab_size + 1 (mask_token_id)
# New: vocab_size + 2 (mask_token_id, wrong_token_id)
mask_token_id = meta_vocab_size      # Current
wrong_token_id = meta_vocab_size + 1 # New
extended_vocab_size = meta_vocab_size + 2
```

#### 1.2 Training Type Configuration (train.py)
```python
# Add new configuration parameter
training_type = 'unmasking'  # 'unmasking' or 'remasking'
remasking_loss_type = 'corrupted_only'  # 'corrupted_only' or 'all_positions'

# Different checkpoint naming
if training_type == 'remasking':
    ckpt_filename = f'ckpt_remasking_{remasking_loss_type}_{iter_num}.pt'
    wandb_run_name = f'{wandb_run_name}_remasking_{remasking_loss_type}'
else:
    ckpt_filename = f'ckpt_unmasking_{iter_num}.pt'
```

#### 1.3 Data Generation (train.py)
```python
def get_batch_remasking(split):
    # Same position selection logic as get_batch()
    # Replace selected positions with wrong_token_id instead of mask_token_id
    # Return (corrupted_x, original_y, corruption_mask)
    pass

def get_batch(split):
    if training_type == 'remasking':
        return get_batch_remasking(split)
    else:
        return get_batch_unmasking(split)  # Current implementation
```

### Phase 2: Loss Function Implementation

#### 2.1 Loss Function (Two Approaches to Test)

##### Approach A: Corrupted Positions Only (Like Unmasking)
```python
# Only compute loss where corruption occurred
masked_logits = logits_flat[corruption_mask]
masked_targets = targets_flat[corruption_mask]
loss = F.cross_entropy(masked_logits, masked_targets)
```

##### Approach B: All Positions (Full Supervision)
```python
# Compute loss on all positions
logits_flat = logits.view(-1, logits.size(-1))
targets_flat = targets.view(-1)
loss = F.cross_entropy(logits_flat, targets_flat)
```

**Rationale for Approach B**: It's an error to:
- NOT fix a wrong token (false negative)
- Fix a correct token (false positive)

Both approaches will be implemented and compared.

#### 2.2 Unified Loss Computation
```python
def compute_loss(logits, targets, mask, training_type, loss_type):
    if training_type == 'unmasking':
        # Current implementation: only masked positions
        return compute_masked_loss(logits, targets, mask)
    elif training_type == 'remasking':
        if loss_type == 'corrupted_only':
            return compute_masked_loss(logits, targets, mask)
        elif loss_type == 'all_positions':
            return compute_full_loss(logits, targets)
```

#### 2.3 Experimental Comparison
- Train two remasking models with different loss functions
- Compare performance in sampling quality
- Determine optimal approach for future development

### Phase 3: Integration

#### 3.1 Model Architecture Considerations
- **Both models**: Same transformer architecture with token prediction heads
- **Both models**: Output `(vocab_size + 2)`-sized logits for identical architecture
- **Unmasking model**: Uses `mask_token_id`, `wrong_token_id` unused but present
- **Remasking model**: Uses `wrong_token_id`, `mask_token_id` unused but present
- **Future merging**: Identical architecture enables model combination later

#### 3.2 Model Loading (sample.py)
```python
# Load appropriate model based on checkpoint type
if 'remasking' in checkpoint_name:
    # Remasking model: predict tokens, identify wrong_token_id positions
    remasking_model = load_remasking_model()
else:
    # Current unmasking model
    unmasking_model = load_unmasking_model()
```

#### 3.3 Intelligent Remasking Logic
```python
def intelligent_remask(tokens, remasking_model, num_to_remask):
    # Get model predictions
    logits, _ = remasking_model(tokens, None)
    
    # Find positions where model predicts wrong_token_id
    predicted_tokens = torch.argmax(logits, dim=-1)
    wrong_positions = (predicted_tokens == wrong_token_id)
    
    # Remask based on model confidence about "wrongness"
    # Implementation details TBD based on experimental results
    pass
```

## Training Strategy

### Phase 1: Foundation (This Implementation)
- Extend vocabulary: add `wrong_token_id`
- Implement `get_batch_remasking()` with `wrong_token_id` corruption
- Test both loss approaches: corrupted-only vs all-positions
- Same transformer architecture as unmasking model

### Phase 2: Comparison & Optimization
- Train models with both loss functions
- Compare sampling quality improvements
- Benchmark against random remasking baseline
- Select optimal approach

### Phase 3: Integration & Advanced Features
- Integrate best remasking model into sampling
- Experiment with confidence-based remasking thresholds
- Develop model merging strategies

### Phase 4: Advanced Corruption Strategies (Future)
- Multiple wrong token types
- Context-aware corruption patterns
- Hierarchical remasking decisions

## Expected Benefits

1. **Contextual Awareness**: Model learns which tokens fit poorly in context
2. **Coherence Preservation**: Avoid remasking tokens that are already correct
3. **Adaptive Strategy**: Different remasking patterns for different content types
4. **Quality Improvement**: Focus demasking effort on truly problematic tokens

## Files to Modify

1. **train.py**: 
   - Add `training_type` and `remasking_loss_type` parameters
   - Implement `get_batch_remasking()` function
   - Update checkpoint naming and wandb logging
   - Add loss computation variants

2. **model.py**: 
   - Update vocabulary size calculation
   - Ensure compatibility for future model merging

3. **sample.py**: 
   - Add remasking model loading
   - Replace random remasking with intelligent remasking
   - Handle different vocabulary sizes

4. **configurator.py**: 
   - Add new configuration parameters

## Success Metrics

### Training Metrics
1. **Loss Convergence**: Both loss approaches achieve stable convergence
2. **Accuracy**: High accuracy on predicting original tokens at corrupted positions
3. **Comparison**: Determine which loss function (corrupted-only vs all-positions) performs better

### Sampling Quality Metrics
1. **Coherence**: Generated text maintains better contextual coherence
2. **Efficiency**: Fewer diffusion iterations needed for high-quality output
3. **Consistency**: More deterministic and predictable generation process

### Experimental Comparisons
1. **Random vs Intelligent Remasking**: Direct quality comparison
2. **Loss Function Variants**: Performance comparison between approaches A and B
3. **Architecture Compatibility**: Verify future model merging feasibility

## Development Milestones

1. **Milestone 1**: Vocabulary extension and basic remasking data generation
2. **Milestone 2**: Both loss function implementations working
3. **Milestone 3**: Successful remasking model training
4. **Milestone 4**: Integration with sample.py and quality evaluation
5. **Milestone 5**: Performance comparison and approach selection

This approach transforms the remasking step from random guessing to learned intelligence, making the diffusion process more efficient and effective while maintaining architectural compatibility for future model merging.