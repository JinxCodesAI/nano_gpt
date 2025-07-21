# Rotary Embeddings Implementation Plan for nanoGPT

## Overview
This document outlines the step-by-step plan to introduce Rotary Position Embeddings (RoPE) as a configurable option in nanoGPT. RoPE is an alternative to the standard learned position embeddings that provides better extrapolation capabilities and relative position understanding.

## Current Architecture Analysis

### Key Components
- **model.py**: Contains the main GPT model implementation
  - `GPTConfig`: Dataclass for model configuration
  - `CausalSelfAttention`: Standard multi-head attention with learned position embeddings
  - `GPT`: Main model class with token and position embeddings
- **train.py**: Training script with configuration system
- **configurator.py**: Configuration override system

### Current Position Embedding System
- Uses `nn.Embedding` for position embeddings (`wpe` in transformer)
- Position embeddings are added to token embeddings before processing
- Fixed maximum sequence length (`block_size`)

## Implementation Plan

### Phase 1: Design and Structure

#### 1.1 Rotary Embedding Module Design
Create a new rotary embedding module that implements RoPE functionality:

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        # Implementation of RoPE
```

#### 1.2 Configuration Extension
Extend `GPTConfig` to include rotary embedding options:

```python
@dataclass
class GPTConfig:
    # ... existing fields ...
    use_rotary_embeddings: bool = False
    rotary_base: float = 10000.0
    rotary_max_position_embeddings: int = 2048
```

### Phase 2: Core Implementation

#### 2.1 Create Rotary Embedding Module
- Implement RoPE rotation matrix computation
- Add frequency computation for different dimensions
- Implement efficient rotation application to query/key vectors

#### 2.2 Modify CausalSelfAttention
- Add rotary embedding support to attention mechanism
- Conditionally apply rotary embeddings based on configuration
- Maintain backward compatibility with existing models

#### 2.3 Update GPT Model
- Modify position embedding handling
- Add conditional logic for rotary vs standard embeddings
- Update model initialization to handle new parameters

### Phase 3: Configuration and Integration

#### 3.1 Training Script Updates
- Add new configuration parameters to train.py
- Update model initialization to pass rotary parameters
- Ensure configuration override system works with new options

#### 3.2 Configuration Files
- Create example configuration files using rotary embeddings
- Update existing configs to show rotary embedding usage
- Add documentation for new configuration options

### Phase 4: Testing and Validation

#### 4.1 Backward Compatibility
- Ensure existing models load correctly without rotary embeddings
- Test checkpoint compatibility
- Verify training continuation works

#### 4.2 Performance Testing
- Compare memory usage between rotary and standard embeddings
- Validate training convergence with rotary embeddings
- Test extrapolation capabilities

## Detailed Implementation Steps

### Step 1: Create Rotary Embedding Module
**File**: `model.py` (add new class)
- Implement `RotaryPositionalEmbedding` class
- Add rotation matrix computation
- Implement efficient rotation application

### Step 2: Update GPTConfig
**File**: `model.py`
- Add `use_rotary_embeddings: bool = False`
- Add `rotary_base: float = 10000.0`
- Add `rotary_max_position_embeddings: int = 2048`

### Step 3: Modify CausalSelfAttention
**File**: `model.py`
- Add rotary embedding parameter to __init__
- Update forward method to apply rotary embeddings
- Handle both rotary and standard attention paths

### Step 4: Update GPT Model
**File**: `model.py`
- Modify __init__ to handle rotary embeddings
- Update position embedding creation logic
- Ensure proper initialization

### Step 5: Update Training Script
**File**: `train.py`
- Add new configuration parameters
- Update model initialization
- Add documentation for new options

### Step 6: Create Configuration Examples
**Files**: `config/`
- Create `train_gpt2_rotary.py`
- Create `finetune_shakespeare_rotary.py`
- Add inline documentation

### Step 7: Testing and Validation
- Create test script for rotary embeddings
- Add unit tests for rotation matrices
- Validate model loading/saving

## Code Structure Changes

### Files to Modify
1. **model.py**
   - Add RotaryPositionalEmbedding class
   - Update GPTConfig dataclass
   - Modify CausalSelfAttention class
   - Update GPT class initialization

2. **train.py**
   - Add new configuration parameters
   - Update model initialization calls

3. **config/** directory
   - Add new configuration examples
   - Update documentation

### New Files to Create
1. **config/train_gpt2_rotary.py** - Example config for rotary embeddings
2. **test_rotary.py** - Test script for rotary embeddings
3. **ROTARY_EMBEDDINGS_PLAN.md** - This documentation

## Configuration Usage

### Basic Usage
```python
# In config file or command line
use_rotary_embeddings = True
rotary_base = 10000.0
rotary_max_position_embeddings = 2048
```

### Command Line Override
```bash
python train.py --use_rotary_embeddings=True --rotary_base=50000.0
```

## Technical Details

### Rotary Embedding Implementation
RoPE applies rotation matrices to query and key vectors based on their positions:
- For position `m`, apply rotation by angle `m * θ_i` to each dimension pair
- `θ_i = base^(-2i/d)` where `d` is the embedding dimension
- Efficient implementation using complex numbers and rotation matrices

### Memory Considerations
- Rotary embeddings don't require additional parameters for position encoding
- May slightly increase computation due to rotation operations
- Better extrapolation means potentially smaller models for same performance

### Compatibility Notes
- Models trained with standard embeddings will not work with rotary embeddings
- Checkpoints contain different parameter structures
- Training must be consistent within a single run

## Testing Checklist

- [ ] Model initializes correctly with rotary embeddings enabled
- [ ] Model trains without errors
- [ ] Loss decreases normally during training
- [ ] Model generates coherent text
- [ ] Backward compatibility maintained for non-rotary models
- [ ] Configuration overrides work correctly
- [ ] Memory usage is reasonable
- [ ] Performance is comparable to standard embeddings

## Future Enhancements

- Support for different rotary embedding variants (e.g., YaRN, NTK)
- Dynamic scaling for longer sequences
- Integration with flash attention optimizations
- Performance benchmarking tools

## References

- [RoPE Paper](https://arxiv.org/abs/2104.09864) - Rotary Position Embedding original paper
- [Hugging Face Implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py) - Reference implementation
- [LLaMA Implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) - Production usage example