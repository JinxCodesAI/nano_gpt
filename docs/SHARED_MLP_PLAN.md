# Shared MLP Implementation Plan for nanoGPT

## Overview
This document outlines the step-by-step plan to introduce shared MLP (Multi-Layer Perceptron) functionality across transformer blocks in nanoGPT. This feature allows multiple blocks to share the same MLP parameters, reducing model size while maintaining architectural flexibility.

## Current Architecture Analysis

### Key Components
- **model.py**: Contains the main GPT model implementation
  - `GPTConfig`: Dataclass for model configuration
  - `MLP`: Feed-forward network implementation
  - `Block`: Transformer block containing attention and MLP
  - `GPT`: Main model class managing all blocks

### Current MLP System
- Each `Block` instance creates its own independent `MLP` module
- MLP parameters are duplicated across all blocks
- No sharing mechanism exists between blocks

## Implementation Plan

### Phase 1: Design and Structure

#### 1.1 MLP Sharing Architecture Design
Create a new MLP manager system that handles sharing across blocks:

```python
class MLPManager:
    def __init__(self, config):
        # Manages shared MLP instances based on sharing configuration
        pass
    
    def get_mlp(self, layer_idx, head_group_idx):
        # Returns appropriate MLP instance for given position
        pass
```

#### 1.2 Configuration Extension
Extend `GPTConfig` to include MLP sharing options:

```python
@dataclass
class GPTConfig:
    # ... existing fields ...
    shared_MLP_heads: int = 1  # Number of attention heads sharing one MLP
    shared_MLP_layers: int = 1  # Number of transformer layers sharing one MLP
```

### Phase 2: Core Implementation

#### 2.1 Create MLP Manager
- Implement MLPManager class to handle MLP sharing logic
- Add validation for divisibility requirements
- Implement efficient MLP instance management

#### 2.2 Modify MLP Class
- Make MLP class compatible with sharing (no changes needed, already stateless)
- Ensure proper parameter initialization

#### 2.3 Update Block Class
- Modify Block initialization to accept MLP from manager
- Add layer and head indexing information to Block
- Update Block to use shared MLP instances

#### 2.4 Update GPT Model
- Instantiate MLPManager in GPT initialization
- Create shared MLP instances based on configuration
- Pass appropriate MLP instances to each Block

### Phase 3: Configuration and Validation

#### 3.1 Validation Logic
- Validate that n_layer is divisible by shared_MLP_layers
- Validate that n_head is divisible by shared_MLP_heads
- Provide clear error messages for invalid configurations

#### 3.2 Backward Compatibility
- Ensure default values maintain existing behavior
- Handle checkpoint loading/saving with shared MLPs
- Maintain parameter count reporting accuracy

## Detailed Implementation Steps

### Step 1: Update GPTConfig
**File**: `model.py`
- Add `shared_MLP_heads: int = 1`
- Add `shared_MLP_layers: int = 1`
- Add validation methods for divisibility

### Step 2: Create MLPManager Class
**File**: `model.py` (add new class)
- Implement MLP instance creation and management
- Handle indexing logic for layer/head grouping
- Provide clean interface for Block instances

### Step 3: Modify Block Class
**File**: `model.py`
- Add layer_idx and head_group_idx parameters to __init__
- Accept MLP instance from MLPManager instead of creating new one
- Update forward pass to use provided MLP

### Step 4: Update GPT Initialization
**File**: `model.py`
- Instantiate MLPManager with configuration
- Create shared MLP instances based on sharing parameters
- Pass appropriate MLP to each Block during initialization

### Step 5: Update Parameter Counting
**File**: `model.py`
- Modify get_detailed_param_count() to handle shared MLPs
- Update parameter counting logic to avoid double-counting shared parameters

### Step 6: Testing and Validation
- Create test configurations for different sharing patterns
- Validate parameter reduction calculations
- Test backward compatibility

## Code Structure Changes

### Files to Modify
1. **model.py**
   - Add MLPManager class
   - Update GPTConfig dataclass
   - Modify Block class for MLP sharing
   - Update GPT class initialization
   - Update parameter counting methods

### New Classes to Create
1. **MLPManager**: Manages shared MLP instances
2. **SharedMLPConfig**: Helper class for indexing shared MLPs

## Technical Details

### MLP Sharing Logic
- **shared_MLP_heads=1**: Each attention head gets its own MLP (current behavior)
- **shared_MLP_heads=2**: Every 2 attention heads share one MLP
- **shared_MLP_layers=1**: Each transformer layer has its own MLP
- **shared_MLP_layers=2**: Every 2 transformer layers share one MLP

### Indexing System
```python
# MLP indexing formula
mlp_layer_group = layer_idx // shared_MLP_layers
mlp_head_group = head_idx // shared_MLP_heads
mlp_key = (mlp_layer_group, mlp_head_group)
```

### Memory Considerations
- Parameter reduction factor: (shared_MLP_layers * shared_MLP_heads)
- Memory savings scale linearly with sharing factors
- Computation remains identical (same forward pass)

## Configuration Usage

### Basic Usage
```python
# In config file or command line
shared_MLP_heads = 2  # Share MLP across 2 attention heads
shared_MLP_layers = 3  # Share MLP across 3 transformer layers
```

### Validation Rules
- n_layer must be divisible by shared_MLP_layers
- n_head must be divisible by shared_MLP_heads
- Both parameters must be positive integers

### Command Line Override
```bash
python train.py --shared_MLP_heads=2 --shared_MLP_layers=3
```

## Testing Checklist

- [ ] Model initializes correctly with shared MLPs
- [ ] Parameter count reduces appropriately with sharing
- [ ] Validation catches invalid configurations
- [ ] Backward compatibility maintained (shared_MLP_heads=1, shared_MLP_layers=1)
- [ ] Training works with shared MLPs
- [ ] Model generates coherent text
- [ ] Checkpoint saving/loading works correctly
- [ ] Parameter counting is accurate with sharing

## Example Configurations

### No Sharing (Current Behavior)
```python
shared_MLP_heads = 1
shared_MLP_layers = 1
```

### Moderate Sharing
```python
shared_MLP_heads = 2
shared_MLP_layers = 2
# Reduces MLP parameters by 4x
```

### Maximum Sharing
```python
shared_MLP_heads = n_head
shared_MLP_layers = n_layer
# Single MLP shared across entire model
```

## Future Enhancements

- Support for different sharing patterns (e.g., alternating sharing)
- Dynamic sharing configuration during training
- Integration with mixture of experts
- Performance benchmarking tools

## References

- [Parameter Sharing in Transformers](https://arxiv.org/abs/1906.05909) - Universal Transformers paper
- [ALBERT](https://arxiv.org/abs/1909.11942) - Parameter sharing across layers