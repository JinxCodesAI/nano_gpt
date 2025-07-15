# Mixture of Experts (MoE) Implementation Plan for nanoGPT

## Overview
This document outlines the step-by-step plan to introduce Mixture of Experts (MoE) architecture as a configurable option in nanoGPT. MoE is a sparse activation technique that allows scaling model capacity while maintaining computational efficiency by selectively activating only a subset of expert networks for each input token.

## Current Architecture Analysis

### Key Components
- **model.py**: Contains the main GPT model implementation
  - `GPTConfig`: Dataclass for model configuration
  - `CausalSelfAttention`: Multi-head attention mechanism
  - `MLP`: Standard feed-forward network (SwiGLU could be added)
  - `Block`: Transformer block with attention + MLP
  - `GPT`: Main model class
- **train.py**: Training script with configuration system
- **config/**: Configuration files for different training scenarios

### Current Feed-Forward System
- Uses a single MLP layer (`c_fc -> GELU -> c_proj`) for all tokens
- Fixed intermediate size (4x hidden size)
- All parameters are active for every token
- No expert selection or routing mechanism

## MoE Architecture Overview

### Core Concepts from DeepSeek Implementation
- **Expert Networks**: Multiple parallel MLP experts (e.g., 8-64 experts)
- **Router/Gate**: Learned gating mechanism to route tokens to experts
- **Top-K Routing**: Each token is routed to top-K experts (typically 1-2)
- **Load Balancing**: Techniques to ensure expert utilization balance
- **Shared Experts**: Optional shared experts for all tokens
- **Expert Parallelism**: Support for distributed expert training

### Key Components from DeepSeek
- `MoEGate`: Gating mechanism for expert selection
- `DeepseekV3MoE`: Main MoE module with expert management
- `DeepseekV3MLP`: Individual expert network
- Expert placement logic in transformer blocks

## Implementation Plan

### Phase 1: Design and Structure

#### 1.1 MoE Module Design
Create modular MoE components inspired by DeepSeek:

```python
class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Gating mechanism for expert selection
        self.top_k = config.moe_top_k
        self.num_experts = config.moe_num_experts
        self.gate = nn.Linear(config.n_embd, self.num_experts, bias=False)

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Collection of expert networks
        self.experts = nn.ModuleList([
            MLP(config, expert=True) for _ in range(config.moe_num_experts)
        ])
        self.gate = MoEGate(config)
```

#### 1.2 Configuration Extension
Extend `GPTConfig` to include MoE parameters:

```python
@dataclass
class GPTConfig:
    # ... existing fields ...
    
    # MoE parameters
    use_moe: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_expert_capacity_factor: float = 1.0
    moe_aux_loss_weight: float = 0.01
    moe_first_k_dense_replace: int = 0  # First K layers use dense MLP
    moe_layer_freq: int = 1  # Every Nth layer uses MoE
    moe_shared_experts: int = 0  # Number of shared experts
```

### Phase 2: Core Implementation

#### 2.1 Create MoE Components
- Implement `MoEGate` with top-k routing
- Create `MoELayer` with expert management
- Add load balancing loss computation
- Implement expert capacity management

#### 2.2 Modify MLP for Expert Mode
- Add expert-specific configuration
- Support different intermediate sizes for experts
- Add optional SwiGLU activation (from DeepSeek)

#### 2.3 Update Block Architecture
- Conditional MoE vs dense MLP selection
- Expert placement logic based on layer index
- Integration with existing transformer blocks

### Phase 3: Advanced Features

#### 3.1 Load Balancing
- Implement auxiliary loss for expert balancing
- Add routing probability regularization
- Support different gating strategies (sigmoid, softmax)

#### 3.2 Expert Parallelism Support
- Add distributed training support
- Implement expert sharding across GPUs
- Add communication primitives for expert routing

#### 3.3 Memory Optimization
- Implement expert caching
- Add gradient checkpointing for experts
- Support expert offloading

### Phase 4: Configuration and Integration

#### 4.1 Training Script Updates
- Add MoE-specific configuration parameters
- Update optimizer to handle expert parameters
- Add MoE-specific logging and metrics

#### 4.2 Configuration Files
- Create example MoE configurations
- Add documentation for MoE hyperparameters
- Provide tuning guidelines

## Detailed Implementation Steps

### Step 1: Create MoE Gate Module
**File**: `model.py` (add new class)
- Implement `MoEGate` class with top-k routing
- Add gating score computation
- Implement expert selection logic
- Add load balancing loss computation

### Step 2: Create Expert MLP
**File**: `model.py` (modify existing MLP)
- Add expert mode to `MLP` class
- Support configurable intermediate sizes
- Add SwiGLU activation option
- Ensure expert-specific initialization

### Step 3: Create MoE Layer
**File**: `model.py` (add new class)
- Implement `MoELayer` with expert collection
- Add expert routing and combination
- Implement efficient expert computation
- Add shared expert support

### Step 4: Update GPTConfig
**File**: `model.py`
- Add all MoE configuration parameters
- Set reasonable defaults for experimentation
- Add validation for MoE parameters

### Step 5: Modify Block Architecture
**File**: `model.py`
- Update `Block` class to support MoE
- Add conditional logic for MoE vs dense MLP
- Implement expert placement strategy

### Step 6: Update GPT Model
**File**: `model.py`
- Modify initialization for MoE support
- Update parameter counting for MoE models
- Add MoE-specific metrics and logging

### Step 7: Update Training Script
**File**: `train.py`
- Add MoE configuration parameters
- Update optimizer configuration for experts
- Add MoE-specific loss computation
- Implement MoE monitoring

### Step 8: Create Configuration Examples
**Files**: `config/`
- Create `train_shakespeare_char_moe.py`
- Create `train_gpt2_moe.py`
- Add MoE tuning guidelines

## Code Structure Changes

### Files to Modify
1. **model.py**
   - Add MoEGate class
   - Add MoELayer class
   - Modify MLP class for expert mode
   - Update Block class
   - Update GPTConfig dataclass
   - Update GPT class initialization

2. **train.py**
   - Add MoE configuration parameters
   - Update loss computation for auxiliary losses
   - Add MoE-specific logging
   - Update optimizer configuration

3. **config/** directory
   - Add new MoE configuration examples
   - Update documentation

### New Files to Create
1. **config/train_shakespeare_char_moe.py** - Basic MoE config
2. **config/train_gpt2_moe.py** - GPT-2 scale MoE config
3. **moe_utils.py** - MoE-specific utilities
4. **test_moe.py** - MoE testing script

## Technical Implementation Details

### MoE Gate Implementation
```python
class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.gate = nn.Linear(config.n_embd, self.num_experts, bias=False)
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = x.shape
        x = x.view(-1, hidden_size)  # (batch*seq_len, hidden_size)
        
        # Compute gating scores
        gate_scores = F.softmax(self.gate(x), dim=-1)  # (batch*seq_len, num_experts)
        
        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        
        # Normalize scores
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
        
        # Compute auxiliary loss for load balancing
        aux_loss = self._compute_aux_loss(gate_scores)
        
        return top_k_indices, top_k_scores, aux_loss
```

### Expert Placement Strategy
```python
def should_use_moe(layer_idx, config):
    """Determine if a layer should use MoE based on configuration."""
    if not config.use_moe:
        return False
    
    # Skip first K layers
    if layer_idx < config.moe_first_k_dense_replace:
        return False
    
    # Use MoE every N layers
    if (layer_idx - config.moe_first_k_dense_replace) % config.moe_layer_freq == 0:
        return True
    
    return False
```

### Load Balancing Loss
```python
def _compute_aux_loss(self, gate_scores):
    """Compute auxiliary loss for expert load balancing."""
    # Compute fraction of tokens routed to each expert
    expert_fractions = gate_scores.mean(dim=0)
    
    # Compute uniform target
    uniform_target = torch.ones_like(expert_fractions) / self.num_experts
    
    # L2 loss between actual and uniform distributions
    aux_loss = torch.sum((expert_fractions - uniform_target) ** 2)
    
    return aux_loss
```

## Configuration Usage

### Basic MoE Configuration
```python
# In config file
use_moe = True
moe_num_experts = 8
moe_top_k = 2
moe_first_k_dense_replace = 2  # First 2 layers use dense MLP
moe_layer_freq = 2  # Every 2nd layer uses MoE
moe_aux_loss_weight = 0.01
```

### Advanced MoE Configuration
```python
# Large-scale MoE
use_moe = True
moe_num_experts = 64
moe_top_k = 2
moe_expert_capacity_factor = 1.25
moe_shared_experts = 2  # 2 shared experts for all tokens
moe_aux_loss_weight = 0.005
```

### Command Line Override
```bash
python train.py --use_moe=True --moe_num_experts=16 --moe_top_k=2
```

## Memory and Performance Considerations

### Memory Usage
- **Expert Parameters**: Total parameters scale with `num_experts`, but only `top_k` experts are active per token
- **Activation Memory**: Reduced compared to dense model with same parameter count
- **Routing Overhead**: Small additional memory for gating and routing

### Training Considerations
- **Gradient Accumulation**: May need adjustment for expert balancing
- **Learning Rate**: May need reduction for stable expert training
- **Batch Size**: Larger batches help with expert utilization
- **Auxiliary Loss**: Important for expert balancing, but don't overweight

### Inference Optimizations
- **Expert Caching**: Cache expert computations for repeated tokens
- **Dynamic Routing**: Adjust routing based on input characteristics
- **Expert Pruning**: Remove unused experts for deployment

## Testing Checklist

- [ ] Model initializes correctly with MoE enabled
- [ ] Expert routing works correctly (tokens routed to appropriate experts)
- [ ] Load balancing loss decreases during training
- [ ] Model trains without numerical instabilities
- [ ] Memory usage scales appropriately with expert count
- [ ] Backward compatibility maintained for non-MoE models
- [ ] Configuration overrides work correctly
- [ ] Expert placement strategy works as expected
- [ ] Shared experts function correctly
- [ ] Distributed training support works (future enhancement)

## Future Enhancements

### Advanced Routing Strategies
- **Learnable Routing**: Trainable routing functions
- **Expert Choice**: Allow experts to choose tokens
- **Hierarchical Routing**: Multi-level expert selection
- **Dynamic Expert Creation**: Adaptive expert addition/removal

### Optimization Features
- **Expert Parallelism**: Distributed expert training
- **Expert Offloading**: CPU/GPU expert management
- **Sparse Expert Updates**: Only update active experts
- **Expert Quantization**: Reduce expert memory usage

### Advanced Architectures
- **Hierarchical MoE**: Nested expert structures
- **Task-Specific Experts**: Experts for different tasks
- **Temporal Experts**: Experts for different time steps
- **Multi-Modal Experts**: Experts for different modalities

## References

- [DeepSeek-MoE Paper](https://arxiv.org/abs/2401.06066) - DeepSeek's MoE implementation
- [GShard Paper](https://arxiv.org/abs/2006.16668) - Early MoE transformer work
- [Switch Transformers](https://arxiv.org/abs/2101.03961) - Scaling MoE models
- [BASE Layer](https://arxiv.org/abs/2002.08933) - Balanced assignment of experts
- [Tutel MoE](https://github.com/microsoft/tutel) - Efficient MoE implementation