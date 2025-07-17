# Design Document

## Overview

The dual-mode GPT architecture extends the existing GPT model to support two operational modes: generator and reward. This design enables efficient implementation of GRPO training by sharing the computationally expensive transformer layers while using specialized heads for different tasks.

The key innovation is the separation of the model into:
- **Shared Trunk**: Transformer layers that process input sequences (embeddings, blocks, layer norm)
- **Task-Specific Heads**: Output layers specialized for either language modeling or reward prediction

## Architecture

### High-Level Architecture

```
Input Tokens
     ↓
Token Embeddings + Position Embeddings (if not using rotary)
     ↓
Dropout Layer
     ↓
Transformer Blocks (Shared Trunk)
     ↓
Final Layer Norm
     ↓
┌─────────────────┬─────────────────┐
│  Generator Mode │  Reward Mode    │
│                 │                 │
│  Language Model │  Sequence       │
│  Head           │  Pooling        │
│  (Linear)       │  ↓              │
│                 │  MLP + ReLU     │
│                 │  ↓              │
│                 │  Softmax        │
└─────────────────┴─────────────────┘
```

### Component Design

#### 1. GPTConfig Extension

The configuration class is extended with a `mode` parameter:

```python
@dataclass
class GPTConfig:
    # ... existing parameters ...
    mode: str = 'generator'  # 'generator' or 'reward'
```

#### 2. Shared Transformer Trunk

The transformer trunk remains unchanged and includes:
- Token embeddings (`wte`)
- Position embeddings (`wpe`) - only when not using rotary embeddings
- Dropout layer (`drop`)
- Transformer blocks (`h`)
- Final layer normalization (`ln_f`)

#### 3. Generator Head

For `mode='generator'`:
- Uses existing `lm_head` (Linear layer: n_embd → vocab_size)
- Maintains weight tying with token embeddings
- Outputs logits for next-token prediction
- Computes cross-entropy loss when targets provided

#### 4. Reward Head

For `mode='reward'`:
- **Pooling Layer**: Extracts the last token's hidden state as sequence representation
- **MLP**: Two-layer network (n_embd → reward_head_hidden_dim → 2) with ReLU activation, reward_head_hidden_dim default is 256
- **Softmax**: Converts raw scores to probability distribution [P(natural), P(synthetic)]
- Computes MSE loss when targets provided

### Data Models

#### Input/Output Specifications

**Generator Mode:**
- Input: Token indices (batch_size, sequence_length)
- Output: Logits (batch_size, sequence_length, vocab_size) or (batch_size, 1, vocab_size) for inference
- Loss: Cross-entropy loss against target tokens

**Reward Mode:**
- Input: Token indices (batch_size, sequence_length)
- Output: Probabilities (batch_size, 2) representing [P(natural), P(synthetic)]
- Loss: MSE loss against target probability distribution

#### Configuration Parameters

The reward head architecture uses fixed dimensions:
- Hidden layer size: 256 neurons
- Output size: 2 (binary classification)
- Activation: ReLU for hidden layer, Softmax for output

## Components and Interfaces

### Modified GPT Class

#### Constructor Changes
```python
def __init__(self, config):
    # Validate mode parameter
    assert config.mode in ['generator', 'reward']
    
    # Create shared transformer trunk
    self.transformer = nn.ModuleDict(...)
    
    # Create mode-specific heads
    if config.mode == 'generator':
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight
    elif config.mode == 'reward':
        self.reward_head = nn.Sequential(
            nn.Linear(config.n_embd, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=-1)
        )
```

#### Forward Pass Logic
```python
def forward(self, idx, targets=None):
    # Shared trunk processing
    x = self.process_shared_trunk(idx)
    
    # Mode-specific head processing
    if self.config.mode == 'generator':
        return self.generator_forward(x, targets)
    elif self.config.mode == 'reward':
        return self.reward_forward(x, targets)
```

### Interface Compatibility

The dual-mode design maintains backward compatibility:
- Existing generator functionality unchanged
- Same method signatures for `forward()`, `generate()`, etc.
- Configuration defaults to generator mode
- No breaking changes to existing training scripts

## Error Handling

### Configuration Validation
- Assert valid mode values during initialization
- Clear error messages for invalid configurations
- Graceful handling of missing parameters

### Runtime Error Handling
- Validate input tensor shapes for both modes
- Handle edge cases in reward head pooling
- Proper error propagation from loss calculations

### Weight Loading Compatibility
- Support loading generator checkpoints into reward mode (trunk weights only)
- Handle missing reward head weights gracefully
- Maintain compatibility with existing checkpoint format

## Testing Strategy

### Unit Tests
1. **Configuration Tests**
   - Valid mode parameter acceptance
   - Invalid mode parameter rejection
   - Default mode behavior

2. **Architecture Tests**
   - Correct head creation for each mode
   - Shared trunk weight sharing verification
   - Weight initialization correctness

3. **Forward Pass Tests**
   - Generator mode output shapes and types
   - Reward mode output shapes and types
   - Loss calculation accuracy for both modes

4. **Compatibility Tests**
   - Backward compatibility with existing models
   - Checkpoint loading/saving functionality
   - Parameter counting accuracy

### Integration Tests
1. **Training Loop Integration**
   - Generator mode training compatibility
   - Reward mode training functionality
   - Mode switching during training cycles

2. **Memory and Performance**
   - Memory usage comparison between modes
   - Forward pass timing benchmarks
   - GPU utilization efficiency

### Validation Tests
1. **Mathematical Correctness**
   - Reward head probability distribution validation
   - Loss function correctness
   - Gradient flow verification

2. **GRPO Integration**
   - Compatibility with reward model training pipeline
   - Proper weight freezing for trunk parameters
   - Adversarial training loop functionality