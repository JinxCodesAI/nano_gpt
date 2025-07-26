# Layer-Specific Operations Guide

This guide explains how to use the flexible layer-specific operations for freezing/unfreezing layers and setting LoRA ranks on individual layers in nanoGPT.

## Overview

The GPT model now supports fine-grained control over individual layers, allowing you to:
- Freeze/unfreeze specific components (attention, MLP, layer norms, embeddings)
- Set LoRA ranks for specific layers independently
- Configure these operations to trigger automatically during training based on loss thresholds

## Layer Naming Convention

### Transformer Block Components (0-based indexing)

Each transformer block contains multiple components that can be controlled individually:

| Layer Name | Description | Example |
|------------|-------------|---------|
| `attn.N` | Attention mechanism in layer N | `attn.2` = attention in layer 2 |
| `mlp.N` | MLP/feed-forward network in layer N | `mlp.0` = MLP in layer 0 |
| `ln_1.N` | First LayerNorm in layer N (pre-attention) | `ln_1.5` = first LayerNorm in layer 5 |
| `ln_2.N` | Second LayerNorm in layer N (pre-MLP) | `ln_2.3` = second LayerNorm in layer 3 |

### Global Components

| Layer Name | Description |
|------------|-------------|
| `wte` | Token embeddings (word token embeddings) |
| `wpe` | Position embeddings (only if not using rotary embeddings) |
| `ln_f` | Final layer normalization |
| `lm_head` | Language model head (output projection) |

## Available Operations

### 1. Freeze Layer (`freeze_layer`)

Freezes all parameters in a specific layer component, preventing gradient updates.

**Python API:**
```python
# Freeze attention in layer 2
model.freeze_layer("attn.2")

# Freeze token embeddings
model.freeze_layer("wte")

# Freeze MLP in layer 0
model.freeze_layer("mlp.0")
```

**Training Config:**
```python
{
    "name": "freeze_layer",
    "value": "attn.2",  # Layer name as string
    "trigger_loss": 2.5,
    "max_wait_iters": 1000
}
```

### 2. Unfreeze Layer (`unfreeze_layer`)

Unfreezes all parameters in a specific layer component, enabling gradient updates.

**Python API:**
```python
# Unfreeze attention in layer 2
model.unfreeze_layer("attn.2")

# Unfreeze token embeddings
model.unfreeze_layer("wte")
```

**Training Config:**
```python
{
    "name": "unfreeze_layer",
    "value": "mlp.1",  # Layer name as string
    "trigger_loss": 2.0,
    "max_wait_iters": 1500
}
```

### 3. Set Layer LoRA Rank (`set_layer_lora_rank`)

Sets the LoRA rank for a specific layer. Currently supports attention layers (`attn.N`) and token embeddings (`wte`).

**Python API:**
```python
# Set attention layer 3 to LoRA rank 16
model.set_layer_lora_rank("attn.3", 16)

# Set token embeddings to LoRA rank 8
model.set_layer_lora_rank("wte", 8)

# Disable LoRA for attention layer 1 (rank 0)
model.set_layer_lora_rank("attn.1", 0)
```

**Training Config:**
```python
{
    "name": "set_layer_lora_rank",
    "value": ["attn.3", 16],  # [layer_name, rank] as list
    "trigger_loss": 2.8,
    "max_wait_iters": 1000
}
```

## Complete Training Configuration Example

Here's a comprehensive example showing how to use these operations in your training configuration:

```python
# In your training config file
operations = [
    # Phase 1: Start with frozen embeddings, only train transformer layers
    {
        "name": "freeze_layer",
        "value": "wte",
        "trigger_loss": float('inf'),  # Execute immediately
        "max_wait_iters": 0
    },
    {
        "name": "freeze_layer", 
        "value": "lm_head",
        "trigger_loss": float('inf'),
        "max_wait_iters": 0
    },
    
    # Phase 2: When loss drops, add LoRA to specific attention layers
    {
        "name": "set_layer_lora_rank",
        "value": ["attn.0", 8],  # Early layer gets lower rank
        "trigger_loss": 3.0,
        "max_wait_iters": 1000
    },
    {
        "name": "set_layer_lora_rank", 
        "value": ["attn.5", 16],  # Middle layer gets higher rank
        "trigger_loss": 2.8,
        "max_wait_iters": 1000
    },
    {
        "name": "set_layer_lora_rank",
        "value": ["attn.11", 8], # Final layer gets lower rank
        "trigger_loss": 2.6,
        "max_wait_iters": 1000
    },
    
    # Phase 3: Gradually unfreeze specific components
    {
        "name": "unfreeze_layer",
        "value": "ln_f",  # Unfreeze final layer norm first
        "trigger_loss": 2.4,
        "max_wait_iters": 1500
    },
    {
        "name": "set_layer_lora_rank",
        "value": ["wte", 12],  # Add LoRA to embeddings
        "trigger_loss": 2.2,
        "max_wait_iters": 1500
    },
    
    # Phase 4: Fine-tune specific problematic layers
    {
        "name": "unfreeze_layer",
        "value": "attn.2",  # Maybe layer 2 needs more training
        "trigger_loss": 2.0,
        "max_wait_iters": 2000
    },
    {
        "name": "freeze_layer",
        "value": "mlp.0",   # Freeze MLP that might be overfitting
        "trigger_loss": 1.8,
        "max_wait_iters": 2000
    }
]
```

## Advanced Usage Patterns

### Progressive LoRA Rank Scaling
```python
# Start with low ranks, increase as training progresses
operations = [
    {"name": "set_layer_lora_rank", "value": ["attn.6", 4], "trigger_loss": 3.0, "max_wait_iters": 1000},
    {"name": "set_layer_lora_rank", "value": ["attn.6", 8], "trigger_loss": 2.5, "max_wait_iters": 1500},
    {"name": "set_layer_lora_rank", "value": ["attn.6", 16], "trigger_loss": 2.0, "max_wait_iters": 2000},
]
```

### Layer-wise Unfreezing Strategy
```python
# Unfreeze layers from top to bottom as loss improves
operations = [
    {"name": "unfreeze_layer", "value": "attn.11", "trigger_loss": 2.8, "max_wait_iters": 1000},
    {"name": "unfreeze_layer", "value": "attn.10", "trigger_loss": 2.6, "max_wait_iters": 1000},
    {"name": "unfreeze_layer", "value": "attn.9", "trigger_loss": 2.4, "max_wait_iters": 1000},
    # ... continue for other layers
]
```

### Component-Specific Training
```python
# Train only attention layers first, then MLPs
operations = [
    # Freeze all MLPs initially
    *[{"name": "freeze_layer", "value": f"mlp.{i}", "trigger_loss": float('inf'), "max_wait_iters": 0} 
      for i in range(12)],  # Assuming 12 layers
    
    # Unfreeze MLPs when attention is well-trained
    *[{"name": "unfreeze_layer", "value": f"mlp.{i}", "trigger_loss": 2.0, "max_wait_iters": 1500}
      for i in range(12)]
]
```

## Legacy Function Support

The following legacy functions are maintained for backward compatibility but show deprecation warnings:

- `resize_lora_rank(rank)` → Use `set_layer_lora_rank("attn.X", rank)` for each layer
- `resize_embedding_rank(rank)` → Use `set_layer_lora_rank("wte", rank)`  
- `set_embedding_finetune_mode(enabled)` → Use `freeze_layer`/`unfreeze_layer` for specific control
- `set_embedding_freeze_mode(enabled)` → Use `freeze_layer`/`unfreeze_layer` for specific control

## Best Practices

### 1. Start Conservative
Begin with most layers frozen and gradually unfreeze as training progresses:
```python
# Freeze everything except a few key layers initially
initial_frozen = ["wte", "lm_head"] + [f"attn.{i}" for i in range(8, 12)]
```

### 2. Use Different LoRA Ranks by Depth
Early layers often need lower ranks, middle layers higher ranks:
```python
# Lower ranks for early layers (more general features)
{"name": "set_layer_lora_rank", "value": ["attn.0", 4], ...}
{"name": "set_layer_lora_rank", "value": ["attn.1", 4], ...}

# Higher ranks for middle layers (task-specific features)  
{"name": "set_layer_lora_rank", "value": ["attn.6", 16], ...}
{"name": "set_layer_lora_rank", "value": ["attn.7", 16], ...}

# Medium ranks for final layers (output formatting)
{"name": "set_layer_lora_rank", "value": ["attn.11", 8], ...}
```

### 3. Monitor Training Closely
Use different trigger thresholds to create training phases:
- Phase 1 (3.0+ loss): Basic structure learning
- Phase 2 (2.5-3.0 loss): Add LoRA adapters  
- Phase 3 (2.0-2.5 loss): Fine-tune specific components
- Phase 4 (<2.0 loss): Final optimization

### 4. Experiment with Component Priorities
Some components may be more important for your specific task:
```python
# For code generation, attention might be more critical
# For text generation, MLPs might need more capacity
# For domain adaptation, embeddings might need the most change
```

## Troubleshooting

### Common Issues

1. **"Layer index out of range"**: Check your model's `n_layer` configuration
2. **"Unknown component"**: Verify layer name spelling (case-sensitive)
3. **LoRA not supported**: Only `attn.X` and `wte` support LoRA currently
4. **Position embeddings not found**: `wpe` only exists when not using rotary embeddings

### Debugging Tips

```python
# Check current layer configuration
print(f"Model has {model.config.n_layer} layers")
print(f"Using rotary embeddings: {model.config.use_rotary_embeddings}")

# Check parameter counts
param_counts = model.get_detailed_param_count()
print("Trainable parameters by component:")
for component, counts in param_counts.items():
    print(f"  {component}: {counts['trainable']:,} / {counts['total']:,}")
```

This flexible system allows you to create sophisticated training strategies tailored to your specific use case and dataset characteristics.