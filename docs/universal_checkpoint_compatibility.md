# Universal Checkpoint Compatibility

## Overview

The Universal Checkpoint Compatibility feature enables seamless loading of model checkpoints between different adapter configurations, specifically between standard (non-LoRA) and LoRA-enabled models. This eliminates the need for manual checkpoint conversion and provides maximum flexibility in training workflows.

## Key Benefits

### 🔄 **Seamless Architecture Switching**
- Load any checkpoint into any model architecture
- Switch between standard and LoRA configurations without conversion
- Resume training with different adapter settings

### 🚀 **Training Flexibility**
- Start training with a standard model, resume with LoRA fine-tuning
- Begin with LoRA training, deploy as a standard model
- Experiment with different LoRA ranks without losing progress

### 📦 **Deployment Ready**
- All checkpoints are saved in deployable standard format
- No post-processing required for model deployment
- Universal compatibility across different inference setups

### 🔧 **Backward Compatible**
- Existing checkpoints continue to work without modification
- Gradual migration path for existing training pipelines
- No breaking changes to current workflows

## Technical Implementation

### Universal Checkpoint Format

Every checkpoint is saved using the `get_merged_state_dict()` method, which:

1. **Merges LoRA weights**: Calculates `W_merged = W_main + (B @ A * scale)`
2. **Standardizes keys**: Converts `main_weight.weight` to standard `.weight` format
3. **Creates universal format**: Results in checkpoints that look like standard models

```python
# Example: LoRA weights are automatically merged during save
checkpoint = {
    'model': raw_model.get_merged_state_dict(),  # Universal format
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
    'config': config,
}
```

### Smart Loading Logic

The resume functionality includes intelligent key remapping:

1. **Key Comparison**: Compares checkpoint keys with target model keys
2. **Automatic Remapping**: Maps `.weight` to `.main_weight.weight` when needed
3. **Graceful Handling**: Uses `strict=False` for missing LoRA-specific weights
4. **Optimizer Compatibility**: Handles parameter count changes gracefully

```python
# Example: Automatic key remapping during load
for k, v in state_dict.items():
    lora_key_equivalent = k.replace('.weight', '.main_weight.weight')
    if k in model_state_dict:
        final_state_dict[k] = v  # Direct copy
    elif lora_key_equivalent in model_state_dict:
        final_state_dict[lora_key_equivalent] = v  # Remap to LoRA
```

## Usage Examples

### Example 1: Standard → LoRA Training

```bash
# Step 1: Train a standard model
python train.py config/train_standard.py

# Step 2: Resume with LoRA enabled (automatic compatibility)
python train.py config/train_with_lora.py
```

### Example 2: LoRA → Standard Deployment

```bash
# Step 1: Fine-tune with LoRA
python train.py config/finetune_lora.py

# Step 2: Resume as standard model for deployment
python train.py config/deploy_standard.py
```

### Example 3: LoRA Rank Experimentation

```bash
# Start with rank 16
python train.py config/lora_rank_16.py

# Resume with rank 32 (weights are merged and new adapters initialized)
python train.py config/lora_rank_32.py
```

## Configuration

No special configuration is required. The feature works automatically with existing config files:

```python
# Standard model config
embedding_mode = 'standard'
attn_lora_rank = 0
embedding_rank = 0

# LoRA model config  
embedding_mode = 'lora'
attn_lora_rank = 16
embedding_rank = 16
lora_alpha = 1.0
```

## Testing

The implementation includes comprehensive automated tests:

### Test Coverage
- ✅ Standard → LoRA checkpoint loading
- ✅ LoRA → Standard checkpoint loading  
- ✅ Weight merging accuracy
- ✅ Key remapping functionality
- ✅ Optimizer state handling
- ✅ Error resilience

### Running Tests

```bash
# Run the comprehensive compatibility test suite
python test_checkpoint_compatibility.py
```

The test suite creates small models, trains them, saves checkpoints, and verifies that loading works in both directions.

## Error Handling

### Optimizer State Mismatches
When switching between architectures with different parameter counts:

```
WARNING: Could not load optimizer state from checkpoint: loaded state dict contains a parameter group that doesn't match the size of optimizer's group
   This is expected when switching between LoRA and non-LoRA models.
   Optimizer will start fresh with the configured learning rate.
```

This is expected behavior and not an error. The optimizer will initialize fresh with the configured learning rate.

### Missing Keys
LoRA-specific weights (`lora_A.weight`, `lora_B.weight`) are intentionally missing from universal checkpoints and are initialized from scratch when needed.

## Best Practices

### 1. **Consistent Base Architecture**
Ensure the base model architecture (n_layer, n_head, n_embd) remains consistent when switching between configurations.

### 2. **Learning Rate Adjustment**
When resuming with a fresh optimizer, consider adjusting the learning rate appropriately.

### 3. **Validation Monitoring**
Monitor validation loss when switching architectures to ensure training stability.

### 4. **Checkpoint Naming**
Use descriptive checkpoint names to track which configurations were used:
```
out_standard/ckpt.pt
out_lora_rank16/ckpt.pt
out_lora_rank32/ckpt.pt
```

## Implementation Files

- **`model.py`**: Contains `get_merged_state_dict()` method
- **`train.py`**: Contains smart loading logic in resume block
- **`test_checkpoint_compatibility.py`**: Comprehensive test suite
- **`config/test_checkpoint_compatibility*.py`**: Test configurations

## Troubleshooting

### Issue: "Unexpected keys" warnings
**Solution**: These are normal when loading universal checkpoints. The smart loader handles them automatically.

### Issue: Training instability after resume
**Solution**: The optimizer starts fresh. Consider using a lower learning rate or warmup period.

### Issue: Memory usage differences
**Solution**: LoRA models use less memory. Adjust batch size accordingly when switching.

This feature provides a robust, flexible checkpoint system that adapts to your training needs without manual intervention.
