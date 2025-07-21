# Training Orchestrator Implementation

This document describes the complete implementation of the Dynamic Training Orchestrator for GPT model training, as specified in `docs/training_morfing_operations.md`.

## Overview

The Training Orchestrator enables dynamic adjustment of both training hyperparameters and model architecture during training based on validation loss thresholds and timeout conditions. This implementation includes all operations from the specification, including complex architectural transformations with optimizer state preservation.

## Features Implemented

### ✅ Core Infrastructure
- **Absolute Value Operations**: Direct setting of training parameters and architectural values
- **Scaling Schedule**: YAML/JSON configuration loading for operation sequences
- **Main Training Loop Integration**: Automatic trigger checking and operation execution
- **Operation Execution Framework**: Master function handling all operation types

### ✅ Hyperparameter Operations
- `set_lr`: Set learning rate to specified absolute value
- `set_batch_size`: Set batch size to specified absolute value
- `set_grad_accum`: Set gradient accumulation steps to specified absolute value
- `reset_lr_schedule`: Reset learning rate warmup/decay schedule offset
- `set_warmup_iters`: Set warmup iterations to specified absolute value
- `set_eval_iters`: Set evaluation iterations to specified absolute value
- `set_eval_interval`: Set evaluation frequency to specified absolute value

### ✅ Architectural Operations
- `stack_layers`: Duplicate existing layers using explicit layer map for model depth growth
- `widen_mlp`: Widen MLP layers to specified absolute width using Net2WiderNet algorithm
- `set_attn_lora_rank`: Set attention LoRA rank to specified absolute value
- `set_embedding_lora_rank`: Set embedding LoRA rank to specified absolute value
- `merge_lora_weights`: Merge LoRA adapters into main weights and reset

### ✅ Advanced Features
- **LoRA Support**: Complete LoRA embedding and linear layer implementations
- **Optimizer State Preservation**: Name-based state transfer with parameter names saved in checkpoints
- **Universal Checkpoint Compatibility**: Seamless loading between LoRA and non-LoRA configurations
- **DDP Safety**: All operations work correctly in distributed training
- **Torch.compile Compatibility**: Proper wrapper management for compiled models

### ✅ Testing & Validation
- **Comprehensive Test Suite**: All operations tested individually and in combination
- **Critical Bug Fixes Applied**: Fixed LoRAEmbedding matrix transposition and optimizer state transfer
- **Integration Tests**: Complete scaling schedule simulation and checkpoint compatibility
- **Architectural Operation Tests**: Layer stacking validation with negative index checks
- **Sample Configurations**: Multiple example schedules for different scenarios

## Usage

### 1. Configuration

Create a scaling schedule configuration file (YAML or JSON):

```json
[
    {
        "name": "set_lr",
        "value": 0.0012,
        "trigger_loss": 6.0,
        "max_wait_iters": 1000,
        "reevaluate": false
    },
    {
        "name": "stack_layers",
        "value": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "trigger_loss": 5.5,
        "max_wait_iters": 2000,
        "reevaluate": true
    },
    {
        "name": "widen_mlp",
        "value": 4096,
        "trigger_loss": 5.2,
        "max_wait_iters": 2000,
        "reevaluate": true
    },
    {
        "name": "reset_lr_schedule",
        "value": null,
        "trigger_loss": 5.19,
        "max_wait_iters": 500,
        "reevaluate": false
    },
    {
        "name": "set_attn_lora_rank",
        "value": 64,
        "trigger_loss": 4.8,
        "max_wait_iters": 3000,
        "reevaluate": false
    },
    {
        "name": "merge_lora_weights",
        "value": null,
        "trigger_loss": 4.5,
        "max_wait_iters": 5000,
        "reevaluate": true
    }
]
```

### 2. Training Script Usage

Set the scaling schedule file in your training configuration:

```bash
python train.py --scaling_schedule_file=configs/demo_scaling_schedule.json
```

Or modify the `scaling_schedule_file` parameter in `train.py` directly.

### 3. Model Configuration for LoRA

When using LoRA operations, configure the model with LoRA parameters:

```python
# In config/train_gpt2_lora.py
embedding_mode = 'lora'
embedding_rank = 32  # Absolute LoRA rank for embedding layer
attn_lora_rank = 16  # Absolute LoRA rank for attention layers
lora_alpha = 2.0     # LoRA scaling factor
```

### 4. Monitoring

The orchestrator will log operation executions with detailed information:

```
=== SCALING OPERATION TRIGGERED (DDP SYNC) ===
Operation: stack_layers
Trigger reason: Loss threshold
Current val loss: 5.4, Trigger loss: 5.5
Iterations since last op: 1500, Max wait: 2000

Performing architectural operation: stack_layers
Re-stacking layers based on map: [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]. New depth will be 12.
Model now has 12 layers.
Re-configuring optimizer after architectural change...
Transferred optimizer state for 156 / 312 parameters
Re-compiling the model...
Re-wrapping model in DDP...
Architectural operation completed successfully.

Re-evaluating validation loss after operation...
New val loss after operation: 5.35
=== SCALING OPERATION COMPLETE ===
```

## Configuration Files

### Sample Configurations
- `configs/demo_scaling_schedule.json`: Complete example with all operation types
- `configs/first_scaling_schedule.json`: Current production configuration  
- `configs/sample_scaling_schedule.json`: (Removed) Legacy basic operations only

### Configuration Format

Each operation requires:
- `name`: Operation function name
- `value`: Parameter for the operation (absolute value or layer map for architectural ops)
- `trigger_loss`: Validation loss threshold to trigger operation
- `max_wait_iters`: Maximum iterations to wait before forcing operation
- `reevaluate`: Whether to immediately re-evaluate validation loss after operation

## Testing

### Run Operation Tests
```bash
python test_operations.py          # Test all architectural operations
python test_critical_fixes.py      # Verify critical bug fixes
python test_operations_integration.py  # Integration testing
```

### Run Manual Simulation
```bash
python test_manual.py              # Manual orchestrator simulation
```

## Implementation Details

### Configuration Parameters
Located in `train.py` lines 80-88:
- `embedding_mode`: 'standard' or 'lora' for embedding layer type
- `attn_lora_rank`: Absolute LoRA rank for attention layers (0 disables LoRA)
- `embedding_rank`: Absolute LoRA rank for embedding layer (0 disables LoRA)
- `lora_alpha`: LoRA scaling factor
- `scaling_schedule_file`: Path to scaling schedule config file (YAML/JSON)
- `scaling_schedule`: Will be loaded from file or set programmatically
- `target_architecture_config`: Global state for target architecture

### Operation Execution
The `execute_operation()` function (`train.py:539-643`) handles all operations with:

**Hyperparameter Operations**: Direct parameter updates with absolute values
**Architectural Operations**:
1. Model unwrapping from DDP/compile wrappers
2. Optimizer state preservation using parameter names
3. Model architectural transformation with absolute values/layer maps
4. Optimizer recreation and state transfer
5. Re-application of DDP and compile wrappers

**Validation includes**:
- Valid operation names and parameters
- Proper layer map format for stack_layers operations
- Positive values for architectural parameters

### Trigger Logic
Operations are triggered when either:
1. **Loss Threshold**: `current_val_loss < trigger_loss`
2. **Timeout**: `(current_iter - iter_of_last_op) >= max_wait_iters`

### Optimizer State Preservation
The `transfer_optimizer_state()` function (`train.py:114-144`) preserves training stability by:
1. **Name-based Mapping**: Uses parameter names saved in checkpoints instead of object identity
2. **State Transfer**: Copies Adam momentum and variance for preserved parameters
3. **Robust Handling**: Works correctly with architectural changes that create new parameter objects
4. **Checkpoint Integration**: Parameter names are saved in checkpoints to enable state transfer on resume

### LoRA Implementation
Complete LoRA support in `model.py`:
- **LoRAEmbedding**: LoRA adapter for embedding layers with merge/reset
- **LoRALinear**: LoRA adapter for linear layers with proper initialization
- **GPTConfig Updates**: New LoRA configuration parameters

## Critical Fixes Applied

### 1. LoRAEmbedding Matrix Transposition Bug (FIXED)
**Problem**: Incorrect matrix transposition in get_merged_state_dict causing dimension mismatch
**Solution**: Removed erroneous .T at end of LoRAEmbedding merge calculation

### 2. Optimizer State Transfer Enhancement (IMPLEMENTED)
**Problem**: Optimizer momentum lost when switching between LoRA configurations
**Solution**: Save parameter names in checkpoints and use transfer_optimizer_state function

### 3. Operations System Refactor (COMPLETED)
**Problem**: Relative multiplier/divisor system was confusing and error-prone
**Solution**: Moved to absolute value operations for clarity and reliability

## Future Extensions

**Potential enhancements**:
- Masked Structural Growth (MSG) for mathematically rigorous growth
- Adaptive scheduling based on gradient analysis
- More sophisticated LoRA rank adjustment with state interpolation

## Files Modified/Added

### Core Implementation
- `train.py`: Complete orchestrator with absolute operations (lines 80-88: config params, 539-643: execution)
- `model.py`: LoRA modules, GPTConfig updates, architectural operations (stack_layers, widen_mlp, etc.)
- `configs/`: Configuration files directory
  - Various example configurations for different scenarios

### Testing
- `test_checkpoint_compatibility.py`: Universal checkpoint compatibility testing
- `test_stack_layers_validation.py`: Layer stacking validation testing
- Various config files in `config/` directory for testing different scenarios

### Documentation
- `docs/Training_Orchestration.md`: Complete implementation documentation (this file)
- `docs/training_morfing_operations.md`: Original functional specification
- `docs/advanced operations_review.md`: Critical issues review and fixes applied
- `docs/universal_checkpoint_compatibility.md`: Checkpoint system documentation

## Status

✅ **COMPLETE**: All operations refactored to use absolute values for clarity and reliability
✅ **COMPLETE**: Critical bugs fixed including LoRAEmbedding transposition and optimizer state transfer
✅ **COMPLETE**: Universal checkpoint compatibility between LoRA and non-LoRA configurations
✅ **COMPLETE**: Robust optimizer state preservation with parameter names saved in checkpoints
✅ **COMPLETE**: Comprehensive validation including negative index checks for stack_layers
✅ **COMPLETE**: DDP and torch.compile compatibility maintained

**The implementation is production-ready with all critical issues resolved and comprehensive testing completed.**
