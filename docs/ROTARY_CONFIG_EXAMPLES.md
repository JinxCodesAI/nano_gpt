# Rotary Embeddings Configuration Examples

This document provides example configurations for using rotary embeddings in nanoGPT.

## Basic Rotary Embeddings Configuration

### train_gpt2_rotary.py
```python
# config for training GPT-2 (124M) with rotary embeddings
# launch as: torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_rotary.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M-rotary'

# rotary embedding settings
use_rotary_embeddings = True
rotary_base = 10000.0
rotary_max_position_embeddings = 2048

# training settings (same as standard GPT-2)
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8
max_iters = 600000
lr_decay_iters = 600000

eval_interval = 1000
eval_iters = 200
log_interval = 10
weight_decay = 1e-1
```

### finetune_shakespeare_rotary.py
```python
import time

out_dir = 'out-shakespeare-rotary'
eval_interval = 5
eval_iters = 40
wandb_log = False
wandb_project = 'shakespeare'
wandb_run_name = 'ft-rotary-' + str(time.time())

dataset = 'shakespeare'
init_from = 'gpt2-xl'

# enable rotary embeddings for fine-tuning
use_rotary_embeddings = True
rotary_base = 10000.0
rotary_max_position_embeddings = 1024

always_save_checkpoint = False
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20
learning_rate = 3e-5
decay_lr = False
```

## Advanced Rotary Configurations

### Large Base for Better Extrapolation
```python
use_rotary_embeddings = True
rotary_base = 50000.0  # Larger base for better long-sequence handling
rotary_max_position_embeddings = 4096
```

### Custom Max Position
```python
use_rotary_embeddings = True
rotary_base = 10000.0
rotary_max_position_embeddings = 8192  # Support for longer sequences
```

## Command Line Usage Examples

### Basic Rotary Training
```bash
python train.py --use_rotary_embeddings=True --rotary_base=10000.0
```

### Custom Rotary Parameters
```bash
python train.py config/train_gpt2.py --use_rotary_embeddings=True --rotary_base=50000.0
```

### Fine-tuning with Rotary
```bash
python train.py config/finetune_shakespeare.py --use_rotary_embeddings=True
```

## Configuration Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_rotary_embeddings` | bool | False | Enable rotary position embeddings instead of learned embeddings |
| `rotary_base` | float | 10000.0 | Base for frequency computation in rotary embeddings |
| `rotary_max_position_embeddings` | int | 2048 | Maximum sequence length for rotary embedding precomputation |

## Compatibility Notes

- Models trained with `use_rotary_embeddings=False` cannot be loaded with `use_rotary_embeddings=True`
- Checkpoints include the rotary embedding configuration
- Training must be consistent within a single run
- Rotary embeddings typically use slightly less memory (no learned position embeddings)