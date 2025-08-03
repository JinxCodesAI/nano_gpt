
# nanoGPT

![nanoGPT](assets/nanogpt.jpg)

This is a fork of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) with modular architecture for training GPT models. The codebase has been refactored into a modular training system with dynamic scaling capabilities and architectural operations.

The original repository focuses on simplicity and readability. This fork extends that foundation by adding:

- Modular training architecture with separate modules for configuration, scheduling, evaluation, and operations
- Dynamic training orchestrator that can modify model architecture and hyperparameters during training
- Layer-specific operations for fine-grained control over model components
- Support for LoRA (Low-Rank Adaptation) training
- Scaling schedules that trigger operations based on validation loss thresholds
- Robust resumption capabilities with checkpoint management
- Multi-GPU distributed training support

## Training System Overview

### Main Training Script

The main training script is `train_refactored.py`, which provides a modular architecture with the following components:

- **Configuration Management** (`training/config.py`): Centralized configuration handling with dataset-specific defaults
- **Training Scheduler** (`training/scheduler.py`): Manages scheduled operations and learning rate scheduling  
- **Operations Engine** (`training/operations.py`): Executes architectural transformations and parameter modifications
- **Evaluation System** (`training/evaluation.py`): Handles model evaluation and analysis
- **Resume System** (`training/resume.py`): Robust checkpoint management and state restoration
- **Utilities** (`training/utils.py`): Timing profilers, batch management, and system monitoring

### Key Features

- **Dynamic Training Orchestrator**: Automatically triggers operations based on validation loss thresholds or iteration timeouts
- **Layer-Specific Operations**: Freeze/unfreeze individual layers, set LoRA ranks per layer, fine-grained control
- **Scaling Schedules**: YAML/JSON configuration files define sequences of operations to execute during training
- **DDP-Safe Architecture**: Full multi-GPU distributed training support with synchronized operations
- **Memory Management**: VRAM monitoring, batch size optimization, emergency checkpoint handling

## Install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm psutil pyyaml
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
- `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
- `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
- `tiktoken` for OpenAI's fast BPE code <3
- `wandb` for optional logging <3
- `tqdm` for progress bars <3
- `psutil` for system monitoring <3
- `pyyaml` for scaling schedule configuration <3

## Quick Start

### Basic Training

Train a character-level GPT on Shakespeare data:

```sh
# Prepare data
python data/shakespeare_char/prepare.py

# Train using the refactored training script
python train_refactored.py config/train_shakespeare_char.py
```

### Using Dynamic Scaling Operations

Train with automatic architectural scaling based on validation loss:

```sh
# Create a scaling schedule configuration
cat > configs/basic_scaling.yaml << EOF
- name: change_lr
  value: 0.5
  trigger_loss: 2.0
  max_wait_iters: 1000
- name: set_layer_lora_rank
  value: ["attn.0", 8]
  trigger_loss: 1.8
  max_wait_iters: 1500
EOF

# Train with scaling schedule
python train_refactored.py config/train_shakespeare_char.py --scaling_schedule_file=configs/basic_scaling.yaml
```

### Multi-GPU Training

For distributed training across multiple GPUs:

```sh
# Single node, 4 GPUs
torchrun --standalone --nproc_per_node=4 train_refactored.py config/train_gpt2.py

# Multi-node training
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=MASTER_IP --master_port=1234 train_refactored.py
```

## Advanced Training Features

### Scaling Schedule Configuration

The training orchestrator supports dynamic operations triggered by validation loss thresholds. Create YAML configuration files to define operation sequences:

```yaml
# Example scaling schedule
- name: change_lr
  value: 2.0              # Double learning rate
  trigger_loss: 6.0       # When val loss < 6.0
  max_wait_iters: 50000   # Or after 50k iterations

- name: set_layer_lora_rank
  value: ["attn.2", 16]   # Set attention layer 2 to LoRA rank 16
  trigger_loss: 5.5
  max_wait_iters: 75000

- name: freeze_layer
  value: "mlp.0"          # Freeze MLP in layer 0
  trigger_loss: 5.0
  max_wait_iters: 100000
```

### Layer-Specific Operations

Control individual model components:

- **Freeze/Unfreeze**: `freeze_layer`, `unfreeze_layer` for components like `attn.N`, `mlp.N`, `wte`, `lm_head`
- **LoRA Ranks**: Set LoRA ranks for specific layers with `set_layer_lora_rank`
- **Fine-grained Control**: Target specific transformer blocks and components

See `docs/LAYER_OPERATIONS_GUIDE.md` for complete layer naming conventions and usage patterns.

### Training Configuration

The system uses `training/config.py` for centralized configuration management:

```python
# Configure via configurator.py or command line
config = TrainingConfig()
config.scaling_schedule_file = "configs/my_schedule.yaml"
config.batch_size = 32
config.learning_rate = 6e-4
config.max_iters = 600000
```

### Memory Management and Monitoring

- **VRAM Monitoring**: Automatic VRAM usage tracking and warnings
- **Batch Size Optimization**: `adjust_batch_size` operation automatically finds optimal batch sizes
- **Emergency Checkpoints**: Automatic checkpoint saving on memory pressure
- **System Resource Monitoring**: CPU, memory, and GPU utilization tracking

### Resume and Checkpoint Management

Robust checkpoint handling with smart parameter transfer:

```sh
# Resume training from checkpoint
python train_refactored.py --init_from=resume --out_dir=path/to/checkpoints

# Resume with parameter overrides
python train_refactored.py --init_from=resume --learning_rate=3e-4 --batch_size=64
```

Features:
- Smart state dict loading with LoRA compatibility
- Optimizer state transfer across architectural changes
- Scaling schedule state restoration
- Emergency checkpoint fallback

## Documentation

### Core Documentation

- **`docs/README_ORCHESTRATOR.md`**: Complete guide to the Dynamic Training Orchestrator with DDP safety, configuration examples, and usage patterns
- **`docs/LAYER_OPERATIONS_GUIDE.md`**: Layer-specific operations guide covering freeze/unfreeze operations, LoRA rank setting, and naming conventions

### Training Modules

- **`training/config.py`**: Configuration management with validation and dataset-specific defaults
- **`training/scheduler.py`**: Training scheduler, learning rate scheduling, and early stopping
- **`training/operations.py`**: Architectural operations and model transformations
- **`training/evaluation.py`**: Model evaluation and analysis systems
- **`training/resume.py`**: Checkpoint management and state restoration
- **`training/utils.py`**: Utilities for timing, batch management, and system monitoring

### Configuration Examples

Sample scaling schedules are provided in the `configs/` directory demonstrating different training strategies and operation sequences.

## Usage Examples

### Basic Training

```sh
# Train from scratch
python train_refactored.py config/train_shakespeare_char.py

# Train with LoRA
python train_refactored.py config/train_shakespeare_char.py --attn_lora_rank=16 --embedding_rank=8
```

### Finetuning with Scaling Operations

```sh
# Prepare data
python data/shakespeare/prepare.py

# Finetune with dynamic operations
python train_refactored.py config/finetune_shakespeare.py --scaling_schedule_file=configs/finetune_schedule.yaml
```

### Advanced Configuration

```sh
# Custom configuration with overrides
python train_refactored.py \
  --init_from=gpt2-medium \
  --batch_size=32 \
  --learning_rate=3e-4 \
  --max_iters=50000 \
  --scaling_schedule_file=configs/progressive_scaling.yaml \
  --wandb_log=True \
  --wandb_project=my_experiment
```

## Sampling and Inference

Sample from trained models using the standard `sample.py` script:

```sh
# Sample from pre-trained models
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100

# Sample from your trained model
python sample.py --out_dir=out-shakespeare-char --start="ROMEO:"

# Use prompt from file
python sample.py --start=FILE:prompt.txt --out_dir=path/to/model
```

## Technical Notes

### Performance

- Uses PyTorch 2.0 with `torch.compile()` for improved performance
- Built-in timing profiler tracks forward/backward, evaluation, and operation execution times
- VRAM monitoring and automatic batch size optimization
- Multi-GPU DDP support with synchronized operations

### Architecture Features

- Modular design separating concerns (config, scheduling, operations, evaluation)
- Dynamic architectural transformations during training
- LoRA support for parameter-efficient fine-tuning
- Layer-wise freezing and unfreezing capabilities
- Robust checkpoint management with emergency fallback

### Memory Management

- Automatic VRAM usage monitoring
- Smart batch size calculation based on available memory
- Emergency checkpoint system for memory pressure situations
- Efficient parameter transfer during architectural changes

## Troubleshooting

### Common Issues

- **PyTorch Compile Errors**: Add `--compile=False` to disable torch.compile if experiencing issues
- **Memory Issues**: Use `adjust_batch_size` operation or set smaller batch sizes manually
- **DDP Issues**: Ensure all processes have the same scaling schedule configuration
- **Resume Issues**: Check checkpoint compatibility and use emergency checkpoints if main checkpoint is corrupted

### Debugging

- Check `logs/` directory for detailed training logs
- Use `--file_logging=True` for comprehensive logging
- Monitor VRAM usage in training output
- Review scaling schedule execution in logs

### Configuration Validation

The system validates configurations at startup and will report specific errors for:
- Missing checkpoint directories when resuming
- Invalid scaling schedule files
- Inconsistent model architecture parameters
- Missing vocabulary remapping files

## Original Repository

This fork extends the original [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) repository. For background on GPT architecture and language modeling, see Andrej Karpathy's [Zero To Hero series](https://karpathy.ai/zero-to-hero.html) and the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY).

## License

MIT License - see the original repository for details.
