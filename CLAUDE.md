# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanoGPT is a minimal, educational implementation of GPT (Generative Pre-trained Transformer) optimized for training and finetuning medium-sized language models. The codebase prioritizes simplicity and readability over abstraction - the core model and training loop are contained in just two ~300-line files.

## Key Commands

### Training
```bash
# Train character-level GPT on Shakespeare (quick start)
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py

# Train GPT-2 124M reproduction on OpenWebText (requires 8x A100 GPUs)
python data/openwebtext/prepare.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# Single GPU training with custom parameters
python train.py --batch_size=32 --compile=False --device=cpu

# Finetune pretrained model
python train.py config/finetune_shakespeare.py
```

### Sampling/Inference
```bash
# Sample from trained model
python sample.py --out_dir=out-shakespeare-char

# Sample from OpenAI GPT-2 models
python sample.py --init_from=gpt2-xl --start="What is the answer to life?" --num_samples=5
```

### Benchmarking
```bash
# Performance benchmarking
python bench.py

# Evaluate pretrained models
python train.py config/eval_gpt2.py
```

### Dependencies
Install with: `pip install torch numpy transformers datasets tiktoken wandb tqdm`

## Architecture

### Core Files
- **`model.py`**: Complete GPT implementation (~300 lines)
  - `GPTConfig`: Model configuration dataclass
  - `GPT`: Main model class with transformer blocks
  - `CausalSelfAttention`: Multi-head attention with Flash Attention support
  - `MLP`: Feed-forward network
  - `LayerNorm`: Custom layer normalization
  - `Block`: Transformer block combining attention and MLP

- **`train.py`**: Training loop and distributed training setup (~300 lines)
  - Supports single GPU and multi-GPU (DDP) training
  - Configurable via command line args or config files
  - Handles checkpointing, evaluation, and logging
  - Gradient accumulation for large effective batch sizes

- **`sample.py`**: Text generation from trained models
  - Supports temperature and top-k sampling
  - Can load from checkpoints or OpenAI pretrained models

### Configuration System
Uses `configurator.py` for flexible parameter overriding:
- Config files (e.g., `config/train_gpt2.py`) set base parameters
- Command line args (`--batch_size=32`) override specific values
- All scripts use: `exec(open('configurator.py').read())`

### Data Pipeline
- **Preparation scripts**: `data/{dataset}/prepare.py` tokenize raw text into binary files
- **Binary format**: `train.bin` and `val.bin` contain tokenized sequences as uint16 arrays
- **Datasets supported**: Shakespeare (character-level), OpenWebText (GPT-2 tokenizer)

## Development Notes

### Model Initialization
- Supports three init modes: `'scratch'`, `'resume'` (from checkpoint), or `'gpt2*'` (OpenAI weights)
- GPT-2 weight loading enables transfer learning and finetuning
- Automatic parameter counting and weight initialization

### Training Features
- **Flash Attention**: Automatically enabled for PyTorch >= 2.0
- **Compilation**: `torch.compile()` for ~2x speedup (disable with `--compile=False`)
- **Mixed Precision**: Supports bfloat16/float16 for memory efficiency
- **Distributed Training**: DDP support for multi-GPU/multi-node setups
- **Gradient Accumulation**: Simulates larger batch sizes on smaller hardware

### Device Support
- CUDA GPUs (primary target)
- CPU training (slower, use `--device=cpu --compile=False`)
- Apple Silicon (`--device=mps` for M1/M2 Macs)

### Configuration Patterns
Each config file in `config/` follows the pattern:
```python
# Override default parameters from train.py
batch_size = 12
block_size = 1024
# ... other overrides
```

### No Testing Framework
This codebase has no formal test suite - it prioritizes simplicity and educational clarity over extensive testing infrastructure.