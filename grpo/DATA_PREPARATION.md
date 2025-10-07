# GRPO Data Preparation Guide

## Overview

GRPO training uses the same streaming data infrastructure as standard training. The data provider runs continuously in the background, generating masked input files as they are consumed by the training process.

## Running Data Preparation

### Command

```bash
python prepare.py grpo/grpo_config.py
```

### Expected Behavior

When you run the command, you should see:

```
Loaded composition config from data\char_diffusion\config\complex.py
```

Then the process will appear to "hang" - **this is normal!** The provider is running in an infinite loop, waiting to generate files when needed.

**IMPORTANT:** You do NOT need to wait for this to finish. You can:
1. Leave it running in the background (recommended)
2. Stop it with Ctrl+C after a few seconds (some files will be generated)
3. Start training immediately in another terminal - it will consume files as they're generated

### How It Works

1. **Initial Setup**: The provider checks if data files already exist
2. **Backlog Management**: It maintains a backlog of pre-generated files (default: 2 files)
3. **Continuous Generation**: When backlog is low, it generates new files
4. **Sleep When Full**: When backlog is full, it sleeps and checks periodically

### File Generation

Files are generated in:
- `data/char_diffusion/train/` - Training data
- `data/char_diffusion/val/` - Validation data

File naming convention: `{timestamp}-{sequence}-{batches}.pt`

Example: `1234567890-0-100.pt` (timestamp 1234567890, sequence 0, 100 batches)

### Stopping Data Generation

Press `Ctrl+C` to stop the provider. It will exit cleanly.

## Configuration

Key parameters in `grpo_config.py`:

```python
# Dataset
dataset = 'char_diffusion'
block_size = 1024
batch_size = 1

# Data streaming
data_prefer_queue = True
data_cache_files = 1
data_wait_sleep_seconds = 1.0
data_wait_timeout_seconds = None
data_stream_verbose = False

# Composition config (for multi-stage masking)
composition_config = 'complex'
```

### Composition Config

The `composition_config` parameter loads stage-based masking configuration from:
`data/char_diffusion/config/{composition_config}.py`

This defines:
- `use_all_stages_for_training`: Which stages to use for training
- `unmasking_stages`: Stage definitions for progressive unmasking
- `validation_stages`: Stages for validation data

## Running Training Without Pre-generating Data

You don't need to run `prepare.py` separately! The training script will automatically consume data from the queue, and the provider can run in parallel.

### Option 1: Start Training Directly

```bash
python grpo/train_grpo.py grpo/grpo_config.py
```

The training will wait for data files to appear. If no files exist, it will wait until the provider generates them.

### Option 2: Run Provider and Training in Parallel

**Terminal 1** (Data Provider):
```bash
python prepare.py grpo/grpo_config.py
```

**Terminal 2** (Training):
```bash
python grpo/train_grpo.py grpo/grpo_config.py
```

This is the recommended approach for continuous training.

## Troubleshooting

### "No data files found"

**Cause**: Provider hasn't generated files yet or files were deleted.

**Solution**: 
1. Run `python prepare.py grpo/grpo_config.py` in a separate terminal
2. Wait for files to appear in `data/char_diffusion/train/`
3. Start training

### "Provider not generating files"

**Cause**: Backlog is full (already has `max_backlog_files` files).

**Solution**: This is normal! The provider waits until files are consumed by training.

### "Loaded composition config but nothing happens"

**Cause**: This is expected behavior. The provider is running and waiting.

**Solution**: 
- Check `data/char_diffusion/train/` for generated files
- If files exist, the provider is sleeping (backlog full)
- If no files, wait a moment - generation is in progress

### "Training hangs waiting for data"

**Cause**: No data files available and provider not running.

**Solution**: Start the provider in a separate terminal.

## Data Generation Parameters

### From `grpo_config.py`

- `batch_size`: Number of samples per batch (affects file size)
- `block_size`: Sequence length
- `mask_probability`: Fraction of tokens to mask (default: 0.15)

### From Provider Defaults

- `batches_per_file`: Number of batches per file (default: 100)
- `max_backlog_files`: Maximum files to keep in queue (default: 2)
- `sleep_seconds`: Sleep duration when backlog full (default: 2.0)

## File Format

Each `.pt` file contains:

```python
{
    'tensors': {
        'x': torch.Tensor,      # Masked input (batch_size, block_size)
        'y': torch.Tensor,      # Original targets (batch_size, block_size)
        # Optional: 'attention_mask', 'metadata', etc.
    },
    'metadata': {
        'timestamp': float,
        'sequence': int,
        'num_batches': int,
        # Dataset-specific metadata
    }
}
```

## Best Practices

1. **Start provider first**: Generate initial files before training
2. **Monitor backlog**: Check `data/char_diffusion/train/` periodically
3. **Adjust backlog size**: Increase `max_backlog_files` for faster training
4. **Use separate terminals**: Run provider and training in parallel
5. **Clean old files**: Provider automatically removes consumed files

## Advanced: Custom Data Generation

To customize data generation, modify:
- `data/char_diffusion/prepare_streaming.py` - Provider implementation
- `data/char_diffusion/config/complex.py` - Stage configuration
- `grpo/grpo_config.py` - Training configuration

## Summary

- Data preparation runs continuously in the background
- "Loaded composition config" message is normal - provider is running
- No need to pre-generate all data - provider generates on-demand
- Training and provider can run in parallel
- Press Ctrl+C to stop provider when done

