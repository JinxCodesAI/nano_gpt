# Experiment Runner

A comprehensive experiment runner for running multiple diffusion training experiments sequentially with proper error handling, logging, and result organization.

## Features

- **Sequential execution**: Runs experiments one after another to avoid resource conflicts
- **Robust error handling**: Failed experiments don't stop the entire batch
- **Real-time logging**: Full stdout/stderr capture with immediate flushing and streaming
- **Organized output**: Each experiment gets its own directory with logs and checkpoints
- **Global overrides**: Easy configuration of batch size, wandb settings, device, etc.
- **Result tracking**: JSON summaries and metadata for each experiment
- **Real-time monitoring**: See progress in master log and individual experiment logs

## Quick Start

### Basic Usage

```bash
# Run all configs in a directory (use "cuda" not "gpu" for GPU)
python run_experiments.py --configs config/shkspr_char_diff/*.py --device cuda --batch_size 8

# Run specific configs
python run_experiments.py --configs easy_first.py moderate_first.py --batch_size 16

# CPU training with wandb disabled
python run_experiments.py --configs *.py --device cpu --wandb_enabled false --batch_size 4
```

### Global Override Options

- `--batch_size INT`: Override batch size for all configs
- `--wandb_project STR`: Override wandb project name  
- `--wandb_enabled true/false`: Enable/disable wandb logging
- `--device STR`: Override device (cuda, cpu, mps, cuda:0, etc.) - Use "cuda" not "gpu"
- `--compile true/false`: Enable/disable torch.compile
- `--max_iters INT`: Override maximum training iterations
- `--learning_rate FLOAT`: Override learning rate
- `--base_out_dir STR`: Base output directory (default: experiments)

## Output Structure

Each experiment batch creates a timestamped directory:

```
experiments/
└── experiment_20240128_143022/
    ├── master_log.txt              # Master log for entire batch
    ├── experiment_summary.json     # Summary of all experiments
    ├── easy_first/                 # Individual experiment directory
    │   ├── stdout.log             # Training stdout
    │   ├── stderr.log             # Training stderr  
    │   ├── combined.log           # Combined stdout/stderr
    │   ├── config_used.py         # Copy of config file used
    │   ├── experiment_metadata.json # Experiment metadata
    │   ├── experiment_result.json   # Individual result
    │   └── ckpt_unmasking_*.pt     # Training checkpoints
    ├── moderate_first/
    │   └── ...
    └── super_hard_first/
        └── ...
```

## Log Files Explained

- **master_log.txt**: High-level progress and summary for the entire batch
- **stdout.log**: Complete stdout from the training process (real-time flushed)
- **stderr.log**: Complete stderr (warnings, errors, real-time flushed)
- **combined.log**: Interleaved stdout/stderr with timestamps (real-time flushed)
- **experiment_result.json**: Status, duration, errors, checkpoint info

**Real-time Logging**: All log files are flushed immediately as output is produced by the training process, ensuring you can monitor progress in real-time using `tail -f` commands.

## Error Handling

- **Individual failures**: Failed experiments are logged but don't stop the batch
- **Exception handling**: Python exceptions are caught and logged with traceback
- **Process monitoring**: Non-zero exit codes are detected and reported
- **Resource cleanup**: Processes are properly terminated on interruption
- **State preservation**: Partial results are saved even if batch is interrupted

## Monitoring Progress

### Real-time Monitoring
```bash
# Watch master log
tail -f experiments/experiment_*/master_log.txt

# Watch specific experiment
tail -f experiments/experiment_*/config_name/stdout.log
```

### Check Results
```bash
# View summary
python -c "import json; print(json.dumps(json.load(open('experiments/experiment_*/experiment_summary.json')), indent=2))"

# Check individual result
cat experiments/experiment_*/config_name/experiment_result.json
```

## Examples

### Quick Test Run
```bash
python run_experiments.py \
  --configs config/shkspr_char_diff/easy_first.py \
  --max_iters 1000 \
  --batch_size 8 \
  --base_out_dir quick_test
```

### Production Run with Wandb
```bash
python run_experiments.py \
  --configs config/shkspr_char_diff/*.py \
  --wandb_project "diffusion_experiments_v2" \
  --batch_size 16 \
  --base_out_dir production_run \
  --learning_rate 1e-3
```

### CPU Development Run
```bash
python run_experiments.py \
  --configs config/shkspr_char_diff/easy_first.py \
  --device cpu \
  --compile false \
  --wandb_enabled false \
  --batch_size 4 \
  --max_iters 500
```

### Multi-GPU Setup
```bash
# Run on specific GPU
python run_experiments.py --configs *.py --device cuda:1 --batch_size 32

# For multi-GPU training, set environment variables
CUDA_VISIBLE_DEVICES=0,1 python run_experiments.py --configs *.py
```

## Configuration Override

The experiment runner can override any configuration parameter from your config files. Global overrides take precedence over config file settings.

### Wandb Override Examples
```bash
# Disable wandb for all experiments
python run_experiments.py --configs *.py --wandb_enabled false

# Use custom project name
python run_experiments.py --configs *.py --wandb_project "my_experiment_batch"

# Enable wandb even if configs have it disabled
python run_experiments.py --configs *.py --wandb_enabled true --wandb_project "override_project"
```

## Troubleshooting

### Common Issues

1. **Config files not found**: Ensure paths are correct and use quotes for glob patterns
   ```bash
   python run_experiments.py --configs "config/shkspr_char_diff/*.py"
   ```

2. **Out of memory**: Reduce batch size globally
   ```bash
   python run_experiments.py --configs *.py --batch_size 4
   ```

3. **CUDA errors**: Switch to CPU or specific GPU
   ```bash
   python run_experiments.py --configs *.py --device cpu
   python run_experiments.py --configs *.py --device cuda:0
   ```

4. **Wandb issues**: Disable wandb or check authentication
   ```bash
   python run_experiments.py --configs *.py --wandb_enabled false
   ```

### Debugging Failed Experiments

1. Check the stderr.log file for the failed experiment
2. Look at experiment_result.json for error details
3. Check master_log.txt for high-level information
4. Verify config file syntax if experiments fail to start

### Interrupting and Resuming

- **Ctrl+C**: Gracefully stops current experiment and saves results
- **Kill signal**: Results saved up to the interruption point
- **Resuming**: Currently no built-in resume (start new batch), but checkpoints are preserved

## Performance Tips

- Use `--compile true` for faster training (requires PyTorch 2.0+)
- Set appropriate batch sizes based on GPU memory
- Use `--device cuda:X` to specify GPU for multi-GPU systems
- Monitor GPU memory usage during experiments
- Consider running shorter experiments first to validate setup

## Integration with Training Code

The experiment runner works by:
1. Passing config file as positional argument and `--out_dir=path` to the training command
2. Adding any global overrides as `--key=value` arguments
3. Using the existing configurator.py mechanism in train_run.py
4. Capturing all output via subprocess with real-time streaming

This means it's compatible with the existing training infrastructure and config system.