# File-Based Logging System

The training script (`train.py`) uses a modular `TrainingLogger` class (defined in `logger.py`) that provides comprehensive file-based logging functionality.

## Overview

The logging system creates timestamped log files that contain:
1. Complete configuration dump at startup
2. Training progress including validation results
3. Automatic flushing to ensure data persistence even if training is interrupted

## Architecture

The logging functionality is implemented as a separate `TrainingLogger` class in `logger.py`, which is imported and used by `train.py`. This modular design keeps the training script clean and makes the logging system reusable.

## Configuration

### Default Settings

The logging system uses the following default configuration:

```python
log_dir = 'logs'        # Directory for log files
file_logging = True     # Enable/disable file logging
```

### Customizing Log Directory

You can customize the log directory in several ways:

#### 1. Command Line Override
```bash
python train.py --log_dir=my_custom_logs
```

#### 2. Configuration File
Create a config file (e.g., `config/my_config.py`):
```python
log_dir = 'experiments/run_001/logs'
```

Then run:
```bash
python train.py config/my_config.py
```

#### 3. Disable Logging
To disable file logging entirely:
```bash
python train.py --file_logging=False
```

## Log File Format

### File Naming
Log files are automatically named with timestamps:
```
log_run_YYYYMMDD_HHMMSS.txt
```

Example: `log_run_20240716_143052.txt`

### Log File Structure

#### 1. Header Section
```
================================================================================
Training run started at 2024-07-16 14:30:52
================================================================================
Configuration:
----------------------------------------
batch_size: 12
block_size: 1024
compile: True
dataset: openwebtext
...
----------------------------------------
```

#### 2. Training Progress
Each validation round logs:
```
[2024-07-16 14:35:22] step 2000: train loss 3.2145, val loss 3.1892
[2024-07-16 14:40:15] step 4000: train loss 2.9876, val loss 2.9654
[2024-07-16 14:45:08] step 6000: train loss 2.7543, val loss 2.7321
```

#### 3. Footer Section
```
================================================================================
Training run ended at 2024-07-16 16:45:30
================================================================================
```

## Features

### Automatic Directory Creation
The system automatically creates the log directory if it doesn't exist.

### Immediate Flushing
All log entries are immediately flushed to disk, ensuring data persistence even if training is interrupted unexpectedly.

### Master Process Only
In distributed training (DDP), only the master process (rank 0) performs logging to avoid duplicate entries.

### Complete Configuration Capture
The system captures all configuration parameters, including:
- Command line overrides
- Configuration file settings
- Default values

## Usage Examples

### Basic Usage
```bash
python train.py
```
Creates log file in `logs/log_run_YYYYMMDD_HHMMSS.txt`

### Custom Log Directory
```bash
python train.py --log_dir=experiment_logs
```

### With Configuration File
```bash
python train.py config/shakespeare.py --log_dir=shakespeare_logs
```

### Distributed Training
```bash
torchrun --standalone --nproc_per_node=4 train.py
```
Only the master process (rank 0) will create and write to the log file.

## TrainingLogger Class

The `TrainingLogger` class provides the following key methods:

### Initialization
```python
from logger import TrainingLogger

# Create logger instance
logger = TrainingLogger(log_dir='logs', enabled=True)
```

### Key Methods
- `setup(config)`: Initialize logging and dump configuration
- `log(message)`: Log a timestamped message
- `log_step(iter_num, train_loss, val_loss)`: Log training step results
- `close()`: Close log file and write footer

### Context Manager Support
```python
with TrainingLogger(log_dir='logs', enabled=True) as logger:
    logger.setup(config)
    logger.log_step(100, 2.5, 2.4)
    # Automatically closed when exiting context
```

## Integration with Existing Logging

The file logging system works alongside existing logging mechanisms:
- Console output (print statements) continue to work as before
- Weights & Biases logging (if enabled) continues to function
- File logging adds an additional persistent record

## Troubleshooting

### Permission Issues
If you encounter permission errors when creating log files:
1. Ensure the specified log directory is writable
2. Check disk space availability
3. Verify the process has write permissions to the target directory

### Missing Log Files
If log files are not created:
1. Verify `file_logging=True` in configuration
2. Check that you're running as the master process in DDP setups
3. Ensure the log directory path is valid

### Incomplete Logs
If logs appear incomplete:
- The system uses line buffering and explicit flushing
- Logs should be complete even if training is interrupted
- Check for disk space issues if logs stop abruptly
