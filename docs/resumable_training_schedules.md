# Resumable Training Schedules

This document describes the resumable training schedules feature implemented in the training orchestrator.

## Overview

The resumable training schedules feature allows training runs to be interrupted and resumed while preserving the state of which operations have already been completed. This prevents re-execution of operations that have already been performed and ensures consistent training progression.

## Key Features

1. **Stateful Operations**: Each operation in a schedule now tracks whether it has been completed
2. **Resume-Aware Loading**: Schedule loading behavior differs based on whether training is starting from scratch or resuming
3. **Persistent State**: Operation completion status is saved to the schedule file automatically
4. **Backward Compatibility**: Existing schedule files without completion status are handled gracefully

## Implementation Details

### Schedule File Format

Each operation in a schedule file can optionally include a `completed` field. If not present, it defaults to `false`:

```json
{
  "name": "change_lr",
  "value": 0.8,
  "trigger_loss": 15.0,
  "max_wait_iters": 50,
  "desc": "First LR reduction",
  "reevaluate": false
}
```

The `completed` field will be automatically added when the schedule is saved:

```json
{
  "name": "change_lr",
  "value": 0.8,
  "trigger_loss": 15.0,
  "max_wait_iters": 50,
  "desc": "First LR reduction",
  "reevaluate": false,
  "completed": false
}
```

### Key Functions

#### `save_scaling_schedule(file_path, schedule_data)`
- Saves the updated schedule data back to YAML or JSON files
- Automatically detects file format based on extension
- Preserves formatting and structure

#### `load_scaling_schedule(file_path, init_from)`
- Loads schedule and handles completion status based on `init_from` parameter
- If `init_from == 'scratch'`: resets all operations to `completed: false`
- If `init_from == 'resume'`: preserves existing completion status
- Adds `completed: false` to operations that don't have this field

### Main Loop Changes

The orchestration loop now:
1. Finds the next uncompleted operation instead of always taking the first one
2. Marks operations as completed when they succeed
3. Saves the updated schedule to disk after each completion
4. Logs when all operations are complete

## Usage

### Starting from Scratch

```python
# In your config file
init_from = 'scratch'
scaling_schedule_file = 'configs/my_schedule.json'
```

When starting from scratch:
- All operations are reset to `completed: false`
- Operations execute in order as they meet trigger conditions

### Resuming Training

```python
# In your config file  
init_from = 'resume'
scaling_schedule_file = 'configs/my_schedule.json'
```

When resuming:
- Existing completion status is preserved
- Only uncompleted operations are considered for execution
- Completed operations are skipped

### Example Workflow

1. **Initial Training Run**:
   ```bash
   python train.py config/my_config.py
   ```
   - Operations execute and are marked as completed
   - Training is interrupted (Ctrl+C)

2. **Resume Training**:
   ```bash
   # Edit config to set: init_from = 'resume'
   python train.py config/my_config.py
   ```
   - Only remaining uncompleted operations will execute
   - Previously completed operations are skipped

## Utilities

### Updating Existing Schedule Files

Use the provided utility to add `completed` fields to existing schedule files:

```bash
# Update a specific file
python update_schedule_format.py configs/my_schedule.json

# Update all files in configs/ directory
python update_schedule_format.py --configs-dir

# Skip creating backup files
python update_schedule_format.py --no-backup configs/my_schedule.json
```

### Testing

Run the test suite to verify functionality:

```bash
python test_resumable.py
```

### Demo

Create and run a demonstration:

```bash
python demo_resumable.py
```

## Backward Compatibility

- **No modifications required**: Existing schedule files without `completed` fields work seamlessly
- **Automatic handling**: Missing `completed` fields are treated as `false` and automatically added on save
- **Zero breaking changes**: No changes required to existing configurations or workflows

## Benefits

1. **Fault Tolerance**: Training can be safely interrupted and resumed
2. **Experimentation**: Easy to test different continuation strategies
3. **Resource Management**: Avoid re-executing expensive operations
4. **Debugging**: Clear visibility into which operations have completed
5. **Flexibility**: Can restart from scratch or resume as needed

## Technical Notes

- Schedule state is saved after each successful operation
- DDP (Distributed Data Parallel) synchronization is maintained
- File I/O errors are handled gracefully
- Both JSON and YAML formats are supported
- Operation completion is tracked in-memory and persisted to disk

## Migration Guide

For existing users:

1. **No action required** - existing schedules work perfectly as-is
2. **Optional**: Run `update_schedule_format.py` to add explicit `completed` fields (not necessary)
3. **Recommended**: Test resume functionality with your specific schedules
4. **Best Practice**: Use version control for schedule files to track changes

### New Schedule Files

When creating new schedule files, you can omit the `completed` field entirely:

```json
[
  {
    "name": "change_lr",
    "value": 0.8,
    "trigger_loss": 10.0,
    "max_wait_iters": 100,
    "reevaluate": false
  }
]
```

The system will automatically handle completion tracking.

## Troubleshooting

### Common Issues

1. **File Permission Errors**: Ensure write permissions for schedule files
2. **Format Errors**: Verify JSON/YAML syntax is valid
3. **Missing Fields**: Use the update utility to add required fields

### Debugging

- Check console output for schedule loading messages
- Verify `completed` fields in schedule files after operations
- Use the test script to validate basic functionality

## Future Enhancements

Potential improvements:
- Operation rollback capabilities
- Conditional operation dependencies
- Schedule validation tools
- Web-based schedule editor
- Integration with experiment tracking systems
