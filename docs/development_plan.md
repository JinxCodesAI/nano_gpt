# Enhanced Data Augmentation Feature

## Overview

This feature implements a mixed training data approach to prevent overfitting by combining natural training data with model-generated enhanced data. The implementation modifies the `get_batch()` function to probabilistically mix real data samples with self-generated continuations.

## Feature Description

### How It Works

For each training batch when `split == 'train'`:
1. **Batch Composition**: Each element in the batch is determined to be either "natural" or "enhanced" based on configurable probability `p`
2. **Natural Elements**: Generated same way as before - random samples from training data
3. **Enhanced Elements**: Sampled from pre-generated rotating buffer of enhanced samples
4. **Result**: Combined batch of natural + enhanced rows, each of length `block_size`

### Enhanced Sample Generation Pipeline

**Background Generation Process**:
1. **Parallel Generation**: Continuously generates enhanced samples in background using separate thread/process
2. **Rotating Buffer**: Maintains buffer of N pre-generated enhanced samples (configurable size)
3. **Buffer Management**: When buffer is full, oldest samples are replaced with newly generated ones
4. **Model Synchronization**: Regenerates buffer contents when inference model is updated from checkpoint

### Configuration Parameters

- `enhanced_data_probability` (float, 0.0-1.0): Probability of enhanced vs natural data per batch element
- `min_prefix_length` (int): Minimum prefix length for enhanced data generation  
- `max_prefix_length` (int): Maximum prefix length for enhanced data generation
- `enhanced_generation_temperature` (float): Temperature for model generation
- `enhanced_generation_top_k` (int): Top-k sampling for model generation
- `enhanced_buffer_size` (int): Maximum number of pre-generated enhanced samples in rotating buffer
- `enhanced_generation_batch_size` (int): Batch size for parallel enhanced sample generation

## Implementation Details

### Modified Functions

#### `get_batch(split)`
- **Location**: `train.py:116-131`
- **Changes**: Added logic to handle mixed natural/enhanced data when `split == 'train'` and `enhanced_data_probability > 0`
- **Dependencies**: Requires access to separate inference model for generation
- **Behavior**: Returns same tensor shapes as original, but with mixed data sources

#### Training Loop Modifications
- **Inference Model Management**: Initialize separate inference model copy
- **Checkpoint Synchronization**: Reload inference model when new checkpoint is saved (validation loss improvement)
- **Memory Efficiency**: Inference model only created when `enhanced_data_probability > 0`

#### New Classes and Functions Added

1. **`EnhancedSampleBuffer` class**
   - Thread-safe rotating buffer for pre-generated enhanced samples
   - Methods: `get_samples(n)`, `add_samples(samples)`, `clear()`, `is_full()`, `size()`
   - Handles buffer overflow by replacing oldest samples
   - **Sample Reuse**: Samples are reused (sampling with replacement) for efficiency when generation is slower than consumption

2. **`EnhancedSampleGenerator` class**
   - Background generator running in separate thread
   - Continuously generates enhanced samples and adds to buffer
   - Methods: `start()`, `stop()`, `update_model(checkpoint_path)`, `is_running()`

3. **`determine_batch_composition(batch_size, probability)`**
   - Returns boolean mask indicating which batch elements should be enhanced
   - Uses reproducible random sampling

4. **`generate_enhanced_samples_batch(inference_model, data, batch_size, device, ctx)`**
   - Generates batch of enhanced samples for buffer
   - Returns list of (x, y) tensor pairs ready for training

5. **`sample_random_fragments(sequences, block_size)`**
   - Extracts random fragments of exact `block_size` from generated sequences
   - Handles edge cases where sequences are shorter than expected

### Memory and Performance Considerations

- **Separate Inference Model**: Maintains a dedicated copy of the model for generation (always in eval mode)
- **Background Generation**: Enhanced samples generated in parallel thread, no training bottleneck
- **Rotating Buffer**: Pre-generated samples ready for immediate use during training
- **Memory Management**: 
  - Buffer size configurable to balance memory usage vs sample diversity
  - Efficient buffer replacement strategy (FIFO)
  - Enhanced generation batched to optimize GPU utilization
- **Performance**: 
  - Zero latency for enhanced sample retrieval during training
  - Minimal overhead when `enhanced_data_probability = 0.0` (disabled)
  - Background generation rate adjustable based on consumption
  - **Sample Reuse**: Graceful handling when generation is slower than consumption by reusing available samples
- **Thread Safety**: Thread-safe buffer operations for concurrent access
- **Device Consistency**: Maintains proper device placement for all tensors

### Edge Cases Handled

1. **Short Sequences**: When generated sequence is shorter than `block_size`
2. **Device Mismatch**: Ensures all tensors are on correct device
3. **Empty Batches**: When no enhanced elements are selected
4. **Model Compilation**: Compatible with PyTorch 2.0 model compilation
5. **Inference Model Lag**: Handles cases where inference model is behind training model
6. **Buffer Underflow**: Graceful fallback to natural data when buffer is empty, sample reuse when buffer has fewer samples than requested
7. **Thread Termination**: Proper cleanup of background generation thread
8. **Memory Pressure**: Buffer size auto-adjustment based on available memory

## Testing Strategy

### Unit Tests
- `test_enhanced_sample_buffer()`: Buffer operations, thread safety, FIFO behavior
- `test_enhanced_sample_generator()`: Background generation, model updates, thread lifecycle
- `test_determine_batch_composition()`: Verify probability distributions
- `test_generate_enhanced_samples_batch()`: Check batch generation functionality  
- `test_sample_random_fragments()`: Validate fragment sampling
- `test_enhanced_get_batch()`: Integration test for modified `get_batch()`

### Integration Tests
- **Shape Consistency**: Verify output tensors match expected shapes
- **Content Validation**: Check that enhanced data contains model-generated content
- **Performance**: Benchmark overhead with different probability settings
- **Memory Usage**: Monitor memory consumption during generation

### Configuration Tests
- **Parameter Validation**: Test edge cases for all config parameters
- **Reproducibility**: Ensure consistent results with same random seeds
- **Backward Compatibility**: Verify original behavior when disabled

## Usage Examples

### Basic Usage (10% enhanced data)
```bash
python train.py --enhanced_data_probability=0.1
```

### Custom Configuration
```bash
python train.py --enhanced_data_probability=0.2 \
                --min_prefix_length=128 \
                --max_prefix_length=256 \
                --enhanced_generation_temperature=0.9
```

### Disabled (Default)
```bash
python train.py  # enhanced_data_probability=0.0 by default
```

## Expected Benefits

1. **Reduced Overfitting**: Model sees variations of training data rather than exact repeats
2. **Improved Generalization**: Exposure to model's own generation style during training
3. **Configurable Impact**: Tunable probability allows gradual introduction
4. **Backward Compatible**: Zero impact when disabled (default state)

## Future Enhancements

- **Dynamic Probability**: Adjust probability based on training progress
- **Quality Filtering**: Filter generated sequences based on perplexity or other metrics
- **Multi-Model Generation**: Use ensemble of models for enhanced data generation
- **Adaptive Prefix Length**: Dynamically adjust prefix lengths based on model performance