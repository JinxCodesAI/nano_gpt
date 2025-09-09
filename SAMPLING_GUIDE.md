# Diffusion Sampling Guide

This guide explains how to use the new diffusion-based sampling implementation in `sample.py`.

## Quick Start

### Basic Diffusion Sampling
```bash
python sample.py --sampling_method=diffusion --num_samples=4 --sequence_length=512
```

### Standard Autoregressive Sampling
```bash
python sample.py --sampling_method=standard --max_new_tokens=100 --start_text="To be or not to be"
```

## Key Features

### 1. Diffusion-based Iterative Demasking
- Starts with all tokens masked
- Iteratively predicts and unmasks tokens over multiple steps
- Supports various remasking schedules (linear, custom)
- Optional intelligent remasking using model confidence

### 2. Multiple Sampling Methods
- **Diffusion**: Iterative demasking approach (like sample2.py)  
- **Standard**: Traditional autoregressive generation

### 3. Advanced Sampling Controls
- **Temperature**: Controls randomness (0.1 = deterministic, 2.0 = very random)
- **Top-p (Nucleus)**: Sample from top cumulative probability mass
- **Repetition penalty**: Discourage repetitive text
- **Custom schedules**: Define your own masking ratios

## Configuration Options

### Model Loading
```python
checkpoint_name = 'big_boy2.pt'  # Your trained checkpoint
remasking_checkpoint_name = None  # Optional remasking model
```

### Diffusion Parameters
```python
diffusion_iterations = 50        # Number of demasking steps
start_ratio = 0.95              # Start with 95% tokens masked
end_ratio = 0.05                # End with 5% tokens masked
temperature = 0.8               # Sampling temperature
top_p = 1.0                     # Nucleus sampling (1.0 = disabled)
```

### Remasking Strategy
```python
randomness_strength = 0.4       # 0.0 = pure model-guided, 1.0 = pure random
intelligent_remasking = True    # Use model confidence for remasking
```

### Custom Schedules
```python
schedule_type = 'custom'
masking_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
```

## Advanced Usage

### 1. Using a Remasking Model
If you have a trained remasking model (for intelligent token selection):
```bash
python sample.py --remasking_checkpoint_name=remasking_model.pt --randomness_strength=0.2
```

### 2. Custom Masking Schedule
For fine-grained control over the demasking process:
```bash
python sample.py --schedule_type=custom --masking_ratios="[0.95,0.8,0.6,0.4,0.2,0.05]"
```

### 3. Debug Mode
For detailed analysis of the generation process:
```bash
python sample.py --log_debug=True --use_verbose_logging=True
```

### 4. High-Quality Generation
For maximum quality (slower):
```bash
python sample.py --diffusion_iterations=100 --temperature=0.6 --intelligent_remasking=True
```

## Understanding the Output

The script provides:
1. **Generated text**: The final samples
2. **Self-confidence scores**: Model's confidence in its predictions
3. **Statistics**: Character distributions and patterns
4. **Progress tracking**: Shows masking ratios during generation

## Comparison with sample2.py

### What's Implemented:
- ✅ Iterative demasking algorithm
- ✅ Multiple sampling schedules  
- ✅ Intelligent remasking using model confidence
- ✅ Self-confidence scoring
- ✅ Nucleus sampling and repetition penalties
- ✅ Debug and verbose logging modes
- ✅ Both causal and bidirectional attention support

### Adaptations for Current Project:
- 🔄 Compatible with your checkpoint format
- 🔄 Uses your existing data/vocabulary structure  
- 🔄 Integrates with your model architecture
- 🔄 Supports both compiled and non-compiled models
- 🔄 Works with your configurator.py system

### Key Differences:
- Uses `mask_token_id = vocab_size` (extends vocabulary by 1)
- Simplified model loading to match your checkpoint format
- Integrated both diffusion and standard sampling modes
- Added progress tracking and cleaner output formatting

## Troubleshooting

### Common Issues:

1. **"Checkpoint not found"**
   - Check that the checkpoint exists in the `out/` directory
   - Verify the `checkpoint_name` parameter

2. **"Vocabulary file not found"**
   - Ensure the dataset has a `meta.pkl` file
   - Check the dataset name matches your training data

3. **Memory issues**
   - Reduce `sequence_length` or `num_samples`
   - Use `dtype='float16'` instead of `bfloat16`

4. **Poor generation quality**
   - Increase `diffusion_iterations`
   - Lower `temperature` for more deterministic output
   - Enable `intelligent_remasking=True`

## Examples

### Shakespeare Generation
```bash
python sample.py \
  --checkpoint_name=optimal2_3400.pt \
  --sampling_method=diffusion \
  --sequence_length=256 \
  --diffusion_iterations=25 \
  --temperature=0.7 \
  --num_samples=3
```

### Creative Writing
```bash  
python sample.py \
  --sampling_method=diffusion \
  --sequence_length=512 \
  --temperature=1.0 \
  --top_p=0.9 \
  --intelligent_remasking=True \
  --diffusion_iterations=50
```

The implementation provides a powerful and flexible text generation system that combines the sophistication of diffusion-based approaches with the reliability of your existing training infrastructure.