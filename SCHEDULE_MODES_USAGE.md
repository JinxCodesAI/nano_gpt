# Schedule Modes Usage Examples

## Quick Start

### Ratio Mode (Default)
```bash
# Use default ratio mode with standard parameters
python sample_simple.py \
    --checkpoint-name your_model.pt \
    --schedule-mode ratio \
    --start-ratio 0.95 \
    --end-ratio 0.05 \
    --iterations 15
```

### Threshold Mode
```bash
# Use threshold mode for adaptive masking
python sample_simple.py \
    --checkpoint-name your_model.pt \
    --schedule-mode threshold \
    --start-ratio 0.95 \
    --end-ratio 0.05 \
    --iterations 15
```

## Detailed Examples

### Example 1: Conservative Ratio Mode
Masks fewer tokens at the start, good for high-quality models:
```bash
python sample_simple.py \
    --checkpoint-name your_model.pt \
    --schedule-mode ratio \
    --start-ratio 0.80 \
    --end-ratio 0.05 \
    --iterations 15 \
    --num-samples 8 \
    --sequence-length 1024
```

Expected output:
```
Iteration 1/15: 13107/16384 masked (80.0%)
Iteration 2/15: 11796/16384 masked (72.0%)
...
Iteration 15/15: 819/16384 masked (5.0%)
```

### Example 2: Aggressive Threshold Mode
Starts with very low threshold (masks almost everything):
```bash
python sample_simple.py \
    --checkpoint-name your_model.pt \
    --schedule-mode threshold \
    --start-ratio 0.98 \
    --end-ratio 0.02 \
    --iterations 20 \
    --verbose
```

Expected behavior:
- Early iterations: masks almost all tokens (wrongness > 0.02)
- Middle iterations: adaptive masking based on quality
- Late iterations: may finish early if quality is good

### Example 3: Custom Ratios/Thresholds
Use custom schedule instead of linear:
```bash
python sample_simple.py \
    --checkpoint-name your_model.pt \
    --schedule-mode ratio \
    --masking-ratios "0.95,0.90,0.85,0.75,0.65,0.50,0.35,0.20,0.10,0.05"
```

### Example 4: Threshold Mode with Judge Evaluation
```bash
python sample_simple.py \
    --checkpoint-name your_model.pt \
    --judge-checkpoint-name judge_model.pt \
    --schedule-mode threshold \
    --start-ratio 0.95 \
    --end-ratio 0.05 \
    --quality-metric judge \
    --num-samples 16
```

## Comparing Both Modes

Run both modes on the same seed to compare:

```bash
# Ratio mode
python sample_simple.py \
    --checkpoint-name your_model.pt \
    --schedule-mode ratio \
    --seed 42 \
    --num-samples 4 \
    --verbose

# Threshold mode
python sample_simple.py \
    --checkpoint-name your_model.pt \
    --schedule-mode threshold \
    --seed 42 \
    --num-samples 4 \
    --verbose
```

## When to Use Each Mode

### Use Ratio Mode When:
- You want **predictable** generation time
- You need **consistent** behavior across runs
- You're doing **batch generation** with time constraints
- Your model quality is **variable** across different inputs

### Use Threshold Mode When:
- You want **adaptive** generation based on quality
- You can tolerate **variable** generation time
- You want to **finish early** when quality is good
- Your model has a **reliable critic head**

## Advanced Usage

### Combining with Other Parameters

```bash
# Threshold mode with seed text and custom placement
python sample_simple.py \
    --checkpoint-name your_model.pt \
    --schedule-mode threshold \
    --seed-text "Once upon a time" \
    --seed-placement prefix \
    --start-ratio 0.95 \
    --end-ratio 0.05

# Ratio mode with temperature and top-p sampling
python sample_simple.py \
    --checkpoint-name your_model.pt \
    --schedule-mode ratio \
    --temperature 0.9 \
    --top-p 0.95 \
    --start-ratio 0.90 \
    --end-ratio 0.10

# Threshold mode with randomness for diversity
python sample_simple.py \
    --checkpoint-name your_model.pt \
    --schedule-mode threshold \
    --randomness-strength 0.2 \
    --start-ratio 0.95 \
    --end-ratio 0.05
```

## Troubleshooting

### Threshold Mode Finishes Too Early
If threshold mode consistently finishes after just a few iterations:
- **Increase** `start-ratio` (e.g., 0.98 or 0.99)
- **Increase** `end-ratio` (e.g., 0.10 or 0.15)
- Check if your critic head is properly trained

### Threshold Mode Never Finishes Early
If threshold mode always completes all iterations:
- **Decrease** `end-ratio` (e.g., 0.02 or 0.01)
- Increase number of `iterations`
- Your model might need more training

### Ratio Mode Masks Too Many/Few Tokens
Adjust `start-ratio` and `end-ratio`:
- For **more aggressive** masking: increase both values
- For **more conservative** masking: decrease both values

## Performance Notes

- **Ratio mode**: Consistent performance, predictable time
- **Threshold mode**: Variable performance, may be faster if finishing early
- Both modes use the same critic evaluation, so quality should be comparable
- Threshold mode may produce slightly better results by adapting to quality

## Testing Your Configuration

Use the test scripts to verify behavior:

```bash
# Test basic functionality
python test_schedule_modes.py

# Simulate full generation
python test_full_schedule.py
```

