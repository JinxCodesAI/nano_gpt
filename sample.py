"""
Diffusion-based text generation using iterative demasking
Compatible with the current training setup and checkpoint format
"""
import os
import time
import math
import pickle
from contextlib import nullcontext
from collections import Counter

from enum import Enum

import torch
from model import GPTConfig, GPT, ModelMode
from sample_utils import (
    linear_remasking_schedule, nucleus_sample, apply_remasking,
    calculate_selfconfidence_ratio, predict_and_sample_tokens, apply_remasking_step,
    calculate_judge_scores,
)


# Global timestamp to mark the beginning of generation (aligned with the
# "Starting diffusion generation" log inside diffusion_generate)
generation_start_wall_time = None

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Model loading
init_from = 'resume'  # 'resume' to load from checkpoint
out_dir = 'out-char-diffusion'
checkpoint_name = '8500_1.81_all_LMod_enabled(epoch 2).pt'  # Main model checkpoint

# Generation parameters
num_samples = 16  # Number of samples to generate
sequence_length = 1024  # Total length of generated sequence
max_new_tokens = 100  # For regular sampling (non-diffusion)
seed = -1
device = 'cuda'
dtype = 'float16'  # Use float16 for RTX 2060 compatibility
compile = False  # Use PyTorch 2.0 compilation (disabled due to triton issues)

# Sampling method
sampling_method = 'diffusion'  # 'diffusion' or 'standard'

# Diffusion parameters (only used if sampling_method='diffusion')
temperature = 0.8  # Temperature for sampling
top_p = 1.0  # Nucleus sampling parameter (1.0 = disabled)
repetition_penalty = 1.0  # Penalty for repeating recent tokens
repetition_window = 10  # Look back window for repetition penalty
diffusion_iterations = 30  # Number of demasking iterations
start_ratio = 0.95  # Initial ratio of tokens to remask (95%)
end_ratio = 0.05   # Final ratio of tokens to remask (5%)

# Remasking parameters
randomness_strength = 0.4  # Balance between random (1.0) and model-guided (0.0) remasking
intelligent_remasking = True  # Enable self-confidence based remasking when no remasking model

# Quality metric configuration
class QualityMetric(Enum):
    NONE = 'None'
    SELF = 'Self'
    JUDGE = 'Judge'

quality_metric = QualityMetric.JUDGE
# Judge (sequence scorer) checkpoint name (relative to out_dir); required if quality_metric == QualityMetric.JUDGE
judge_checkpoint_name = 'scoring_p90_0.0096_epoch_3.pt'

# Schedule parameters
schedule_type = 'linear'  # 'linear' or 'custom'
masking_ratios = None  # For custom schedule: list of ratios for each iteration

# Logging parameters
use_verbose_logging = True  # Print detailed progress
log_debug = False  # Enable detailed debug logging
show_progress = True  # Show basic progress information

# Standard sampling parameters (only used if sampling_method='standard')
start_text = ""  # Starting text for standard sampling
# Ensure remasking_model is defined (optional component); defaults to None for inference
remasking_model = None

std_temperature = 0.8
top_k = 200  # Top-k sampling

# -----------------------------------------------------------------------------
exec(open('configurator.py').read()) # overrides from command line or config file

if seed == -1:
    seed = int.from_bytes(os.urandom(4), byteorder='little')

# Validation
if sampling_method not in ['diffusion', 'standard']:
    raise ValueError(f"sampling_method must be 'diffusion' or 'standard', got '{sampling_method}'")

if schedule_type == 'custom' and masking_ratios is not None:
    diffusion_iterations = len(masking_ratios)
    print(f"Using custom schedule with {diffusion_iterations} iterations")

# Set random seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Device setup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_model_from_checkpoint(checkpoint_path, device, compile_model=False):
    """Load model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model configuration
    if 'model_args' in checkpoint:
        model_args = checkpoint['model_args']
    else:
        # Fallback for older checkpoints
        raise ValueError("Checkpoint missing 'model_args'. Please check checkpoint format.")

    # Ensure backward compatibility
    if 'attention_type' not in model_args:
        model_args['attention_type'] = 'causal'
    if 'position_encoding' not in model_args:
        model_args['position_encoding'] = 'absolute'

    # Inference-time safety: do not chain-load any additional pretrained checkpoint
    # Some judge checkpoints may carry init_from_checkpoint used during training; disable it here
    if 'init_from_checkpoint' in model_args and model_args.get('init_from_checkpoint'):
        # Print once to aid debugging, then clear
        print(f"Ignoring training-only init_from_checkpoint: {model_args['init_from_checkpoint']}")
        model_args['init_from_checkpoint'] = None

    # Print model configuration for debugging
    print(f"Model config from checkpoint:")
    print(f"  - vocab_size: {model_args.get('vocab_size')}")
    print(f"  - block_size: {model_args.get('block_size')}")
    print(f"  - attention_type: {model_args.get('attention_type')}")
    print(f"  - position_encoding: {model_args.get('position_encoding')}")

    # Create model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # Load state dict
    state_dict = checkpoint['model']

    # Remove compilation prefix if present
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    if compile_model:
        print("Compiling model...")
        model = torch.compile(model)

    print(f"Model loaded successfully")
    print(f"  - Parameters: {model.get_num_params()/1e6:.1f}M")

    return model, checkpoint

def load_vocabulary(checkpoint, fallback_dataset='shakespeare_char'):
    """Load vocabulary from checkpoint or fallback dataset"""
    dataset_name = None

    # Try to get dataset name from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            dataset_name = config.get('dataset')
        elif hasattr(config, 'get'):
            dataset_name = config.get('dataset')

    if not dataset_name:
        dataset_name = fallback_dataset
        print(f"Dataset name not found in checkpoint, using fallback: {dataset_name}")

    meta_path = os.path.join('data', dataset_name, 'meta.pkl')

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Vocabulary file not found: {meta_path}")

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    stoi, itos = meta['stoi'], meta['itos']
    vocab_size = meta['vocab_size']

    print(f"Vocabulary loaded from {dataset_name}: {vocab_size} tokens")

    return stoi, itos, vocab_size, dataset_name

# -----------------------------------------------------------------------------
# Diffusion generation functions
# -----------------------------------------------------------------------------

def log_iteration_progress(iteration, total_iterations, tokens, mask_token_id, decode_fn):
    """Log progress during diffusion iterations"""
    if not show_progress:
        return

    masked_count = (tokens == mask_token_id).sum().item()
    total_tokens = tokens.numel()
    masked_ratio = masked_count / total_tokens

    print(f"Iteration {iteration+1}/{total_iterations}: {masked_count}/{total_tokens} tokens masked ({masked_ratio:.1%})")

    if iteration == 0 or iteration == total_iterations - 1:
        # Show sample for first and last iterations
        sample_text = decode_fn(tokens[0])
        preview = sample_text[:100] + ('...' if len(sample_text) > 100 else '')
        print(f"  Sample: {preview}")

def diffusion_generate(model, batch_size, total_length, iterations, mask_token_id, vocab_size,
                      decode_fn, remasking_model=None, verbose=False):
    """
    Generate text using diffusion-based iterative demasking

    Returns:
        Generated tokens (batch_size, total_length)
    """
    # Start with all positions masked
    tokens = torch.full((batch_size, total_length), mask_token_id, dtype=torch.long, device=device)

    # Debug: Check initial token setup
    if verbose:
        print(f"DEBUG: Initial tokens shape: {tokens.shape}")
        print(f"DEBUG: mask_token_id: {mask_token_id}, vocab_size: {vocab_size}")
        print(f"DEBUG: All tokens set to mask_token_id: {torch.all(tokens == mask_token_id).item()}")
        print(f"DEBUG: Token value range: {tokens.min().item()} to {tokens.max().item()}")

    # Mark the wall-clock start of generation aligned with the first progress log
    global generation_start_wall_time
    generation_start_wall_time = time.time()

    if verbose or show_progress:
        print(f"Starting diffusion generation:")
        print(f"  - Samples: {batch_size}")
        print(f"  - Length: {total_length}")
        print(f"  - Iterations: {iterations}")
        print(f"  - Temperature: {temperature}")
        if remasking_model is not None:
            print(f"  - Using remasking model")
        elif intelligent_remasking:
            print(f"  - Using intelligent self-remasking")
        else:
            print(f"  - Using random remasking")
        print("=" * 60)

    for iteration in range(iterations):
        if verbose or show_progress:
            log_iteration_progress(iteration, iterations, tokens, mask_token_id, decode_fn)

        # Step 1: Predict tokens for masked positions
        tokens, prediction_tokens, logits = predict_and_sample_tokens(
            model=model,
            tokens=tokens,
            mask_token_id=mask_token_id,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            vocab_size=vocab_size,
            device=device,
            verbose=verbose and use_verbose_logging,
            return_logits=True
        )

        # Step 2: Remask for next iteration (except last iteration)
        if iteration < iterations - 1:
            tokens = apply_remasking_step(
                tokens=tokens,
                prediction_tokens=prediction_tokens,
                iteration=iteration,
                iterations=iterations,
                schedule_type=schedule_type,
                masking_ratios=masking_ratios,
                start_ratio=start_ratio,
                end_ratio=end_ratio,
                remasking_model=remasking_model,
                randomness_strength=randomness_strength,
                mask_token_id=mask_token_id,
                device=device,
                base_model=model,
                intelligent_remasking=intelligent_remasking,
                verbose=verbose and use_verbose_logging,
                logits_from_predict=logits
            )

    return tokens

def standard_generate(model, start_ids, max_new_tokens, temperature=1.0, top_k=None):
    """
    Standard autoregressive generation (batched across num_samples)
    """
    model.eval()

    # Build initial batch
    if start_ids:
        idx = torch.tensor([start_ids] * num_samples, dtype=torch.long, device=device)
    else:
        # If no start text, randomize the first token per sample
        idx = torch.randint(0, vocab_size, (num_samples, 1), device=device)

    # Single batched generate call
    all_generated = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return all_generated

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

# Load main model
main_checkpoint_path = os.path.join(out_dir, checkpoint_name)
model, checkpoint = load_model_from_checkpoint(main_checkpoint_path, device, compile)

# Load vocabulary
stoi, itos, meta_vocab_size, dataset_name = load_vocabulary(checkpoint)

# Get model's actual vocabulary size from checkpoint
model_vocab_size = model.config.vocab_size
print(f"Model vocab_size from checkpoint: {model_vocab_size}")
print(f"Meta vocab_size from data: {meta_vocab_size}")

# Use the vocabulary size from the model checkpoint
vocab_size = model_vocab_size

# According to char_diffusion prepare_streaming.py:
# mask_token_id = len(chars) and vocab_size = len(chars) + 1
# So mask_token_id = vocab_size - 1 (last token in vocabulary)
mask_token_id = vocab_size - 1
print(f"Using mask_token_id: {mask_token_id} ('{itos[mask_token_id] if mask_token_id < len(itos) else '[MASK]'}')")
print(f"Actual vocabulary size for sampling: {vocab_size - 1} (excluding mask token)")

# Create decode functions
def decode(token_ids):
    """Decode tokens to text"""
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()

    result = []
    for token_id in token_ids:
        if token_id == mask_token_id:
            result.append('[MASK]')
        elif token_id < len(itos):
            result.append(itos[token_id])
        else:
            result.append('[UNK]')
    return ''.join(result)

def decode_with_mask_char(token_ids, mask_char='#'):
    """Decode tokens to text with custom mask character"""
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()

    result = []
    for token_id in token_ids:
        if token_id == mask_token_id:
            result.append(mask_char)
        elif token_id < len(itos):
            result.append(itos[token_id])
        else:
            result.append('[UNK]')
    return ''.join(result)

# Load judge (sequence scorer) model if requested
judge_model = None
if quality_metric == QualityMetric.JUDGE:
    if judge_checkpoint_name is None:
        raise ValueError("quality_metric is 'Judge' but judge_checkpoint_name is not provided")
    judge_checkpoint_path = os.path.join(out_dir, judge_checkpoint_name)
    if not os.path.exists(judge_checkpoint_path):
        raise FileNotFoundError(f"Judge checkpoint not found: {judge_checkpoint_path}")
    judge_model, _ = load_model_from_checkpoint(judge_checkpoint_path, device, compile_model=False)
    if getattr(judge_model.config, 'mode', None) != ModelMode.SEQUENCE_SCORER:
        raise ValueError("Judge model must be configured with mode=SEQUENCE_SCORER")

# Print generation settings
print(f"\n{'='*60}")
print(f"GENERATION SETTINGS")
print(f"{'='*60}")
print(f"Method: {sampling_method}")
print(f"Samples: {num_samples}")
print(f"Device: {device} ({dtype})")
print(f"Seed: {seed}")

if sampling_method == 'diffusion':
    print(f"Sequence length: {sequence_length}")
    print(f"Iterations: {diffusion_iterations}")
    print(f"Schedule: {schedule_type} ({start_ratio:.1%} → {end_ratio:.1%})")
    print(f"Temperature: {temperature}, Top-p: {top_p}")
    print(f"Repetition penalty: {repetition_penalty} (window: {repetition_window})")
else:
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {std_temperature}, Top-k: {top_k}")
    print(f"Start text: '{start_text}'")

print(f"{'='*60}")

# Generate samples
start_time = time.time()
# Metrics for judge evaluation (filled when metric == QualityMetric.JUDGE)
judge_time = None
judge_tokens_total = None
judge_tokens_per_sec = None


with torch.no_grad():
    with ctx:
        if sampling_method == 'diffusion':
            # Diffusion generation
            generated_tokens = diffusion_generate(
                model=model,
                batch_size=num_samples,
                total_length=sequence_length,
                iterations=diffusion_iterations,
                mask_token_id=mask_token_id,
                vocab_size=vocab_size,  # Full vocab size (includes mask token)
                decode_fn=decode,
                remasking_model=remasking_model,
                verbose=use_verbose_logging
            )
            _end_wall = time.time()
            # Calculate quality metric
            metric = quality_metric
            confidence_scores = None
            judge_scores = None
            if metric == QualityMetric.SELF:
                print("\nCalculating self-confidence scores...")
                confidence_scores = calculate_selfconfidence_ratio(
                    model=model,
                    tokens=generated_tokens,
                    mask_token_id=mask_token_id,
                    device=device,
                    ctx=ctx
                )
            elif metric == QualityMetric.JUDGE:
                if judge_model is None:
                    raise ValueError("quality_metric 'Judge' requires a valid judge model to be loaded")
                print("\nEvaluating quality with Judge model...")
                _judge_start = time.time()
                judge_scores = calculate_judge_scores(
                    judge_model=judge_model,
                    tokens=generated_tokens,
                    device=device,
                    ctx=ctx
                )
                judge_time = time.time() - _judge_start
                # Compute judge token throughput: batch_size * effective sequence length (with CLS if used)
                judge_seq_len = int(generated_tokens.size(1))
                cls_id = getattr(judge_model.config, 'cls_token_id', None)
                if cls_id is not None:
                    judge_seq_len += 1
                max_t = getattr(judge_model.config, 'block_size', judge_seq_len)
                judge_tokens_per_sample = int(min(judge_seq_len, max_t))
                judge_tokens_total = int(num_samples * judge_tokens_per_sample)
                judge_tokens_per_sec = (judge_tokens_total / judge_time) if judge_time > 0 else float('inf')
            else:
                # No quality metric requested
                pass

        else:
            # Standard generation
            start_ids = [stoi[c] for c in start_text if c in stoi] if start_text else []
            generated_tokens = standard_generate(
                model=model,
                start_ids=start_ids,
                max_new_tokens=max_new_tokens,
                temperature=std_temperature,
                top_k=top_k
            )
            _end_wall = time.time()

generation_time = time.time() - start_time

# Display results
print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Generation completed in {generation_time:.2f} seconds")
print(f"{'='*60}")

for i in range(num_samples):
    print(f"\n{'─'*40}")
    print(f"SAMPLE {i+1}/{num_samples}")
    if sampling_method == 'diffusion':
        metric = quality_metric
        if metric == QualityMetric.SELF and 'confidence_scores' in locals() and confidence_scores is not None:
            confidence = confidence_scores[i]
            raw_prob = math.exp(confidence) if confidence > -100 else 0.0
            print(f"Self-confidence: {confidence:.4f} (prob: {raw_prob:.6f})")
        elif metric == QualityMetric.JUDGE and 'judge_scores' in locals() and judge_scores is not None:
            score = float(judge_scores[i].item()) if hasattr(judge_scores, 'shape') else float(judge_scores[i])
            print(f"Quality score (Judge): {score:.4f}")
    print(f"{'─'*40}")

    sample_text = decode(generated_tokens[i])
    print(sample_text)

    # Show some statistics
    if sampling_method == 'diffusion':
        char_counts = Counter(sample_text)
        total_chars = len(sample_text)

        # Show most common characters
        top_chars = char_counts.most_common(5)
        char_stats = [f"'{c}': {count}" for c, count in top_chars]
        print(f"\nStats: {total_chars} chars, most common: {', '.join(char_stats)}")

# Performance summary based on full wall time from generation start to this point

if generation_start_wall_time is not None:
    _full_time = _end_wall - generation_start_wall_time
else:
    _full_time = _end_wall - start_time
_tokens_per_sample = sequence_length if sampling_method == 'diffusion' else max_new_tokens
_total_tokens = num_samples * _tokens_per_sample
_time_per_sample = _full_time / max(1, num_samples)
_tokens_per_sec = (_total_tokens / _full_time) if _full_time > 0 else float('inf')

print(f"\n{'='*60}")
print("PERFORMANCE SUMMARY")
print(f"Total wall time: {_full_time:.2f} s | Time per sample: {_time_per_sample:.2f} s")
print(f"Throughput: {_tokens_per_sec:.2f} tokens/s (tokens: {_total_tokens})")
if judge_time is not None:
    _judge_time_per_sample = judge_time / max(1, num_samples)
    print(f"Judge time: {judge_time:.2f} s | Time per sample: {_judge_time_per_sample:.2f} s")
    if judge_tokens_total is not None and judge_tokens_per_sec is not None:
        print(f"Judge throughput: {judge_tokens_per_sec:.2f} tokens/s (judge tokens: {judge_tokens_total})")

print(f"\n{'='*60}")
print(f"GENERATION COMPLETE")
print(f"{'='*60}")