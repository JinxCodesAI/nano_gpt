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
import torch.nn.functional as F
from model import GPTConfig, GPT, ModelMode
from sample_utils import (
    linear_remasking_schedule, nucleus_sample, apply_remasking,
    calculate_selfconfidence_ratio, predict_and_sample_tokens, apply_remasking_step,
    calculate_judge_scores,
)

from core.common.timings import TimingAccumulator, print_global_summary
from timings_singleton import set_global_timer, get_global_timer


# Global timestamp to mark the beginning of generation (aligned with the
# "Starting diffusion generation" log inside diffusion_generate)
generation_start_wall_time = None

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Model loading
init_from = 'resume'  # 'resume' to load from checkpoint
out_dir = 'out-char-diffusion'
checkpoint_name = '1.69_MLM_8500.pt'  # Main model checkpoint

# Generation parameters
num_samples = 16  # Number of samples to generate
sequence_length = 1024  # Total length of generated sequence
max_new_tokens = 100  # For regular sampling (non-diffusion)
seed = -1
device = 'cuda'
dtype = 'float16'  # Use float16 for RTX 2060 compatibility
compile = False  # Use PyTorch 2.0 compilation (disabled due to triton issues)

# Sampling method
sampling_method = 'diffusion'  # 'diffusion' or 'multinomial' 

seed_text = "\nWell, sir; what did this gentleman to her?\n"

class SeedPlacement(Enum):
    PREFIX = 'prefix'
    RANDOM_PLACEMENT = 'random'

# Seed placement mode
seed_placement = SeedPlacement.RANDOM_PLACEMENT


# Sampling parameters (used by both diffusion and multinomial)
temperature = 0.8  # Temperature for sampling
iterations = 15  # Number of iterations (demasking for diffusion, resampling for multinomial)

# Diffusion-specific parameters (only used if sampling_method='diffusion')
top_p = 1.0  # Nucleus sampling parameter (1.0 = disabled)
start_ratio = 0.95  # Initial ratio of tokens to remask (95%)
end_ratio = 0.05   # Final ratio of tokens to remask (5%)

# Remasking parameters (diffusion only)
randomness_strength = 0.2  # Balance between random (1.0) and model-guided (0.0) remasking
intelligent_remasking = True  # Enable self-confidence based remasking when no remasking model

# Quality metric configuration
class QualityMetric(Enum):
    NONE = 'None'
    SELF = 'Self'
    JUDGE = 'Judge'

quality_metric = QualityMetric.JUDGE
# Judge (sequence scorer) checkpoint name (relative to out_dir); required if quality_metric == QualityMetric.JUDGE
judge_checkpoint_name =  'padded_judge_0.0155.pt'# 'scoring_p90_0.0096_epoch_3.pt'

# Schedule parameters (diffusion only)
schedule_type = 'linear'  # 'linear' or 'custom'
masking_ratios = None # For custom schedule: list of ratios for each iteration

# Logging parameters
use_verbose_logging = True  # Print detailed progress
log_debug = False  # Enable detailed debug logging
show_progress = True  # Show basic progress information

# Ensure remasking_model is defined (optional component); defaults to None for inference
remasking_model = None

# -----------------------------------------------------------------------------
exec(open('configurator.py').read()) # overrides from command line or config file

if seed == -1:
    seed = int.from_bytes(os.urandom(4), byteorder='little')

# Validation
if sampling_method not in ['diffusion', 'multinomial']:
    raise ValueError(f"sampling_method must be 'diffusion' or 'multinomial', got '{sampling_method}'")

if schedule_type == 'custom' and masking_ratios is not None:
    iterations = len(masking_ratios)
    print(f"Using custom schedule with {iterations} iterations")

# Set random seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Device setup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# Initialize and register global timing accumulator for this run
_global_timer = TimingAccumulator()
if device_type == 'cuda':
    _global_timer.set_cuda_sync(True)
set_global_timer(_global_timer)

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

    # Inference-time safety: do not chain-load any additional pretrained checkpoint
    # Some judge checkpoints may carry init_from_checkpoint used during training; disable it here
    if 'init_from_checkpoint' in model_args and model_args.get('init_from_checkpoint'):
        # Print once to aid debugging, then clear
        print(f"Ignoring training-only init_from_checkpoint: {model_args['init_from_checkpoint']}")
        model_args['init_from_checkpoint'] = None

    # Filter out deprecated config fields (for backward compatibility with old checkpoints)
    deprecated_fields = {'mode', 'num_token_classes', 'binary_classification', 'attention_type', 'position_encoding'}
    filtered_model_args = {k: v for k, v in model_args.items() if k not in deprecated_fields}

    # Store the old mode if present (we'll set it after model creation)
    old_mode = model_args.get('mode', None)

    # Print model configuration for debugging
    print(f"Model config from checkpoint:")
    print(f"  - vocab_size: {filtered_model_args.get('vocab_size')}")
    print(f"  - block_size: {filtered_model_args.get('block_size')}")
    if old_mode:
        print(f"  - mode (deprecated): {old_mode} - will be set after model creation")

    # Create model
    gptconf = GPTConfig(**filtered_model_args)
    model = GPT(gptconf)

    # Set mode based on old config if present
    if old_mode:
        if old_mode == 'sequence_scorer' or old_mode == ModelMode.SEQUENCE_SCORER:
            model.set_mode(ModelMode.SEQUENCE_SCORER)
            print(f"  - Set model mode to SEQUENCE_SCORER")
        elif old_mode == 'language_model' or old_mode == ModelMode.LANGUAGE_MODEL:
            model.set_mode(ModelMode.LANGUAGE_MODEL)
            print(f"  - Set model mode to LANGUAGE_MODEL")
        # token_classifier is deprecated, default to LANGUAGE_MODEL
        elif old_mode == 'token_classifier':
            model.set_mode(ModelMode.LANGUAGE_MODEL)
            print(f"  - WARNING: token_classifier mode deprecated, using LANGUAGE_MODEL")

    # Load state dict
    state_dict = checkpoint['model']

    # Remove compilation prefix if present
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # Handle old checkpoints with single head (backward compatibility)
    # New models have both lm_head and sequence_head, old models had only one
    has_lm_head = any(k.startswith('lm_head.') for k in state_dict.keys())
    has_sequence_head = any(k.startswith('sequence_head.') for k in state_dict.keys())

    if not has_lm_head and has_sequence_head:
        # Old SEQUENCE_SCORER checkpoint - initialize lm_head from wte (weight tying)
        print("  - Old SEQUENCE_SCORER checkpoint detected, initializing missing lm_head")
        if 'transformer.wte.weight' in state_dict:
            state_dict['lm_head.weight'] = state_dict['transformer.wte.weight'].clone()
    elif has_lm_head and not has_sequence_head:
        # Old LANGUAGE_MODEL checkpoint - initialize sequence_head with small random weights
        print("  - Old LANGUAGE_MODEL checkpoint detected, initializing missing sequence_head")
        # Get embedding dimension from existing weights
        n_embd = state_dict['transformer.wte.weight'].shape[1]
        # Initialize sequence_head components with small random weights
        state_dict['sequence_head.base_predictor.weight'] = torch.randn(1, n_embd) * 0.01
        state_dict['sequence_head.base_predictor.bias'] = torch.zeros(1)
        state_dict['sequence_head.log_temperature'] = torch.zeros(1)

    model.load_state_dict(state_dict, strict=False)  # Use strict=False to allow missing keys in edge cases
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

    return stoi, itos, vocab_size, dataset_name, meta

# -----------------------------------------------------------------------------
# Diffusion generation functions
# -----------------------------------------------------------------------------

def apply_seed_text(tokens, seed_ids, placement, batch_size, total_length, device, verbose=False):
    """
    Apply seed text to tokens and create protected mask

    Args:
        tokens: Token tensor to modify (batch_size, total_length)
        seed_ids: List of seed token IDs
        placement: SeedPlacement enum (PREFIX or RANDOM_PLACEMENT)
        batch_size: Batch size
        total_length: Sequence length
        device: Device to use
        verbose: Enable verbose logging

    Returns:
        tuple: (modified_tokens, protected_mask)
    """
    protected_mask = torch.zeros((batch_size, total_length), dtype=torch.bool, device=device)

    if seed_ids is not None and len(seed_ids) > 0:
        seed_len = min(len(seed_ids), total_length)
        # Determine placement start index
        if placement == SeedPlacement.RANDOM_PLACEMENT:
            max_start = total_length - seed_len
            if max_start >= 0:
                start_idx = int(torch.randint(0, max_start + 1, (1,), device=device).item())
            else:
                start_idx = 0
                seed_len = total_length
        else:
            start_idx = 0
        seed_tensor = torch.tensor(seed_ids[:seed_len], dtype=torch.long, device=device)
        tokens[:, start_idx:start_idx+seed_len] = seed_tensor.unsqueeze(0)
        protected_mask[:, start_idx:start_idx+seed_len] = True
        if verbose:
            print(f"DEBUG: Applied seed length: {seed_len} at index: {start_idx}")

    return tokens, protected_mask

def log_iteration_progress(iteration, total_iterations, tokens, mask_token_id, decode_fn,
                          previous_tokens=None, mode='diffusion'):
    """Log progress during generation iterations

    Args:
        iteration: Current iteration number
        total_iterations: Total number of iterations
        tokens: Current token tensor
        mask_token_id: ID of mask token
        decode_fn: Function to decode tokens to text
        previous_tokens: Previous iteration tokens (for multinomial mode to track changes)
        mode: 'diffusion' or 'multinomial' - determines what to log
    """
    if not show_progress:
        return

    # Guard against incorrect types to fail fast if tokens is not a tensor
    if not torch.is_tensor(tokens):
        raise TypeError(f"tokens must be a torch.Tensor, got {type(tokens)}")

    if mode == 'diffusion':
        masked_count = (tokens == mask_token_id).sum().item()
        total_tokens = tokens.numel()
        masked_ratio = masked_count / total_tokens
        print(f"Iteration {iteration+1}/{total_iterations}: {masked_count}/{total_tokens} tokens masked ({masked_ratio:.1%})")
    elif mode == 'multinomial':
        if previous_tokens is not None:
            # Calculate how many tokens changed from previous iteration
            changed_count = (tokens != previous_tokens).sum().item()
            total_tokens = tokens.numel()
            changed_ratio = changed_count / total_tokens
            print(f"Iteration {iteration+1}/{total_iterations}: {changed_count}/{total_tokens} tokens changed ({changed_ratio:.1%})")
        else:
            # First iteration - all tokens are being generated
            total_tokens = tokens.numel()
            print(f"Iteration {iteration+1}/{total_iterations}: {total_tokens}/{total_tokens} tokens initialized (100.0%)")

    if iteration == 0 or iteration == total_iterations - 1:
        # Show sample for first and last iterations
        sample_text = decode_fn(tokens[0])
        preview = sample_text[:100] + ('...' if len(sample_text) > 100 else '')
        print(f"  Sample: {preview}")

def diffusion_generate(model, batch_size, total_length, iterations, mask_token_id, vocab_size,
                      decode_fn, remasking_model=None, verbose=False, seed_ids=None, placement=SeedPlacement.PREFIX):
    """
    Generate text using diffusion-based iterative demasking

    Returns:
        Generated tokens (batch_size, total_length)
    """
    # Start with all positions masked
    tokens = torch.full((batch_size, total_length), mask_token_id, dtype=torch.long, device=device)

    # Apply seed text (never to be masked) if provided
    tokens, protected_mask = apply_seed_text(
        tokens=tokens,
        seed_ids=seed_ids,
        placement=placement,
        batch_size=batch_size,
        total_length=total_length,
        device=device,
        verbose=verbose
    )

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
        elif getattr(getattr(model, 'config', object()), 'add_critic_head', False):
            print(f"  - Using critic-guided remasking")
        elif intelligent_remasking:
            print(f"  - Using intelligent self-remasking")
        else:
            print(f"  - Using random remasking")
        print("=" * 60)

    for iteration in range(iterations):
        if verbose or show_progress:
            log_iteration_progress(iteration, iterations, tokens, mask_token_id, decode_fn)

        # Step 1: Predict tokens for masked positions
        _t = get_global_timer()
        _cm = _t.measure('predict_and_sample') if _t is not None else nullcontext()
        with _cm:
            prediction_tokens, logits = predict_and_sample_tokens(
                model=model,
                tokens=tokens,
                mask_token_id=mask_token_id,
                temperature=temperature,
                top_p=top_p,
                vocab_size=vocab_size,
                device=device,
                verbose=verbose and use_verbose_logging,
                return_logits=True,
                pad_token_id=pad_token_id,
                base_vocab_size=base_vocab_size
            )

        tokens = prediction_tokens

        # Step 2: Remask for next iteration (except last iteration)
        if iteration < iterations - 1:
            _t = get_global_timer()
            _cm = _t.measure('remask') if _t is not None else nullcontext()
            with _cm:
                _remask_result = apply_remasking_step(
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
                    intelligent_remasking=(False if (remasking_model is None and getattr(getattr(model, 'config', object()), 'add_critic_head', False)) else intelligent_remasking),
                    verbose=verbose and use_verbose_logging,
                    logits_from_predict=logits,
                    protected_mask=protected_mask
                )
            # apply_remasking_step returns (remasked_tokens, min_wrongness, remasked_indices)
            if isinstance(_remask_result, tuple):
                remasked_tokens = _remask_result[0]
                # Early termination signal in threshold mode (nothing to mask)
                if remasked_tokens is None:
                    break
                tokens = remasked_tokens
            else:
                # Backward-compat path
                tokens = _remask_result

    return tokens

def multinomial_generate(model, batch_size, total_length, iterations, mask_token_id, vocab_size,
                        decode_fn, temperature=0.8, verbose=False, seed_ids=None,
                        placement=SeedPlacement.PREFIX, pad_token_id=None, base_vocab_size=None):
    """
    Generate text using multinomial sampling with iterative full-sequence resampling

    Unlike diffusion which only updates masked positions and applies remasking,
    multinomial sampling:
    1. Starts with fully masked sentence
    2. Runs model and applies softmax
    3. Samples from softmax and replaces ALL tokens (not just masked ones)
    4. Repeats for specified iterations without any remasking step

    Args:
        model: The language model
        batch_size: Number of samples to generate
        total_length: Length of sequences to generate
        iterations: Number of resampling iterations
        mask_token_id: ID of mask token
        vocab_size: Full vocabulary size
        decode_fn: Function to decode tokens to text
        temperature: Sampling temperature
        verbose: Enable verbose logging
        seed_ids: Optional seed text token IDs (protected from changes)
        placement: Seed placement mode
        pad_token_id: ID of pad token (excluded from sampling)
        base_vocab_size: Base vocabulary size (exclude special tokens)

    Returns:
        Generated tokens (batch_size, total_length)
    """
    # Start with all positions masked
    tokens = torch.full((batch_size, total_length), mask_token_id, dtype=torch.long, device=device)

    # Apply seed text (never to be changed) if provided
    tokens, protected_mask = apply_seed_text(
        tokens=tokens,
        seed_ids=seed_ids,
        placement=placement,
        batch_size=batch_size,
        total_length=total_length,
        device=device,
        verbose=verbose
    )

    # Mark the wall-clock start of generation
    global generation_start_wall_time
    generation_start_wall_time = time.time()

    if verbose or show_progress:
        print(f"Starting multinomial generation:")
        print(f"  - Samples: {batch_size}")
        print(f"  - Length: {total_length}")
        print(f"  - Iterations: {iterations}")
        print(f"  - Temperature: {temperature}")
        print("=" * 60)

    previous_tokens = None

    for iteration in range(iterations):
        if verbose or show_progress:
            log_iteration_progress(iteration, iterations, tokens, mask_token_id, decode_fn,
                                 previous_tokens=previous_tokens, mode='multinomial')

        # Store previous tokens for change tracking
        previous_tokens = tokens.clone()

        # Step 1: Run model forward pass
        _t = get_global_timer()
        _cm = _t.measure('predict_and_sample') if _t is not None else nullcontext()
        with _cm:
            # Get logits for all positions
            dummy_targets = torch.zeros_like(tokens)
            with torch.no_grad():
                logits, _ = model(tokens, targets=dummy_targets)

            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Exclude special tokens from sampling
            logits[:, :, mask_token_id] = float('-inf')
            if pad_token_id is not None:
                logits[:, :, pad_token_id] = float('-inf')
            if base_vocab_size is not None and logits.shape[-1] > base_vocab_size:
                logits[:, :, base_vocab_size:] = float('-inf')
            if vocab_size is not None and logits.shape[-1] > vocab_size:
                logits = logits[:, :, :vocab_size]

            # Sample from softmax for all positions using torch.multinomial
            probs = F.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)

            # Reshape for multinomial sampling
            batch_size_actual, seq_len, vocab_size_actual = probs.shape
            probs_flat = probs.view(-1, vocab_size_actual)  # (batch_size * seq_len, vocab_size)
            sampled_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)  # (batch_size * seq_len,)
            sampled_tokens = sampled_flat.view(batch_size_actual, seq_len)  # (batch_size, seq_len)

            # Update tokens, but preserve protected (seed) positions
            new_tokens = tokens.clone()
            new_tokens[~protected_mask] = sampled_tokens[~protected_mask]

        tokens = new_tokens

    return tokens

# Load main model
main_checkpoint_path = os.path.join(out_dir, checkpoint_name)
model, checkpoint = load_model_from_checkpoint(main_checkpoint_path, device, compile)

# Load vocabulary
stoi, itos, meta_vocab_size, dataset_name, meta = load_vocabulary(checkpoint)

# Get model's actual vocabulary size from checkpoint
model_vocab_size = model.config.vocab_size
print(f"Model vocab_size from checkpoint: {model_vocab_size}")
print(f"Meta vocab_size from data: {meta_vocab_size}")

# Use the vocabulary size from the model checkpoint
vocab_size = model_vocab_size

# Load special token IDs from metadata
mask_token_id = meta.get('mask_token_id', None)
pad_token_id = meta.get('pad_token_id', None)
base_vocab_size = meta.get('base_vocab_size', None)

# Fallback to old logic if metadata doesn't have new token info
if mask_token_id is None:
    # Old approach: mask_token_id = vocab_size - 1
    mask_token_id = vocab_size - 1
    print(f"Warning: Using fallback mask_token_id calculation: {mask_token_id}")
else:
    print(f"Loaded mask_token_id from metadata: {mask_token_id}")

if pad_token_id is not None:
    print(f"Loaded pad_token_id from metadata: {pad_token_id}")

if base_vocab_size is not None:
    print(f"Loaded base_vocab_size from metadata: {base_vocab_size}")

print(f"Using mask_token_id: {mask_token_id} ('{itos[mask_token_id] if mask_token_id < len(itos) else '[MASK]'}')")
if pad_token_id is not None and pad_token_id < len(itos):
    print(f"Using pad_token_id: {pad_token_id} ('{itos[pad_token_id]}')")
print(f"Total vocab_size: {vocab_size}")
print(f"Base vocab_size (content tokens): {base_vocab_size if base_vocab_size else vocab_size - 1}")
# Tokenize seed text to insert and protect from remasking
seed_ids = [stoi[c] for c in seed_text if c in stoi] if seed_text else []
if use_verbose_logging and seed_ids:
    print(f"Seed text configured: length {len(seed_ids)} | placement: {seed_placement.name}")


# Create decode functions
def decode(token_ids):
    """Decode tokens to text, handling mask and pad tokens"""
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()

    result = []
    for token_id in token_ids:
        if token_id == mask_token_id:
            result.append('[MASK]')
        elif pad_token_id is not None and token_id == pad_token_id:
            result.append('[PAD]')
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
        elif pad_token_id is not None and token_id == pad_token_id:
            result.append('[PAD]')
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
    # Set judge model to SEQUENCE_SCORER mode
    judge_model.set_mode(ModelMode.SEQUENCE_SCORER)

# Print generation settings
print(f"\n{'='*60}")
print(f"GENERATION SETTINGS")
print(f"{'='*60}")
print(f"Method: {sampling_method}")
print(f"Samples: {num_samples}")
print(f"Device: {device} ({dtype})")
print(f"Seed: {seed}")

print(f"Sequence length: {sequence_length}")
print(f"Iterations: {iterations}")
print(f"Temperature: {temperature}")

if sampling_method == 'diffusion':
    print(f"Schedule: {schedule_type} ({start_ratio:.1%} → {end_ratio:.1%})")
    print(f"Top-p: {top_p}")
    # Informational: Critic-guided remasking will be used if available (no external remasking model)
    if remasking_model is None and getattr(getattr(model, 'config', object()), 'add_critic_head', False):
        print("Critic-guided remasking: enabled (using model's critic head)")

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
                iterations=iterations,
                mask_token_id=mask_token_id,
                vocab_size=vocab_size,  # Full vocab size (includes mask token)
                decode_fn=decode,
                remasking_model=remasking_model,
                verbose=use_verbose_logging,
                seed_ids=seed_ids,
                placement=seed_placement
            )
        elif sampling_method == 'multinomial':
            # Multinomial generation
            generated_tokens = multinomial_generate(
                model=model,
                batch_size=num_samples,
                total_length=sequence_length,
                iterations=iterations,
                mask_token_id=mask_token_id,
                vocab_size=vocab_size,
                decode_fn=decode,
                temperature=temperature,
                verbose=use_verbose_logging,
                seed_ids=seed_ids,
                placement=seed_placement,
                pad_token_id=pad_token_id,
                base_vocab_size=base_vocab_size
            )
        _end_wall = time.time()

        # Calculate quality metric (for both diffusion and multinomial)
        metric = quality_metric
        confidence_scores = None
        judge_scores = None
        if metric == QualityMetric.SELF:
            print("\nCalculating self-confidence scores...")
            _t = get_global_timer()
            _cm = _t.measure('self_conf_eval') if _t is not None else nullcontext()
            with _cm:
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
            _t = get_global_timer()
            _cm = _t.measure('judge_eval') if _t is not None else nullcontext()
            with _cm:
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


generation_time = time.time() - start_time

# Compute critic percentiles per sample if critic head is available
critic_percentiles = None
if sampling_method == 'diffusion' and getattr(getattr(model, 'config', object()), 'add_critic_head', False):
    with torch.no_grad():
        if ctx is not None:
            with ctx:
                _critic_scores = model.critic_scores(generated_tokens)
        else:
            _critic_scores = model.critic_scores(generated_tokens)
    # Exclude PAD and MASK tokens from percentile computation
    _valid = (generated_tokens != mask_token_id)
    if pad_token_id is not None:
        _valid = _valid & (generated_tokens != pad_token_id)
    _q = torch.tensor([0.1, 0.5, 0.9], device=generated_tokens.device, dtype=torch.float32)
    _per_list = []
    for b in range(generated_tokens.size(0)):
        _s = _critic_scores[b][_valid[b]]
        if _s.numel() == 0:
            _per_list.append(torch.tensor([float('nan'), float('nan'), float('nan')], device='cpu'))
        else:
            # Convert logits to probabilities before percentile computation
            _prob = torch.sigmoid(_s.float())
            _per_list.append(torch.quantile(_prob, _q).cpu())
    critic_percentiles = torch.stack(_per_list, dim=0)

# Display results
print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Generation completed in {generation_time:.2f} seconds")
print(f"{'='*60}")

for i in range(num_samples):
    print(f"\n{'─'*40}")
    print(f"SAMPLE {i+1}/{num_samples}")

    # Display quality metrics (for both diffusion and multinomial)
    metric = quality_metric
    if metric == QualityMetric.SELF and 'confidence_scores' in locals() and confidence_scores is not None:
        confidence = confidence_scores[i]
        raw_prob = math.exp(confidence) if confidence > -100 else 0.0
        print(f"Self-confidence: {confidence:.4f} (prob: {raw_prob:.6f})")
    elif metric == QualityMetric.JUDGE and 'judge_scores' in locals() and judge_scores is not None:
        score = float(judge_scores[i].item()) if hasattr(judge_scores, 'shape') else float(judge_scores[i])
        print(f"Quality score (Judge): {score:.4f}")

    # Critic token score percentiles (if computed, diffusion only)
    if sampling_method == 'diffusion' and 'critic_percentiles' in locals() and critic_percentiles is not None:
        p10, p50, p90 = [float(x) for x in critic_percentiles[i].tolist()]
        print(f"Critic probabilities p10/median/p90: {p10:.4f} / {p50:.4f} / {p90:.4f}")

    print(f"{'─'*40}")

    sample_text = decode(generated_tokens[i])
    print(sample_text)

    # Show some statistics (for both methods)
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
_tokens_per_sample = sequence_length  # Both diffusion and multinomial use sequence_length
_total_tokens = num_samples * _tokens_per_sample
_time_per_sample = _full_time / max(1, num_samples)
_tokens_per_sec = (_total_tokens / _full_time) if _full_time > 0 else float('inf')

# Model-only throughput based on timer window (first measure start → last measure end)
try:
    from core.common.timings import get_global_timer as _get_t
    _t = _get_t()
    _timer_total = _t.get_total_time() if _t is not None else _full_time
except Exception:
    _timer_total = _full_time
_model_tokens_per_sec = (_total_tokens / _timer_total) if _timer_total > 0 else float('inf')
from core.common.timings import print_global_hierarchical_summary as _print_h
_print_h(title="OPERATION TIMING (hierarchical)", show_counts=True)
# Defer throughput print to the bottom near judge info
_throughput_line = f"Throughput: {_model_tokens_per_sec:.2f} tokens/s (tokens: {_total_tokens})"

if judge_time is not None:
    _judge_time_per_sample = judge_time / max(1, num_samples)
    print(f"Judge time: {judge_time:.2f} s | Time per sample: {_judge_time_per_sample:.2f} s")
    print(f"Average judge score: {judge_scores.mean().item():.4f}")
    print(f"Best judge score: {judge_scores.max().item():.4f}")
    print(f"Worst judge score: {judge_scores.min().item():.4f}")
    if judge_tokens_total is not None and judge_tokens_per_sec is not None:
        print(f"Judge throughput: {judge_tokens_per_sec:.2f} tokens/s (judge tokens: {judge_tokens_total})")

# Print model-only throughput at the bottom, near judge info
print(_throughput_line)

print(f"\n{'='*60}")
print(f"GENERATION COMPLETE")
print(f"{'='*60}")