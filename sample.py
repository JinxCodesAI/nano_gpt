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

import torch
from model import GPTConfig, GPT
from sample_utils import (
    linear_remasking_schedule, nucleus_sample, apply_remasking, 
    calculate_selfconfidence_ratio, predict_and_sample_tokens, apply_remasking_step
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Model loading
init_from = 'resume'  # 'resume' to load from checkpoint
out_dir = 'out'
checkpoint_name = '7250_1.76_all_LMod_enabled.pt'  # Main model checkpoint
remasking_checkpoint_name = None  # Optional: remasking model checkpoint

# Generation parameters
num_samples = 4  # Number of samples to generate
sequence_length = 512  # Total length of generated sequence
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
diffusion_iterations = 50  # Number of demasking iterations
start_ratio = 0.95  # Initial ratio of tokens to remask (95%)
end_ratio = 0.05   # Final ratio of tokens to remask (5%)

# Remasking parameters
randomness_strength = 0.4  # Balance between random (1.0) and model-guided (0.0) remasking
intelligent_remasking = True  # Enable self-confidence based remasking when no remasking model

# Schedule parameters
schedule_type = 'linear'  # 'linear' or 'custom'
masking_ratios = None  # For custom schedule: list of ratios for each iteration

# Logging parameters
use_verbose_logging = True  # Print detailed progress
log_debug = False  # Enable detailed debug logging
show_progress = True  # Show basic progress information

# Standard sampling parameters (only used if sampling_method='standard')
start_text = ""  # Starting text for standard sampling
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
        tokens, prediction_tokens = predict_and_sample_tokens(
            model=model,
            tokens=tokens,
            mask_token_id=mask_token_id,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            vocab_size=vocab_size,
            device=device,
            verbose=verbose and use_verbose_logging
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
                verbose=verbose and use_verbose_logging
            )
    
    return tokens

def standard_generate(model, start_ids, max_new_tokens, temperature=1.0, top_k=None):
    """
    Standard autoregressive generation
    """
    model.eval()
    
    if len(start_ids) == 0:
        # Start with a random token if no start text
        start_ids = [torch.randint(0, vocab_size, (1,)).item()]
    
    for batch_idx in range(num_samples):
        if batch_idx == 0:
            idx = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
        else:
            # Generate different samples by using different starting tokens
            idx = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            if len(start_ids) == 0:
                idx = torch.randint(0, vocab_size, (1, 1), device=device)
        
        generated = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
        
        if batch_idx == 0:
            all_generated = generated
        else:
            all_generated = torch.cat([all_generated, generated], dim=0)
    
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

# Load optional remasking model
remasking_model = None
if remasking_checkpoint_name is not None:
    remasking_checkpoint_path = os.path.join(out_dir, remasking_checkpoint_name)
    if os.path.exists(remasking_checkpoint_path):
        try:
            remasking_model, _ = load_model_from_checkpoint(remasking_checkpoint_path, device, compile)
            print("Remasking model loaded successfully")
        except Exception as e:
            print(f"Failed to load remasking model: {e}")
            remasking_model = None
    else:
        print(f"Remasking checkpoint not found: {remasking_checkpoint_path}")

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
            
            # Calculate self-confidence scores
            print("\nCalculating self-confidence scores...")
            confidence_scores = calculate_selfconfidence_ratio(
                model=model,
                tokens=generated_tokens,
                mask_token_id=mask_token_id,
                device=device,
                ctx=ctx
            )
            
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
            confidence_scores = [0.0] * num_samples  # Not calculated for standard sampling

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
        confidence = confidence_scores[i]
        raw_prob = math.exp(confidence) if confidence > -100 else 0.0
        print(f"Self-confidence: {confidence:.4f} (prob: {raw_prob:.6f})")
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

print(f"\n{'='*60}")
print(f"GENERATION COMPLETE")
print(f"{'='*60}")