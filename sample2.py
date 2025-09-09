"""
Simplified diffusion-based text generation using iterative demasking
Supports only remasking_binary models or no remasking model (random)
"""
import os
import pickle
import math
from collections import Counter
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
from sample_utils import linear_remasking_schedule, nucleus_sample, apply_remasking, calculate_selfconfidence_ratio, predict_and_sample_tokens, apply_remasking_step

# Configuration
init_from = 'resume'
out_dir = 'out'
checkpoint_name = 'big_boy2.pt' #'35.75_58.2_UM.pt' Decent models optimal2_3400.pt, 
remasking_checkpoint_name = None #'ckpt_remasking_binary_600.pt'  # Optional: remasking_binary model checkpoint
num_samples = 8  # Number of samples to generate
sequence_length = 1024  # Total length of generated sequence
seed = -1
device = 'cuda'
dtype = 'float16'
compile = False

# Generation parameters
temperature = 0.8  # Temperature for sampling (1.0 = no change, <1.0 = more deterministic, >1.0 = more random)
top_p = 1   # Nucleus sampling parameter (1.0 = disabled, <1.0 = only sample from top cumulative probability mass)
repetition_penalty = 1  # Penalty for repeating recent tokens (>1.0 = discourage repetition)
repetition_window = 10  # Look back this many tokens for repetition penalty
diffusion_iterations = 100  # Number of demasking iterations
start_ratio = 0.99  # Initial ratio of tokens to remask (99%)
end_ratio = 0.05   # Final ratio of tokens to remask (5%)

# Debug parameters
log_debug = False  # Enable detailed debug logging (character distributions, repetition patterns, etc.)
use_verbose_logging = False  # Print detailed progress and analysis after each iteration
# Schedule parameters
schedule_type = 'custom'  # 'linear' or 'custom' - type of masking schedule to use
#masking_ratios = [0.85,0.75,0.65,0.55,0.45,0.35,0.25,0.15,0.05,0.65,0.60,0.65,0.6,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1]    # Array of masking ratios for 'custom' schedule (overrides diffusion_iterations)
masking_ratios = [0.85,0.816,0.782,0.748,0.714,0.68,0.646,0.612,0.578,0.544,0.51,0.476,0.442,0.408,0.374,0.34,0.306,0.272,0.238,0.204,0.17,0.136,0.102,0.068,0.034]   # Array of masking ratios for 'custom' schedule (overrides diffusion_iterations)

# Remasking parameters (only used if remasking model is available)
randomness_strength = 0.4 # Balance between random (1.0) and model-guided (0.0) remasking

# Intelligent remasking parameters
intelligent_remasking = True  # Enable selfmasking when no remasking model is available

if seed == -1:
    seed = int.from_bytes(os.urandom(4), byteorder='little')

exec(open('configurator.py').read()) # overrides from command line or config file

# Validate schedule parameters
if schedule_type not in ['linear', 'custom']:
    raise ValueError(f"schedule_type must be 'linear' or 'custom', got '{schedule_type}'")

if schedule_type == 'custom':
    if masking_ratios is None or len(masking_ratios) == 0:
        raise ValueError("masking_ratios cannot be None or empty when schedule_type='custom'")
    # Override diffusion_iterations with length of masking_ratios
    diffusion_iterations = len(masking_ratios)
    print(f"Using custom schedule with {diffusion_iterations} iterations from masking_ratios")



def log_generation_start(batch_size, iterations, total_length, remasking_model, randomness_strength, intelligent_remasking=False):
    """Log generation start information"""
    print(f"Starting diffusion generation: {batch_size} samples, {iterations} iterations")
    print(f"Total length: {total_length} (all tokens start masked)")
    if remasking_model is not None:
        print(f"Using remasking_binary model with randomness_strength={randomness_strength}")
    elif intelligent_remasking:
        print(f"Using intelligent selfmasking with randomness_strength={randomness_strength}")
    else:
        print("Using pure random remasking")
    print("=" * 80)


def log_iteration_start(iteration, iterations, tokens, mask_token_id, total_length, decode_mask_fn):
    """Log iteration start information"""
    masked_positions = (tokens == mask_token_id)
    num_masked_per_sample = masked_positions.sum(dim=1)
    avg_masked = num_masked_per_sample.float().mean().item()
    print(f"\nIteration {iteration + 1}/{iterations}")
    print(f"Average tokens masked: {avg_masked:.1f}/{total_length} ({avg_masked/total_length*100:.1f}%)")
    
    # Show first sample as example
    if decode_mask_fn:
        masked_sequence = decode_mask_fn(tokens[0].tolist())
        print(f"Sample 1 BEFORE: {masked_sequence[:100]}{'...' if len(masked_sequence) > 100 else ''}")


def log_sample_after_prediction(prediction_tokens, decode_fn):
    """Log sample after prediction"""
    if decode_fn:
        unmasked_sequence = decode_fn(prediction_tokens[0].tolist())
        print(f"Sample 1 AFTER:  {unmasked_sequence[:100]}{'...' if len(unmasked_sequence) > 100 else ''}")


def log_debug_prediction_analysis(sample_idx, logits, mask_indices, masked_logits, new_tokens, 
                                 mask_token_id, vocab_size, itos, stoi, log_debug):
    """Log detailed debug analysis during prediction"""
    if not (sample_idx == 0):  # Only debug first sample
        return
        
    # DEBUG: Check logits dimensions and values
    print(f"  DEBUG: Model output shape: {logits.shape}, Masked positions: {len(mask_indices)}")
    print(f"  DEBUG: Logits shape for masked positions: {masked_logits.shape}")
    print(f"  DEBUG: vocab_size={vocab_size}, mask_token_id={mask_token_id}")
    
    if masked_logits.shape[-1] > 0:
        mask_logit = masked_logits[0, mask_token_id] if mask_token_id < masked_logits.shape[-1] else float('-inf')
        vocab_logit_mean = masked_logits[0, :vocab_size].mean().item()
        print(f"  DEBUG: First position - mask_token logit: {mask_logit:.3f}, vocab mean logit: {vocab_logit_mean:.3f}")
        
        # MODEL CONFIDENCE ANALYSIS
        probs = torch.softmax(masked_logits, dim=-1)
        vocab_probs = probs[:, :vocab_size]  # Only vocab tokens
        
        # Calculate entropy (measure of uncertainty)
        entropy = -(vocab_probs * torch.log(vocab_probs + 1e-10)).sum(dim=-1)
        avg_entropy = entropy.mean().item()
        max_entropy = math.log(vocab_size)  # Maximum possible entropy
        normalized_entropy = avg_entropy / max_entropy
        
        # Calculate max probability (measure of confidence)
        max_probs = vocab_probs.max(dim=-1)[0]
        avg_confidence = max_probs.mean().item()
        
        print(f"  🧠 MODEL CONFIDENCE: Avg max prob: {avg_confidence:.3f}, Normalized entropy: {normalized_entropy:.3f}")
        
        # Check if model is very confident about punctuation
        punct_probs = vocab_probs[:, :13].sum(dim=-1)  # Sum of all punctuation probs
        high_punct_positions = (punct_probs > 0.7).sum().item()
        if high_punct_positions > 0:
            avg_punct_confidence = punct_probs[punct_probs > 0.7].mean().item() if high_punct_positions > 0 else 0
            print(f"  ⚠️  HIGH PUNCTUATION CONFIDENCE: {high_punct_positions} positions with >70% punct prob (avg: {avg_punct_confidence:.3f})")
            
            # Show which punctuation characters are being strongly predicted
            punct_logits = masked_logits[:, :13]
            punct_top_probs, punct_top_tokens = torch.topk(torch.softmax(punct_logits, dim=-1), 3, dim=-1)
            high_punct_masks = punct_probs > 0.7
            if high_punct_masks.sum() > 0:
                for pos_idx in range(min(3, high_punct_masks.sum().item())):
                    actual_pos_idx = high_punct_masks.nonzero()[pos_idx].item()
                    top_punct_chars = [itos[t.item()] if t.item() < len(itos) else f"[{t.item()}]" for t in punct_top_tokens[actual_pos_idx]]
                    top_punct_probs_vals = [f"{p.item():.3f}" for p in punct_top_probs[actual_pos_idx]]
                    punct_alternatives = ", ".join([f"'{c}':{p}" for c, p in zip(top_punct_chars, top_punct_probs_vals)])
                    print(f"    High-confidence punct position: {punct_alternatives}")

    # DEBUG: Check what tokens were actually sampled
    mask_count = (new_tokens == mask_token_id).sum().item()
    vocab_count = (new_tokens < vocab_size).sum().item()
    print(f"  DEBUG: Sampled {mask_count} mask tokens, {vocab_count} vocab tokens out of {len(new_tokens)}")
    if mask_count > 0:
        print(f"  DEBUG: This is suspicious - model predicting mask tokens!")
    
    # Count non-alphabetic characters
    alphabetic_tokens = set(range(13, 39)) | set(range(39, 65))  # A-Z, a-z
    non_alpha_count = sum(1 for token in new_tokens.tolist() if token not in alphabetic_tokens and token < vocab_size)
    print(f"  DEBUG: Non-alphabetic tokens: {non_alpha_count}/{len(new_tokens)} ({non_alpha_count/len(new_tokens)*100:.1f}%)")
    
    # Find most common token
    from collections import Counter
    token_counts = Counter(new_tokens.tolist())
    most_common_token, most_common_count = token_counts.most_common(1)[0]
    most_common_char = itos[most_common_token] if most_common_token < len(itos) else f"[{most_common_token}]"
    print(f"  DEBUG: Most common token: '{most_common_char}' (ID={most_common_token}) x{most_common_count}")
    
    # Show token distribution for non-alphabetic chars
    punct_counts = {token: count for token, count in token_counts.items() if token < 13 and count > 1}
    if punct_counts:
        punct_info = [f"'{itos[t]}' x{c}" for t, c in sorted(punct_counts.items())]
        print(f"  DEBUG: Repeated punctuation: {', '.join(punct_info)}")
    
    # DETAILED LOGGING: Analyze model predictions for specific patterns
    if log_debug:
        dash_token_id = stoi.get('-', None)
        if dash_token_id is not None:
            dash_positions = (new_tokens == dash_token_id)
            dash_count = dash_positions.sum().item()
            if dash_count > 0:
                print(f"  🔍 DASH ANALYSIS: Generated {dash_count} dashes at positions {dash_positions.nonzero().squeeze().tolist()}")
                
                # Check what was predicted for dash positions
                dash_mask_indices = mask_indices[dash_positions]
                if len(dash_mask_indices) > 0:
                    dash_logits = masked_logits[dash_positions]
                    dash_probs = torch.softmax(dash_logits, dim=-1)
                    dash_prob_for_dash = dash_probs[:, dash_token_id].mean().item()
                    
                    # Check top alternatives
                    top_probs, top_tokens = torch.topk(dash_probs, 5, dim=-1)
                    print(f"  🔍 DASH PREDICTIONS: Avg prob for dash={dash_prob_for_dash:.3f}")
                    for pos_idx in range(min(3, len(dash_logits))):  # Show first 3 dash positions
                        pos_top_tokens = [itos[t.item()] if t.item() < len(itos) else f"[{t.item()}]" for t in top_tokens[pos_idx]]
                        pos_top_probs = [f"{p.item():.3f}" for p in top_probs[pos_idx]]
                        alternatives = ", ".join([f"'{t}':{p}" for t, p in zip(pos_top_tokens, pos_top_probs)])
                        print(f"    Position {dash_mask_indices[pos_idx].item()}: {alternatives}")
        
        # SEQUENCE PATTERN ANALYSIS: Look for consecutive identical tokens
        new_tokens_list = new_tokens.tolist()
        consecutive_patterns = {}
        i = 0
        while i < len(new_tokens_list):
            current_token = new_tokens_list[i]
            if current_token < vocab_size:  # Only analyze vocab tokens
                consecutive_count = 1
                j = i + 1
                while j < len(new_tokens_list) and new_tokens_list[j] == current_token:
                    consecutive_count += 1
                    j += 1
                
                if consecutive_count > 1:
                    token_char = itos[current_token] if current_token < len(itos) else f"[{current_token}]"
                    if token_char not in consecutive_patterns:
                        consecutive_patterns[token_char] = []
                    consecutive_patterns[token_char].append((i, consecutive_count))
                i = j
            else:
                i += 1
        
        if consecutive_patterns:
            print(f"  🔍 REPETITION PATTERNS:")
            for char, occurrences in consecutive_patterns.items():
                for start_pos, count in occurrences:
                    abs_pos = mask_indices[start_pos].item()
                    print(f"    '{char}' x{count} at positions {abs_pos}-{abs_pos+count-1}")
        
        # CONTEXT ANALYSIS: Check what tokens surround frequently repeated patterns
        if dash_token_id is not None and (new_tokens == dash_token_id).sum().item() > 2:
            print(f"  🔍 CONTEXT ANALYSIS: Checking context around dash predictions")
            # This would need the full tokens sequence passed in - simplified for now
            print(f"    Context analysis simplified - dash count: {(new_tokens == dash_token_id).sum().item()}")


def log_iteration_summary(iteration, tokens, mask_token_id, itos, stoi):
    """Log iteration summary with pattern analysis"""
    print(f"  📊 ITERATION {iteration + 1} SUMMARY:")
    
    # Analyze the full sequence for patterns after each iteration
    full_sample = tokens[0].tolist()  # First sample
    unmasked_positions = [i for i, token in enumerate(full_sample) if token != mask_token_id]
    
    if len(unmasked_positions) > 0:
        unmasked_tokens = [full_sample[i] for i in unmasked_positions]
        
        # Count dash sequences
        dash_token_id = stoi.get('-', None)
        if dash_token_id is not None:
            dash_sequences = []
            current_dash_seq = 0
            max_dash_seq = 0
            
            for token in unmasked_tokens:
                if token == dash_token_id:
                    current_dash_seq += 1
                    max_dash_seq = max(max_dash_seq, current_dash_seq)
                else:
                    if current_dash_seq > 1:  # Record sequences of 2+
                        dash_sequences.append(current_dash_seq)
                    current_dash_seq = 0
            
            # Don't forget the last sequence
            if current_dash_seq > 1:
                dash_sequences.append(current_dash_seq)
            
            total_dashes = sum(1 for t in unmasked_tokens if t == dash_token_id)
            if total_dashes > 0:
                print(f"    🔸 Dashes: {total_dashes} total, max sequence: {max_dash_seq}")
                if dash_sequences:
                    seq_counts = Counter(dash_sequences)
                    seq_info = [f"{length}x{count}" for length, count in sorted(seq_counts.items())]
                    print(f"    🔸 Dash sequences (2+): {', '.join(seq_info)}")
        
        # Character type distribution
        punct_count = sum(1 for t in unmasked_tokens if t < 13)  # First 13 are punctuation
        alpha_count = sum(1 for t in unmasked_tokens if 13 <= t < 65)  # Letters
        total_unmasked = len(unmasked_tokens)
        
        print(f"    🔸 Character types: {punct_count} punct ({punct_count/total_unmasked*100:.1f}%), {alpha_count} letters ({alpha_count/total_unmasked*100:.1f}%)")
        
        # Most common tokens in this iteration
        from collections import Counter
        token_counter = Counter(unmasked_tokens)
        top_tokens = token_counter.most_common(5)
        top_info = [f"'{itos[t] if t < len(itos) else f'[{t}]'}' x{c}" for t, c in top_tokens]
        print(f"    🔸 Most common: {', '.join(top_info)}")


def diffusion_generate(model, batch_size, total_length, iterations, remasking_model, mask_token_id,
                      randomness_strength, decode_fn, decode_mask_fn, verbose=True, temperature=1.0,
                      top_p=1.0, schedule_type='linear', masking_ratios=None, repetition_penalty=1.0,
                      repetition_window=10, log_debug=False, intelligent_remasking=False):
    """
    Generate text samples using diffusion-based iterative demasking

    Args:
        model: Trained diffusion model
        batch_size: Number of samples to generate
        total_length: Length of sequence to generate
        iterations: Number of demasking iterations
        remasking_model: Optional remasking_binary model
        mask_token_id: ID of mask token
        randomness_strength: Balance between random and model-guided remasking (0-1)
        decode_fn: Function to decode tokens to text
        decode_mask_fn: Function to decode with mask characters
        verbose: Whether to print progress
        temperature: Temperature for sampling
        top_p: Nucleus sampling parameter (1.0 = disabled, <1.0 = only sample from top cumulative probability mass)
        schedule_type: 'linear' or 'custom' - type of masking schedule to use
        masking_ratios: Array of masking ratios for 'custom' schedule (overrides iterations)
        intelligent_remasking: Enable selfmasking using base model when remasking_model is None

    Returns:
        Generated tokens (batch_size, total_length)
    """
    # Start with all positions masked
    tokens = torch.full((batch_size, total_length), mask_token_id, dtype=torch.long, device=device)
    
    if verbose:
        log_generation_start(batch_size, iterations, total_length, remasking_model, randomness_strength, intelligent_remasking)
    
    for iteration in range(iterations):
        if verbose:
            log_iteration_start(iteration, iterations, tokens, mask_token_id, total_length, decode_mask_fn)
        
        # Step 1: Predict tokens for all masked positions
        debug_logging_fn = log_debug_prediction_analysis if log_debug else None
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
            debug_logging_fn=debug_logging_fn,
            itos=itos,
            stoi=stoi,
            verbose=verbose,
            log_debug=log_debug
        )
        
        if verbose:
            log_sample_after_prediction(prediction_tokens, decode_fn)
        
        # Step 2: Remask tokens for next iteration (except final iteration)
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
                verbose=verbose
            )
            
            if verbose and decode_mask_fn:
                remasked_sequence = decode_mask_fn(tokens[0].tolist())
                print(f"Sample 1 REMASKED: {remasked_sequence[:100]}{'...' if len(remasked_sequence) > 100 else ''}")
        
        # ITERATION SUMMARY LOGGING
        if verbose and log_debug:
            log_iteration_summary(iteration, tokens, mask_token_id, itos, stoi)
        
        if verbose:
            print("=" * 80)
    
    return tokens

# Initialize
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load main model
print(f"Loading main model from {checkpoint_name}...")
ckpt_path = os.path.join(out_dir, checkpoint_name)
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

model_args = checkpoint['model_args']
if 'attention_type' not in model_args:
    model_args['attention_type'] = 'causal'  # Backward compatibility

model_config = GPTConfig(**model_args)
model = GPT(model_config)

# Load model weights
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

print(f"Main model loaded (attention: {model_config.attention_type})")

# Load optional remasking model
remasking_model = None
if remasking_checkpoint_name is not None:
    print(f"Loading remasking model from {remasking_checkpoint_name}...")
    remasking_ckpt_path = os.path.join(out_dir, remasking_checkpoint_name)
    
    if os.path.exists(remasking_ckpt_path):
        remasking_checkpoint = torch.load(remasking_ckpt_path, map_location=device, weights_only=False)
        
        remasking_model_args = remasking_checkpoint['model_args']
        if 'attention_type' not in remasking_model_args:
            remasking_model_args['attention_type'] = 'causal'
        
        # Verify it's a binary classification model
        if not remasking_model_args.get('binary_classification', False):
            print("Warning: Remasking model is not binary classification, skipping")
            remasking_model = None
        else:
            remasking_config = GPTConfig(**remasking_model_args)
            remasking_model = GPT(remasking_config)
            
            # Load weights
            remasking_state_dict = remasking_checkpoint['model']
            for k, v in list(remasking_state_dict.items()):
                if k.startswith(unwanted_prefix):
                    remasking_state_dict[k[len(unwanted_prefix):]] = remasking_state_dict.pop(k)
            remasking_model.load_state_dict(remasking_state_dict)
            
            remasking_model.eval()
            remasking_model.to(device)
            if compile:
                remasking_model = torch.compile(remasking_model)
            
            print(f"Remasking model loaded (binary classification)")
    else:
        print(f"Remasking checkpoint not found: {remasking_ckpt_path}")

# Load vocabulary
dataset_name = None
if 'config' in checkpoint:
    config = checkpoint['config']
    if hasattr(config, 'get'):
        dataset_name = config.get('dataset')
    elif hasattr(config, '__getitem__'):
        try:
            dataset_name = config['dataset']
        except (KeyError, TypeError):
            pass

if not dataset_name:
    if 'shakespeare' in checkpoint_name.lower():
        dataset_name = 'shakespeare_char'
    else:
        dataset_name = 'shakespeare_char'
    print(f"Using default dataset: {dataset_name}")

meta_path = os.path.join('data', dataset_name, 'meta.pkl')

if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    vocab_size = meta['vocab_size']
    mask_token_id = vocab_size  # Mask token ID
    
    def decode(token_ids):
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
        result = []
        for token_id in token_ids:
            if token_id == mask_token_id:
                result.append(mask_char)
            elif token_id < len(itos):
                result.append(itos[token_id])
            else:
                result.append('[UNK]')
        return ''.join(result)
    
    print(f"Vocabulary loaded: size={vocab_size}, mask_token_id={mask_token_id}")
else:
    raise FileNotFoundError(f"Meta file not found: {meta_path}")

# Generate samples
print(f"\n{'='*80}")
print(f"GENERATION SETTINGS")
print(f"Samples: {num_samples}, Length: {sequence_length}, Iterations: {diffusion_iterations}")
print(f"Temperature: {temperature}, Top-p: {top_p}, Repetition penalty: {repetition_penalty}, Seed: {seed}")
print(f"Remasking schedule: {start_ratio:.1%} → {end_ratio:.1%}")
if remasking_model is not None:
    print(f"Randomness strength: {randomness_strength} (0=pure model, 1=pure random)")
print(f"{'='*80}")

with torch.no_grad():
    with ctx:
        generated_tokens = diffusion_generate(
            model=model,
            batch_size=num_samples,
            total_length=sequence_length,
            iterations=diffusion_iterations,
            remasking_model=remasking_model,
            mask_token_id=mask_token_id,
            randomness_strength=randomness_strength,
            decode_fn=decode,
            decode_mask_fn=decode_with_mask_char,
            verbose=use_verbose_logging,
            temperature=temperature,
            top_p=top_p,
            schedule_type=schedule_type,
            masking_ratios=masking_ratios,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            log_debug=log_debug,
            intelligent_remasking=intelligent_remasking
        )
        
        # Calculate self-confidence scores for all samples
        confidence_scores = calculate_selfconfidence_ratio(
            model=model,
            tokens=generated_tokens,
            mask_token_id=mask_token_id,
            device=device,
            ctx=ctx
        )
        
        # Display results
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS")
        print(f"{'='*80}")
        
        for i in range(num_samples):
            print(f"\n{'─'*60}")
            print(f"SAMPLE {i+1}/{num_samples}")
            log_prob = confidence_scores[i]
            raw_prob = math.exp(log_prob) if log_prob != float('-inf') else 0.0
            print(f"Self-confidence score (avg log-prob): {log_prob:.4f} (raw prob: {raw_prob:.4f})")
            print(f"{'─'*60}")
            sample_text = decode(generated_tokens[i].tolist())
            print(sample_text)
        
        print(f"\n{'='*80}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*80}")