"""
Simplified diffusion-based text generation using iterative demasking
Supports only remasking_binary models or no remasking model (random)
"""
import os
import pickle
import math
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration
init_from = 'resume'
out_dir = 'out'
checkpoint_name = 'optimal2_3400.pt' #'35.75_58.2_UM.pt'
remasking_checkpoint_name = None #'ckpt_remasking_binary_600.pt'  # Optional: remasking_binary model checkpoint
num_samples = 1  # Number of samples to generate
sequence_length = 1024  # Total length of generated sequence
seed = -1
device = 'cpu'
dtype = 'float32'
compile = False

# Generation parameters
temperature = 1.0  # Temperature for sampling (1.0 = no change, <1.0 = more deterministic, >1.0 = more random)
diffusion_iterations = 100  # Number of demasking iterations
start_ratio = 0.99  # Initial ratio of tokens to remask (99%)
end_ratio = 0.05   # Final ratio of tokens to remask (5%)

# Schedule parameters
schedule_type = 'custom'  # 'linear' or 'custom' - type of masking schedule to use
#masking_ratios = [0.85,0.75,0.65,0.55,0.45,0.35,0.25,0.15,0.05,0.65,0.60,0.65,0.6,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1]    # Array of masking ratios for 'custom' schedule (overrides diffusion_iterations)
masking_ratios = [0.85,0.816,0.782,0.748,0.714,0.68,0.646,0.612,0.578,0.544,0.51,0.476,0.442,0.408,0.374,0.34,0.306,0.272,0.238,0.204,0.17,0.136,0.102,0.068,0.034]   # Array of masking ratios for 'custom' schedule (overrides diffusion_iterations)

# Remasking parameters (only used if remasking model is available)
randomness_strength = 0.4 # Balance between random (1.0) and model-guided (0.0) remasking

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

# -----------------------------------------------------------------------------

def linear_remasking_schedule(total_iterations, current_iteration):
    """Linear decrease in remask ratio from start_ratio to end_ratio"""
    progress = current_iteration / (total_iterations - 1) if total_iterations > 1 else 1.0
    return start_ratio - progress * (start_ratio - end_ratio)

def apply_remasking(tokens, remask_ratio, remasking_model, randomness_strength, mask_token_id, device, verbose=False):
    """
    Apply remasking using either random selection or model-guided selection
    
    Args:
        tokens: Current token sequence (batch_size, seq_len)
        remask_ratio: Fraction of tokens to remask
        remasking_model: Optional remasking_binary model
        randomness_strength: Balance between random (1.0) and model-guided (0.0)
        mask_token_id: ID of mask token
        device: Device to run on
        verbose: Whether to print debug info
    
    Returns:
        tokens: Updated token sequence with remasked positions
    """
    batch_size, seq_len = tokens.shape
    
    # Calculate target number of masks based on total sequence length
    target_masked = int(seq_len * remask_ratio)
    
    # Count currently masked positions
    currently_masked = (tokens == mask_token_id).sum(dim=1)  # Count per sample
    
    # Calculate how many additional masks needed per sample
    additional_masks_needed = target_masked - currently_masked
    additional_masks_needed = torch.clamp(additional_masks_needed, min=0)  # Don't unmask
    
    # Check if any samples need additional masks
    if additional_masks_needed.sum() == 0:
        if verbose:
            print(f"  No tokens to remask (ratio: {remask_ratio:.3f})")
        return tokens
    
    if remasking_model is None:
        # Pure random remasking
        if verbose:
            avg_additional = additional_masks_needed.float().mean().item()
            avg_currently_masked = currently_masked.float().mean().item()
            print(f"  Using pure random remasking: adding {avg_additional:.1f} masks (currently {avg_currently_masked:.1f}/{seq_len}, target {target_masked})")
        
        for batch_idx in range(batch_size):
            num_additional = additional_masks_needed[batch_idx].item()
            if num_additional > 0:
                # Only select from unmasked positions
                unmasked_positions = (tokens[batch_idx] != mask_token_id)
                unmasked_indices = torch.where(unmasked_positions)[0]
                if len(unmasked_indices) > 0:
                    num_to_select = min(num_additional, len(unmasked_indices))
                    selected_positions = torch.randperm(len(unmasked_indices), device=device)[:num_to_select]
                    remask_indices = unmasked_indices[selected_positions]
                    tokens[batch_idx, remask_indices] = mask_token_id
    else:
        # Model-guided remasking with randomness
        if verbose:
            avg_additional = additional_masks_needed.float().mean().item()
            avg_currently_masked = currently_masked.float().mean().item()
            print(f"  Using model-guided remasking: randomness={randomness_strength:.2f}, adding {avg_additional:.1f} masks (currently {avg_currently_masked:.1f}/{seq_len}, target {target_masked})")
        
        with torch.no_grad():
            # Get model predictions for all samples
            logits, _ = remasking_model(tokens, None)
            
            # Get probabilities for "remask" class (class 1 for remasking_binary)
            probs = torch.softmax(logits, dim=-1)  # (batch_size, seq_len, 2)
            wrong_token_probs = probs[:, :, 1]  # (batch_size, seq_len)
            
            for batch_idx in range(batch_size):
                num_additional = additional_masks_needed[batch_idx].item()
                if num_additional > 0:
                    # Only consider unmasked positions for remasking
                    unmasked_positions = (tokens[batch_idx] != mask_token_id)
                    unmasked_indices = torch.where(unmasked_positions)[0]
                    if len(unmasked_indices) > 0:
                        # Get probabilities only for unmasked positions
                        unmasked_model_probs = wrong_token_probs[batch_idx, unmasked_indices]
                        unmasked_rand_probs = torch.rand(len(unmasked_indices), device=device)
                        
                        # Combine random and model probabilities
                        combined_probs = randomness_strength * unmasked_rand_probs + (1 - randomness_strength) * unmasked_model_probs
                        
                        # Select top positions by combined probability
                        num_to_select = min(num_additional, len(unmasked_indices))
                        _, top_indices = torch.topk(combined_probs, num_to_select)
                        remask_indices = unmasked_indices[top_indices]
                        tokens[batch_idx, remask_indices] = mask_token_id
                        
                        if verbose and batch_idx == 0:  # Show stats for first sample
                            model_prob_range = f"{unmasked_model_probs.min().item():.3f}-{unmasked_model_probs.max().item():.3f}"
                            selected_model_probs = unmasked_model_probs[top_indices]
                            selected_rand_probs = unmasked_rand_probs[top_indices]
                            avg_model_prob = selected_model_probs.mean().item()
                            avg_rand_prob = selected_rand_probs.mean().item()
                            print(f"    Model prob range: {model_prob_range}")
                            print(f"    Selected positions - avg model prob: {avg_model_prob:.3f}, avg rand prob: {avg_rand_prob:.3f}")
    
    return tokens

def diffusion_generate(model, batch_size, total_length, iterations, remasking_model, mask_token_id,
                      randomness_strength, decode_fn, decode_mask_fn, verbose=True, temperature=1.0,
                      schedule_type='linear', masking_ratios=None):
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
        schedule_type: 'linear' or 'custom' - type of masking schedule to use
        masking_ratios: Array of masking ratios for 'custom' schedule (overrides iterations)

    Returns:
        Generated tokens (batch_size, total_length)
    """
    # Start with all positions masked
    tokens = torch.full((batch_size, total_length), mask_token_id, dtype=torch.long, device=device)
    
    if verbose:
        print(f"Starting diffusion generation: {batch_size} samples, {iterations} iterations")
        print(f"Total length: {total_length} (all tokens start masked)")
        if remasking_model is not None:
            print(f"Using remasking_binary model with randomness_strength={randomness_strength}")
        else:
            print("Using pure random remasking")
        print("=" * 80)
    
    for iteration in range(iterations):
        if verbose:
            masked_positions = (tokens == mask_token_id)
            num_masked_per_sample = masked_positions.sum(dim=1)
            avg_masked = num_masked_per_sample.float().mean().item()
            print(f"\nIteration {iteration + 1}/{iterations}")
            print(f"Average tokens masked: {avg_masked:.1f}/{total_length} ({avg_masked/total_length*100:.1f}%)")
            
            # Show first sample as example
            if decode_mask_fn:
                masked_sequence = decode_mask_fn(tokens[0].tolist())
                print(f"Sample 1 BEFORE: {masked_sequence[:100]}{'...' if len(masked_sequence) > 100 else ''}")
        
        # Step 1: Predict tokens for all masked positions
        masked_positions = (tokens == mask_token_id)
        total_masked = masked_positions.sum().item()
        
        if total_masked > 0:
            with torch.no_grad():
                # Get predictions for all samples
                dummy_targets = torch.zeros_like(tokens)
                logits, _ = model(tokens, dummy_targets)
                
                # Sample new tokens for masked positions
                for sample_idx in range(batch_size):
                    sample_masked = masked_positions[sample_idx]
                    if sample_masked.sum() > 0:
                        mask_indices = torch.where(sample_masked)[0]
                        masked_logits = logits[sample_idx, mask_indices]
                        
                        # DEBUG: Check logits dimensions and values
                        if verbose and sample_idx == 0:
                            print(f"  DEBUG: Model output shape: {logits.shape}, Masked positions: {len(mask_indices)}")
                            print(f"  DEBUG: Logits shape for masked positions: {masked_logits.shape}")
                            print(f"  DEBUG: vocab_size={vocab_size}, mask_token_id={mask_token_id}")
                            if masked_logits.shape[-1] > 0:
                                mask_logit = masked_logits[0, mask_token_id] if mask_token_id < masked_logits.shape[-1] else float('-inf')
                                vocab_logit_mean = masked_logits[0, :vocab_size].mean().item()
                                print(f"  DEBUG: First position - mask_token logit: {mask_logit:.3f}, vocab mean logit: {vocab_logit_mean:.3f}")
                        
                        # Apply temperature
                        if temperature != 1.0:
                            masked_logits = masked_logits / temperature
                        
                        probs = torch.softmax(masked_logits, dim=-1)
                        new_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                        
                        # DEBUG: Check what tokens were actually sampled
                        if verbose and sample_idx == 0:
                            mask_count = (new_tokens == mask_token_id).sum().item()
                            vocab_count = (new_tokens < vocab_size).sum().item()
                            print(f"  DEBUG: Sampled {mask_count} mask tokens, {vocab_count} vocab tokens out of {len(new_tokens)}")
                            if mask_count > 0:
                                print(f"  DEBUG: This is suspicious - model predicting mask tokens!")
                        
                        tokens[sample_idx, mask_indices] = new_tokens
        
        # Save tokens after prediction for accurate display
        prediction_tokens = tokens.clone()  # Always save the actual prediction results
        
        if verbose:
            predicted_masks = (prediction_tokens == mask_token_id).sum().item()
            if predicted_masks > 0:
                print(f"  ⚠️  Model predicted {predicted_masks} mask tokens during prediction step!")
        
        if verbose and decode_fn:
            unmasked_sequence = decode_fn(prediction_tokens[0].tolist())
            print(f"Sample 1 AFTER:  {unmasked_sequence[:100]}{'...' if len(unmasked_sequence) > 100 else ''}")
        
        # Step 2: Remask tokens for next iteration (except final iteration)
        if iteration < iterations - 1:
            if schedule_type == 'custom':
                # Use the next ratio from the custom schedule
                remask_ratio = masking_ratios[iteration + 1]
            else:
                # Use linear schedule
                remask_ratio = linear_remasking_schedule(iterations, iteration + 1)
            
            # Debug: Track masks before remasking (using actual prediction results)
            masks_before = (prediction_tokens == mask_token_id).sum().item()
            
            tokens = apply_remasking(tokens, remask_ratio, remasking_model, randomness_strength, 
                                   mask_token_id, device, verbose)
            
            # Debug: Track masks after remasking
            if verbose:
                masks_after = (tokens == mask_token_id).sum().item()
                mask_change = masks_after - masks_before
                print(f"  Masks: {masks_before} → {masks_after} (change: {mask_change:+d})")
            
            if verbose and decode_mask_fn:
                remasked_sequence = decode_mask_fn(tokens[0].tolist())
                print(f"Sample 1 REMASKED: {remasked_sequence[:100]}{'...' if len(remasked_sequence) > 100 else ''}")
        
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
print(f"Temperature: {temperature}, Seed: {seed}")
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
            verbose=True,
            temperature=temperature,
            schedule_type=schedule_type,
            masking_ratios=masking_ratios
        )
        
        # Display results
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS")
        print(f"{'='*80}")
        
        for i in range(num_samples):
            print(f"\n{'─'*60}")
            print(f"SAMPLE {i+1}/{num_samples}")
            print(f"{'─'*60}")
            sample_text = decode(generated_tokens[i].tolist())
            print(sample_text)
        
        print(f"\n{'='*80}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*80}")