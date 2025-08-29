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
checkpoint_name = '35.75_58.2_UM.pt'  # Main model checkpoint to load
remasking_checkpoint_name = 'ckpt_remasking_binary_600.pt'  # Optional: remasking_binary model checkpoint
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

# Remasking parameters (only used if remasking model is available)
randomness_strength = 1 # Balance between random (1.0) and model-guided (0.0) remasking

if seed == -1:
    seed = int.from_bytes(os.urandom(4), byteorder='little')

exec(open('configurator.py').read()) # overrides from command line or config file
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
    num_to_remask = int(seq_len * remask_ratio)
    
    if num_to_remask == 0:
        if verbose:
            print(f"  No tokens to remask (ratio: {remask_ratio:.3f})")
        return tokens
    
    if remasking_model is None:
        # Pure random remasking
        if verbose:
            print(f"  Using pure random remasking: {num_to_remask}/{seq_len} tokens ({remask_ratio:.1%})")
        
        for batch_idx in range(batch_size):
            # Random selection for each sample
            remask_indices = torch.randperm(seq_len, device=device)[:num_to_remask]
            tokens[batch_idx, remask_indices] = mask_token_id
    else:
        # Model-guided remasking with randomness
        if verbose:
            print(f"  Using model-guided remasking: randomness={randomness_strength:.2f}, target={num_to_remask}/{seq_len} tokens ({remask_ratio:.1%})")
        
        with torch.no_grad():
            # Get model predictions for all samples
            logits, _ = remasking_model(tokens, None)
            
            # Get probabilities for "remask" class (class 1 for remasking_binary)
            probs = torch.softmax(logits, dim=-1)  # (batch_size, seq_len, 2)
            wrong_token_probs = probs[:, :, 1]  # (batch_size, seq_len)
            
            for batch_idx in range(batch_size):
                # Generate random probabilities for each position
                rand_probs = torch.rand(seq_len, device=device)
                
                # Combine random and model probabilities
                combined_probs = randomness_strength * rand_probs + (1 - randomness_strength) * wrong_token_probs[batch_idx]
                
                # Select top positions by combined probability
                _, top_indices = torch.topk(combined_probs, num_to_remask)
                tokens[batch_idx, top_indices] = mask_token_id
                
                if verbose and batch_idx == 0:  # Show stats for first sample
                    model_prob_range = f"{wrong_token_probs[batch_idx].min().item():.3f}-{wrong_token_probs[batch_idx].max().item():.3f}"
                    selected_model_probs = wrong_token_probs[batch_idx, top_indices]
                    selected_rand_probs = rand_probs[top_indices]
                    avg_model_prob = selected_model_probs.mean().item()
                    avg_rand_prob = selected_rand_probs.mean().item()
                    print(f"    Model prob range: {model_prob_range}")
                    print(f"    Selected positions - avg model prob: {avg_model_prob:.3f}, avg rand prob: {avg_rand_prob:.3f}")
    
    return tokens

def diffusion_generate(model, batch_size, total_length, iterations, remasking_model, mask_token_id, 
                      randomness_strength, decode_fn, decode_mask_fn, verbose=True, temperature=1.0):
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
                        
                        # Apply temperature
                        if temperature != 1.0:
                            masked_logits = masked_logits / temperature
                        
                        probs = torch.softmax(masked_logits, dim=-1)
                        new_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                        tokens[sample_idx, mask_indices] = new_tokens
        
        if verbose and decode_fn:
            unmasked_sequence = decode_fn(tokens[0].tolist())
            print(f"Sample 1 AFTER:  {unmasked_sequence[:100]}{'...' if len(unmasked_sequence) > 100 else ''}")
        
        # Step 2: Remask tokens for next iteration (except final iteration)
        if iteration < iterations - 1:
            remask_ratio = linear_remasking_schedule(iterations, iteration + 1)
            tokens = apply_remasking(tokens, remask_ratio, remasking_model, randomness_strength, 
                                   mask_token_id, device, verbose)
            
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
            temperature=temperature
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