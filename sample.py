"""
Diffusion-based text generation using iterative demasking
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
checkpoint_name = '35.75_58.2_UM.pt'  # Specific checkpoint to load
remasking_checkpoint_name = '1.23_remasking_bin.pt'  # Optional: checkpoint for remasking model, if None uses random remasking
remasking_model_type = 'auto'  # 'remasking', 'remasking_binary', or 'auto' to detect from checkpoint
num_samples = 1  # Number of samples to generate
sequence_length = 1024  # Total length of generated sequence
seed = -1
device = 'cpu'
dtype = 'float32'
compile = False

if seed == -1:
    seed = int.from_bytes(os.urandom(4), byteorder='little')

# Sampling parameters for non-deterministic generation
temperature = 1.0  # Temperature for sampling (1.0 = no change, <1.0 = more deterministic, >1.0 = more random)
seed_offset_multiplier = 42  # Multiplier for seed separation between samples

use_intelligent_remasking = False
use_mixed_remasking = False
remasking_confidence_threshold = 0.58
remasking_schedule = 'linear'
diffusion_iterations = 20
min_random_ratio = 0.9
start_ratio = 0.99
end_ratio = 0.05

exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

def linear_remasking_schedule(total_iterations, current_iteration):
    """Linear decrease in remask ratio from high to low"""
    # Start with high remasking (e.g., 80%), end with low (e.g., 10%)
    progress = current_iteration / (total_iterations - 1) if total_iterations > 1 else 1.0
    return start_ratio - progress * (start_ratio - end_ratio)

def exponential_remasking_schedule(total_iterations, current_iteration):
    """Exponential decrease in remask ratio"""
    progress = current_iteration / (total_iterations - 1) if total_iterations > 1 else 1.0
    # Exponential decay
    decay_factor = -2.0  # Controls steepness
    exp_progress = (math.exp(decay_factor * progress) - 1) / (math.exp(decay_factor) - 1)
    return start_ratio - exp_progress * (start_ratio - end_ratio)

def intelligent_remask(tokens, remasking_model, num_to_remask, mask_token_id, wrong_token_id, remask_wrong_id, model_type, device, verbose=False):
    """
    Use remasking model to intelligently select positions to remask
    
    Args:
        tokens: Current token sequence
        remasking_model: Trained remasking model
        num_to_remask: Number of positions to remask
        mask_token_id: ID of mask token
        wrong_token_id: ID of wrong token (for remasking models)
        remask_wrong_id: ID of wrong token (for remasking_binary models)
        model_type: Type of remasking model ('remasking' or 'remasking_binary')
        device: Device to run on
        verbose: Whether to print debug info
    """
    with torch.no_grad():
        # Get model predictions
        logits, _ = remasking_model(tokens, None)
        
        # Select appropriate target token based on model type
        target_token_id = remask_wrong_id if model_type == 'remasking_binary' else wrong_token_id
        
        # Get probabilities for target token at each position
        probs = torch.softmax(logits[0], dim=-1)
        target_probs = probs[:, target_token_id]
        
        if verbose:
            model_name = "[REMASK_WRONG]" if model_type == 'remasking_binary' else "[WRONG]"
            print(f"  {model_name} token probabilities range: {target_probs.min().item():.4f} to {target_probs.max().item():.4f}")
        
        # Select top positions by target token probability
        if num_to_remask > 0:
            top_probs, top_indices = torch.topk(target_probs, min(num_to_remask, tokens.size(1)))
            tokens[0, top_indices] = mask_token_id
            
            if verbose:
                lowest_qualifying_prob = top_probs[-1].item()
                model_name = "[REMASK_WRONG]" if model_type == 'remasking_binary' else "[WRONG]"
                print(f"  Selected {len(top_indices)} positions with highest {model_name} probabilities")
                print(f"  Lowest qualifying probability: {lowest_qualifying_prob:.4f}")
        else:
            if verbose:
                print("  No positions to remask (num_to_remask = 0)")
    
    return tokens

def intelligent_remask_threshold(tokens, remasking_model, threshold, mask_token_id, wrong_token_id, remask_wrong_id, model_type, device, verbose=False):
    """
    Use remasking model with probability threshold to select positions to remask
    
    Args:
        tokens: Current token sequence
        remasking_model: Trained remasking model
        threshold: Target token probability threshold for remasking
        mask_token_id: ID of mask token
        wrong_token_id: ID of wrong token (for remasking models)
        remask_wrong_id: ID of wrong token (for remasking_binary models)
        model_type: Type of remasking model ('remasking' or 'remasking_binary')
        device: Device to run on
        verbose: Whether to print debug info
    
    Returns:
        tokens: Updated token sequence
        num_remasked: Number of tokens that were remasked
    """
    with torch.no_grad():
        # Get model predictions
        logits, _ = remasking_model(tokens, None)
        
        # Select appropriate target token based on model type
        target_token_id = remask_wrong_id if model_type == 'remasking_binary' else wrong_token_id
        
        # Get probabilities for target token at each position
        probs = torch.softmax(logits[0], dim=-1)
        target_probs = probs[:, target_token_id]
        
        # Find positions where target probability exceeds threshold
        threshold_mask = target_probs > threshold
        num_remasked = threshold_mask.sum().item()
        
        if verbose:
            model_name = "[REMASK_WRONG]" if model_type == 'remasking_binary' else "[WRONG]"
            print(f"  {model_name} token probabilities range: {target_probs.min().item():.4f} to {target_probs.max().item():.4f}")
            print(f"  Threshold: {threshold:.4f}")
            print(f"  Positions above threshold: {num_remasked}/{tokens.size(1)}")
        
        # Remask positions above threshold
        if num_remasked > 0:
            tokens[0, threshold_mask] = mask_token_id
            
            if verbose:
                min_remasked_prob = target_probs[threshold_mask].min().item()
                print(f"  Lowest remasked probability: {min_remasked_prob:.4f}")
        else:
            if verbose:
                print("  No positions above threshold")
    
    return tokens, num_remasked

def mixed_intelligent_remask(tokens, remasking_model, num_to_remask, threshold, mask_token_id, wrong_token_id, remask_wrong_id, model_type, device, min_random_ratio=0.0, verbose=False):
    """
    Mixed remasking: combines intelligent candidate selection with schedule-based count
    
    Args:
        tokens: Current token sequence
        remasking_model: Trained remasking model
        num_to_remask: Number of positions to remask (from schedule)
        threshold: Target token probability threshold for candidates
        mask_token_id: ID of mask token
        wrong_token_id: ID of wrong token (for remasking models)
        remask_wrong_id: ID of wrong token (for remasking_binary models)
        model_type: Type of remasking model ('remasking' or 'remasking_binary')
        device: Device to run on
        min_random_ratio: Minimum fraction that must be randomly selected
        verbose: Whether to print debug info
    
    Returns:
        tokens: Updated token sequence
        intelligent_count: Number of intelligently selected tokens
        random_count: Number of randomly selected tokens
    """
    with torch.no_grad():
        # Calculate minimum random tokens required
        min_random_tokens = int(num_to_remask * min_random_ratio)
        max_intelligent_tokens = num_to_remask - min_random_tokens
        
        # Get model predictions
        logits, _ = remasking_model(tokens, None)
        
        # Select appropriate target token based on model type
        target_token_id = remask_wrong_id if model_type == 'remasking_binary' else wrong_token_id
        
        # Get probabilities for target token at each position
        probs = torch.softmax(logits[0], dim=-1)
        target_probs = probs[:, target_token_id]
        
        # Find candidates above threshold
        threshold_mask = target_probs > threshold
        num_candidates = threshold_mask.sum().item()
        
        if verbose:
            model_name = "[REMASK_WRONG]" if model_type == 'remasking_binary' else "[WRONG]"
            print(f"  {model_name} token probabilities range: {target_probs.min().item():.4f} to {target_probs.max().item():.4f}")
            print(f"  Threshold: {threshold:.4f}")
            print(f"  Schedule requires: {num_to_remask} tokens to remask")
            print(f"  Min random ratio: {min_random_ratio:.2f} → min {min_random_tokens} random, max {max_intelligent_tokens} intelligent")
            print(f"  Intelligent candidates above threshold: {num_candidates}/{tokens.size(1)} tokens ({num_candidates/tokens.size(1)*100:.1f}%)")
        
        # Determine how many intelligent tokens we can actually use
        intelligent_tokens_to_use = min(num_candidates, max_intelligent_tokens)
        random_tokens_to_use = num_to_remask - intelligent_tokens_to_use
        
        # Step 1: Select intelligent tokens (if any)
        intelligent_positions = []
        if intelligent_tokens_to_use > 0:
            candidate_probs = target_probs[threshold_mask]
            _, top_indices_in_candidates = torch.topk(candidate_probs, intelligent_tokens_to_use)
            candidate_positions = torch.where(threshold_mask)[0]
            intelligent_positions = candidate_positions[top_indices_in_candidates]
            tokens[0, intelligent_positions] = mask_token_id
        
        # Step 2: Select random tokens for the remainder
        random_positions = []
        if random_tokens_to_use > 0:
            # Find positions not already selected by intelligent selection
            all_positions = torch.arange(tokens.size(1), device=device)
            if len(intelligent_positions) > 0:
                # Create mask for positions not already selected
                selected_mask = torch.zeros(tokens.size(1), dtype=torch.bool, device=device)
                selected_mask[intelligent_positions] = True
                available_positions = all_positions[~selected_mask]
            else:
                available_positions = all_positions
            
            if len(available_positions) >= random_tokens_to_use:
                random_indices = torch.randperm(len(available_positions))[:random_tokens_to_use]
                random_positions = available_positions[random_indices]
                tokens[0, random_positions] = mask_token_id
            else:
                # Not enough positions available, take all remaining
                random_positions = available_positions
                tokens[0, random_positions] = mask_token_id
                random_tokens_to_use = len(available_positions)
        
        intelligent_count = intelligent_tokens_to_use
        random_count = len(random_positions)
        
        if verbose:
            intelligent_pct = (intelligent_count / num_to_remask * 100) if num_to_remask > 0 else 0
            random_pct = (random_count / num_to_remask * 100) if num_to_remask > 0 else 0
            actual_random_ratio = random_count / num_to_remask if num_to_remask > 0 else 0
            
            print(f"  ✓ INTELLIGENT SELECTION: Used {intelligent_count}/{num_to_remask} tokens ({intelligent_pct:.1f}%)")
            if intelligent_count > 0:
                candidate_probs = target_probs[threshold_mask]
                if intelligent_count == len(candidate_probs):
                    print(f"    - All {intelligent_count} candidates above threshold selected")
                    min_intelligent_prob = candidate_probs.min().item()
                    max_intelligent_prob = candidate_probs.max().item()
                else:
                    # Only top N were selected
                    _, top_indices = torch.topk(candidate_probs, intelligent_count)
                    selected_probs = candidate_probs[top_indices]
                    print(f"    - Top {intelligent_count} candidates by probability (limited by min_random_ratio)")
                    min_intelligent_prob = selected_probs.min().item()
                    max_intelligent_prob = selected_probs.max().item()
                print(f"    - Probability range: {min_intelligent_prob:.4f} to {max_intelligent_prob:.4f}")
            else:
                if num_candidates == 0:
                    print(f"    - No candidates above threshold {threshold:.4f}")
                else:
                    print(f"    - {num_candidates} candidates available but limited by min_random_ratio")
            
            print(f"  ✗ RANDOM SELECTION: Used {random_count}/{num_to_remask} tokens ({random_pct:.1f}%)")
            print(f"    - Actual random ratio: {actual_random_ratio:.2f} (required min: {min_random_ratio:.2f})")
            if random_count < min_random_tokens:
                print(f"    - WARNING: Could not meet minimum random requirement ({random_count} < {min_random_tokens})")
    
    return tokens, intelligent_count, random_count

def diffusion_generate_batch(model, batch_size, total_length, iterations, schedule='linear', mask_token_id=None, wrong_token_id=None, remask_wrong_id=None, remasking_model=None, remasking_model_type='remasking', decode_fn=None, decode_mask_fn=None, verbose=True, use_threshold_remasking=False, use_mixed_remasking=False, threshold=0.1, temperature=1.0):
    """
    Generate multiple text samples using diffusion-based iterative demasking with batching

    Args:
        model: Trained diffusion model
        batch_size: Number of samples to generate in parallel
        total_length: Total length of sequence to generate
        iterations: Number of demasking/remasking iterations
        schedule: Remasking schedule ('linear' or 'exponential')
        mask_token_id: ID of the mask token
        wrong_token_id: ID of the wrong token (for remasking model)
        remask_wrong_id: ID of the wrong token (for remasking_binary model)
        remasking_model: Optional trained remasking model for intelligent remasking
        remasking_model_type: Type of remasking model ('remasking' or 'remasking_binary')
        decode_fn: Function to decode tokens to text (handles mask tokens)
        decode_mask_fn: Function to decode tokens with mask character
        verbose: Whether to print iteration results
        use_threshold_remasking: If True, use probability threshold instead of schedule
        use_mixed_remasking: If True, use mixed random+intelligent remasking
        threshold: Target token probability threshold for remasking
        temperature: Temperature for sampling (1.0 = no change, <1.0 = more deterministic, >1.0 = more random)
    
    Returns:
        torch.Tensor: Generated tokens of shape (batch_size, total_length)
    """

    # Start with ALL positions masked for all samples in batch
    tokens = torch.full((batch_size, total_length), mask_token_id, dtype=torch.long, device=device)

    if verbose:
        print(f"Starting batched diffusion generation: {batch_size} samples, {iterations} iterations, schedule: {schedule}")
        print(f"Total length: {total_length} (all tokens start masked)")
        print("=" * 80)

    for iteration in range(iterations):
        # Show state before unmasking (show stats for all samples)
        if verbose:
            masked_positions = (tokens == mask_token_id)
            num_masked_per_sample = masked_positions.sum(dim=1)
            avg_masked = num_masked_per_sample.float().mean().item()
            print(f"\nIteration {iteration + 1}/{iterations}")
            print(f"Average tokens masked: {avg_masked:.1f}/{total_length} ({avg_masked/total_length*100:.1f}%)")

            # Show first sample's sequence with mask characters as example
            if decode_mask_fn:
                masked_sequence = decode_mask_fn(tokens[0].tolist())
                print(f"Sample 1 BEFORE unmasking: {masked_sequence[:100]}{'...' if len(masked_sequence) > 100 else ''}")

        # Step 1: Predict tokens for all masked positions (batch processing)
        masked_positions = (tokens == mask_token_id)
        total_masked = masked_positions.sum().item()

        if total_masked > 0:
            with torch.no_grad():
                # Get logits for all positions and all samples in batch
                dummy_targets = torch.zeros_like(tokens)
                logits, _ = model(tokens, dummy_targets)

            # Process each sample in the batch
            for sample_idx in range(batch_size):
                sample_masked_positions = masked_positions[sample_idx]
                if sample_masked_positions.sum() > 0:
                    # Sample new tokens for masked positions with temperature
                    mask_indices = torch.where(sample_masked_positions)[0]
                    masked_logits = logits[sample_idx, mask_indices]
                    
                    # Apply temperature to logits before softmax
                    if temperature != 1.0:
                        masked_logits = masked_logits / temperature
                    
                    probs = torch.softmax(masked_logits, dim=-1)
                    new_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

                    # Replace masked tokens with predictions
                    tokens[sample_idx, mask_indices] = new_tokens

        # Show state after unmasking (show first sample as example)
        if verbose and decode_fn:
            unmasked_sequence = decode_fn(tokens[0].tolist())
            print(f"Sample 1 AFTER unmasking:  {unmasked_sequence[:100]}{'...' if len(unmasked_sequence) > 100 else ''}")

        # Step 2: Re-mask some tokens for next iteration (batch processing)
        # Stop remasking on the final iteration (iteration == iterations - 1)
        if iteration < iterations - 1:
            # Calculate how many tokens to re-mask from schedule
            if schedule == 'linear':
                remask_ratio = linear_remasking_schedule(iterations, iteration + 1)
            elif schedule == 'exponential':
                remask_ratio = exponential_remasking_schedule(iterations, iteration + 1)
            else:
                remask_ratio = 0.5  # Default

            num_to_remask = int(total_length * remask_ratio)

            if num_to_remask > 0:
                if verbose:
                    print(f"Using random remasking for batch (ratio: {remask_ratio:.3f})...")
                
                # Apply random remasking to each sample in the batch
                for sample_idx in range(batch_size):
                    # Random remasking for each sample independently
                    all_positions = torch.arange(total_length, device=device)
                    remask_indices = torch.randperm(total_length, device=device)[:num_to_remask]
                    positions_to_remask = all_positions[remask_indices]
                    tokens[sample_idx, positions_to_remask] = mask_token_id

                if verbose:
                    # Show state after remasking (first sample as example)
                    if decode_mask_fn:
                        remasked_sequence = decode_mask_fn(tokens[0].tolist())
                        print(f"Sample 1 AFTER remasking:  {remasked_sequence[:100]}{'...' if len(remasked_sequence) > 100 else ''}")
                    print(f"Re-masked {num_to_remask} tokens per sample (ratio: {remask_ratio:.2f})")
            else:
                if verbose:
                    print(f"Schedule requires 0 tokens to remask - generation converged at iteration {iteration + 1}")
                break

        if verbose:
            print("=" * 80)

    return tokens

def diffusion_generate(model, total_length, iterations, schedule='linear', mask_token_id=None, wrong_token_id=None, remask_wrong_id=None, remasking_model=None, remasking_model_type='remasking', decode_fn=None, decode_mask_fn=None, verbose=True, use_threshold_remasking=False, use_mixed_remasking=False, threshold=0.1, temperature=1.0):
    """
    Generate single text sample using diffusion-based iterative demasking (backward compatibility)
    """
    # Use batch function with batch_size=1 and return single result
    batch_results = diffusion_generate_batch(
        model, 1, total_length, iterations, schedule, mask_token_id, wrong_token_id, remask_wrong_id,
        remasking_model, remasking_model_type, decode_fn, decode_mask_fn, verbose,
        use_threshold_remasking, use_mixed_remasking, threshold, temperature
    )
    return batch_results[0]

# Set initial seed (will be modified per sample)
base_seed = seed
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load trained diffusion model
print(f"Loading model from {checkpoint_name}...")
ckpt_path = os.path.join(out_dir, checkpoint_name)
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

# Handle backward compatibility for attention_type
model_args = checkpoint['model_args']
if 'attention_type' not in model_args:
    print("Warning: attention_type not found in checkpoint, defaulting to 'causal' for backward compatibility")
    model_args['attention_type'] = 'causal'

model_config = GPTConfig(**model_args)
print(f"Model uses {model_config.attention_type} attention")
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

# Load remasking model if specified
remasking_model = None
detected_remasking_type = None
if use_intelligent_remasking and remasking_checkpoint_name is not None:
    print(f"Loading remasking model from {remasking_checkpoint_name}...")
    remasking_ckpt_path = os.path.join(out_dir, remasking_checkpoint_name)
    
    if os.path.exists(remasking_ckpt_path):
        remasking_checkpoint = torch.load(remasking_ckpt_path, map_location=device, weights_only=False)
        
        # Handle backward compatibility for attention_type
        remasking_model_args = remasking_checkpoint['model_args']
        if 'attention_type' not in remasking_model_args:
            print("Warning: attention_type not found in remasking checkpoint, defaulting to 'causal' for backward compatibility")
            remasking_model_args['attention_type'] = 'causal'
        
        remasking_model_config = GPTConfig(**remasking_model_args)
        print(f"Remasking model uses {remasking_model_config.attention_type} attention")
        remasking_model = GPT(remasking_model_config)
        
        # Load remasking model weights
        remasking_state_dict = remasking_checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(remasking_state_dict.items()):
            if k.startswith(unwanted_prefix):
                remasking_state_dict[k[len(unwanted_prefix):]] = remasking_state_dict.pop(k)
        remasking_model.load_state_dict(remasking_state_dict)
        
        remasking_model.eval()
        remasking_model.to(device)
        if compile:
            remasking_model = torch.compile(remasking_model)
        
        # Detect remasking model type
        if remasking_model_type == 'auto':
            # Check if checkpoint has training_type or infer from checkpoint name
            if 'config' in remasking_checkpoint and 'training_type' in remasking_checkpoint['config']:
                detected_remasking_type = remasking_checkpoint['config']['training_type']
            elif 'remasking_binary' in remasking_checkpoint_name:
                detected_remasking_type = 'remasking_binary'
            elif 'remasking' in remasking_checkpoint_name:
                detected_remasking_type = 'remasking'
            else:
                detected_remasking_type = 'remasking'  # Default fallback
        else:
            detected_remasking_type = remasking_model_type
            
        print(f"Remasking model loaded successfully (type: {detected_remasking_type})")
    else:
        print(f"Warning: Remasking checkpoint {remasking_ckpt_path} not found, falling back to random remasking")
        use_intelligent_remasking = False
elif use_intelligent_remasking:
    print("Warning: use_intelligent_remasking=True but no remasking_checkpoint_name specified, falling back to random remasking")
    use_intelligent_remasking = False

# Load vocabulary and setup encoding/decoding
dataset_name = None

# Try to extract dataset name from checkpoint config
if 'config' in checkpoint:
    config = checkpoint['config']
    # Handle case where config might be a complex object or dictionary
    if hasattr(config, 'get') and callable(getattr(config, 'get')):
        # config is a dictionary-like object
        dataset_name = config.get('dataset')
    elif hasattr(config, '__getitem__'):
        # config supports indexing but might not be a dict
        try:
            dataset_name = config['dataset']
        except (KeyError, TypeError):
            pass
    elif hasattr(config, 'dataset'):
        # config is an object with dataset attribute
        dataset_name = getattr(config, 'dataset', None)

# Fallback: try to infer from checkpoint filename or use default
if not dataset_name:
    print("Warning: Could not extract dataset from checkpoint config")
    if 'shakespeare' in checkpoint_name.lower():
        dataset_name = 'shakespeare_char'
        print(f"Inferred dataset from filename: {dataset_name}")
    else:
        dataset_name = 'shakespeare_char'  # Default fallback
        print(f"Using default dataset: {dataset_name}")

meta_path = os.path.join('data', dataset_name, 'meta.pkl')

if os.path.exists(meta_path):
    print(f"Loading vocabulary from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    # Setup encoding/decoding functions
    stoi, itos = meta['stoi'], meta['itos']
    vocab_size = meta['vocab_size']
    mask_token_id = vocab_size  # Mask token is next ID after vocabulary
    wrong_token_id = vocab_size + 1  # Wrong token is next ID after mask (for remasking models)
    remask_good_id = vocab_size + 2  # Good token for binary remasking models
    remask_wrong_id = vocab_size + 3  # Wrong token for binary remasking models

    def encode(text):
        return [stoi[c] for c in text]

    def decode(token_ids):
        # Handle mask tokens by replacing them with a placeholder
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
        # Decode tokens using specified mask character for masked positions
        result = []
        for token_id in token_ids:
            if token_id == mask_token_id:
                result.append(mask_char)
            elif token_id < len(itos):
                result.append(itos[token_id])
            else:
                result.append('[UNK]')
        return ''.join(result)

    print(f"Vocabulary size: {vocab_size}, mask_token_id: {mask_token_id}")
else:
    raise FileNotFoundError(f"Meta file not found: {meta_path}. Please ensure the dataset '{dataset_name}' exists in the data directory.")

# Run diffusion generation with batching
with torch.no_grad():
    with ctx:
        if num_samples == 1:
            # Single sample - use original behavior for backward compatibility
            current_seed = base_seed
            torch.manual_seed(current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(current_seed)
            
            print(f"\n{'='*80}")
            print(f"SINGLE SAMPLE GENERATION (seed: {current_seed}, temperature: {temperature})")
            print(f"{'='*80}")
            generated_tokens = diffusion_generate(
                model,
                sequence_length,
                diffusion_iterations,
                remasking_schedule,
                mask_token_id,
                wrong_token_id,
                remask_wrong_id,
                remasking_model,
                detected_remasking_type or 'remasking',
                decode,
                decode_with_mask_char,
                verbose=True,
                use_threshold_remasking=use_intelligent_remasking and not use_mixed_remasking,
                use_mixed_remasking=use_mixed_remasking,
                threshold=remasking_confidence_threshold,
                temperature=temperature
            )
            print(f"\nFINAL RESULT:")
            print(decode(generated_tokens.tolist()))
            print('='*80)
            
        else:
            # Multiple samples - use batching for efficiency
            # Set seed to ensure different results with different seeds per sample
            batch_seeds = [base_seed + k * seed_offset_multiplier for k in range(num_samples)]
            torch.manual_seed(batch_seeds[0])  # Use first seed as base
            if torch.cuda.is_available():
                torch.cuda.manual_seed(batch_seeds[0])
            
            print(f"\n{'='*80}")
            print(f"BATCH GENERATION: {num_samples} samples (temperature: {temperature})")
            print(f"Seeds: {batch_seeds}")
            print(f"{'='*80}")
            
            # Generate all samples in batch
            all_generated_tokens = diffusion_generate_batch(
                model,
                num_samples,
                sequence_length,
                diffusion_iterations,
                remasking_schedule,
                mask_token_id,
                wrong_token_id,
                remask_wrong_id,
                remasking_model,
                detected_remasking_type or 'remasking',
                decode,
                decode_with_mask_char,
                verbose=True,
                use_threshold_remasking=False,  # Disable intelligent remasking for batch mode
                use_mixed_remasking=False,
                threshold=remasking_confidence_threshold,
                temperature=temperature
            )
            
            # Display all generated samples at the end
            print(f"\n{'='*80}")
            print(f"ALL GENERATED SAMPLES")
            print(f"{'='*80}")
            
            for k in range(num_samples):
                print(f"\n{'─'*60}")
                print(f"SAMPLE {k+1}/{num_samples} (seed: {batch_seeds[k]})")
                print(f"{'─'*60}")
                sample_text = decode(all_generated_tokens[k].tolist())
                print(sample_text)
                print(f"{'─'*60}")
            
            print(f"\n{'='*80}")
            print(f"GENERATION COMPLETE - {num_samples} samples generated")
            print(f"{'='*80}")
