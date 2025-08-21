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
checkpoint_name = 'BI2_10000.pt'  # Specific checkpoint to load
remasking_checkpoint_name = None  # Optional: checkpoint for remasking model, if None uses random remasking
use_intelligent_remasking = False  # Set to True to use remasking model instead of random
num_samples = 1  # Number of samples to generate
sequence_length = 1024  # Total length of generated sequence
start_ratio = 1.0  # Start with all tokens masked
end_ratio = 0.1
seed = 1337
device = 'cpu'
dtype = 'float32'
compile = False

# Diffusion generation parameters
diffusion_iterations = 100        # Number of demasking/remasking rounds
remasking_schedule = 'linear'    # 'linear' or 'exponential'
remasking_confidence_threshold = 0.1  # Minimum confidence for remasking decisions
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

def intelligent_remask(tokens, remasking_model, num_to_remask, mask_token_id, wrong_token_id, device, verbose=False):
    """
    Use remasking model to intelligently select positions to remask
    
    Args:
        tokens: Current token sequence
        remasking_model: Trained remasking model
        num_to_remask: Number of positions to remask
        mask_token_id: ID of mask token
        wrong_token_id: ID of wrong token
        device: Device to run on
        verbose: Whether to print debug info
    """
    with torch.no_grad():
        # Get model predictions
        logits, _ = remasking_model(tokens, None)
        predicted_tokens = torch.argmax(logits, dim=-1)
        
        # Find positions where model predicts wrong_token_id
        # These are positions the model thinks should be remasked
        wrong_positions = (predicted_tokens[0] == wrong_token_id)
        
        if verbose:
            print(f"  Model identified {wrong_positions.sum().item()} positions as needing remasking")
        
        # Select positions to remask based on model's predictions
        if wrong_positions.sum() > num_to_remask:
            # Get probabilities for wrong_token_id predictions
            probs = torch.softmax(logits[0], dim=-1)
            wrong_probs = probs[:, wrong_token_id]
            
            # Select top positions by confidence
            _, top_indices = torch.topk(wrong_probs, num_to_remask)
            tokens[0, top_indices] = mask_token_id
            
            if verbose:
                print(f"  Selected top {num_to_remask} positions by confidence")
        elif wrong_positions.sum() > 0:
            # Remask all positions model thinks are wrong
            tokens[0, wrong_positions] = mask_token_id
            
            if verbose:
                print(f"  Remasked all {wrong_positions.sum().item()} identified positions")
        else:
            # Fallback to random if model doesn't identify any positions
            if verbose:
                print("  No positions identified, falling back to random remasking")
            all_positions = torch.arange(tokens.size(1), device=device)
            remask_indices = torch.randperm(tokens.size(1))[:num_to_remask]
            tokens[0, remask_indices] = mask_token_id
    
    return tokens

def diffusion_generate(model, total_length, iterations, schedule='linear', mask_token_id=None, wrong_token_id=None, remasking_model=None, decode_fn=None, decode_mask_fn=None, verbose=True):
    """
    Generate text using diffusion-based iterative demasking

    Args:
        model: Trained diffusion model
        total_length: Total length of sequence to generate
        iterations: Number of demasking/remasking iterations
        schedule: Remasking schedule ('linear' or 'exponential')
        mask_token_id: ID of the mask token
        wrong_token_id: ID of the wrong token (for remasking model)
        remasking_model: Optional trained remasking model for intelligent remasking
        decode_fn: Function to decode tokens to text (handles mask tokens)
        decode_mask_fn: Function to decode tokens with mask character
        verbose: Whether to print iteration results
    """

    # Start with ALL positions masked (pure diffusion approach)
    tokens = torch.full((1, total_length), mask_token_id, dtype=torch.long, device=device)

    if verbose:
        print(f"Starting diffusion generation with {iterations} iterations, schedule: {schedule}")
        print(f"Total length: {total_length} (all tokens start masked)")
        print("=" * 60)

    for iteration in range(iterations):
        # Show state before unmasking
        if verbose:
            masked_positions = (tokens[0] == mask_token_id)
            num_masked = masked_positions.sum().item()
            print(f"\nIteration {iteration + 1}/{iterations}")
            print(f"Tokens masked: {num_masked}/{total_length} ({num_masked/total_length*100:.1f}%)")

            # Show sequence with mask characters
            if decode_mask_fn:
                masked_sequence = decode_mask_fn(tokens[0].tolist())
                print(f"BEFORE unmasking: {masked_sequence}")

        # Step 1: Predict tokens for all masked positions
        masked_positions = (tokens[0] == mask_token_id)
        num_masked = masked_positions.sum().item()

        if num_masked > 0:
            with torch.no_grad():
                # Get logits for all positions
                dummy_targets = torch.zeros_like(tokens[0])
                logits, _ = model(tokens, dummy_targets)

            # Sample new tokens for masked positions
            mask_indices = torch.where(masked_positions)[0]
            masked_logits = logits[0, mask_indices]
            probs = torch.softmax(masked_logits, dim=-1)
            new_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Replace masked tokens with predictions
            tokens[0, mask_indices] = new_tokens

        # Show state after unmasking
        if verbose and decode_fn:
            unmasked_sequence = decode_fn(tokens[0].tolist())
            print(f"AFTER unmasking:  {unmasked_sequence}")

        # Step 2: Re-mask some tokens for next iteration (except on final iteration)
        if iteration < iterations - 1:
            # Calculate how many tokens to re-mask
            if schedule == 'linear':
                remask_ratio = linear_remasking_schedule(iterations, iteration + 1)
            elif schedule == 'exponential':
                remask_ratio = exponential_remasking_schedule(iterations, iteration + 1)
            else:
                remask_ratio = 0.5  # Default

            # Re-mask all positions (pure diffusion - no protected prompt)
            all_positions = torch.arange(total_length, device=device)
            num_to_remask = int(total_length * remask_ratio)

            if num_to_remask > 0:
                if remasking_model is not None:
                    # Use intelligent remasking
                    if verbose:
                        print(f"Using intelligent remasking model...")
                    tokens = intelligent_remask(
                        tokens, remasking_model, num_to_remask, 
                        mask_token_id, wrong_token_id, device, verbose
                    )
                else:
                    # Fallback to random remasking
                    if verbose:
                        print(f"Using random remasking...")
                    remask_indices = torch.randperm(total_length)[:num_to_remask]
                    positions_to_remask = all_positions[remask_indices]
                    tokens[0, positions_to_remask] = mask_token_id

                if verbose:
                    # Show state after remasking
                    if decode_mask_fn:
                        remasked_sequence = decode_mask_fn(tokens[0].tolist())
                        print(f"AFTER remasking:  {remasked_sequence}")
                    print(f"Re-masked {num_to_remask} tokens (ratio: {remask_ratio:.2f})")

        if verbose:
            print("=" * 80)

    return tokens[0]

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load trained diffusion model
print(f"Loading model from {checkpoint_name}...")
ckpt_path = os.path.join(out_dir, checkpoint_name)
checkpoint = torch.load(ckpt_path, map_location=device)

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
if use_intelligent_remasking and remasking_checkpoint_name is not None:
    print(f"Loading remasking model from {remasking_checkpoint_name}...")
    remasking_ckpt_path = os.path.join(out_dir, remasking_checkpoint_name)
    
    if os.path.exists(remasking_ckpt_path):
        remasking_checkpoint = torch.load(remasking_ckpt_path, map_location=device)
        
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
            
        print(f"Remasking model loaded successfully")
    else:
        print(f"Warning: Remasking checkpoint {remasking_ckpt_path} not found, falling back to random remasking")
        use_intelligent_remasking = False
elif use_intelligent_remasking:
    print("Warning: use_intelligent_remasking=True but no remasking_checkpoint_name specified, falling back to random remasking")
    use_intelligent_remasking = False

# Load vocabulary and setup encoding/decoding
if 'config' in checkpoint and 'dataset' in checkpoint['config']:
    dataset_name = checkpoint['config']['dataset']
    meta_path = os.path.join('data', dataset_name, 'meta.pkl')

    if os.path.exists(meta_path):
        print(f"Loading vocabulary from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        # Setup encoding/decoding functions
        stoi, itos = meta['stoi'], meta['itos']
        vocab_size = meta['vocab_size']
        mask_token_id = vocab_size  # Mask token is next ID after vocabulary
        wrong_token_id = vocab_size + 1  # Wrong token is next ID after mask

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
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
else:
    raise ValueError("Checkpoint does not contain dataset configuration")

# Run diffusion generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            print(f"\n{'='*80}")
            print(f"SAMPLE {k+1}/{num_samples}")
            print(f"{'='*80}")
            generated_tokens = diffusion_generate(
                model,
                sequence_length,
                diffusion_iterations,
                remasking_schedule,
                mask_token_id,
                wrong_token_id,
                remasking_model,
                decode,
                decode_with_mask_char,
                verbose=True
            )
            print(f"\nFINAL RESULT:")
            print(decode(generated_tokens.tolist()))
            print('='*80)
