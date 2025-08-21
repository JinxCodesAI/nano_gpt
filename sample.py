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
checkpoint_name = 'smooth_ckpt_5000.pt'  # Specific checkpoint to load
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

def diffusion_generate(model, total_length, iterations, schedule='linear', mask_token_id=None, decode_fn=None, decode_mask_fn=None, verbose=True):
    """
    Generate text using diffusion-based iterative demasking

    Args:
        model: Trained diffusion model
        total_length: Total length of sequence to generate
        iterations: Number of demasking/remasking iterations
        schedule: Remasking schedule ('linear' or 'exponential')
        mask_token_id: ID of the mask token
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
                # Randomly select positions to re-mask
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
                decode,
                decode_with_mask_char,
                verbose=True
            )
            print(f"\nFINAL RESULT:")
            print(decode(generated_tokens.tolist()))
            print('='*80)
