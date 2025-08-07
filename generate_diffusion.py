#generate_diffusion.py

import os
import time
import torch
import pickle
from contextlib import nullcontext
from model import GPTConfig, GPT


def robust_decode(tokens, itos_map, special_tokens):
    """
    Decodes a list of tokens, gracefully handling special tokens and unknown tokens.
    """
    chars = []
    for token_id in tokens:
        if token_id in special_tokens:
            chars.append(special_tokens[token_id])
        else:
            # Use .get() to provide a default for unknown tokens instead of crashing
            chars.append(itos_map.get(token_id, f'[UNK:{token_id}]'))
    return "".join(chars)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
out_dir = 'out-shakespeare-char-diffusion'
device = 'cpu'
max_new_tokens = 256 # Generate a longer sequence
num_samples = 50
temperature = 0.8
top_k = 200
seed = 1337

# --- Progressive Denoising Schedule ---
max_steps = 15 # Number of refinement steps
remask_schedule = 'cosine' # 'cosine', 'linear', or 'constant'
initial_remask_rate = 0.95 # Start by re-masking 95% of the draft

special_token_map = {
    66: "[MASK]",
    67: "[WRONG]"
}

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=torch.float16)

ckpt_path = os.path.join(out_dir, 'ckpt_1754497186.054105_5000.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# --- Decoder Loading ---
meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
decode = lambda l: ''.join(l) # Default decoder if meta not found
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    itos = meta['itos']
    decode = lambda l: ''.join([itos[i] for i in l])
    print(f"Loaded character decoder with vocab size {meta['vocab_size']}")
else:
    print("No meta.pkl found, using default character decoder.")

# --- Generation ---
mask_token_id = gptconf.mask_token_id
wrong_token_id = gptconf.wrong_token_id
assert mask_token_id is not None and wrong_token_id is not None, "Special tokens not found in model config"

# Create a batch of starting sequences
start_ids = torch.full((num_samples, max_new_tokens), mask_token_id, dtype=torch.long, device=device)

print(f"Generating {num_samples} samples with progressive denoising...")
t0 = time.time()

with torch.no_grad():
    with ctx:
        # Generate all samples in parallel
        y = model.generate_diffusion(
            start_ids, 
            max_steps=max_steps, 
            temperature=temperature, 
            top_k=top_k,
            remask_schedule=remask_schedule,
            initial_remask_rate=initial_remask_rate
        )
        t1 = time.time()
        total_time = t1 - t0
        total_tokens_generated = num_samples * max_new_tokens
        tokens_per_second = total_tokens_generated / total_time
        
        # --- Decode and Print ---
        for i in range(num_samples):
            tokens = y[i].tolist()
            print(f"\n--- Sample {i+1} ---")
            generated_text = robust_decode(tokens, itos, special_token_map)
            print(generated_text)
        
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        print(f"Total tokens generated: {total_tokens_generated}")
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Tokens per second: {tokens_per_second:.2f}")
        print("="*50)
