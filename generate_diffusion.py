import os
import torch
from model import GPTConfig, GPT
# import tiktoken # If you have a tokenizer

# --- Parameters ---
out_dir = 'out-shakespeare-char-diffusion'  # Updated to match config
device = 'cuda'
max_new_tokens = 200  # Generate more tokens for Shakespeare
num_samples = 1
temperature = 0.8
top_k = 200
seed = 1337
max_steps = 25  # Number of refinement steps

# --- Load Model ---
torch.manual_seed(seed)
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
# Remove unwanted prefix '_orig_mod.' from state_dict keys if necessary
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# --- Prepare Initial Sequence ---
mask_token_id = gptconf.mask_token_id
wrong_token_id = gptconf.wrong_token_id
assert mask_token_id is not None and wrong_token_id is not None, "Special tokens not found in model config"
start_ids = torch.full((num_samples, max_new_tokens), mask_token_id, dtype=torch.long, device=device)

# --- Try to load character-level decoder ---
import pickle
data_dir = os.path.join('data', 'shakespeare_char')
meta_path = os.path.join(data_dir, 'meta.pkl')
decode = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    decode = meta['decode']
    print(f"Loaded character decoder with vocab size {meta['vocab_size']}")

# --- Generate ---
print("Generating with diffusion model...")
with torch.no_grad():
    y = model.generate_diffusion(start_ids, max_steps, temperature=temperature, top_k=top_k)
    
    # --- Decode and Print ---
    for i in range(num_samples):
        tokens = y[i].tolist()
        print(f"\n--- Sample {i+1} ---")
        if decode is not None:
            # Character-level decoding
            try:
                generated_text = ''.join([decode[token] for token in tokens if token < len(decode)])
                print("Generated text:")
                print(repr(generated_text))  # Use repr to show special characters
                print("\nGenerated text (formatted):")
                print(generated_text)
            except (IndexError, KeyError) as e:
                print(f"Decoding error: {e}")
                print("Raw token IDs:")
                print(tokens)
        else:
            print("No decoder available. Raw token IDs:")
            print(tokens)