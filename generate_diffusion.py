import os
import torch
from model import GPTConfig, GPT
# import tiktoken # If you have a tokenizer

# --- Parameters ---
out_dir = 'out-diffusion'
device = 'cuda'
max_new_tokens = 100
num_samples = 1
temperature = 0.8
top_k = 200
seed = 1337
max_steps = 25 # Number of refinement steps

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
assert mask_token_id is not None, "Mask token not found in model config"
start_ids = torch.full((num_samples, max_new_tokens), mask_token_id, dtype=torch.long, device=device)

# --- Generate ---
print("Generating with diffusion model...")
with torch.no_grad():
    y = model.generate_diffusion(start_ids, max_steps, temperature=temperature, top_k=top_k)
    # --- Decode and Print ---
    # enc = tiktoken.get_encoding("gpt2")
    # print(enc.decode(y[0].tolist()))
    print("Generated token IDs:")
    print(y[0].tolist())