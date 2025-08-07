# In a new file: visualize_generation.py (or replace the old one)

import os
import torch
import pickle
import math
from contextlib import nullcontext
from model import GPTConfig, GPT
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
out_dir = 'out-shakespeare-char-diffusion'
device = 'cpu'
max_new_tokens = 200
temperature = 0.8
top_k = 200
seed = 1337

# --- Progressive Denoising Schedule ---
manual_remask_schedule = [
    0.95, 0.90, 0.85, 0.82, 0.80, 0.70, 0.60, 
    0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.00
]
max_steps = len(manual_remask_schedule)

# -----------------------------------------------------------------------------
# Model and Decoder Loading
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

mask_token_id = gptconf.mask_token_id
replace_token_id = gptconf.replace_token_id or gptconf.wrong_token_id
assert mask_token_id is not None and replace_token_id is not None, "Special tokens not found"

itos = {}
meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    itos = meta['itos']

def robust_decode(tokens, itos_map, special_tokens):
    chars = []
    for token_id in tokens:
        if token_id in special_tokens:
            chars.append(special_tokens[token_id])
        else:
            chars.append(itos_map.get(token_id, f'[UNK:{token_id}]'))
    return "".join(chars)

special_token_map = { mask_token_id: "░", replace_token_id: "[WRONG]" }

# -----------------------------------------------------------------------------
# Scoring Function (Integrated into the script)
# -----------------------------------------------------------------------------
def score_sequence(sequence_ids, model_to_score_with):
    """Calculates the 'self-doubt' score for a given sequence of token IDs."""
    with torch.no_grad():
        logits, _ = model_to_score_with.forward(sequence_ids)
        probs = F.softmax(logits, dim=-1)
        prob_wrong = probs[:, :, replace_token_id]
        prob_not_wrong = 1.0 - prob_wrong
        epsilon = 1e-9
        log_total_correctness = -torch.sum(torch.log(prob_not_wrong)).item()
        avg_log_correctness = log_total_correctness / sequence_ids.shape[1]
        return avg_log_correctness

# -----------------------------------------------------------------------------
# Main Generation and Scoring Loop
# -----------------------------------------------------------------------------
start_ids = torch.full((1, max_new_tokens), mask_token_id, dtype=torch.long, device=device)
idx = start_ids

print("="*80)
print("STARTING GENERATION VISUALIZATION AND SCORING")
print(f"Using manual schedule with {max_steps} steps.")
print("="*80)

with torch.no_grad():
    with ctx:
        # --- Step 0: Initial Burst ---
        print(f"\n--- STEP 0: INITIAL DRAFT ---")
        logits, _ = model.forward(idx)
        probs = F.softmax(logits, dim=-1)
        # This is our first full draft
        current_draft = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(1, -1)
        
        # Score this initial draft
        initial_score = score_sequence(current_draft, model)
        print(robust_decode(current_draft[0].tolist(), itos, special_token_map))
        print(f"--- SCORE (Avg Log Correctness): {initial_score:.4f} ---")
        
        # The main loop starts with this first draft
        idx = current_draft

        # --- Iterative Refinement Loop ---
        for step in range(max_steps):
            print(f"\n--- STEP {step + 1}/{max_steps}: REFINE ---")
            
            # --- Part A: The "Proofreading" Step ---
            logits, _ = model.forward(idx)
            wrong_logits = logits[:, :, replace_token_id]
            
            remask_rate = manual_remask_schedule[step]
            num_to_remask = int(max_new_tokens * remask_rate)
            
            if num_to_remask > 0:
                _, indices_to_remask = torch.topk(wrong_logits, num_to_remask, dim=-1)
                remask_mask = torch.zeros_like(idx, dtype=torch.bool)
                remask_mask.scatter_(-1, indices_to_remask, True)
                
                # --- MODIFICATION: Create the scaffold for the next step ---
                scaffold = idx.clone()
                scaffold[remask_mask] = mask_token_id
                
                print(f"Proofread (Re-mask Rate: {remask_rate:.2%}, {num_to_remask} tokens masked). Scaffold:")
                print(robust_decode(scaffold[0].tolist(), itos, special_token_map))
            else:
                print("Re-mask rate is 0. Skipping proofread.")
                scaffold = idx.clone()
                remask_mask = torch.zeros_like(idx, dtype=torch.bool)

            # --- Part B: The "Generative" Step ---
            logits, _ = model.forward(scaffold)
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, :, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(1, -1)

            # Create the new draft by filling in the blanks of the scaffold
            new_draft = torch.where(remask_mask, idx_next, idx)
            
            # --- Part C: Score the New Draft ---
            current_score = score_sequence(new_draft, model)
            
            print("Refined Draft:")
            print(robust_decode(new_draft[0].tolist(), itos, special_token_map))
            print(f"--- SCORE (Avg Log Correctness): {current_score:.4f} ---")
            
            # The new draft becomes the input for the next iteration
            idx = new_draft

print("\n" + "="*80)
print("FINAL OUTPUT")
print("="*80)
print(robust_decode(idx[0].tolist(), itos, special_token_map))
print(f"--- FINAL SCORE: {current_score:.4f} ---")