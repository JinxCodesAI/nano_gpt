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
device = 'cpu' # Use CPU for easier debugging if needed
max_new_tokens = 200
temperature = 0.8
top_k = 200
seed = 1337

# --- Progressive Denoising Schedule ---
# --- NEW: Confidence Threshold Schedule ---
# The number of elements determines the number of refinement steps.
# Each value is the MINIMUM confidence (P(NOT WRONG)) a token must have to survive the edit.
confidence_threshold_schedule = [
    0.01,  # Step 1: Keep any token the model is >1% sure about (very aggressive initial edit)
    0.50,  # Step 2
    0.50,  # Step 3
    0.50,  # Step 4
    0.50,  # Step 5
    0.65,  # Step 6
    0.75,  # Step 7: Getting stricter
    0.98,  # Step 8: Final polish, keep only very high-confidence tokens
]
max_steps = len(confidence_threshold_schedule)
# --- END NEW ---

# -----------------------------------------------------------------------------
# Model and Decoder Loading
# -----------------------------------------------------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=torch.float16)

ckpt_path = os.path.join(out_dir, 'ckpt_1754497186.054105_5000.pt') # Assuming a generic ckpt.pt
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
wrong_token_id = gptconf.wrong_token_id
assert mask_token_id is not None and wrong_token_id is not None, "Special tokens not found"

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

special_token_map = { mask_token_id: "░", wrong_token_id: "[WRONG]" }

# -----------------------------------------------------------------------------
# Scoring Function
# -----------------------------------------------------------------------------
def score_sequence(sequence_ids, model_to_score_with):
    with torch.no_grad():
        logits, _ = model_to_score_with.forward(sequence_ids)
        probs = F.softmax(logits, dim=-1)
        prob_wrong = probs[:, :, wrong_token_id]
        prob_not_wrong = 1.0 - prob_wrong
        epsilon = 1e-9
        # Corrected the log calculation to be negative log likelihood
        log_total_correctness = torch.log(prob_not_wrong + epsilon).sum().item()
        avg_log_correctness = log_total_correctness / sequence_ids.shape[1]
        return avg_log_correctness

# -----------------------------------------------------------------------------
# Main Generation and Scoring Loop
# -----------------------------------------------------------------------------
start_ids = torch.full((1, max_new_tokens), mask_token_id, dtype=torch.long, device=device)
idx = start_ids

print("="*80)
print("STARTING GENERATION VISUALIZATION AND SCORING")
print(f"Using confidence threshold schedule with {max_steps} steps.")
print("="*80)

with torch.no_grad():
    with ctx:
        # --- Step 0: Initial Burst ---
        print(f"\n--- STEP 0: INITIAL DRAFT ---")
        logits, _ = model.forward(idx)
        probs = F.softmax(logits, dim=-1)
        current_draft = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(1, -1)
        
        initial_score = score_sequence(current_draft, model)
        print(robust_decode(current_draft[0].tolist(), itos, special_token_map))
        print(f"--- SCORE (Avg Log Correctness): {initial_score:.4f} ---")
        
        idx = current_draft

        # --- Iterative Refinement Loop ---
        for step in range(max_steps):
            print(f"\n--- STEP {step + 1}/{max_steps}: REFINE ---")
            
            # --- Part A: The "Proofreading" Step ---
            logits, _ = model.forward(idx)
            probs = F.softmax(logits, dim=-1)
            prob_wrong = probs[:, :, wrong_token_id]
            prob_not_wrong = 1.0 - prob_wrong
            
            # --- MODIFICATION: Use the confidence threshold for this step ---
            confidence_threshold = confidence_threshold_schedule[step]
            
            # Re-mask any token where the model's confidence is BELOW the threshold.
            remask_mask = prob_not_wrong > confidence_threshold
            num_to_remask = remask_mask.sum().item()
            # --- END MODIFICATION ---
            
            if num_to_remask > 0:
                scaffold = idx.clone()
                scaffold[remask_mask] = mask_token_id
                
                print(f"Proofread (Confidence Threshold: {confidence_threshold:.2%}, {num_to_remask} tokens masked). Scaffold:")
                print(robust_decode(scaffold[0].tolist(), itos, special_token_map))
            else:
                print(f"Confidence threshold {confidence_threshold:.2%} met by all tokens. Skipping proofread.")
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

            new_draft = torch.where(remask_mask, idx_next, idx)
            
            # --- Part C: Score the New Draft ---
            current_score = score_sequence(new_draft, model)
            
            print("Refined Draft:")
            print(robust_decode(new_draft[0].tolist(), itos, special_token_map))
            print(f"--- SCORE (Avg Log Correctness): {current_score:.4f} ---")
            
            idx = new_draft

print("\n" + "="*80)
print("FINAL OUTPUT")
print("="*80)
print(robust_decode(idx[0].tolist(), itos, special_token_map))
print(f"--- FINAL SCORE: {current_score:.4f} ---")