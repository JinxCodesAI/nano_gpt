# In a new file: evaluate_on_validation.py

import os
import torch
import pickle
import numpy as np
from contextlib import nullcontext
from model import GPTConfig, GPT
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
out_dir = 'out-shakespeare-char-diffusion'
device = 'cpu'

# --- Fragment Selection ---
# You can change these values to explore different parts of the validation set.
val_fragment_start = 1500
val_fragment_length = 200

# -----------------------------------------------------------------------------
# Model and Decoder Loading
# -----------------------------------------------------------------------------
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

# --- Load the character-level decoder ---
meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
if not os.path.exists(meta_path):
    raise FileNotFoundError("meta.pkl not found. Cannot decode the text.")

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
itos = meta['itos']
decode = lambda l: ''.join([itos[i] for i in l])

# -----------------------------------------------------------------------------
# Scoring Logic
# -----------------------------------------------------------------------------
wrong_token_id = gptconf.wrong_token_id
assert wrong_token_id is not None, "Wrong token not found in model config"

def score_sequence(sequence_ids, model_to_score_with):
    """Calculates the 'self-doubt' score for a given sequence of token IDs."""
    with torch.no_grad():
        with ctx:
            logits, _ = model_to_score_with.forward(sequence_ids)
            probs = F.softmax(logits, dim=-1)
            
            # Get the probability the model assigns to the [WRONG] token at each position
            prob_wrong = probs[:, :, wrong_token_id]
            
            # The "correctness" probability is 1 minus the "wrongness" probability
            prob_not_wrong = 1.0 - prob_wrong
            
            # Add a small epsilon for numerical stability before taking the log
            epsilon = 1e-9
            log_total_correctness = torch.log(prob_not_wrong + epsilon).sum().item()
            
            # Normalize by the length of the sequence to get a comparable score
            avg_log_correctness = log_total_correctness / sequence_ids.shape[1]
            
            return avg_log_correctness

# -----------------------------------------------------------------------------
# Main Evaluation
# -----------------------------------------------------------------------------
# 1. Load the validation data using numpy memmap
data_dir = os.path.join('data', checkpoint['config']['dataset'])
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# 2. Extract the desired fragment
start_index = val_fragment_start
end_index = start_index + val_fragment_length
if end_index > len(val_data):
    raise ValueError(f"Fragment end index {end_index} is out of bounds for validation set of size {len(val_data)}")
fragment_token_ids = torch.from_numpy(val_data[start_index:end_index].astype(np.int64))

# 3. Add a batch dimension and move to the correct device
idx = fragment_token_ids.to(device).unsqueeze(0)

# 4. Decode the fragment for human readability
human_readable_text = decode(fragment_token_ids.tolist())

# 5. Score the fragment using the trained model
validation_score = score_sequence(idx, model)

# -----------------------------------------------------------------------------
# Print Results and Analysis
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("EVALUATING MODEL CONFIDENCE ON A REAL VALIDATION FRAGMENT")
print("="*80)

print("\n--- Validation Text Fragment ---")
print(human_readable_text)
print("-" * 30)

print(f"\n--- SCORE (Avg Log Correctness): {validation_score:.4f} ---")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("This score measures the model's 'self-doubt' about the provided text.")
print("A score closer to 0.0 means the model is highly confident the text is correct.")
print("A large negative score means the model believes the text likely contains errors.")
print("\n--- Interpretation ---")
print("Compare this score to the score the model gives its own generated text (e.g., -0.0002).")
print("If this score is significantly lower (more negative), it means the model is 'narcissistic':")
print("it is more confident in its own simple, repetitive patterns than in the complex, nuanced")
print("patterns of real human writing, which it has not fully mastered.")