"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
from typing import List, Optional

import torch
import tiktoken
from model import GPTConfig, GPT
from sampling_utils import (
    apply_deletions,
    apply_insertions,
    apply_re_noise,
    build_cooldown_mask,
    compute_noise_ratio,
    deletion_scores_from_probs,
    score_gaps_for_insertion,
    select_topk_mask,
    uncertainty_from_logprobs,
)

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-char-random-replacement' # ignored if init_from is not 'resume'
ckpt_name = 'new_hope_2_9000.pt'
start = "\ROMEO:\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 900 # number of tokens generated in each sample
max_iterations = 20 # maximum number of diffusion iterations per sample
fix_prompt_during_diffusion = True # keep conditioning text fixed at every iteration when True
temperature = 1.5 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
seed = 42
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

# --- Re-noise schedule knobs ---
noise_schedule = 'cosine'   # 'linear' or 'cosine' (anything else -> no re-noise)
noise_start    = 0.0       # fraction of positions to randomize at iter 0 (e.g., 0.05-0.30)
noise_end      = 0.0       # fraction at the last iteration (usually 0.0)

# --- Structural edit knobs ---
# edit_schedule: interpolation curve for mapping iteration -> edit ratios. Accepts "linear" or
#                "cosine", defaults to cosine decay. Any other value disables edits.
edit_schedule        = 'cosine'
# insert_ratio_start/end: fraction of the currently active (non-prompt) length that we try to fill
#                         with new insertions at the beginning/end of diffusion. Values in the
#                         0.00–0.10 range tend to preserve coherence without overwhelming updates.
insert_ratio_start   = 0.2
insert_ratio_end     = 0.00
# delete_ratio_start/end: fraction of the active region eligible for deletion early/late in the
#                         schedule. Keep these below ~0.10 to avoid deleting large spans at once.
delete_ratio_start   = 0.2
delete_ratio_end     = 0.00
# delete_margin: probability buffer applied before considering a removal beneficial. Larger margins
#                make deletions rarer; values around 0.01–0.05 generally work well.
delete_margin        = 0.02
# delete_lambda: weight for the second-order deletion heuristic that looks one token ahead. Raising
#                this emphasizes deletions that also improve the following token. Typical range is
#                0.1–0.5.
delete_lambda        = 0.30
# length_target_mode: optional policy that nudges the sequence toward a target length. "none" leaves
#                     lengths unconstrained, while "to_max_new" gradually pushes toward
#                     initial_length + max_new_tokens.
length_target_mode   = 'none'
# cooldown_distance: radius of edit suppression in tokens. Recently edited positions are skipped
#                    for this many steps on each side to avoid thrashing. 0 disables cooldown.
cooldown_distance    = 5

exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    if ' ' not in stoi:
        raise ValueError("Space character not found in dataset vocabulary; cannot perform space-based re-noising.")
    space_token_id = stoi[' ']
    _encode_unknown_cache = set()

    def encode(s: str):
        missing = set()
        ids = []
        for c in s:
            idx = stoi.get(c)
            if idx is None:
                missing.add(c)
                idx = space_token_id
            ids.append(idx)
        if missing:
            unseen = missing.difference(_encode_unknown_cache)
            if unseen:
                printable = ", ".join(repr(ch) for ch in sorted(unseen))
                print(f"Warning: substituting space for unknown characters: {printable}")
                _encode_unknown_cache.update(unseen)
        return ids

    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    space_token_ids = encode(" ")
    if not space_token_ids:
        raise ValueError("Encoder did not return a token id for a single space character.")
    if len(space_token_ids) != 1:
        raise ValueError(f"Expected a single token id for space, got: {space_token_ids}")
    space_token_id = space_token_ids[0]

GREEN = "\033[92m"
ORANGE = "\033[38;5;208m"
RED_BACKGROUND = "\033[41m"
WHITE_BACKGROUND = "\033[47m"
RESET = "\033[0m"

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
prompt = torch.tensor(start_ids, dtype=torch.long, device=device)
initial_length = prompt.size(0)
block_size = model.config.block_size
if initial_length > block_size:
    raise ValueError(f"Prompt is longer ({initial_length}) than model block size ({block_size}).")

# run diffusion-style generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            seq_length = block_size
            max_token_pos = min(seq_length, initial_length + max_new_tokens)

            x = torch.zeros((1, seq_length), dtype=torch.long, device=device)
            x[0, :initial_length] = prompt

            if temperature <= 0:
                raise ValueError("temperature must be greater than zero to perform sampling.")

            prev_decoded = None
            total_log_likelihood = 0.0  # Track cumulative log likelihood over diffusion steps.
            total_token_count = 0       # Track number of token evaluations for running averages.
            iteration = 0
            last_insert_indices: List[int] = []
            last_delete_indices: List[int] = []

            logits, _ = model(x)
            last_log_probs = torch.log_softmax(logits, dim=-1).detach()

            while iteration < max_iterations:
                active_len = max(0, max_token_pos - initial_length)
                r_ins = compute_noise_ratio(iteration, max_iterations, edit_schedule, insert_ratio_start, insert_ratio_end)
                r_del = compute_noise_ratio(iteration, max_iterations, edit_schedule, delete_ratio_start, delete_ratio_end)
                k_ins = max(0, int(r_ins * active_len))
                k_del = max(0, int(r_del * active_len))
                token_index_map: List[Optional[int]] = list(range(max_token_pos))

                if length_target_mode == 'to_max_new':
                    target_len = min(block_size, initial_length + max_new_tokens)
                    error = max_token_pos - target_len
                    if error > 0:
                        k_del = min(active_len, k_del + min(3, error))
                    elif error < 0:
                        extra = min(3, -error)
                        capacity = max(0, block_size - max_token_pos)
                        k_ins = min(k_ins + extra, capacity + active_len)

                # Insertion scoring and application
                u = uncertainty_from_logprobs(last_log_probs[:, :max_token_pos, :], x[:, :max_token_pos])
                gap_scores = score_gaps_for_insertion(u)
                allowed_gap_count = max(0, gap_scores.numel() - initial_length)
                k_ins = min(k_ins, allowed_gap_count)
                cooldown_mask_gaps = build_cooldown_mask(
                    gap_scores.numel(), last_insert_indices, cooldown_distance, gap_scores.device
                )
                sel_gaps = select_topk_mask(
                    gap_scores,
                    k_ins,
                    forbid_lo=0,
                    forbid_hi=initial_length,
                    additional_forbid=cooldown_mask_gaps,
                )
                gap_indices = torch.nonzero(sel_gaps, as_tuple=False).flatten().tolist()
                insertion_positions_pre_deletion: List[int] = []
                if gap_indices:
                    max_token_pos, applied_insertions = apply_insertions(
                        x,
                        gap_indices,
                        max_token_pos=max_token_pos,
                        block_size=block_size,
                        fill_id=space_token_id,
                    )
                    offset = 0
                    for idx in applied_insertions:
                        insertion_positions_pre_deletion.append(idx + offset)
                        offset += 1
                else:
                    applied_insertions = []
                    insertion_positions_pre_deletion = []

                if insertion_positions_pre_deletion:
                    for pos in insertion_positions_pre_deletion:
                        if 0 <= pos <= len(token_index_map):
                            token_index_map.insert(pos, None)

                # Deletion scoring and application
                last_probs = last_log_probs.exp()
                del_scores = deletion_scores_from_probs(
                    last_probs[:, :max_token_pos, :],
                    x[:, :max_token_pos],
                    margin=delete_margin,
                    lam=delete_lambda,
                )
                allowed_delete_count = max(0, del_scores.numel() - initial_length)
                k_del = min(k_del, allowed_delete_count)
                cooldown_mask_del = build_cooldown_mask(
                    del_scores.numel(), last_delete_indices, cooldown_distance, del_scores.device
                )
                sel_del = select_topk_mask(
                    del_scores,
                    k_del,
                    forbid_lo=0,
                    forbid_hi=initial_length,
                    additional_forbid=cooldown_mask_del,
                )
                del_indices = torch.nonzero(sel_del, as_tuple=False).flatten().tolist()
                pre_deletion_tokens = None
                if del_indices:
                    pre_deletion_tokens = x[0, :max_token_pos].tolist()
                    max_token_pos, applied_deletions = apply_deletions(
                        x,
                        del_indices,
                        max_token_pos=max_token_pos,
                        initial_length=initial_length,
                        fill_id=space_token_id,
                        prior_insertions=applied_insertions,
                    )
                else:
                    applied_deletions = []

                if applied_deletions:
                    for del_idx in sorted(applied_deletions, reverse=True):
                        if 0 <= del_idx < len(token_index_map):
                            token_index_map.pop(del_idx)

                last_insert_indices = applied_insertions
                last_delete_indices = applied_deletions

                if fix_prompt_during_diffusion:
                    x[0, :initial_length] = prompt

                beta_s = compute_noise_ratio(iteration, max_iterations, noise_schedule, noise_start, noise_end)
                x = apply_re_noise(
                    x=x,
                    max_token_pos=max_token_pos,
                    initial_length=initial_length,
                    replace_character_id=space_token_id,
                    ratio=beta_s,
                    device=device,
                )
                logits, _ = model(x)
                logits = logits / temperature

                # Convert logits to log-probabilities for every token position, then exponentiate
                # to obtain probabilities for sampling.
                log_probs = torch.log_softmax(logits, dim=-1)
                probs = log_probs.exp()

                # Slice out both probability and log-probability distributions for the active portion
                # of the sequence. Cast to float32 for stable multinomial sampling and log accumulation.
                active_log_probs = log_probs[0, :max_token_pos, :].to(dtype=torch.float)
                active_probs = probs[0, :max_token_pos, :].to(dtype=torch.float)

                # Draw a token for every active position via multinomial sampling. We first collapse the
                # sequence dimension so multinomial can treat each position independently, then restore
                # the original shape to align with the sequence layout. Accumulate the log likelihood of
                # the selected tokens to report overall generation confidence.
                flat_probs = active_probs.view(-1, active_probs.size(-1))
                sampled_indices = torch.multinomial(flat_probs, 1)

                sampled = sampled_indices.view(1, -1)

                # Write the sampled tokens back into the working sequence window, overwriting any previous
                # proposals. Optionally restore the original prompt tokens so the conditioning text stays
                # fixed throughout diffusion updates when the flag is enabled.
                x[:, :max_token_pos] = sampled
                if fix_prompt_during_diffusion:
                    x[0, :initial_length] = prompt
                    sampled[:, :initial_length] = prompt

                # Evaluate log probabilities for the tokens that remain active after the optional prompt fix.
                current_tokens = sampled[0, :max_token_pos].unsqueeze(-1)
                iteration_log_probs = active_log_probs.gather(-1, current_tokens).squeeze(-1)
                iteration_mean_log_prob = iteration_log_probs.mean().item()

                # Zero out any positions beyond the active window so that the next iteration continues to
                # treat them as padding (i.e., not part of the diffusion process yet).
                if max_token_pos < seq_length:
                    x[:, max_token_pos:] = 0

                decoded = decode(x[0, :max_token_pos].tolist())

                deletion_display = None
                if applied_deletions and pre_deletion_tokens:
                    pre_deletion_decoded = decode(pre_deletion_tokens)
                    deletion_set = set(applied_deletions)
                    highlighted_pre_deletion = []
                    for idx, char in enumerate(pre_deletion_decoded):
                        if idx in deletion_set:
                            highlighted_pre_deletion.append(f"{RED_BACKGROUND}{char}{RESET}")
                        else:
                            highlighted_pre_deletion.append(char)
                    deletion_display = "".join(highlighted_pre_deletion)

                colored_chars = []
                green_count = 0
                orange_count = 0
                for idx, char in enumerate(decoded):
                    source_idx = token_index_map[idx] if idx < len(token_index_map) else None
                    if source_idx is None:
                        base_color = ORANGE
                        orange_count += 1
                        colored_chars.append(f"{WHITE_BACKGROUND}{base_color}{char}{RESET}")
                        continue
                    same_as_prev = (
                        prev_decoded is not None
                        and source_idx < len(prev_decoded)
                        and char == prev_decoded[source_idx]
                    )
                    if same_as_prev:
                        base_color = GREEN
                        green_count += 1
                    else:
                        base_color = ORANGE
                        orange_count += 1

                    colored_chars.append(f"{base_color}{char}{RESET}")

                print(
                    f"Iteration {iteration}, sample {k} | "
                    f"mean log prob: {iteration_mean_log_prob:.4f} | "
                    f"changed {orange_count}/{green_count+orange_count} | "
                    f"ins {len(applied_insertions)} | del {len(applied_deletions)}"
                )
                if deletion_display is not None:
                    print(f"Before deletion: {deletion_display}")
                print("".join(colored_chars))
                print("==========================================================================================================================")
                prev_decoded = decoded

                last_log_probs = log_probs.detach()

                iteration += 1
                if iteration >= max_iterations:
                    break

            final_tokens = x[0, :max_token_pos].tolist()
            print(decode(final_tokens))
            print('---------------')
