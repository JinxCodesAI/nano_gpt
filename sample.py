"""
Sample from a trained model
"""
from typing import List, Optional

import torch
from display import DiffusionDisplay
from model_setup import ModelSetup
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
ckpt_name = 'lora_37750_overfit.pt'
start = "JHONNY:\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
encoder_start = None  # optional separate encoder conditioning text (string or "FILE:path")
num_samples = 1 # number of samples to draw
max_new_tokens = 900 # number of tokens generated in each sample
max_iterations = 20 # maximum number of diffusion iterations per sample
fix_prompt_during_diffusion = True # keep conditioning text fixed at every iteration when True
temperature = 1.5 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
seed = 43
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
insert_ratio_start   = 0.00
insert_ratio_end     = 0.00
# delete_ratio_start/end: fraction of the active region eligible for deletion early/late in the
#                         schedule. Keep these below ~0.10 to avoid deleting large spans at once.
delete_ratio_start   = 0.00
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
setup = ModelSetup(
    init_from=init_from,
    out_dir=out_dir,
    ckpt_name=ckpt_name,
    device=device,
    dtype=dtype,
    compile_model=compile,
    start=start,
    encoder_start=encoder_start,
)
model = setup.model
decode = setup.decode
space_token_id = setup.space_token_id
prompt = setup.prompt
initial_length = setup.initial_length
block_size = setup.block_size
encoder_prompt = setup.encoder_prompt
ctx = setup.autocast_context()

# run diffusion-style generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            seq_length = block_size
            max_token_pos = min(seq_length, initial_length + max_new_tokens)

            x = torch.zeros((1, seq_length), dtype=torch.long, device=device)
            x[0, :initial_length] = prompt

            g_ctx = None
            if getattr(model.config, "use_encoder_guidance", False):
                enc_x = encoder_prompt.unsqueeze(0).to(device)
                g_ctx = model.encode(enc_x)

            if temperature <= 0:
                raise ValueError("temperature must be greater than zero to perform sampling.")

            total_log_likelihood = 0.0  # Track cumulative log likelihood over diffusion steps.
            total_token_count = 0       # Track number of token evaluations for running averages.
            iteration = 0
            last_insert_indices: List[int] = []
            last_delete_indices: List[int] = []

            logits, _ = model(x, g=g_ctx)
            last_log_probs = torch.log_softmax(logits, dim=-1).detach()
            display = DiffusionDisplay(decode)

            while iteration < max_iterations:
                display.start_iteration(max_token_pos)
                active_len = max(0, max_token_pos - initial_length)
                r_ins = compute_noise_ratio(iteration, max_iterations, edit_schedule, insert_ratio_start, insert_ratio_end)
                r_del = compute_noise_ratio(iteration, max_iterations, edit_schedule, delete_ratio_start, delete_ratio_end)
                k_ins = max(0, int(r_ins * active_len))
                k_del = max(0, int(r_del * active_len))

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
                if gap_indices:
                    max_token_pos, applied_insertions = apply_insertions(
                        x,
                        gap_indices,
                        max_token_pos=max_token_pos,
                        block_size=block_size,
                        fill_id=space_token_id,
                    )
                else:
                    applied_insertions = []

                display.register_insertions(applied_insertions)

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
                pre_deletion_tokens: Optional[List[int]] = None
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

                display.capture_pre_deletion(pre_deletion_tokens, applied_deletions)
                display.register_deletions(applied_deletions)

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
                logits, _ = model(x, g=g_ctx)
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

                tokens_for_display = x[0, :max_token_pos].tolist()
                display.emit_iteration(
                    iteration=iteration,
                    sample_index=k,
                    mean_log_prob=iteration_mean_log_prob,
                    insert_count=len(applied_insertions),
                    delete_count=len(applied_deletions),
                    tokens=tokens_for_display,
                )
                last_log_probs = log_probs.detach()

                iteration += 1
                if iteration >= max_iterations:
                    break

            final_tokens = x[0, :max_token_pos].tolist()
            display.emit_final_tokens(final_tokens)
