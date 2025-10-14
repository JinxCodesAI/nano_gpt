"""Evaluate a saved diffusion checkpoint (historical GPT-2 large config name retained)."""

# Directory containing the checkpoint manager output (`ckpt.pt`).
out_dir = 'out-your-run'

batch_size = 8
eval_iters = 500
eval_only = True
wandb_log = False

# Resume from a repository-generated checkpoint; GPT-2 weights are legacy only.
init_from = 'resume'
