"""Evaluate a saved diffusion checkpoint (historical GPT-2 XL config name retained)."""

# Set this to the directory whose CheckpointManager wrote `ckpt.pt`.
out_dir = 'out-your-run'

batch_size = 8
eval_iters = 500
eval_only = True
wandb_log = False

# Evaluation resumes from local checkpoints only.
init_from = 'resume'
