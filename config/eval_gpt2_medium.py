"""Evaluate a saved diffusion checkpoint (historical GPT-2 medium config name retained)."""

# Update to the directory holding the checkpoint you wish to evaluate.
out_dir = 'out-your-run'

batch_size = 8

eval_iters = 500
eval_only = True
wandb_log = False

# All evaluation now resumes from repository-produced checkpoints.
init_from = 'resume'
