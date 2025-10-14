"""Evaluate a saved diffusion checkpoint (historical GPT-2 config name retained)."""

# Point this to the training run directory that contains `ckpt.pt`
out_dir = 'out-your-run'

batch_size = 8

# Use more iterations to get a stable estimate of the loss
# when evaluating a stored checkpoint.
eval_iters = 500

eval_only = True

wandb_log = False

# Resume from the checkpoint manager in `out_dir`; GPT-2 weights are no longer supported.
init_from = 'resume'
