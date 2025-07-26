      
# Base config for fast, small-scale testing of the Training Orchestrator.
# This is NOT for achieving a good loss, but for verifying functionality.

device = 'cpu'
init_from = 'scratch'
out_dir = 'test-base'
wandb_log = False # Disable wandb for quick tests
wandb_project = 'owt-test'
wandb_run_name='orchestrator-test'

# Make the model very small for fast initialization and training
n_layer = 1
n_head = 4
n_embd = 128
n_hidden = 256 # Explicitly set for clarity
block_size = 64 # Small context size

# Make the training loop very short
max_iters = 300 # Just enough to trigger operations and see if it survives
lr_decay_iters = 300
gradient_accumulation_steps = 4
batch_size = 8

# Evaluate very frequently to check triggers quickly
eval_interval = 50
eval_iters = 10
log_interval = 5

# For debugging, it's often better to disable compile to get clearer stack traces
compile = False
scaling_schedule_file = 'configs/test_base_example.json'

    