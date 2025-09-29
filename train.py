"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import pickle
from contextlib import nullcontext


import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, ModelMode
import torch._dynamo

from dataset_consumer import DatasetConsumer
from checkpoint_manager import CheckpointManager
from loss_modifiers import create_loss_modifier_pipeline
from core.scheduler import CosineLRScheduler
from core.evaluator import Evaluator
from core.logger import create_logger
from core.training_step import TrainingStep
from core.trainer import Trainer

torch._dynamo.config.suppress_errors = True

# -----------------------------------------------------------------------------
# default config values designed for a small GPT-like model on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'run' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
target_size = None # target sequence length, defaults to block_size if None
vocab_size = None # vocab size of the tokenizer
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
attention_type = 'causal' # 'causal' for autoregressive, 'bidirectional' for BERT-style
position_encoding = 'absolute' # 'absolute' or 'rotary'
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# loss modifiers (all disabled by default for backward compatibility)
loss_modifiers_enabled = False # master switch for all loss modifiers
entropy_modifier_enabled = False # enable entropy-based loss modification
entropy_modifier_weight = 1.0 # weight factor for entropy modification
entropy_modifier_threshold = 0.0 # threshold for filtering low-entropy positions
entropy_modifier_eps = 1e-8 # small value to prevent log(0) in entropy calculation
target_smoothing_enabled = False # enable label smoothing
target_smoothing_factor = 0.1 # label smoothing factor (0.0 = no smoothing)
target_smoothing_special_tokens = [] # special token IDs to exclude from smoothing
target_smoothing_exclude_padding = True # exclude padding tokens from loss calculation
target_smoothing_padding_token = -100 # padding token ID to exclude
mask_ratio_weight_enabled = False # enable mask ratio based loss weighting
mask_ratio_weight_power = 0.5 # power for inverse square root weighting
mask_ratio_weight_min = 0.1 # minimum weight to prevent extreme values
mask_ratio_weight_max = 10.0 # maximum weight to prevent extreme values
mask_ratio_weight_eps = 1e-8 # small value to prevent division by zero
# multi-mode configuration
model_mode = 'language_model'  # 'language_model', 'token_classifier', 'sequence_scorer'
num_token_classes = 2  # For token classification
cls_token_id = None  # For sequence scoring
freeze_transformer = False  # Feature extraction mode
init_from_checkpoint = None  # Path to pretrained model
unfreeze_at_iteration = None  # Dynamic unfreezing
unfreeze_lr_multiplier = 0.1  # LR multiplier when unfreezing
seq_scorer_log_abs_rel_err = True  # running average abs(target - pred)/max(|target|, eps)
# critic head (optional, default disabled)
add_critic_head = False
critic_alpha = 0.5
critic_target_scope = 'all'

# -----------------------------------------------------------------------------
exec(open('configurator.py').read()) # overrides from command line or config file
from config.validator import validate_config
validate_config(globals())
# Recompute config_keys AFTER applying external config so new keys are included
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging

# initialize loss modifier pipeline
loss_modifier_pipeline = create_loss_modifier_pipeline(config)
# Store message for later logging after logger is initialized
enabled_modifiers_msg = None
if not loss_modifier_pipeline.is_empty():
    enabled_modifiers_msg = f"Enabled loss modifiers: {', '.join(loss_modifier_pipeline.get_enabled_modifier_names())}"
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size

# initialize logger early so all subsequent operations can use it
logger = create_logger(
    wandb_log=wandb_log,
    wandb_project=wandb_project,
    wandb_run_name=wandb_run_name,
    config=config,
    master_process=master_process,
    loss_modifier_pipeline=loss_modifier_pipeline
)

logger.log_info(f"tokens per iteration will be: {tokens_per_iter:,}")

# Log loss modifier status if any are enabled
if enabled_modifiers_msg:
    logger.log_info(enabled_modifiers_msg)

if master_process:
    os.makedirs(out_dir, exist_ok=True)
checkpoint_manager = CheckpointManager(out_dir)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loader with pre-computed batches
data_dir = os.path.join('data', dataset)

# initialize streaming data consumer (config-driven)
consumer = DatasetConsumer(
    data_dir=data_dir,
    batch_size=batch_size,
    block_size=block_size,
    target_size=target_size,
    device_type=device_type,
    prefer_queue=globals().get('data_prefer_queue', True),
    cache_files=globals().get('data_cache_files', 1),
    wait_sleep_seconds=globals().get('data_wait_sleep_seconds', 1.0),
    wait_timeout_seconds=globals().get('data_wait_timeout_seconds', None),
    verbose=globals().get('data_stream_verbose', False),
)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9


# derive vocab_size from consumer meta
meta = consumer.meta
meta_vocab_size = meta.get('vocab_size', vocab_size)
# Ensure vocab covers CLS token if provided
effective_vocab_size = meta_vocab_size
if cls_token_id is not None:
    if effective_vocab_size is None:
        effective_vocab_size = int(cls_token_id) + 1
    else:
        effective_vocab_size = max(int(effective_vocab_size), int(cls_token_id) + 1)
logger.log_info(f"found vocab_size = {meta_vocab_size} (from consumer.meta); effective_vocab_size = {effective_vocab_size}")
# attach dataset meta to config to inform checkpoint naming (contains training_type)
config['meta'] = meta

# Critic-related token ids from dataset meta (if provided)
meta_mask_token_id = meta.get('mask_token_id', None)
meta_pad_token_id = meta.get('pad_token_id', None)

# provide config to checkpoint manager early so it can resolve training_type-based paths
checkpoint_manager.set_metadata(model_args={}, config=config)


# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, attention_type=attention_type,
                  position_encoding=position_encoding,
                  # multi-mode parameters
                  mode=ModelMode(model_mode),
                  num_token_classes=num_token_classes,
                  cls_token_id=cls_token_id,
                  freeze_transformer=freeze_transformer,
                  init_from_checkpoint=init_from_checkpoint,
                  unfreeze_at_iteration=unfreeze_at_iteration,
                  unfreeze_lr_multiplier=unfreeze_lr_multiplier,
                  # critic head parameters
                  add_critic_head=add_critic_head,
                  critic_alpha=critic_alpha,
                  critic_target_scope=critic_target_scope,
                  start_critic_iteration=start_critic_iteration,
                  end_critic_iteration=end_critic_iteration,
                  mask_token_id=meta_mask_token_id,
                  pad_token_id=meta_pad_token_id) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    logger.log_info("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if effective_vocab_size is None:
        raise ValueError("vocab_size must be provided by consumer.meta or config; no default fallback")
    model_args['vocab_size'] = effective_vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, logger=logger)
elif init_from == 'resume':
    logger.log_info(f"Resuming training from {out_dir}")
    resolved_ckpt_path = checkpoint_manager.resolve_load_path()
    logger.log_info(f"Attempting to load checkpoint: {resolved_ckpt_path}")

    # resume training from a checkpoint via CheckpointManager
    checkpoint = checkpoint_manager.load(device=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, logger=logger)
    state_dict = checkpoint['model']
    checkpoint_manager.load_model_state(model, state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size  # so that the checkpoint will have the right value

checkpoint_manager.set_metadata(model_args=model_args, config=config)
model.to(device)
raw_model = model.module if ddp else model  # ensure raw model is registered
checkpoint_manager.register_model(raw_model)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
checkpoint_manager.register_optimizer(optimizer)

if init_from == 'resume':
    # If resuming past the unfreeze point, mirror optimizer param group structure
    if unfreeze_at_iteration is not None and iter_num >= unfreeze_at_iteration:
        raw_model.unfreeze_transformer_weights()
        raw_model.extend_optimizer_with_unfrozen(optimizer)
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

if compile:
    logger.log_info("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    try:
        model = torch.compile(model)
    except Exception as e:
        logger.log_warning(f"torch.compile failed ({e}); falling back to eager mode. Set compile=False to silence.")
        model = unoptimized_model

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# initialize learning rate scheduler
scheduler = CosineLRScheduler(
    learning_rate=learning_rate,
    warmup_iters=warmup_iters,
    lr_decay_iters=lr_decay_iters,
    min_lr=min_lr,
    decay_lr=decay_lr
)

# initialize evaluator
evaluator = Evaluator(
    model=model,
    consumer=consumer,
    loss_modifier_pipeline=loss_modifier_pipeline,
    eval_iters=eval_iters,
    ctx=ctx,
    device=device,
    min_zero_for_stats=globals().get('eval_zero_stats_min_zeros', 0),
    max_extra_batches_for_zero_stats=globals().get('eval_zero_stats_max_extra_batches', 0),
    reset_val_stream_each_eval=globals().get('eval_reset_val_stream', False),
)

# initialize training step handler
training_step = TrainingStep(
    model=model,
    optimizer=optimizer,
    scaler=scaler,
    gradient_accumulation_steps=gradient_accumulation_steps,
    grad_clip=grad_clip,
    ddp=ddp,
    ctx=ctx,
    scheduler=scheduler,
    unfreeze_at_iteration=unfreeze_at_iteration,
    unfreeze_lr_multiplier=unfreeze_lr_multiplier,
    logger=logger,
)


# Delegate the main training loop to the Trainer orchestrator
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    evaluator=evaluator,
    logger=logger,
    training_step=training_step,
    checkpoint_manager=checkpoint_manager,
    consumer=consumer,
    device=device,
    ddp=ddp,
    master_process=master_process,
    eval_interval=eval_interval,
    log_interval=log_interval,
    max_iters=max_iters,
    always_save_checkpoint=always_save_checkpoint,
    eval_only=eval_only,
    batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    iter_num=iter_num,
    best_val_loss=best_val_loss,
)

trainer.train()

if ddp:
    destroy_process_group()

