"""
GRPO Training Script: Main entry point for Group-Relative Policy Optimization.

This script orchestrates GRPO training by:
1. Loading generator, reference, and judge models
2. Setting up optimizer and scheduler
3. Initializing data consumer
4. Running the GRPO training loop

Usage:
    python grpo/train_grpo.py config/grpo_config.py
    python grpo/train_grpo.py --group_size=16 --kl_beta=0.2
"""

import os
import sys
import time
import pickle
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPTConfig, GPT, ModelMode
from dataset_consumer import DatasetConsumer
from checkpoint_manager import CheckpointManager
from core.scheduler import CosineLRScheduler
from core.logger import create_logger

from grpo_training_step import GRPOTrainingStep
from grpo_trainer import GRPOTrainer

# -----------------------------------------------------------------------------
# Load default configuration
# -----------------------------------------------------------------------------
exec(open('grpo/grpo_config.py').read())

# Override with command line or config file
exec(open('configurator.py').read())

# Validate configuration
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list))]
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size

# Create output directory
if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Random seed
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Initialize logger
# -----------------------------------------------------------------------------

logger = create_logger(
    wandb_log=wandb_log,
    wandb_project=wandb_project,
    wandb_run_name=wandb_run_name,
    config=config,
    master_process=master_process,
    loss_modifier_pipeline=None  # Not used in GRPO
)

logger.log_info(f"GRPO Training Configuration:")
logger.log_info(f"  group_size: {group_size}")
logger.log_info(f"  kl_beta: {kl_beta}")
logger.log_info(f"  batch_size: {batch_size}")
logger.log_info(f"  gradient_accumulation_steps: {gradient_accumulation_steps}")
logger.log_info(f"  effective_batch_size: {batch_size * group_size * gradient_accumulation_steps}")
logger.log_info(f"  learning_rate: {learning_rate}")
logger.log_info(f"  max_iters: {max_iters}")
logger.log_info(f"  tokens per iteration: {tokens_per_iter:,}")

# -----------------------------------------------------------------------------
# Load models
# -----------------------------------------------------------------------------

def load_model_from_checkpoint(checkpoint_path, device, compile_model=False):
    """Load model from checkpoint."""
    logger.log_info(f"Loading model from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_args' in checkpoint:
        model_args = checkpoint['model_args']
    else:
        raise ValueError("Checkpoint missing 'model_args'")
    
    # Ensure backward compatibility
    if 'attention_type' not in model_args:
        model_args['attention_type'] = 'causal'
    if 'position_encoding' not in model_args:
        model_args['position_encoding'] = 'absolute'
    
    # Clear init_from_checkpoint to avoid chain-loading
    if 'init_from_checkpoint' in model_args:
        model_args['init_from_checkpoint'] = None
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, logger=logger)
    
    state_dict = checkpoint['model']
    
    # Remove compilation prefix if present
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    
    if compile_model:
        logger.log_info("Compiling model...")
        model = torch.compile(model)
    
    logger.log_info(f"Model loaded: {model.get_num_params()/1e6:.1f}M parameters")
    
    return model, checkpoint

# Load generator model (to be trained)
logger.log_info("Loading generator model...")
generator_model, gen_checkpoint = load_model_from_checkpoint(
    generator_checkpoint, device, compile_model=False
)
generator_model.train()

# Load reference model (frozen copy for KL penalty)
logger.log_info("Loading reference model...")
if reference_checkpoint is not None:
    reference_model, _ = load_model_from_checkpoint(
        reference_checkpoint, device, compile_model=False
    )
else:
    # Use same checkpoint as generator
    reference_model, _ = load_model_from_checkpoint(
        generator_checkpoint, device, compile_model=False
    )
reference_model.eval()
for param in reference_model.parameters():
    param.requires_grad = False
logger.log_info("Reference model frozen")

# Load judge model (frozen, provides reward signal)
logger.log_info("Loading judge model...")
judge_model, _ = load_model_from_checkpoint(
    judge_checkpoint, device, compile_model=False
)
judge_model.eval()
for param in judge_model.parameters():
    param.requires_grad = False
logger.log_info("Judge model frozen")

# Verify judge is a sequence scorer
if getattr(judge_model.config, 'mode', None) != ModelMode.SEQUENCE_SCORER:
    raise ValueError("Judge model must be configured with mode=SEQUENCE_SCORER")

# -----------------------------------------------------------------------------
# Initialize data consumer
# -----------------------------------------------------------------------------

data_dir = os.path.join('data', dataset)

consumer = DatasetConsumer(
    data_dir=data_dir,
    batch_size=batch_size,
    block_size=block_size,
    target_size=target_size,
    device_type=device_type,
    prefer_queue=data_prefer_queue,
    cache_files=data_cache_files,
    wait_sleep_seconds=data_wait_sleep_seconds,
    wait_timeout_seconds=data_wait_timeout_seconds,
    verbose=data_stream_verbose,
)

# Get metadata
meta = consumer.meta
meta_vocab_size = meta.get('vocab_size')
mask_token_id = meta.get('mask_token_id')
pad_token_id = meta.get('pad_token_id')
base_vocab_size = meta.get('base_vocab_size')

logger.log_info(f"Dataset metadata:")
logger.log_info(f"  vocab_size: {meta_vocab_size}")
logger.log_info(f"  mask_token_id: {mask_token_id}")
logger.log_info(f"  pad_token_id: {pad_token_id}")
logger.log_info(f"  base_vocab_size: {base_vocab_size}")

# -----------------------------------------------------------------------------
# Initialize optimizer and scheduler
# -----------------------------------------------------------------------------

optimizer = generator_model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)

scheduler = CosineLRScheduler(
    learning_rate=learning_rate,
    warmup_iters=warmup_iters,
    lr_decay_iters=lr_decay_iters,
    min_lr=min_lr,
    decay_lr=True
)

# Gradient scaler for mixed precision
# Try new API first, fall back to old API for compatibility
try:
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
except TypeError:
    # Fallback for older PyTorch versions
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# -----------------------------------------------------------------------------
# Initialize checkpoint manager
# -----------------------------------------------------------------------------

checkpoint_manager = CheckpointManager(out_dir)
model_args = gen_checkpoint.get('model_args', {})
checkpoint_manager.set_metadata(model_args=model_args, config=config)
checkpoint_manager.register_model(generator_model)
checkpoint_manager.register_optimizer(optimizer)

# -----------------------------------------------------------------------------
# Initialize GRPO training step
# -----------------------------------------------------------------------------

grpo_config = {
    'group_size': group_size,
    'kl_beta': kl_beta,
    'grad_clip': grad_clip,
    'gradient_accumulation_steps': gradient_accumulation_steps,
    'mask_token_id': mask_token_id,
    'pad_token_id': pad_token_id,
    'base_vocab_size': base_vocab_size,
    'vocab_size': meta_vocab_size,
    'temperature': temperature,
    'top_p': top_p,
    'device': device,
}

grpo_step = GRPOTrainingStep(
    generator_model=generator_model,
    reference_model=reference_model,
    judge_model=judge_model,
    optimizer=optimizer,
    scaler=scaler,
    config=grpo_config,
    ctx=ctx,
    ddp=ddp,
)

# -----------------------------------------------------------------------------
# Initialize GRPO trainer
# -----------------------------------------------------------------------------

trainer = GRPOTrainer(
    generator_model=generator_model,
    reference_model=reference_model,
    judge_model=judge_model,
    optimizer=optimizer,
    scheduler=scheduler,
    training_step=grpo_step,
    consumer=consumer,
    checkpoint_manager=checkpoint_manager,
    logger=logger,
    device=device,
    ddp=ddp,
    master_process=master_process,
    log_interval=log_interval,
    save_interval=save_interval,
    sample_interval=sample_interval,
    max_iters=max_iters,
    batch_size=batch_size,
    iter_num=0,
)

# -----------------------------------------------------------------------------
# Start training
# -----------------------------------------------------------------------------

logger.log_info("Starting GRPO training...")
trainer.train()

if ddp:
    destroy_process_group()

logger.log_info("GRPO training complete!")

