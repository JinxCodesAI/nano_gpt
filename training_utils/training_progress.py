"""
Training progress management for diffusion training.
Handles stage progression for unmasking training and learning rate scheduling.
"""

import math

from .training_config import TrainingContext, UnmaskingStageType


def update_stage_progress(training_ctx: TrainingContext, val_loss: float):
    """
    Update stage progress for unmasking training based on validation loss.
    Returns True if stage was advanced, False otherwise.
    """
    if training_ctx.training_type != 'unmasking':
        return False
    
    stage_config = training_ctx.get_current_stage_config()
    if stage_config is None:
        return False
    
    # Check if validation loss improved
    if val_loss < training_ctx.best_val_loss_this_stage:
        training_ctx.best_val_loss_this_stage = val_loss
        training_ctx.val_loss_stale_count = 0
        print(f"  Stage {training_ctx.current_stage}: New best val loss {val_loss:.4f}, reset stale count to 0")
        return False
    else:
        training_ctx.val_loss_stale_count += 1
        print(f"  Stage {training_ctx.current_stage}: Val loss stale count {training_ctx.val_loss_stale_count}/{stage_config.get_val_loss_stale_count()}")
        
        # Check if we should advance to next stage
        if training_ctx.val_loss_stale_count >= stage_config.get_val_loss_stale_count():
            advanced = training_ctx.advance_stage()
            if advanced:
                new_stage_config = training_ctx.get_current_stage_config()
                stage_type = new_stage_config.get_stage_type()
                print(f"\n*** ADVANCING TO STAGE {training_ctx.current_stage} ({stage_type.value}) ***")
                if stage_type == UnmaskingStageType.STICKY:
                    config = new_stage_config.config
                    print(f"  Target masked ratio: {config.target_masked_ratio}")
                    print(f"  P1 probability: {config.p1_probability}")
                    print(f"  P2 probability: {config.p2_probability}")
                elif stage_type == UnmaskingStageType.RANDOM:
                    config = new_stage_config.config
                    print(f"  Max masked ratio: {config.max_masked_ratio}")
                print(f"  Val loss stale count limit: {new_stage_config.get_val_loss_stale_count()}")
                print("*** STAGE ADVANCEMENT COMPLETE ***\n")
                return True
            else:
                print(f"  Stage {training_ctx.current_stage}: Reached final stage, continuing training")
                return False
        
        return False


def get_lr(it, ctx: TrainingContext):
    """Learning rate decay scheduler (cosine with warmup)"""
    # 1) linear warmup for warmup_iters steps
    if it < ctx.warmup_iters:
        return ctx.learning_rate * (it + 1) / (ctx.warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > ctx.lr_decay_iters:
        return ctx.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - ctx.warmup_iters) / (ctx.lr_decay_iters - ctx.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return ctx.min_lr + coeff * (ctx.learning_rate - ctx.min_lr)