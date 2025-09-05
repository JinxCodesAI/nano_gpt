"""
Training logging utilities.

This module handles all training logging including:
- Console output with flushing
- Detailed timing breakdowns
- Mask statistics logging
- Entropy penalty logging
- WandB integration
- Validation timing logs
"""

import sys
import torch
import math
from typing import Optional, Dict, Any
from .entropy_utils import get_current_entropy_penalty
from .training_config import UnmaskingStageType


class TrainingLogger:
    """Handles all training logging and output."""
    
    def __init__(self, wandb_log: bool = False, master_process: bool = True, 
                 log_interval: int = 20, eval_interval: int = 200):
        """
        Initialize training logger.
        
        Args:
            wandb_log: Whether to log to WandB
            master_process: Whether this is the master process (for DDP)
            log_interval: Interval for iteration logging
            eval_interval: Interval for validation logging
        """
        self.wandb_log = wandb_log
        self.master_process = master_process
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.wandb = None
        
        if wandb_log and master_process:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not available, disabling wandb logging")
                self.wandb_log = False
    
    def print_and_flush(self, msg: str):
        """Print message and immediately flush stdout for real-time logging."""
        if self.master_process:
            print(msg)
            sys.stdout.flush()
    
    def log_iteration_stats(self, iter_num: int, loss: float, dt: float, mfu: float,
                           timer, training_ctx, Y: Optional[torch.Tensor] = None,
                           model = None, eval_only: bool = False):
        """
        Log iteration statistics including timing and training metrics.
        
        Args:
            iter_num: Current iteration number
            loss: Training loss
            dt: Time delta for this iteration
            mfu: Model flops utilization
            timer: Timer instance for detailed timing
            training_ctx: Training context
            Y: Target tensor (for sequence scoring)
            model: Model (for sequence scoring predictions)
            eval_only: Whether in eval-only mode
        """
        if iter_num % self.log_interval != 0 or not self.master_process:
            return
        
        # Calculate sequence scoring ratio for relative prediction error
        if (training_ctx.training_type == 'sequence_scoring' and 
            Y is not None and Y.dim() == 1 and model is not None):
            with torch.no_grad():
                # Get predictions from the model
                logits, _ = model(training_ctx.current_batch_X, None)
                if not hasattr(model, 'sequence_head'):
                    raise RuntimeError(
                        f"Model is missing 'sequence_head' attribute for sequence_scoring training! "
                        f"This indicates a model architecture mismatch."
                    )
                
                # For sequence scoring, logits IS already the final predictions
                predictions = logits
                
                # Calculate absolute error: abs(target - prediction)
                absolute_errors = torch.abs(Y - predictions)
                avg_absolute_error = absolute_errors.mean().item()
                
                self.print_and_flush(f"iter {iter_num}: loss {loss:.4f}, time {dt*1000:.2f}ms, mfu {mfu*100:.2f}%, ratio {avg_absolute_error:.3f}")
        else:
            self.print_and_flush(f"iter {iter_num}: loss {loss:.4f}, time {dt*1000:.2f}ms, mfu {mfu*100:.2f}%")
        
        # Log detailed timing breakdown
        self.log_timing_breakdown(timer, dt)
        
        # Log mask statistics for unmasking only
        if (training_ctx.training_type == 'unmasking' and 
            hasattr(training_ctx, 'current_batch_mask') and 
            training_ctx.current_batch_mask is not None):
            self.log_mask_statistics(training_ctx.current_batch_mask, training_ctx)
        
        # Log entropy penalty if enabled
        if training_ctx.enable_entropy_penalty:
            self.log_entropy_penalty(iter_num, training_ctx)
        
        # Log to WandB
        if self.wandb_log and not eval_only:
            log_dict = {
                "iter": iter_num,
                "train/loss": loss,
                "lr": training_ctx.current_lr if hasattr(training_ctx, 'current_lr') else 0.0,
                "mfu": mfu * 100
            }
            
            if training_ctx.enable_entropy_penalty:
                current_entropy_penalty = get_current_entropy_penalty(iter_num, training_ctx)
                log_dict["entropy_penalty"] = current_entropy_penalty
                log_dict["entropy_multiplier_ema"] = training_ctx.entropy_multiplier_ema
            
            self.log_to_wandb(log_dict)
    
    def log_timing_breakdown(self, timer, dt: float):
        """Log detailed timing breakdown."""
        # Enhanced logging with detailed timing - use recent measurements only
        data_time = timer.get_recent_average('data_generation') * 1000
        forward_time = timer.get_recent_average('forward_pass') * 1000
        backward_time = timer.get_recent_average('backward_pass') * 1000
        grad_accum_time = timer.get_recent_average('gradient_accumulation_loop') * 1000
        grad_proc_time = timer.get_recent_average('gradient_processing') * 1000
        optimizer_time = timer.get_recent_average('optimizer_operations') * 1000
        param_check_time = timer.get_recent_average('parameter_stability_check') * 1000
        cleanup_time = timer.get_recent_average('cleanup_operations') * 1000
        gpu_sync_time = timer.get_recent_average('gpu_synchronization') * 1000
        loss_proc_time = timer.get_recent_average('loss_processing') * 1000
        judge_model_time = timer.get_recent_average('judge_model_calculation') * 1000
        instability_time = timer.get_recent_average('instability_detection') * 1000
        
        # Calculate total of measured components
        measured_total = grad_accum_time + grad_proc_time + optimizer_time + param_check_time + cleanup_time + gpu_sync_time + judge_model_time
        total_time = dt * 1000
        unaccounted_time = total_time - measured_total
        
        self.print_and_flush(f"  data: {data_time:.1f}ms, grad_accum: {grad_accum_time:.1f}ms (fw: {forward_time:.1f}ms, bw: {backward_time:.1f}ms)")
        self.print_and_flush(f"  grad_proc: {grad_proc_time:.1f}ms, optimizer: {optimizer_time:.1f}ms, param_check: {param_check_time:.1f}ms")
        self.print_and_flush(f"  loss_proc: {loss_proc_time:.1f}ms, judge_model: {judge_model_time:.1f}ms, instability: {instability_time:.1f}ms")
        self.print_and_flush(f"  cleanup: {cleanup_time:.1f}ms, gpu_sync: {gpu_sync_time:.1f}ms")
        self.print_and_flush(f"  measured: {measured_total:.1f}ms, unaccounted: {unaccounted_time:.1f}ms ({unaccounted_time/total_time*100:.1f}%)")
    
    def log_mask_statistics(self, mask: torch.Tensor, training_ctx):
        """Log mask statistics for unmasking training."""
        mask_counts_per_seq = mask.sum(dim=1).cpu()
        mask_mean = mask_counts_per_seq.float().mean().item()
        mask_var = mask_counts_per_seq.float().var().item() if mask_counts_per_seq.numel() > 1 else 0.0
        mask_min = mask_counts_per_seq.min().item()
        mask_max = mask_counts_per_seq.max().item()
        self.print_and_flush(f"  mask_counts ({training_ctx.training_type}): mean={mask_mean:.6f}, var={mask_var:.6f}, min={mask_min:.6f}, max={mask_max:.6f}")
    
    def log_entropy_penalty(self, iter_num: int, training_ctx):
        """Log entropy penalty information."""
        current_entropy_penalty = get_current_entropy_penalty(iter_num, training_ctx)
        if current_entropy_penalty > 0:
            self.print_and_flush(f"  entropy_penalty: {current_entropy_penalty:.4f}, multiplier_ema: {training_ctx.entropy_multiplier_ema:.4f} (max: {training_ctx.max_entropy_penalty})")
    
    def log_validation_timing(self, timer, iter_num: int):
        """Log validation timing breakdown."""
        if iter_num % self.eval_interval == 0:
            val_time = timer.get_average('validation') * 1000
            val_data_time = timer.get_average('validation_data_generation') * 1000
            val_forward_time = timer.get_average('validation_forward_pass') * 1000
            val_loss_time = timer.get_average('validation_loss_computation') * 1000
            self.print_and_flush(f"  validation: {val_time:.2f}ms (data: {val_data_time:.2f}ms, forward: {val_forward_time:.2f}ms, loss: {val_loss_time:.2f}ms)")
    
    def log_masking_statistics(self, iter_num: int, mask: torch.Tensor, training_ctx):
        """Log masking statistics periodically."""
        stage_config = training_ctx.get_current_stage_config()
        if (stage_config and iter_num % (self.log_interval * 10) == 0 and 
            training_ctx.training_type != 'sequence_scoring'):
            mask_ratio = mask.float().mean().item()
            stage_type = stage_config.get_stage_type()
            stage_info = f"Masking: stage={training_ctx.current_stage} ({stage_type.value}), actual_ratio={mask_ratio:.3f}"
            
            if stage_type == UnmaskingStageType.STICKY:
                config = stage_config.config
                stage_info += f", target={config.target_masked_ratio:.1f}, p1={config.p1_probability:.1f}, p2={config.p2_probability:.1f}"
            elif stage_type == UnmaskingStageType.RANDOM:
                config = stage_config.config
                stage_info += f", max={config.max_masked_ratio:.1f}"
            elif stage_type == UnmaskingStageType.SPAN:
                config = stage_config.config
                stage_info += f", spans={config.spans_count}"
            
            self.print_and_flush(stage_info)
    
    def log_validation_results(self, iter_num: int, losses: Dict[str, Any], lr: float, training_ctx):
        """Log validation results with detailed information."""
        self.print_and_flush(f"\n--- Starting validation at iteration {iter_num} ---")
        
        # Print basic losses
        self.print_and_flush(f"--- Validation complete ---")
        self.print_and_flush(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}")
        
        # Print entropy penalty information if enabled
        if training_ctx.enable_entropy_penalty:
            current_entropy_penalty = get_current_entropy_penalty(iter_num, training_ctx)
            self.print_and_flush(f"entropy penalty: {current_entropy_penalty:.4f}, multiplier EMA: {training_ctx.entropy_multiplier_ema:.4f}")
        
        # Print stage information for unmasking training
        if 'current_stage' in losses:
            stage_config = training_ctx.get_current_stage_config()
            stage_type = stage_config.get_stage_type()
            stage_info = f"Stage {losses['current_stage']} ({stage_type.value}): "
            
            if stage_type == UnmaskingStageType.STICKY:
                config = stage_config.config
                stage_info += f"target_ratio={config.target_masked_ratio:.1f}, p1={config.p1_probability:.1f}, p2={config.p2_probability:.1f}"
            elif stage_type == UnmaskingStageType.RANDOM:
                config = stage_config.config
                stage_info += f"max_ratio={config.max_masked_ratio:.1f}"
            elif stage_type == UnmaskingStageType.SPAN:
                config = stage_config.config
                stage_info += f"spans_count={config.spans_count}"
            
            stage_info += f", stale_count={losses.get('val_loss_stale_count', 0)}"
            self.print_and_flush(stage_info)
        
        # Print model vs random statistics if available
        if 'val_model_vs_random' in losses:
            self.print_and_flush(f"  val model vs random: {losses['val_model_vs_random']:.2f}x better")
            self.print_and_flush(f"  val avg correct prob: {losses['val_avg_correct_prob']:.4f} (random: {1.0/training_ctx.extended_vocab_size:.4f})")
            if 'val_signal_to_noise' in losses:
                self.print_and_flush(f"  val signal to noise: {losses['val_signal_to_noise']:.2f} (median: {losses.get('val_signal_to_noise_median', 0.0):.2f})")
            if 'val_most_likely_accuracy' in losses:
                self.print_and_flush(f"  Most likely guess correct P %: {losses['val_most_likely_accuracy']:.1f}%")
        
        self.print_and_flush("")  # Add blank line for readability
    
    def log_to_wandb(self, log_dict: Dict[str, Any]):
        """Log dictionary to WandB if enabled."""
        if self.wandb_log and self.wandb is not None and self.master_process:
            self.wandb.log(log_dict)
