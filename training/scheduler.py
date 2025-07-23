"""
Training scheduler for managing operations and monitoring training progress.
"""
import time
import math
from typing import Dict, Any, List, Optional, Tuple
from .operations import execute_operation


class TrainingScheduler:
    """Manages scheduled operations during training."""
    
    def __init__(self, operations: List[Dict[str, Any]], training_logger):
        self.operations = operations
        self.training_logger = training_logger
        
        # Track operation state
        self.completed_operations = set()
        self.operation_wait_counters = {}
        self.last_loss_improvement = {}
        
        # Initialize wait counters
        for i, op in enumerate(self.operations):
            self.operation_wait_counters[i] = 0
            self.last_loss_improvement[i] = float('inf')
    
    def check_and_execute_operations(self, iter_num: int, current_val_loss: float,
                                   model, optimizer, compile_enabled: bool, ddp_enabled: bool,
                                   ddp_local_rank: int, master_process: bool, data_dir: str,
                                   weight_decay: float, learning_rate: float, beta1: float, 
                                   beta2: float, device_type: str, 
                                   target_architecture_config: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Check if any operations should be executed and execute them.
        
        Returns:
            Tuple of (updated_model, updated_optimizer, hyperparameter_updates)
        """
        hyperparameter_updates = {}
        
        for i, op in enumerate(self.operations):
            if i in self.completed_operations:
                continue
            
            # Check if operation should be triggered
            should_execute, trigger_reason = self._should_execute_operation(
                i, op, current_val_loss, iter_num
            )
            
            if should_execute:
                try:
                    # Execute the operation
                    if self._is_architectural_operation(op['name']):
                        model, optimizer = execute_operation(
                            op, trigger_reason, current_val_loss, iter_num,
                            target_architecture_config, model, optimizer,
                            compile_enabled, ddp_enabled, ddp_local_rank,
                            master_process, data_dir, weight_decay, learning_rate,
                            beta1, beta2, device_type, self.training_logger
                        )
                    else:
                        # Handle hyperparameter operations
                        hyperparameter_updates[op['name']] = op['value']
                    
                    # Mark operation as completed
                    self.completed_operations.add(i)
                    
                    if master_process:
                        print(f"✓ Operation {i+1}/{len(self.operations)} completed: {op['name']}")
                
                except Exception as e:
                    if master_process:
                        print(f"✗ Operation {i+1} failed: {op['name']} - {str(e)}")
                    # Don't mark as completed, allow retry
        
        return model, optimizer, hyperparameter_updates
    
    def _should_execute_operation(self, op_index: int, op: Dict[str, Any], 
                                current_val_loss: float, iter_num: int) -> Tuple[bool, str]:
        """Determine if an operation should be executed."""
        trigger_loss = op['trigger_loss']
        max_wait_iters = op.get('max_wait_iters', 0)
        
        # Check loss-based trigger
        loss_triggered = current_val_loss <= trigger_loss
        
        # Check time-based trigger
        self.operation_wait_counters[op_index] += 1
        time_triggered = self.operation_wait_counters[op_index] >= max_wait_iters
        
        # Determine trigger reason
        if loss_triggered and time_triggered:
            return True, "loss and time threshold"
        elif loss_triggered:
            return True, "loss threshold"
        elif time_triggered and max_wait_iters > 0:
            return True, "time threshold"
        
        return False, ""
    
    def _is_architectural_operation(self, op_name: str) -> bool:
        """Check if an operation is architectural (requires model reconstruction)."""
        architectural_ops = [
            'stack_layers', 'widen_mlp', 'set_attn_lora_rank',
            'set_embedding_lora_rank', 'merge_lora_weights',
            'resize_vocabulary', 'set_embedding_finetune_mode', 'set_embedding_freeze_mode',
            'freeze_layer', 'unfreeze_layer', 'set_layer_lora_rank'
        ]
        return op_name in architectural_ops
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of operation progress."""
        total_operations = len(self.operations)
        completed_operations = len(self.completed_operations)
        
        # Get next pending operation
        next_operation = None
        for i, op in enumerate(self.operations):
            if i not in self.completed_operations:
                next_operation = {
                    'index': i + 1,
                    'name': op['name'],
                    'trigger_loss': op['trigger_loss'],
                    'max_wait_iters': op.get('max_wait_iters', 0),
                    'wait_counter': self.operation_wait_counters[i]
                }
                break
        
        return {
            'total_operations': total_operations,
            'completed_operations': completed_operations,
            'progress_percentage': (completed_operations / total_operations * 100) if total_operations > 0 else 100,
            'next_operation': next_operation,
            'all_completed': completed_operations == total_operations
        }
    
    def reset_operation_counters(self):
        """Reset wait counters for all operations."""
        for i in range(len(self.operations)):
            self.operation_wait_counters[i] = 0


class LearningRateScheduler:
    """Manages learning rate scheduling with support for resets."""
    
    def __init__(self, learning_rate: float, warmup_iters: int, lr_decay_iters: int,
                 min_lr: float, decay_lr: bool = True):
        self.base_learning_rate = learning_rate
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.decay_lr = decay_lr
        self.schedule_offset = 0
    
    def get_lr(self, iter_num: int) -> float:
        """Get learning rate for current iteration."""
        if not self.decay_lr:
            return self.base_learning_rate
        
        # Adjust iteration based on schedule offset
        adjusted_iter = iter_num - self.schedule_offset
        
        # Linear warmup
        if adjusted_iter < self.warmup_iters:
            return self.base_learning_rate * adjusted_iter / self.warmup_iters
        
        # Past decay period
        if adjusted_iter > self.lr_decay_iters:
            return self.min_lr
        
        # Cosine decay
        decay_ratio = (adjusted_iter - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.base_learning_rate - self.min_lr)
    
    def reset_schedule(self, iter_num: int):
        """Reset the learning rate schedule from current iteration."""
        self.schedule_offset = iter_num
    
    def update_params(self, learning_rate: Optional[float] = None, 
                     warmup_iters: Optional[int] = None,
                     lr_decay_iters: Optional[int] = None,
                     min_lr: Optional[float] = None):
        """Update scheduler parameters."""
        if learning_rate is not None:
            self.base_learning_rate = learning_rate
        if warmup_iters is not None:
            self.warmup_iters = warmup_iters
        if lr_decay_iters is not None:
            self.lr_decay_iters = lr_decay_iters
        if min_lr is not None:
            self.min_lr = min_lr


class EarlyStoppingMonitor:
    """Monitor training progress and implement early stopping."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.best_weights = None
        self.wait_counter = 0
        self.stopped_epoch = -1
    
    def check(self, val_loss: float, model) -> bool:
        """
        Check if training should be stopped.
        
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.wait_counter = 0
            
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            # No improvement
            self.wait_counter += 1
        
        # Check if we should stop
        if self.wait_counter >= self.patience:
            self.stopped_epoch = self.wait_counter
            
            # Restore best weights if requested
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                print(f"Restored best weights from {self.patience} epochs ago")
            
            return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current early stopping status."""
        return {
            'best_loss': self.best_loss,
            'wait_counter': self.wait_counter,
            'patience': self.patience,
            'should_stop_soon': self.wait_counter >= self.patience - 2
        }