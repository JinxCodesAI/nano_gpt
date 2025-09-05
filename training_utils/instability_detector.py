"""
Training instability detection and recovery utilities.

This module handles all forms of training instability detection:
- Logits containing NaN/Inf values
- Loss becoming NaN/Inf
- Gradient instability (NaN/Inf gradients)
- Parameter corruption (NaN/Inf parameters)
- Automatic recovery via checkpoint reloading
"""

import torch
import math
from typing import Optional, Tuple, Any
from .checkpoint_manager import CheckpointManager


class InstabilityDetector:
    """Detects and handles training instabilities with automatic recovery."""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        """
        Initialize instability detector.
        
        Args:
            checkpoint_manager: CheckpointManager instance for recovery
        """
        self.checkpoint_manager = checkpoint_manager
    
    def check_logits_stability(self, logits: torch.Tensor, iter_num: int) -> bool:
        """
        Check if logits contain NaN or Inf values.
        
        Args:
            logits: Model logits to check
            iter_num: Current iteration number
            
        Returns:
            True if stable, False if unstable
        """
        if not torch.isfinite(logits).all():
            print(f"\n*** INSTABILITY DETECTED at iter {iter_num} ***")
            print(f"Logits contain NaN/Inf: {torch.isnan(logits).sum().item()} NaN, {torch.isinf(logits).sum().item()} Inf")
            print(f"Logits stats: min={logits.min().item():.6f}, max={logits.max().item():.6f}, mean={logits.mean().item():.6f}")
            return False
        return True
    
    def check_loss_stability(self, loss: torch.Tensor, iter_num: int, 
                           loss_type: str = "training") -> bool:
        """
        Check if loss is finite.
        
        Args:
            loss: Loss tensor to check
            iter_num: Current iteration number
            loss_type: Type of loss (training, validation, final)
            
        Returns:
            True if stable, False if unstable
        """
        if not torch.isfinite(loss):
            if loss_type == "validation":
                print(f"\n*** VALIDATION INSTABILITY at iter {iter_num} ***")
                print(f"Val loss is {loss.item()}: {'NaN' if torch.isnan(loss) else 'Inf'}")
                print("NaN detected in validation - model has become unstable")
            elif loss_type == "final":
                print(f"\n*** FINAL LOSS INSTABILITY at iter {iter_num} ***")
                print(f"Final loss is {loss.item()}: {'NaN' if torch.isnan(loss) else 'Inf'}")
            else:
                print(f"\n*** LOSS INSTABILITY at iter {iter_num} ***")
                print(f"Loss is {loss.item()}: {'NaN' if torch.isnan(loss) else 'Inf'}")
            return False
        return True
    
    def check_gradient_stability(self, model, grad_norm: Optional[torch.Tensor], 
                               iter_num: int, grad_clip: float = 0.0) -> bool:
        """
        Check gradient stability.
        
        Args:
            model: Model to check gradients for
            grad_norm: Gradient norm (if clipping enabled)
            iter_num: Current iteration number
            grad_clip: Gradient clipping threshold
            
        Returns:
            True if stable, False if unstable
        """
        if grad_clip != 0.0 and grad_norm is not None:
            # Check for true instability (NaN/Inf gradients)
            if not torch.isfinite(grad_norm):
                if iter_num == 0:
                    print(f"\n*** INITIALIZATION PROBLEM at iter {iter_num} ***")
                    print(f"Gradient norm is {grad_norm.item()}: {'NaN' if torch.isnan(grad_norm) else 'Inf'}")
                    print("This suggests model initialization or loss computation issues")
                    
                    # Check a few key statistics
                    print("\nModel parameter stats:")
                    for name, param in list(model.named_parameters())[:3]:  # First 3 params
                        print(f"  {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
                        if param.grad is not None:
                            print(f"    grad: mean={param.grad.data.mean().item():.6f}, std={param.grad.data.std().item():.6f}")
                else:
                    print(f"\n*** GRADIENT INSTABILITY at iter {iter_num} ***")
                    print(f"Gradient norm is {grad_norm.item()}: {'NaN' if torch.isnan(grad_norm) else 'Inf'}")
                
                # Check individual parameter gradients
                nan_params = 0
                inf_params = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            nan_params += 1
                        if torch.isinf(param.grad).any():
                            inf_params += 1
                print(f"Parameters with NaN gradients: {nan_params}, with Inf gradients: {inf_params}")
                return False
            
            # Only warn about large gradients after initial iterations (when lr > 0)
            if iter_num > 10 and grad_norm > grad_clip * 10:
                print(f"WARNING: Large gradient norm at iter {iter_num}: {grad_norm.item():.4f} (clip threshold: {grad_clip})")
        
        else:
            # Check gradient norms even without clipping
            total_norm = 0.0
            nan_grads = False
            inf_grads = False
            
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    if torch.isnan(param_norm):
                        nan_grads = True
                    if torch.isinf(param_norm):
                        inf_grads = True
                    total_norm += param_norm.item() ** 2
            
            total_norm = total_norm ** (1. / 2)
            
            if nan_grads or inf_grads:
                print(f"\n*** GRADIENT INSTABILITY at iter {iter_num} (no clipping) ***")
                print(f"NaN gradients: {nan_grads}, Inf gradients: {inf_grads}")
                print(f"Total gradient norm: {total_norm:.6f}")
                return False
        
        return True
    
    def check_parameter_stability(self, model, iter_num: int) -> bool:
        """
        Check if model parameters contain NaN or Inf values.
        
        Args:
            model: Model to check
            iter_num: Current iteration number
            
        Returns:
            True if stable, False if unstable
        """
        nan_params = 0
        inf_params = 0
        param_names_with_issues = []
        
        for name, param in model.named_parameters():
            if param.data is not None:
                if torch.isnan(param.data).any():
                    nan_params += 1
                    param_names_with_issues.append(f"{name}(NaN)")
                if torch.isinf(param.data).any():
                    inf_params += 1
                    param_names_with_issues.append(f"{name}(Inf)")
        
        if nan_params > 0 or inf_params > 0:
            print(f"\n*** PARAMETER INSTABILITY at iter {iter_num} ***")
            print(f"Parameters with NaN values: {nan_params}, with Inf values: {inf_params}")
            print(f"Affected parameters: {param_names_with_issues[:10]}")  # Show first 10
            if len(param_names_with_issues) > 10:
                print(f"... and {len(param_names_with_issues) - 10} more")
            return False
        
        return True
    
    def check_validation_stability(self, losses: dict, iter_num: int) -> bool:
        """
        Check validation loss stability.
        
        Args:
            losses: Dictionary containing train/val losses
            iter_num: Current iteration number
            
        Returns:
            True if stable, False if unstable
        """
        train_loss_finite = math.isfinite(losses['train'])
        val_loss_finite = math.isfinite(losses['val'])
        
        if not train_loss_finite or not val_loss_finite:
            print(f"\n*** VALIDATION INSTABILITY at iter {iter_num} ***")
            print(f"Train loss: {losses['train']} ({'finite' if train_loss_finite else 'NaN/Inf'})")
            print(f"Val loss: {losses['val']} ({'finite' if val_loss_finite else 'NaN/Inf'})")
            print("NaN detected in validation - model has become unstable")
            print("*** TERMINATING TRAINING ***")
            return False
        
        return True
    
    def attempt_recovery(self, model, optimizer, training_ctx, scaler, 
                        get_batch_func) -> Tuple[bool, Any, Any, Any]:
        """
        Attempt to recover from instability by reloading from checkpoint.
        
        Args:
            model: Model to recover
            optimizer: Optimizer to recover
            training_ctx: Training context to update
            scaler: Gradient scaler to reset
            get_batch_func: Function to get new batch
            
        Returns:
            Tuple of (success, new_X, new_Y, new_mask) or (False, None, None, None)
        """
        print("*** ATTEMPTING RECOVERY FROM CHECKPOINT ***")
        
        if self.checkpoint_manager.reload_from_latest(model, optimizer, training_ctx):
            # Generate new batch to avoid same problematic data
            X, Y, mask = get_batch_func('train', training_ctx)
            
            # Reset scaler state
            scaler = torch.cuda.amp.GradScaler(enabled=scaler.is_enabled())
            optimizer.zero_grad(set_to_none=True)
            
            return True, X, Y, mask
        else:
            return False, None, None, None
