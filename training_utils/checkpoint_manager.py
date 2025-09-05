"""
Checkpoint management utilities for training.

This module handles all checkpoint-related operations including:
- Finding and loading checkpoints
- Saving checkpoints with proper naming
- Handling state dict prefix issues
- Automatic recovery from training failures
"""

import os
import glob
import torch
from typing import Optional, Dict, Any, Tuple


class CheckpointManager:
    """Manages checkpoint operations for training."""
    
    def __init__(self, out_dir: str, training_type: str, device: str):
        """
        Initialize checkpoint manager.
        
        Args:
            out_dir: Output directory for checkpoints
            training_type: Type of training (unmasking, token_classification, sequence_scoring)
            device: Device for loading checkpoints
        """
        self.out_dir = out_dir
        self.training_type = training_type
        self.device = device
    
    def _extract_iter_num(self, filename: str) -> int:
        """Extract iteration number from checkpoint filename."""
        basename = os.path.basename(filename)
        # Extract number from ckpt_{type}_{XXX}.pt
        parts = basename.split('_')
        for part in parts:
            if part.replace('.pt', '').isdigit():
                return int(part.replace('.pt', ''))
        return 0
    
    def _get_checkpoint_patterns(self) -> Tuple[str, Optional[str]]:
        """Get checkpoint patterns for current training type."""
        if self.training_type == 'unmasking':
            pattern = os.path.join(self.out_dir, 'ckpt_unmasking_*.pt')
            fallback = os.path.join(self.out_dir, 'ckpt_*unmasking*.pt')
        elif self.training_type in ['token_classification', 'remasking_binary']:
            pattern = os.path.join(self.out_dir, 'ckpt_token_classifier_*.pt')
            fallback = os.path.join(self.out_dir, 'ckpt_*remasking*.pt')
        elif self.training_type == 'sequence_scoring':
            pattern = os.path.join(self.out_dir, 'ckpt_sequence_scorer_*.pt')
            fallback = None
        else:
            pattern = os.path.join(self.out_dir, f'ckpt_{self.training_type}_*.pt')
            fallback = None
        
        return pattern, fallback
    
    def find_latest_checkpoint(self, ckpt_filename: Optional[str] = None) -> str:
        """
        Find the latest checkpoint file.
        
        Args:
            ckpt_filename: Specific checkpoint filename to use, or None for latest
            
        Returns:
            Path to checkpoint file
            
        Raises:
            FileNotFoundError: If no checkpoint found
        """
        if ckpt_filename is not None:
            ckpt_path = os.path.join(self.out_dir, ckpt_filename)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found")
            return ckpt_path
        
        # Find latest checkpoint
        pattern, fallback_pattern = self._get_checkpoint_patterns()
        ckpt_files = glob.glob(pattern)
        
        # Try fallback pattern if no files found
        if not ckpt_files and fallback_pattern:
            ckpt_files = glob.glob(fallback_pattern)
        
        if not ckpt_files:
            # Final fallback to old naming convention
            ckpt_path = os.path.join(self.out_dir, 'ckpt.pt')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"No {self.training_type} checkpoint files found in {self.out_dir}")
            return ckpt_path
        
        # Find the latest checkpoint
        latest_ckpt = max(ckpt_files, key=self._extract_iter_num)
        return latest_ckpt
    
    def _handle_state_dict_prefixes(self, state_dict: Dict[str, torch.Tensor], 
                                   target_model) -> Dict[str, torch.Tensor]:
        """Handle _orig_mod prefix issues between compiled and non-compiled models."""
        current_keys = set(target_model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        # Determine if we need to add or remove _orig_mod prefix
        if (any(k.startswith('_orig_mod.') for k in current_keys) and 
            not any(k.startswith('_orig_mod.') for k in checkpoint_keys)):
            # Current model is compiled, but checkpoint doesn't have prefix - add prefix
            print("Adding _orig_mod prefix to checkpoint keys for compiled model")
            new_state = {}
            for k, v in state_dict.items():
                new_state[f'_orig_mod.{k}'] = v
            return new_state
        elif (not any(k.startswith('_orig_mod.') for k in current_keys) and 
              any(k.startswith('_orig_mod.') for k in checkpoint_keys)):
            # Current model is not compiled, but checkpoint has prefix - remove prefix
            print("Removing _orig_mod prefix from checkpoint keys for non-compiled model")
            unwanted_prefix = '_orig_mod.'
            new_state = {}
            for k, v in state_dict.items():
                if k.startswith(unwanted_prefix):
                    new_state[k[len(unwanted_prefix):]] = v
                else:
                    new_state[k] = v
            return new_state
        
        return state_dict
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint dictionary
        """
        print(f"Loading checkpoint: {os.path.basename(checkpoint_path)}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        return checkpoint
    
    def save_checkpoint(self, model, optimizer, iter_num: int, best_val_loss: float,
                       config: Dict[str, Any], training_context: Optional[Dict[str, Any]] = None,
                       model_args: Optional[Dict[str, Any]] = None) -> str:
        """
        Save checkpoint to file.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            iter_num: Current iteration number
            best_val_loss: Best validation loss so far
            config: Training configuration
            training_context: Training context state (optional)
            model_args: Model arguments (optional)
            
        Returns:
            Path to saved checkpoint
        """
        # Get raw model (unwrap DDP if needed)
        raw_model = model.module if hasattr(model, 'module') else model
        
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': config,
        }
        
        if model_args is not None:
            checkpoint['model_args'] = model_args
            
        if training_context is not None:
            checkpoint['training_context'] = training_context
        
        # Generate checkpoint filename based on training type
        if self.training_type == 'unmasking':
            ckpt_filename = f'ckpt_unmasking_{iter_num}.pt'
        elif self.training_type in ['token_classification', 'remasking_binary']:
            ckpt_filename = f'ckpt_token_classifier_{iter_num}.pt'
        elif self.training_type == 'sequence_scoring':
            ckpt_filename = f'ckpt_sequence_scorer_{iter_num}.pt'
        else:
            ckpt_filename = f'ckpt_{self.training_type}_{iter_num}.pt'
        
        ckpt_path = os.path.join(self.out_dir, ckpt_filename)
        print(f"Saving checkpoint to {ckpt_path}")
        torch.save(checkpoint, ckpt_path)
        return ckpt_path
    
    def reload_from_latest(self, model, optimizer, training_ctx) -> bool:
        """
        Reload model and optimizer from the latest checkpoint during training.
        
        Args:
            model: Model to reload
            optimizer: Optimizer to reload
            training_ctx: Training context to update
            
        Returns:
            True if reload successful, False otherwise
        """
        print(f"\n*** RELOADING FROM CHECKPOINT ***")
        
        try:
            # Find latest checkpoint
            ckpt_path = self.find_latest_checkpoint()
            print(f"Reloading from checkpoint: {os.path.basename(ckpt_path)}")
            
            # Load checkpoint
            checkpoint = self.load_checkpoint(ckpt_path)
            
            # Get raw model (unwrap DDP/compilation if needed)
            raw_model = model.module if hasattr(model, 'module') else model
            
            # Handle state dict prefixes
            model_state = self._handle_state_dict_prefixes(checkpoint['model'], raw_model)
            
            # Reload model state
            raw_model.load_state_dict(model_state)
            
            # Reload optimizer state
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Update training context
            # Step back iteration to avoid immediately hitting the same problematic iteration
            training_ctx.iter_num = checkpoint['iter_num'] - 1
            
            # Restore training context state if available
            if 'training_context' in checkpoint:
                ctx_state = checkpoint['training_context']
                training_ctx.current_stage = ctx_state.get('current_stage', 0)
                training_ctx.val_loss_stale_count = ctx_state.get('val_loss_stale_count', 0)
                training_ctx.best_val_loss_this_stage = ctx_state.get('best_val_loss_for_stage', float('inf'))
                training_ctx.entropy_multiplier_ema = ctx_state.get('entropy_multiplier_ema', 1.0)
                print(f"Training context restored: stage={training_ctx.current_stage}, entropy_ema={training_ctx.entropy_multiplier_ema:.4f}")
            
            print(f"Model and optimizer reloaded from iteration {training_ctx.iter_num}")
            print("*** CHECKPOINT RELOAD COMPLETE ***\n")
            return True
            
        except Exception as e:
            print(f"Checkpoint reload failed: {e}")
            print("*** RECOVERY FAILED - TERMINATING TRAINING ***")
            return False
