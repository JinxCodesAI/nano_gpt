"""
Model initialization utilities.

This module handles complex model initialization including:
- Model creation from scratch vs resume
- Transfer learning setup
- Unmasking model loading for sequence scoring
- Model compilation and DDP wrapping
- Model configuration validation
"""

import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Dict, Any, Tuple
from model import GPTConfig, GPT, ModelMode
from .checkpoint_manager import CheckpointManager


class ModelInitializer:
    """Handles model initialization and configuration."""
    
    def __init__(self, device: str, device_type: str):
        """
        Initialize model initializer.
        
        Args:
            device: Device for model placement
            device_type: Device type for autocast context
        """
        self.device = device
        self.device_type = device_type
    
    def create_model_from_scratch(self, model_args: Dict[str, Any], 
                                 extended_vocab_size: int) -> GPT:
        """
        Create a new model from scratch.
        
        Args:
            model_args: Model configuration arguments
            extended_vocab_size: Vocabulary size including special tokens
            
        Returns:
            Initialized GPT model
        """
        print("Initializing a new model from scratch")
        model_args['vocab_size'] = extended_vocab_size
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        return model
    
    def resume_model_from_checkpoint(self, checkpoint_manager: CheckpointManager,
                                   model_args: Dict[str, Any],
                                   ckpt_filename: Optional[str] = None) -> Tuple[GPT, int, float, Optional[Dict]]:
        """
        Resume model from checkpoint.
        
        Args:
            checkpoint_manager: CheckpointManager instance
            model_args: Model configuration arguments
            ckpt_filename: Specific checkpoint filename or None for latest
            
        Returns:
            Tuple of (model, iter_num, best_val_loss, training_context)
        """
        print(f"Resuming training from {checkpoint_manager.out_dir}")
        
        # Find and load checkpoint
        ckpt_path = checkpoint_manager.find_latest_checkpoint(ckpt_filename)
        checkpoint = checkpoint_manager.load_checkpoint(ckpt_path)
        
        checkpoint_model_args = checkpoint['model_args']
        
        # Force these config attributes to be equal otherwise we can't resume training
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        
        # Also restore use_rope setting if it exists in checkpoint
        if 'use_rope' in checkpoint_model_args:
            model_args['use_rope'] = checkpoint_model_args['use_rope']
        
        # Create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
        # Load state dict with prefix handling
        state_dict = checkpoint['model']
        state_dict = checkpoint_manager._handle_state_dict_prefixes(state_dict, model)
        model.load_state_dict(state_dict)
        
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        
        # Extract training context if available
        training_context = checkpoint.get('training_context', None)
        if training_context:
            print(f"Restoring training context state:")
            print(f"  Stage: {training_context.get('current_stage', 0)}")
            print(f"  Val loss stale count: {training_context.get('val_loss_stale_count', 0)}")
            print(f"  Best val loss for stage: {training_context.get('best_val_loss_for_stage', float('inf'))}")
        
        return model, iter_num, best_val_loss, training_context
    
    def load_unmasking_model(self, checkpoint_path: str, extended_vocab_size: int,
                           block_size: int) -> GPT:
        """
        Load unmasking model for sequence scoring.
        
        Args:
            checkpoint_path: Path to unmasking model checkpoint
            extended_vocab_size: Expected vocabulary size
            block_size: Expected block size
            
        Returns:
            Loaded unmasking model
        """
        print(f"Loading unmasking model for sequence scoring from {checkpoint_path}")
        
        # Load the unmasking model checkpoint
        unmasking_checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        unmasking_model_args = unmasking_checkpoint['model_args']
        
        # Create unmasking model with same architecture
        unmasking_gptconf = GPTConfig(**unmasking_model_args)
        unmasking_model = GPT(unmasking_gptconf)
        
        # Load the state dict
        unmasking_state_dict = unmasking_checkpoint['model']
        # Handle potential _orig_mod prefix issues
        unwanted_prefix = '_orig_mod.'
        for k, v in list(unmasking_state_dict.items()):
            if k.startswith(unwanted_prefix):
                unmasking_state_dict[k[len(unwanted_prefix):]] = unmasking_state_dict.pop(k)
        
        unmasking_model.load_state_dict(unmasking_state_dict)
        unmasking_model.to(self.device)
        unmasking_model.eval()  # Set to eval mode for inference
        
        print(f"Unmasking model loaded successfully:")
        print(f"  - Model parameters: {unmasking_model.get_num_params()/1e6:.2f}M")
        print(f"  - Vocab size: {unmasking_model_args.get('vocab_size', 'unknown')}")
        print(f"  - Block size: {unmasking_model_args.get('block_size', 'unknown')}")
        
        # Verify compatibility
        if unmasking_model_args.get('vocab_size') != extended_vocab_size:
            print(f"WARNING: Unmasking model vocab size ({unmasking_model_args.get('vocab_size')}) != current vocab size ({extended_vocab_size})")
        
        if unmasking_model_args.get('block_size', 1024) < (block_size - 1):
            print(f"WARNING: Unmasking model block size ({unmasking_model_args.get('block_size')}) < required size ({block_size - 1})")
        
        return unmasking_model
    
    def setup_transfer_learning(self, model: GPT, init_from_checkpoint: str) -> GPT:
        """
        Setup transfer learning from a pretrained checkpoint.
        
        Args:
            model: Model to initialize with transfer learning
            init_from_checkpoint: Path to pretrained checkpoint
            
        Returns:
            Model with loaded pretrained weights
        """
        if not os.path.exists(init_from_checkpoint):
            raise FileNotFoundError(f"Transfer learning checkpoint not found: {init_from_checkpoint}")
        
        print(f"Loading pretrained weights from: {init_from_checkpoint}")
        
        # Load pretrained checkpoint
        pretrained_checkpoint = torch.load(init_from_checkpoint, map_location=self.device, weights_only=False)
        pretrained_state_dict = pretrained_checkpoint['model']
        
        # Handle prefix issues
        unwanted_prefix = '_orig_mod.'
        for k, v in list(pretrained_state_dict.items()):
            if k.startswith(unwanted_prefix):
                pretrained_state_dict[k[len(unwanted_prefix):]] = pretrained_state_dict.pop(k)
        
        # Load compatible weights
        model_state_dict = model.state_dict()
        compatible_weights = {}
        
        for name, param in pretrained_state_dict.items():
            if name in model_state_dict and model_state_dict[name].shape == param.shape:
                compatible_weights[name] = param
            else:
                print(f"Skipping incompatible weight: {name}")
        
        print(f"Loaded {len(compatible_weights)}/{len(model_state_dict)} compatible weights")
        
        # Load the compatible weights
        model.load_state_dict(compatible_weights, strict=False)
        
        return model
    
    def compile_and_wrap_model(self, model: GPT, compile_flag: bool = False,
                              ddp_flag: bool = False, ddp_local_rank: Optional[int] = None) -> GPT:
        """
        Compile model and wrap in DDP if needed.
        
        Args:
            model: Model to compile/wrap
            compile_flag: Whether to compile the model
            ddp_flag: Whether to wrap in DDP
            ddp_local_rank: Local rank for DDP
            
        Returns:
            Compiled/wrapped model
        """
        # Move model to device first
        model.to(self.device)
        
        # Compile the model if requested
        if compile_flag:
            print("Compiling the model... (takes a ~minute)")
            model = torch.compile(model)  # requires PyTorch 2.0
        
        # Wrap model into DDP container if needed
        if ddp_flag:
            if ddp_local_rank is None:
                raise ValueError("ddp_local_rank must be provided when ddp_flag is True")
            model = DDP(model, device_ids=[ddp_local_rank])
        
        return model
    
    def validate_model_configuration(self, training_type: str, attention_type: str,
                                   num_token_classes: int, cls_token_id: Optional[int] = None) -> ModelMode:
        """
        Validate model configuration and return appropriate model mode.
        
        Args:
            training_type: Type of training
            attention_type: Attention mechanism type
            num_token_classes: Number of token classes
            cls_token_id: CLS token ID for sequence scoring
            
        Returns:
            Appropriate ModelMode
        """
        if training_type in ['token_classification', 'remasking_binary']:
            if attention_type != 'bidirectional':
                print("WARNING: Token classification requires bidirectional attention")
            
            print(f"Token classification mode enabled:")
            print(f"  - Attention type: {attention_type}")
            print(f"  - Number of classes: {num_token_classes}")
            
            return ModelMode.TOKEN_CLASSIFIER
        
        elif training_type == 'sequence_scoring':
            if attention_type != 'bidirectional':
                print("WARNING: Sequence scoring requires bidirectional attention")
            
            print(f"Sequence scoring mode enabled:")
            print(f"  - Attention type: {attention_type}")
            print(f"  - CLS token ID: {cls_token_id}")
            
            return ModelMode.SEQUENCE_SCORER
        
        elif training_type == 'unmasking':
            print(f"Language modeling mode enabled (unmasking)")
            return ModelMode.LANGUAGE_MODEL
        
        else:
            raise ValueError(f"Unknown training type: {training_type}")
    
    def crop_model_block_size(self, model: GPT, block_size: int, model_args: Dict[str, Any]):
        """
        Crop model block size if needed.
        
        Args:
            model: Model to crop
            block_size: Desired block size
            model_args: Model arguments to update
        """
        if block_size < model.config.block_size:
            model.crop_block_size(block_size)
            model_args['block_size'] = block_size
