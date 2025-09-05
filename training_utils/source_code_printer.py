"""
Source code and configuration printing utilities.

This module handles printing of source code and global variables for reproducibility.
Useful for research and debugging to capture the exact state of the codebase.
"""

import sys
import os
from typing import Dict, Any, List


class SourceCodePrinter:
    """Handles printing of source code and global variables for reproducibility."""
    
    @staticmethod
    def get_local_python_files() -> List[str]:
        """
        Get all local Python files that are imported.
        
        Returns:
            List of local Python filenames
        """
        local_files = set()
        
        # Get all local Python files that are imported
        for module_name, module in sys.modules.items():
            if hasattr(module, '__file__') and module.__file__:
                file_path = module.__file__
                # Only include .py files in current directory (not packages/libraries)
                if file_path.endswith('.py') and os.path.dirname(file_path) == os.getcwd():
                    local_files.add(os.path.basename(file_path))
        
        # Always include the main script
        local_files.add('train_run.py')
        local_files.add('train_run2.py')  # Include the refactored version too
        
        # Convert to sorted list for consistent output
        return sorted(local_files)
    
    @staticmethod
    def print_file_contents(filenames: List[str]):
        """
        Print contents of specified files.
        
        Args:
            filenames: List of filenames to print
        """
        for filename in filenames:
            print(f"\n--- {filename} ---")
            try:
                with open(filename, 'r') as f:
                    print(f.read())
            except FileNotFoundError:
                print(f"File {filename} not found")
    
    @staticmethod
    def print_global_variables(globals_dict: Dict[str, Any]):
        """
        Print global variables for reproducibility.
        
        Args:
            globals_dict: Dictionary of global variables
        """
        print("\n" + "=" * 80)
        print("GLOBAL VARIABLES:")
        print("=" * 80)
        
        for name, value in sorted(globals_dict.items()):
            if not name.startswith('_') and not callable(value):
                print(f"{name} = {value}")
        
        print("\n" + "=" * 80)
    
    @staticmethod
    def print_source_code_and_globals(globals_dict: Dict[str, Any]):
        """
        Print source code and global variables for full reproducibility.
        
        Args:
            globals_dict: Dictionary of global variables from the main script
        """
        print("=" * 80)
        print("SOURCE CODE:")
        print("=" * 80)
        
        # Get and print local files
        local_files = SourceCodePrinter.get_local_python_files()
        SourceCodePrinter.print_file_contents(local_files)
        
        # Print global variables
        SourceCodePrinter.print_global_variables(globals_dict)
    
    @staticmethod
    def print_configuration_summary(config: Dict[str, Any]):
        """
        Print a summary of key configuration parameters.
        
        Args:
            config: Configuration dictionary
        """
        print("=" * 80)
        print("CONFIGURATION SUMMARY:")
        print("=" * 80)
        
        # Key training parameters
        key_params = [
            'training_type', 'dataset', 'batch_size', 'block_size', 'learning_rate',
            'max_iters', 'n_layer', 'n_head', 'n_embd', 'attention_type', 'use_rope',
            'grad_clip', 'dropout', 'warmup_iters', 'lr_decay_iters', 'min_lr'
        ]
        
        print("Training Parameters:")
        for param in key_params:
            if param in config:
                print(f"  {param}: {config[param]}")
        
        # Model-specific parameters
        if config.get('training_type') == 'token_classification':
            print(f"  num_token_classes: {config.get('num_token_classes', 'N/A')}")
        
        if config.get('training_type') == 'sequence_scoring':
            print(f"  unmasking_model_checkpoint: {config.get('unmasking_model_checkpoint', 'N/A')}")
        
        # Transfer learning parameters
        if config.get('init_from_checkpoint'):
            print("Transfer Learning:")
            print(f"  init_from_checkpoint: {config.get('init_from_checkpoint')}")
            print(f"  freeze_transformer: {config.get('freeze_transformer', False)}")
            print(f"  unfreeze_at_iteration: {config.get('unfreeze_at_iteration', 'N/A')}")
        
        # Unmasking stages
        if config.get('unmasking_stages'):
            print("Unmasking Stages:")
            for i, stage in enumerate(config['unmasking_stages']):
                print(f"  Stage {i}: {stage}")
        
        print("=" * 80)
    
    @staticmethod
    def print_training_start_banner(training_type: str, model_params: int, 
                                   vocab_size: int, block_size: int):
        """
        Print a banner with key training information.
        
        Args:
            training_type: Type of training
            model_params: Number of model parameters
            vocab_size: Vocabulary size
            block_size: Block size
        """
        print("\n" + "=" * 80)
        print("TRAINING STARTED")
        print("=" * 80)
        print(f"Training Type: {training_type}")
        print(f"Model Parameters: {model_params/1e6:.2f}M")
        print(f"Vocabulary Size: {vocab_size}")
        print(f"Block Size: {block_size}")
        print("=" * 80 + "\n")
    
    @staticmethod
    def print_model_architecture_summary(model):
        """
        Print a summary of the model architecture.
        
        Args:
            model: Model to summarize
        """
        print("=" * 80)
        print("MODEL ARCHITECTURE:")
        print("=" * 80)
        
        # Get raw model (unwrap DDP if needed)
        raw_model = model.module if hasattr(model, 'module') else model
        
        print(f"Model Type: {type(raw_model).__name__}")
        print(f"Total Parameters: {raw_model.get_num_params()/1e6:.2f}M")
        
        if hasattr(raw_model, 'config'):
            config = raw_model.config
            print(f"Layers: {config.n_layer}")
            print(f"Heads: {config.n_head}")
            print(f"Embedding Dimension: {config.n_embd}")
            print(f"Block Size: {config.block_size}")
            print(f"Vocabulary Size: {config.vocab_size}")
            print(f"Attention Type: {getattr(config, 'attention_type', 'causal')}")
            print(f"Use RoPE: {getattr(config, 'use_rope', False)}")
            print(f"Dropout: {config.dropout}")
            print(f"Bias: {config.bias}")
            
            if hasattr(config, 'mode'):
                print(f"Model Mode: {config.mode}")
            
            if hasattr(config, 'num_token_classes'):
                print(f"Token Classes: {config.num_token_classes}")
            
            if hasattr(config, 'cls_token_id'):
                print(f"CLS Token ID: {config.cls_token_id}")
        
        print("=" * 80 + "\n")
