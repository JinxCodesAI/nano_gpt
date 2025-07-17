#!/usr/bin/env python3
"""
Configuration and validation for reward data preparation.

This module provides configuration dataclasses and validation logic for the
configurable reward data preparation system.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# Custom exception classes
class RewardDataPrepError(Exception):
    """Base exception for reward data preparation errors."""
    pass


class TokenizationMismatchError(RewardDataPrepError):
    """Raised when tokenization methods are incompatible."""
    pass


class BinaryFileError(RewardDataPrepError):
    """Raised when binary files are invalid or incompatible."""
    pass


class ConfigurationError(RewardDataPrepError):
    """Raised when command-line parameters are invalid."""
    pass


@dataclass
class RewardDataConfig:
    """Configuration for reward data preparation."""
    
    # Required parameters
    model_path: str
    
    # Input configuration
    input_mode: str = 'text'  # 'text' or 'binary'
    data_path: Optional[str] = None     # Raw text file path
    train_bin: Optional[str] = None     # Existing train.bin path
    val_bin: Optional[str] = None       # Existing val.bin path
    
    # Tokenization configuration
    tokenization: str = 'auto'  # 'auto', 'bpe', 'char'
    meta_path: Optional[str] = None       # meta.pkl path for char tokenization
    
    # Generation parameters (existing)
    output_dir: str = 'data/reward_dataset'
    train_split: float = 0.9
    samples_per_chunk: int = 10
    temperature: float = 1.0
    top_k: Optional[int] = None
    device: str = 'auto'
    
    # Additional metadata
    _validation_errors: List[str] = field(default_factory=list, init=False)
    _warnings: List[str] = field(default_factory=list, init=False)
    
    def add_validation_error(self, error: str):
        """Add a validation error message."""
        self._validation_errors.append(error)
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self._warnings.append(warning)
    
    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return len(self._validation_errors) > 0
    
    def get_errors(self) -> List[str]:
        """Get all validation errors."""
        return self._validation_errors.copy()
    
    def get_warnings(self) -> List[str]:
        """Get all warnings."""
        return self._warnings.copy()


@dataclass
class TokenizationInfo:
    """Information about tokenization configuration."""
    
    method: str           # 'bpe' or 'char'
    vocab_size: int      # Size of vocabulary
    meta_path: Optional[str] = None # Path to meta.pkl if char tokenization
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for saving with dataset."""
        return {
            'method': self.method,
            'vocab_size': self.vocab_size,
            'meta_path': self.meta_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenizationInfo':
        """Deserialize from saved dataset."""
        return cls(
            method=data['method'],
            vocab_size=data['vocab_size'],
            meta_path=data.get('meta_path')
        )


class ConfigurationValidator:
    """Validates reward data preparation configuration."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_config(self, config: RewardDataConfig) -> bool:
        """
        Validate the complete configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        self._validate_required_parameters(config)
        self._validate_input_mode(config)
        self._validate_tokenization_config(config)
        self._validate_file_paths(config)
        self._validate_generation_parameters(config)
        self._check_parameter_conflicts(config)
        
        return not config.has_errors()
    
    def _validate_required_parameters(self, config: RewardDataConfig):
        """Validate required parameters."""
        if not config.model_path:
            config.add_validation_error("model_path is required")
        elif not os.path.exists(config.model_path):
            config.add_validation_error(f"Model file not found: {config.model_path}")
    
    def _validate_input_mode(self, config: RewardDataConfig):
        """Validate input mode configuration."""
        if config.input_mode not in ['text', 'binary']:
            config.add_validation_error(f"Invalid input_mode: {config.input_mode}. Must be 'text' or 'binary'")
            return
        
        if config.input_mode == 'text':
            if not config.data_path:
                config.add_validation_error("data_path is required when input_mode is 'text'")
            
            # Check for conflicting binary parameters
            if config.train_bin or config.val_bin:
                config.add_warning("train_bin and val_bin are ignored in text mode")
        
        elif config.input_mode == 'binary':
            if not config.train_bin or not config.val_bin:
                config.add_validation_error("Both train_bin and val_bin are required when input_mode is 'binary'")
            
            # Check for conflicting text parameters
            if config.data_path:
                config.add_warning("data_path is ignored in binary mode")
    
    def _validate_tokenization_config(self, config: RewardDataConfig):
        """Validate tokenization configuration."""
        if config.tokenization not in ['auto', 'bpe', 'char']:
            config.add_validation_error(f"Invalid tokenization: {config.tokenization}. Must be 'auto', 'bpe', or 'char'")
            return
        
        if config.tokenization == 'char':
            if not config.meta_path:
                # Try to infer meta_path from data_path
                if config.data_path:
                    data_dir = os.path.dirname(config.data_path) if os.path.isfile(config.data_path) else config.data_path
                    inferred_meta_path = os.path.join(data_dir, 'meta.pkl')
                    if os.path.exists(inferred_meta_path):
                        config.meta_path = inferred_meta_path
                        config.add_warning(f"Inferred meta_path: {inferred_meta_path}")
                    else:
                        config.add_validation_error("meta_path is required when tokenization is 'char'")
                else:
                    config.add_validation_error("meta_path is required when tokenization is 'char'")
        
        elif config.tokenization == 'bpe':
            if config.meta_path:
                config.add_warning("meta_path is ignored when tokenization is 'bpe'")
    
    def _validate_file_paths(self, config: RewardDataConfig):
        """Validate file paths exist and are accessible."""
        # Validate data_path
        if config.data_path and not os.path.exists(config.data_path):
            config.add_validation_error(f"Data file not found: {config.data_path}")
        
        # Validate binary files
        if config.train_bin and not os.path.exists(config.train_bin):
            config.add_validation_error(f"Train binary file not found: {config.train_bin}")
        
        if config.val_bin and not os.path.exists(config.val_bin):
            config.add_validation_error(f"Validation binary file not found: {config.val_bin}")
        
        # Validate meta_path
        if config.meta_path and not os.path.exists(config.meta_path):
            config.add_validation_error(f"Meta file not found: {config.meta_path}")
        
        # Validate output directory can be created
        try:
            os.makedirs(config.output_dir, exist_ok=True)
        except Exception as e:
            config.add_validation_error(f"Cannot create output directory {config.output_dir}: {str(e)}")
    
    def _validate_generation_parameters(self, config: RewardDataConfig):
        """Validate generation parameters."""
        if not (0.0 < config.train_split < 1.0):
            config.add_validation_error(f"train_split must be between 0 and 1, got {config.train_split}")
        
        if config.samples_per_chunk <= 0:
            config.add_validation_error(f"samples_per_chunk must be positive, got {config.samples_per_chunk}")
        
        if config.temperature <= 0:
            config.add_validation_error(f"temperature must be positive, got {config.temperature}")
        
        if config.top_k is not None and config.top_k <= 0:
            config.add_validation_error(f"top_k must be positive if specified, got {config.top_k}")
    
    def _check_parameter_conflicts(self, config: RewardDataConfig):
        """Check for parameter conflicts and provide helpful warnings."""
        # Check for both text and binary inputs specified
        has_text_input = bool(config.data_path)
        has_binary_input = bool(config.train_bin and config.val_bin)
        
        if has_text_input and has_binary_input:
            if config.input_mode == 'binary':
                config.add_warning("Both text and binary inputs specified. Using binary mode as requested.")
            else:
                config.add_warning("Both text and binary inputs specified. Using text mode as requested.")
        
        # Check tokenization method compatibility with input mode
        if config.input_mode == 'binary' and config.tokenization == 'char':
            if not config.meta_path:
                config.add_warning("Character tokenization with binary input requires meta_path for validation")
    
    def print_validation_results(self, config: RewardDataConfig):
        """Print validation results to console."""
        if config.get_warnings():
            print("\nWarnings:")
            for warning in config.get_warnings():
                print(f"  WARNING: {warning}")

        if config.get_errors():
            print("\nValidation Errors:")
            for error in config.get_errors():
                print(f"  ERROR: {error}")
            print("\nPlease fix the above errors before proceeding.")
        else:
            print("\nConfiguration validation passed")
    
    def suggest_fixes(self, config: RewardDataConfig) -> List[str]:
        """
        Suggest fixes for common configuration errors.
        
        Args:
            config: Configuration with errors
            
        Returns:
            List of suggested fixes
        """
        suggestions = []
        
        for error in config.get_errors():
            if "model_path" in error and "not found" in error:
                suggestions.append("Check that the model checkpoint file exists and the path is correct")
            
            elif "data_path is required" in error:
                suggestions.append("Specify --data_path when using text input mode")
            
            elif "train_bin and val_bin are required" in error:
                suggestions.append("Specify both --train_bin and --val_bin when using binary input mode")
            
            elif "meta_path is required" in error:
                suggestions.append("Specify --meta_path when using character tokenization, or use --tokenization auto")
            
            elif "not found" in error:
                suggestions.append("Check that all file paths are correct and files exist")
        
        return suggestions
