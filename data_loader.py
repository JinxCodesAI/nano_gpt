#!/usr/bin/env python3
"""
DataLoader for unified data loading in reward data preparation.

This module provides a unified interface for loading data from both raw text files
and existing binary files, with validation and error handling.
"""

import os
import logging
import numpy as np
from typing import List, Tuple, Optional
from tokenization_manager import TokenizationManager, TokenizationError


class DataLoadError(Exception):
    """Base exception for data loading errors."""
    pass


class BinaryFileError(DataLoadError):
    """Raised when binary files are invalid or incompatible."""
    pass


class TextFileError(DataLoadError):
    """Raised when text files cannot be processed."""
    pass


class DataLoader:
    """
    Unified data loader for reward model data preparation.
    
    Supports loading from both raw text files and existing binary files,
    with automatic validation and error handling.
    """
    
    def __init__(self, tokenization_manager: TokenizationManager):
        """
        Initialize DataLoader.
        
        Args:
            tokenization_manager: Configured TokenizationManager instance
        """
        self.tokenizer = tokenization_manager
        self.logger = logging.getLogger(__name__)
        
        if not self.tokenizer.encoder:
            raise DataLoadError("TokenizationManager must be initialized before creating DataLoader")
    
    def load_from_text(self, text_path: str, train_split: float = 0.9) -> Tuple[List[int], List[int]]:
        """
        Load and split raw text file using the configured tokenization method.
        
        Args:
            text_path: Path to raw text file
            train_split: Fraction of data to use for training (default: 0.9)
            
        Returns:
            Tuple of (train_tokens, val_tokens)
            
        Raises:
            TextFileError: If text file cannot be loaded or processed
        """
        if not os.path.exists(text_path):
            raise TextFileError(f"Text file not found: {text_path}")
        
        if not (0.0 < train_split < 1.0):
            raise TextFileError(f"Invalid train_split: {train_split}. Must be between 0 and 1.")
        
        try:
            self.logger.info(f"Loading text data from {text_path}")
            
            # Read text file
            with open(text_path, 'r', encoding='utf-8') as f:
                data = f.read()
            
            if not data.strip():
                raise TextFileError(f"Text file is empty: {text_path}")
            
            # Tokenize the data
            self.logger.info(f"Tokenizing data using {self.tokenizer.tokenization_type} method")
            tokens = self.tokenizer.encode(data)
            
            if not tokens:
                raise TextFileError("Tokenization resulted in empty token list")
            
            # Split the data
            n = len(tokens)
            split_idx = int(n * train_split)
            
            train_tokens = tokens[:split_idx]
            val_tokens = tokens[split_idx:]
            
            self.logger.info(f"Data split complete:")
            self.logger.info(f"  Total tokens: {n:,}")
            self.logger.info(f"  Train tokens: {len(train_tokens):,}")
            self.logger.info(f"  Val tokens: {len(val_tokens):,}")
            
            return train_tokens, val_tokens
            
        except TokenizationError as e:
            raise TextFileError(f"Tokenization failed: {str(e)}")
        except Exception as e:
            raise TextFileError(f"Failed to load text file {text_path}: {str(e)}")
    
    def load_from_binary(self, train_bin: str, val_bin: str) -> Tuple[List[int], List[int]]:
        """
        Load existing binary train.bin and val.bin files.
        
        Args:
            train_bin: Path to train.bin file
            val_bin: Path to val.bin file
            
        Returns:
            Tuple of (train_tokens, val_tokens)
            
        Raises:
            BinaryFileError: If binary files are invalid or incompatible
        """
        # Validate file existence
        if not os.path.exists(train_bin):
            raise BinaryFileError(f"Train binary file not found: {train_bin}")
        
        if not os.path.exists(val_bin):
            raise BinaryFileError(f"Validation binary file not found: {val_bin}")
        
        try:
            self.logger.info(f"Loading binary data from {train_bin} and {val_bin}")
            
            # Load binary files
            train_tokens = self._load_binary_file(train_bin, "train")
            val_tokens = self._load_binary_file(val_bin, "val")
            
            # Validate token compatibility
            self._validate_token_compatibility(train_tokens, "train")
            self._validate_token_compatibility(val_tokens, "val")
            
            self.logger.info(f"Binary data loaded successfully:")
            self.logger.info(f"  Train tokens: {len(train_tokens):,}")
            self.logger.info(f"  Val tokens: {len(val_tokens):,}")
            
            return train_tokens, val_tokens
            
        except Exception as e:
            if isinstance(e, BinaryFileError):
                raise
            raise BinaryFileError(f"Failed to load binary files: {str(e)}")
    
    def _load_binary_file(self, file_path: str, split_name: str) -> List[int]:
        """
        Load a single binary file and convert to token list.
        
        Args:
            file_path: Path to binary file
            split_name: Name of the split (for error messages)
            
        Returns:
            List of tokens
            
        Raises:
            BinaryFileError: If file cannot be loaded or is invalid
        """
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise BinaryFileError(f"{split_name} binary file is empty: {file_path}")
            
            # Load as uint16 (standard format for token files)
            tokens_array = np.fromfile(file_path, dtype=np.uint16)
            
            if len(tokens_array) == 0:
                raise BinaryFileError(f"No tokens loaded from {split_name} file: {file_path}")
            
            # Convert to list
            tokens = tokens_array.tolist()
            
            self.logger.debug(f"Loaded {len(tokens):,} tokens from {file_path}")
            
            return tokens
            
        except Exception as e:
            if isinstance(e, BinaryFileError):
                raise
            raise BinaryFileError(f"Failed to load {split_name} binary file {file_path}: {str(e)}")
    
    def _validate_token_compatibility(self, tokens: List[int], split_name: str):
        """
        Validate that tokens are compatible with the current tokenization method.
        
        Args:
            tokens: List of tokens to validate
            split_name: Name of the split (for error messages)
            
        Raises:
            BinaryFileError: If tokens are incompatible
        """
        if not tokens:
            raise BinaryFileError(f"{split_name} token list is empty")
        
        # Check token range compatibility
        if not self.tokenizer.validate_tokens(tokens):
            max_token = max(tokens)
            min_token = min(tokens)
            raise BinaryFileError(
                f"Token range incompatible in {split_name} data. "
                f"Found tokens in range [{min_token}, {max_token}], "
                f"but vocab size is {self.tokenizer.vocab_size}"
            )
        
        self.logger.debug(f"Token validation passed for {split_name} data")
    
    def validate_binary_files(self, train_bin: str, val_bin: str) -> bool:
        """
        Validate binary file format and compatibility without loading full data.
        
        Args:
            train_bin: Path to train.bin file
            val_bin: Path to val.bin file
            
        Returns:
            True if files are valid and compatible
            
        Raises:
            BinaryFileError: If validation fails
        """
        # Check file existence
        for file_path, name in [(train_bin, "train"), (val_bin, "val")]:
            if not os.path.exists(file_path):
                raise BinaryFileError(f"{name} binary file not found: {file_path}")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise BinaryFileError(f"{name} binary file is empty: {file_path}")
            
            # Check if file size is reasonable (multiple of 2 bytes for uint16)
            if file_size % 2 != 0:
                raise BinaryFileError(f"{name} binary file has invalid size (not multiple of 2): {file_path}")
        
        # Sample validation - check first few tokens from each file
        try:
            for file_path, name in [(train_bin, "train"), (val_bin, "val")]:
                # Load just the first 100 tokens for validation
                sample_tokens = np.fromfile(file_path, dtype=np.uint16, count=100).tolist()
                if sample_tokens and not self.tokenizer.validate_tokens(sample_tokens):
                    max_token = max(sample_tokens)
                    raise BinaryFileError(
                        f"Sample tokens from {name} file exceed vocab size. "
                        f"Max token: {max_token}, vocab size: {self.tokenizer.vocab_size}"
                    )
        
        except Exception as e:
            if isinstance(e, BinaryFileError):
                raise
            raise BinaryFileError(f"Binary file validation failed: {str(e)}")
        
        self.logger.info("Binary file validation passed")
        return True
    
    def get_data_info(self, train_tokens: List[int], val_tokens: List[int]) -> dict:
        """
        Get information about loaded data.
        
        Args:
            train_tokens: Training tokens
            val_tokens: Validation tokens
            
        Returns:
            Dictionary with data statistics
        """
        info = {
            'train_size': len(train_tokens),
            'val_size': len(val_tokens),
            'total_size': len(train_tokens) + len(val_tokens),
            'train_split_ratio': len(train_tokens) / (len(train_tokens) + len(val_tokens)),
            'tokenization_method': self.tokenizer.tokenization_type,
            'vocab_size': self.tokenizer.vocab_size
        }
        
        if train_tokens:
            info['train_token_range'] = (min(train_tokens), max(train_tokens))
        
        if val_tokens:
            info['val_token_range'] = (min(val_tokens), max(val_tokens))
        
        return info
