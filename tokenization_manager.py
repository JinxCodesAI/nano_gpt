#!/usr/bin/env python3
"""
TokenizationManager for handling different tokenization methods in reward data preparation.

This module provides a unified interface for both BPE (tiktoken) and character-level tokenization,
with automatic detection capabilities based on file structure and metadata.
"""

import os
import pickle
import logging
from typing import List, Optional, Tuple, Dict, Any, Callable
import tiktoken


class TokenizationError(Exception):
    """Base exception for tokenization-related errors."""
    pass


class TokenizationDetectionError(TokenizationError):
    """Raised when tokenization method cannot be detected."""
    pass


class MetaFileError(TokenizationError):
    """Raised when meta.pkl file is invalid or missing."""
    pass


class TokenizationManager:
    """
    Manages tokenization methods for reward data preparation.
    
    Supports both BPE (tiktoken GPT-2) and character-level tokenization with
    automatic detection based on file structure and metadata.
    """
    
    def __init__(self, data_path: Optional[str] = None, meta_path: Optional[str] = None):
        """
        Initialize TokenizationManager.
        
        Args:
            data_path: Path to data directory for auto-detection
            meta_path: Explicit path to meta.pkl file for character tokenization
        """
        self.tokenization_type: Optional[str] = None  # 'bpe' or 'char'
        self.vocab_size: Optional[int] = None
        self.encoder: Optional[Callable[[str], List[int]]] = None
        self.decoder: Optional[Callable[[List[int]], str]] = None
        self.meta_path: Optional[str] = meta_path
        self.meta_data: Optional[Dict[str, Any]] = None
        self.tiktoken_encoder: Optional[tiktoken.Encoding] = None
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize tokenization if paths provided
        if data_path or meta_path:
            self._initialize_tokenization(data_path, meta_path)
    
    def _initialize_tokenization(self, data_path: Optional[str], meta_path: Optional[str]):
        """Initialize tokenization based on provided paths."""
        if meta_path:
            # Explicit character tokenization
            self.load_char_tokenization(meta_path)
        elif data_path:
            # Auto-detect tokenization method
            detected_method = self.detect_tokenization_method(data_path)
            if detected_method == 'char':
                # Look for meta.pkl in data directory
                auto_meta_path = os.path.join(data_path, 'meta.pkl')
                if os.path.exists(auto_meta_path):
                    self.load_char_tokenization(auto_meta_path)
                else:
                    raise MetaFileError(f"Character tokenization detected but meta.pkl not found in {data_path}")
            else:
                self.load_bpe_tokenization()
    
    def detect_tokenization_method(self, data_path: str) -> str:
        """
        Auto-detect tokenization method from file structure.
        
        Args:
            data_path: Path to data directory or file
            
        Returns:
            'bpe' or 'char' indicating detected tokenization method
            
        Raises:
            TokenizationDetectionError: If method cannot be determined
        """
        if not os.path.exists(data_path):
            raise TokenizationDetectionError(f"Data path does not exist: {data_path}")
        
        # If data_path is a file, check its parent directory
        if os.path.isfile(data_path):
            data_dir = os.path.dirname(data_path)
        else:
            data_dir = data_path
        
        # Check for meta.pkl file (indicates character-level tokenization)
        meta_path = os.path.join(data_dir, 'meta.pkl')
        if os.path.exists(meta_path):
            self.logger.info(f"Found meta.pkl at {meta_path}, using character-level tokenization")
            return 'char'
        
        # Check for specific directory names that indicate character tokenization
        if 'char' in os.path.basename(data_dir).lower():
            self.logger.info(f"Directory name contains 'char', assuming character-level tokenization")
            return 'char'
        
        # Default to BPE tokenization
        self.logger.info("No character tokenization indicators found, using BPE tokenization")
        return 'bpe'
    
    def load_char_tokenization(self, meta_path: str):
        """
        Load character-level tokenization from meta.pkl file.
        
        Args:
            meta_path: Path to meta.pkl file
            
        Raises:
            MetaFileError: If meta.pkl file is invalid or missing
        """
        if not os.path.exists(meta_path):
            raise MetaFileError(f"Meta file not found: {meta_path}")
        
        try:
            with open(meta_path, 'rb') as f:
                self.meta_data = pickle.load(f)
            
            # Validate meta data structure
            required_keys = ['vocab_size', 'itos', 'stoi']
            for key in required_keys:
                if key not in self.meta_data:
                    raise MetaFileError(f"Missing required key '{key}' in meta.pkl")
            
            self.tokenization_type = 'char'
            self.vocab_size = self.meta_data['vocab_size']
            self.meta_path = meta_path
            
            # Create encoder and decoder functions
            stoi = self.meta_data['stoi']
            itos = self.meta_data['itos']
            
            self.encoder = lambda s: [stoi[c] for c in s]
            self.decoder = lambda l: ''.join([itos[i] for i in l])
            
            self.logger.info(f"Loaded character tokenization: vocab_size={self.vocab_size}")
            
        except Exception as e:
            raise MetaFileError(f"Failed to load meta.pkl from {meta_path}: {str(e)}")
    
    def load_bpe_tokenization(self):
        """Initialize tiktoken BPE encoding (GPT-2)."""
        try:
            self.tiktoken_encoder = tiktoken.get_encoding("gpt2")
            self.tokenization_type = 'bpe'
            self.vocab_size = self.tiktoken_encoder.n_vocab
            
            # Create encoder and decoder functions
            self.encoder = lambda s: self.tiktoken_encoder.encode_ordinary(s)
            self.decoder = lambda l: self.tiktoken_encoder.decode(l)
            
            self.logger.info(f"Loaded BPE tokenization: vocab_size={self.vocab_size}")
            
        except Exception as e:
            raise TokenizationError(f"Failed to initialize BPE tokenization: {str(e)}")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text using the configured tokenization method.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
            
        Raises:
            TokenizationError: If tokenization is not initialized
        """
        if self.encoder is None:
            raise TokenizationError("Tokenization not initialized. Call load_char_tokenization() or load_bpe_tokenization() first.")
        
        try:
            return self.encoder(text)
        except Exception as e:
            raise TokenizationError(f"Failed to encode text: {str(e)}")
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode tokens using the configured tokenization method.
        
        Args:
            tokens: List of token IDs to decode
            
        Returns:
            Decoded text string
            
        Raises:
            TokenizationError: If tokenization is not initialized
        """
        if self.decoder is None:
            raise TokenizationError("Tokenization not initialized. Call load_char_tokenization() or load_bpe_tokenization() first.")
        
        try:
            return self.decoder(tokens)
        except Exception as e:
            raise TokenizationError(f"Failed to decode tokens: {str(e)}")
    
    def validate_tokens(self, tokens: List[int]) -> bool:
        """
        Validate that all tokens are within the vocabulary range.
        
        Args:
            tokens: List of token IDs to validate
            
        Returns:
            True if all tokens are valid, False otherwise
        """
        if self.vocab_size is None:
            return False
        
        return all(0 <= token < self.vocab_size for token in tokens)
    
    def get_tokenization_info(self) -> Dict[str, Any]:
        """
        Get information about the current tokenization configuration.
        
        Returns:
            Dictionary containing tokenization metadata
        """
        return {
            'method': self.tokenization_type,
            'vocab_size': self.vocab_size,
            'meta_path': self.meta_path,
            'is_initialized': self.encoder is not None and self.decoder is not None
        }
    
    def is_compatible_with(self, other_info: Dict[str, Any]) -> bool:
        """
        Check if this tokenization is compatible with another configuration.
        
        Args:
            other_info: Tokenization info dictionary from another instance
            
        Returns:
            True if configurations are compatible
        """
        if not self.encoder or not other_info.get('is_initialized'):
            return False
        
        return (self.tokenization_type == other_info.get('method') and
                self.vocab_size == other_info.get('vocab_size'))
