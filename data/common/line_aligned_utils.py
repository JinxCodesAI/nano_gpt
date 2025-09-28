"""
Shared utilities for line-aligned variable-length sequence generation.
This module provides common functionality for data providers that need to generate
sequences consisting of full lines with proper padding.
"""
from __future__ import annotations

import torch
from typing import List, Tuple, Dict, Any


class LineAlignedSequenceBuilder:
    """
    Helper class for building line-aligned variable-length sequences.
    Ensures sequences consist only of full lines and handles padding appropriately.
    """
    
    def __init__(self, 
                 lines_ids: List[List[int]], 
                 newline_token_id: int = None,
                 pad_token_id: int = None):
        """
        Initialize the builder with line data.
        
        Args:
            lines_ids: List of lines, where each line is a list of token IDs
            newline_token_id: Token ID for newline character (optional)
            pad_token_id: Token ID for padding (optional)
        """
        self.lines_ids = lines_ids
        self.newline_token_id = newline_token_id
        self.pad_token_id = pad_token_id
        
        # Precompute tensors for efficient line packing
        self.line_lens = torch.tensor([len(x) for x in lines_ids], dtype=torch.long)
        self.cumsum = self.line_lens.cumsum(dim=0)
        self.line_offsets = torch.cat([torch.tensor([0], dtype=torch.long), self.cumsum[:-1]])
        self.tokens_flat = torch.tensor([t for line in lines_ids for t in line], dtype=torch.long)
        
        # Valid start lines: skip blank-only lines (just a single '\n')
        self.valid_starts = torch.tensor([i for i, ids in enumerate(lines_ids) if len(ids) > 1], dtype=torch.long)
    
    def build_variable_length_sequences(self, 
                                      count: int, 
                                      block_size: int, 
                                      rng: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build variable-length line-aligned sequences.
        
        Args:
            count: Number of sequences to generate
            block_size: Maximum sequence length
            rng: Random number generator
            
        Returns:
            x: Tensor of shape (count, block_size) containing sequences
            content_lengths: Tensor of shape (count,) with actual content lengths
        """
        if len(self.lines_ids) == 0:
            raise ValueError("No lines available")
        
        # Start with zeros (will be filled with actual content)
        x = torch.zeros((count, block_size), dtype=torch.long)
        content_lengths = torch.zeros(count, dtype=torch.long)
        
        if self.valid_starts.numel() == 0:
            raise ValueError("No valid (non-blank) start lines")
            
        starts = self.valid_starts[torch.randint(0, self.valid_starts.numel(), (count,), generator=rng)]
        
        # Use full block_size for content
        B = block_size
        s_prev = torch.where(starts > 0, self.cumsum[starts - 1], torch.zeros_like(starts, dtype=torch.long))
        thresholds = s_prev + B
        last_idx = torch.searchsorted(self.cumsum, thresholds, right=True) - 1
        
        for b in range(count):
            s = int(starts[b].item())
            e = int(last_idx[b].item())
            if e >= s:
                sum_full = int((self.cumsum[e] - (self.cumsum[s - 1] if s > 0 else 0)).item())
                start_ptr = int(self.line_offsets[s].item())
                if sum_full > 0:
                    # Fill with actual content
                    content_len = min(sum_full, block_size)
                    x[b, :content_len] = self.tokens_flat[start_ptr:start_ptr + content_len]
                    content_lengths[b] = content_len
            else:
                # Single line case
                budget = B
                take = max(0, int(budget) - 1)
                if take > 0:
                    p0 = int(self.line_offsets[s].item())
                    content_len = min(take, block_size)
                    x[b, :content_len] = self.tokens_flat[p0:p0 + content_len]
                    content_lengths[b] = content_len
                else:
                    content_lengths[b] = 0
                
                # Add newline if there's space and we have a newline token
                if self.newline_token_id is not None and content_lengths[b] < block_size:
                    x[b, content_lengths[b]] = self.newline_token_id
                    content_lengths[b] += 1
        
        return x, content_lengths
    
    def apply_padding(self, 
                     x: torch.Tensor, 
                     content_lengths: torch.Tensor) -> torch.Tensor:
        """
        Apply padding to sequences after content.
        
        Args:
            x: Input sequences
            content_lengths: Actual content lengths for each sequence
            
        Returns:
            x with padding applied
        """
        if self.pad_token_id is None:
            return x
            
        count, block_size = x.shape
        for b in range(count):
            if content_lengths[b] < block_size:
                x[b, content_lengths[b]:] = self.pad_token_id
        
        return x


def prepare_line_data(data: str, stoi: Dict[str, int]) -> Tuple[List[str], List[List[int]], Dict[str, int]]:
    """
    Prepare line data from raw text for line-aligned sequence generation.

    Args:
        data: Raw text data
        stoi: External string-to-index mapping to use for tokenization. Must be provided.

    Returns:
        lines: List of lines (with newlines kept)
        lines_ids: List of lines converted to token IDs
        stoi: String to index mapping
    """
    if stoi is None:
        raise ValueError("stoi must be provided; do not auto-build from data")

    # Split into lines (keeping newlines)
    lines = data.splitlines(keepends=True)
    lines_ids = [[stoi[c] for c in line] for line in lines]

    return lines, lines_ids, stoi


def create_line_aligned_builder(data: str,
                              newline_token_id: int = None,
                              pad_token_id: int = None,
                              stoi: Dict[str, int] = None) -> Tuple[LineAlignedSequenceBuilder, LineAlignedSequenceBuilder, Dict[str, int]]:
    """
    Create train and validation line-aligned sequence builders from text data.

    Args:
        data: Raw text data
        newline_token_id: Token ID for newline character
        pad_token_id: Token ID for padding
        stoi: External string-to-index mapping (required)

    Returns:
        train_builder: Builder for training data
        val_builder: Builder for validation data
        stoi: String to index mapping
    """
    if stoi is None:
        raise ValueError("stoi must be provided; do not auto-build from data")
    lines, lines_ids, stoi = prepare_line_data(data, stoi=stoi)

    # Create train/val splits (90/10)
    n = len(lines_ids)
    train_lines_ids = lines_ids[:int(n * 0.9)]
    val_lines_ids = lines_ids[int(n * 0.9):]

    train_builder = LineAlignedSequenceBuilder(train_lines_ids, newline_token_id, pad_token_id)
    val_builder = LineAlignedSequenceBuilder(val_lines_ids, newline_token_id, pad_token_id)

    return train_builder, val_builder, stoi
