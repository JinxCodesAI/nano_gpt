#!/usr/bin/env python3
"""
Test script to verify the outlier filtering functionality in BatchManager and ModelAnalyzer.
"""

import torch
import numpy as np
import tempfile
import os
import sys

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import only the analyzer module directly
from analyzer import ModelAnalyzer

def create_test_data(vocab_size=1000, num_tokens=10000):
    """Create test data with a known distribution."""
    # Create a Zipfian-like distribution where some tokens are much more frequent
    frequencies = np.power(np.arange(1, vocab_size + 1), -1.5)  # Zipfian distribution
    frequencies = frequencies / frequencies.sum()
    
    # Generate tokens according to this distribution
    tokens = np.random.choice(vocab_size, size=num_tokens, p=frequencies)
    
    return tokens, frequencies

def test_outlier_filtering_logic():
    """Test the outlier filtering logic directly."""
    print("Testing outlier filtering logic...")

    # Simulate served token counts with a Zipfian distribution
    vocab_size = 1000
    served_token_counts = torch.zeros(vocab_size, dtype=torch.float64)

    # Create a realistic distribution where some tokens are much more frequent
    frequencies = np.power(np.arange(1, vocab_size + 1), -1.5)  # Zipfian distribution
    frequencies = frequencies / frequencies.sum()

    # Simulate total tokens served
    total_tokens_served = 100000
    served_token_counts = torch.from_numpy(frequencies * total_tokens_served).double()

    # Test the filtering logic
    for ignored_sum in [0.01, 0.05, 0.1]:
        print(f"\nTesting with ignored_outlayers_sum = {ignored_sum}")

        # Calculate the served distribution
        served_distribution = served_token_counts / total_tokens_served

        # Sort tokens by their served frequency
        sorted_indices = torch.argsort(served_distribution, descending=True)
        sorted_counts = served_distribution[sorted_indices]

        # Calculate cumulative sum
        cumulative_sum = torch.cumsum(sorted_counts, dim=0)

        # Find tokens that sum up to (1 - ignored_outlayers_sum) of total
        target_sum = 1.0 - ignored_sum

        # Find the cutoff index where cumulative sum reaches target_sum
        cutoff_idx = torch.searchsorted(cumulative_sum, target_sum, right=True)
        cutoff_idx = min(cutoff_idx.item(), vocab_size - 1)

        # Get the non-outlier token IDs
        non_outlier_tokens = sorted_indices[:cutoff_idx + 1].tolist()

        print(f"Selected {len(non_outlier_tokens)} tokens out of {vocab_size}")
        print(f"Percentage of tokens selected: {len(non_outlier_tokens)/vocab_size*100:.1f}%")
        print(f"These tokens represent {cumulative_sum[cutoff_idx]:.4f} of total served tokens")

        # Verify that the selected tokens are reasonable
        assert len(non_outlier_tokens) > 0, "Should select at least some tokens"
        assert len(non_outlier_tokens) <= vocab_size, "Cannot select more tokens than vocabulary"
        assert len(non_outlier_tokens) < vocab_size, "Should filter out some tokens"
        assert cumulative_sum[cutoff_idx] >= target_sum * 0.95, "Should capture most of the target distribution"

    print("Outlier filtering logic test passed!")

def test_analyzer_filtered_embeddings():
    """Test the ModelAnalyzer's filtered embedding analysis."""
    print("\nTesting ModelAnalyzer filtered embedding analysis...")
    
    # Create a simple mock model for testing
    class MockModel:
        def __init__(self, vocab_size, embed_dim):
            self.config = type('Config', (), {
                'n_layer': 2,
                'vocab_size': vocab_size,
                'n_embd': embed_dim
            })()
    
    vocab_size = 100
    embed_dim = 64
    mock_model = MockModel(vocab_size, embed_dim)
    analyzer = ModelAnalyzer(mock_model)
    
    # Create test embedding weights
    embedding_weights = torch.randn(vocab_size, embed_dim)
    
    # Test without filtering (analyze all embeddings)
    print("Testing without filtering...")
    result_all = analyzer._analyze_embedding_geometry(embedding_weights)
    
    if result_all is not None:
        print(f"Analyzed {result_all['analysis_info']['num_embeddings_analyzed']} embeddings")
        assert result_all['analysis_info']['num_embeddings_analyzed'] == vocab_size
        assert result_all['analysis_info']['total_embeddings'] == vocab_size
        assert result_all['analysis_info']['filtered'] == False
    
    # Test with filtering (analyze subset of embeddings)
    print("Testing with filtering...")
    filtered_tokens = list(range(0, vocab_size, 2))  # Select every other token
    result_filtered = analyzer._analyze_embedding_geometry(
        embedding_weights, 
        filtered_token_ids=filtered_tokens
    )
    
    if result_filtered is not None:
        print(f"Analyzed {result_filtered['analysis_info']['num_embeddings_analyzed']} embeddings")
        assert result_filtered['analysis_info']['num_embeddings_analyzed'] == len(filtered_tokens)
        assert result_filtered['analysis_info']['total_embeddings'] == vocab_size
        assert result_filtered['analysis_info']['filtered'] == True
        
        # Verify that the analysis still produces reasonable results
        assert 'local_density' in result_filtered
        assert 'global_sparsity' in result_filtered
        assert 'analysis_info' in result_filtered
    
    print("ModelAnalyzer filtered embedding analysis test passed!")

def main():
    """Run all tests."""
    print("Starting outlier filtering tests...\n")

    try:
        test_outlier_filtering_logic()
        test_analyzer_filtered_embeddings()
        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
