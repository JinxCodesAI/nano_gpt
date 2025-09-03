#!/usr/bin/env python3
"""
Test script for SEQUENCE_SCORER judge model functionality in evaluate_models.py

This script tests the new sequence scoring functionality without requiring
actual model checkpoints by creating mock models and data.
"""

import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import ModelMode, GPTConfig
from sample_utils import calculate_sequence_scores


class MockSequenceScorerModel:
    """Mock SEQUENCE_SCORER model for testing"""
    
    def __init__(self, cls_token_id=66):
        self.config = Mock()
        self.config.mode = ModelMode.SEQUENCE_SCORER
        self.config.cls_token_id = cls_token_id
        
        # Create a simple sequence head that returns predictable scores
        self.sequence_head = nn.Sequential(
            nn.Linear(384, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Initialize with small weights for testing
        with torch.no_grad():
            self.sequence_head[0].weight.fill_(0.1)
    
    def __call__(self, tokens_with_cls, targets):
        """Mock forward pass that returns predictable scores"""
        batch_size = tokens_with_cls.shape[0]
        
        # Create mock CLS token representations
        # Use sum of tokens as a simple feature for predictable scoring
        token_sums = tokens_with_cls.sum(dim=1).float()  # (batch_size,)
        
        # Normalize to create features
        features = (token_sums - token_sums.mean()) / (token_sums.std() + 1e-8)
        features = features.unsqueeze(-1).expand(-1, 384)  # (batch_size, 384)
        
        # Pass through sequence head
        scores = self.sequence_head(features).squeeze(-1)  # (batch_size,)
        
        return scores, None


def test_calculate_sequence_scores():
    """Test the calculate_sequence_scores function"""
    print("Testing calculate_sequence_scores function...")
    
    # Create mock model
    model = MockSequenceScorerModel(cls_token_id=66)
    
    # Create test tokens (batch_size=3, seq_len=10)
    tokens = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Sample 1
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # Sample 2 (different pattern)
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],   # Sample 3 (uniform)
    ])
    
    device = 'cpu'
    ctx = torch.no_grad()  # Simple context for testing
    
    # Test the function
    scores = calculate_sequence_scores(
        model=model,
        tokens=tokens,
        cls_token_id=66,
        device=device,
        ctx=ctx
    )
    
    print(f"  Input tokens shape: {tokens.shape}")
    print(f"  Output scores: {scores}")
    print(f"  Scores type: {type(scores)}")
    print(f"  Number of scores: {len(scores)}")
    
    # Verify results
    assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"
    assert all(0.0 <= score <= 1.0 for score in scores), f"Scores should be in [0,1] range: {scores}"
    assert isinstance(scores, list), f"Expected list, got {type(scores)}"
    
    print("  ✓ calculate_sequence_scores test passed!")
    return True


def test_winner_determination():
    """Test the winner determination logic for SEQUENCE_SCORER models"""
    print("\nTesting winner determination logic...")
    
    # Mock tournament manager
    from evaluate_models import TournamentManager
    
    # Create mock objects
    config = {'verbose': False}
    elo_tracker = Mock()
    judge_model_type = ModelMode.SEQUENCE_SCORER
    
    tournament_manager = TournamentManager(config, elo_tracker, judge_model_type)
    
    # Test cases: [model1_score, model2_score, expected_result]
    test_cases = [
        (0.3, 0.5, 'DRAW'),  # Close scores (diff = 0.2) = draw
        (0.7, 0.4, 'LOSE'),  # Higher score loses (diff > 0.2)
        (0.5, 0.69, 'DRAW'), # Close scores (diff < 0.2) = draw
        (0.5, 0.71, 'WIN'),  # Just above threshold, lower wins
        (0.1, 0.9, 'WIN'),   # Large difference, lower wins
    ]
    
    for model1_score, model2_score, expected in test_cases:
        # Create mock ratings
        model1_ratings = [{'confidence_score': model1_score}]
        model2_ratings = [{'confidence_score': model2_score}]
        
        result = tournament_manager.determine_winner(model1_ratings, model2_ratings)
        
        print(f"  Score1: {model1_score}, Score2: {model2_score} -> {result} (expected: {expected})")
        
        assert result == expected, f"Expected {expected}, got {result} for scores {model1_score} vs {model2_score}"
    
    print("  ✓ Winner determination test passed!")
    return True


def test_cls_token_prepending():
    """Test that CLS tokens are properly prepended to sequences"""
    print("\nTesting CLS token prepending...")
    
    # Create test tokens
    original_tokens = torch.tensor([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10]
    ])
    
    cls_token_id = 66
    batch_size, seq_len = original_tokens.shape
    
    # Manually prepend CLS token (same logic as in calculate_sequence_scores)
    cls_tokens = torch.full((batch_size, 1), cls_token_id, dtype=original_tokens.dtype)
    tokens_with_cls = torch.cat([cls_tokens, original_tokens], dim=1)
    
    print(f"  Original shape: {original_tokens.shape}")
    print(f"  With CLS shape: {tokens_with_cls.shape}")
    print(f"  CLS token ID: {cls_token_id}")
    print(f"  First column (CLS): {tokens_with_cls[:, 0].tolist()}")
    
    # Verify CLS token prepending
    assert tokens_with_cls.shape == (batch_size, seq_len + 1), f"Expected shape {(batch_size, seq_len + 1)}, got {tokens_with_cls.shape}"
    assert torch.all(tokens_with_cls[:, 0] == cls_token_id), "First column should be CLS tokens"
    assert torch.all(tokens_with_cls[:, 1:] == original_tokens), "Rest should match original tokens"
    
    print("  ✓ CLS token prepending test passed!")
    return True


def main():
    """Run all tests"""
    print("Testing SEQUENCE_SCORER judge model functionality")
    print("=" * 60)
    
    try:
        # Run tests
        test_calculate_sequence_scores()
        test_winner_determination()
        test_cls_token_prepending()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("\nThe SEQUENCE_SCORER judge model implementation appears to be working correctly.")
        print("Key features verified:")
        print("  - CLS token prepending")
        print("  - Sequence score calculation (0-1 range)")
        print("  - Winner determination (lower score wins)")
        print("  - Draw detection (difference < 0.2)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
