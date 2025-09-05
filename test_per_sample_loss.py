"""
Unit tests for per-sample loss processing functionality.
Tests verify that per-sample processing matches original behavior when wrongness_factor=1.0.
"""

import torch
import math
import pytest
from training_utils.loss_processing import calculate_per_sample_losses, apply_per_sample_modifications
from training_utils.entropy_utils import calculate_wrong_answer_entropy, calculate_wrong_answer_entropy_per_sample, get_current_entropy_penalty
from training_utils.training_config import TrainingContext


def create_test_training_context(enable_entropy_penalty=False, batch_size=4):
    """Create a test training context with minimal required fields."""
    ctx = TrainingContext()
    ctx.batch_size = batch_size
    ctx.extended_vocab_size = 100
    ctx.enable_entropy_penalty = enable_entropy_penalty
    ctx.max_entropy_penalty = 0.5
    ctx.entropy_penalty_start_iter = 1000
    ctx.max_iters = 10000
    ctx.entropy_multiplier_ema = 1.0
    ctx.entropy_multiplier_ema_factor = 0.99
    return ctx


def test_per_sample_losses_basic():
    """Test basic per-sample loss calculation matches aggregated cross-entropy."""
    batch_size, seq_len, vocab_size = 4, 8, 100
    device = torch.device('cpu')
    
    # Create test data
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    mask = torch.randint(0, 2, (batch_size, seq_len), device=device).bool()
    
    # Ensure at least some positions are masked
    mask[:, 0] = True
    
    # Calculate per-sample losses
    per_sample_losses, per_sample_mask_counts = calculate_per_sample_losses(logits, targets, mask)
    
    # Calculate original aggregated loss for comparison
    logits_reshaped = logits.view(-1, vocab_size)
    targets_reshaped = targets.view(-1)
    mask_reshaped = mask.view(-1)
    
    original_loss = torch.nn.functional.cross_entropy(
        logits_reshaped[mask_reshaped],
        targets_reshaped[mask_reshaped],
        reduction='mean'
    )
    
    # The mean of per-sample losses (weighted by mask counts) should match original
    valid_samples = per_sample_mask_counts > 0
    if valid_samples.any():
        aggregated_loss = per_sample_losses[valid_samples].mean()
        
        # Should be close (allowing for numerical precision differences)
        assert torch.allclose(aggregated_loss, original_loss, atol=1e-2), \
            f"Aggregated per-sample loss {aggregated_loss} doesn't match original {original_loss}"
    
    # Check shapes
    assert per_sample_losses.shape == (batch_size,), f"Expected shape ({batch_size},), got {per_sample_losses.shape}"
    assert per_sample_mask_counts.shape == (batch_size,), f"Expected shape ({batch_size},), got {per_sample_mask_counts.shape}"


def test_per_sample_losses_soft_targets():
    """Test per-sample loss calculation with soft targets."""
    batch_size, seq_len, vocab_size = 2, 4, 50
    device = torch.device('cpu')
    
    # Create test data with soft targets
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    targets = torch.softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)  # Soft targets
    mask = torch.ones(batch_size, seq_len, device=device).bool()
    
    # Calculate per-sample losses
    per_sample_losses, per_sample_mask_counts = calculate_per_sample_losses(logits, targets, mask)
    
    # Calculate original aggregated loss for comparison
    logits_reshaped = logits.view(-1, vocab_size)
    targets_reshaped = targets.view(-1, vocab_size)
    mask_reshaped = mask.view(-1)
    
    original_loss = torch.nn.functional.cross_entropy(
        logits_reshaped[mask_reshaped],
        targets_reshaped[mask_reshaped],
        reduction='mean'
    )
    
    # The mean of per-sample losses should match original
    aggregated_loss = per_sample_losses.mean()
    assert torch.allclose(aggregated_loss, original_loss, atol=1e-2), \
        f"Aggregated per-sample loss {aggregated_loss} doesn't match original {original_loss}"


def test_entropy_per_sample_matches_global():
    """Test that per-sample entropy calculation matches global when averaged."""
    batch_size, seq_len, vocab_size = 3, 6, 80
    device = torch.device('cpu')
    
    # Create test data
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    mask = torch.ones(batch_size, seq_len, device=device).bool()
    
    # Calculate global entropy
    global_entropy = calculate_wrong_answer_entropy(logits, targets, vocab_size)
    
    # Calculate per-sample entropies
    per_sample_entropies = calculate_wrong_answer_entropy_per_sample(logits, targets, mask, vocab_size)
    
    # Average should be close to global (allowing for numerical differences)
    avg_per_sample_entropy = per_sample_entropies.mean()
    assert torch.allclose(avg_per_sample_entropy, global_entropy, atol=1e-5), \
        f"Average per-sample entropy {avg_per_sample_entropy} doesn't match global {global_entropy}"


def test_apply_per_sample_modifications_no_penalty():
    """Test per-sample modifications with no entropy penalty and wrongness_factor=1.0."""
    batch_size = 4
    device = torch.device('cpu')
    
    # Create test data
    per_sample_losses = torch.rand(batch_size, device=device)
    logits = torch.randn(batch_size, 8, 100, device=device)
    targets = torch.randint(0, 100, (batch_size, 8), device=device)
    mask = torch.ones(batch_size, 8, device=device).bool()
    wrongness_factor = torch.ones(batch_size, device=device)
    
    # Create training context with no entropy penalty
    training_ctx = create_test_training_context(enable_entropy_penalty=False, batch_size=batch_size)
    
    # Apply modifications
    modified_losses = apply_per_sample_modifications(
        per_sample_losses, logits, targets, mask, training_ctx, iter_num=5000, wrongness_factor=wrongness_factor
    )
    
    # Should be identical to original losses
    assert torch.allclose(modified_losses, per_sample_losses), \
        "Modified losses should match original when no penalty and wrongness_factor=1.0"


def test_apply_per_sample_modifications_with_entropy():
    """Test per-sample modifications with entropy penalty enabled."""
    batch_size = 3
    device = torch.device('cpu')
    
    # Create test data
    per_sample_losses = torch.rand(batch_size, device=device)
    logits = torch.randn(batch_size, 6, 50, device=device)
    targets = torch.randint(0, 50, (batch_size, 6), device=device)
    mask = torch.ones(batch_size, 6, device=device).bool()
    wrongness_factor = torch.ones(batch_size, device=device)
    
    # Create training context with entropy penalty
    training_ctx = create_test_training_context(enable_entropy_penalty=True, batch_size=batch_size)
    
    # Apply modifications at iteration where penalty is active
    iter_num = 5000
    modified_losses = apply_per_sample_modifications(
        per_sample_losses, logits, targets, mask, training_ctx, iter_num=iter_num, wrongness_factor=wrongness_factor
    )
    
    # Should be different from original losses (entropy penalty applied)
    assert not torch.allclose(modified_losses, per_sample_losses), \
        "Modified losses should differ from original when entropy penalty is applied"
    
    # All modified losses should be >= original (penalty increases loss)
    assert torch.all(modified_losses >= per_sample_losses), \
        "Entropy penalty should increase losses"


def test_wrongness_factor_scaling():
    """Test that wrongness factor correctly scales per-sample losses."""
    batch_size = 4
    device = torch.device('cpu')
    
    # Create test data
    per_sample_losses = torch.rand(batch_size, device=device)
    logits = torch.randn(batch_size, 8, 100, device=device)
    targets = torch.randint(0, 100, (batch_size, 8), device=device)
    mask = torch.ones(batch_size, 8, device=device).bool()
    
    # Create different wrongness factors
    wrongness_factor = torch.tensor([1.0, 2.0, 0.5, 3.0], device=device)
    
    # Create training context with no entropy penalty
    training_ctx = create_test_training_context(enable_entropy_penalty=False, batch_size=batch_size)
    
    # Apply modifications
    modified_losses = apply_per_sample_modifications(
        per_sample_losses, logits, targets, mask, training_ctx, iter_num=5000, wrongness_factor=wrongness_factor
    )
    
    # Check that losses are scaled correctly
    expected_losses = per_sample_losses * wrongness_factor
    assert torch.allclose(modified_losses, expected_losses), \
        f"Expected {expected_losses}, got {modified_losses}"


def test_wrongness_factor_shape_validation():
    """Test that wrongness factor shape validation works correctly."""
    batch_size = 4
    device = torch.device('cpu')
    
    # Create test data
    per_sample_losses = torch.rand(batch_size, device=device)
    logits = torch.randn(batch_size, 8, 100, device=device)
    targets = torch.randint(0, 100, (batch_size, 8), device=device)
    mask = torch.ones(batch_size, 8, device=device).bool()
    
    # Create wrongness factor with wrong shape
    wrongness_factor = torch.ones(batch_size + 1, device=device)  # Wrong size
    
    # Create training context
    training_ctx = create_test_training_context(enable_entropy_penalty=False, batch_size=batch_size)
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="wrongness_factor shape"):
        apply_per_sample_modifications(
            per_sample_losses, logits, targets, mask, training_ctx, iter_num=5000, wrongness_factor=wrongness_factor
        )


if __name__ == "__main__":
    # Run basic tests
    test_per_sample_losses_basic()
    test_per_sample_losses_soft_targets()
    test_entropy_per_sample_matches_global()
    test_apply_per_sample_modifications_no_penalty()
    test_apply_per_sample_modifications_with_entropy()
    test_wrongness_factor_scaling()
    test_wrongness_factor_shape_validation()
    print("All tests passed!")
