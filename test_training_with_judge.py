#!/usr/bin/env python3
"""
Test training with judge model enabled.

This script creates mock models and runs a few training iterations to verify:
1. Judge model loads correctly
2. Wrongness factor values are in expected range (0-10)
3. Training proceeds without errors
4. Loss calculation works properly
"""

import os
import sys
import torch
import tempfile
from contextlib import nullcontext

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import GPTConfig, GPT, ModelMode
from training_utils import TrainingContext
from training_utils.model_initializer import ModelInitializer
from training_utils.loss_processing import (
    calculate_per_sample_losses,
    apply_per_sample_modifications,
    calculate_predicted_masking_ratio,
    calculate_wrongness_factor
)

def create_mock_checkpoint(model, filepath):
    """Create a mock checkpoint file for testing."""
    checkpoint = {
        'model': model.state_dict(),
        'model_args': {
            'n_layer': model.config.n_layer,
            'n_head': model.config.n_head,
            'n_embd': model.config.n_embd,
            'block_size': model.config.block_size,
            'bias': model.config.bias,
            'vocab_size': model.config.vocab_size,
            'dropout': model.config.dropout,
            'attention_type': model.config.attention_type,
            'use_rope': model.config.use_rope,
            'mode': model.config.mode,
            'num_token_classes': getattr(model.config, 'num_token_classes', 2),
            'cls_token_id': getattr(model.config, 'cls_token_id', None)
        },
        'iter_num': 0,
        'best_val_loss': 999.0,
        'config': {}
    }
    
    torch.save(checkpoint, filepath)
    print(f"Created mock checkpoint: {filepath}")

def test_training_with_judge_model():
    """Test training loop with judge model enabled."""
    print("Testing training with judge model...")
    
    device = 'cpu'
    vocab_size = 100
    block_size = 64
    batch_size = 4
    
    # Create temporary directory for mock checkpoints
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock unmasking model
        unmasking_config = GPTConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
            mode=ModelMode.LANGUAGE_MODEL,
            attention_type='bidirectional'
        )
        unmasking_model = GPT(unmasking_config)
        unmasking_checkpoint_path = os.path.join(temp_dir, 'unmasking_model.pt')
        create_mock_checkpoint(unmasking_model, unmasking_checkpoint_path)
        
        # Create mock judge model
        judge_config = GPTConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
            mode=ModelMode.SEQUENCE_SCORER,
            attention_type='bidirectional',
            cls_token_id=vocab_size - 1
        )
        judge_model = GPT(judge_config)
        judge_checkpoint_path = os.path.join(temp_dir, 'judge_model.pt')
        create_mock_checkpoint(judge_model, judge_checkpoint_path)
        
        # Create training context
        training_ctx = TrainingContext(
            training_type='unmasking',
            batch_size=batch_size,
            block_size=block_size,
            device=device,
            device_type=device,
            extended_vocab_size=vocab_size,
            mask_token_id=vocab_size - 10,
            cls_token_id=vocab_size - 1,
            enable_entropy_penalty=False
        )
        
        # Initialize model initializer and load models
        model_initializer = ModelInitializer(device, device)
        
        # Load unmasking model
        unmasking_model_loaded = model_initializer.load_unmasking_model(
            unmasking_checkpoint_path, vocab_size, block_size
        )
        training_ctx.unmasking_model = unmasking_model_loaded
        
        # Load judge model
        judge_model_loaded = model_initializer.load_judge_model(
            judge_checkpoint_path, vocab_size, block_size
        )
        training_ctx.judge_model = judge_model_loaded
        
        print("✓ Models loaded successfully")
        
        # Simulate training iteration
        ctx = nullcontext()
        
        # Create mock training data
        X = torch.randint(0, vocab_size - 10, (batch_size, block_size), device=device)
        Y = torch.randint(0, vocab_size - 10, (batch_size, block_size), device=device)
        mask = torch.rand(batch_size, block_size, device=device) < 0.3
        
        # Create mock main model for logits
        main_model_config = GPTConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
            mode=ModelMode.LANGUAGE_MODEL,
            attention_type='bidirectional'
        )
        main_model = GPT(main_model_config)
        main_model.to(device)
        main_model.eval()
        
        # Forward pass to get logits
        with torch.no_grad():
            logits, _ = main_model(X, None)
        
        print(f"Input shapes - X: {X.shape}, Y: {Y.shape}, mask: {mask.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Mask ratio: {mask.float().mean().item():.3f}")
        
        # Test the complete pipeline
        if mask.any():
            # Step 1: Calculate per-sample losses
            per_sample_losses, per_sample_mask_counts = calculate_per_sample_losses(logits, Y, mask)
            print(f"Per-sample losses: {per_sample_losses.tolist()}")
            
            # Step 2: Calculate wrongness factor using judge model
            predicted_masking_ratios = calculate_predicted_masking_ratio(Y, mask, training_ctx, ctx)
            real_masking_ratios = mask.float().mean(dim=1)
            wrongness_factor = calculate_wrongness_factor(predicted_masking_ratios, real_masking_ratios)
            
            print(f"Predicted ratios: {predicted_masking_ratios.tolist()}")
            print(f"Real ratios: {real_masking_ratios.tolist()}")
            print(f"Wrongness factors: {wrongness_factor.tolist()}")
            
            # Verify wrongness factor is in expected range
            assert torch.all(wrongness_factor >= 0.0), "Wrongness factor should be >= 0"
            assert torch.all(wrongness_factor <= 10.0), "Wrongness factor should be <= 10"
            
            # Step 3: Apply per-sample modifications
            modified_per_sample_losses = apply_per_sample_modifications(
                per_sample_losses, logits, Y, mask, training_ctx, 0, wrongness_factor
            )
            
            print(f"Modified losses: {modified_per_sample_losses.tolist()}")
            
            # Step 4: Final aggregation
            valid_samples = per_sample_mask_counts > 0
            if valid_samples.any():
                final_loss = modified_per_sample_losses[valid_samples].mean()
                print(f"Final aggregated loss: {final_loss.item():.4f}")
                
                # Verify loss is reasonable
                assert not torch.isnan(final_loss), "Loss should not be NaN"
                assert not torch.isinf(final_loss), "Loss should not be infinite"
                assert final_loss.item() > 0, "Loss should be positive"
                
                print("✓ Loss calculation completed successfully")
            else:
                print("No valid samples with masks")
        else:
            print("No masked positions in batch")
        
        print("✓ Training simulation completed successfully")

def main():
    """Run the training validation test."""
    print("Running training validation with judge model...\n")
    
    try:
        test_training_with_judge_model()
        print("\n🎉 Training validation passed!")
        print("Judge model integration is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Training validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
