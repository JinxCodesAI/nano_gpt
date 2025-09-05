#!/usr/bin/env python3
"""
Test the exact training call that was failing.
"""

import torch
from contextlib import nullcontext
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import GPTConfig, GPT, ModelMode
from training_utils import TrainingContext
from training_utils.loss_processing import calculate_predicted_masking_ratio, calculate_wrongness_factor

def test_training_call():
    """Test the exact call from train_run2.py that was failing."""
    print("Testing the exact training call...")
    
    device = 'cpu'
    vocab_size = 100
    block_size = 64
    batch_size = 4
    
    # Create training context (similar to train_run2.py)
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
    
    training_ctx.judge_model = GPT(judge_config)
    training_ctx.judge_model.to(device)
    training_ctx.judge_model.eval()
    
    # Create test data that matches train_run2.py
    # logits from: logits, loss = model(X, Y)
    logits = torch.randn(batch_size, block_size, vocab_size, device=device)
    
    # Y can be either 2D or 3D depending on training configuration
    # Test both cases
    
    print("\nTest Case 1: Y is 2D (hard targets)")
    Y_2d = torch.randint(0, vocab_size - 10, (batch_size, block_size), device=device)
    mask = torch.rand(batch_size, block_size, device=device) < 0.3
    ctx = nullcontext()
    
    try:
        # This is the exact call from train_run2.py line 571
        predicted_masking_ratios = calculate_predicted_masking_ratio(logits, Y_2d, mask, training_ctx, ctx)
        
        print(f"  ✅ Success! Shape: {predicted_masking_ratios.shape}")
        print(f"  Values: {predicted_masking_ratios.tolist()}")
        
        # Test the rest of the pipeline
        real_masking_ratios = mask.float().mean(dim=1)
        wrongness_factor = calculate_wrongness_factor(predicted_masking_ratios, real_masking_ratios)
        
        print(f"  Real ratios: {real_masking_ratios.tolist()}")
        print(f"  Wrongness factors: {wrongness_factor.tolist()}")
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nTest Case 2: Y is 3D (soft targets)")
    Y_3d = torch.randn(batch_size, block_size, vocab_size, device=device)
    Y_3d = torch.softmax(Y_3d, dim=-1)
    
    try:
        # This is the exact call from train_run2.py line 571
        predicted_masking_ratios = calculate_predicted_masking_ratio(logits, Y_3d, mask, training_ctx, ctx)
        
        print(f"  ✅ Success! Shape: {predicted_masking_ratios.shape}")
        print(f"  Values: {predicted_masking_ratios.tolist()}")
        
        # Test the rest of the pipeline
        real_masking_ratios = mask.float().mean(dim=1)
        wrongness_factor = calculate_wrongness_factor(predicted_masking_ratios, real_masking_ratios)
        
        print(f"  Real ratios: {real_masking_ratios.tolist()}")
        print(f"  Wrongness factors: {wrongness_factor.tolist()}")
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nTest Case 3: No judge model (fallback)")
    training_ctx.judge_model = None
    
    try:
        predicted_masking_ratios = calculate_predicted_masking_ratio(logits, Y_2d, mask, training_ctx, ctx)
        
        print(f"  ✅ Success! Shape: {predicted_masking_ratios.shape}")
        print(f"  Values (should be all 1s): {predicted_masking_ratios.tolist()}")
        
        if torch.allclose(predicted_masking_ratios, torch.ones_like(predicted_masking_ratios)):
            print("  ✅ Correctly returns all 1s when no judge model")
        else:
            print("  ❌ Should return all 1s when no judge model")
            return False
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run the test."""
    print("Testing exact training call that was failing...")
    
    try:
        success = test_training_call()
        
        if success:
            print("\n🎉 All tests passed!")
            print("The ValueError: too many values to unpack (expected 2) is fixed!")
            print("Training should now work correctly.")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
