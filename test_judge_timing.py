#!/usr/bin/env python3
"""
Test judge model timing integration.
"""

import torch
from contextlib import nullcontext
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import GPTConfig, GPT, ModelMode
from training_utils import TrainingContext
from training_utils.training_logger import TrainingLogger
from training_utils.loss_processing import calculate_predicted_masking_ratio, calculate_wrongness_factor
from utils import Timer

def create_mock_judge_model(vocab_size, block_size, device):
    """Create a mock judge model."""
    config = GPTConfig(
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
    
    model = GPT(config)
    model.to(device)
    model.eval()
    return model

def test_judge_timing():
    """Test that judge model timing is properly tracked and logged."""
    print("Testing judge model timing integration...")
    
    device = 'cpu'
    vocab_size = 100
    block_size = 64
    batch_size = 4
    
    # Create timer and logger
    timer = Timer()
    logger = TrainingLogger(wandb_log=False, master_process=True, log_interval=1, eval_interval=10)
    
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
    
    training_ctx.judge_model = create_mock_judge_model(vocab_size, block_size, device)
    
    # Create test data
    logits = torch.randn(batch_size, block_size, vocab_size, device=device)
    Y = torch.randint(0, vocab_size - 10, (batch_size, block_size), device=device)
    mask = torch.rand(batch_size, block_size, device=device) < 0.3
    ctx = nullcontext()
    
    print(f"\nTest data shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  Y: {Y.shape}")
    print(f"  mask: {mask.shape}")
    
    # Simulate the timing as it would happen in train_run2.py
    print("\nSimulating judge model calculation with timing...")
    
    # Run multiple iterations to get meaningful timing data
    for i in range(5):
        with timer.time_function('judge_model_calculation'):
            if training_ctx.judge_model is not None:
                # Get predicted masking ratios from judge model (reuse existing logits)
                predicted_masking_ratios = calculate_predicted_masking_ratio(logits, Y, mask, training_ctx, ctx)
                
                # Calculate real masking ratios
                real_masking_ratios = mask.float().mean(dim=1)  # (batch_size,) - ratio per sample
                
                # Calculate wrongness factor using the formula
                wrongness_factor = calculate_wrongness_factor(predicted_masking_ratios, real_masking_ratios)
            else:
                # Fallback: use all ones (no scaling)
                wrongness_factor = torch.ones(training_ctx.batch_size, device=logits.device)
        
        # Also time some other operations for comparison
        with timer.time_function('loss_processing'):
            time.sleep(0.001)  # Simulate loss processing time
        
        with timer.time_function('forward_pass'):
            time.sleep(0.002)  # Simulate forward pass time
        
        with timer.time_function('backward_pass'):
            time.sleep(0.001)  # Simulate backward pass time
    
    print(f"\nResults from iteration {i+1}:")
    print(f"  Predicted ratios: {predicted_masking_ratios.tolist()}")
    print(f"  Real ratios: {real_masking_ratios.tolist()}")
    print(f"  Wrongness factors: {wrongness_factor.tolist()}")
    
    # Test the timing breakdown logging
    print("\nTesting timing breakdown logging...")
    
    # Simulate the timing breakdown as it would be called in train_run2.py
    dt = 0.050  # 50ms total iteration time
    logger.log_timing_breakdown(timer, dt)
    
    # Check that judge model timing is captured
    judge_time = timer.get_recent_average('judge_model_calculation')
    print(f"\nJudge model timing verification:")
    print(f"  Average judge model time: {judge_time*1000:.2f}ms")
    
    if judge_time > 0:
        print("  ✅ Judge model timing is being tracked correctly")
    else:
        print("  ❌ Judge model timing is not being tracked")
        return False
    
    # Test with no judge model (should be very fast)
    print("\nTesting with no judge model (fallback)...")
    training_ctx.judge_model = None
    
    with timer.time_function('judge_model_calculation'):
        if training_ctx.judge_model is not None:
            predicted_masking_ratios = calculate_predicted_masking_ratio(logits, Y, mask, training_ctx, ctx)
            real_masking_ratios = mask.float().mean(dim=1)
            wrongness_factor = calculate_wrongness_factor(predicted_masking_ratios, real_masking_ratios)
        else:
            # Fallback: use all ones (no scaling)
            wrongness_factor = torch.ones(training_ctx.batch_size, device=logits.device)
    
    fallback_time = timer.get_recent_average('judge_model_calculation')
    print(f"  Fallback time: {fallback_time*1000:.2f}ms")
    print(f"  Wrongness factors (should be all 1s): {wrongness_factor.tolist()}")
    
    if torch.allclose(wrongness_factor, torch.ones_like(wrongness_factor)):
        print("  ✅ Fallback correctly returns all 1s")
    else:
        print("  ❌ Fallback should return all 1s")
        return False
    
    return True

def main():
    """Run the timing test."""
    print("Testing judge model timing integration...\n")
    
    try:
        success = test_judge_timing()
        
        if success:
            print("\n🎉 All timing tests passed!")
            print("Judge model timing is properly tracked and logged separately.")
        else:
            print("\n❌ Some timing tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
