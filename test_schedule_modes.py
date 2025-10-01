"""
Test script to verify ratio vs threshold schedule modes
"""
import torch
from sample_utils import apply_remasking_step

def test_schedule_modes():
    """Test both ratio and threshold schedule modes"""
    
    # Setup
    device = 'cpu'
    batch_size = 2
    seq_len = 100
    mask_token_id = 99
    
    # Create mock model with critic head
    class MockModel:
        class Config:
            add_critic_head = True
        config = Config()
        
        def critic_scores(self, tokens):
            # Return mock critic scores (logits)
            # Higher values = more likely to be wrong
            scores = torch.randn(tokens.shape, device=tokens.device)
            return scores
    
    model = MockModel()
    
    # Create prediction tokens (all non-masked)
    prediction_tokens = torch.randint(0, 90, (batch_size, seq_len), device=device)
    
    print("="*60)
    print("Testing RATIO mode (default)")
    print("="*60)
    
    # Test ratio mode with 50% masking
    result_ratio = apply_remasking_step(
        tokens=prediction_tokens.clone(),
        prediction_tokens=prediction_tokens.clone(),
        iteration=0,
        iterations=10,
        schedule_type='linear',
        start_ratio=0.5,
        end_ratio=0.5,
        mask_token_id=mask_token_id,
        device=device,
        base_model=model,
        intelligent_remasking=False,
        verbose=True,
        schedule_mode='ratio',
    )
    
    if result_ratio is not None:
        masked_count = (result_ratio == mask_token_id).sum().item()
        total = result_ratio.numel()
        print(f"Ratio mode: {masked_count}/{total} masked ({masked_count/total:.1%})")
        print(f"Expected: ~{int(total*0.5)}/{total} masked (50.0%)")
    else:
        print("ERROR: Ratio mode returned None")
    
    print("\n" + "="*60)
    print("Testing THRESHOLD mode")
    print("="*60)
    print("Note: threshold is inverted, so start_ratio=0.95 means threshold=0.05")
    print("      (mask tokens with wrongness > 0.05, i.e., almost everything)")
    print()

    # Test threshold mode with start_ratio=0.95 (threshold=0.05, should mask many tokens)
    result_threshold_high = apply_remasking_step(
        tokens=prediction_tokens.clone(),
        prediction_tokens=prediction_tokens.clone(),
        iteration=0,
        iterations=10,
        schedule_type='linear',
        start_ratio=0.95,  # Inverted: threshold=0.05 (mask almost everything)
        end_ratio=0.95,
        mask_token_id=mask_token_id,
        device=device,
        base_model=model,
        intelligent_remasking=False,
        verbose=True,
        schedule_mode='threshold',
    )

    if result_threshold_high is not None:
        masked_count = (result_threshold_high == mask_token_id).sum().item()
        total = result_threshold_high.numel()
        print(f"Threshold mode (ratio=0.95, threshold=0.05): {masked_count}/{total} masked ({masked_count/total:.1%})")
    else:
        print("Threshold mode (ratio=0.95, threshold=0.05): No tokens masked (early termination)")

    # Test threshold mode with end_ratio=0.05 (threshold=0.95, should mask few tokens)
    result_threshold_low = apply_remasking_step(
        tokens=prediction_tokens.clone(),
        prediction_tokens=prediction_tokens.clone(),
        iteration=0,
        iterations=10,
        schedule_type='linear',
        start_ratio=0.05,  # Inverted: threshold=0.95 (mask only very wrong tokens)
        end_ratio=0.05,
        mask_token_id=mask_token_id,
        device=device,
        base_model=model,
        intelligent_remasking=False,
        verbose=True,
        schedule_mode='threshold',
    )

    if result_threshold_low is not None:
        masked_count = (result_threshold_low == mask_token_id).sum().item()
        total = result_threshold_low.numel()
        print(f"Threshold mode (ratio=0.05, threshold=0.95): {masked_count}/{total} masked ({masked_count/total:.1%})")
    else:
        print("Threshold mode (ratio=0.05, threshold=0.95): No tokens masked (early termination)")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)

if __name__ == '__main__':
    test_schedule_modes()

