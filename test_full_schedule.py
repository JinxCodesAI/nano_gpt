"""
Test script to simulate full generation with both schedule modes
Shows actual masked token counts at each iteration
"""
import torch
from sample_utils import apply_remasking_step, linear_remasking_schedule

def simulate_generation(schedule_mode='ratio', iterations=15, seq_len=16384, batch_size=1):
    """Simulate a full generation to show masked token counts"""
    
    device = 'cpu'
    mask_token_id = 99
    start_ratio = 0.95
    end_ratio = 0.05
    
    # Create mock model with critic head
    class MockModel:
        class Config:
            add_critic_head = True
        config = Config()
        
        def critic_scores(self, tokens):
            # Return mock critic scores (logits) with some variation
            # Simulate decreasing wrongness as generation progresses
            scores = torch.randn(tokens.shape, device=tokens.device) * 0.5
            return scores
    
    model = MockModel()
    
    # Start with all masked
    tokens = torch.full((batch_size, seq_len), mask_token_id, dtype=torch.long, device=device)
    
    print("="*70)
    print(f"Simulating {schedule_mode.upper()} mode generation")
    print("="*70)
    print(f"Sequence length: {seq_len}")
    print(f"Iterations: {iterations}")
    print(f"Start ratio: {start_ratio:.1%}, End ratio: {end_ratio:.1%}")
    print("="*70)
    
    for iteration in range(iterations):
        # Count currently masked tokens
        masked_count = (tokens == mask_token_id).sum().item()
        total = tokens.numel()
        masked_pct = masked_count / total * 100
        
        print(f"Iteration {iteration+1}/{iterations}: {masked_count}/{total} masked ({masked_pct:.1f}%)")
        
        # Predict tokens (unmask everything for simulation)
        prediction_tokens = torch.randint(0, 90, (batch_size, seq_len), device=device)
        
        # Apply remasking for next iteration (except last)
        if iteration < iterations - 1:
            remasked = apply_remasking_step(
                tokens=tokens,
                prediction_tokens=prediction_tokens,
                iteration=iteration,
                iterations=iterations,
                schedule_type='linear',
                start_ratio=start_ratio,
                end_ratio=end_ratio,
                mask_token_id=mask_token_id,
                device=device,
                base_model=model,
                intelligent_remasking=False,
                verbose=False,
                schedule_mode=schedule_mode,
            )
            
            if remasked is None:
                print(f"  -> Early termination: no tokens exceed threshold")
                tokens = prediction_tokens
                break
            
            tokens = remasked
        else:
            tokens = prediction_tokens
    
    print("="*70)
    print()

def main():
    seq_len = 16384
    iterations = 15
    
    # Test ratio mode
    simulate_generation(schedule_mode='ratio', iterations=iterations, seq_len=seq_len)
    
    # Test threshold mode
    simulate_generation(schedule_mode='threshold', iterations=iterations, seq_len=seq_len)
    
    print("="*70)
    print("COMPARISON")
    print("="*70)
    print("RATIO mode:")
    print("  - Masks a fixed percentage of tokens at each iteration")
    print("  - Percentage decreases linearly from start_ratio to end_ratio")
    print("  - Always completes all iterations")
    print("  - Masked count is predictable and consistent")
    print()
    print("THRESHOLD mode:")
    print("  - Masks ALL tokens with wrongness above threshold")
    print("  - Threshold increases linearly (inverted from ratio)")
    print("  - Can finish early if no tokens exceed threshold")
    print("  - Masked count depends on actual token wrongness scores")
    print("  - More adaptive to generation quality")
    print("="*70)

if __name__ == '__main__':
    main()

