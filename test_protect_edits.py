#!/usr/bin/env python3
"""
Test script for the protect edits functionality
"""

import sys
import torch
from pathlib import Path

# Add visualization directory to path
sys.path.append(str(Path(__file__).parent / "visualization"))

def test_apply_remasking_with_protection():
    """Test that apply_remasking respects protected positions"""
    from diffusion_utils import apply_remasking
    
    print("Testing apply_remasking with protected positions...")
    
    # Create test tokens (batch_size=1, seq_len=10)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    mask_token_id = 11
    device = torch.device('cpu')
    protected_positions = {2, 5, 8}  # Protect positions 2, 5, and 8
    
    print(f"Original tokens: {tokens}")
    print(f"Protected positions: {protected_positions}")
    
    # Test with high remask ratio to force remasking
    remasked_tokens = apply_remasking(
        tokens=tokens,
        remask_ratio=0.8,  # Remask 80% of tokens
        remasking_model=None,
        randomness_strength=1.0,  # Pure random
        mask_token_id=mask_token_id,
        device=device,
        base_model=None,
        intelligent_remasking=False,
        verbose=True,
        protected_positions=protected_positions
    )
    
    print(f"Remasked tokens: {remasked_tokens}")
    
    # Check that protected positions were not masked
    for pos in protected_positions:
        original_token = tokens[0, pos].item()
        remasked_token = remasked_tokens[0, pos].item()
        if remasked_token == mask_token_id:
            print(f"ERROR: Protected position {pos} was masked!")
            return False
        elif remasked_token != original_token:
            print(f"ERROR: Protected position {pos} was changed from {original_token} to {remasked_token}!")
            return False
        else:
            print(f"SUCCESS: Protected position {pos} kept original token {original_token}")
    
    return True

def test_empty_protected_positions():
    """Test that apply_remasking works with empty protected positions"""
    from diffusion_utils import apply_remasking
    
    print("\nTesting apply_remasking with empty protected positions...")
    
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    mask_token_id = 6
    device = torch.device('cpu')
    
    # Test with empty set
    remasked_tokens = apply_remasking(
        tokens=tokens,
        remask_ratio=0.5,
        remasking_model=None,
        randomness_strength=1.0,
        mask_token_id=mask_token_id,
        device=device,
        base_model=None,
        intelligent_remasking=False,
        verbose=False,
        protected_positions=set()  # Empty set
    )
    
    print(f"Original tokens: {tokens}")
    print(f"Remasked tokens: {remasked_tokens}")
    print("SUCCESS: Function works with empty protected positions")
    return True

if __name__ == "__main__":
    print("Running protect edits feature tests...\n")
    
    try:
        # Test 1: Basic protection functionality
        success1 = test_apply_remasking_with_protection()
        
        # Test 2: Empty protected positions
        success2 = test_empty_protected_positions()
        
        if success1 and success2:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)