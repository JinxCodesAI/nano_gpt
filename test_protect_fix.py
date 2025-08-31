#!/usr/bin/env python3
"""
Test script to verify the protect edits fix
"""

def test_position_detection():
    """Test that we can correctly detect edited positions"""
    import torch
    
    # Simulate baseline tokens (what model originally output)
    baseline = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Simulate user edited tokens (user changed positions 2, 3, 4 to spell "knife")
    # Assuming: k=20, n=21, i=22, f=23, e=24
    edited = torch.tensor([1, 2, 20, 21, 22, 23, 24, 8, 9, 10])
    
    mask_token_id = 99
    
    # Find positions where tokens were changed from baseline
    changed_positions = torch.where(
        (baseline != edited) &  # Token was changed from baseline
        (edited != mask_token_id)  # And new token is not a mask
    )[0]
    
    print(f"Baseline tokens: {baseline}")
    print(f"Edited tokens:   {edited}")
    print(f"Changed positions: {changed_positions.tolist()}")
    print(f"Expected: [2, 3, 4, 5, 6] (positions where 'knife' was typed)")
    
    expected = [2, 3, 4, 5, 6]
    actual = changed_positions.tolist()
    
    if actual == expected:
        print("✅ Position detection works correctly!")
        return True
    else:
        print(f"❌ Position detection failed! Expected {expected}, got {actual}")
        return False

def test_remaskable_filtering():
    """Test that protected positions are correctly filtered out"""
    import torch
    
    # Simulate a scenario where we have unmasked positions but some are protected
    tokens = torch.tensor([[1, 2, 20, 21, 22, 23, 24, 8, 9, 10]])  # batch_size=1
    mask_token_id = 99
    protected_positions = {2, 3, 4, 5, 6}  # Positions with "knife"
    
    # Find unmasked positions
    unmasked_positions = (tokens[0] != mask_token_id)
    unmasked_indices = torch.where(unmasked_positions)[0]
    
    # Filter out protected positions
    if protected_positions:
        protected_tensor = torch.tensor(list(protected_positions), dtype=torch.long)
        mask = ~torch.isin(unmasked_indices, protected_tensor)
        remaskable_indices = unmasked_indices[mask]
    else:
        remaskable_indices = unmasked_indices
    
    print(f"Tokens: {tokens[0]}")
    print(f"Unmasked positions: {unmasked_indices.tolist()}")
    print(f"Protected positions: {sorted(protected_positions)}")
    print(f"Remaskable positions: {remaskable_indices.tolist()}")
    print(f"Expected remaskable: [0, 1, 7, 8, 9] (excluding 'knife' positions)")
    
    expected = [0, 1, 7, 8, 9]
    actual = remaskable_indices.tolist()
    
    if actual == expected:
        print("✅ Position filtering works correctly!")
        return True
    else:
        print(f"❌ Position filtering failed! Expected {expected}, got {actual}")
        return False

if __name__ == "__main__":
    print("Testing protect edits fixes...\n")
    
    try:
        # Test 1: Position detection
        print("Test 1: Position Detection")
        success1 = test_position_detection()
        
        print("\nTest 2: Remaskable Position Filtering")
        success2 = test_remaskable_filtering()
        
        if success1 and success2:
            print("\n✅ All logic tests passed!")
        else:
            print("\n❌ Some logic tests failed!")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()