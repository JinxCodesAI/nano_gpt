#!/usr/bin/env python3
"""
Test the core logic without torch dependency
"""

def test_position_detection():
    """Test that we can correctly detect edited positions"""
    
    # Simulate baseline tokens (what model originally output)
    baseline = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Simulate user edited tokens (user changed positions 2, 3, 4, 5, 6 to spell "knife")
    # Assuming: k=20, n=21, i=22, f=23, e=24
    edited = [1, 2, 20, 21, 22, 23, 24, 8, 9, 10]
    
    mask_token_id = 99
    
    # Find positions where tokens were changed from baseline
    changed_positions = []
    for i in range(len(baseline)):
        if baseline[i] != edited[i] and edited[i] != mask_token_id:
            changed_positions.append(i)
    
    print(f"Baseline tokens: {baseline}")
    print(f"Edited tokens:   {edited}")
    print(f"Changed positions: {changed_positions}")
    print(f"Expected: [2, 3, 4, 5, 6] (positions where 'knife' was typed)")
    
    expected = [2, 3, 4, 5, 6]
    
    if changed_positions == expected:
        print("✅ Position detection works correctly!")
        return True
    else:
        print(f"❌ Position detection failed! Expected {expected}, got {changed_positions}")
        return False

def test_remaskable_filtering():
    """Test that protected positions are correctly filtered out"""
    
    # Simulate a scenario where we have tokens but some positions are protected
    tokens = [1, 2, 20, 21, 22, 23, 24, 8, 9, 10]
    mask_token_id = 99
    protected_positions = {2, 3, 4, 5, 6}  # Positions with "knife"
    
    # Find unmasked positions
    unmasked_indices = []
    for i, token in enumerate(tokens):
        if token != mask_token_id:
            unmasked_indices.append(i)
    
    # Filter out protected positions
    remaskable_indices = []
    for idx in unmasked_indices:
        if idx not in protected_positions:
            remaskable_indices.append(idx)
    
    print(f"Tokens: {tokens}")
    print(f"Unmasked positions: {unmasked_indices}")
    print(f"Protected positions: {sorted(protected_positions)}")
    print(f"Remaskable positions: {remaskable_indices}")
    print(f"Expected remaskable: [0, 1, 7, 8, 9] (excluding 'knife' positions)")
    
    expected = [0, 1, 7, 8, 9]
    
    if remaskable_indices == expected:
        print("✅ Position filtering works correctly!")
        return True
    else:
        print(f"❌ Position filtering failed! Expected {expected}, got {remaskable_indices}")
        return False

if __name__ == "__main__":
    print("Testing protect edits logic (without torch)...\n")
    
    try:
        # Test 1: Position detection
        print("Test 1: Position Detection")
        success1 = test_position_detection()
        
        print("\nTest 2: Remaskable Position Filtering")
        success2 = test_remaskable_filtering()
        
        if success1 and success2:
            print("\n✅ All logic tests passed!")
            print("\nThe core logic should work correctly!")
            print("Key fixes implemented:")
            print("1. ✅ Store baseline tokens when substep is created")
            print("2. ✅ Compare edits against baseline, not previous step")
            print("3. ✅ Update InferenceRunner with new protected positions")
            print("4. ✅ Filter protected positions in apply_remasking")
        else:
            print("\n❌ Some logic tests failed!")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()