#!/usr/bin/env python3
"""
Test script to check if train_utils imports work correctly
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

print("Testing train_utils imports...")

try:
    # Test individual imports that were failing
    print("1. Testing TrainingContext...")
    from train_utils import TrainingContext
    print("   ✓ TrainingContext imported")
    
    print("2. Testing apply_target_driven_sticky_masking_gpu...")
    from train_utils import apply_target_driven_sticky_masking_gpu
    print("   ✓ apply_target_driven_sticky_masking_gpu imported")
    
    print("3. Testing apply_synthetic_corruption...")
    from train_utils import apply_synthetic_corruption
    print("   ✓ apply_synthetic_corruption imported")
    
    print("4. Testing find_double_newline_indices...")
    from train_utils import find_double_newline_indices
    print("   ✓ find_double_newline_indices imported")
    
    print("5. Testing load_synthetic_model...")
    from train_utils import load_synthetic_model
    print("   ✓ load_synthetic_model imported")
    
    print("6. Testing apply_random_masking_gpu...")
    from train_utils import apply_random_masking_gpu
    print("   ✓ apply_random_masking_gpu imported")
    
    print("\n✅ All train_utils imports successful!")
    
    # Now test if we can import the problematic functions that were missing
    print("\n7. Testing if missing functions are now available...")
    
    # These should now work because we defined them as wrappers in model_explorer.py
    print("   Cannot test wrapper functions here (they're in model_explorer.py)")
    print("   But the original missing functions were:")
    print("   - apply_random_corruption_gpu")  
    print("   - apply_sticky_corruption_gpu")
    print("   - apply_fragment_corruption_gpu")
    print("   These are now defined as compatibility wrappers.")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)