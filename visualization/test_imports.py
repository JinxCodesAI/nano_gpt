#!/usr/bin/env python3
"""
Test script to check if enhanced_explorer.py imports work correctly
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    # Test train_utils imports
    from train_utils import (
        TrainingContext, UnmaskingStage, apply_target_driven_sticky_masking_gpu,
        find_double_newline_indices, StickyStageConfig
    )
    print("✓ train_utils imports successful")
    
    # Test model imports  
    from model import GPT, GPTConfig
    print("✓ model imports successful")
    
    print("All imports successful!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)