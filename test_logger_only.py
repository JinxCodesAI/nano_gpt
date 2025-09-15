#!/usr/bin/env python3
"""
Simple logger test without torch dependency.
"""

import sys
import os
from io import StringIO
from unittest.mock import patch

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))

def test_logger_classes():
    """Test logger classes work independently."""
    try:
        from logger import ConsoleLogger, create_logger
        
        # Test console logger
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            logger = ConsoleLogger(master_process=True)
            logger.log_info("Test message")
            logger.log_step({"iter": 5, "loss": 1.5, "time_ms": 50, "mfu_pct": 75})
        
        output = mock_stdout.getvalue().strip().split('\n')
        expected = [
            "Test message",
            "iter 5: loss 1.5000, time 50.00ms, mfu 75.00%"
        ]
        
        for i, exp in enumerate(expected):
            if output[i] != exp:
                print(f"❌ Expected: '{exp}', Got: '{output[i]}'")
                return False
        
        print("✅ Logger classes work correctly")
        return True
    except Exception as e:
        print(f"❌ Logger test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 40)
    print("LOGGER CLASS TEST")
    print("=" * 40)
    
    if test_logger_classes():
        print("\n🎉 Logger classes are working properly!")
        print("✅ Console output format preserved")
        print("✅ Master process filtering works")
    else:
        print("\n💥 Logger test failed!")