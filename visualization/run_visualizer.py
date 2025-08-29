#!/usr/bin/env python3
"""
Direct launcher for Diffusion Inference Visualizer
Bypasses dependency checks - assumes you have PyQt6, torch, numpy installed
"""

import sys
import os
from pathlib import Path

def main():
    """Direct launcher"""
    print("🚀 Starting Diffusion Inference Visualizer...")
    
    # Change to parent directory so sample.py imports work correctly
    original_cwd = os.getcwd()
    parent_dir = Path(__file__).parent.parent
    print(f"Changing to: {parent_dir}")
    os.chdir(parent_dir)
    
    # Add visualization directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        from inference_visualizer import main as run_visualizer
        run_visualizer()
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("\n💡 Troubleshooting:")
        print("- Make sure you have: pip install PyQt6 torch numpy")
        print("- Ensure model files exist in ./out/ directory")
        print("- Check that data files exist in ./data/shakespeare_char/")
        return 1
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
        
    return 0

if __name__ == "__main__":
    sys.exit(main())