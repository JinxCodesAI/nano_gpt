#!/usr/bin/env python3
"""
Launch script for Diffusion Inference Visualizer
Checks dependencies and launches the new inference visualization application
"""

import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import PyQt6
        print("✓ PyQt6 found")
    except ImportError:
        print("✗ PyQt6 not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt6"])
        
    try:
        import torch
        print("✓ PyTorch found")
    except ImportError:
        print("✗ PyTorch not found. Please install PyTorch manually.")
        print("Visit: https://pytorch.org/get-started/locally/")
        return False
        
    try:
        import numpy
        print("✓ NumPy found")
    except ImportError:
        print("✗ NumPy not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
        
    return True

def main():
    """Main launcher"""
    print("🚀 Diffusion Inference Visualizer - Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("📦 Checking dependencies...")
    if not check_dependencies():
        print("❌ Failed to install required dependencies!")
        return 1
        
    print("✅ All dependencies satisfied!")
    print("🚀 Starting Inference Visualizer...")
    
    # Import and run the inference visualizer
    try:
        # Change to parent directory so sample.py imports work correctly
        import os
        original_cwd = os.getcwd()
        parent_dir = Path(__file__).parent.parent
        os.chdir(parent_dir)
        
        # Add visualization directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        try:
            from inference_visualizer import main as run_visualizer
            run_visualizer()
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("\n💡 Troubleshooting:")
        print("- Ensure model files exist in ../out/ directory")
        print("- Check that data files exist in ../data/shakespeare_char/")
        print("- Try running: pip install PyQt6 torch numpy")
        print("- For old explorer, use: python model_explorer.py")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())