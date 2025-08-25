#!/usr/bin/env python3
"""
Launch script for Model Explorer
Checks dependencies and launches the application
"""

import sys
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
    print("🚀 Diffusion Model Explorer - Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("📦 Checking dependencies...")
    if not check_dependencies():
        print("❌ Failed to install required dependencies!")
        return 1
        
    print("✅ All dependencies satisfied!")
    print("🚀 Starting Model Explorer...")
    
    # Import and run the model explorer
    try:
        from model_explorer import main as run_basic
        run_basic()
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("\n💡 Troubleshooting:")
        print("- Ensure model files exist in ../out/ directory")
        print("- Check that data files exist in ../data/shakespeare_char/")
        print("- Try running: pip install PyQt6 torch numpy")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())