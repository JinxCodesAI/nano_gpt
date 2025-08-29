#!/usr/bin/env python3
"""
Development setup script for visualizer
Checks environment and provides helpful info
"""

import sys
import os
from pathlib import Path

def check_environment():
    """Check development environment"""
    print("🔧 Diffusion Visualizer Development Setup")
    print("=" * 50)
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python version: {python_version}")
    
    # Check required directories
    parent_dir = Path(__file__).parent.parent
    
    print("\n📁 Directory Structure:")
    directories_to_check = [
        parent_dir / "out",
        parent_dir / "data" / "shakespeare_char",
        parent_dir / "visualization"
    ]
    
    for dir_path in directories_to_check:
        if dir_path.exists():
            print(f"  ✓ {dir_path.relative_to(parent_dir)}")
        else:
            print(f"  ✗ {dir_path.relative_to(parent_dir)} (missing)")
    
    # Check dependencies
    print("\n📦 Dependencies:")
    deps_to_check = [
        ("PyQt6", "PyQt6"),
        ("torch", "torch"), 
        ("numpy", "numpy as np")
    ]
    
    for dep_name, import_name in deps_to_check:
        try:
            exec(f"import {import_name}")
            print(f"  ✓ {dep_name}")
        except ImportError:
            print(f"  ✗ {dep_name} (missing)")
    
    # Check key files
    print("\n📄 Key Files:")
    files_to_check = [
        parent_dir / "model.py",
        parent_dir / "sample.py", 
        parent_dir / "out" / "35.75_58.2_UM.pt",
        parent_dir / "data" / "shakespeare_char" / "meta.pkl"
    ]
    
    for file_path in files_to_check:
        if file_path.exists():
            print(f"  ✓ {file_path.relative_to(parent_dir)}")
        else:
            print(f"  ✗ {file_path.relative_to(parent_dir)} (missing)")
    
    print("\n🚀 Launch Options:")
    print("  python launch.py          # Full launcher with dependency checks")
    print("  python run_direct.py      # Direct launch for development")
    print("  pip install -r requirements.txt  # Install dependencies")

if __name__ == "__main__":
    check_environment()