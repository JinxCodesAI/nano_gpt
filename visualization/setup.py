#!/usr/bin/env python3
"""
Setup script for Model Explorer
Verifies file structure and dependencies
"""

import sys
import os
from pathlib import Path

def check_file_structure():
    """Check if required files exist"""
    print("🔍 Checking file structure...")
    
    required_files = [
        "../data/shakespeare_char/train.bin",
        "../data/shakespeare_char/meta.pkl",
        "../train_utils.py", 
        "../model.py",
        "../utils.py"
    ]
    
    optional_model_files = [
        "../out/14.6_unmasking_no_noise.pt",
        "../out/1.23_remasking_bin.pt", 
        "../out/1.35_remask.pt"
    ]
    
    missing_required = []
    missing_models = []
    
    # Check required files
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_required.append(file_path)
        else:
            print(f"✅ {file_path}")
            
    # Check model files
    for file_path in optional_model_files:
        if not Path(file_path).exists():
            missing_models.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    # Report results
    if missing_required:
        print(f"\n❌ Missing required files:")
        for file_path in missing_required:
            print(f"   - {file_path}")
        print("\n💡 These files are required for the application to work.")
        print("   Make sure you're running from the correct directory.")
        return False
        
    if missing_models:
        print(f"\n⚠️  Missing model files:")
        for file_path in missing_models:
            print(f"   - {file_path}")
        print("\n💡 Model files are optional but needed for predictions.")
        print("   Train models first or adjust paths in the application.")
        
    print(f"\n✅ File structure check complete!")
    return True

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        print("   Python 3.8+ is required")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    dependencies = [
        "PyQt6",
        "numpy", 
        "torch"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep.lower().replace('-', '_'))
            print(f"✅ {dep} already installed")
        except ImportError:
            print(f"📦 Installing {dep}...")
            import subprocess
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                print(f"✅ {dep} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {dep}: {e}")
                return False
                
    return True

def main():
    """Main setup function"""
    print("🚀 Model Explorer Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return 1
        
    # Check file structure
    if not check_file_structure():
        return 1
        
    # Install dependencies
    if not install_dependencies():
        return 1
        
    print("\n🎉 Setup complete!")
    print("\n🚀 You can now run the application:")
    print("   python launch.py")
    print("   python model_explorer.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())