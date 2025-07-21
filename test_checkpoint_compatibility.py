#!/usr/bin/env python3
"""
Test script for universal checkpoint compatibility.

This script tests the ability to:
1. Train a standard (non-LoRA) model and save a checkpoint
2. Resume training from that checkpoint with LoRA enabled
3. Verify that the loading process works without errors

Usage:
    python test_checkpoint_compatibility.py
"""

import os
import sys
import subprocess
import shutil
import time

def run_training(config_file, description):
    """Run training with the specified config file."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Config: {config_file}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, "train.py", f"config/{config_file}"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            return True
        else:
            print(f"❌ FAILED: {description}")
            print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
            print("STDERR:", result.stderr[-1000:])  # Last 1000 chars
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description} took too long")
        return False
    except Exception as e:
        print(f"💥 ERROR: {description} - {str(e)}")
        return False

def cleanup_test_directory():
    """Clean up the test output directory."""
    test_dir = "out-checkpoint-compatibility-test"
    if os.path.exists(test_dir):
        print(f"🧹 Cleaning up test directory: {test_dir}")
        shutil.rmtree(test_dir)

def check_checkpoint_exists():
    """Check if the checkpoint file was created."""
    checkpoint_path = "out-checkpoint-compatibility-test/ckpt.pt"
    if os.path.exists(checkpoint_path):
        print(f"✅ Checkpoint file exists: {checkpoint_path}")
        return True
    else:
        print(f"❌ Checkpoint file missing: {checkpoint_path}")
        return False

def cleanup_reverse_test_directory():
    """Clean up the reverse test output directory."""
    test_dir = "out-checkpoint-compatibility-reverse-test"
    if os.path.exists(test_dir):
        print(f"🧹 Cleaning up reverse test directory: {test_dir}")
        shutil.rmtree(test_dir)

def check_reverse_checkpoint_exists():
    """Check if the reverse test checkpoint file was created."""
    checkpoint_path = "out-checkpoint-compatibility-reverse-test/ckpt.pt"
    if os.path.exists(checkpoint_path):
        print(f"✅ Reverse checkpoint file exists: {checkpoint_path}")
        return True
    else:
        print(f"❌ Reverse checkpoint file missing: {checkpoint_path}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting Universal Checkpoint Compatibility Test")
    print("This test verifies that checkpoints can be loaded between LoRA and non-LoRA models")

    # Test 1: Standard -> LoRA
    print("\n" + "="*80)
    print("TEST 1: Loading Standard Checkpoint into LoRA Model")
    print("="*80)

    # Clean up any previous test runs
    cleanup_test_directory()

    # Test 1a: Train a standard model and save checkpoint
    success1a = run_training(
        "test_checkpoint_compatibility.py",
        "Training standard (non-LoRA) model"
    )

    if not success1a:
        print("❌ First test failed, aborting")
        return False

    # Check if checkpoint was created
    if not check_checkpoint_exists():
        print("❌ Checkpoint not created, aborting")
        return False

    # Small delay to ensure file system consistency
    time.sleep(1)

    # Test 1b: Resume training with LoRA enabled
    success1b = run_training(
        "test_checkpoint_compatibility_lora.py",
        "Resuming with LoRA enabled"
    )

    if not success1b:
        print("\n❌ TEST 1 FAILED!")
        print("The LoRA model could not load from the standard checkpoint")
        return False

    print("\n✅ TEST 1 PASSED: Standard -> LoRA compatibility verified!")
    cleanup_test_directory()

    # Test 2: LoRA -> Standard
    print("\n" + "="*80)
    print("TEST 2: Loading LoRA Checkpoint into Standard Model")
    print("="*80)

    # Clean up any previous reverse test runs
    cleanup_reverse_test_directory()

    # Test 2a: Train a LoRA model and save checkpoint
    success2a = run_training(
        "test_checkpoint_compatibility_reverse.py",
        "Training LoRA model"
    )

    if not success2a:
        print("❌ LoRA training test failed, aborting")
        return False

    # Check if checkpoint was created
    if not check_reverse_checkpoint_exists():
        print("❌ LoRA checkpoint not created, aborting")
        return False

    # Small delay to ensure file system consistency
    time.sleep(1)

    # Test 2b: Resume training with standard model
    success2b = run_training(
        "test_checkpoint_compatibility_reverse_standard.py",
        "Resuming with standard model"
    )

    if success2b:
        print("\n✅ TEST 2 PASSED: LoRA -> Standard compatibility verified!")
        cleanup_reverse_test_directory()

        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Standard -> LoRA checkpoint loading works")
        print("✅ LoRA -> Standard checkpoint loading works")
        print("✅ Universal checkpoint compatibility fully verified!")
        return True
    else:
        print("\n❌ TEST 2 FAILED!")
        print("The standard model could not load from the LoRA checkpoint")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
