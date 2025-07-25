#!/usr/bin/env python3
"""
Test script to verify the integration of resume improvements and refactored structure.
"""
import os
import sys

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        # Test core imports
        from training.config import TrainingConfig, load_scaling_schedule
        print("✓ training.config imported successfully")
        
        from training.resume import (
            find_checkpoint_path, load_checkpoint_with_fallback, 
            apply_model_parameter_overrides, apply_smart_state_dict_loading
        )
        print("✓ training.resume imported successfully")
        
        from training.utils import TimingProfiler, BatchManager
        print("✓ training.utils imported successfully")
        
        from training.scheduler import TrainingScheduler, LearningRateScheduler
        print("✓ training.scheduler imported successfully")
        
        from training.operations import log_detailed_params, log_model_architecture
        print("✓ training.operations imported successfully")
        
        from training.evaluation import get_val_batch
        print("✓ training.evaluation imported successfully")
        
        # Test logger
        from logger import TrainingLogger
        print("✓ logger imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config():
    """Test TrainingConfig functionality."""
    print("\nTesting TrainingConfig...")
    
    try:
        from training.config import TrainingConfig
        
        # Create config
        config = TrainingConfig()
        print("✓ TrainingConfig created successfully")
        
        # Test methods
        model_args = config.get_model_args()
        print(f"✓ get_model_args() returned {len(model_args)} parameters")
        
        overrideable = config.get_overrideable_params()
        print(f"✓ get_overrideable_params() returned {len(overrideable)} parameters")
        
        training_overrideable = config.get_training_overrideable_params()
        print(f"✓ get_training_overrideable_params() returned {len(training_overrideable)} parameters")
        
        config_dict = config.to_dict()
        print(f"✓ to_dict() returned {len(config_dict)} settings")
        
        # Test update from dict
        config.update_from_dict({'batch_size': 16, 'learning_rate': 1e-4})
        assert config.batch_size == 16
        assert config.learning_rate == 1e-4
        print("✓ update_from_dict() works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ TrainingConfig test failed: {e}")
        return False

def test_logger():
    """Test TrainingLogger functionality."""
    print("\nTesting TrainingLogger...")
    
    try:
        from logger import TrainingLogger
        
        # Create logger
        logger = TrainingLogger(log_dir='test_logs', file_enabled=True)
        print("✓ TrainingLogger created successfully")
        
        # Test setup
        logger.setup({'test_param': 'test_value'})
        print("✓ Logger setup completed")
        
        # Test logging methods
        logger.log("Test message")
        logger.log_metrics( {'iter':100, 'loss': 0.5, 'lr': 1e-4})
        logger.log_operation_start(100, 'test_op', 'test_value', 'test_trigger', 0.5, 0.6, 1000)
        print("✓ All logging methods work")
        
        # Clean up
        logger.close()
        
        # Remove test log file
        if os.path.exists(logger.log_file_path):
            os.remove(logger.log_file_path)
        if os.path.exists('test_logs'):
            os.rmdir('test_logs')
        
        print("✓ Logger cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"✗ TrainingLogger test failed: {e}")
        return False

def test_schedulers():
    """Test scheduler functionality."""
    print("\nTesting schedulers...")
    
    try:
        from training.scheduler import LearningRateScheduler
        
        # Create LR scheduler
        lr_scheduler = LearningRateScheduler(
            learning_rate=1e-3,
            warmup_iters=100,
            lr_decay_iters=1000,
            min_lr=1e-5
        )
        print("✓ LearningRateScheduler created successfully")
        
        # Test LR calculation
        lr_0 = lr_scheduler.get_lr(0)
        lr_50 = lr_scheduler.get_lr(50)
        lr_100 = lr_scheduler.get_lr(100)
        lr_500 = lr_scheduler.get_lr(500)
        lr_1000 = lr_scheduler.get_lr(1000)
        
        print(f"✓ LR at iter 0: {lr_0:.6f}")
        print(f"✓ LR at iter 50: {lr_50:.6f}")
        print(f"✓ LR at iter 100: {lr_100:.6f}")
        print(f"✓ LR at iter 500: {lr_500:.6f}")
        print(f"✓ LR at iter 1000: {lr_1000:.6f}")
        
        # Test parameter updates
        lr_scheduler.update_params(learning_rate=2e-3)
        new_lr = lr_scheduler.get_lr(100)
        print(f"✓ Updated LR at iter 100: {new_lr:.6f}")
        
        # Test schedule reset
        lr_scheduler.reset_schedule(500)
        reset_lr = lr_scheduler.get_lr(500)
        print(f"✓ Reset LR at iter 500: {reset_lr:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Scheduler test failed: {e}")
        return False

def test_timing_profiler():
    """Test TimingProfiler functionality."""
    print("\nTesting TimingProfiler...")
    
    try:
        from training.utils import TimingProfiler
        import time
        
        # Create profiler
        profiler = TimingProfiler()
        print("✓ TimingProfiler created successfully")
        
        # Test timing sections
        with profiler.time_section("test_section"):
            time.sleep(0.01)  # 10ms
        
        with profiler.time_section("another_section"):
            time.sleep(0.005)  # 5ms
        
        # Test summary
        summary = profiler.get_summary()
        print("✓ Timing summary generated:")
        print(summary)
        
        return True
        
    except Exception as e:
        print(f"✗ TimingProfiler test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Integration of Resume Improvements and Refactored Structure")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config,
        test_logger,
        test_schedulers,
        test_timing_profiler
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration successful!")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
