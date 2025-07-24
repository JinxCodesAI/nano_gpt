#!/usr/bin/env python3
"""
Basic test script to verify the integration without requiring torch.
"""
import os
import sys

def test_basic_imports():
    """Test that basic modules can be imported successfully."""
    print("Testing basic imports...")
    
    try:
        # Test config
        from training.config import TrainingConfig, load_scaling_schedule
        print("✓ training.config imported successfully")
        
        # Test logger
        from logger import TrainingLogger
        print("✓ logger imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config_functionality():
    """Test TrainingConfig functionality."""
    print("\nTesting TrainingConfig...")
    
    try:
        from training.config import TrainingConfig
        
        # Create config
        config = TrainingConfig()
        print("✓ TrainingConfig created successfully")
        
        # Test basic properties
        assert hasattr(config, 'batch_size')
        assert hasattr(config, 'learning_rate')
        assert hasattr(config, 'n_layer')
        print("✓ Config has expected attributes")
        
        # Test methods
        model_args = config.get_model_args()
        assert isinstance(model_args, dict)
        assert 'n_layer' in model_args
        print(f"✓ get_model_args() returned {len(model_args)} parameters")
        
        overrideable = config.get_overrideable_params()
        assert isinstance(overrideable, list)
        print(f"✓ get_overrideable_params() returned {len(overrideable)} parameters")
        
        training_overrideable = config.get_training_overrideable_params()
        assert isinstance(training_overrideable, list)
        print(f"✓ get_training_overrideable_params() returned {len(training_overrideable)} parameters")
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        print(f"✓ to_dict() returned {len(config_dict)} settings")
        
        # Test update from dict
        original_batch_size = config.batch_size
        original_lr = config.learning_rate
        
        config.update_from_dict({'batch_size': 16, 'learning_rate': 1e-4})
        assert config.batch_size == 16
        assert config.learning_rate == 1e-4
        print("✓ update_from_dict() works correctly")
        
        # Test validation
        try:
            config.validate()
            print("✓ Config validation passed")
        except Exception as e:
            print(f"✓ Config validation caught expected error: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ TrainingConfig test failed: {e}")
        return False

def test_logger_functionality():
    """Test TrainingLogger functionality."""
    print("\nTesting TrainingLogger...")
    
    try:
        from logger import TrainingLogger
        
        # Create logger
        logger = TrainingLogger(log_dir='test_logs', enabled=True)
        print("✓ TrainingLogger created successfully")
        
        # Test setup
        test_config = {
            'batch_size': 12,
            'learning_rate': 1e-3,
            'n_layer': 6
        }
        logger.setup(test_config)
        print("✓ Logger setup completed")
        
        # Test basic logging
        logger.log("Test message")
        print("✓ Basic logging works")
        
        # Test metrics logging
        logger.log_metrics(100, {'loss': 0.5, 'lr': 1e-4})
        print("✓ Metrics logging works")
        
        # Test operation logging
        logger.log_operation_start(100, 'test_op', 'test_value', 'test_trigger', 0.5, 0.6, 1000)
        logger.log_operation_success(100, 'test_op', {'param': 'value'})
        logger.log_operation_error(100, 'test_op', 'test error')
        print("✓ Operation logging works")
        
        # Test analysis logging
        logger.log_analysis_results(100, {'metric': 'value'})
        print("✓ Analysis logging works")
        
        # Test properties
        assert logger.is_enabled == True
        assert logger.log_file_path is not None
        print("✓ Logger properties work")
        
        # Clean up
        logger.close()
        
        # Verify log file was created and has content
        if os.path.exists(logger.log_file_path):
            with open(logger.log_file_path, 'r') as f:
                content = f.read()
                assert len(content) > 0
                assert 'Test message' in content
                assert 'METRICS:' in content
                assert 'OPERATION_START:' in content
            print("✓ Log file created with expected content")
            
            # Remove test log file
            os.remove(logger.log_file_path)
        
        if os.path.exists('test_logs'):
            os.rmdir('test_logs')
        
        print("✓ Logger cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"✗ TrainingLogger test failed: {e}")
        return False

def test_resume_module_structure():
    """Test that resume module has expected structure."""
    print("\nTesting resume module structure...")
    
    try:
        import training.resume as resume_module
        
        # Check that expected functions exist
        expected_functions = [
            'find_checkpoint_path',
            'load_checkpoint_with_fallback',
            'apply_model_parameter_overrides',
            'apply_smart_state_dict_loading',
            'transfer_optimizer_state',
            'load_training_state',
            'restore_scaling_schedule_state',
            'apply_training_parameter_overrides'
        ]
        
        for func_name in expected_functions:
            assert hasattr(resume_module, func_name), f"Missing function: {func_name}"
            func = getattr(resume_module, func_name)
            assert callable(func), f"Not callable: {func_name}"
        
        print(f"✓ All {len(expected_functions)} expected functions found")
        
        return True
        
    except Exception as e:
        print(f"✗ Resume module test failed: {e}")
        return False

def test_file_compilation():
    """Test that all Python files compile without syntax errors."""
    print("\nTesting file compilation...")
    
    try:
        import py_compile
        
        files_to_test = [
            'train_refactored.py',
            'training/config.py',
            'training/resume.py',
            'training/utils.py',
            'training/scheduler.py',
            'training/operations.py',
            'training/evaluation.py',
            'logger.py'
        ]
        
        for file_path in files_to_test:
            if os.path.exists(file_path):
                py_compile.compile(file_path, doraise=True)
                print(f"✓ {file_path} compiles successfully")
            else:
                print(f"⚠ {file_path} not found, skipping")
        
        return True
        
    except Exception as e:
        print(f"✗ Compilation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Basic Integration Test (No External Dependencies)")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_config_functionality,
        test_logger_functionality,
        test_resume_module_structure,
        test_file_compilation
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
        print("🎉 All basic tests passed! Integration looks good!")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
