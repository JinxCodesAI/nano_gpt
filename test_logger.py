"""
Test script for the TrainingLogger class.
"""

import os
import tempfile
import shutil
from logger import TrainingLogger


def test_logger_basic_functionality():
    """Test basic logger functionality."""
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp()
    
    try:
        # Test configuration
        config = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'max_iters': 1000,
            'dataset': 'test_data'
        }
        
        # Initialize logger
        logger = TrainingLogger(log_dir=test_dir, enabled=True)
        
        # Setup logging
        logger.setup(config)
        
        # Test logging messages
        logger.log("Test message 1")
        logger.log_step(100, 2.5432, 2.4321)
        logger.log_step(200, 2.1234, 2.0987)
        logger.log("Test message 2")
        
        # Close logger
        logger.close()
        
        # Verify log file was created
        log_files = [f for f in os.listdir(test_dir) if f.startswith('log_run_')]
        assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"
        
        # Read and verify log content
        log_path = os.path.join(test_dir, log_files[0])
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Check for expected content
        assert "Training run started at" in content
        assert "Configuration:" in content
        assert "batch_size: 32" in content
        assert "learning_rate: 0.001" in content
        assert "step 100: train loss 2.5432, val loss 2.4321" in content
        assert "step 200: train loss 2.1234, val loss 2.0987" in content
        assert "Test message 1" in content
        assert "Test message 2" in content
        assert "Training run ended at" in content
        
        print("✓ Basic functionality test passed")
        
    finally:
        # Clean up
        shutil.rmtree(test_dir)


def test_logger_disabled():
    """Test logger when disabled."""
    test_dir = tempfile.mkdtemp()
    
    try:
        # Initialize disabled logger
        logger = TrainingLogger(log_dir=test_dir, enabled=False)
        
        # Setup logging (should do nothing)
        logger.setup({'test': 'config'})
        
        # Try logging (should do nothing)
        logger.log("This should not be logged")
        logger.log_step(100, 1.0, 1.0)
        
        # Close logger
        logger.close()
        
        # Verify no log files were created
        log_files = [f for f in os.listdir(test_dir) if f.startswith('log_run_')]
        assert len(log_files) == 0, f"Expected 0 log files, found {len(log_files)}"
        
        print("✓ Disabled logger test passed")
        
    finally:
        # Clean up
        shutil.rmtree(test_dir)


def test_logger_context_manager():
    """Test logger as context manager."""
    test_dir = tempfile.mkdtemp()
    
    try:
        config = {'test_param': 'test_value'}
        
        # Use logger as context manager
        with TrainingLogger(log_dir=test_dir, enabled=True) as logger:
            logger.setup(config)
            logger.log("Context manager test")
            logger.log_step(50, 3.0, 2.9)
        
        # Verify log file was created and closed properly
        log_files = [f for f in os.listdir(test_dir) if f.startswith('log_run_')]
        assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"
        
        # Read and verify content
        log_path = os.path.join(test_dir, log_files[0])
        with open(log_path, 'r') as f:
            content = f.read()
        
        assert "test_param: test_value" in content
        assert "Context manager test" in content
        assert "step 50: train loss 3.0000, val loss 2.9000" in content
        assert "Training run ended at" in content
        
        print("✓ Context manager test passed")
        
    finally:
        # Clean up
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    print("Running TrainingLogger tests...")
    
    test_logger_basic_functionality()
    test_logger_disabled()
    test_logger_context_manager()
    
    print("\n✅ All tests passed!")
