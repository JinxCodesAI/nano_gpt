#!/usr/bin/env python3
"""
Unit tests for the Training Orchestrator functionality.
Tests the basic operations and scaling schedule logic.
"""

import unittest
import tempfile
import os
import json
import yaml
from unittest.mock import patch, MagicMock

# Import the functions we want to test
# We'll need to mock some globals since train.py has module-level execution
import sys
sys.path.append('.')

class TestTrainingOrchestrator(unittest.TestCase):
    """Test cases for Training Orchestrator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock global variables that would be set in train.py
        self.mock_globals = {
            'lr_multiplier': 1.0,
            'batch_size_multiplier': 1.0,
            'grad_accum_multiplier': 1.0,
            'lr_schedule_offset': 0,
            'gradient_accumulation_steps': 5,
            'batch_size': 12,
            'iter_num': 1000
        }
    
    def test_load_scaling_schedule_yaml(self):
        """Test loading scaling schedule from YAML file"""
        # Create a temporary YAML file
        schedule_data = [
            {
                'name': 'change_lr',
                'value': 2.0,
                'trigger_loss': 6.0,
                'max_wait_iters': 50000,
                'reevaluate': False
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(schedule_data, f)
            temp_file = f.name
        
        try:
            # Import and test the function
            from train import load_scaling_schedule
            result = load_scaling_schedule(temp_file)
            
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['name'], 'change_lr')
            self.assertEqual(result[0]['value'], 2.0)
            self.assertEqual(result[0]['trigger_loss'], 6.0)
            self.assertEqual(result[0]['max_wait_iters'], 50000)
            self.assertEqual(result[0]['reevaluate'], False)
        finally:
            os.unlink(temp_file)
    
    def test_load_scaling_schedule_json(self):
        """Test loading scaling schedule from JSON file"""
        schedule_data = [
            {
                'name': 'change_batch_size',
                'value': 1.5,
                'trigger_loss': 5.5,
                'max_wait_iters': 75000,
                'reevaluate': False
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schedule_data, f)
            temp_file = f.name
        
        try:
            from train import load_scaling_schedule
            result = load_scaling_schedule(temp_file)
            
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['name'], 'change_batch_size')
            self.assertEqual(result[0]['value'], 1.5)
        finally:
            os.unlink(temp_file)
    
    def test_load_scaling_schedule_invalid_file(self):
        """Test loading scaling schedule from non-existent file"""
        from train import load_scaling_schedule
        result = load_scaling_schedule('non_existent_file.yaml')
        self.assertEqual(result, [])
    
    def test_load_scaling_schedule_invalid_format(self):
        """Test loading scaling schedule with invalid format"""
        # Create a file with invalid format (not a list)
        invalid_data = {'not': 'a list'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_data, f)
            temp_file = f.name
        
        try:
            from train import load_scaling_schedule
            result = load_scaling_schedule(temp_file)
            self.assertEqual(result, [])
        finally:
            os.unlink(temp_file)

class TestExecuteOperation(unittest.TestCase):
    """Test cases for execute_operation function"""
    
    def setUp(self):
        """Set up test environment with mocked globals"""
        # We need to patch the global variables in train module
        self.patcher_globals = patch.multiple(
            'train',
            lr_multiplier=1.0,
            batch_size_multiplier=1.0,
            grad_accum_multiplier=1.0,
            lr_schedule_offset=0,
            gradient_accumulation_steps=5,
            batch_size=12,
            iter_num=1000
        )
        self.mock_globals = self.patcher_globals.start()
    
    def tearDown(self):
        """Clean up patches"""
        self.patcher_globals.stop()
    
    def test_change_lr_operation(self):
        """Test change_lr operation"""
        from train import execute_operation
        
        op = {
            'name': 'change_lr',
            'value': 2.0,
            'trigger_loss': 6.0,
            'max_wait_iters': 50000,
            'reevaluate': False
        }
        
        # Mock the global lr_multiplier
        with patch('train.lr_multiplier', 1.0) as mock_lr:
            result = execute_operation(op)
            self.assertTrue(result)
    
    def test_change_batch_size_operation(self):
        """Test change_batch_size operation"""
        from train import execute_operation
        
        op = {
            'name': 'change_batch_size',
            'value': 2.0,
            'trigger_loss': 5.5,
            'max_wait_iters': 75000,
            'reevaluate': False
        }
        
        with patch('train.batch_size', 12) as mock_batch_size:
            with patch('train.batch_size_multiplier', 1.0) as mock_multiplier:
                result = execute_operation(op)
                self.assertTrue(result)
    
    def test_invalid_operation(self):
        """Test handling of invalid operation"""
        from train import execute_operation
        
        op = {
            'name': 'invalid_operation',
            'value': 1.0,
            'trigger_loss': 5.0,
            'max_wait_iters': 1000,
            'reevaluate': False
        }
        
        result = execute_operation(op)
        self.assertFalse(result)
    
    def test_negative_value_validation(self):
        """Test validation of negative values"""
        from train import execute_operation
        
        op = {
            'name': 'change_lr',
            'value': -1.0,  # Invalid negative value
            'trigger_loss': 6.0,
            'max_wait_iters': 50000,
            'reevaluate': False
        }
        
        result = execute_operation(op)
        self.assertFalse(result)

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
