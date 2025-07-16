"""
Training Logger for NanoGPT

This module provides a TrainingLogger class that handles file-based logging
for training runs, including configuration dumping and progress tracking.
"""

import os
from datetime import datetime


class TrainingLogger:
    """
    A logger class for training runs that creates timestamped log files
    and handles configuration dumping and progress logging.
    """
    
    def __init__(self, log_dir='logs', enabled=True):
        """
        Initialize the training logger.
        
        Args:
            log_dir (str): Directory where log files will be stored
            enabled (bool): Whether logging is enabled
        """
        self.log_dir = log_dir
        self.enabled = enabled
        self.log_file = None
        self.log_path = None
        
    def setup(self, config=None):
        """
        Set up the logging system by creating the log directory and file.
        
        Args:
            config (dict): Configuration dictionary to log at startup
        """
        if not self.enabled:
            return
            
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"log_run_{timestamp}.txt"
        self.log_path = os.path.join(self.log_dir, log_filename)
        
        # Open log file with line buffering for immediate flush
        self.log_file = open(self.log_path, 'w', buffering=1)
        
        # Log startup header
        self._write_header()
        
        # Log configuration if provided
        if config:
            self._log_config(config)
            
    def _write_header(self):
        """Write the header section to the log file."""
        if self.log_file is None:
            return
            
        self.log_file.write("=" * 80 + "\n")
        self.log_file.write(f"Training run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("=" * 80 + "\n")
        self.log_file.flush()
        
    def _log_config(self, config):
        """
        Log the configuration dictionary to the file.
        
        Args:
            config (dict): Configuration dictionary to log
        """
        if self.log_file is None:
            return
            
        self.log_file.write("Configuration:\n")
        self.log_file.write("-" * 40 + "\n")
        
        for key, value in sorted(config.items()):
            self.log_file.write(f"{key}: {value}\n")
            
        self.log_file.write("-" * 40 + "\n")
        self.log_file.flush()
        
    def log(self, message):
        """
        Log a message with timestamp.
        
        Args:
            message (str): Message to log
        """
        if self.log_file is None:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"[{timestamp}] {message}\n")
        self.log_file.flush()
        
    def log_step(self, iter_num, train_loss, val_loss):
        """
        Log a training step with losses.
        
        Args:
            iter_num (int): Current iteration number
            train_loss (float): Training loss
            val_loss (float): Validation loss
        """
        message = f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}"
        self.log(message)
        
    def close(self):
        """Close the log file and write footer."""
        if self.log_file is None:
            return
            
        # Write footer
        self.log_file.write("=" * 80 + "\n")
        self.log_file.write(f"Training run ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("=" * 80 + "\n")
        
        # Close file
        self.log_file.close()
        self.log_file = None
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures log file is closed."""
        self.close()
        
    @property
    def is_enabled(self):
        """Check if logging is enabled."""
        return self.enabled
        
    @property
    def log_file_path(self):
        """Get the path to the current log file."""
        return self.log_path
