"""
Logging abstractions for training pipeline.

This module provides modular logging classes extracted from train.py to improve
testability and allow flexible logging configurations (console, wandb, etc.).
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class Logger(ABC):
    """Abstract base class for all loggers."""
    
    @abstractmethod
    def log_step(self, metrics: Dict[str, Any]) -> None:
        """
        Log training step metrics (called every log_interval).
        
        Args:
            metrics: Dictionary containing step metrics like loss, lr, mfu, iter, etc.
        """
        pass
    
    @abstractmethod
    def log_eval(self, metrics: Dict[str, Any]) -> None:
        """
        Log evaluation metrics (called every eval_interval).
        
        Args:
            metrics: Dictionary containing eval metrics like train/val losses, lr, mfu, iter, etc.
        """
        pass
    
    @abstractmethod
    def log_info(self, message: str) -> None:
        """
        Log general information messages.
        
        Args:
            message: Info message to log
        """
        pass
    
    @abstractmethod  
    def log_checkpoint(self, message: str) -> None:
        """
        Log checkpoint-related messages.
        
        Args:
            message: Checkpoint message to log
        """
        pass


class ConsoleLogger(Logger):
    """
    Console logger that handles all print statements from train.py.
    
    Replicates the exact same console output format as the original code.
    """
    
    def __init__(self, master_process: bool = True):
        """
        Initialize console logger.
        
        Args:
            master_process: Whether this is the master process (for DDP)
        """
        self.master_process = master_process
    
    def log_step(self, metrics: Dict[str, Any]) -> None:
        """Log training step to console (every log_interval)."""
        if not self.master_process:
            return
            
        iter_num = metrics.get('iter', 0)
        loss = metrics.get('loss', 0.0)
        dt_ms = metrics.get('time_ms', 0.0)
        mfu_pct = metrics.get('mfu_pct', 0.0)
        seqvar = metrics.get('seqvar_loss_ratio', None)
        seqcorr = metrics.get('seqcorr_loss_ratio', None)

        extras = []
        if seqvar is not None:
            extras.append(f"seqvar {seqvar:.4f}")
        if seqcorr is not None:
            extras.append(f"seqcorr {seqcorr:.4f}")
        extras_str = (", " + ", ".join(extras)) if extras else ""
        print(f"iter {iter_num}: loss {loss:.4f}, time {dt_ms:.2f}ms, mfu {mfu_pct:.2f}%{extras_str}")
    
    def log_eval(self, metrics: Dict[str, Any]) -> None:
        """Log evaluation results to console (every eval_interval)."""
        if not self.master_process:
            return
            
        iter_num = metrics.get('iter', 0)
        train_loss = metrics.get('train/loss', 0.0)
        val_loss = metrics.get('val/loss', 0.0)
        seqvar_avg = metrics.get('loss_modifiers/SequenceScorerVarianceModifier.loss_ratio_avg', None)
        seqcorr_avg = metrics.get('loss_modifiers/SequenceScorerCorrelationModifier.loss_ratio_avg', None)

        parts = [f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}"]
        if seqvar_avg is not None:
            parts.append(f"seqvar loss_ratio_avg {seqvar_avg:.4f}")
        if seqcorr_avg is not None:
            parts.append(f"seqcorr loss_ratio_avg {seqcorr_avg:.4f}")
        print(", ".join(parts))

    def log_info(self, message: str) -> None:
        """Log general info message to console."""
        if self.master_process:
            print(message)
    
    def log_checkpoint(self, message: str) -> None:
        """Log checkpoint message to console."""
        if self.master_process:
            print(message)


class WandBLogger(Logger):
    """
    Weights & Biases logger that handles wandb.log calls.
    
    Replicates exact same logging behavior as original code with proper
    interval-based logging and loss modifier metrics integration.
    """
    
    def __init__(self, project: str, run_name: str, config: Dict[str, Any], 
                 master_process: bool = True, enabled: bool = True):
        """
        Initialize WandB logger.
        
        Args:
            project: WandB project name
            run_name: WandB run name  
            config: Configuration dict to log to WandB
            master_process: Whether this is the master process (for DDP)
            enabled: Whether WandB logging is enabled
        """
        self.master_process = master_process
        self.enabled = enabled
        self.wandb = None
        
        if self.enabled and self.master_process:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(project=project, name=run_name, config=config)
            except ImportError:
                print("Warning: wandb not available, disabling wandb logging")
                self.enabled = False
    
    def log_step(self, metrics: Dict[str, Any]) -> None:
        """
        Log training step metrics to WandB (every log_interval).
        
        Note: Only logs basic step metrics, not evaluation data.
        """
        if not (self.enabled and self.master_process and self.wandb):
            return
        
        # For step logging, we typically don't log to wandb every step
        # Only evaluation metrics go to wandb in the original code
        pass
    
    def log_eval(self, metrics: Dict[str, Any]) -> None:
        """
        Log evaluation metrics to WandB (every eval_interval).
        
        This replicates the exact wandb.log call from the original code.
        """
        if not (self.enabled and self.master_process and self.wandb):
            return
        
        # Create log dict with core metrics
        log_dict = {
            "iter": metrics.get('iter', 0),
            "train/loss": metrics.get('train/loss', 0.0),
            "val/loss": metrics.get('val/loss', 0.0), 
            "lr": metrics.get('lr', 0.0),
            "mfu": metrics.get('mfu_pct', 0.0),  # Already as percentage
        }
        
        # Add flattened loss modifier metrics if present
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key.startswith("loss_modifiers/"):
                log_dict[key] = value

        self.wandb.log(log_dict)

    def log_info(self, message: str) -> None:
        """WandB doesn't log info messages directly."""
        pass
    
    def log_checkpoint(self, message: str) -> None:
        """WandB doesn't log checkpoint messages directly."""
        pass


class CompositeLogger(Logger):
    """
    Composite logger that combines multiple loggers.
    
    Allows logging to console + wandb simultaneously while maintaining
    the exact same behavior as the original train.py.
    """
    
    def __init__(self, loggers: List[Logger]):
        """
        Initialize composite logger.
        
        Args:
            loggers: List of logger instances to combine
        """
        self.loggers = loggers
    
    def log_step(self, metrics: Dict[str, Any]) -> None:
        """Forward step logging to all loggers."""
        for logger in self.loggers:
            logger.log_step(metrics)
    
    def log_eval(self, metrics: Dict[str, Any]) -> None:
        """Forward eval logging to all loggers.""" 
        for logger in self.loggers:
            logger.log_eval(metrics)
    
    def log_info(self, message: str) -> None:
        """Forward info logging to all loggers."""
        for logger in self.loggers:
            logger.log_info(message)
    
    def log_checkpoint(self, message: str) -> None:
        """Forward checkpoint logging to all loggers."""
        for logger in self.loggers:
            logger.log_checkpoint(message)


def create_logger(wandb_log: bool, wandb_project: str, wandb_run_name: str, 
                  config: Dict[str, Any], master_process: bool = True) -> Logger:
    """
    Factory function to create appropriate logger based on configuration.
    
    Args:
        wandb_log: Whether to enable WandB logging
        wandb_project: WandB project name
        wandb_run_name: WandB run name
        config: Configuration dict
        master_process: Whether this is the master process
        
    Returns:
        Logger instance (ConsoleLogger, WandBLogger, or CompositeLogger)
    """
    loggers = []
    
    # Always add console logger
    loggers.append(ConsoleLogger(master_process=master_process))
    
    # Add WandB logger if enabled
    if wandb_log:
        loggers.append(WandBLogger(
            project=wandb_project,
            run_name=wandb_run_name, 
            config=config,
            master_process=master_process,
            enabled=True
        ))
    
    if len(loggers) == 1:
        return loggers[0]
    else:
        return CompositeLogger(loggers)