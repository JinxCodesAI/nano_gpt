"""
Base class for loss modifiers in the modular loss modification system.

All loss modifiers should inherit from BaseLossModifier and implement the required methods.
This ensures a consistent interface and allows for easy composition and extension.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch


class BaseLossModifier(ABC):
    """
    Abstract base class for all loss modifiers.
    
    Loss modifiers can adjust the loss function during training to implement
    various training strategies like label smoothing, entropy-based weighting,
    or other loss modifications.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the loss modifier with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters for this modifier
        """
        self.config = config
        self.enabled = config.get('enabled', False)
        self._metrics = {}
    
    @abstractmethod
    def modify_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        loss: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        """
        Apply the loss modification to the input loss.
        
        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size)
            targets: Target token IDs (batch_size, seq_len)
            loss: Original loss value (scalar tensor)
            **kwargs: Additional arguments that may be needed by specific modifiers
            
        Returns:
            Modified loss tensor (scalar)
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics collected by this modifier during the last forward pass.
        
        Returns:
            Dictionary of metric names and values for logging/monitoring
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if this modifier is enabled."""
        return self.enabled
    
    def reset_metrics(self) -> None:
        """Reset internal metrics for the next iteration."""
        self._metrics = {}