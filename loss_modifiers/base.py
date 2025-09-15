"""
Base class for loss modifiers in the modular loss modification system.

All loss modifiers should inherit from BaseLossModifier and implement the required methods.
This ensures a consistent interface and allows for easy composition and extension.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch

# Import ModelMode - we'll need to handle the import path
try:
    from model import ModelMode
except ImportError:
    # Handle case where model.py isn't in path during isolated testing
    from enum import Enum
    class ModelMode(Enum):
        LANGUAGE_MODEL = "language_model"
        TOKEN_CLASSIFIER = "token_classifier" 
        SEQUENCE_SCORER = "sequence_scorer"


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
        model_mode: ModelMode = None,
        **kwargs
    ) -> torch.Tensor | dict:
        """
        Apply the loss modification to the input loss.

        Args:
            logits: Model output logits
            targets: Target values 
            loss: Original loss value (scalar tensor)
            model_mode: Current model mode (for mode-specific behavior)
            **kwargs: Additional arguments that may be needed by specific modifiers.
                Common keys supported by the pipeline:
                - per_position_loss: Optional (batch_size, seq_len) tensor with per-position losses
                - mask: Optional boolean mask (batch_size, seq_len) of valid positions
                - ignore_index: Optional int for ignored target positions

        Returns:
            Either:
            - A scalar tensor 'loss' (modified loss), or
            - A dict that may contain:
                {'loss': <scalar tensor>, 'per_position_loss': <(B,T) tensor>}.
              Returning 'per_position_loss' indicates this modifier replaces the
              per-position loss; the pipeline will aggregate it to a scalar at the end.
        """
        pass

    @abstractmethod  
    def supports_mode(self, mode: ModelMode) -> bool:
        """
        Check if this modifier supports the given model mode.
        
        Args:
            mode: Model mode to check
            
        Returns:
            True if modifier is compatible with this mode
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