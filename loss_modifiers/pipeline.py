"""
Pipeline for orchestrating multiple loss modifiers in sequence.

The LossModifierPipeline manages multiple loss modifiers and applies them
in sequence to the loss function during training. It provides a clean interface
for the training loop and handles metric collection from all modifiers.
"""

from typing import List, Dict, Any, Optional
import torch
from .base import BaseLossModifier


class LossModifierPipeline:
    """
    Pipeline for applying multiple loss modifiers in sequence.
    
    This class coordinates multiple loss modifiers, applying them in the order
    they were added to the pipeline. It also collects metrics from all modifiers
    for monitoring and logging purposes.
    """
    
    def __init__(self, modifiers: Optional[List[BaseLossModifier]] = None):
        """
        Initialize the pipeline with optional list of modifiers.
        
        Args:
            modifiers: List of loss modifiers to include in the pipeline
        """
        self.modifiers = modifiers or []
        self._enabled_modifiers = []
        self._update_enabled_modifiers()
    
    def add_modifier(self, modifier: BaseLossModifier) -> None:
        """
        Add a new modifier to the pipeline.
        
        Args:
            modifier: Loss modifier instance to add
        """
        self.modifiers.append(modifier)
        self._update_enabled_modifiers()
    
    def _update_enabled_modifiers(self) -> None:
        """Update the list of enabled modifiers for efficient processing."""
        self._enabled_modifiers = [mod for mod in self.modifiers if mod.is_enabled()]
    
    def modify_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        loss: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        """
        Apply all enabled modifiers to the loss in sequence.
        
        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size)
            targets: Target token IDs (batch_size, seq_len)
            loss: Original loss value (scalar tensor)
            **kwargs: Additional arguments passed to modifiers
            
        Returns:
            Modified loss tensor after applying all enabled modifiers
        """
        # If no enabled modifiers, return original loss (zero overhead)
        if not self._enabled_modifiers:
            return loss
        
        modified_loss = loss
        for modifier in self._enabled_modifiers:
            modified_loss = modifier.modify_loss(logits, targets, modified_loss, **kwargs)
        
        return modified_loss
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Collect metrics from all enabled modifiers.
        
        Returns:
            Dictionary with metrics from all modifiers, prefixed by modifier name
        """
        all_metrics = {}
        for modifier in self._enabled_modifiers:
            modifier_name = modifier.__class__.__name__
            modifier_metrics = modifier.get_metrics()
            for key, value in modifier_metrics.items():
                all_metrics[f"{modifier_name}.{key}"] = value
        return all_metrics
    
    def reset_all_metrics(self) -> None:
        """Reset metrics for all modifiers."""
        for modifier in self._enabled_modifiers:
            modifier.reset_metrics()
    
    def is_empty(self) -> bool:
        """Check if the pipeline has any enabled modifiers."""
        return len(self._enabled_modifiers) == 0
    
    def get_enabled_modifier_names(self) -> List[str]:
        """Get names of all enabled modifiers for logging."""
        return [mod.__class__.__name__ for mod in self._enabled_modifiers]