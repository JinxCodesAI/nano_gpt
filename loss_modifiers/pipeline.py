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
            **kwargs: Additional arguments passed to modifiers. Common keys:
                - per_position_loss: Optional (B,T) tensor with per-position losses
                - mask: Optional boolean mask (B,T) for valid positions

        Returns:
            Modified loss tensor after applying all enabled modifiers
        """
        # If no enabled modifiers, return original loss (zero overhead)
        if not self._enabled_modifiers:
            return loss

        # Track optional per-position loss tensor through the pipeline
        per_position_loss = kwargs.get('per_position_loss', None)
        # Whether any modifier in this pipeline produced/replaced per_position_loss
        per_position_loss_owned = False
        extra_kwargs = dict(kwargs)

        modified_loss = loss
        for modifier in self._enabled_modifiers:
            # Pass current per_position_loss along
            if per_position_loss is not None:
                extra_kwargs['per_position_loss'] = per_position_loss
            result = modifier.modify_loss(logits, targets, modified_loss, **extra_kwargs)

            # Support both scalar return and dict return with possible per_position_loss replacement
            if isinstance(result, dict):
                if 'per_position_loss' in result:
                    per_position_loss = result['per_position_loss']
                    per_position_loss_owned = True
                if 'loss' in result and result['loss'] is not None:
                    modified_loss = result['loss']
            else:
                modified_loss = result

        # Aggregate only if a modifier produced/replaced per_position_loss and no later modifier
        # provided a final scalar loss to use
        if per_position_loss_owned:
            mask = kwargs.get('mask', None)
            if mask is not None:
                mask_f = mask.float()
                modified_loss = (per_position_loss * mask_f).sum() / (mask_f.sum() + 1e-8)
            else:
                modified_loss = per_position_loss.mean()

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