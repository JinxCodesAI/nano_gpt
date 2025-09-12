"""
Modular loss modifier system for flexible training loss adjustments.

This package provides a modular system for applying various loss modifications
during training. All modifiers are opt-in and can be combined in any configuration.

Usage:
    from loss_modifiers import create_loss_modifier_pipeline
    
    # Create pipeline with desired modifiers
    pipeline = create_loss_modifier_pipeline(config)
    
    # Use in training loop
    modified_loss = pipeline.modify_loss(logits, targets, original_loss)
"""

from .base import BaseLossModifier
from .pipeline import LossModifierPipeline
from .entropy_modifier import EntropyModifier
from .target_smoothing_modifier import TargetSmoothingModifier
from .mask_ratio_weight_modifier import MaskRatioWeightModifier

__all__ = [
    'BaseLossModifier',
    'LossModifierPipeline',
    'EntropyModifier',
    'TargetSmoothingModifier',
    'MaskRatioWeightModifier',
    'create_loss_modifier_pipeline',
]


def create_loss_modifier_pipeline(config):
    """
    Factory function to create a loss modifier pipeline from configuration.
    
    Args:
        config: Configuration dictionary or object with loss modifier settings
        
    Returns:
        LossModifierPipeline: Configured pipeline with enabled modifiers
    """
    modifiers = []
    
    # Helper function to get config value
    def get_config_value(key, default=None):
        if hasattr(config, key):
            return getattr(config, key)
        elif isinstance(config, dict):
            return config.get(key, default)
        else:
            return default
    
    # Check if any modifiers are enabled
    loss_modifiers_enabled = get_config_value('loss_modifiers_enabled', False)
    if not loss_modifiers_enabled:
        return LossModifierPipeline([])  # Return empty pipeline
    
    # Entropy Modifier
    entropy_enabled = get_config_value('entropy_modifier_enabled', False)
    if entropy_enabled:
        entropy_config = {
            'enabled': True,
            'verbose': get_config_value('entropy_modifier_verbose', False),
            'weight': get_config_value('entropy_modifier_weight', 1.0),
            'entropy_threshold': get_config_value('entropy_modifier_threshold', 0.0),
            'eps': get_config_value('entropy_modifier_eps', 1e-8),
        }
        modifiers.append(EntropyModifier(entropy_config))
    
    # Target Smoothing Modifier
    smoothing_enabled = get_config_value('target_smoothing_enabled', False)
    if smoothing_enabled:
        special_tokens_raw = get_config_value('target_smoothing_special_tokens', [])
        
        # Parse special tokens - handle both list and comma-delimited string formats
        if isinstance(special_tokens_raw, str):
            if special_tokens_raw.strip():
                special_tokens = [int(x.strip()) for x in special_tokens_raw.split(',')]
            else:
                special_tokens = []
        elif isinstance(special_tokens_raw, list):
            special_tokens = special_tokens_raw
        else:
            special_tokens = []
        
        smoothing_config = {
            'enabled': True,
            'smoothing_factor': get_config_value('target_smoothing_factor', 0.1),
            'special_token_ids': special_tokens,
            'exclude_padding': get_config_value('target_smoothing_exclude_padding', True),
            'padding_token_id': get_config_value('target_smoothing_padding_token', -100),
        }
        modifiers.append(TargetSmoothingModifier(smoothing_config))
    
    # Mask Ratio Weight Modifier
    mask_weight_enabled = get_config_value('mask_ratio_weight_enabled', False)
    if mask_weight_enabled:
        mask_weight_config = {
            'enabled': True,
            'power': get_config_value('mask_ratio_weight_power', 0.5),
            'min_weight': get_config_value('mask_ratio_weight_min', 0.1),
            'max_weight': get_config_value('mask_ratio_weight_max', 10.0),
            'eps': get_config_value('mask_ratio_weight_eps', 1e-8),
        }
        modifiers.append(MaskRatioWeightModifier(mask_weight_config))
    
    return LossModifierPipeline(modifiers)