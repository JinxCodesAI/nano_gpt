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
            'weight': get_config_value('entropy_modifier_weight', 1.0),
            'use_for_weighting': get_config_value('entropy_modifier_use_for_weighting', False),
            'entropy_threshold': get_config_value('entropy_modifier_threshold', 0.0),
            'eps': get_config_value('entropy_modifier_eps', 1e-8),
        }
        modifiers.append(EntropyModifier(entropy_config))
    
    # Target Smoothing Modifier
    smoothing_enabled = get_config_value('target_smoothing_enabled', False)
    if smoothing_enabled:
        smoothing_config = {
            'enabled': True,
            'smoothing_factor': get_config_value('target_smoothing_factor', 0.1),
            'special_token_ids': get_config_value('target_smoothing_special_tokens', []),
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
            'use_sequence_level': get_config_value('mask_ratio_weight_sequence_level', True),
        }
        modifiers.append(MaskRatioWeightModifier(mask_weight_config))
    
    return LossModifierPipeline(modifiers)