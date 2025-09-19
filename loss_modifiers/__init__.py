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
from .sequence_judge_weight_modifier import SequenceScoringJudgeWeightModifier
from .sequence_variance_modifier import SequenceScorerVarianceModifier
from .sequence_correlation_modifier import SequenceScorerCorrelationModifier
from .metrics_collector import MetricsCollectorModifier

__all__ = [
    'BaseLossModifier',
    'LossModifierPipeline',
    'EntropyModifier',
    'TargetSmoothingModifier',
    'MaskRatioWeightModifier',
    'SequenceScoringJudgeWeightModifier',
    'SequenceScorerVarianceModifier',
    'SequenceScorerCorrelationModifier',
    'MetricsCollectorModifier',
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

    # Judge Weight Modifier (Sequence scoring-based)
    judge_weight_enabled = get_config_value('judge_weight_modifier_enabled', False)
    if judge_weight_enabled:
        judge_weight_config = {
            'enabled': True,
            'judge_weight_checkpoint': get_config_value('judge_weight_checkpoint', None),
            'judge_weight_exponent': get_config_value('judge_weight_exponent', 1.0),
            'judge_weight_min_factor': get_config_value('judge_weight_min_factor', 0.1),
            'judge_weight_max_factor': get_config_value('judge_weight_max_factor', 10.0),
            'judge_weight_eps': get_config_value('judge_weight_eps', 1e-6),
            # Inherit device/dtype from main run config
            'device': get_config_value('device', 'cuda' if __import__('torch').cuda.is_available() else 'cpu'),
            'dtype': get_config_value('dtype', 'float32'),
        }
        modifiers.append(SequenceScoringJudgeWeightModifier(judge_weight_config))

    # Sequence Scorer Variance Modifier
    seq_var_enabled = get_config_value('sequence_variance_enabled', False)
    if seq_var_enabled:
        seq_var_config = {
            'enabled': True,
            'sequence_variance_scale': get_config_value('sequence_variance_scale', 2.0),
            'sequence_variance_alpha': get_config_value('sequence_variance_alpha', 1.5),
            'sequence_variance_eps': get_config_value('sequence_variance_eps', 1e-8),
        }
        modifiers.append(SequenceScorerVarianceModifier(seq_var_config))

    # Sequence Scorer Correlation Modifier
    seq_corr_enabled = get_config_value('sequence_correlation_enabled', False)
    if seq_corr_enabled:
        seq_corr_config = {
            'enabled': True,
            'sequence_correlation_alpha': get_config_value('sequence_correlation_alpha', 4.0),
            'sequence_correlation_eps': get_config_value('sequence_correlation_eps', 1e-8),
        }
        modifiers.append(SequenceScorerCorrelationModifier(seq_corr_config))

    # Generic metrics collector (currently: sequence scorer AARE)
    if get_config_value('seq_scorer_log_abs_rel_err', False):
        metrics_config = {
            'enabled': True,
            'collect_sequence_scorer_aare': True,
            'ema_alpha': get_config_value('sequence_metrics_ema_alpha', 0.1),
            'eps': get_config_value('sequence_metrics_eps', 1e-6),
        }
        modifiers.append(MetricsCollectorModifier(metrics_config))

    return LossModifierPipeline(modifiers)