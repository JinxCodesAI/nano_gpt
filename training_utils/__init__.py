"""
Modular training utilities for diffusion training.

This package contains the split components of the original train_utils.py file,
organized into logical modules for better maintainability and code organization.

Modules:
    training_config: Configuration classes and enums
    data_loading: Data loading, prefetching, and caching utilities
    validation_sets: Validation set creation and management
    masking_strategies: Masking and corruption functions
    batch_generation: Core batch generation logic
    model_evaluation: Loss estimation and evaluation
    training_progress: Stage progression and learning rate scheduling
    entropy_utils: Entropy penalty and label smoothing utilities
"""

import sys
sys.modules['train_utils'] = sys.modules[__name__]

# Export commonly used classes and functions for easy importing
from .training_config import (
    UnmaskingStageType,
    BaseStageConfig,
    StickyStageConfig,
    RandomStageConfig,
    SpanStageConfig,
    UnmaskingStage,
    TrainingContext
)

from .data_loading import (
    start_prefetch,
    stop_prefetch,
    clear_validation_cache
)

from .validation_sets import (
    create_unmasking_validation_set,
    get_unmasking_validation_batch,
    get_unmasking_training_batch_all_stages,
    create_remasking_validation_set,
    get_remasking_validation_batch
)

from .masking_strategies import (
    load_synthetic_model,
    apply_random_masking_gpu,
    apply_stage_masking,
    apply_target_driven_sticky_masking_gpu,
    apply_corruption_gpu
)

from .batch_generation import (
    get_batch,
    get_batch_unmasking,
    get_batch_remasking_binary
)

from .model_evaluation import (
    estimate_loss
)

from .training_progress import (
    update_stage_progress,
    get_lr
)

from .entropy_utils import (
    calculate_wrong_answer_entropy,
    get_current_entropy_penalty,
    update_entropy_multiplier_ema,
    apply_label_smoothing
)

# All exported symbols for wildcard imports (import * from utils)
__all__ = [
    # Configuration classes
    'UnmaskingStageType',
    'BaseStageConfig', 
    'StickyStageConfig',
    'RandomStageConfig',
    'SpanStageConfig',
    'UnmaskingStage',
    'TrainingContext',
    
    # Data loading functions
    'start_prefetch',
    'stop_prefetch', 
    'clear_validation_cache',
    
    # Validation set functions
    'create_unmasking_validation_set',
    'get_unmasking_validation_batch',
    'get_unmasking_training_batch_all_stages', 
    'create_remasking_validation_set',
    'get_remasking_validation_batch',
    
    # Masking strategy functions
    'load_synthetic_model',
    'apply_random_masking_gpu',
    'apply_stage_masking',
    'apply_target_driven_sticky_masking_gpu',
    'apply_corruption_gpu',
    
    # Batch generation functions
    'get_batch',
    'get_batch_unmasking',
    'get_batch_remasking_binary',
    
    # Model evaluation functions
    'estimate_loss',
    
    # Training progress functions
    'update_stage_progress',
    'get_lr',
    
    # Entropy utility functions
    'calculate_wrong_answer_entropy',
    'get_current_entropy_penalty', 
    'update_entropy_multiplier_ema',
    'apply_label_smoothing'
]