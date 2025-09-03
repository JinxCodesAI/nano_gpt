"""
Training configuration classes and enums for diffusion training.
Contains all configuration data structures used across the training pipeline.
"""

from enum import Enum
from abc import ABC, abstractmethod
from typing import Union
from dataclasses import dataclass


class UnmaskingStageType(Enum):
    """Enumeration of available unmasking stage types"""
    STICKY = "sticky"
    RANDOM = "random"
    SPAN = "span"


@dataclass
class BaseStageConfig(ABC):
    """Base class for stage-specific configuration"""
    val_loss_stale_count: int
    
    @abstractmethod
    def get_stage_type(self) -> UnmaskingStageType:
        """Return the stage type"""
        pass


@dataclass
class StickyStageConfig(BaseStageConfig):
    """Configuration for sticky masking stages"""
    target_masked_ratio: float
    p1_probability: float
    p2_probability: float
    
    def get_stage_type(self) -> UnmaskingStageType:
        return UnmaskingStageType.STICKY


@dataclass
class RandomStageConfig(BaseStageConfig):
    """Configuration for random masking stages"""
    max_masked_ratio: float
    
    def get_stage_type(self) -> UnmaskingStageType:
        return UnmaskingStageType.RANDOM


@dataclass
class SpanStageConfig(BaseStageConfig):
    """Configuration for span masking stages"""
    spans_count: int
    
    def get_stage_type(self) -> UnmaskingStageType:
        return UnmaskingStageType.SPAN


@dataclass
class UnmaskingStage:
    """Configuration for a single stage of unmasking training"""
    config: Union[StickyStageConfig, RandomStageConfig, SpanStageConfig]
    
    def get_stage_type(self) -> UnmaskingStageType:
        return self.config.get_stage_type()
    
    def get_val_loss_stale_count(self) -> int:
        return self.config.val_loss_stale_count


@dataclass
class TrainingContext:
    """Configuration class for training parameters to avoid long parameter lists"""
    # Training configuration  
    training_type: str = 'unmasking'  # Options: 'unmasking', 'token_classification', 'sequence_scoring', 'remasking_binary' (backward compatibility)
    num_token_classes: int = 2  # Number of classes for token classification (flexible, not just binary)
    batch_size: int = 16
    block_size: int = 1024
    max_iters: int = 12000  # Maximum training iterations

    # Transfer learning support
    freeze_transformer: bool = False  # For transfer learning: freeze transformer, train only head
    init_from_checkpoint: str = None  # Path to pretrained checkpoint for transfer learning

    # Dynamic unfreezing parameters for two-stage training
    unfreeze_at_iteration: int = None  # Iteration at which to unfreeze transformer (None = never unfreeze)
    unfreeze_lr_multiplier: float = 0.1  # Learning rate multiplier when unfreezing
    transformer_frozen: bool = False  # Track current frozen state (set automatically)
    
    # Sequence scoring support - reference to unmasking model for reconstruction
    unmasking_model = None  # Pretrained unmasking model for sequence reconstruction
    
    # Device configuration
    device: str = 'cuda'
    device_type: str = 'cuda'
    seed_offset: int = 0
    
    # Data configuration
    data_dir: str = 'data/shakespeare_char'
    meta_vocab_size: int = None
    vocab_size: int = None  # extended_vocab_size (meta_vocab_size + 15 reserved special tokens)
    use_paragraph_boundaries: bool = True  # If True, start samples at paragraph boundaries (double newlines)
    use_all_stages_for_training: bool = False  # If True, generate training batches from all stages like validation
    weight_loss_by_mask_ratio: bool = False  # If True, weight loss by sqrt(1.0 / mask_ratio) to balance gradient magnitude
    
    # Token IDs
    mask_token_id: int = None
    wrong_token_id: int = None
    remask_good_id: int = None
    remask_wrong_id: int = None
    extended_vocab_size: int = None
    cls_token_id: int = None  # For sequence scoring mode
    
    # Training iteration and stage tracking
    iter_num: int = 0
    current_stage: int = 0
    val_loss_stale_count: int = 0
    best_val_loss_this_stage: float = float('inf')
    
    # Unmasking stages (only for unmasking training)
    unmasking_stages: list = None
    validation_stages: list = None  # Separate stages for validation set creation
    
    # Remasking strategy - only random supported
    
    # Remasking corruption parameters
    # Random corruption parameters
    guaranteed_unmasked_max: float = 0.95
    guaranteed_unmasked_min: float = 0.1
    sticky_transition_start: int = 1000
    sticky_transition_end: int = 6000
    random_mask_warmup: int = 1000
    
    # Sticky corruption parameters
    sticky_rounds: int = 4
    sticky_p1_p2_multiplier: float = 1.2
    sticky_p1_divisor: float = 2.0
    p1_p2_ratio: float = 1.0  # For remasking sticky masking: if 1.0 use random, else use sticky
    
    # Evaluation parameters
    eval_iters: int = 20
    
    # Learning rate parameters
    warmup_iters: int = 2000
    lr_decay_iters: int = 8000
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    
    # Entropy penalty parameters (incentivize uniform wrong answer distributions)
    enable_entropy_penalty: bool = False
    max_entropy_penalty: float = 0.5  # Penalty for concentrated wrong answers
    entropy_penalty_start_iter: int = 6000
    
    # Entropy multiplier tracking
    entropy_multiplier_ema: float = 1.0  # Exponential moving average of entropy multiplier
    entropy_multiplier_ema_factor: float = 0.99  # EMA decay factor
    
    # Label smoothing parameters
    uncertainty_factor: float = 0.0  # Label smoothing factor: 0 = no smoothing, >0 = apply smoothing
    
    def __post_init__(self):
        # Default unmasking stages if not provided
        if self.training_type == 'unmasking' and self.unmasking_stages is None:
            self.unmasking_stages = [
                UnmaskingStage(StickyStageConfig(target_masked_ratio=0.2, p1_probability=0.3, p2_probability=0.0, val_loss_stale_count=5)),
                UnmaskingStage(StickyStageConfig(target_masked_ratio=0.4, p1_probability=0.2, p2_probability=0.8, val_loss_stale_count=5)),
                UnmaskingStage(StickyStageConfig(target_masked_ratio=0.6, p1_probability=0.1, p2_probability=0.9, val_loss_stale_count=10)),
            ]
        
        # Default validation stages if not provided (same as training stages by default)
        if self.training_type == 'unmasking' and self.validation_stages is None:
            self.validation_stages = self.unmasking_stages
        
        # Keep both meta_vocab_size (for random token generation) and extended_vocab_size (for model)
        # vocab_size is kept for backward compatibility but should equal extended_vocab_size
        self.vocab_size = self.extended_vocab_size
    
    def get_current_stage_config(self) -> UnmaskingStage:
        """Get configuration for current unmasking stage"""
        if self.training_type not in ['unmasking', 'sequence_scoring'] or not self.unmasking_stages:
            return None
        
        if self.current_stage >= len(self.unmasking_stages):
            # Return last stage if we've exceeded all stages
            return self.unmasking_stages[-1]
        
        return self.unmasking_stages[self.current_stage]
    
    def advance_stage(self):
        """Advance to next unmasking stage and reset stale count"""
        if self.current_stage < len(self.unmasking_stages) - 1:
            self.current_stage += 1
            self.val_loss_stale_count = 0
            self.best_val_loss_this_stage = float('inf')
            return True
        return False