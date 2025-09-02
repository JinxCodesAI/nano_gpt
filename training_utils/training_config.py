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
class UnmaskingStage:
    """Configuration for a single stage of unmasking training"""
    config: Union[StickyStageConfig, RandomStageConfig]
    
    def get_stage_type(self) -> UnmaskingStageType:
        return self.config.get_stage_type()
    
    def get_val_loss_stale_count(self) -> int:
        return self.config.val_loss_stale_count


@dataclass
class TrainingContext:
    """Configuration class for training parameters to avoid long parameter lists"""
    # Dataset configuration (NEW - replaces dataset-specific fields)
    dataset_config: Any = None  # DatasetConfig instance
    
    # Training configuration
    training_type: str = 'remasking'
    batch_size: int = 16
    block_size: int = 1024
    max_iters: int = 12000  # Maximum training iterations
    
    # Device configuration
    device: str = 'cuda'
    device_type: str = 'cuda'
    seed_offset: int = 0
    
    # Training iteration and stage tracking
    iter_num: int = 0
    current_stage: int = 0
    val_loss_stale_count: int = 0
    best_val_loss_this_stage: float = float('inf')
    
    # Training-specific loss computation parameters (KEEP - control loss during training)
    weight_loss_by_mask_ratio: bool = False  # Weight loss by sqrt(1.0 / mask_ratio) to balance gradient magnitude
    enable_entropy_penalty: bool = False     # Apply entropy penalty to loss
    max_entropy_penalty: float = 0.5          # Maximum entropy penalty multiplier
    entropy_penalty_start_iter: int = 6000  # Iteration to start applying entropy penalty
    uncertainty_factor: float = 0.0           # Label smoothing factor for loss computation
    
    # Remasking corruption parameters (KEEP for remasking training type)
    guaranteed_unmasked_max: float = 0.95
    guaranteed_unmasked_min: float = 0.1
    sticky_transition_start: int = 1000
    sticky_transition_end: int = 6000
    random_mask_warmup: int = 1000
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
    
    # Entropy multiplier tracking
    entropy_multiplier_ema: float = 1.0  # Exponential moving average of entropy multiplier
    entropy_multiplier_ema_factor: float = 0.99  # EMA decay factor
    
    # Transfer learning parameters
    transfer_learning_mode: str = 'from_scratch'  # 'from_scratch', 'feature_extraction', 'fine_tuning'
    pretrained_checkpoint_path: str = None  # Path to pretrained checkpoint for transfer learning
    model_mode: str = 'language_model'  # Target model mode: 'language_model', 'token_classifier', or 'sequence_classifier'
    
    def __post_init__(self):
        # Validate that dataset_config is provided
        if self.dataset_config is None:
            raise ValueError("dataset_config is required in TrainingContext")
    
    def get_current_stage_config(self):
        """Get configuration for current unmasking stage - delegate to dataset configuration"""
        if self.training_type != 'unmasking' or not self.dataset_config:
            return None
        
        return self.dataset_config.get_stage_config_for_iteration(self.iter_num)
    
    def advance_stage(self):
        """Advance to next unmasking stage and reset stale count"""
        if not self.dataset_config or not hasattr(self.dataset_config.training_config, 'UNMASKING_STAGES'):
            return False
            
        stages = self.dataset_config.training_config.UNMASKING_STAGES
        if self.current_stage < len(stages) - 1:
            self.current_stage += 1
            self.val_loss_stale_count = 0
            self.best_val_loss_this_stage = float('inf')
            return True
        return False