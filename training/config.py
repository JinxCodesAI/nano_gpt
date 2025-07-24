"""
Training configuration management for nanoGPT.
"""
import os
import json
import yaml
from typing import Dict, Any, List, Optional

class TrainingConfig:
    """Manages training configuration with default values and overrides."""
    
    def __init__(self):
        # I/O settings
        self.out_dir = 'out'
        self.eval_interval = 2000
        self.log_interval = 1
        self.eval_iters = 200
        self.eval_only = False
        self.always_save_checkpoint = True
        self.init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
        
        # File logging
        self.log_dir = 'logs'
        self.file_logging = True
        
        # WandB logging
        self.wandb_log = False
        self.wandb_project = 'owt'
        self.wandb_run_name = 'gpt2'
        
        # Data settings
        self.dataset = 'fineweb10B'
        self.train_shard_filenames = ['train.bin']
        self.num_train_shards = 1
        
        # Training hyperparameters
        self.gradient_accumulation_steps = 5 * 8
        self.batch_size = 12
        self.block_size = 1024
        
        # Model architecture
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.dropout = 0.0
        self.bias = False
        self.n_hidden = None
        
        # Rotary embeddings
        self.use_rotary_embeddings = False
        self.rotary_base = 10000.0
        self.rotary_max_position_embeddings = 2048
        
        # Optimizer settings
        self.learning_rate = 6e-4
        self.max_iters = 600000
        self.weight_decay = 1e-1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        
        # Learning rate decay
        self.decay_lr = True
        self.warmup_iters = 2000
        self.lr_decay_iters = 600000
        self.min_lr = 6e-5
        
        # DDP settings
        self.backend = 'nccl'
        
        # System settings
        self.device = 'cuda'
        self.dtype = 'bfloat16'
        self.compile = True
        
        # LoRA settings
        self.embedding_mode = 'standard'
        self.attn_lora_rank = 0
        self.embedding_rank = 0
        self.lora_alpha = 1.0
        
        # Scaling schedule
        self.scaling_schedule_file = None
        self.scaling_schedule = []
        self.target_architecture_config = None
        
        # Analysis settings
        self.ignored_outlayers_sum = 0.01
        
        # Vocabulary settings
        self.shrunken_vocab_size = None
        self.vocab_remapping_file = None
        self.RARE_TOKEN_ID = None
        
        # Apply dataset-specific defaults
        self._apply_dataset_defaults()
    
    def _apply_dataset_defaults(self):
        """Apply dataset-specific configuration defaults."""
        if self.dataset == 'fineweb10B':
            self.num_train_shards = 103
            self.train_shard_filenames = [f"fineweb_train_{i:06d}.bin" for i in range(1, self.num_train_shards + 1)]
        else:
            self.num_train_shards = len(self.train_shard_filenames)
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Reapply dataset-specific defaults if dataset changed
        self._apply_dataset_defaults()
    
    def update_from_file(self, config_path: str):
        """Update configuration from a YAML or JSON file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
        
        self.update_from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def get_model_args(self) -> Dict[str, Any]:
        """Get model configuration arguments."""
        return {
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'n_embd': self.n_embd,
            'block_size': self.block_size,
            'bias': self.bias,
            'vocab_size': None,  # Will be set later
            'dropout': self.dropout,
            'n_hidden': self.n_hidden,
            'use_rotary_embeddings': self.use_rotary_embeddings,
            'rotary_base': self.rotary_base,
            'rotary_max_position_embeddings': self.rotary_max_position_embeddings,
            'embedding_mode': self.embedding_mode,
            'embedding_rank': self.embedding_rank,
            'attn_lora_rank': self.attn_lora_rank,
            'lora_alpha': self.lora_alpha
        }
    
    def get_overrideable_params(self) -> List[str]:
        """Get list of parameters that can be overridden during resume."""
        return [
            'n_hidden', 'n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
            'dropout', 'use_rotary_embeddings', 'rotary_base', 'rotary_max_position_embeddings',
            'embedding_mode', 'embedding_rank', 'attn_lora_rank', 'lora_alpha'
        ]

    def get_training_overrideable_params(self) -> List[str]:
        """Get list of training parameters that can be overridden during resume."""
        return [
            'learning_rate', 'max_iters', 'weight_decay', 'beta1', 'beta2', 'grad_clip',
            'decay_lr', 'warmup_iters', 'lr_decay_iters', 'min_lr', 'batch_size',
            'gradient_accumulation_steps', 'eval_interval', 'eval_iters'
        ]
    
    def validate(self):
        """Validate configuration settings."""
        errors = []
        
        # Check required paths exist
        if self.init_from == 'resume' and not os.path.exists(self.out_dir):
            errors.append(f"Resume directory does not exist: {self.out_dir}")
        
        # Check shrunken vocab consistency
        if self.shrunken_vocab_size is not None:
            if self.vocab_remapping_file is None:
                errors.append("vocab_remapping_file is required when using shrunken_vocab_size")
            elif not os.path.exists(self.vocab_remapping_file):
                errors.append(f"Vocab remapping file not found: {self.vocab_remapping_file}")
        
        # Check scaling schedule file
        if self.scaling_schedule_file is not None and not os.path.exists(self.scaling_schedule_file):
            errors.append(f"Scaling schedule file not found: {self.scaling_schedule_file}")
        
        # Check model architecture consistency
        if self.n_embd % self.n_head != 0:
            errors.append(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        
        # Check LoRA settings
        if self.embedding_mode == 'lora' and self.embedding_rank <= 0:
            errors.append("embedding_rank must be > 0 when embedding_mode is 'lora'")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
    
    def __repr__(self):
        return f"TrainingConfig({len(self.to_dict())} settings)"


def load_scaling_schedule(file_path: str, init_from: str) -> List[Dict[str, Any]]:
    """Load scaling schedule from file."""
    if not os.path.exists(file_path):
        print(f"Scaling schedule file not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                data = yaml.safe_load(f)
            elif file_path.endswith('.json'):
                data = json.load(f)
            else:
                print(f"Unsupported scaling schedule file format: {file_path}")
                return []
        
        # Extract operations from the data structure
        if isinstance(data, dict) and 'operations' in data:
            operations = data['operations']
        elif isinstance(data, list):
            operations = data
        else:
            print("Invalid scaling schedule format: expected list of operations or dict with 'operations' key")
            return []
        
        print(f"Loaded {len(operations)} operations from scaling schedule")
        return operations
        
    except Exception as e:
        print(f"Error loading scaling schedule: {e}")
        return []


def save_scaling_schedule(file_path: str, schedule_data: Dict[str, Any]):
    """Save scaling schedule to file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                yaml.safe_dump(schedule_data, f, default_flow_style=False, indent=2)
            elif file_path.endswith('.json'):
                json.dump(schedule_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        
        print(f"Saved scaling schedule to {file_path}")
        
    except Exception as e:
        print(f"Error saving scaling schedule: {e}")