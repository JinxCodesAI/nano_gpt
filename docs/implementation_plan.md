# Implementation Plan: Multi-Mode Transformer with Transfer Learning

## Overview
This document outlines the step-by-step implementation plan to extend the current GPT implementation with sequence classification and token classification capabilities, while maintaining full backward compatibility with the existing MLM (Masked Language Modeling) training.

## Current State Analysis

### Existing Implementation (model.py + train.py)
- **Architecture**: Standard GPT with causal/bidirectional attention
- **Training**: Single mode operation (language modeling)
- **Features**: Position encoding options, modular loss modifier system, distributed training
- **Loss Modifiers**: Entropy modifier, target smoothing, mask ratio weighting
- **Scheduler**: Cosine LR scheduler with linear warmup
- **Strengths**: Clean, focused implementation with good training infrastructure

### Target Features
- **Multi-mode operation**: Language model, token classifier, sequence scorer
- **Transfer learning**: Pretrained model loading, freezing/unfreezing
- **Advanced heads**: Mode-specific output layers with proper initialization
- **Training modes**: Feature extraction vs. full fine-tuning
- **Enhanced loss modifiers**: Mode-aware modifier filtering
- **Advanced scheduling**: Transfer learning optimized schedulers

## Implementation Strategy

### Phase 1: Core Architecture Extensions

#### 1.1 Add ModelMode Enum and Configuration
**File**: `model.py`
**Location**: After imports, before existing classes

```python
from enum import Enum

class ModelMode(Enum):
    """Defines the operational modes for the transformer model"""
    LANGUAGE_MODEL = "language_model"      # Current MLM functionality
    TOKEN_CLASSIFIER = "token_classifier"  # Per-token classification
    SEQUENCE_SCORER = "sequence_scorer"    # Sequence-level 0-1 scoring
```

**Extend GPTConfig dataclass**:
```python
@dataclass
class GPTConfig:
    # Existing fields remain unchanged
    block_size: int = 1024
    vocab_size: int = 50304
    # ... all existing fields ...
    
    # New fields for multi-mode support
    mode: ModelMode = ModelMode.LANGUAGE_MODEL
    num_token_classes: int = 2  # For token classification
    cls_token_id: int = None  # For sequence scoring
    
    # Transfer learning support
    freeze_transformer: bool = False
    init_from_checkpoint: str = None
    unfreeze_at_iteration: int = None
    unfreeze_lr_multiplier: float = 0.1
    
    # Backward compatibility
    binary_classification: bool = False  # Legacy support
    
    def __post_init__(self):
        # Handle backward compatibility
        if self.binary_classification and self.mode == ModelMode.LANGUAGE_MODEL:
            self.mode = ModelMode.TOKEN_CLASSIFIER
            self.num_token_classes = 2
        
        # Enforce bidirectional attention for classification tasks
        if self.mode in [ModelMode.TOKEN_CLASSIFIER, ModelMode.SEQUENCE_SCORER]:
            if self.attention_type != 'bidirectional':
                self.attention_type = 'bidirectional'
```

#### 1.2 Extend GPT Class Constructor
**File**: `model.py`
**Method**: `GPT.__init__`

**Add after existing transformer creation**:
```python
# Create mode-specific output heads
if self.config.mode == ModelMode.LANGUAGE_MODEL:
    # Existing language modeling head
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.transformer.wte.weight = self.lm_head.weight  # Weight tying
elif self.config.mode == ModelMode.TOKEN_CLASSIFIER:
    # Token-level classification head
    self.lm_head = nn.Linear(config.n_embd, config.num_token_classes, bias=False)
    self._log_info(f"Token classifier head: {config.num_token_classes} classes per token")
elif self.config.mode == ModelMode.SEQUENCE_SCORER:
    # Sequence-level scoring head
    self.sequence_head = nn.Sequential(
        nn.Linear(config.n_embd, 1, bias=False),
        nn.Sigmoid()
    )
    # Initialize with small weights for stability
    with torch.no_grad():
        self.sequence_head[0].weight.normal_(0.0, 0.01)
    self._log_info("Sequence scorer head: continuous score 0-1")
```

**Add transfer learning initialization**:
```python
# Transfer learning: load pretrained weights if specified
if config.init_from_checkpoint is not None:
    self._load_pretrained_checkpoint(config.init_from_checkpoint)

# Freeze transformer if requested
if config.freeze_transformer:
    self.freeze_transformer_weights()
```

#### 1.3 Add Transfer Learning Methods
**File**: `model.py`
**Methods**: Add to GPT class

```python
def _load_pretrained_checkpoint(self, checkpoint_path):
    """Load pretrained weights, excluding heads"""
    self._log_info(f"Loading pretrained weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    state_dict = checkpoint.get('model', checkpoint)
    
    # Filter transformer weights, exclude heads
    transformer_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace('_orig_mod.', '')  # Remove torch.compile prefix
        if (clean_key.startswith('transformer.') and 
            not clean_key.startswith('lm_head') and 
            not clean_key.startswith('sequence_head')):
            transformer_state_dict[clean_key] = v
    
    # Load with strict=False to allow missing head weights
    missing_keys, unexpected_keys = self.load_state_dict(transformer_state_dict, strict=False)
    self._log_info(f"Loaded transformer weights: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected")

def freeze_transformer_weights(self):
    """Freeze transformer for feature extraction"""
    self._log_info("Freezing transformer weights for feature extraction")
    for param in self.transformer.parameters():
        param.requires_grad = False
    # Keep heads trainable
    if hasattr(self, 'lm_head'):
        for param in self.lm_head.parameters():
            param.requires_grad = True
    if hasattr(self, 'sequence_head'):
        for param in self.sequence_head.parameters():
            param.requires_grad = True

def unfreeze_transformer_weights(self):
    """Unfreeze transformer for full fine-tuning"""
    self._log_info("Unfreezing transformer weights for fine-tuning")
    for param in self.transformer.parameters():
        param.requires_grad = True

def get_frozen_status(self):
    """Check if transformer is frozen"""
    for param in self.transformer.parameters():
        if param.requires_grad:
            return False
    return True
```

#### 1.4 Update Forward Method
**File**: `model.py`
**Method**: `GPT.forward`

**Replace existing forward logic with mode-specific handling**:
```python
def forward(self, idx, targets=None, loss_modifiers=None):
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

    # Forward through transformer (unchanged)
    tok_emb = self.transformer.wte(idx)
    if hasattr(self.transformer, 'wpe'):
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
    else:
        x = self.transformer.drop(tok_emb)
        
    for block in self.transformer.h:
        x = block(x)
    x = self.transformer.ln_f(x)

    # Mode-specific output and loss computation
    if self.config.mode == ModelMode.SEQUENCE_SCORER:
        return self._forward_sequence_scorer(x, targets, loss_modifiers)
    elif self.config.mode == ModelMode.TOKEN_CLASSIFIER:
        return self._forward_token_classifier(x, targets, loss_modifiers)
    else:  # LANGUAGE_MODEL
        return self._forward_language_model(x, targets, loss_modifiers)

def _forward_sequence_scorer(self, x, targets, loss_modifiers):
    """Sequence scoring forward pass"""
    cls_output = x[:, 0, :]  # Extract [CLS] token
    logits = self.sequence_head(cls_output).squeeze(-1)
    
    if targets is not None:
        base_loss = F.mse_loss(logits, targets.float())
        
        # Apply loss modifiers if available and compatible with sequence scoring
        if loss_modifiers is not None and not loss_modifiers.is_empty():
            # Note: Some modifiers may not be applicable to sequence scoring
            loss = loss_modifiers.modify_loss(
                logits.unsqueeze(-1), targets, base_loss,
                model_mode=self.config.mode
            )
        else:
            loss = base_loss
    else:
        loss = None
    
    return logits, loss

def _forward_token_classifier(self, x, targets, loss_modifiers):
    """Token classification forward pass"""
    logits = self.lm_head(x)
    
    if targets is not None:
        num_classes = self.config.num_token_classes
        if targets.dim() == 3:  # Soft targets
            base_loss = F.cross_entropy(logits.view(-1, num_classes), targets.view(-1, num_classes))
        else:  # Hard targets with dynamic weighting
            base_loss = self._compute_weighted_classification_loss(logits, targets, num_classes)
        
        # Apply loss modifiers if available
        if loss_modifiers is not None and not loss_modifiers.is_empty():
            mask = targets != -1  # Valid token mask
            loss = loss_modifiers.modify_loss(
                logits, targets, base_loss, mask=mask,
                ignore_index=-1, model_mode=self.config.mode
            )
        else:
            loss = base_loss
    else:
        loss = None
    
    return logits, loss

def _forward_language_model(self, x, targets, loss_modifiers):
    """Language modeling forward pass (existing logic)"""
    if targets is not None:
        logits = self.lm_head(x)
        if loss_modifiers is not None and not loss_modifiers.is_empty():
            # Existing loss modifier logic
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = targets.view(-1)
            per_position_loss = F.cross_entropy(
                flat_logits, flat_targets,
                ignore_index=self.config.ignore_index,
                reduction='none'
            )
            per_position_loss = per_position_loss.view(x.size(0), x.size(1))
            mask = targets != self.config.ignore_index
            base_loss = (per_position_loss * mask.float()).sum() / (mask.float().sum() + 1e-8)
            loss = loss_modifiers.modify_loss(
                logits, targets, base_loss, mask=mask, 
                per_position_loss=per_position_loss, 
                ignore_index=self.config.ignore_index,
                model_mode=self.config.mode
            )
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.config.ignore_index)
    else:
        # Inference optimization
        logits = self.lm_head(x[:, [-1], :])
        loss = None
    
    return logits, loss
```

### Phase 1.5: Enhanced Loss Modifier System

#### 1.5.1 Extend BaseLossModifier for Mode Awareness
**File**: `loss_modifiers/base.py`
**Method**: Update `modify_loss` signature

```python
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
        **kwargs: Additional arguments
        
    Returns:
        Modified loss or dict with loss and metrics
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
```

#### 1.5.2 Update Existing Loss Modifiers
**Files**: `loss_modifiers/entropy_modifier.py`, `loss_modifiers/target_smoothing_modifier.py`, `loss_modifiers/mask_ratio_weight_modifier.py`

**Add mode support methods**:
```python
def supports_mode(self, mode: ModelMode) -> bool:
    """Check mode compatibility"""
    if mode == ModelMode.LANGUAGE_MODEL:
        return True
    elif mode == ModelMode.TOKEN_CLASSIFIER:
        return True  # Entropy and smoothing work for classification
    elif mode == ModelMode.SEQUENCE_SCORER:
        return False  # Most modifiers don't apply to sequence-level MSE loss
    return False

def modify_loss(self, logits, targets, loss, model_mode=None, **kwargs):
    """Mode-aware loss modification"""
    if model_mode and not self.supports_mode(model_mode):
        # Return unmodified loss for unsupported modes
        return loss
    
    # Existing modification logic...
```

#### 1.5.3 Update Loss Modifier Pipeline
**File**: `loss_modifiers/pipeline.py`
**Method**: Update `modify_loss` to filter compatible modifiers

```python
def modify_loss(self, logits, targets, loss, model_mode=None, **kwargs):
    """Apply all compatible modifiers in sequence"""
    if self.is_empty():
        return loss
    
    current_loss = loss
    per_position_loss = kwargs.get('per_position_loss', None)
    
    # Filter modifiers by mode compatibility
    compatible_modifiers = [
        m for m in self.modifiers 
        if m.is_enabled() and (model_mode is None or m.supports_mode(model_mode))
    ]
    
    for modifier in compatible_modifiers:
        result = modifier.modify_loss(
            logits, targets, current_loss, 
            model_mode=model_mode, **kwargs
        )
        
        if isinstance(result, dict):
            current_loss = result.get('loss', current_loss)
            if 'per_position_loss' in result:
                per_position_loss = result['per_position_loss']
                kwargs['per_position_loss'] = per_position_loss
        else:
            current_loss = result
    
    return current_loss
```

### Phase 2: Training Infrastructure Updates

#### 2.1 Update Training Script Configuration
**File**: `train.py`
**Location**: After existing config variables (around line 103)

```python
# Model mode configuration
model_mode = 'language_model'  # 'language_model', 'token_classifier', 'sequence_scorer'
num_token_classes = 2  # For token classification
cls_token_id = None  # For sequence scoring
freeze_transformer = False  # Feature extraction mode
init_from_checkpoint = None  # Path to pretrained model
unfreeze_at_iteration = None  # Dynamic unfreezing
unfreeze_lr_multiplier = 0.1  # LR multiplier when unfreezing
```

#### 2.2 Update Model Initialization
**File**: `train.py`
**Location**: Model initialization section (around line 197)

```python
# Update model_args to include new parameters
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout, 
    attention_type=attention_type, position_encoding=position_encoding,
    # New parameters
    mode=ModelMode[model_mode.upper()],
    num_token_classes=num_token_classes,
    cls_token_id=cls_token_id,
    freeze_transformer=freeze_transformer,
    init_from_checkpoint=init_from_checkpoint,
    unfreeze_at_iteration=unfreeze_at_iteration,
    unfreeze_lr_multiplier=unfreeze_lr_multiplier
)
```

#### 2.3 Add Dynamic Unfreezing Logic
**File**: `train.py`
**Location**: Training loop, before optimizer step (around line 340)

```python
# Dynamic unfreezing support
if (unfreeze_at_iteration is not None and 
    iter_num == unfreeze_at_iteration and 
    raw_model.get_frozen_status()):
    
    logger.log_info(f"Unfreezing transformer at iteration {iter_num}")
    raw_model.unfreeze_transformer_weights()
    
    # Adjust learning rate for stability
    for param_group in optimizer.param_groups:
        param_group['lr'] *= unfreeze_lr_multiplier
    
    logger.log_info(f"Reduced learning rate by factor {unfreeze_lr_multiplier}")
```

#### 2.4 Enhanced Learning Rate Scheduling
**File**: `core/scheduler.py`
**Location**: Add new scheduler classes

```python
class TransferLearningScheduler(LRScheduler):
    """
    Specialized scheduler for transfer learning with feature extraction and fine-tuning phases.
    
    Features:
    - Lower learning rates optimized for fine-tuning
    - Different schedules for frozen vs unfrozen phases
    - Smooth transitions when unfreezing
    """
    
    def __init__(
        self,
        base_lr: float = 5e-5,           # Lower LR for fine-tuning
        head_lr_multiplier: float = 10.0,  # Higher LR for new heads
        warmup_iters: int = 500,         # Shorter warmup
        feature_extraction_iters: int = 2000,  # Frozen phase duration
        unfreeze_lr_drop: float = 0.1,   # LR reduction when unfreezing
        decay_lr: bool = True,
        min_lr_ratio: float = 0.1
    ):
        self.base_lr = base_lr
        self.head_lr_multiplier = head_lr_multiplier
        self.warmup_iters = warmup_iters
        self.feature_extraction_iters = feature_extraction_iters
        self.unfreeze_lr_drop = unfreeze_lr_drop
        self.decay_lr = decay_lr
        self.min_lr_ratio = min_lr_ratio
    
    def get_lr(self, iter_num: int, is_frozen: bool = True) -> dict:
        """
        Get learning rates for different parameter groups.
        
        Returns:
            dict with 'transformer' and 'head' learning rates
        """
        # Phase 1: Feature extraction (transformer frozen)
        if iter_num < self.feature_extraction_iters and is_frozen:
            if iter_num < self.warmup_iters:
                # Warmup for head only
                progress = (iter_num + 1) / (self.warmup_iters + 1)
                head_lr = self.base_lr * self.head_lr_multiplier * progress
            else:
                # Constant LR for head, frozen transformer
                head_lr = self.base_lr * self.head_lr_multiplier
            
            return {
                'transformer': 0.0,  # Frozen
                'head': head_lr
            }
        
        # Phase 2: Full fine-tuning (transformer unfrozen)
        else:
            if not is_frozen:
                # Apply LR drop when first unfreezing
                effective_base_lr = self.base_lr * self.unfreeze_lr_drop
            else:
                effective_base_lr = self.base_lr
            
            # Cosine decay from unfreeze point
            if self.decay_lr and iter_num > self.feature_extraction_iters:
                decay_progress = (iter_num - self.feature_extraction_iters) / max(1, iter_num - self.feature_extraction_iters + 1000)
                decay_factor = 0.5 * (1.0 + math.cos(math.pi * min(decay_progress, 1.0)))
                transformer_lr = self.min_lr_ratio * effective_base_lr + decay_factor * (effective_base_lr - self.min_lr_ratio * effective_base_lr)
            else:
                transformer_lr = effective_base_lr
            
            return {
                'transformer': transformer_lr,
                'head': transformer_lr * 2.0  # Slightly higher for head
            }

class WarmupOnlyScheduler(LRScheduler):
    """
    Simple warmup-only scheduler for short fine-tuning runs.
    Good for classification tasks with small datasets.
    """
    
    def __init__(
        self,
        learning_rate: float,
        warmup_iters: int,
        hold_iters: int = None,  # Hold at peak LR, then decay
        min_lr_ratio: float = 0.1
    ):
        self.learning_rate = learning_rate
        self.warmup_iters = warmup_iters
        self.hold_iters = hold_iters or warmup_iters
        self.min_lr_ratio = min_lr_ratio
    
    def get_lr(self, iter_num: int) -> float:
        """Simple warmup then hold/decay schedule"""
        if iter_num < self.warmup_iters:
            # Linear warmup
            return self.learning_rate * (iter_num + 1) / (self.warmup_iters + 1)
        elif iter_num < self.warmup_iters + self.hold_iters:
            # Hold at peak
            return self.learning_rate
        else:
            # Linear decay to minimum
            decay_progress = (iter_num - self.warmup_iters - self.hold_iters) / max(1, iter_num - self.warmup_iters - self.hold_iters + 100)
            return self.learning_rate * (self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 - min(decay_progress, 1.0)))

class AdaptiveScheduler(LRScheduler):
    """
    Adaptive scheduler that adjusts based on validation performance.
    Useful for fine-tuning when optimal schedule length is unknown.
    """
    
    def __init__(
        self,
        initial_lr: float,
        patience: int = 3,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        warmup_iters: int = 100
    ):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.best_loss = float('inf')
        self.wait_count = 0
        self.reductions = 0
    
    def get_lr(self, iter_num: int) -> float:
        """Get current learning rate"""
        if iter_num < self.warmup_iters:
            # Linear warmup to current LR
            return self.current_lr * (iter_num + 1) / (self.warmup_iters + 1)
        return self.current_lr
    
    def step(self, val_loss: float) -> bool:
        """
        Update scheduler based on validation loss.
        Returns True if LR was reduced.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait_count = 0
            return False
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                old_lr = self.current_lr
                self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                self.wait_count = 0
                self.reductions += 1
                return self.current_lr < old_lr
        return False
```

### Phase 3: Dataset and Training Support

#### 3.1 Update Evaluator for Multi-Mode Support
**File**: `core/evaluator.py`
**Method**: Update evaluation logic to handle different modes

```python
def evaluate(self):
    """Mode-aware evaluation"""
    losses = {}
    
    for split in ['train', 'val']:
        losses_list = []
        for _ in range(self.eval_iters):
            X, Y = self.consumer.get_batch(split, self.device)
            
            with self.ctx:
                # All modes now support loss modifiers with mode filtering
                logits, loss = self.model(X, Y, loss_modifiers=self.loss_modifier_pipeline)
            
            losses_list.append(loss.item())
        
        losses[split] = np.mean(losses_list)
    
    return losses
```

#### 3.2 Create Configuration Templates
**File**: `config/token_classifier_config.py`

```python
"""Configuration template for token classification"""

# Override default config for token classification
model_mode = 'token_classifier'
attention_type = 'bidirectional'  # Required for classification
num_token_classes = 2  # Adjust as needed
freeze_transformer = True  # Start with feature extraction
unfreeze_at_iteration = 5000  # Unfreeze after warmup
init_from_checkpoint = 'path/to/pretrained/model.pt'

# Classification-specific training settings
learning_rate = 5e-5  # Lower LR for fine-tuning
warmup_iters = 1000
max_iters = 20000
eval_interval = 500

# Enable compatible loss modifiers for classification
loss_modifiers_enabled = True
entropy_modifier_enabled = True  # Works well for classification
target_smoothing_enabled = True  # Label smoothing for classification
mask_ratio_weight_enabled = False  # Not typically used for classification
```

**File**: `config/sequence_scorer_config.py`

```python
"""Configuration template for sequence scoring"""

# Override default config for sequence scoring
model_mode = 'sequence_scorer'
attention_type = 'bidirectional'  # Required for classification
cls_token_id = 101  # [CLS] token ID from tokenizer
freeze_transformer = True  # Start with feature extraction
unfreeze_at_iteration = 3000  # Unfreeze after initial training
init_from_checkpoint = 'path/to/pretrained/model.pt'

# Sequence scoring specific settings
learning_rate = 1e-4  # Higher LR for small head
warmup_iters = 500
max_iters = 15000
eval_interval = 250

# Most loss modifiers don't apply to sequence scoring (MSE loss)
loss_modifiers_enabled = True
entropy_modifier_enabled = False  # N/A for MSE loss
target_smoothing_enabled = False  # N/A for regression
mask_ratio_weight_enabled = False  # N/A for sequence-level task
```

### Phase 4: Validation and Testing

#### 4.1 Create Validation Script
**File**: `validate_multimode.py`

```python
"""Validation script for multi-mode functionality"""

import torch
from model import GPT, GPTConfig, ModelMode

def test_mode_switching():
    """Test model creation in different modes"""
    base_config = GPTConfig(
        n_layer=4, n_head=4, n_embd=128, 
        vocab_size=1000, block_size=64
    )
    
    # Test language model mode
    lm_config = GPTConfig(**base_config.__dict__, mode=ModelMode.LANGUAGE_MODEL)
    lm_model = GPT(lm_config)
    
    # Test token classifier mode  
    tc_config = GPTConfig(**base_config.__dict__, 
                         mode=ModelMode.TOKEN_CLASSIFIER,
                         num_token_classes=3)
    tc_model = GPT(tc_config)
    
    # Test sequence scorer mode
    ss_config = GPTConfig(**base_config.__dict__,
                         mode=ModelMode.SEQUENCE_SCORER,
                         cls_token_id=0)
    ss_model = GPT(ss_config)
    
    print("All modes created successfully!")

def test_transfer_learning():
    """Test transfer learning functionality"""
    # Create and save a base model
    base_config = GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=100)
    base_model = GPT(base_config)
    torch.save({'model': base_model.state_dict()}, 'test_checkpoint.pt')
    
    # Load into classifier
    classifier_config = GPTConfig(
        **base_config.__dict__,
        mode=ModelMode.TOKEN_CLASSIFIER,
        num_token_classes=2,
        init_from_checkpoint='test_checkpoint.pt',
        freeze_transformer=True
    )
    classifier = GPT(classifier_config)
    
    assert classifier.get_frozen_status(), "Transformer should be frozen"
    print("Transfer learning test passed!")

if __name__ == "__main__":
    test_mode_switching()
    test_transfer_learning()
```

## Implementation Order

### Week 1: Core Architecture
1. Add ModelMode enum and extend GPTConfig
2. Update GPT.__init__ with mode-specific heads
3. Add transfer learning methods
4. Test basic model creation

### Week 2: Forward Pass Updates
1. Implement mode-specific forward methods
2. Update loss computation for each mode
3. Test forward/backward passes
4. Validate gradient flows

### Week 3: Training Infrastructure
1. Update train.py configuration
2. Add dynamic unfreezing logic
3. Update evaluator for multi-mode support
4. Create configuration templates

### Week 4: Validation and Integration
1. Create comprehensive test suite
2. Validate backward compatibility
3. Performance testing
4. Documentation updates

## Backward Compatibility Guarantees

1. **Existing MLM Training**: All existing configurations with `training_type = 'MLM'` will work unchanged
2. **Configuration Compatibility**: New parameters have sensible defaults
3. **Model Loading**: Existing checkpoints can be loaded as base models
4. **API Stability**: All existing methods maintain their signatures

## Risk Mitigation

1. **Gradual Integration**: Each mode can be implemented and tested independently
2. **Feature Flags**: New functionality is opt-in via configuration
3. **Comprehensive Testing**: Validation scripts ensure no regressions
4. **Rollback Plan**: Changes are additive and can be easily reverted

## Success Metrics

1. **Functionality**: All three modes work correctly
2. **Performance**: No degradation in existing MLM training
3. **Transfer Learning**: Successful fine-tuning from pretrained models
4. **Code Quality**: Clean, maintainable implementation
5. **Documentation**: Clear usage examples and API documentation