"""
Example: Token Classification with Transfer Learning

This example shows how to:
1. Load a pretrained language model
2. Add a token classification head
3. Use feature extraction mode (frozen transformer)
4. Gradually unfreeze for full fine-tuning
"""

# To run this example:
# python train.py --config=config/token_classifier_config.py \
#   --init_from_checkpoint=path/to/pretrained/model.pt \
#   --dataset=your_token_classification_dataset \
#   --num_token_classes=3 \
#   --max_iters=10000

# The configuration will be loaded from config/token_classifier_config.py:

"""
Example config/token_classifier_config.py content:

# Token classification mode
model_mode = 'token_classifier'
num_token_classes = 3  # Adjust for your task

# Transfer learning settings  
freeze_transformer = True  # Start with feature extraction
unfreeze_at_iteration = 2000  # Unfreeze after 2000 iterations
init_from_checkpoint = 'checkpoints/pretrained_lm.pt'

# Fine-tuning optimized settings
learning_rate = 5e-5  # Lower LR for fine-tuning
warmup_iters = 500
max_iters = 8000
batch_size = 16

# Enable helpful loss modifiers
loss_modifiers_enabled = True
entropy_modifier_enabled = True  # Good for classification
target_smoothing_enabled = True  # Label smoothing
target_smoothing_factor = 0.1
"""

import torch
from model import GPT, GPTConfig, ModelMode
from loss_modifiers import create_loss_modifier_pipeline

def create_token_classifier_model():
    """Create a token classification model with transfer learning"""
    
    config = GPTConfig(
        # Base model architecture (should match pretrained model)
        n_layer=12,
        n_head=12, 
        n_embd=768,
        vocab_size=50304,
        block_size=1024,
        
        # Token classification specific
        mode=ModelMode.TOKEN_CLASSIFIER,
        num_token_classes=3,  # Adjust for your task
        
        # Transfer learning
        init_from_checkpoint='checkpoints/pretrained_lm.pt',
        freeze_transformer=True,  # Start frozen
        unfreeze_at_iteration=2000,
        unfreeze_lr_multiplier=0.1,
    )
    
    model = GPT(config)
    return model

def create_loss_modifiers():
    """Create loss modifier pipeline for token classification"""
    
    config = {
        'loss_modifiers_enabled': True,
        
        # Entropy modifier - good for classification
        'entropy_modifier_enabled': True,
        'entropy_modifier_weight': 1.0,
        
        # Label smoothing - helps with overfitting
        'target_smoothing_enabled': True,
        'target_smoothing_factor': 0.1,
        
        # Skip mask ratio weighting for classification
        'mask_ratio_weight_enabled': False,
    }
    
    return create_loss_modifier_pipeline(config)

def example_training_step(model, loss_modifier_pipeline):
    """Example of a training step with the new functionality"""
    
    # Example token classification data
    batch_size, seq_len = 4, 128
    
    # Input tokens
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    
    # Token classification targets (0, 1, 2 for 3 classes, -1 for padding)
    targets = torch.randint(-1, 3, (batch_size, seq_len))  # -1 = ignore padding
    
    # Forward pass with loss modifiers
    logits, loss = model(input_ids, targets, loss_modifiers=loss_modifier_pipeline)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Target shape: {targets.shape}")  
    print(f"Output logits shape: {logits.shape}")  # [batch, seq_len, num_classes]
    print(f"Loss: {loss.item():.4f}")
    
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)
    print(f"Predictions shape: {predictions.shape}")
    
    return loss

def example_unfreezing(model, optimizer, current_iteration):
    """Example of dynamic unfreezing during training"""
    
    unfreeze_iteration = 2000
    
    if (current_iteration == unfreeze_iteration and 
        model.get_frozen_status()):
        
        print(f"Unfreezing transformer at iteration {current_iteration}")
        model.unfreeze_transformer_weights()
        
        # Reduce learning rate for stability
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
            
        print(f"Reduced learning rate to {param_group['lr']}")

if __name__ == "__main__":
    print("Token Classification Example")
    print("=" * 40)
    
    # Create model and loss modifiers
    model = create_token_classifier_model()
    loss_pipeline = create_loss_modifiers()
    
    print(f"Model created in mode: {model.config.mode}")
    print(f"Number of classes: {model.config.num_token_classes}")
    print(f"Frozen status: {model.get_frozen_status()}")
    
    # Example training step
    print("\nExample training step:")
    loss = example_training_step(model, loss_pipeline)
    
    print("\nThis example shows the new multi-mode functionality!")
    print("To run actual training, use the train.py script with token_classifier_config.py")