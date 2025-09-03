#!/usr/bin/env python3
"""
Test script to debug sequence scoring batch generation in isolation
"""

import sys
import os
import torch
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training_utils.training_config import TrainingContext, UnmaskingStage, StickyStageConfig
from training_utils.batch_generation import get_batch_sequence_scoring
from model import GPT, GPTConfig, ModelMode

def create_test_context():
    """Create a minimal training context for testing"""
    
    # Copy important configuration from optimal5.py and train_run.py
    ctx = TrainingContext()
    
    # Basic settings
    ctx.training_type = 'sequence_scoring'
    ctx.batch_size = 4  # Small batch for testing
    ctx.block_size = 256  # Small block size for faster testing
    ctx.device = 'cpu'
    ctx.device_type = 'cpu'
    
    # Data settings
    ctx.data_dir = 'data/shakespeare_char'
    ctx.meta_vocab_size = 65  # Shakespeare char vocab size
    ctx.mask_token_id = 65
    ctx.extended_vocab_size = 80  # meta_vocab_size + 15 reserved
    ctx.cls_token_id = 66  # First special token after mask_token_id
    
    # Unmasking stages (copied from optimal5.py)
    ctx.unmasking_stages = [
        UnmaskingStage(StickyStageConfig(
            target_masked_ratio=0.4, 
            p1_probability=0.15, 
            p2_probability=0.3, 
            val_loss_stale_count=6
        )),
    ]
    
    # Other settings
    ctx.use_paragraph_boundaries = False
    ctx.current_stage = 0
    ctx.iter_num = 0
    
    print("Test context created:")
    print(f"  batch_size: {ctx.batch_size}")
    print(f"  block_size: {ctx.block_size}")
    print(f"  meta_vocab_size: {ctx.meta_vocab_size}")
    print(f"  mask_token_id: {ctx.mask_token_id}")
    print(f"  cls_token_id: {ctx.cls_token_id}")
    print(f"  Stage 0 target_ratio: {ctx.unmasking_stages[0].config.target_masked_ratio}")
    
    return ctx

def create_test_unmasking_model(ctx):
    """Create a dummy unmasking model for testing"""
    print("\nCreating test unmasking model...")
    
    config = GPTConfig(
        block_size=ctx.block_size,
        vocab_size=ctx.extended_vocab_size,
        n_layer=2,  # Small model for testing
        n_head=2,
        n_embd=64,
        dropout=0.0,
        mode=ModelMode.LANGUAGE_MODEL,
        attention_type='bidirectional'
    )
    
    model = GPT(config)
    model.eval()
    
    print(f"  Model params: {model.get_num_params()}")
    print(f"  Model vocab_size: {config.vocab_size}")
    
    return model

def test_sequence_scoring_generation():
    """Test sequence scoring batch generation"""
    print("=" * 60)
    print("TESTING SEQUENCE SCORING BATCH GENERATION")
    print("=" * 60)
    
    # Create test context
    ctx = create_test_context()
    
    # Create dummy unmasking model
    unmasking_model = create_test_unmasking_model(ctx)
    ctx.unmasking_model = unmasking_model
    
    # Check if data directory exists
    if not os.path.exists(ctx.data_dir):
        print(f"\nWARNING: Data directory {ctx.data_dir} not found!")
        print("Creating dummy data for testing...")
        
        # Create dummy data files
        os.makedirs(ctx.data_dir, exist_ok=True)
        
        # Create dummy train.bin (small Shakespeare-like data)
        dummy_data = np.random.randint(0, ctx.meta_vocab_size, size=10000, dtype=np.uint16)
        with open(os.path.join(ctx.data_dir, 'train.bin'), 'wb') as f:
            dummy_data.tobytes()
            f.write(dummy_data.tobytes())
        
        print(f"  Created dummy train.bin with {len(dummy_data)} tokens")
    
    try:
        print(f"\nGenerating sequence scoring batch...")
        print(f"  Split: train")
        print(f"  Context batch_size: {ctx.batch_size}")
        print(f"  Context block_size: {ctx.block_size}")
        
        # Test batch generation
        X, Y, mask = get_batch_sequence_scoring('train', ctx)
        
        print(f"\nBatch generation successful!")
        print(f"  X shape: {X.shape} (should be [{ctx.batch_size}, {ctx.block_size}])")
        print(f"  Y shape: {Y.shape} (should be [{ctx.batch_size}])")
        print(f"  mask shape: {mask.shape}")
        
        print(f"\nData analysis:")
        print(f"  X dtype: {X.dtype}")
        print(f"  Y dtype: {Y.dtype}")
        print(f"  X range: [{X.min().item()}, {X.max().item()}]")
        print(f"  Y range: [{Y.min().item():.4f}, {Y.max().item():.4f}]")
        print(f"  Y mean: {Y.mean().item():.4f}")
        print(f"  Y std: {Y.std().item():.4f}")
        
        print(f"\nMask analysis:")
        print(f"  mask dtype: {mask.dtype}")
        print(f"  Total mask ratio: {mask.float().mean().item():.4f}")
        print(f"  Per-sequence mask ratios: {mask.float().mean(dim=1).tolist()}")
        
        print(f"\nFirst sequence analysis:")
        first_seq = X[0]
        first_target = Y[0]
        first_mask = mask[0]
        
        print(f"  First sequence tokens: {first_seq[:10].tolist()}... (showing first 10)")
        print(f"  First target (masking ratio): {first_target.item():.4f}")
        print(f"  CLS token (should be {ctx.cls_token_id}): {first_seq[0].item()}")
        print(f"  Mask for first sequence: {first_mask.sum().item()}/{len(first_mask)} = {first_mask.float().mean().item():.4f}")
        
        # Test MSE loss calculation
        print(f"\nTesting MSE loss calculation:")
        dummy_predictions = torch.sigmoid(torch.randn_like(Y))  # Random predictions 0-1
        mse_loss = torch.nn.functional.mse_loss(dummy_predictions, Y.float())
        print(f"  Dummy predictions: {dummy_predictions.tolist()}")
        print(f"  Targets: {Y.tolist()}")
        print(f"  MSE loss: {mse_loss.item():.6f}")
        
        # Test with predictions close to targets
        close_predictions = Y + torch.randn_like(Y) * 0.01  # Very close predictions
        close_predictions = torch.clamp(close_predictions, 0, 1)  # Clamp to [0,1]
        close_loss = torch.nn.functional.mse_loss(close_predictions, Y.float())
        print(f"  Close predictions: {close_predictions.tolist()}")
        print(f"  Close MSE loss: {close_loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR during batch generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sequence_scoring_generation()
    print(f"\nTest result: {'SUCCESS' if success else 'FAILED'}")