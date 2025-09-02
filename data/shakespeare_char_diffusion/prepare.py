"""
Enhanced preparation script for Shakespeare character diffusion training dataset.
Pre-generates training and validation data for all iterations to decouple data generation from training.
"""
import os
import sys
import pickle
import numpy as np
import torch
from typing import Dict, Any, List, Tuple

# Add parent directories to import training utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from training_config import (UNMASKING_STAGES, VALIDATION_STAGES, BLOCK_SIZE, SPECIAL_TOKEN_OFFSET,
                              VALIDATION_SAMPLES_PER_STAGE, VALIDATION_SAMPLES_PER_FILE, VALIDATION_BATCH_SIZE)
from data_utils import create_iteration_mapping, apply_stage_masking


def create_enhanced_meta() -> Dict[str, Any]:
    """Create enhanced metadata with training information"""
    # Load original meta from parent dataset
    parent_meta_path = os.path.join('..', 'shakespeare_char', 'meta.pkl')
    with open(parent_meta_path, 'rb') as f:
        base_meta = pickle.load(f)
    
    # Add training-specific metadata
    enhanced_meta = {
        **base_meta,
        'extended_vocab_size': base_meta['vocab_size'] + SPECIAL_TOKEN_OFFSET,
        'special_tokens': {
            'mask_token_id': base_meta['vocab_size'],
            'wrong_token_id': base_meta['vocab_size'] + 1,
            'remask_good_id': base_meta['vocab_size'] + 2,
            'remask_wrong_id': base_meta['vocab_size'] + 3,
        },
        'dataset_type': 'character',
        'block_size': BLOCK_SIZE,  # CONSTRAINT: Fixed for this dataset
        'supported_model_modes': ['language_model', 'token_classifier'],
        'training_stages': len(UNMASKING_STAGES),
        'validation_stages': len(VALIDATION_STAGES),
        'batch_cache_size': 1000,
    }
    return enhanced_meta


def copy_tokenized_data():
    """Copy tokenized data from parent dataset"""
    parent_train = os.path.join('..', 'shakespeare_char', 'train.bin')
    parent_val = os.path.join('..', 'shakespeare_char', 'val.bin')
    
    # Copy the binary data files
    with open(parent_train, 'rb') as src, open('train.bin', 'wb') as dst:
        dst.write(src.read())
    
    with open(parent_val, 'rb') as src, open('val.bin', 'wb') as dst:
        dst.write(src.read())


def prepare_cached_data(meta: Dict[str, Any]):
    """Pre-compute expensive operations and cache them"""
    # Load data for preprocessing
    train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')
    
    # For shakespeare dataset, we don't use paragraph boundaries, but we'll create 
    # valid indices arrays for consistency
    train_indices = np.arange(len(train_data) - BLOCK_SIZE, dtype=np.int64)
    val_indices = np.arange(len(val_data) - BLOCK_SIZE, dtype=np.int64)
    
    # Save cached valid indices
    np.save('cached_data/train_valid_indices.npy', train_indices)
    np.save('cached_data/val_valid_indices.npy', val_indices)
    
    print(f"Cached {len(train_indices)} training valid indices")
    print(f"Cached {len(val_indices)} validation valid indices")


def generate_validation_sample_for_stage(stage_config: Dict, sample_idx: int,
                                        val_data: np.memmap, valid_indices: np.ndarray, 
                                        meta: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a single validation sample for a specific stage"""
    device = 'cpu'  # Generate on CPU for storage
    
    # Deterministic sampling based on stage and sample index
    torch.manual_seed(42 + sample_idx)  
    if len(valid_indices) > 0:
        ix_idx = torch.randint(len(valid_indices), (1,)).numpy()[0]
        ix = valid_indices[ix_idx]
    else:
        ix = torch.randint(len(val_data) - BLOCK_SIZE, (1,)).numpy()[0]
    
    # Load single sequence
    x_np = val_data[ix:ix + BLOCK_SIZE].astype(np.int64)
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)  # Add batch dimension
    
    # Apply stage masking
    mask_token_id = meta['special_tokens']['mask_token_id']
    meta_vocab_size = meta['vocab_size']
    
    masked_x, mask = apply_stage_masking(x, stage_config, mask_token_id, meta_vocab_size)
    
    # Target is original x
    y = x.clone()
    
    return masked_x.squeeze(0), y.squeeze(0), mask.squeeze(0)  # Remove batch dimension


def prepare_validation_pools(meta: Dict):
    """Generate fixed validation pools for all stages - dataset layer responsibility"""
    print("Generating fixed validation pools for all stages...")
    
    # Create validation directory structure
    validation_dir = 'validation'
    os.makedirs(validation_dir, exist_ok=True)
    
    # Load validation data
    val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')
    val_indices = np.load('cached_data/val_valid_indices.npy')
    
    # Generate validation pools for each stage
    for stage_idx, stage_config in enumerate(VALIDATION_STAGES):
        print(f"  Generating validation pool for stage {stage_idx+1}/{len(VALIDATION_STAGES)}")
        
        stage_samples = []
        
        # Generate fixed number of samples for this stage
        for sample_idx in range(VALIDATION_SAMPLES_PER_STAGE):
            sample = generate_validation_sample_for_stage(
                stage_config, sample_idx, val_data, val_indices, meta
            )
            stage_samples.append(sample)
        
        # Save samples in files of VALIDATION_SAMPLES_PER_FILE size
        num_files = (len(stage_samples) + VALIDATION_SAMPLES_PER_FILE - 1) // VALIDATION_SAMPLES_PER_FILE
        
        for file_idx in range(num_files):
            start_idx = file_idx * VALIDATION_SAMPLES_PER_FILE
            end_idx = min(start_idx + VALIDATION_SAMPLES_PER_FILE, len(stage_samples))
            file_samples = stage_samples[start_idx:end_idx]
            
            filename = f'{validation_dir}/stage_{stage_idx}_file_{file_idx}.pt'
            torch.save(file_samples, filename)
        
        print(f"    Saved {len(stage_samples)} samples in {num_files} files")
    
    # Save validation metadata
    validation_meta = {
        'num_stages': len(VALIDATION_STAGES),
        'samples_per_stage': VALIDATION_SAMPLES_PER_STAGE,
        'samples_per_file': VALIDATION_SAMPLES_PER_FILE,
        'total_samples': len(VALIDATION_STAGES) * VALIDATION_SAMPLES_PER_STAGE
    }
    
    with open(f'{validation_dir}/validation_meta.pkl', 'wb') as f:
        pickle.dump(validation_meta, f)
    
    print(f"✓ Validation pools generated: {validation_meta['num_stages']} stages, {validation_meta['total_samples']} total samples")


def generate_training_batch_for_iteration(iteration: int, stage_config: Dict,
                                       train_data: np.memmap, valid_indices: np.ndarray,
                                       meta: Dict, batch_size: int = 192) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate training batch for a specific iteration and stage"""
    device = 'cpu'  # Generate on CPU for storage
    
    # Use iteration as seed for reproducible but varied training data
    torch.manual_seed(1337 + iteration)
    
    # Sample indices
    if len(valid_indices) > 0:
        ix_indices = torch.randint(len(valid_indices), (batch_size,)).numpy()
        ix_np = valid_indices[ix_indices]
    else:
        ix_np = torch.randint(len(train_data) - BLOCK_SIZE, (batch_size,)).numpy()
    
    # Vectorized data loading
    ix_expanded = ix_np[:, None] + np.arange(BLOCK_SIZE)[None, :]
    x_np = train_data[ix_expanded].astype(np.int64)
    x = torch.from_numpy(x_np).to(device)
    
    # Apply stage masking
    mask_token_id = meta['special_tokens']['mask_token_id']
    meta_vocab_size = meta['vocab_size']
    
    masked_x, mask = apply_stage_masking(x, stage_config, mask_token_id, meta_vocab_size)
    
    # Target is original x
    y = x.clone()
    
    return masked_x, y, mask


def pre_generate_training_data(max_iters: int, eval_interval: int, meta: Dict):
    """Pre-generate training and validation data for all iterations"""
    os.makedirs('prepared_batches', exist_ok=True)
    
    # Load data
    train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')
    
    # Load cached valid indices
    train_indices = np.load('cached_data/train_valid_indices.npy')
    val_indices = np.load('cached_data/val_valid_indices.npy')
    
    # Create iteration-to-stage mapping
    iteration_mapping = create_iteration_mapping(max_iters, UNMASKING_STAGES)
    
    print(f"Pre-generating training and validation data for {max_iters} iterations...")
    
    # Generate data for training iterations and validation points
    validation_iterations = list(range(0, max_iters, eval_interval))
    total_batches = len(validation_iterations)
    
    for i, iteration in enumerate(validation_iterations):
        print(f"Generating batch {i+1}/{total_batches} (iteration {iteration})", end=" ... ")
        
        # Get current stage for this iteration
        stage_idx = iteration_mapping.get(iteration, len(UNMASKING_STAGES) - 1)
        current_stage_config = UNMASKING_STAGES[stage_idx]
        
        # Generate training batch
        train_batch = generate_training_batch_for_iteration(
            iteration, current_stage_config, train_data, train_indices, meta
        )
        torch.save(train_batch, f'prepared_batches/train_iter_{iteration:04d}.pt')
        
        # NOTE: Validation data is now generated separately in prepare_validation_pools()
        # Training no longer generates per-iteration validation batches
        print("✓")
    
    print(f"Pre-generated {total_batches} training and validation batch files")


def main():
    """Main preparation function"""
    print("=== Enhanced Shakespeare Character Diffusion Dataset Preparation ===")
    
    # Setup directory structure
    os.makedirs('cached_data', exist_ok=True)
    os.makedirs('prepared_batches', exist_ok=True)
    
    # Create enhanced metadata
    print("Creating enhanced metadata...")
    meta = create_enhanced_meta()
    with open('meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    print(f"✓ Enhanced metadata created (vocab_size: {meta['vocab_size']}, extended: {meta['extended_vocab_size']})")
    
    # Copy tokenized data from parent dataset
    print("Copying tokenized data from parent dataset...")
    copy_tokenized_data()
    print("✓ Tokenized data copied")
    
    # Pre-compute cached data structures
    print("Pre-computing cached data structures...")
    prepare_cached_data(meta)
    print("✓ Cached data prepared")
    
    # Generate fixed validation pools for all stages
    print("Generating fixed validation pools...")
    prepare_validation_pools(meta)
    print("✓ Fixed validation pools generated")

    # Pre-generate training data for 8000 iterations with eval every 200 iterations
    print("Pre-generating training batches...")
    pre_generate_training_data(max_iters=8000, eval_interval=200, meta=meta)
    print("✓ All training data pre-generated")
    
    # Summary
    train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')
    
    print("\n=== Dataset Summary ===")
    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")
    print(f"Original vocab size: {meta['vocab_size']}")
    print(f"Extended vocab size: {meta['extended_vocab_size']}")
    print(f"Block size constraint: {meta['block_size']}")
    print(f"Training stages: {meta['training_stages']}")
    print(f"Validation stages: {meta['validation_stages']}")
    print(f"Prepared batch files: {len([f for f in os.listdir('prepared_batches') if f.endswith('.pt')])}")
    
    print("\n✅ Shakespeare character diffusion dataset preparation complete!")
    print(f"Dataset ready for training with block_size={BLOCK_SIZE}")


if __name__ == '__main__':
    main()