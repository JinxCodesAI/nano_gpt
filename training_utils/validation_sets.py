"""
Validation set creation and management for diffusion training.
Handles creation of consistent validation sets across different training stages.
"""

import os
import numpy as np
import torch

from .training_config import TrainingContext, UnmaskingStageType
from .data_loading import find_double_newline_indices, get_data_cache, get_valid_indices_cache, get_validation_caches, set_validation_caches

# Global cache for sequence scoring validation set
sequence_scoring_val_set = None


def create_unmasking_validation_set(ctx: TrainingContext):
    """Create complete validation set with samples evenly distributed across all stages"""
    from .masking_strategies import apply_stage_masking  # Import here to avoid circular dependency
    from .entropy_utils import apply_label_smoothing  # Import here to avoid circular dependency
    
    val_batch_cache, progressive_val_cache, progressive_val_full_cache, unmasking_val_set, remasking_val_set = get_validation_caches()
    
    if unmasking_val_set is not None:
        print("Using existing validation set from cache")
        return  # Already created
    
    print("Creating validation set with samples from all stages...")
    
    data_cache = get_data_cache()
    valid_indices_cache = get_valid_indices_cache()
    
    # Cache validation data if not already cached
    if data_cache['val'] is None:
        data_cache['val'] = np.memmap(os.path.join(ctx.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        if ctx.use_paragraph_boundaries:
            valid_indices_cache['val'] = find_double_newline_indices(data_cache['val'], ctx.meta_vocab_size, ctx.block_size)
        else:
            valid_indices_cache['val'] = np.array([])
    
    data = data_cache['val']
    valid_indices = valid_indices_cache['val']
    
    total_samples = ctx.eval_iters * ctx.batch_size
    num_stages = len(ctx.validation_stages)
    samples_per_stage = total_samples // num_stages
    
    validation_batches = []
    
    # Generate samples for each stage
    torch.manual_seed(42)  # Fixed seed for reproducible validation set
    
    for stage_idx, stage_config in enumerate(ctx.validation_stages):
        stage_samples = samples_per_stage
        # Handle remainder samples
        if stage_idx < (total_samples % num_stages):
            stage_samples += 1
            
        stage_type = stage_config.get_stage_type()
        stage_info = f"  Stage {stage_idx} ({stage_type.value}): {stage_samples} samples"
        if stage_type == UnmaskingStageType.STICKY:
            config = stage_config.config
            stage_info += f" (target_ratio={config.target_masked_ratio:.1f}, p1={config.p1_probability:.1f}, p2={config.p2_probability:.1f})"
        elif stage_type == UnmaskingStageType.RANDOM:
            config = stage_config.config
            stage_info += f" (max_ratio={config.max_masked_ratio:.1f})"
        print(stage_info)
        
        # Generate batches for this stage
        stage_batches = []
        samples_generated = 0
        
        while samples_generated < stage_samples:
            batch_size = min(ctx.batch_size, stage_samples - samples_generated)
            
            # Sample data indices
            if len(valid_indices) == 0:
                ix_np = torch.randint(len(data) - ctx.block_size, (batch_size,)).numpy()
            else:
                ix_indices = torch.randint(len(valid_indices), (batch_size,)).numpy()
                ix_np = valid_indices[ix_indices]
            
            # Load data with vectorized indexing
            ix_expanded = ix_np[:, None] + np.arange(ctx.block_size)[None, :]  # (batch_size, block_size)
            x_np = data[ix_expanded].astype(np.int64)
            
            # Convert to tensor
            x = torch.from_numpy(x_np)
            if ctx.device_type == 'cuda':
                x = x.pin_memory().to(ctx.device, non_blocking=True)
            else:
                x = x.to(ctx.device)
            
            # Apply stage-specific masking
            masked_x, mask = apply_stage_masking(x, stage_config, ctx.mask_token_id, ctx.meta_vocab_size)
            
            # Apply label smoothing to targets if enabled
            y = x.clone()
            if ctx.uncertainty_factor > 0.0:
                # Determine special token IDs to exclude from smoothing
                special_token_ids = []
                if ctx.mask_token_id is not None and ctx.mask_token_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.mask_token_id)
                if ctx.wrong_token_id is not None and ctx.wrong_token_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.wrong_token_id)
                if ctx.remask_good_id is not None and ctx.remask_good_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.remask_good_id)
                if ctx.remask_wrong_id is not None and ctx.remask_wrong_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.remask_wrong_id)
                
                y = apply_label_smoothing(y, ctx.uncertainty_factor, ctx.extended_vocab_size, 
                                          special_token_ids=special_token_ids, device=ctx.device)
            
            stage_batches.append((masked_x.clone(), y.clone(), mask.clone()))
            samples_generated += batch_size
        
        validation_batches.extend(stage_batches)
    
    torch.manual_seed(1337 + ctx.seed_offset)  # Reset seed
    unmasking_val_set = validation_batches
    
    # Update global cache
    set_validation_caches(val_batch_cache, progressive_val_cache, progressive_val_full_cache, unmasking_val_set, remasking_val_set)
    
    print(f"Validation set created: {len(validation_batches)} batches, {sum(b[0].size(0) for b in validation_batches)} total samples")


def get_unmasking_validation_batch(ctx: TrainingContext, batch_idx=None):
    """Get a specific batch from the pre-created validation set"""
    val_batch_cache, progressive_val_cache, progressive_val_full_cache, unmasking_val_set, remasking_val_set = get_validation_caches()
    
    if unmasking_val_set is None:
        create_unmasking_validation_set(ctx)
        val_batch_cache, progressive_val_cache, progressive_val_full_cache, unmasking_val_set, remasking_val_set = get_validation_caches()
    
    if batch_idx is None:
        batch_idx = 0
    
    # Handle batch index wrapping
    batch_idx = batch_idx % len(unmasking_val_set)
    return unmasking_val_set[batch_idx]


def get_unmasking_training_batch_all_stages(ctx: TrainingContext):
    """Generate fresh training batch with samples distributed across all stages"""
    from .masking_strategies import apply_stage_masking  # Import here to avoid circular dependency
    from .entropy_utils import apply_label_smoothing  # Import here to avoid circular dependency
    
    data_cache = get_data_cache()
    valid_indices_cache = get_valid_indices_cache()
    
    # Cache training data if not already cached
    if data_cache['train'] is None:
        data_cache['train'] = np.memmap(os.path.join(ctx.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        if ctx.use_paragraph_boundaries:
            valid_indices_cache['train'] = find_double_newline_indices(data_cache['train'], ctx.meta_vocab_size, ctx.block_size)
        else:
            valid_indices_cache['train'] = np.array([])
    
    data = data_cache['train']
    valid_indices = valid_indices_cache['train']
    
    num_stages = len(ctx.unmasking_stages)
    samples_per_stage = ctx.batch_size // num_stages
    remainder = ctx.batch_size % num_stages
    
    # Create mixed batch with samples from all stages
    all_masked_x = []
    all_x = []
    all_masks = []
    
    for stage_idx, stage_config in enumerate(ctx.unmasking_stages):
        # Determine number of samples for this stage
        stage_samples = samples_per_stage + (1 if stage_idx < remainder else 0)
        
        if stage_samples > 0:
            # Sample data indices for this stage
            if len(valid_indices) == 0:
                ix_np = torch.randint(len(data) - ctx.block_size, (stage_samples,)).numpy()
            else:
                ix_indices = torch.randint(len(valid_indices), (stage_samples,)).numpy()
                ix_np = valid_indices[ix_indices]
            
            # Load data with vectorized indexing
            ix_expanded = ix_np[:, None] + np.arange(ctx.block_size)[None, :]  # (stage_samples, block_size)
            x_np = data[ix_expanded].astype(np.int64)
            
            # Convert to tensor
            x = torch.from_numpy(x_np)
            if ctx.device_type == 'cuda':
                x = x.pin_memory().to(ctx.device, non_blocking=True)
            else:
                x = x.to(ctx.device)
            
            # Apply masking for this stage
            masked_x, mask = apply_stage_masking(x, stage_config, ctx.mask_token_id, ctx.meta_vocab_size)
            
            # Apply label smoothing to targets if enabled
            y = x.clone()
            if ctx.uncertainty_factor > 0.0:
                # Determine special token IDs to exclude from smoothing
                special_token_ids = []
                if ctx.mask_token_id is not None and ctx.mask_token_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.mask_token_id)
                if ctx.wrong_token_id is not None and ctx.wrong_token_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.wrong_token_id)
                if ctx.remask_good_id is not None and ctx.remask_good_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.remask_good_id)
                if ctx.remask_wrong_id is not None and ctx.remask_wrong_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.remask_wrong_id)
                
                y = apply_label_smoothing(y, ctx.uncertainty_factor, ctx.extended_vocab_size, 
                                          special_token_ids=special_token_ids, device=ctx.device)
            
            all_masked_x.append(masked_x)
            all_x.append(y)  # Use smoothed targets
            all_masks.append(mask)
    
    # Concatenate all stages back into batch
    final_masked_x = torch.cat(all_masked_x, dim=0)
    final_x = torch.cat(all_x, dim=0)
    final_mask = torch.cat(all_masks, dim=0)
    
    # Shuffle the batch to mix stages randomly
    perm = torch.randperm(final_masked_x.size(0))
    final_masked_x = final_masked_x[perm]
    final_x = final_x[perm]
    final_mask = final_mask[perm]
    
    return final_masked_x, final_x, final_mask


def create_sequence_scoring_validation_set(ctx: TrainingContext, force_recreate=False):
    """Create complete validation set for sequence scoring training using validation_stages"""
    val_batch_cache, progressive_val_cache, progressive_val_full_cache, unmasking_val_set, remasking_val_set = get_validation_caches()

    # For sequence scoring, we'll store in a new cache variable (we'll need to extend the cache system)
    # For now, let's create it fresh each time but with consistent seed
    if not force_recreate:
        print("Creating sequence scoring validation set...")
    else:
        print("Force recreating sequence scoring validation set...")

    data_cache = get_data_cache()
    valid_indices_cache = get_valid_indices_cache()

    # Cache validation data if not already cached
    if data_cache['val'] is None:
        data_cache['val'] = np.memmap(os.path.join(ctx.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        if ctx.use_paragraph_boundaries:
            valid_indices_cache['val'] = find_double_newline_indices(data_cache['val'], ctx.meta_vocab_size, ctx.block_size)
        else:
            valid_indices_cache['val'] = np.array([])

    data = data_cache['val']
    valid_indices = valid_indices_cache['val']

    # Use validation_stages instead of unmasking_stages
    available_stages = ctx.validation_stages
    if not available_stages:
        raise ValueError("No validation_stages configured for sequence scoring validation")

    num_stages = len(available_stages)
    total_samples = ctx.eval_iters * ctx.batch_size
    samples_per_stage = total_samples // num_stages
    remainder = total_samples % num_stages

    validation_batches = []

    # Set consistent seed for validation set creation
    torch.manual_seed(1337 + ctx.seed_offset + 1000)  # Different seed from unmasking validation

    for stage_idx, stage_config in enumerate(available_stages):
        stage_samples = samples_per_stage + (1 if stage_idx < remainder else 0)
        print(f"  Stage {stage_idx}: {stage_samples} samples using {stage_config.config.get_stage_type().value}")

        stage_batches = []
        samples_generated = 0

        while samples_generated < stage_samples:
            batch_size = min(ctx.batch_size, stage_samples - samples_generated)

            # Sample data indices from validation split
            if len(valid_indices) == 0:
                ix_np = torch.randint(len(data) - ctx.block_size, (batch_size,)).numpy()
            else:
                ix_indices = torch.randint(len(valid_indices), (batch_size,)).numpy()
                ix_np = valid_indices[ix_indices]

            # Load data with vectorized indexing
            ix_expanded = ix_np[:, None] + np.arange(ctx.block_size)[None, :]  # (batch_size, block_size)
            x_np = data[ix_expanded].astype(np.int64)

            # Convert to tensor
            x = torch.from_numpy(x_np)
            if ctx.device_type == 'cuda':
                x = x.pin_memory().to(ctx.device, non_blocking=True)
            else:
                x = x.to(ctx.device)

            # Apply the same sequence scoring workflow as training
            # Step 1: Create unmasked version using unmasking model
            with torch.no_grad():
                unmasked_logits, _ = ctx.unmasking_model(x, None)
                unmasked_tokens = torch.argmax(unmasked_logits, dim=-1)

            # Step 2: Apply masking strategy for this stage
            mask = stage_config.apply_masking(x, ctx)

            # Step 3: Create masked input
            masked_x = x.clone()
            masked_x[mask] = ctx.mask_token_id

            # Step 4: Calculate masking ratios (targets)
            masking_ratios = mask.float().mean(dim=1)

            stage_batches.append((masked_x.clone(), masking_ratios.clone(), mask.clone()))
            samples_generated += batch_size

        validation_batches.extend(stage_batches)

    torch.manual_seed(1337 + ctx.seed_offset)  # Reset seed

    # Store in a global variable for now (we'll need to extend the cache system later)
    global sequence_scoring_val_set
    sequence_scoring_val_set = validation_batches

    print(f"Sequence scoring validation set created: {len(validation_batches)} batches, {sum(b[0].size(0) for b in validation_batches)} total samples")


def create_remasking_validation_set(ctx: TrainingContext, force_recreate=False):
    """Create complete validation set for remasking training with progressive corruption intensities"""
    from .masking_strategies import apply_corruption_gpu  # Import here to avoid circular dependency
    from .entropy_utils import apply_label_smoothing  # Import here to avoid circular dependency
    
    val_batch_cache, progressive_val_cache, progressive_val_full_cache, unmasking_val_set, remasking_val_set = get_validation_caches()
    
    if remasking_val_set is not None and not force_recreate:
        print("Using existing remasking validation set from cache")
        return  # Already created
    
    if force_recreate:
        print("Force recreating remasking validation set...")
        remasking_val_set = None
    
    print("Creating remasking validation set with progressive corruption intensities...")
    
    data_cache = get_data_cache()
    valid_indices_cache = get_valid_indices_cache()
    
    # Cache validation data if not already cached
    if data_cache['val'] is None:
        data_cache['val'] = np.memmap(os.path.join(ctx.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        if ctx.use_paragraph_boundaries:
            valid_indices_cache['val'] = find_double_newline_indices(data_cache['val'], ctx.meta_vocab_size, ctx.block_size)
        else:
            valid_indices_cache['val'] = np.array([])
    
    data = data_cache['val']
    valid_indices = valid_indices_cache['val']
    
    total_samples = ctx.eval_iters * ctx.batch_size
    validation_batches = []
    
    # Use fixed seed for reproducible validation set
    torch.manual_seed(42)  
    
    # Use a fixed mid-training corruption level for consistent validation
    # This represents a reasonable difficulty level for evaluation
    fixed_validation_iter = ctx.max_iters // 2  # Mid-training corruption level
    
    # Calculate what the corruption rate will be for validation
    corruption_min = 1.0 - ctx.guaranteed_unmasked_max
    corruption_max = 1.0 - ctx.guaranteed_unmasked_min
    if fixed_validation_iter < ctx.random_mask_warmup:
        progress = fixed_validation_iter / ctx.random_mask_warmup
        val_corruption_rate = corruption_min + progress * (corruption_max - corruption_min)
    else:
        val_corruption_rate = corruption_max
    
    print(f"  Using fixed validation corruption level: iter {fixed_validation_iter} = {val_corruption_rate:.1%} corruption")
    
    for k in range(ctx.eval_iters):
        # Sample data indices
        if len(valid_indices) == 0:
            ix_np = torch.randint(len(data) - ctx.block_size, (ctx.batch_size,)).numpy()
        else:
            ix_indices = torch.randint(len(valid_indices), (ctx.batch_size,)).numpy()
            ix_np = valid_indices[ix_indices]
        
        # Load data with vectorized indexing
        ix_expanded = ix_np[:, None] + np.arange(ctx.block_size)[None, :]
        x_np = data[ix_expanded].astype(np.int64)
        
        # Convert to tensor
        x = torch.from_numpy(x_np)
        if ctx.device_type == 'cuda':
            x = x.pin_memory().to(ctx.device, non_blocking=True)
        else:
            x = x.to(ctx.device)
        
        # Apply corruption using the fixed validation iteration (disable debug to avoid spam)
        corrupted_x, mask = apply_corruption_gpu(x, fixed_validation_iter, ctx.guaranteed_unmasked_max, ctx.guaranteed_unmasked_min,
                                                ctx.sticky_transition_start, ctx.sticky_transition_end, ctx.meta_vocab_size, 
                                                ctx.random_mask_warmup, ctx.p1_p2_ratio, debug=False)
        
        if ctx.training_type == 'remasking_binary':
            # Binary targets: 0=keep, 1=remask
            y = torch.full_like(x, ctx.remask_good_id)
            y[mask] = ctx.remask_wrong_id
        else:  # remasking
            # Target: original tokens at correct positions, wrong_token_id at corrupted positions
            y = x.clone()
            y[mask] = ctx.wrong_token_id
        
        # Apply label smoothing if enabled
        if ctx.uncertainty_factor > 0.0:
            if ctx.training_type == 'remasking_binary':
                # For binary classification, smooth over the 2 classes
                y = apply_label_smoothing(y, ctx.uncertainty_factor, ctx.extended_vocab_size, 
                                          device=ctx.device)
            else:  # remasking
                # For remasking, exclude special tokens from smoothing
                special_token_ids = []
                if ctx.wrong_token_id is not None and ctx.wrong_token_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.wrong_token_id)
                if ctx.mask_token_id is not None and ctx.mask_token_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.mask_token_id)
                if ctx.remask_good_id is not None and ctx.remask_good_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.remask_good_id)
                if ctx.remask_wrong_id is not None and ctx.remask_wrong_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.remask_wrong_id)
                
                y = apply_label_smoothing(y, ctx.uncertainty_factor, ctx.extended_vocab_size, 
                                          special_token_ids=special_token_ids, device=ctx.device)
        
        validation_batches.append((corrupted_x.clone(), y.clone(), mask.clone()))
    
    torch.manual_seed(1337 + ctx.seed_offset)  # Reset seed
    remasking_val_set = validation_batches
    
    # Update global cache
    set_validation_caches(val_batch_cache, progressive_val_cache, progressive_val_full_cache, unmasking_val_set, remasking_val_set)
    
    print(f"Remasking validation set created: {len(validation_batches)} batches, {sum(b[0].size(0) for b in validation_batches)} total samples")


def get_sequence_scoring_validation_batch(ctx: TrainingContext, batch_idx=None):
    """Get a specific batch from the pre-created sequence scoring validation set"""
    global sequence_scoring_val_set

    if 'sequence_scoring_val_set' not in globals() or sequence_scoring_val_set is None:
        create_sequence_scoring_validation_set(ctx, force_recreate=False)

    if batch_idx is None:
        batch_idx = 0

    # Handle batch index wrapping
    batch_idx = batch_idx % len(sequence_scoring_val_set)
    return sequence_scoring_val_set[batch_idx]


def get_remasking_validation_batch(ctx: TrainingContext, batch_idx=None):
    """Get a specific batch from the pre-created remasking validation set"""
    val_batch_cache, progressive_val_cache, progressive_val_full_cache, unmasking_val_set, remasking_val_set = get_validation_caches()
    
    if remasking_val_set is None:
        create_remasking_validation_set(ctx, force_recreate=False)
        val_batch_cache, progressive_val_cache, progressive_val_full_cache, unmasking_val_set, remasking_val_set = get_validation_caches()
    
    if batch_idx is None:
        batch_idx = 0
    
    # Handle batch index wrapping
    batch_idx = batch_idx % len(remasking_val_set)
    return remasking_val_set[batch_idx]