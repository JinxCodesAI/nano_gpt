"""
Core batch generation logic for diffusion training.
Handles batch creation for different training types.
"""

import os
import numpy as np
import torch

from .training_config import TrainingContext, UnmaskingStageType
from .data_loading import (
    find_double_newline_indices, 
    start_prefetch, 
    get_data_cache, 
    get_valid_indices_cache, 
    get_prefetch_queue,
    get_validation_caches, 
    set_validation_caches,
    _prefetch_enabled
)
from .validation_sets import (
    get_unmasking_validation_batch, 
    get_unmasking_training_batch_all_stages,
    get_remasking_validation_batch
)
from .masking_strategies import (
    apply_stage_masking, 
    apply_corruption_gpu,
    get_progressive_validation_iterations
)


def apply_unmasking_strategy(x, split, ctx: TrainingContext, validation_sample_idx=None):
    """
    Shared masking logic for unmasking and sequence scoring.
    Returns: masked_x, mask, permutation_applied
    """
    
    permutation_applied = None
    # For validation: create samples from all stages for comprehensive evaluation
    if split == 'val':
        torch.manual_seed(42 + (validation_sample_idx or 0))
        
        # Distribute validation samples across all validation stages
        if ctx.validation_stages and len(ctx.validation_stages) > 1:
            stage_idx = (validation_sample_idx or 0) % len(ctx.validation_stages)
            stage_config = ctx.validation_stages[stage_idx]
            # Apply stage-specific masking
            masked_x, mask = apply_stage_masking(x, stage_config, ctx.mask_token_id, ctx.meta_vocab_size)
        else:
            # Fallback to current stage if no stages defined
            stage_config = ctx.get_current_stage_config()
            if stage_config is None:
                raise ValueError(f"No stage configuration available for {ctx.training_type} training")
            # Apply stage-specific masking
            masked_x, mask = apply_stage_masking(x, stage_config, ctx.mask_token_id, ctx.meta_vocab_size)
    else:
        # For training: distribute batch samples across all stages 
        if ctx.unmasking_stages and ctx.current_stage >= 0:
            # Use all stages if use_all_stages_for_training is enabled, otherwise up to current stage
            if ctx.use_all_stages_for_training:
                available_stages = ctx.unmasking_stages  # Use ALL stages
            else:
                available_stages = ctx.unmasking_stages[:ctx.current_stage + 1]  # Up to current stage
            num_stages = len(available_stages)
            
            # Distribute batch_size samples evenly across all available stages
            samples_per_stage = ctx.batch_size // num_stages
            remainder = ctx.batch_size % num_stages
            
            # Create mixed batch with samples from all stages
            all_masked_x = []
            all_masks = []
            
            start_idx = 0
            for stage_idx, stage_config in enumerate(available_stages):
                # Determine number of samples for this stage
                stage_samples = samples_per_stage + (1 if stage_idx < remainder else 0)
                
                if stage_samples > 0:
                    # Get DIFFERENT subset of data for this stage (FIX: was x[:stage_samples])
                    stage_x = x[start_idx:start_idx + stage_samples]
                    start_idx += stage_samples
                    
                    # Apply masking for this stage
                    stage_masked_x, stage_mask = apply_stage_masking(stage_x, stage_config, ctx.mask_token_id, ctx.meta_vocab_size)
                    
                    all_masked_x.append(stage_masked_x)
                    all_masks.append(stage_mask)
            
            # Concatenate all stages back into batch
            masked_x = torch.cat(all_masked_x, dim=0)
            mask = torch.cat(all_masks, dim=0)
            
            # Shuffle the batch to mix stages randomly
            perm = torch.randperm(masked_x.size(0))
            masked_x = masked_x[perm]
            mask = mask[perm]
            permutation_applied = perm
            
            # Debug: Log what stages are being used (disabled for sequence_scoring to reduce log noise)
            if ctx.iter_num % 20 == 0 and ctx.training_type != 'sequence_scoring':
                stage_counts = [samples_per_stage + (1 if i < remainder else 0) for i in range(num_stages)]
                stage_types = [stage.config.get_stage_type().value for stage in available_stages]
                print(f"  DEBUG SHARED: Using {num_stages} stages with counts {stage_counts}, types {stage_types}")
            
            # Log stage distribution occasionally for training only
            if split == 'train' and ctx.iter_num % 1500 == 0 and ctx.iter_num > 0:
                stage_counts = [samples_per_stage + (1 if i < remainder else 0) for i in range(num_stages)]
                print(f"Training iter {ctx.iter_num}: Mixed batch from {num_stages} stages: {stage_counts}")
        else:
            # Fallback to current stage configuration
            stage_config = ctx.get_current_stage_config()
            if stage_config is None:
                raise ValueError(f"No stage configuration available for {ctx.training_type} training")
            
            # Apply single stage masking (fallback case)
            masked_x, mask = apply_stage_masking(x, stage_config, ctx.mask_token_id, ctx.meta_vocab_size)
    
    if split == 'val':
        torch.manual_seed(1337 + ctx.seed_offset)
    
    
    return masked_x, mask, permutation_applied


def get_batch(split, ctx: TrainingContext, validation_sample_idx=None):
    """Main batch generation function that delegates to specific training type functions"""
    if ctx.training_type == 'unmasking':
        return get_batch_unmasking(split, ctx, validation_sample_idx)
    elif ctx.training_type in ['remasking_binary', 'token_classification']:
        # Support both old and new names for backward compatibility
        return get_batch_token_classification(split, ctx, validation_sample_idx)
    elif ctx.training_type == 'sequence_scoring':
        return get_batch_sequence_scoring(split, ctx, validation_sample_idx)
    else:
        raise ValueError(f"Unsupported training type: {ctx.training_type}")


def get_batch_unmasking(split, ctx: TrainingContext, validation_sample_idx=None):
    """Stage-based unmasking with target-driven sticky masking"""
    from .entropy_utils import apply_label_smoothing  # Import here to avoid circular dependency
    
    val_batch_cache, progressive_val_cache, progressive_val_full_cache, unmasking_val_set, remasking_val_set = get_validation_caches()
    data_cache = get_data_cache()
    valid_indices_cache = get_valid_indices_cache()
    prefetch_queue = get_prefetch_queue()

    # For validation, use the pre-created validation set distributed across all stages
    if split == 'val':
        return get_unmasking_validation_batch(ctx, validation_sample_idx)
    
    # For training, check if we should generate batch from all stages
    if split == 'train' and ctx.use_all_stages_for_training:
        return get_unmasking_training_batch_all_stages(ctx)

    # Cache memory-mapped data and valid indices - MAJOR SPEEDUP
    if data_cache[split] is None:
        if split == 'train':
            data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
        # Cache the expensive valid indices computation
        if ctx.use_paragraph_boundaries:
            print(f"Computing valid indices for {split} (paragraph boundaries)... (one-time cost)")
            valid_indices_cache[split] = find_double_newline_indices(data_cache[split], ctx.meta_vocab_size, ctx.block_size)
        else:
            print(f"Using random sampling for {split} (no paragraph boundaries)")
            valid_indices_cache[split] = np.array([])  # Empty array indicates random sampling
        print(f"Found {len(valid_indices_cache[split])} valid indices for {split}")
        
        # Start prefetching for training data
        if split == 'train':
            start_prefetch(ctx)

    # Try to get prefetched data for training
    x_np = None
    if split == 'train' and _prefetch_enabled:
        try:
            x_np = prefetch_queue.get_nowait()
        except:
            pass  # Queue empty, generate normally
    
    # Generate data if not prefetched
    if x_np is None:
        data = data_cache[split]
        valid_indices = valid_indices_cache[split]
        
        # Fast index sampling - all on CPU to avoid GPU-CPU sync
        if len(valid_indices) == 0:
            if split == 'val':
                torch.manual_seed(42)
                ix_np = torch.randint(len(data) - ctx.block_size, (ctx.batch_size,)).numpy()
                torch.manual_seed(1337 + ctx.seed_offset)
            else:
                ix_np = torch.randint(len(data) - ctx.block_size, (ctx.batch_size,)).numpy()
        else:
            if split == 'val':
                torch.manual_seed(42)
                ix_indices = torch.randint(len(valid_indices), (ctx.batch_size,)).numpy()
                ix_np = valid_indices[ix_indices]
                torch.manual_seed(1337 + ctx.seed_offset)
            else:
                ix_indices = torch.randint(len(valid_indices), (ctx.batch_size,)).numpy()
                ix_np = valid_indices[ix_indices]

        # VECTORIZED DATA LOADING - Use advanced indexing for parallel loading
        ix_expanded = ix_np[:, None] + np.arange(ctx.block_size)[None, :]  # (batch_size, block_size)
        x_np = data[ix_expanded].astype(np.int64)
    
    # Single GPU transfer with pinned memory
    x = torch.from_numpy(x_np)
    if ctx.device_type == 'cuda':
        x = x.pin_memory().to(ctx.device, non_blocking=True)
    else:
        x = x.to(ctx.device)

    # Apply shared masking strategy
    masked_x, mask, perm = apply_unmasking_strategy(x, split, ctx, validation_sample_idx)
    
    # Apply same permutation to original x if mixed batch was created
    if perm is not None:
        x = x[perm]

    # Target is original x
    y = x.clone()
    
    # Apply label smoothing if enabled
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

    # Cache validation batch for consistency
    if split == 'val':
        if validation_sample_idx is not None:
            cache_key = f"unmasking_{validation_sample_idx}"
            progressive_val_cache[cache_key] = (masked_x, y, mask)
        else:
            val_batch_cache = (masked_x, y, mask)
        
        # Update global cache
        set_validation_caches(val_batch_cache, progressive_val_cache, progressive_val_full_cache, unmasking_val_set, remasking_val_set)

    return masked_x, y, mask


def get_batch_token_classification(split, ctx: TrainingContext, validation_sample_idx=None):
    """
    Token-level classification training with flexible number of classes.
    Refactored from get_batch_remasking_binary to support multi-class classification.
    """
    from .entropy_utils import apply_label_smoothing  # Import here to avoid circular dependency
    
    # Use pre-created validation set for validation
    if split == 'val':
        return get_remasking_validation_batch(ctx, validation_sample_idx)
    
    # Training data generation - fast implementation 
    data_cache = get_data_cache()
    valid_indices_cache = get_valid_indices_cache()
    prefetch_queue = get_prefetch_queue()

    # Use same data caching and prefetching as remasking
    if data_cache[split] is None:
        if split == 'train':
            data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
        if ctx.use_paragraph_boundaries:
            print(f"Computing valid indices for {split} (paragraph boundaries)... (one-time cost)")
            valid_indices_cache[split] = find_double_newline_indices(data_cache[split], ctx.meta_vocab_size, ctx.block_size)
        else:
            print(f"Using random sampling for {split} (no paragraph boundaries)")
            valid_indices_cache[split] = np.array([])  # Empty array indicates random sampling
        print(f"Found {len(valid_indices_cache[split])} valid indices for {split}")
        
        if split == 'train':
            start_prefetch(ctx)

    # Try to get prefetched data for training (reuse existing prefetch system)
    x_np = None
    if split == 'train' and _prefetch_enabled:
        try:
            x_np = prefetch_queue.get_nowait()
        except:
            pass

    # Generate data if not prefetched (same as remasking)
    if x_np is None:
        data = data_cache[split]
        valid_indices = valid_indices_cache[split]
        
        if len(valid_indices) == 0:
            if split == 'val':
                torch.manual_seed(42)
                ix_np = torch.randint(len(data) - ctx.block_size, (ctx.batch_size,)).numpy()
                torch.manual_seed(1337 + ctx.seed_offset)
            else:
                ix_np = torch.randint(len(data) - ctx.block_size, (ctx.batch_size,)).numpy()
        else:
            if split == 'val':
                torch.manual_seed(42)
                ix_indices = torch.randint(len(valid_indices), (ctx.batch_size,)).numpy()
                ix_np = valid_indices[ix_indices]
                torch.manual_seed(1337 + ctx.seed_offset)
            else:
                ix_indices = torch.randint(len(valid_indices), (ctx.batch_size,)).numpy()
                ix_np = valid_indices[ix_indices]

        ix_expanded = ix_np[:, None] + np.arange(ctx.block_size)[None, :]  # (batch_size, block_size)
        x_np = data[ix_expanded].astype(np.int64)
    
    # Single GPU transfer
    x = torch.from_numpy(x_np)
    if ctx.device_type == 'cuda':
        x = x.pin_memory().to(ctx.device, non_blocking=True)
    else:
        x = x.to(ctx.device)

    # Determine which iteration to use for corruption strategy
    if split == 'val' and validation_sample_idx is not None:
        # For progressive validation, use the specific validation iteration
        progressive_iterations = get_progressive_validation_iterations(ctx.eval_iters, ctx.max_iters)
        corruption_iter = progressive_iterations[validation_sample_idx % len(progressive_iterations)]
    else:
        corruption_iter = ctx.iter_num

    # Use unified corruption function (random or sticky based on p1_p2_ratio)
    corrupted_x, mask = apply_corruption_gpu(x, corruption_iter, ctx.guaranteed_unmasked_max, ctx.guaranteed_unmasked_min,
                                           ctx.sticky_transition_start, ctx.sticky_transition_end, ctx.meta_vocab_size, 
                                           ctx.random_mask_warmup, ctx.p1_p2_ratio, debug=True)

    # Token classification targets: support flexible number of classes
    if ctx.num_token_classes == 2:
        # Backward compatibility: binary classification using existing remask_good_id/remask_wrong_id
        y = torch.full_like(x, ctx.remask_good_id)
        y[mask] = ctx.remask_wrong_id
    else:
        # Multi-class classification: generate class labels based on corruption patterns
        # For multi-class, use simple 0/1 labels for now (can be extended later)
        # Class 0: uncorrupted, Class 1: corrupted (for backwards compatibility with binary)
        y = torch.zeros_like(x, dtype=torch.long)
        y[mask] = 1
    
    # Apply label smoothing if enabled
    if ctx.uncertainty_factor > 0.0:
        if ctx.num_token_classes == 2:
            # Binary classification: smooth over remask_good_id, remask_wrong_id
            y = apply_label_smoothing(y, ctx.uncertainty_factor, ctx.extended_vocab_size, 
                                      device=ctx.device)
        else:
            # Multi-class: smooth over the actual number of classes
            y = apply_label_smoothing(y, ctx.uncertainty_factor, ctx.num_token_classes, 
                                      device=ctx.device)

    # No caching needed for training - validation uses pre-created set
    return corrupted_x, y, mask


def get_batch_sequence_scoring(split, ctx: TrainingContext, validation_sample_idx=None):
    """
    Sequence-level scoring training with masking + unmasking model reconstruction workflow:
    1. Sample sequences and apply masking (same as unmasking training)
    2. Calculate actual masking ratios for each sequence
    3. Use unmasking_model to reconstruct masked sequences
    4. Return reconstructed sequences with [CLS] token and masking ratios as targets
    """
    
    if ctx.unmasking_model is None:
        raise ValueError("Sequence scoring requires unmasking_model in TrainingContext")
    
    # Step 1: Sample sequences and apply masking (reuse unmasking logic)
    # Use existing data loading infrastructure from unmasking
    from .entropy_utils import apply_label_smoothing  # Import here to avoid circular dependency
    
    val_batch_cache, progressive_val_cache, progressive_val_full_cache, unmasking_val_set, remasking_val_set = get_validation_caches()
    data_cache = get_data_cache()
    valid_indices_cache = get_valid_indices_cache()
    prefetch_queue = get_prefetch_queue()

    # For validation, we need to create a modified validation batch with sequence scoring workflow
    if split == 'val':
        # For now, fall through to training logic to apply the sequence scoring workflow
        # TODO: Could optimize by pre-creating sequence scoring validation sets
        pass

    # Training data generation - identical to get_batch_unmasking until masking step
    if data_cache[split] is None:
        if split == 'train':
            data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
        if ctx.use_paragraph_boundaries:
            print(f"Computing valid indices for {split} (paragraph boundaries)... (one-time cost)")
            valid_indices_cache[split] = find_double_newline_indices(data_cache[split], ctx.meta_vocab_size, ctx.block_size - 1)  # -1 for [CLS] token
        else:
            print(f"Using random sampling for {split} (no paragraph boundaries)")
            valid_indices_cache[split] = np.array([])
        print(f"Found {len(valid_indices_cache[split])} valid indices for {split}")

    # Generate clean text data (same as unmasking - will mask later)
    data = data_cache[split]
    valid_indices = valid_indices_cache[split]

    # Handle validation seed consistency
    if len(valid_indices) == 0:
        if split == 'val':
            torch.manual_seed(42 + (validation_sample_idx or 0))
            ix_np = torch.randint(len(data) - (ctx.block_size - 1), (ctx.batch_size,)).numpy()
            torch.manual_seed(1337 + ctx.seed_offset)
        else:
            ix_np = torch.randint(len(data) - (ctx.block_size - 1), (ctx.batch_size,)).numpy()
    else:
        if split == 'val':
            torch.manual_seed(42 + (validation_sample_idx or 0))
            ix_indices = torch.randint(len(valid_indices), (ctx.batch_size,)).numpy()
            ix_np = valid_indices[ix_indices]
            torch.manual_seed(1337 + ctx.seed_offset)
        else:
            ix_indices = torch.randint(len(valid_indices), (ctx.batch_size,)).numpy()
            ix_np = valid_indices[ix_indices]

    # Load clean text sequences (block_size - 1 tokens to leave room for [CLS])
    ix_expanded = ix_np[:, None] + np.arange(ctx.block_size - 1)[None, :]
    clean_text_data = data[ix_expanded].astype(np.int64)
    
    # Convert to torch tensor
    clean_text = torch.from_numpy(clean_text_data)
    if ctx.device_type == 'cuda':
        clean_text = clean_text.pin_memory().to(ctx.device, non_blocking=True)
    else:
        clean_text = clean_text.to(ctx.device)

    
    # Step 2: Apply masking using shared logic (identical to unmasking training)
    masked_text, mask, perm = apply_unmasking_strategy(clean_text, split, ctx, validation_sample_idx)
    
    # Apply same permutation to original clean_text if mixed batch was created
    if perm is not None:
        clean_text = clean_text[perm]
    

    # Step 3: Calculate actual masking ratios for each sequence (these become our targets)
    masking_ratios = mask.float().mean(dim=1)  # Shape: (batch_size,) - ratio per sequence
    
    # Step 4: Use unmasking model to reconstruct the masked sequences
    ctx.unmasking_model.eval()
    with torch.no_grad():
        # Forward pass through unmasking model to get reconstructed sequences
        reconstructed_logits, _ = ctx.unmasking_model(masked_text, None)
        
        # Get most likely tokens for reconstruction
        reconstructed_tokens = torch.argmax(reconstructed_logits, dim=-1)
        
        # Create final sequences: keep unmasked tokens, use predictions for masked tokens
        final_sequences = clean_text.clone()
        final_sequences[mask] = reconstructed_tokens[mask]

    # Step 5: Prepend [CLS] token to create input for sequence scorer
    cls_token_id = ctx.cls_token_id
    if cls_token_id is None:
        raise ValueError("CLS token ID not set in training context for sequence scoring")
    cls_tokens = torch.full((ctx.batch_size, 1), cls_token_id, device=ctx.device, dtype=torch.long)
    x = torch.cat([cls_tokens, final_sequences], dim=1)  # Shape: (batch_size, block_size)
    
    # Step 6: Masking ratios become our target scores (already 0-1 range)
    y = masking_ratios  # Shape: (batch_size,)
    
    # Return the actual mask that was used for masking (for correct logging statistics)
    # Need to pad mask to match x's shape (which includes CLS token)
    cls_mask = torch.zeros((ctx.batch_size, 1), device=ctx.device, dtype=torch.bool)
    padded_mask = torch.cat([cls_mask, mask], dim=1)  # Shape: (batch_size, block_size)

    return x, y, padded_mask