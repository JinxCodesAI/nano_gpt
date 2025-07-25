"""
Model operations and architectural transformations for nanoGPT training.
"""
import os
import time
import pickle
import torch
from typing import Dict, Any, List, Tuple, Optional


def transfer_optimizer_state(new_optimizer, old_state_dict, old_param_dict, model):
    """Transfer optimizer state after architectural changes by matching parameter names."""
    print("Transferring optimizer state by parameter name matching...")
    
    # Get the new parameter dictionary
    new_param_dict = {name: p for name, p in model.named_parameters()}
    
    # Clear optimizer state completely to avoid KeyError issues
    new_optimizer.state.clear()
    
    # Update param_groups to contain current model parameters
    new_params = [p for p in model.parameters() if p.requires_grad]
    for group in new_optimizer.param_groups:
        group['params'] = new_params
    
    # Transfer matching state
    transferred_count = 0
    total_old_params = len(old_param_dict)
    
    for old_name, old_param in old_param_dict.items():
        if old_name in new_param_dict:
            new_param = new_param_dict[old_name]
            
            # Check if the parameter shapes match
            if old_param.shape == new_param.shape:
                # Find the parameter ID in the old optimizer state
                old_param_id = id(old_param)
                if old_param_id in old_state_dict['state']:
                    # Transfer the state to the new parameter ID
                    new_param_id = id(new_param)
                    # Deep copy the state to avoid reference issues
                    old_state = old_state_dict['state'][old_param_id]
                    new_state = {}
                    for key, value in old_state.items():
                        if torch.is_tensor(value):
                            new_state[key] = value.clone().detach()
                        else:
                            new_state[key] = value
                    new_optimizer.state[new_param_id] = new_state
                    transferred_count += 1
    
    print(f"Successfully transferred optimizer state for {transferred_count}/{total_old_params} parameters")
    print(f"New optimizer now tracks {len(new_params)} parameters")


def transfer_optimizer_state_by_shape(new_optimizer, old_state_dict, model):
    """Transfer optimizer state by matching parameter shapes when names don't match."""
    print("Transferring optimizer state by shape matching...")
    
    # Group old parameters by shape
    old_params_by_shape = {}
    for param_id, state in old_state_dict['state'].items():
        if 'exp_avg' in state:
            shape = tuple(state['exp_avg'].shape)
            if shape not in old_params_by_shape:
                old_params_by_shape[shape] = []
            old_params_by_shape[shape].append(state)
    
    # Transfer state to new parameters with matching shapes
    new_state = {}
    transferred_count = 0
    
    for param in model.parameters():
        param_shape = tuple(param.shape)
        if param_shape in old_params_by_shape and old_params_by_shape[param_shape]:
            # Use the first available state with this shape
            old_state = old_params_by_shape[param_shape].pop(0)
            new_param_id = id(param)
            new_state[new_param_id] = old_state.copy()
            transferred_count += 1
    
    # Update the new optimizer's state
    new_optimizer.state = new_state
    
    print(f"Successfully transferred optimizer state for {transferred_count} parameters by shape matching")


def log_detailed_params(model_to_log, master_process: bool = True):
    """Log detailed parameter count of the model."""
    if master_process:
        print("\nDetailed parameter count:")
        detailed_params = model_to_log.get_detailed_param_count()
        
        for component, counts in detailed_params.items():
            total = counts['total']
            trainable = counts['trainable']
            frozen = total - trainable
            
            if component == 'total':
                print(f"\n{component.upper()}:")
            else:
                print(f"{component}:")
            
            print(f"  Total: {total:,}")
            print(f"  Trainable: {trainable:,}")
            if frozen > 0:
                print(f"  Frozen: {frozen:,}")


def log_model_architecture(model, iter_num: int, is_initial: bool = False, is_target: bool = False):
    """Log the current model architecture."""
    config = model.config
    
    if is_initial:
        print(f"\n=== INITIAL MODEL ARCHITECTURE (iter {iter_num}) ===")
    elif is_target:
        print(f"\n=== TARGET MODEL ARCHITECTURE ===")
    else:
        print(f"\n=== CURRENT MODEL ARCHITECTURE (iter {iter_num}) ===")
    
    print(f"Layers: {config.n_layer}")
    print(f"Heads: {config.n_head}")
    print(f"Embedding dim: {config.n_embd}")
    print(f"Hidden dim: {config.n_hidden if config.n_hidden else 4 * config.n_embd}")
    print(f"Block size: {config.block_size}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Dropout: {config.dropout}")
    print(f"Bias: {config.bias}")
    print(f"Rotary embeddings: {config.use_rotary_embeddings}")
    
    if config.use_rotary_embeddings:
        print(f"Rotary base: {config.rotary_base}")
        print(f"Rotary max positions: {config.rotary_max_position_embeddings}")
    
    print(f"Embedding mode: {config.embedding_mode}")
    print(f"Embedding LoRA rank: {config.embedding_rank}")
    print(f"Attention LoRA rank: {config.attn_lora_rank}")
    print(f"LoRA alpha: {config.lora_alpha}")
    
    param_count = model.get_num_params() / 1e6
    print(f"Parameters: {param_count:.2f}M")
    print("=" * 50)


def calculate_and_log_target_architecture(initial_config, schedule: List[Dict[str, Any]]):
    """Calculate and log the target architecture after all scheduled operations."""
    print("\n=== ANALYZING SCALING SCHEDULE ===")
    
    # Create a copy of the initial config to simulate changes
    target_config = initial_config.__dict__.copy()
    
    print(f"Initial architecture: {target_config['n_layer']} layers, "
          f"{target_config['n_head']} heads, {target_config['n_embd']} embd, "
          f"{target_config['n_hidden']} hidden")
    
    operations_by_type = {}
    
    for i, op in enumerate(schedule):
        op_name = op['name']
        op_value = op['value']
        
        if op_name not in operations_by_type:
            operations_by_type[op_name] = []
        operations_by_type[op_name].append((i, op_value))
        
        # Simulate the operation's effect on config
        if op_name == 'stack_layers':
            target_config['n_layer'] = len(op_value)
            print(f"Operation {i+1}: stack_layers -> {target_config['n_layer']} layers")
        elif op_name == 'widen_mlp':
            target_config['n_hidden'] = op_value
            print(f"Operation {i+1}: widen_mlp -> {op_value} hidden dim")
        elif op_name == 'set_attn_lora_rank':
            target_config['attn_lora_rank'] = op_value
            print(f"Operation {i+1}: set_attn_lora_rank -> {op_value}")
        elif op_name == 'set_embedding_lora_rank':
            target_config['embedding_rank'] = op_value
            print(f"Operation {i+1}: set_embedding_lora_rank -> {op_value}")
        # Add other operations as needed
    
    print(f"\nTarget architecture: {target_config['n_layer']} layers, "
          f"{target_config['n_head']} heads, {target_config['n_embd']} embd, "
          f"{target_config['n_hidden']} hidden")
    
    print(f"Operations summary:")
    for op_type, ops in operations_by_type.items():
        print(f"  {op_type}: {len(ops)} operations")
    
    return target_config


def calculate_relative_batch_size(current_batch_size: int, scale_factor: float, 
                                master_process: bool = True) -> int:
    """
    Calculate new batch size by scaling current batch size by a factor.
    
    Args:
        current_batch_size: Current batch size
        scale_factor: Factor to scale by (0 < scale_factor <= 1)
        master_process: Whether this is the master process
    
    Returns:
        New batch size rounded down to nearest multiple of 8
    """
    if scale_factor <= 0 or scale_factor > 1:
        raise ValueError(f"scale_factor must be in range (0, 1], got {scale_factor}")
    
    # Calculate new batch size
    new_batch_size_float = current_batch_size * scale_factor
    
    # Round down to nearest multiple of 8, minimum of 8
    new_batch_size = max(8, int(new_batch_size_float // 8) * 8)
    
    if master_process:
        print(f"Scaling batch size: {current_batch_size} × {scale_factor:.3f} = {new_batch_size_float:.1f}")
        print(f"Rounded down to nearest multiple of 8: {new_batch_size}")
    
    return new_batch_size


def calculate_optimal_batch_size(model, current_batch_size: int, max_batch_size: int = 1024, 
                               target_vram_percent: float = 82.0, device_type: str = 'cuda',
                               master_process: bool = True) -> int:
    """
    Calculate optimal batch size based on current VRAM usage.
    
    Args:
        model: The model to test
        current_batch_size: Current batch size
        max_batch_size: Maximum batch size to consider
        target_vram_percent: Target VRAM utilization percentage
        device_type: Device type ('cuda' or 'cpu')
        master_process: Whether this is the master process
    
    Returns:
        Optimal batch size (multiple of 8, preferably power of 2)
    """
    if device_type != 'cuda' or not torch.cuda.is_available():
        return current_batch_size
    
    def get_vram_usage():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3     # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            # Use reserved memory as it's more accurate for actual GPU usage
            used_percent = (reserved / total) * 100
            return reserved, total, used_percent
        return 0, 0, 0
    
    # Get current VRAM state
    current_vram_used, total_vram, current_percent = get_vram_usage()
    
    if master_process:
        print(f"Current VRAM usage: {current_vram_used:.1f}/{total_vram:.1f}GB ({current_percent:.1f}%)")
        print(f"Target VRAM usage: {target_vram_percent:.1f}%")
    
    # If current usage is already optimal, keep current batch size
    if abs(current_percent - target_vram_percent) < 5.0:
        if master_process:
            print(f"Current VRAM usage is optimal, keeping batch size: {current_batch_size}")
        return current_batch_size
    
    # Estimate memory per sample (rough approximation)
    memory_per_sample = current_vram_used / current_batch_size if current_batch_size > 0 else 0.1
    target_vram_used = total_vram * (target_vram_percent / 100.0)
    estimated_optimal_batch = int(target_vram_used / memory_per_sample) if memory_per_sample > 0 else current_batch_size
    
    # Generate candidate batch sizes (multiples of 8, preferably powers of 2)
    candidates = []
    
    # Powers of 2
    power = 3  # Start from 8
    while 2**power <= max_batch_size:
        candidates.append(2**power)
        power += 1
    
    # Multiples of 8 that aren't powers of 2
    for mult in range(3, max_batch_size // 8 + 1):
        candidate = mult * 8
        if candidate <= max_batch_size and candidate not in candidates:
            candidates.append(candidate)
    
    candidates.sort()
    
    # Find the best candidate closest to our estimate
    best_batch_size = current_batch_size
    best_diff = float('inf')
    
    for candidate in candidates:
        # Prefer candidates close to our estimate
        diff = abs(candidate - estimated_optimal_batch)
        if diff < best_diff:
            best_diff = diff
            best_batch_size = candidate
    
    # Safety bounds - don't change too drastically
    min_batch_size = max(8, current_batch_size // 4)
    max_safe_batch_size = min(max_batch_size, current_batch_size * 4)
    best_batch_size = max(min_batch_size, min(best_batch_size, max_safe_batch_size))
    
    if master_process:
        estimated_vram_with_new_batch = memory_per_sample * best_batch_size
        estimated_percent = (estimated_vram_with_new_batch / total_vram) * 100
        print(f"Estimated optimal batch size: {estimated_optimal_batch}")
        print(f"Selected batch size: {best_batch_size} (estimated VRAM: {estimated_percent:.1f}%)")
    
    return best_batch_size


def execute_operation(op: Dict[str, Any], trigger_reason: str, current_val_loss: float, 
                     iter_num: int, target_architecture_config: Optional[Dict[str, Any]],
                     model, optimizer, compile_enabled: bool, ddp_enabled: bool, 
                     ddp_local_rank: int, master_process: bool, data_dir: str,
                     weight_decay: float, learning_rate: float, beta1: float, beta2: float,
                     device_type: str, training_logger, current_batch_size: int = None) -> Tuple[Any, Any]:
    """
    Execute a single training operation (architectural or hyperparameter change).
    
    Returns:
        Tuple of (updated_model, updated_optimizer)
    """
    op_name = op['name']
    op_value = op['value']
    op_label = f"{op_name}({op_value})"
    
    # Log operation start
    if master_process:
        print(f"\n{'='*60}")
        print(f"EXECUTING OPERATION: {op_label}")
        print(f"Trigger: {trigger_reason}")
        print(f"Current validation loss: {current_val_loss:.4f}")
        print(f"Target loss: {op['trigger_loss']}")
        print(f"Max wait iterations: {op['max_wait_iters']}")
        print(f"{'='*60}")
        
        training_logger.log_operation_start(iter_num, op_label, op_value, trigger_reason, 
                                          current_val_loss, op['trigger_loss'], op['max_wait_iters'])
    
    try:
        # Check if this is an architectural operation
        architectural_ops = ['stack_layers', 'widen_mlp', 'set_attn_lora_rank',
                             'set_embedding_lora_rank', 'merge_lora_weights',
                             'resize_vocabulary', 'set_embedding_finetune_mode', 'set_embedding_freeze_mode',
                             'freeze_layer', 'unfreeze_layer', 'set_layer_lora_rank']
        
        if op_name in architectural_ops:
            model, optimizer = _execute_architectural_operation(
                op_name, op_value, model, optimizer, compile_enabled, ddp_enabled,
                ddp_local_rank, master_process, data_dir, weight_decay, learning_rate,
                beta1, beta2, device_type, training_logger, iter_num
            )
        else:
            # Handle non-architectural operations
            if master_process:
                print(f"Executing hyperparameter operation: {op_name}")
            
            # Special handling for adjust_batch_size
            if op_name == 'adjust_batch_size':
                # Get unwrapped model for VRAM calculation
                if compile_enabled:
                    unwrapped_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                    if ddp_enabled:
                        unwrapped_model = unwrapped_model.module if hasattr(unwrapped_model, 'module') else unwrapped_model
                else:
                    unwrapped_model = model.module if ddp_enabled and hasattr(model, 'module') else model
                
                # Extract parameters from op_value
                if isinstance(op_value, dict):
                    batch_size_to_use = op_value.get('current_batch_size', current_batch_size or 32)
                    max_batch_size = op_value.get('max_batch_size', 1024)
                    target_vram_percent = op_value.get('target_vram_percent', 82.0)
                else:
                    # Legacy support - use current_batch_size from training loop or op_value
                    batch_size_to_use = current_batch_size or op_value or 32
                    max_batch_size = 1024
                    target_vram_percent = 82.0
                
                optimal_batch_size = calculate_optimal_batch_size(
                    unwrapped_model, batch_size_to_use, max_batch_size, 
                    target_vram_percent, device_type, master_process
                )
                
                # Return the calculated batch size as the operation value
                op['value'] = optimal_batch_size
                
                if master_process:
                    print(f"Calculated optimal batch size: {optimal_batch_size}")
                    training_logger.log_operation_success(iter_num, op_name, 
                                                        {'calculated_batch_size': optimal_batch_size,
                                                         'original_batch_size': batch_size_to_use})
            
            # Special handling for set_batch_size_relative
            elif op_name == 'set_batch_size_relative':
                batch_size_to_use = current_batch_size or 32
                scale_factor = op_value
                
                if not isinstance(scale_factor, (int, float)):
                    raise ValueError(f"set_batch_size_relative requires a numeric scale factor, got {type(scale_factor)}")
                
                new_batch_size = calculate_relative_batch_size(
                    batch_size_to_use, scale_factor, master_process
                )
                
                # Return the calculated batch size as the operation value
                op['value'] = new_batch_size
                
                if master_process:
                    print(f"Calculated relative batch size: {new_batch_size}")
                    training_logger.log_operation_success(iter_num, op_name, 
                                                        {'new_batch_size': new_batch_size,
                                                         'original_batch_size': batch_size_to_use,
                                                         'scale_factor': scale_factor})
            else:
                # These operations modify global training state and are handled by caller
                # Just log the operation
                if master_process:
                    training_logger.log_operation_success(iter_num, op_name, {'value': op_value})
        
        return model, optimizer
        
    except Exception as e:
        error_msg = f"Operation {op_label} failed: {str(e)}"
        if master_process:
            print(f"ERROR: {error_msg}")
            training_logger.log_operation_failure(iter_num, op_name, error_msg)
        raise


def _execute_architectural_operation(op_name: str, op_value: Any, model, optimizer,
                                   compile_enabled: bool, ddp_enabled: bool, ddp_local_rank: int,
                                   master_process: bool, data_dir: str, weight_decay: float,
                                   learning_rate: float, beta1: float, beta2: float,
                                   device_type: str, training_logger, iter_num: int):

    """Execute an architectural operation that requires model reconstruction."""
    if master_process:
        print(f"Performing architectural operation: {op_name}")
    
    # Reset torch compilation if enabled to avoid state issues
    if compile_enabled:
        torch._dynamo.reset()
    
    # Get unwrapped model
    if compile_enabled:
        unwrapped_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        if ddp_enabled:
            unwrapped_model = unwrapped_model.module if hasattr(unwrapped_model, 'module') else unwrapped_model
    else:
        unwrapped_model = model.module if ddp_enabled and hasattr(model, 'module') else model
    
    # Save optimizer state
    old_optimizer_state = optimizer.state_dict()
    old_param_dict = {name: p for name, p in unwrapped_model.named_parameters()}
    
    # Execute the architectural operation
    start_time = time.time()
    
    if op_name == 'stack_layers':
        unwrapped_model.stack_layers(op_value)
    elif op_name == 'widen_mlp':
        unwrapped_model.widen_mlp(op_value)
    elif op_name == 'set_attn_lora_rank':
        unwrapped_model.resize_lora_rank(op_value)
    elif op_name == 'set_embedding_lora_rank':
        unwrapped_model.resize_embedding_rank(op_value)
    elif op_name == 'merge_lora_weights':
        unwrapped_model.merge_lora_weights()
    elif op_name == 'resize_vocabulary':
        source_token_id, noise_std = op_value
        meta_path = os.path.join(data_dir, 'meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        full_vocab_size = meta['vocab_size']
        unwrapped_model.resize_vocabulary(full_vocab_size, source_token_id, noise_std)
    elif op_name == 'set_embedding_finetune_mode':
        unwrapped_model.set_embedding_finetune_mode(op_value)
    elif op_name == 'set_embedding_freeze_mode':
        unwrapped_model.set_embedding_freeze_mode(op_value)
    elif op_name == 'freeze_layer':
        unwrapped_model.freeze_layer(op_value)
    elif op_name == 'unfreeze_layer':
        unwrapped_model.unfreeze_layer(op_value)
    elif op_name == 'set_layer_lora_rank':
        layer_name, rank = op_value
        unwrapped_model.set_layer_lora_rank(layer_name, rank)
    
    # Re-create optimizer and wrappers
    log_detailed_params(unwrapped_model, master_process)
    if master_process:
        print("Re-configuring optimizer...")
    
    # Create fresh optimizer
    optimizer = unwrapped_model.configure_optimizers(weight_decay, learning_rate, 
                                                   (beta1, beta2), device_type)
    
    # Try to transfer optimizer state, but don't fail if it doesn't work
    if master_process:
        print("Transferring optimizer state...")
    try:
        transfer_optimizer_state(optimizer, old_optimizer_state, old_param_dict, unwrapped_model)
    except Exception as e:
        if master_process:
            print(f"Warning: Failed to transfer optimizer state: {e}")
            print("Starting with fresh optimizer state...")
        # Clear any partial state to ensure clean start
        optimizer.state.clear()
    
    # Rebuild model wrappers
    model = unwrapped_model
    if compile_enabled:
        if master_process:
            print("Re-compiling the model...")
        torch._dynamo.reset()
        model = torch.compile(model)
    
    if ddp_enabled:
        if master_process:
            print("Re-wrapping model in DDP...")
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # Log success
    raw_model = model.module if ddp_enabled and hasattr(model, 'module') else model
    log_model_architecture(raw_model, iter_num)
    
    if master_process:
        training_logger.log_operation_success(iter_num, op_name, 
                                            {'new_config': raw_model.config.__dict__})
    
    return model, optimizer