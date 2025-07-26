"""
Resume functionality for nanoGPT training.

This module handles checkpoint loading, parameter overrides, and state restoration
for resuming training from checkpoints.
"""
import os
import torch
from typing import Dict, Any, Tuple, Optional, List
from .config import TrainingConfig


def find_checkpoint_path(out_dir: str) -> str:
    """
    Find the best available checkpoint path, including emergency checkpoints.
    
    Args:
        out_dir: Output directory to search for checkpoints
        
    Returns:
        Path to the checkpoint file to use
        
    Raises:
        FileNotFoundError: If no checkpoint is found
    """
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    
    # Check for main checkpoint first
    if os.path.exists(ckpt_path):
        return ckpt_path
    
    # Look for emergency checkpoints
    emergency_files = [
        f for f in os.listdir(out_dir) 
        if f.startswith('emergency_ckpt_iter_') and f.endswith('.pt')
    ]
    
    if emergency_files:
        # Use the most recent emergency checkpoint
        emergency_files.sort(key=lambda x: int(x.split('_')[3].split('.')[0]), reverse=True)
        ckpt_path = os.path.join(out_dir, emergency_files[0])
        print(f"Main checkpoint not found. Using emergency checkpoint: {ckpt_path}")
        return ckpt_path
    
    raise FileNotFoundError(f"No checkpoint found in {out_dir}")


def load_checkpoint_with_fallback(ckpt_path: str, device: str, out_dir: str) -> Dict[str, Any]:
    """
    Load checkpoint with fallback to emergency checkpoints if main checkpoint fails.
    
    Args:
        ckpt_path: Primary checkpoint path
        device: Device to load checkpoint on
        out_dir: Output directory for emergency checkpoint search
        
    Returns:
        Loaded checkpoint dictionary
    """
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if checkpoint.get('emergency_save', False):
            print("Loaded from emergency checkpoint - continuing training from emergency save point")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint {ckpt_path}: {e}")
        
        # Try emergency checkpoints as fallback
        emergency_files = [
            f for f in os.listdir(out_dir) 
            if f.startswith('emergency_ckpt_iter_') and f.endswith('.pt')
        ]
        
        if emergency_files:
            emergency_files.sort(key=lambda x: int(x.split('_')[3].split('.')[0]), reverse=True)
            emergency_ckpt_path = os.path.join(out_dir, emergency_files[0])
            print(f"Trying emergency checkpoint: {emergency_ckpt_path}")
            checkpoint = torch.load(emergency_ckpt_path, map_location=device)
            print("Successfully loaded from emergency checkpoint")
            return checkpoint
        else:
            raise e


def apply_model_parameter_overrides(
    checkpoint_model_args: Dict[str, Any], 
    config: TrainingConfig
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply model parameter overrides from current config to checkpoint model args.
    
    Args:
        checkpoint_model_args: Model arguments from checkpoint
        config: Current training configuration
        
    Returns:
        Tuple of (updated_model_args, list_of_changes)
    """
    overrideable_params = config.get_overrideable_params()
    config_changes = []
    
    for param in overrideable_params:
        if param in checkpoint_model_args:
            checkpoint_value = checkpoint_model_args[param]
            current_value = getattr(config, param, None)
            
            # Special handling for vocab_size - if current value is None, use checkpoint value
            if param == 'vocab_size' and current_value is None:
                checkpoint_model_args[param] = checkpoint_value
            elif current_value != checkpoint_value and current_value is not None:
                config_changes.append(f"{param}: {checkpoint_value} -> {current_value}")
                print(f"Overriding {param}: {checkpoint_value} -> {current_value}")
                checkpoint_model_args[param] = current_value  # Use current config value
            else:
                # If no override specified, keep checkpoint value (already in checkpoint_model_args)
                pass
    
    # Force update of base architecture from checkpoint for parameters not overridden
    base_arch_params = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'n_hidden']
    for k in base_arch_params:
        if k in checkpoint_model_args and not any(k in change for change in config_changes):
            # Update config to match checkpoint for consistency
            setattr(config, k, checkpoint_model_args[k])
    
    return checkpoint_model_args, config_changes


def apply_smart_state_dict_loading(model, checkpoint_state_dict: Dict[str, Any]) -> None:
    """
    Apply smart state dict loading with LoRA compatibility.
    
    This handles loading standard weights into LoRA layers and removes compilation prefixes.
    
    Args:
        model: The model to load state into
        checkpoint_state_dict: State dict from checkpoint
    """
    print("Applying smart loader logic for model weights...")
    model_sd = model.state_dict()
    final_state_dict = {}

    for k, v in checkpoint_state_dict.items():
        # Case 1: The key from the checkpoint exists directly in the new model
        if k in model_sd:
            final_state_dict[k] = v
        # Case 2: We are loading a standard weight into a LoRA layer's main_weight
        else:
            lora_key_equivalent = k.replace('.weight', '.main_weight.weight')
            if lora_key_equivalent in model_sd:
                print(f"  Remapping standard weight to LoRA: {k} -> {lora_key_equivalent}")
                final_state_dict[lora_key_equivalent] = v
            else:
                print(f"  Skipping unexpected key from checkpoint: {k}")

    # Remove the compilation wrapper prefix if it exists
    unwanted_prefix = '_orig_mod.'
    for k, v in list(final_state_dict.items()):
        if k.startswith(unwanted_prefix):
            final_state_dict[k[len(unwanted_prefix):]] = final_state_dict.pop(k)

    # Load the prepared state dict.
    # strict=False is essential, as LoRA A/B weights are expected to be missing.
    model.load_state_dict(final_state_dict, strict=False)


def transfer_optimizer_state(new_optimizer, old_state_dict: Dict[str, Any], 
                           old_param_dict: Dict[str, Any], model) -> int:
    """
    Transfer optimizer state from old optimizer to new optimizer for parameters that still exist.
    Uses parameter names as the bridge instead of object identity to handle architectural changes.
    
    Args:
        new_optimizer: New optimizer instance
        old_state_dict: Old optimizer state dict
        old_param_dict: Dictionary mapping parameter names to tensors from old model
        model: Current model
        
    Returns:
        Number of parameters for which state was transferred
    """
    if 'state' not in old_state_dict:
        return 0
    
    # Map old parameter names to their state from the old optimizer state_dict
    state_to_transfer = {}
    old_param_id_to_name = {id(p): name for name, p in old_param_dict.items()}
    
    for param_id, state in old_state_dict['state'].items():
        if param_id in old_param_id_to_name:
            param_name = old_param_id_to_name[param_id]
            state_to_transfer[param_name] = state
    
    # For each parameter in the new model, find its state by name and apply it
    transferred_count = 0
    new_param_name_map = {name: p for name, p in model.named_parameters()}
    
    for param_name, state in state_to_transfer.items():
        if param_name in new_param_name_map:
            param_tensor = new_param_name_map[param_name]
            # Directly set the state in the optimizer
            new_optimizer.state[param_tensor] = state
            transferred_count += 1
    
    total_params = len(list(model.parameters()))
    print(f"Transferred optimizer state for {transferred_count} / {total_params} parameters")
    
    return transferred_count


def transfer_optimizer_state_by_shape(new_optimizer, old_state_dict: Dict[str, Any], model) -> int:
    """
    Fallback function to transfer optimizer state when parameter names are not available.
    Attempts to match parameters by shape and name similarity.
    
    Args:
        new_optimizer: New optimizer instance
        old_state_dict: Old optimizer state dict
        model: Current model
        
    Returns:
        Number of parameters for which state was transferred
    """
    if 'state' not in old_state_dict:
        return 0

    # Get current model parameters
    current_params = {name: param for name, param in model.named_parameters()}

    # This is a simplified approach - we can't perfectly reconstruct the mapping
    # without the old parameter names, but we can try some heuristics
    print("Attempting basic optimizer state transfer by parameter name matching...")
    print("Note: This may not transfer all state due to architectural changes.")

    return 0  # Placeholder - could implement more sophisticated matching


def load_training_state(checkpoint: Dict[str, Any]) -> Tuple[int, float, Dict[str, Any]]:
    """
    Load training state variables from checkpoint.
    
    Args:
        checkpoint: Checkpoint dictionary
        
    Returns:
        Tuple of (iter_num, best_val_loss, execution_state)
    """
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    
    execution_state = {}
    
    # Load execution state variables
    if 'iter_of_last_op' in checkpoint:
        execution_state['iter_of_last_op'] = checkpoint['iter_of_last_op']
        print(f"Restored iter_of_last_op: {execution_state['iter_of_last_op']}")
    
    if 'lr_schedule_offset' in checkpoint:
        execution_state['lr_schedule_offset'] = checkpoint['lr_schedule_offset']
        print(f"Restored lr_schedule_offset: {execution_state['lr_schedule_offset']}")
    
    return iter_num, best_val_loss, execution_state


def restore_scaling_schedule_state(checkpoint: Dict[str, Any], 
                                 current_schedule_file: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """
    Restore scaling schedule state from checkpoint if available.
    
    Args:
        checkpoint: Checkpoint dictionary
        current_schedule_file: Current scaling schedule file path
        
    Returns:
        Restored scaling schedule or None
    """
    if 'scaling_schedule' not in checkpoint or not checkpoint['scaling_schedule']:
        return None
    
    saved_schedule = checkpoint['scaling_schedule']
    saved_schedule_file = checkpoint.get('scaling_schedule_file')
    
    # If we have a current scaling schedule file and it matches the saved one,
    # use the saved state to preserve completion status
    if current_schedule_file and saved_schedule_file == current_schedule_file:
        print("Restoring scaling schedule state from checkpoint (preserving completion status)")
        # Also update the file to match the checkpoint state
        from .config import save_scaling_schedule
        save_scaling_schedule(current_schedule_file, saved_schedule)
        return saved_schedule
    elif saved_schedule:
        print("Warning: Checkpoint has scaling schedule but current config doesn't match.")
        print(f"  Checkpoint file: {saved_schedule_file}")
        print(f"  Current file: {current_schedule_file}")
        print("  Using file-based schedule (some completion status may be lost).")
    
    return None


def apply_training_parameter_overrides(checkpoint: Dict[str, Any], 
                                     config: TrainingConfig) -> List[str]:
    """
    Apply training parameter overrides from current config.
    
    Args:
        checkpoint: Checkpoint dictionary
        config: Current training configuration
        
    Returns:
        List of parameter changes made
    """
    if 'config' not in checkpoint:
        return []
    
    saved_config = checkpoint['config']
    training_overrides = config.get_training_overrideable_params()
    training_param_changes = []
    
    for param in training_overrides:
        if param in saved_config:
            saved_value = saved_config[param]
            current_value = getattr(config, param, None)
            
            if current_value != saved_value:
                training_param_changes.append(f"{param}: {saved_value} -> {current_value}")
                print(f"Training override {param}: {saved_value} -> {current_value}")
                # Keep current value (already in config)
    
    if training_param_changes:
        print(f"Applied {len(training_param_changes)} training parameter overrides:")
        for change in training_param_changes:
            print(f"  - {change}")
    
    return training_param_changes
