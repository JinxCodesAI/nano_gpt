"""
Model evaluation and analysis utilities for nanoGPT training.
"""
import os
import torch
import numpy as np
import concurrent.futures
from typing import Dict, Any, List, Optional, Callable, Tuple


def run_full_analysis_async(analyzer, current_snapshot, prev_snapshot, val_batch, 
                          iter_num: int, filtered_token_ids: Optional[List[int]] = None):
    """Run full model analysis asynchronously."""
    def analysis_task():
        try:
            # Run the analysis
            results = analyzer.run_full_analysis(
                current_snapshot=current_snapshot,
                prev_snapshot=prev_snapshot,
                val_batch=val_batch,
                filtered_token_ids=filtered_token_ids
            )
            return {
                'success': True,
                'results': results,
                'iter_num': iter_num
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'iter_num': iter_num
            }
    
    # Submit task to thread pool
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(analysis_task)
    return future


def analysis_done_callback(future, training_logger, master_process: bool = True):
    """Callback function for when analysis is complete."""
    try:
        result = future.result()
        if result['success']:
            if master_process:
                print(f"Analysis completed for iteration {result['iter_num']}")
                # Log analysis results
                training_logger.log_analysis_results(result['iter_num'], result['results'])
        else:
            if master_process:
                print(f"Analysis failed for iteration {result['iter_num']}: {result['error']}")
                training_logger.log(f"Analysis error: {result['error']}")
    except Exception as e:
        if master_process:
            print(f"Error in analysis callback: {e}")


def get_val_batch(data_dir: str, block_size: int, batch_size: int, device: str,
                 shrunken_vocab_size: Optional[int] = None, 
                 vocab_remapping: Optional[torch.Tensor] = None,
                 rare_token_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a validation batch."""
    val_data_path = os.path.join(data_dir, 'val.bin')
    
    if not os.path.exists(val_data_path):
        raise FileNotFoundError(f"Validation data not found: {val_data_path}")
    
    # Load validation data
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
    
    # Sample random positions for the batch
    total_length = len(val_data)
    max_start_pos = total_length - block_size
    
    if max_start_pos <= 0:
        raise ValueError(f"Validation data too small. Need at least {block_size} tokens.")
    
    # Generate random starting positions
    start_positions = np.random.randint(0, max_start_pos, batch_size)
    
    # Extract sequences
    sequences = []
    for start_pos in start_positions:
        sequence = val_data[start_pos:start_pos + block_size]
        sequences.append(sequence)
    
    # Convert to tensor
    data = torch.tensor(np.array(sequences), dtype=torch.long)
    
    # Apply vocabulary remapping if using shrunken vocabulary
    if shrunken_vocab_size is not None and vocab_remapping is not None:
        data = _apply_vocab_remapping(data, vocab_remapping, rare_token_id)
    
    # Split into input and target
    x = data[:, :-1].contiguous()  # Input sequences
    y = data[:, 1:].contiguous()   # Target sequences
    
    # Move to device
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    
    return x, y


def _apply_vocab_remapping(data: torch.Tensor, vocab_remapping: torch.Tensor, 
                          rare_token_id: int) -> torch.Tensor:
    """Apply vocabulary remapping for shrunken vocabulary."""
    # Create mask for valid tokens
    valid_mask = data < len(vocab_remapping)
    
    # Apply remapping
    remapped_data = torch.where(
        valid_mask,
        vocab_remapping[data.clamp(0, len(vocab_remapping) - 1)],
        rare_token_id
    )
    
    return remapped_data


def create_model_snapshot(model, include_gradients: bool = False) -> Dict[str, Any]:
    """Create a snapshot of model state for analysis."""
    snapshot = {
        'config': model.config.__dict__.copy(),
        'parameters': {},
        'parameter_stats': {}
    }
    
    # Capture parameter values and statistics
    for name, param in model.named_parameters():
        if param.data is not None:
            # Store parameter data
            snapshot['parameters'][name] = param.data.clone().detach()
            
            # Calculate statistics
            param_flat = param.data.flatten()
            snapshot['parameter_stats'][name] = {
                'mean': param_flat.mean().item(),
                'std': param_flat.std().item(),
                'min': param_flat.min().item(),
                'max': param_flat.max().item(),
                'norm': param.norm().item(),
                'shape': list(param.shape),
                'numel': param.numel(),
                'requires_grad': param.requires_grad
            }
            
            # Include gradients if requested and available
            if include_gradients and param.grad is not None:
                grad_flat = param.grad.flatten()
                snapshot['parameter_stats'][name]['grad_mean'] = grad_flat.mean().item()
                snapshot['parameter_stats'][name]['grad_std'] = grad_flat.std().item()
                snapshot['parameter_stats'][name]['grad_norm'] = param.grad.norm().item()
    
    return snapshot


def analyze_parameter_changes(current_snapshot: Dict[str, Any], 
                            prev_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze changes in model parameters between snapshots."""
    if prev_snapshot is None:
        return {'message': 'No previous snapshot available for comparison'}
    
    changes = {
        'parameter_changes': {},
        'summary': {
            'total_parameters_changed': 0,
            'max_change': 0.0,
            'avg_change': 0.0,
            'parameters_with_large_changes': []
        }
    }
    
    current_params = current_snapshot['parameters']
    prev_params = prev_snapshot['parameters']
    
    total_change = 0.0
    num_changed = 0
    large_change_threshold = 0.1
    
    for name in current_params:
        if name in prev_params:
            current_param = current_params[name]
            prev_param = prev_params[name]
            
            if current_param.shape == prev_param.shape:
                # Calculate change
                diff = current_param - prev_param
                change_norm = diff.norm().item()
                relative_change = change_norm / (prev_param.norm().item() + 1e-8)
                
                changes['parameter_changes'][name] = {
                    'absolute_change': change_norm,
                    'relative_change': relative_change,
                    'max_abs_change': diff.abs().max().item()
                }
                
                total_change += change_norm
                num_changed += 1
                
                # Track large changes
                if relative_change > large_change_threshold:
                    changes['summary']['parameters_with_large_changes'].append({
                        'name': name,
                        'relative_change': relative_change
                    })
                
                # Update max change
                if change_norm > changes['summary']['max_change']:
                    changes['summary']['max_change'] = change_norm
    
    # Calculate summary statistics
    if num_changed > 0:
        changes['summary']['total_parameters_changed'] = num_changed
        changes['summary']['avg_change'] = total_change / num_changed
    
    return changes


def evaluate_model_performance(model, val_batch_fn: Callable, num_batches: int = 10,
                             device_type: str = 'cuda', dtype: str = 'bfloat16') -> Dict[str, float]:
    """Evaluate model performance on validation data."""
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    perplexities = []
    
    # Setup autocast context
    if device_type == 'cpu':
        ctx = torch.no_grad()
    else:
        autocast_dtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float16
        ctx = torch.amp.autocast(device_type=device_type, dtype=autocast_dtype)
    
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = val_batch_fn()
            
            with ctx:
                logits, loss = model(x, y)
            
            batch_loss = loss.item()
            batch_tokens = y.numel()
            
            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens
            
            # Calculate perplexity for this batch
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)
    
    model.train()
    
    # Calculate metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    avg_perplexity = np.mean(perplexities)
    
    return {
        'loss': avg_loss,
        'perplexity': avg_perplexity,
        'num_batches': num_batches,
        'total_tokens': total_tokens
    }


def check_model_health(model) -> Dict[str, Any]:
    """Check model health by analyzing parameters and gradients."""
    health_report = {
        'healthy': True,
        'warnings': [],
        'errors': [],
        'statistics': {}
    }
    
    param_stats = []
    grad_stats = []
    
    for name, param in model.named_parameters():
        if param.data is not None:
            # Check for NaN or Inf in parameters
            if torch.isnan(param.data).any():
                health_report['errors'].append(f"NaN found in parameter {name}")
                health_report['healthy'] = False
            
            if torch.isinf(param.data).any():
                health_report['errors'].append(f"Inf found in parameter {name}")
                health_report['healthy'] = False
            
            # Collect parameter statistics
            param_norm = param.data.norm().item()
            param_stats.append(param_norm)
            
            # Check gradients
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    health_report['errors'].append(f"NaN found in gradient of {name}")
                    health_report['healthy'] = False
                
                if torch.isinf(param.grad).any():
                    health_report['errors'].append(f"Inf found in gradient of {name}")
                    health_report['healthy'] = False
                
                grad_norm = param.grad.norm().item()
                grad_stats.append(grad_norm)
                
                # Check for very large gradients
                if grad_norm > 100.0:
                    health_report['warnings'].append(f"Large gradient norm in {name}: {grad_norm:.2f}")
            elif param.requires_grad:
                health_report['warnings'].append(f"No gradient computed for trainable parameter {name}")
    
    # Calculate overall statistics
    if param_stats:
        health_report['statistics']['param_norm_mean'] = np.mean(param_stats)
        health_report['statistics']['param_norm_std'] = np.std(param_stats)
        health_report['statistics']['param_norm_max'] = np.max(param_stats)
    
    if grad_stats:
        health_report['statistics']['grad_norm_mean'] = np.mean(grad_stats)
        health_report['statistics']['grad_norm_std'] = np.std(grad_stats)
        health_report['statistics']['grad_norm_max'] = np.max(grad_stats)
    
    return health_report