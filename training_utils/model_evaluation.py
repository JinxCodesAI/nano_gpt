"""
Model evaluation and loss estimation for diffusion training.
Contains comprehensive validation logic with detailed metrics tracking.
"""

import math
import torch

from .training_config import TrainingContext, UnmaskingStageType
from .batch_generation import get_batch


def estimate_loss(model, torch_ctx, timer, training_ctx: TrainingContext, dataset_config):
    """Estimate loss over either split using many batches"""
    out = {}
    model.eval()
    
    # Add current stage information for unmasking training
    if training_ctx.training_type == 'unmasking':
        stage_config = training_ctx.get_current_stage_config()
        if stage_config:
            out['current_stage'] = training_ctx.current_stage
            stage_type = stage_config.get_stage_type()
            out['stage_type'] = stage_type.value
            if stage_type == UnmaskingStageType.STICKY:
                config = stage_config.config
                out['target_masked_ratio'] = config.target_masked_ratio
                out['p1_probability'] = config.p1_probability
                out['p2_probability'] = config.p2_probability
            elif stage_type == UnmaskingStageType.RANDOM:
                config = stage_config.config
                out['max_masked_ratio'] = config.max_masked_ratio
            out['val_loss_stale_count'] = training_ctx.val_loss_stale_count
    
    for split in ['train', 'val']:
        losses = torch.zeros(training_ctx.eval_iters)
        # Track masked token ratios for all splits
        masked_token_ratios = []

        if split == 'val':
            # For validation, also track model vs random performance
            model_probs = []
            # Track signal to noise ratio (correct prob vs most probable incorrect prob)
            signal_to_noise_ratios = []
            # Track detailed probability breakdown for binary classification by class
            right_probs_p0 = []  # Probabilities for correct predictions where target=0
            right_probs_p1 = []  # Probabilities for correct predictions where target=1
            # Track most likely predictions for accuracy calculation
            most_likely_correct = []
            # For binary classification and remasking, track corruption statistics
            if training_ctx.training_type in ['remasking_binary', 'remasking']:
                total_positions = 0
                corrupted_positions = 0
            else:
                random_prob = 1.0 / training_ctx.extended_vocab_size  # Random chance probability

        # For unmasking, use pre-created validation set with samples from all stages
        # Track per-stage losses for detailed analysis
        stage_losses = {}
        stage_sample_counts = {}
        if split == 'val' and training_ctx.training_type == 'unmasking':
            print(f"Using validation set with samples from all {len(training_ctx.validation_stages)} stages")
            # Initialize per-stage tracking
            for stage_idx in range(len(training_ctx.validation_stages)):
                stage_losses[stage_idx] = []
                stage_sample_counts[stage_idx] = 0
        
        for k in range(training_ctx.eval_iters):
            with timer.time_function('validation_data_generation'):
                if split == 'val' and training_ctx.training_type == 'unmasking':
                    # Use pre-created validation set with batch index
                    X, Y, mask = get_batch(split, dataset_config, training_ctx.iter_num, training_ctx.batch_size, training_ctx.block_size, validation_sample_idx=k)
                    # Determine which stage this batch belongs to based on validation set structure
                    total_samples = training_ctx.eval_iters * training_ctx.batch_size
                    num_stages = len(training_ctx.validation_stages)
                    samples_per_stage = total_samples // num_stages
                    current_sample_idx = k * training_ctx.batch_size
                    current_stage_idx = min(current_sample_idx // samples_per_stage, num_stages - 1)
                elif split == 'val' and training_ctx.training_type in ['remasking_binary', 'remasking']:
                    # Fix: pass validation_sample_idx to get different validation batches
                    X, Y, mask = get_batch(split, dataset_config, training_ctx.iter_num, training_ctx.batch_size, training_ctx.block_size, validation_sample_idx=k)
                    current_stage_idx = None
                else:
                    X, Y, mask = get_batch(split, dataset_config, training_ctx.iter_num, training_ctx.batch_size, training_ctx.block_size)
                    current_stage_idx = None

            # Calculate masked token ratio for this batch
            # Get per-sample ratios to capture the full range of masking rates
            sample_ratios = mask.float().mean(dim=1)  # Shape: (batch_size,) - ratio per sample
            masked_token_ratios.extend(sample_ratios.cpu().tolist())  # Add all individual sample ratios

            with torch_ctx:
                with timer.time_function('validation_forward_pass'):
                    # This is handled in validation_loss_computation section
                    pass
                with timer.time_function('validation_loss_computation'):
                    # Optimized single forward pass for validation
                    logits, loss = model(X, Y)

                    # Apply masking for unmasking training only
                    if training_ctx.training_type == 'unmasking' and mask.any():
                        # Fast validation path - single reshape and boolean indexing
                        # Cross-entropy handles both hard targets (indices) and soft targets (probabilities)
                        logits_reshaped = logits.view(-1, logits.size(-1))
                        mask_reshaped = mask.view(-1)
                        
                        if Y.dim() == 3:
                            # Soft targets (probability distributions)
                            targets_reshaped = Y.view(-1, Y.size(-1))
                            loss = torch.nn.functional.cross_entropy(
                                logits_reshaped[mask_reshaped],
                                targets_reshaped[mask_reshaped],
                                reduction='mean'
                            )
                        else:
                            # Hard targets (token indices)
                            targets_reshaped = Y.view(-1)
                            loss = torch.nn.functional.cross_entropy(
                                logits_reshaped[mask_reshaped],
                                targets_reshaped[mask_reshaped],
                                reduction='mean'
                            )
                        
                        # Apply mask ratio weighting if enabled (same as training)
                        if training_ctx.weight_loss_by_mask_ratio:
                            mask_ratio = mask.float().mean().item()
                            if mask_ratio > 0:
                                weight = (1.0 / mask_ratio) ** 0.5  # sqrt(1.0 / mask_ratio)
                                loss = loss * weight
                    # For remasking variants, model's internal loss is correct

                # For validation, compute model vs random statistics
                if split == 'val':
                    # Get probabilities from logits and flatten for statistics
                    probs = torch.nn.functional.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
                    probs_flat = probs.view(-1, probs.size(-1))  # (batch_size * seq_len, vocab_size)
                    
                    # Handle both hard and soft targets
                    if Y.dim() == 3:
                        # Soft targets - get the most likely class from probability distribution
                        targets_flat = torch.argmax(Y.view(-1, Y.size(-1)), dim=-1)  # (batch_size * seq_len,)
                    else:
                        # Hard targets
                        targets_flat = Y.view(-1)  # (batch_size * seq_len,)
                    
                    # Calculate most likely predictions (argmax)
                    predictions = torch.argmax(probs, dim=-1)  # (batch_size, seq_len)
                    predictions_flat = predictions.view(-1)  # (batch_size * seq_len,)

                    if training_ctx.training_type == 'remasking_binary':
                        # For binary classification, compute accuracy on all positions
                        # Track corruption statistics for proper baseline
                        total_positions += targets_flat.numel()
                        corrupted_positions += (targets_flat == training_ctx.remask_wrong_id).sum().item()
                        
                        # Track validation statistics for summary
                        if split == 'val':
                            # Initialize counters on first batch
                            if k == 0:
                                val_total_class_0, val_total_class_1 = 0, 0
                                val_pred_class_0, val_pred_class_1 = 0, 0
                                val_correct_pred_0, val_correct_pred_1 = 0, 0
                            
                            # Count actual class distribution
                            class_0_count = (targets_flat == 0).sum().item()
                            class_1_count = (targets_flat == 1).sum().item()
                            val_total_class_0 += class_0_count
                            val_total_class_1 += class_1_count
                            
                            # Count predictions
                            pred_0_count = (predictions_flat == 0).sum().item()
                            pred_1_count = (predictions_flat == 1).sum().item()
                            val_pred_class_0 += pred_0_count
                            val_pred_class_1 += pred_1_count
                            
                            # Count correct predictions by class
                            correct_0 = ((predictions_flat == 0) & (targets_flat == 0)).sum().item()
                            correct_1 = ((predictions_flat == 1) & (targets_flat == 1)).sum().item()
                            val_correct_pred_0 += correct_0
                            val_correct_pred_1 += correct_1

                        # Get probabilities for correct binary classification
                        correct_token_probs = probs_flat[range(len(targets_flat)), targets_flat]
                        model_probs.extend(correct_token_probs.cpu().tolist())
                        
                        # Calculate signal to noise ratio for binary classification
                        # For each position, find the most probable incorrect class
                        incorrect_probs = torch.where(targets_flat == 0, probs_flat[:, 1], probs_flat[:, 0])
                        # Calculate signal to noise ratio, capped at 100
                        sn_ratios = torch.clamp(correct_token_probs / (incorrect_probs + 1e-10), max=100.0)
                        signal_to_noise_ratios.extend(sn_ratios.cpu().tolist())
                        
                        # Track detailed probability breakdown by class
                        class_0_mask = (targets_flat == 0)
                        class_1_mask = (targets_flat == 1)
                        
                        # Get probabilities for correct predictions by class
                        if class_0_mask.sum() > 0:
                            class_0_correct_probs = probs_flat[class_0_mask, 0]  # P(class=0) where target=0
                            right_probs_p0.extend(class_0_correct_probs.cpu().tolist())
                        
                        if class_1_mask.sum() > 0:
                            class_1_correct_probs = probs_flat[class_1_mask, 1]  # P(class=1) where target=1
                            right_probs_p1.extend(class_1_correct_probs.cpu().tolist())
                        
                        # Track most likely prediction accuracy for all positions
                        correct_predictions = (predictions_flat == targets_flat).cpu().tolist()
                        most_likely_correct.extend(correct_predictions)
                    elif training_ctx.training_type == 'remasking':
                        # For remasking, compute accuracy on ALL positions (corrupted + uncorrupted)
                        # Track corruption statistics for proper baseline
                        mask_flat = mask.view(-1)  # (batch_size * seq_len,)
                        total_positions += targets_flat.numel()
                        corrupted_positions += mask_flat.sum().item()  # mask indicates corrupted positions

                        # Get probabilities for correct predictions at ALL positions
                        correct_token_probs = probs_flat[range(len(targets_flat)), targets_flat]
                        model_probs.extend(correct_token_probs.cpu().tolist())
                        
                        # Calculate signal to noise ratio for remasking
                        # Create a copy of probabilities and zero out the correct class to find max incorrect
                        probs_masked = probs_flat.clone()
                        probs_masked[range(len(targets_flat)), targets_flat] = 0.0
                        max_incorrect_probs = torch.max(probs_masked, dim=1)[0]
                        # Calculate signal to noise ratio, capped at 100
                        sn_ratios = torch.clamp(correct_token_probs / (max_incorrect_probs + 1e-10), max=100.0)
                        signal_to_noise_ratios.extend(sn_ratios.cpu().tolist())
                        
                        # Track most likely prediction accuracy for all positions
                        correct_predictions = (predictions_flat == targets_flat).cpu().tolist()
                        most_likely_correct.extend(correct_predictions)
                    else:
                        # For unmasking, compute on masked positions only
                        mask_flat = mask.view(-1)  # (batch_size * seq_len,)
                        masked_positions = mask_flat.bool()
                        if masked_positions.sum() > 0:  # Only if there are masked tokens
                            correct_token_probs = probs_flat[masked_positions, targets_flat[masked_positions]]
                            model_probs.extend(correct_token_probs.cpu().tolist())
                            
                            # Calculate signal to noise ratio for unmasking (masked positions only)
                            masked_probs = probs_flat[masked_positions]
                            masked_targets = targets_flat[masked_positions]
                            # Create a copy and zero out correct probabilities to find max incorrect
                            probs_masked = masked_probs.clone()
                            probs_masked[range(len(masked_targets)), masked_targets] = 0.0
                            max_incorrect_probs = torch.max(probs_masked, dim=1)[0]
                            # Calculate signal to noise ratio, capped at 100
                            sn_ratios = torch.clamp(correct_token_probs / (max_incorrect_probs + 1e-10), max=100.0)
                            signal_to_noise_ratios.extend(sn_ratios.cpu().tolist())
                            
                            # Track most likely prediction accuracy for masked positions only
                            correct_predictions = (predictions_flat[masked_positions] == targets_flat[masked_positions]).cpu().tolist()
                            most_likely_correct.extend(correct_predictions)

            losses[k] = loss.item()
            
            # Track per-stage losses for unmasking validation
            if split == 'val' and training_ctx.training_type == 'unmasking' and current_stage_idx is not None:
                stage_losses[current_stage_idx].append(loss.item())
                stage_sample_counts[current_stage_idx] += X.size(0)  # Add batch size

        out[split] = losses.mean()
        
        # Add per-stage validation losses for unmasking
        if split == 'val' and training_ctx.training_type == 'unmasking' and stage_losses:
            for stage_idx, stage_loss_list in stage_losses.items():
                if stage_loss_list:  # Only if we have samples for this stage
                    avg_stage_loss = sum(stage_loss_list) / len(stage_loss_list)
                    out[f'val_stage_{stage_idx}_loss'] = avg_stage_loss
                    out[f'val_stage_{stage_idx}_samples'] = stage_sample_counts[stage_idx]
        
        if split == 'val':
            total_samples = training_ctx.eval_iters * training_ctx.batch_size
            print(f"  Validation complete: {training_ctx.eval_iters} batches processed ({total_samples} samples), avg loss = {out[split]:.4f}")
            
            # Print class distribution summary for binary classification
            if training_ctx.training_type == 'remasking_binary' and 'val_total_class_0' in locals():
                total_targets = val_total_class_0 + val_total_class_1
                total_preds = val_pred_class_0 + val_pred_class_1
                if total_targets > 0 and total_preds > 0:
                    # Class distribution
                    class_0_pct = (val_total_class_0 / total_targets) * 100
                    class_1_pct = (val_total_class_1 / total_targets) * 100
                    
                    # Prediction distribution (for display only)
                    pred_0_pct = (val_pred_class_0 / total_preds) * 100
                    pred_1_pct = (val_pred_class_1 / total_preds) * 100
                    
                    # Accuracy by class
                    acc_0 = (val_correct_pred_0 / val_total_class_0 * 100) if val_total_class_0 > 0 else 0
                    acc_1 = (val_correct_pred_1 / val_total_class_1 * 100) if val_total_class_1 > 0 else 0
                    
                    print(f"  Class distribution: no-mask {val_total_class_0} ({class_0_pct:.1f}%), mask {val_total_class_1} ({class_1_pct:.1f}%)")
                    print(f"  Model predictions: no-mask {val_pred_class_0} ({pred_0_pct:.1f}%), mask {val_pred_class_1} ({pred_1_pct:.1f}%)")
                    print(f"  Accuracy by class: no-mask {acc_0:.1f}%, mask {acc_1:.1f}%")
                    
                    # Print detailed probability breakdown if available
                    if f'{split}_avg_prob_right_p0' in out and f'{split}_avg_prob_right_p1' in out:
                        avg_p_right_p0 = out[f'{split}_avg_prob_right_p0']
                        avg_p_right_p1 = out[f'{split}_avg_prob_right_p1']
                        print(f"  Validation probabilities: avg_p_right_p0={avg_p_right_p0:.3f}, avg_p_right_p1={avg_p_right_p1:.3f}")
                    
                    # Add per-class accuracies and distributions to output for wandb logging
                    out[f'{split}_accuracy_no_mask'] = acc_0
                    out[f'{split}_accuracy_mask'] = acc_1
                    out[f'{split}_class_dist_no_mask'] = class_0_pct
                    out[f'{split}_class_dist_mask'] = class_1_pct
            
            # Print per-stage validation losses for unmasking
            if training_ctx.training_type == 'unmasking' and stage_losses:
                print("  Per-stage validation losses:")
                for stage_idx in range(len(training_ctx.validation_stages)):
                    if stage_idx in stage_losses and stage_losses[stage_idx]:
                        stage_config = training_ctx.validation_stages[stage_idx]
                        stage_type = stage_config.get_stage_type()
                        avg_loss = sum(stage_losses[stage_idx]) / len(stage_losses[stage_idx])
                        sample_count = stage_sample_counts[stage_idx]
                        
                        stage_info = f"    Stage {stage_idx} ({stage_type.value}): {avg_loss:.4f} ({sample_count} samples)"
                        if stage_type == UnmaskingStageType.STICKY:
                            config = stage_config.config
                            stage_info += f" - ratio={config.target_masked_ratio:.1f}"
                        elif stage_type == UnmaskingStageType.RANDOM:
                            config = stage_config.config
                            stage_info += f" - max_ratio={config.max_masked_ratio:.1f}"
                        print(stage_info)

        # Add masked token ratio statistics
        if masked_token_ratios:
            avg_masked_ratio = sum(masked_token_ratios) / len(masked_token_ratios)
            out[f'{split}_masked_token_ratio'] = avg_masked_ratio
            out[f'{split}_min_masked_token_ratio'] = min(masked_token_ratios)
            out[f'{split}_max_masked_token_ratio'] = max(masked_token_ratios)

        # Add model vs random comparison for validation
        if split == 'val' and model_probs:
            # Add signal to noise ratio
            if signal_to_noise_ratios:
                finite_sn_ratios = [r for r in signal_to_noise_ratios if math.isfinite(r)]
                if finite_sn_ratios:
                    avg_signal_to_noise = sum(finite_sn_ratios) / len(finite_sn_ratios)
                    out[f'{split}_signal_to_noise'] = avg_signal_to_noise
                    # Add median signal to noise ratio
                    finite_sn_ratios_sorted = sorted(finite_sn_ratios)
                    n = len(finite_sn_ratios_sorted)
                    if n % 2 == 0:
                        median_sn = (finite_sn_ratios_sorted[n//2 - 1] + finite_sn_ratios_sorted[n//2]) / 2.0
                    else:
                        median_sn = finite_sn_ratios_sorted[n//2]
                    out[f'{split}_signal_to_noise_median'] = median_sn
                else:
                    out[f'{split}_signal_to_noise'] = float('nan')
                    out[f'{split}_signal_to_noise_median'] = float('nan')
            # Calculate most likely prediction accuracy percentage
            if most_likely_correct:
                most_likely_accuracy = (sum(most_likely_correct) / len(most_likely_correct)) * 100.0
                out[f'{split}_most_likely_accuracy'] = most_likely_accuracy
            
            # VALIDATION METRICS STABILITY CHECK
            finite_probs = [p for p in model_probs if math.isfinite(p)]
            if len(finite_probs) == 0:
                print(f"\n*** VALIDATION METRICS INSTABILITY ***")
                print(f"All model probabilities are NaN/Inf (total: {len(model_probs)})")
                print(f"Sample of problematic values: {model_probs[:5]}")
                out[f'{split}_model_vs_random'] = float('nan')
                out[f'{split}_avg_correct_prob'] = float('nan')
                if training_ctx.training_type in ['remasking_binary', 'remasking']:
                    out[f'{split}_corruption_ratio'] = corrupted_positions / total_positions if total_positions > 0 else 0.0
                    out[f'{split}_random_baseline'] = 0.5  # Fallback value
            elif len(finite_probs) < len(model_probs):
                print(f"WARNING: {len(model_probs) - len(finite_probs)}/{len(model_probs)} model probabilities are NaN/Inf")
                avg_model_prob = sum(finite_probs) / len(finite_probs)
            else:
                avg_model_prob = sum(model_probs) / len(model_probs)
            
            # Only proceed if we have valid probabilities
            if len(finite_probs) > 0:
                if training_ctx.training_type == 'remasking_binary':
                    # For binary classification, compare against distribution-aware random baseline
                    corruption_ratio = corrupted_positions / total_positions if total_positions > 0 else 0.0
                    # Random classifier matching the distribution would get:
                    # P(correct) = P(guess_good) * P(actual_good) + P(guess_wrong) * P(actual_wrong)
                    # With optimal random strategy: P(guess_good) = P(actual_good), P(guess_wrong) = P(actual_wrong)
                    random_accuracy = (1 - corruption_ratio) ** 2 + corruption_ratio ** 2
                    prob_ratio = avg_model_prob / random_accuracy if random_accuracy > 0 else float('inf')
                    out[f'{split}_model_vs_random'] = prob_ratio
                    out[f'{split}_avg_correct_prob'] = avg_model_prob
                    out[f'{split}_corruption_ratio'] = corruption_ratio
                    out[f'{split}_random_baseline'] = random_accuracy
                    
                    # Calculate detailed probability breakdown by class
                    if right_probs_p0 or right_probs_p1:
                        finite_right_p0 = [p for p in right_probs_p0 if math.isfinite(p)]
                        finite_right_p1 = [p for p in right_probs_p1 if math.isfinite(p)]
                        
                        avg_p_right_p0 = sum(finite_right_p0) / len(finite_right_p0) if finite_right_p0 else 0.0
                        avg_p_right_p1 = sum(finite_right_p1) / len(finite_right_p1) if finite_right_p1 else 0.0
                        
                        out[f'{split}_avg_prob_right_p0'] = avg_p_right_p0
                        out[f'{split}_avg_prob_right_p1'] = avg_p_right_p1
                elif training_ctx.training_type == 'unmasking':
                    # For unmasking, use uniform random baseline
                    prob_ratio = avg_model_prob / random_prob
                    out[f'{split}_model_vs_random'] = prob_ratio
                    out[f'{split}_avg_correct_prob'] = avg_model_prob
                else:
                    raise ValueError(f"Unsupported training type: {training_ctx.training_type}")

    model.train()
    return out