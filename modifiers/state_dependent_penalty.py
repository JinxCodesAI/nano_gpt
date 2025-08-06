"""
State-dependent penalty modifier for the diffusion loss.
Applies dynamic penalties for destructive edits based on corruption rate.
"""

import torch


class StateDependentPenaltyModifier:
    """
    Applies state-dependent penalty for destructive edits.
    Reduces penalty when corruption rate is high (more errors to fix).
    Increases penalty when corruption rate is low (fewer errors, don't create new ones).
    """
    def __init__(self, penalty_strength=0.5):
        self.penalty_strength = penalty_strength

    def __call__(self, weights, context):
        logits = context['logits']
        inputs = context['inputs']
        targets = context['targets']
        mask_token_id = context['mask_token_id']
        wrong_token_id = context['wrong_token_id']
        
        with torch.no_grad():
            B, T = inputs.shape
            predictions_2d = torch.argmax(logits, dim=-1)
            
            total_destructive_positions = 0
            total_penalty_applied = 0.0
            
            for i in range(B):
                epsilon = 1e-6
                
                # Calculate corruption rate for this sequence
                input_words_mask = (inputs[i] != mask_token_id)
                num_input_words = input_words_mask.sum().item()
                
                if num_input_words > 0:
                    corrupted_words_mask = (targets[i, input_words_mask] == wrong_token_id)
                    num_corrupted_words = corrupted_words_mask.sum().item()
                    corruption_rate = num_corrupted_words / (num_input_words + epsilon)
                    
                    # Effective penalty: lower when corruption rate is high
                    effective_penalty = self.penalty_strength * (1.0 - corruption_rate)
                    
                    # Find positions where a correct word was wrongly changed to [WRONG]
                    destructive_mask = (
                        (inputs[i] == targets[i]) & 
                        (inputs[i] != mask_token_id) & 
                        (predictions_2d[i] == wrong_token_id)
                    )
                    
                    # Apply penalty to weights
                    weights_2d = weights.view(B, T)
                    weights_2d[i, destructive_mask] = effective_penalty
                    
                    # Track metrics
                    destructive_positions = destructive_mask.sum().item()
                    total_destructive_positions += destructive_positions
                    total_penalty_applied += effective_penalty * destructive_positions
        
        # Store metrics in context for logging
        context['total_destructive_positions'] = total_destructive_positions
        context['avg_penalty_applied'] = (
            total_penalty_applied / max(total_destructive_positions, 1)
        )
        context['penalty_strength'] = self.penalty_strength
        
        return weights, context