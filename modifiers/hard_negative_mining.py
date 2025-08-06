"""
Hard negative mining modifier for the diffusion loss.
Applies higher weights to identity positions where input == target.
"""

import torch


class HardNegativeMiningModifier:
    """
    Applies higher weight to identity task positions where input == target.
    This helps prevent the model from randomly guessing on unchanged tokens.
    """
    def __init__(self, weight_identity=3.0):
        self.weight_identity = weight_identity

    def __call__(self, weights, context):
        flat_inputs = context['flat_inputs']
        flat_targets = context['flat_targets']
        mask_token_id = context['mask_token_id']
        wrong_token_id = context['wrong_token_id']
        
        # Identity task: punish guessing random things where target equals input
        identity_task = (
            (flat_inputs == flat_targets) & 
            (flat_inputs != mask_token_id) & 
            (flat_targets != wrong_token_id)
        )
        weights[identity_task] = self.weight_identity
        
        # Store metrics in context for logging
        context['total_identity_tasks'] = identity_task.sum().item()
        context['weight_identity'] = self.weight_identity
        
        return weights, context