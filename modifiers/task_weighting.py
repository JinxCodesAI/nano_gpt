"""
Task-based weighting modifier for the diffusion loss.
Applies different weights to unmask and remask tasks.
"""

import torch


class TaskWeightingModifier:
    """
    Applies weights based on task type:
    - Unmask task: [MASK] → word (weight_unmask)
    - Remask task: word → [WRONG] (weight_remask)
    """
    def __init__(self, weight_unmask=1.0, weight_remask=1.0):
        self.weight_unmask = weight_unmask
        self.weight_remask = weight_remask

    def __call__(self, weights, context):
        flat_inputs = context['flat_inputs']
        flat_targets = context['flat_targets']
        mask_token_id = context['mask_token_id']
        wrong_token_id = context['wrong_token_id']
        
        # Task 1: Un-masking (Input was [MASK], Target is a word)
        unmask_task = (flat_inputs == mask_token_id) & (flat_targets != wrong_token_id)
        weights[unmask_task] = self.weight_unmask
        
        # Task 2: Re-masking (Input was a word, Target is [WRONG])
        remask_task = (flat_inputs != mask_token_id) & (flat_targets == wrong_token_id)
        weights[remask_task] = self.weight_remask
        
        # Store metrics in context for logging
        context['total_unmask_tasks'] = unmask_task.sum().item()
        context['total_remask_tasks'] = remask_task.sum().item()
        context['weight_unmask'] = self.weight_unmask
        context['weight_remask'] = self.weight_remask
        
        return weights, context