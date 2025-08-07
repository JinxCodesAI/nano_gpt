"""
loss.py
Composable loss function for the diffusion model.
Provides a modular architecture where loss modifications can be easily enabled/disabled.
"""

import torch
import torch.nn.functional as F


class DiffusionLoss:
    """
    A composable loss function for the diffusion model.
    Applies a chain of modifier objects to dynamically calculate loss weights.
    """
    def __init__(self, mask_token_id, replace_token_id):
        self.mask_token_id = mask_token_id
        self.replace_token_id = replace_token_id
        self.modifiers = []

    def add_modifier(self, modifier):
        """Registers a new modifier to be applied in the chain."""
        self.modifiers.append(modifier)

    def __call__(self, logits, targets, inputs, log_diagnostics=False):
        # Calculate the base, unweighted loss for every token
        flat_logits = logits.view(-1, logits.size(-1))
        
        # Check if we have soft labels (3D tensor) or hard labels (2D tensor)
        if targets.dim() == 3:  # Soft labels: [batch, seq, vocab_size]
            # Use KL divergence for soft labels
            flat_targets = targets.view(-1, targets.size(-1))  # [batch*seq, vocab_size]
            log_probs = F.log_softmax(flat_logits, dim=-1)
            per_token_loss = F.kl_div(log_probs, flat_targets, reduction='none').sum(dim=-1)
        else:  # Hard labels: [batch, seq]
            # Use cross-entropy for hard labels
            flat_targets = targets.view(-1)
            per_token_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1, reduction='none')

        # Initialize base weights
        weights = torch.ones_like(per_token_loss)

        # Create a context dictionary to pass information down the chain
        context = {
            'logits': logits,
            'targets': targets,
            'inputs': inputs,
            'flat_logits': flat_logits,
            'flat_targets': flat_targets,
            'flat_inputs': inputs.view(-1),
            'per_token_loss': per_token_loss,
            'mask_token_id': self.mask_token_id,
            'replace_token_id': self.replace_token_id,
            'is_soft_labels': targets.dim() == 3,
        }

        # Apply each modifier in the chain
        for modifier in self.modifiers:
            weights, context = modifier(weights, context)

        final_loss = (per_token_loss * weights).mean()

        if log_diagnostics:
            self._log_diagnostics(context)

        return final_loss

    def _log_diagnostics(self, context):
        """Simple logger that prints any metrics saved to the context dict."""
        print("-" * 20 + " LOSS DIAGNOSTICS " + "-" * 20)
        for key, value in context.items():
            if isinstance(value, (float, int)) and key.startswith(('avg_', 'total_', 'weight_', 'penalty_')):
                print(f"{key:<25}: {value:.4f}")