# In a new file: entropy_penalty_modifier.py
# Or at the bottom of loss.py

import torch
import torch.nn.functional as F

class EntropyPenaltyModifier:
    """
    Applies a penalty to the loss for low-entropy (overconfident) wrong guesses
    during un-masking tasks.
    
    This version implements the "override and recalculate" strategy to be fully
    argmax-free and to correctly handle "good confidence".
    """
    def __init__(self, penalty_strength, vocab_size):
        if vocab_size is None:
            raise ValueError("EntropyPenaltyModifier requires vocab_size for max entropy calculation.")
        
        self.penalty_strength = penalty_strength
        self.vocab_size = vocab_size
        self.max_entropy = torch.log(torch.tensor(self.vocab_size))

    def __call__(self, weights, context):
        """
        This modifier directly changes the `weights` tensor.
        """
        # --- Step 1: Get tensors from context ---
        logits = context.get('biased_logits', context['logits'])
        targets = context['targets']
        inputs = context['inputs']
        mask_token_id = context['mask_token_id']
        wrong_token_id = context['wrong_token_id']
        
        # --- Step 2: Identify the positions where the penalty should apply ---
        unmask_task_mask = (inputs == mask_token_id) & (targets != wrong_token_id)
        if not unmask_task_mask.any():
            return weights, context

        # --- Step 3: Calculate entropy of the *incorrect* part of the distribution ---
        # 1. Create a temporary copy of the logits for manipulation
        temp_logits = logits.clone()

        # 2. For every position, find the logit of the correct target token
        # We need to handle the ignore_index (-1) for targets
        valid_targets = targets.clone()
        valid_mask = valid_targets != -1
        valid_targets[~valid_mask] = 0 # Use a dummy index for invalid targets
        
        correct_logits = temp_logits.gather(-1, valid_targets.unsqueeze(-1)).squeeze(-1)
        
        # 3. Override the correct logit with a neutral value (e.g., negative infinity to zero it out in softmax)
        # This effectively removes the "correct" answer from the probability distribution.
        temp_logits.scatter_(-1, valid_targets.unsqueeze(-1), -float('inf'))
        
        # 4. Calculate the softmax and entropy on this modified distribution
        # This entropy now measures the "peakiness" of only the incorrect tokens.
        incorrect_probs = F.softmax(temp_logits, dim=-1)
        entropy_of_incorrect = torch.distributions.Categorical(probs=incorrect_probs).entropy()
        
        # 5. Normalize the penalty
        normalized_entropy_penalty = (self.max_entropy.to(entropy_of_incorrect.device) - entropy_of_incorrect) / self.max_entropy
        
        # --- Step 4: Apply the penalty to the weights tensor (Multiplicative) ---
        penalty_map = torch.zeros_like(weights, dtype=torch.float32).view_as(unmask_task_mask)
        # Apply the penalty only at the un-masking task positions
        penalty_map[unmask_task_mask] = normalized_entropy_penalty[unmask_task_mask]
        
        # The new weight is scaled up based on how overconfident the incorrect portion of the distribution was.
        # Using the multiplicative approach you suggested.
        new_weights = weights * (1 + self.penalty_strength * penalty_map.view(-1))
        
        # --- Step 5: Log diagnostics ---
        context['avg_entropy_penalty_factor'] = penalty_map[unmask_task_mask].mean().item()
        context['penalty_entropy_strength'] = self.penalty_strength

        return new_weights, context