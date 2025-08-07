# entropy_penalty_modifier.py

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
        replace_token_id = context['replace_token_id']
        
        # --- Step 2: Identify the positions where the penalty should apply ---
        # The penalty should only apply to the un-masking task.
        # Note: When targets are soft, they are FloatTensors.
        is_hard_label = targets.dtype == torch.long
        if is_hard_label:
            unmask_task_mask = (inputs == mask_token_id) & (targets != replace_token_id)
        else:
            # For soft labels, an un-masking task is where the input is a mask.
            # We assume all mask inputs are un-masking tasks.
            unmask_task_mask = (inputs == mask_token_id)

        if not unmask_task_mask.any():
            return weights, context

        # --- Step 3: Calculate entropy of the *incorrect* part of the distribution ---
        
        # 1. Get the model's output probability distribution
        probs = F.softmax(logits, dim=-1)
        
        # 2. Get the target probability distribution
        if is_hard_label:
            target_probs = F.one_hot(targets, num_classes=self.vocab_size).float()
        else:
            target_probs = targets # Targets are already a probability distribution

        # 3. Calculate the "incorrect" probability distribution
        # This is the part of the model's confidence that is NOT on the target.
        # We use ReLU to zero out any negative values that might occur if P_model < P_target.
        incorrect_distribution = F.relu(probs - target_probs)
        
        # 4. Re-normalize this distribution so it sums to 1, making it a valid input for entropy.
        # Add a small epsilon to prevent division by zero if the distributions are identical.
        epsilon = 1e-9
        incorrect_distribution = incorrect_distribution / (incorrect_distribution.sum(dim=-1, keepdim=True) + epsilon)
        
        # 5. Calculate the entropy of this "incorrect" distribution.
        # This entropy measures the "peakiness" of the model's mistakes.
        entropy_of_incorrect = torch.distributions.Categorical(probs=incorrect_distribution).entropy()
        
        # 6. Normalize the penalty
        normalized_entropy_penalty = (self.max_entropy.to(entropy_of_incorrect.device) - entropy_of_incorrect) / self.max_entropy
        
        # --- Step 4: Apply the penalty to the weights tensor (Multiplicative) ---
        penalty_map = torch.zeros_like(weights, dtype=torch.float32).view_as(unmask_task_mask)
        # Apply the penalty only at the un-masking task positions
        penalty_map[unmask_task_mask] = normalized_entropy_penalty[unmask_task_mask]
        
        # The new weight is scaled up based on how overconfident the incorrect portion of the distribution was.
        new_weights = weights * (1 + self.penalty_strength * penalty_map.view(-1))
        
        # --- Step 5: Log diagnostics ---
        context['avg_entropy_penalty_factor'] = penalty_map[unmask_task_mask].mean().item()
        context['penalty_entropy_strength'] = self.penalty_strength

        return new_weights, context