# analyzer.py
import torch
import math
from torch.nn import functional as F

class ModelAnalyzer:
    """A helper class to compute and report advanced model metrics."""
    
    def __init__(self, model):
        # Handle DDP-wrapped models by accessing the underlying module
        self.model = model.module if hasattr(model, 'module') else model

    @torch.no_grad()
    def analyze_mlp_weight_rank(self, layer_idx=0, threshold=0.98):
        """
        Analyzes the effective rank of a specific MLP's main weight matrix using SVD.
        
        Args:
            layer_idx (int): The index of the transformer block to analyze.
            threshold (float): The percentage of variance to capture for effective rank.
        
        Returns:
            A tuple of (effective_rank, full_rank, rank_utilization).
        """
        try:
            # Access the weight of the first linear layer in the MLP block
            weight = self.model.transformer.h[layer_idx].mlp.c_fc.weight
            
            # Perform SVD. svdvals is much more efficient than full SVD.
            # Use .float() for stability, as SVD can be sensitive on bfloat16.
            svdvals = torch.linalg.svdvals(weight.float())
            
            # The "energy" of the matrix is the sum of squared singular values
            total_energy = torch.sum(svdvals**2)
            
            # Find how many singular values are needed to capture the threshold of energy
            cumulative_energy = torch.cumsum(svdvals**2, dim=0)
            normalized_cumulative_energy = cumulative_energy / total_energy
            
            # torch.searchsorted finds the index where the threshold is met or exceeded
            effective_rank = torch.searchsorted(normalized_cumulative_energy, threshold).item() + 1
            full_rank = min(weight.shape) # Rank cannot exceed the smallest dimension
            
            rank_utilization = (effective_rank / full_rank) if full_rank > 0 else 0.0
            
            return effective_rank, full_rank, rank_utilization
        except Exception as e:
            print(f"Warning: Could not analyze MLP rank. Error: {e}")
            return -1, -1, -1.0

    @torch.no_grad()
    def analyze_attention_entropy(self, X_batch):
        """
        Calculates the average entropy across all attention heads for a given batch.
        This requires the model to have been modified to return attention scores.
        
        Args:
            X_batch: A batch of input data (X) to be fed to the model.
        
        Returns:
            The average entropy value as a float.
        """
        try:
            # Set the model to evaluation mode and enable attention score return
            self.model.eval()
            # This flag tells our modified forward pass to return the attention scores
            _, _, attention_scores = self.model(X_batch, return_attention=True)
            self.model.train() # Set it back to training mode
            
            if not attention_scores:
                print("Warning: Model did not return attention scores.")
                return -1.0
            
            all_entropies = []
            # attention_scores is a list of tensors, one for each layer
            for layer_att in attention_scores:
                # layer_att shape: (B, nh, T, T)
                # Add a small epsilon for numerical stability to avoid log(0)
                att_probs = layer_att + 1e-9
                
                # Entropy = -sum(p * log(p))
                log_p = torch.log(att_probs)
                entropy = -torch.sum(att_probs * log_p, dim=-1) # Sum over the sequence length dim
                
                # Average entropy for this layer and append
                all_entropies.append(entropy.mean().item())

            return sum(all_entropies) / len(all_entropies) if all_entropies else 0.0
        except Exception as e:
            print(f"Warning: Could not analyze attention entropy. Error: {e}")
            return -1.0
