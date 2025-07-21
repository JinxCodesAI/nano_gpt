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
    def analyze_lora_update_rank(self, lora_layer, threshold=0.98):
        """
        Analyzes the effective rank of the learned LoRA update matrix (ΔW).
        This method now correctly handles both LoRALinear and LoRAEmbedding layers.
        """
        try:
            if not hasattr(lora_layer, 'lora_A') or lora_layer.lora_A is None:
                return -1, -1, -1.0 # Not a LoRA layer or rank is 0

            lora_A_w = lora_layer.lora_A.weight
            lora_B_w = lora_layer.lora_B.weight
            
            # --- THIS IS THE FIX: Handle different layer types ---
            if isinstance(lora_layer, torch.nn.modules.sparse.Embedding) or type(lora_layer).__name__ == 'LoRAEmbedding':
                # For embeddings, the update matrix is B @ A.T
                # A shape: (vocab, rank), B shape: (n_embd, rank)
                # We need B @ A.T -> (n_embd, rank) @ (rank, vocab) -> (n_embd, vocab)
                # Then we transpose the result to match the main weight's (vocab, n_embd) shape.
                learned_update_matrix = (lora_B_w @ lora_A_w.T).T
            else: # Assumes LoRALinear
                # For linear layers, the update matrix is B @ A
                # A shape: (rank, in_features), B shape: (out_features, rank)
                # B @ A -> (out_features, rank) @ (rank, in_features) -> (out_features, in_features)
                learned_update_matrix = lora_B_w @ lora_A_w
            # --- END FIX ---
            
            # Now, analyze this newly constructed matrix using our helper
            full_rank = lora_layer.rank
            eff_rank, _, _ = self._analyze_weight_matrix_rank(learned_update_matrix, threshold)
            
            rank_utilization = (eff_rank / full_rank) if full_rank > 0 else 0.0
            
            return eff_rank, full_rank, rank_utilization

        except Exception as e:
            # Add the layer type to the error for better debugging
            print(f"Warning: Could not analyze LoRA update rank for layer of type {type(lora_layer).__name__}. Error: {e}")
            return -1, -1, -1.0

    # --- ADD THIS PRIVATE HELPER METHOD ---
    @torch.no_grad()
    def _analyze_weight_matrix_rank(self, weight, threshold=0.98):
        """
        Private helper to perform SVD analysis on any given weight matrix.
        """
        # Use .float() for stability, as SVD can be sensitive on half-precision floats.
        svdvals = torch.linalg.svdvals(weight.float())
        
        # The "energy" of the matrix is the sum of squared singular values
        total_energy = torch.sum(svdvals**2)
        if total_energy == 0:
            return 0, min(weight.shape), 0.0
            
        # Find how many singular values are needed to capture the threshold of energy
        cumulative_energy = torch.cumsum(svdvals**2, dim=0)
        normalized_cumulative_energy = cumulative_energy / total_energy
        
        effective_rank = torch.searchsorted(normalized_cumulative_energy, threshold).item() + 1
        full_rank = min(weight.shape) # Rank cannot exceed the smallest dimension
        
        rank_utilization = (effective_rank / full_rank) if full_rank > 0 else 0.0
        
        return effective_rank, full_rank, rank_utilization

    # --- REFACTOR THE EXISTING MLP ANALYSIS FUNCTION ---
    
    @torch.no_grad()
    def analyze_mlp_weight_rank(self, layer_idx=0, threshold=0.98):
        """Analyzes the effective rank of a specific MLP's main weight matrix."""
        try:
            weight = self.model.transformer.h[layer_idx].mlp.c_fc.weight
            return self._analyze_weight_matrix_rank(weight, threshold)
        except Exception as e:
            print(f"Warning: Could not analyze MLP rank. Error: {e}")
            return -1, -1, -1.0

    @torch.no_grad()
    def analyze_attention_weight_rank(self, layer_idx=0, threshold=0.98):
        """
        Analyzes the effective rank of an Attention projection's Q, K, and V
        matrices separately and returns their results.
        """
        try:
            attn_layer = self.model.transformer.h[layer_idx].attn.c_attn
            weight = attn_layer.main_weight.weight if hasattr(attn_layer, 'main_weight') else attn_layer.weight
            
            # The weights for Q, K, and V are concatenated along dimension 0
            # weight shape: (3 * n_embd, n_embd) -> Transposed from nn.Linear
            # nn.Linear weight shape is (out_features, in_features)
            # So, c_attn weight is (2304, 768)
            q_weights, k_weights, v_weights = torch.chunk(weight, 3, dim=0)
            
            q_results = self._analyze_weight_matrix_rank(q_weights, threshold)
            k_results = self._analyze_weight_matrix_rank(k_weights, threshold)
            v_results = self._analyze_weight_matrix_rank(v_weights, threshold)
            
            # Returns a dictionary for clear interpretation
            return {
                'Q': {'eff_rank': q_results[0], 'full_rank': q_results[1], 'util': q_results[2]},
                'K': {'eff_rank': k_results[0], 'full_rank': k_results[1], 'util': k_results[2]},
                'V': {'eff_rank': v_results[0], 'full_rank': v_results[1], 'util': v_results[2]},
            }
        except Exception as e:
            print(f"Warning: Could not analyze Attention rank. Error: {e}")
            return None
            
    # --- ADD THE NEW EMBEDDING ANALYSIS FUNCTION ---
    @torch.no_grad()
    def analyze_embedding_rank(self, threshold=0.98):
        """Analyzes the effective rank of the token embedding matrix."""
        try:
            embedding_layer = self.model.transformer.wte
            # Handle both standard nn.Embedding and our LoRAEmbedding wrapper
            weight = embedding_layer.main_weight.weight if hasattr(embedding_layer, 'main_weight') else embedding_layer.weight
            return self._analyze_weight_matrix_rank(weight, threshold)
        except Exception as e:
            print(f"Warning: Could not analyze Embedding rank. Error: {e}")
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
