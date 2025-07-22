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

    @torch.no_grad()
    def measure_semantic_drift(self, current_embedding_weights, prev_embedding_weights, top_k=50):
        """
        Measures the semantic drift of the embedding layer by comparing two CPU snapshots
        using Orthogonal Procrustes alignment.

        Args:
            current_embedding_weights (torch.Tensor): The CPU snapshot of the current weights.
            prev_embedding_weights (torch.Tensor): The CPU snapshot of the previous weights.
            top_k (int): The number of most and least drifted tokens to report.

        Returns:
            A dictionary containing drift metrics or None if an error occurs.
        """
        try:
            # --- Step 1: Prepare Tensors for Analysis ---
            # The tensors are already CPU snapshots, so we just ensure they are float32.
            X = prev_embedding_weights.clone().float()  # Previous state
            Y = current_embedding_weights.clone().float()  # Current state

            if X.shape != Y.shape:
                print(f"Warning: Cannot measure drift, embedding shapes mismatch. Prev: {X.shape}, Curr: {Y.shape}")
                return None

            # --- Step 2: Orthogonal Procrustes Alignment ---
            X_centered = X - X.mean(dim=0, keepdim=True)
            Y_centered = Y - Y.mean(dim=0, keepdim=True)
            M = Y_centered.T @ X_centered
            U, _, Vh = torch.linalg.svd(M)
            R = Vh.T @ U.T

            # --- Step 3: Align and Compare ---
            X_aligned = X @ R
            cosine_sims = F.cosine_similarity(X_aligned, Y, dim=1)
            euclidean_dists = torch.linalg.norm(X_aligned - Y, dim=1)

            # --- Step 4: Aggregate and Report Results ---
            sorted_sims, indices = torch.sort(cosine_sims)
            most_drifted_tokens = [(idx.item(), sim.item()) for idx, sim in zip(indices[:top_k], sorted_sims[:top_k])]
            least_drifted_tokens = [(idx.item(), sim.item()) for idx, sim in zip(indices[-top_k:], sorted_sims[-top_k:])]

            # Calculate percentiles for more detailed statistics
            cosine_10th = torch.quantile(cosine_sims, 0.1).item()
            cosine_90th = torch.quantile(cosine_sims, 0.9).item()
            euclidean_10th = torch.quantile(euclidean_dists, 0.1).item()
            euclidean_90th = torch.quantile(euclidean_dists, 0.9).item()

            return {
                'average_cosine_similarity': cosine_sims.mean().item(),
                'cosine_similarity_10th_percentile': cosine_10th,
                'cosine_similarity_90th_percentile': cosine_90th,
                'average_euclidean_distance': euclidean_dists.mean().item(),
                'euclidean_distance_10th_percentile': euclidean_10th,
                'euclidean_distance_90th_percentile': euclidean_90th,
                'most_drifted_tokens': most_drifted_tokens,
                'least_drifted_tokens': least_drifted_tokens,
            }

        except Exception as e:
            print(f"Warning: Could not measure semantic drift. Error: {e}")
            return None

    @torch.no_grad()
    def analyze_embedding_geometry(self, embedding_weights, threshold=0.7, chunk_size=512):
        """
        Performs a fully automated analysis of the embedding space geometry
        by calculating the full pairwise similarity matrix in memory-efficient chunks.

        Args:
            embedding_weights (torch.Tensor): The CPU tensor of embedding weights to analyze.
            threshold (float): The cosine similarity value to define a "close neighbor".
            chunk_size (int): How many rows to process at a time to save memory.

        Returns:
            A dictionary containing key metrics about the embedding space or None on error.
        """
        import psutil
        import gc

        try:
            print(f"(Async Geo Analysis) Starting analysis...")

            # --- Step 1: Prepare Weights ---
            print(f"(Async Geo Analysis) Step 1: Preparing weights...")
            # The input is already a CPU tensor, so we just ensure it's float32.
            weights = embedding_weights.clone().float()
            print(f"(Async Geo Analysis) Weights shape: {weights.shape}, dtype: {weights.dtype}")

            # Check memory before normalization
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            print(f"(Async Geo Analysis) Memory before normalization: {memory_before:.1f} MB")

            # L2 normalize the vectors. After this, a matrix multiplication (A @ B.T)
            # is equivalent to calculating the cosine similarity between vectors in A and B.
            weights_norm = F.normalize(weights, p=2, dim=1)
            vocab_size, embedding_dim = weights_norm.shape

            memory_after_norm = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            print(f"(Async Geo Analysis) Memory after normalization: {memory_after_norm:.1f} MB")
            print(f"(Async Geo Analysis) Vocab size: {vocab_size}, Embedding dim: {embedding_dim}")

            # Calculate expected memory usage for similarity matrix
            expected_memory_mb = (vocab_size * vocab_size * 4) / (1024 * 1024)  # 4 bytes per float32
            print(f"(Async Geo Analysis) Expected full similarity matrix memory: {expected_memory_mb:.1f} MB")

            # If the expected memory is too large, increase chunk size or skip analysis
            if expected_memory_mb > 2000:  # More than 2GB
                print(f"(Async Geo Analysis) WARNING: Large vocabulary ({vocab_size}), adjusting chunk size...")
                chunk_size = max(64, min(chunk_size, vocab_size // 10))  # Reduce chunk size for large vocabs
                print(f"(Async Geo Analysis) Adjusted chunk size to: {chunk_size}")

            # --- Step 2: Calculate Metrics in Chunks with Streaming Statistics ---
            print(f"(Async Geo Analysis) Step 2: Processing in chunks with streaming statistics...")
            total_neighbors = 0
            processed_chunks = 0

            # Streaming statistics - compute incrementally to avoid memory accumulation
            total_elements = 0
            running_sum = 0.0
            running_sum_sq = 0.0
            similarity_samples = []  # Keep only a small sample for percentiles
            max_samples = 10000  # Limit samples to prevent memory growth
            sample_size_used = 0  # Track how many samples we actually collected

            print(f"(Async Geo Analysis) Analyzing {vocab_size} tokens (chunk size: {chunk_size})...")
            for i in range(0, vocab_size, chunk_size):
                try:
                    chunk_end = min(i + chunk_size, vocab_size)
                    chunk = weights_norm[i:chunk_end, :]

                    if processed_chunks % 10 == 0:  # Log every 10 chunks
                        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                        print(f"(Async Geo Analysis) Processing chunk {processed_chunks}, tokens {i}-{chunk_end}, memory: {current_memory:.1f} MB")

                    # Calculate similarity for this chunk against the whole vocabulary.
                    sim_matrix_chunk = torch.matmul(chunk, weights_norm.T)

                    # Avoid counting self-similarity (which is always 1.0) by setting
                    # the diagonal elements to a low value before counting neighbors.
                    for j in range(chunk.shape[0]):
                        global_idx = i + j
                        sim_matrix_chunk[j, global_idx] = -1.0

                    # Metric 1: Count "close neighbors" for this chunk.
                    neighbors_in_chunk = (sim_matrix_chunk > threshold).sum().item()
                    total_neighbors += neighbors_in_chunk

                    # Metric 2: Update streaming statistics instead of storing all values
                    chunk_flat = sim_matrix_chunk.flatten()
                    chunk_elements = chunk_flat.numel()

                    # Update running statistics
                    total_elements += chunk_elements
                    running_sum += chunk_flat.sum().item()
                    running_sum_sq += (chunk_flat ** 2).sum().item()

                    # Keep a random sample for percentile calculation (memory-bounded)
                    if len(similarity_samples) < max_samples:
                        # Take a random sample from this chunk
                        sample_size = min(100, chunk_elements)  # Max 100 samples per chunk
                        if sample_size > 0:
                            indices = torch.randperm(chunk_elements)[:sample_size]
                            samples = chunk_flat[indices].tolist()
                            similarity_samples.extend(samples)
                            sample_size_used = len(similarity_samples)

                    # Clean up chunk immediately to prevent memory accumulation
                    del sim_matrix_chunk, chunk_flat
                    processed_chunks += 1

                    # Force garbage collection every 10 chunks to keep memory stable
                    if processed_chunks % 10 == 0:
                        gc.collect()

                except Exception as chunk_error:
                    print(f"(Async Geo Analysis) ERROR in chunk {processed_chunks} (tokens {i}-{chunk_end}): {chunk_error}")
                    # Continue with next chunk instead of failing completely
                    continue

            print(f"(Async Geo Analysis) Step 3: Computing final statistics...")
            # --- Step 3: Compute Final Statistics from Streaming Data ---
            if total_elements == 0:
                print(f"(Async Geo Analysis) ERROR: No similarity values processed!")
                return None

            # Compute mean and std from streaming statistics
            mean_similarity = running_sum / total_elements
            variance = (running_sum_sq / total_elements) - (mean_similarity ** 2)
            std_similarity = math.sqrt(max(0, variance))  # Ensure non-negative

            # Compute percentiles from sample (approximate but memory-efficient)
            if similarity_samples:
                similarity_samples.sort()
                n_samples = len(similarity_samples)
                sim_10th = similarity_samples[int(0.1 * n_samples)]
                sim_90th = similarity_samples[int(0.9 * n_samples)]
            else:
                sim_10th = mean_similarity - std_similarity
                sim_90th = mean_similarity + std_similarity

            avg_neighborhood_size = total_neighbors / vocab_size

            # Create a simple histogram from samples instead of full data
            hist_counts = [0] * 100
            hist_bins = torch.linspace(-0.5, 1.0, 101).tolist()

            if similarity_samples:
                for val in similarity_samples:
                    if -0.5 <= val <= 1.0:
                        bin_idx = min(99, int((val + 0.5) / 1.5 * 100))
                        hist_counts[bin_idx] += 1

            print(f"(Async Geo Analysis) Analysis complete. Avg neighbors: {avg_neighborhood_size:.2f}")
            print(f"(Async Geo Analysis) Processed {total_elements:,} similarity values from {processed_chunks} chunks")

            # Clean up remaining tensors
            del weights_norm, weights
            if 'similarity_samples' in locals():
                del similarity_samples
            gc.collect()

            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"(Async Geo Analysis) Final memory after cleanup: {final_memory:.1f} MB")

            return {
                'local_density': {
                    'average_neighborhood_size': avg_neighborhood_size,
                    'threshold': threshold
                },
                'global_sparsity': {
                    'histogram_counts': hist_counts,
                    'histogram_bins': hist_bins,
                    'mean_similarity': mean_similarity,
                    'std_similarity': std_similarity,
                    'similarity_10th_percentile': sim_10th,
                    'similarity_90th_percentile': sim_90th,
                    'total_elements_processed': total_elements,
                    'sample_size_used': sample_size_used
                }
            }

        except Exception as e:
            print(f"(Async Geo Analysis) CRITICAL ERROR: Could not analyze embedding geometry. Error: {e}")
            print(f"(Async Geo Analysis) Error type: {type(e).__name__}")
            import traceback
            print(f"(Async Geo Analysis) Traceback: {traceback.format_exc()}")

            # Clean up any remaining tensors
            try:
                if 'weights' in locals():
                    del weights
                if 'weights_norm' in locals():
                    del weights_norm
                if 'all_sim_values' in locals():
                    del all_sim_values
                gc.collect()
            except:
                pass

            return None
