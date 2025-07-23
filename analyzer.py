import torch
import math
from torch.nn import functional as F
import gc

class ModelAnalyzer:
    """A helper class to compute and report advanced model metrics."""

    def __init__(self, model):
        # Handle DDP-wrapped models by accessing the underlying module
        self.model = model.module if hasattr(model, 'module') else model

    # --------------------------------------------------------------------------
    # SECTION 1: GENERIC, PRIVATE CORE ANALYSIS FUNCTIONS
    # These are the reusable building blocks for all other analyses.
    # --------------------------------------------------------------------------

    @torch.no_grad()
    def _measure_procrustes_drift(self, tensor_a, tensor_b):
        """
        Generic helper to measure drift between two tensors of the same shape
        using Orthogonal Procrustes analysis.
        """
        # Ensure float32 for stable SVD computation
        X = tensor_a.clone().to(dtype=torch.float32)
        Y = tensor_b.clone().to(dtype=torch.float32)

        # 1. Center the matrices
        X_centered = X - X.mean(dim=0, keepdim=True)
        Y_centered = Y - Y.mean(dim=0, keepdim=True)

        # 2. Compute covariance matrix and SVD
        M = Y_centered.T @ X_centered
        U, _, Vh = torch.linalg.svd(M)
        R = Vh.T @ U.T # Optimal rotation matrix

        # 3. Align and compare
        X_aligned = X @ R
        cosine_sims = F.cosine_similarity(X_aligned, Y, dim=1)

        return {
            'avg_cosine_similarity': cosine_sims.mean().item(),
            'cosine_sim_10th_percentile': torch.quantile(cosine_sims, 0.1).item(),
            'cosine_sim_90th_percentile': torch.quantile(cosine_sims, 0.9).item()
        }

    @torch.no_grad()
    def _analyze_matrix_rank_utilization(self, weight_matrix, threshold=0.98):
        """Generic helper to perform SVD rank analysis on any given weight matrix."""
        # Use .float() for stability
        svdvals = torch.linalg.svdvals(weight_matrix.float())
        total_energy = torch.sum(svdvals**2)
        if total_energy == 0:
            return {'effective_rank': 0, 'full_rank': min(weight_matrix.shape), 'utilization': 0.0}

        cumulative_energy = torch.cumsum(svdvals**2, dim=0)
        normalized_cumulative_energy = cumulative_energy / total_energy

        effective_rank = torch.searchsorted(normalized_cumulative_energy, threshold).item() + 1
        full_rank = min(weight_matrix.shape)
        utilization = (effective_rank / full_rank) if full_rank > 0 else 0.0

        return {'effective_rank': effective_rank, 'full_rank': full_rank, 'utilization': utilization}

    # --------------------------------------------------------------------------
    # SECTION 2: PUBLIC-FACING ORCHESTRATION METHODS
    # These methods use the core helpers to perform a full model analysis.
    # --------------------------------------------------------------------------

    @torch.no_grad()
    def get_model_state_snapshot(self):
        """
        Creates a comprehensive CPU snapshot of key model weights.
        Returns a dictionary mapping layer names to their CPU weight tensors.
        """
        snapshot = {}
        # 1. Token Embeddings
        wte_layer = self.model.transformer.wte
        snapshot['embeddings'] = (wte_layer.main_weight.weight.clone().detach().cpu()
                                  if hasattr(wte_layer, 'main_weight')
                                  else wte_layer.weight.clone().detach().cpu())

        # 2. Transformer Blocks (Attention and FFN layers)
        for i, block in enumerate(self.model.transformer.h):
            # Attention weights (Q, K, V combined)
            attn_layer = block.attn.c_attn
            snapshot[f'attn.{i}.c_attn.weight'] = (attn_layer.main_weight.weight.clone().detach().cpu()
                                                   if hasattr(attn_layer, 'main_weight')
                                                   else attn_layer.weight.clone().detach().cpu())
            # FFN weights
            ffn_layer = block.mlp.c_fc
            snapshot[f'ffn.{i}.c_fc.weight'] = ffn_layer.weight.clone().detach().cpu()

        return snapshot

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
    def run_full_analysis(self, current_snapshot, prev_snapshot=None, filtered_token_ids=None):
        """
        Runs a full suite of analyses on the provided model snapshots.
        This is the main entry point for the asynchronous worker.

        Args:
            current_snapshot: Current model state snapshot
            prev_snapshot: Previous model state snapshot for drift analysis
            filtered_token_ids: List of token IDs to analyze for embedding geometry (if None, analyze all)
        """
        results = {}

        # --- GEOMETRY & RANK ANALYSIS (on current snapshot) ---
        geometry_results = {}
        # 1. Embedding Galaxy Model Analysis
        if 'embeddings' in current_snapshot:
            geometry_results['embeddings'] = self._analyze_embedding_geometry(
                current_snapshot['embeddings'],
                filtered_token_ids=filtered_token_ids
            )

        # 2. FFN and Attention Rank Analysis
        ffn_ranks = {}
        attn_ranks = {}
        for i in range(self.model.config.n_layer):
            # FFN Rank
            ffn_weight = current_snapshot.get(f'ffn.{i}.c_fc.weight')
            if ffn_weight is not None:
                ffn_ranks[f'layer_{i}'] = self._analyze_matrix_rank_utilization(ffn_weight)

            # Attention Q,K,V Ranks
            qkv_weights = current_snapshot.get(f'attn.{i}.c_attn.weight')
            if qkv_weights is not None:
                q, k, v = torch.chunk(qkv_weights, 3, dim=0)
                attn_ranks[f'layer_{i}'] = {
                    'Q': self._analyze_matrix_rank_utilization(q),
                    'K': self._analyze_matrix_rank_utilization(k),
                    'V': self._analyze_matrix_rank_utilization(v)
                }
        geometry_results['ffn_ranks'] = ffn_ranks
        geometry_results['attn_ranks'] = attn_ranks
        results['geometry'] = geometry_results

        # --- DRIFT ANALYSIS (if prev_snapshot is available) ---
        if prev_snapshot:
            drift_results = {}
            for key, current_tensor in current_snapshot.items():
                prev_tensor = prev_snapshot.get(key)
                if prev_tensor is not None and prev_tensor.shape == current_tensor.shape:
                    drift_results[key] = self._measure_procrustes_drift(prev_tensor, current_tensor)
            results['drift'] = drift_results

        return results

    # --------------------------------------------------------------------------
    # SECTION 3: BUG-FIXED EMBEDDING GEOMETRY ANALYSIS
    # This version uses streaming statistics to avoid OOM crashes.
    # --------------------------------------------------------------------------

    @torch.no_grad()
    def _analyze_embedding_geometry(self, embedding_weights, threshold=0.7, chunk_size=512*16, filtered_token_ids=None):
        """
        Memory-efficient analysis of embedding geometry using streaming statistics.
        This method is now a private helper.
        Optimized for CPU to leverage PyTorch's internal parallelism for matmul
        and vectorized operations, and calculates neighbor count percentiles.

        Args:
            embedding_weights: Full embedding weight tensor
            threshold: Similarity threshold for neighbor counting
            chunk_size: Processing chunk size for memory efficiency
            filtered_token_ids: List of token IDs to analyze (if None, analyze all tokens)
        """
        try:
            weights = embedding_weights.clone().float()

            # Filter embeddings if token IDs are provided
            if filtered_token_ids is not None:
                filtered_token_ids = torch.tensor(filtered_token_ids, dtype=torch.long)
                weights = weights[filtered_token_ids]
                print(f"Analyzing {len(filtered_token_ids)} filtered embeddings out of {embedding_weights.shape[0]} total")

            weights_norm = F.normalize(weights, p=2, dim=1)
            vocab_size, _ = weights_norm.shape

            # --- Streaming statistics variables ---
            total_neighbors = 0
            running_sum = 0.0
            running_sum_sq = 0.0
            total_elements_processed = 0 # Track actual elements processed for correct mean/variance

            # Initialize a tensor to store a fixed-size sample of similarities
            max_samples = 20000
            similarity_samples_buffer = torch.zeros(max_samples, dtype=torch.float32)
            current_sample_idx = 0

            # Tensor to store the number of neighbors for each embedding vector
            # This is key for calculating neighbor count percentiles
            per_vector_neighbor_counts = torch.zeros(vocab_size, dtype=torch.int32)

            for i in range(0, vocab_size, chunk_size):
                chunk = weights_norm[i:i+chunk_size, :]
                # This matmul is the primary operation that gets parallelized internally by PyTorch on CPU
                sim_matrix_chunk = torch.matmul(chunk, weights_norm.T)

                # Vectorized operation to ignore self-similarity
                # Create indices for the diagonal within the chunk's slice of the full matrix
                diag_indices_in_chunk = torch.arange(chunk.shape[0], device=sim_matrix_chunk.device)
                global_col_indices = i + diag_indices_in_chunk
                
                # Mask out-of-bounds indices if chunk extends past vocab_size (last chunk)
                valid_diag_mask = global_col_indices < vocab_size
                
                # Apply the -1.0 to ignore self-similarity
                sim_matrix_chunk[diag_indices_in_chunk[valid_diag_mask], global_col_indices[valid_diag_mask]] = -1.0

                # --- Calculate and store per-vector neighbor counts for this chunk ---
                # Count neighbors for each row in sim_matrix_chunk
                chunk_neighbor_counts = (sim_matrix_chunk > threshold).sum(dim=1)
                # Store these counts in the pre-allocated tensor
                # .cpu() ensures it stays on CPU if sim_matrix_chunk was temporarily on GPU
                per_vector_neighbor_counts[i : i + chunk.shape[0]] = chunk_neighbor_counts.cpu()

                total_neighbors += chunk_neighbor_counts.sum().item() # Sum of all neighbors in this chunk

                # Update streaming stats for global similarity
                chunk_flat = sim_matrix_chunk.flatten()
                running_sum += chunk_flat.sum().item()
                running_sum_sq += (chunk_flat**2).sum().item()
                total_elements_processed += chunk_flat.numel() # Use processed elements for stats

                # Update samples using a reservoir-like approach for efficiency
                sample_size = min(100, chunk_flat.numel())
                if sample_size > 0:
                    # ---- START: Fast & Unique Sampling ----
                    
                    # 1. Define how much to over-sample. A 20% buffer plus a constant is very safe.
                    oversample_size = int(sample_size * 1.2) + 5

                    # 2. Generate the slightly larger sample. This is fast but may have duplicates.
                    indices = torch.randint(0, chunk_flat.numel(), (oversample_size,), device=chunk_flat.device)

                    # 3. Remove duplicates.
                    indices = torch.unique(indices)

                    # 4. Check if we have enough unique samples. If not, fall back to the guaranteed (but slow) method.
                    # This fallback will likely never be triggered, but ensures correctness.
                    if indices.numel() < sample_size:
                        indices = torch.randperm(chunk_flat.numel(), device=chunk_flat.device)[:sample_size]
                    else:
                        # 5. Trim the unique indices down to the exact sample_size we need.
                        indices = indices[:sample_size]
                    
                    # ---- END: Fast & Unique Sampling ----

                    current_chunk_samples = chunk_flat[indices]

                    for sample in current_chunk_samples:
                        if current_sample_idx < max_samples:
                            similarity_samples_buffer[current_sample_idx] = sample
                            current_sample_idx += 1
                        else:
                            # Simple replacement if buffer is full, for approximation.
                            # For true reservoir sampling, probability needs adjustment.
                            replace_idx = torch.randint(0, max_samples, (1,)).item()
                            similarity_samples_buffer[replace_idx] = sample

                del sim_matrix_chunk, chunk_flat, chunk_neighbor_counts
                torch.cuda.empty_cache() # In case it was accidentally on GPU, though moved to CPU

            # Finalize calculations
            gc.collect() # Force garbage collection

            avg_neighborhood_size = total_neighbors / vocab_size if vocab_size > 0 else 0.0
            
            mean_similarity = 0.0
            variance = 0.0
            std_similarity = 0.0

            if total_elements_processed > 0:
                mean_similarity = running_sum / total_elements_processed
                variance = (running_sum_sq / total_elements_processed) - (mean_similarity**2)
                std_similarity = math.sqrt(max(0, variance))

            sim_10th, sim_90th = 0.0, 0.0
            # Use only the filled part of the buffer for global similarity quantile calculation
            similarity_samples_final = similarity_samples_buffer[:current_sample_idx]
            if similarity_samples_final.numel() > 0:
                sim_10th = torch.quantile(similarity_samples_final, 0.1).item()
                sim_90th = torch.quantile(similarity_samples_final, 0.9).item()

            # --- Calculate percentiles for neighbor counts ---
            neighbor_10th_percentile = 0.0
            neighbor_90th_percentile = 0.0
            neighbor_99th_percentile = 0.0
            if vocab_size > 0: # Ensure there are embeddings to process
                # Ensure the tensor is float for quantile calculation, then convert back to item
                neighbor_10th_percentile = torch.quantile(per_vector_neighbor_counts.float(), 0.1).item()
                neighbor_90th_percentile = torch.quantile(per_vector_neighbor_counts.float(), 0.9).item()
                neighbor_99th_percentile = torch.quantile(per_vector_neighbor_counts.float(), 0.99).item()


            return {
                'local_density': {
                    'average_neighborhood_size': avg_neighborhood_size,
                    'neighbor_10th_percentile': neighbor_10th_percentile,
                    'neighbor_90th_percentile': neighbor_90th_percentile,
                    'neighbor_99th_percentile': neighbor_99th_percentile,
                },
                'global_sparsity': {
                    'mean_similarity': mean_similarity,
                    'std_similarity': std_similarity,
                    'similarity_10th_percentile': sim_10th,
                    'similarity_90th_percentile': sim_90th
                },
                'analysis_info': {
                    'num_embeddings_analyzed': vocab_size,
                    'total_embeddings': embedding_weights.shape[0],
                    'filtered': filtered_token_ids is not None
                }
            }
        except Exception as e:
            print(f"ERROR in _analyze_embedding_geometry: {e}")
            return None