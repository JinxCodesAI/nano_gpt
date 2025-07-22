# <<< REPLACE THE ENTIRE CONTENTS OF analyzer.py WITH THIS CODE >>>

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
            'cosine_sim_10th_percentile': torch.quantile(cosine_sims, 0.1).item()
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
    def run_full_analysis(self, current_snapshot, prev_snapshot=None):
        """
        Runs a full suite of analyses on the provided model snapshots.
        This is the main entry point for the asynchronous worker.
        """
        results = {}

        # --- GEOMETRY & RANK ANALYSIS (on current snapshot) ---
        geometry_results = {}
        # 1. Embedding Galaxy Model Analysis (the most complex)
        geometry_results['embeddings'] = self._analyze_embedding_geometry(current_snapshot['embeddings'])

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
    def _analyze_embedding_geometry(self, embedding_weights, threshold=0.7, chunk_size=512):
        """
        Memory-efficient analysis of embedding geometry using streaming statistics.
        This method is now a private helper.
        """
        try:
            weights = embedding_weights.clone().float()
            weights_norm = F.normalize(weights, p=2, dim=1)
            vocab_size, _ = weights_norm.shape

            # --- Streaming statistics variables ---
            total_neighbors = 0
            total_elements = 0
            running_sum = 0.0
            running_sum_sq = 0.0
            # Keep a small, fixed-size random sample for percentile calculation
            max_samples = 20000
            similarity_samples = []

            for i in range(0, vocab_size, chunk_size):
                chunk = weights_norm[i:i+chunk_size, :]
                sim_matrix_chunk = torch.matmul(chunk, weights_norm.T)

                for j in range(chunk.shape[0]):
                    sim_matrix_chunk[j, i+j] = -1.0 # Ignore self-similarity

                total_neighbors += (sim_matrix_chunk > threshold).sum().item()

                # Update streaming stats
                chunk_flat = sim_matrix_chunk.flatten()
                running_sum += chunk_flat.sum().item()
                running_sum_sq += (chunk_flat**2).sum().item()
                total_elements += chunk_flat.numel()

                # Update samples
                if len(similarity_samples) < max_samples:
                    sample_size = min(100, chunk_flat.numel())
                    indices = torch.randperm(chunk_flat.numel())[:sample_size]
                    similarity_samples.extend(chunk_flat[indices].tolist())

                del sim_matrix_chunk, chunk_flat

            # Finalize calculations
            gc.collect() # Force garbage collection
            avg_neighborhood_size = total_neighbors / vocab_size
            mean_similarity = running_sum / total_elements
            variance = (running_sum_sq / total_elements) - (mean_similarity**2)
            std_similarity = math.sqrt(max(0, variance))

            sim_10th, sim_90th = 0.0, 0.0
            if similarity_samples:
                similarity_samples = torch.tensor(similarity_samples)
                sim_10th = torch.quantile(similarity_samples, 0.1).item()
                sim_90th = torch.quantile(similarity_samples, 0.9).item()

            return {
                'local_density': {'average_neighborhood_size': avg_neighborhood_size},
                'global_sparsity': {
                    'mean_similarity': mean_similarity,
                    'std_similarity': std_similarity,
                    'similarity_10th_percentile': sim_10th,
                    'similarity_90th_percentile': sim_90th
                }
            }
        except Exception as e:
            print(f"ERROR in _analyze_embedding_geometry: {e}")
            return None
