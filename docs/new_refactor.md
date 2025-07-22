Of course. Based on your request, I have prepared a complete, self-contained guide to refactor your analysis pipeline.

This document is designed for a proficient Python developer who is not an expert in machine learning. It includes a full analysis of the current code, identifies a critical memory bug, proposes a more robust and generic architecture, and provides the complete, final code for all necessary changes.

***

### **Part 1: Analysis, Bug Fixes, and Refactoring Specification**

This document details the plan to evolve the analysis toolkit from an embedding-specific tool into a generic, robust, and memory-efficient framework for analyzing the entire model.

#### **1. Analysis of the Current Implementation & Bug Identification**

A review of the provided code reveals two critical issues that prevent its use in a real-world, long-running training job.

*   **Critical Bug: Memory Leak & Out-of-Memory (OOM) Crash:**
    *   **Problem:** The `analyze_embedding_geometry` function, while correctly processing in chunks, aggregates all similarity values into a single list (`all_sim_values`) and then concatenates them (`torch.cat`). For a standard 50,257-token vocabulary, the resulting similarity matrix (`50257 x 50257`) requires approximately **10 GB of RAM**. This will inevitably cause the Python process to crash from an OOM error on most systems.
    *   **Solution:** We must refactor this function to use **streaming statistics**. Instead of storing all `2.5 billion` similarity values, we will compute the mean, standard deviation, and an approximate percentile distribution on the fly, using a constant, small amount of memory.

*   **Architectural Limitation: Lack of Genericity:**
    *   **Problem:** The current analysis functions are hardcoded to work only on the embedding layer (`wte`). The logic for drift and geometry is broadly applicable to other layers (FFN, Attention), but the current implementation doesn't allow for this.
    *   **Solution:** We will refactor the `ModelAnalyzer` to use generic, private helper methods for the core analysis logic (drift, rank). Public-facing methods will then use these helpers to analyze specific parts of the model, creating a clean, powerful, and extensible API.

#### **2. Proposed Architecture for Refactored `ModelAnalyzer`**

We will restructure `ModelAnalyzer` into a layered API with generic core components.

*   **2.1. Private Core Methods (The "How"):**
    *   `_measure_procrustes_drift(tensor_a, tensor_b)`: A generic function that performs Orthogonal Procrustes alignment and comparison on any two provided weight tensors of the same shape.
    *   `_analyze_matrix_rank_utilization(matrix)`: A generic function that computes the effective rank and utilization of any given weight matrix.

*   **2.2. Public Analysis Methods (The "What"):**
    *   These methods will be responsible for orchestrating a full analysis of the model.
    *   **`get_model_state_snapshot()`**: A new helper method that iterates through the live model and creates a comprehensive CPU snapshot. This snapshot will be a dictionary mapping layer names to their weight tensors (e.g., `{'ffn.0.weight': tensor, 'attn.0.q_weight': tensor, ...}`). This isolates all CPU/snapshot logic into one place.
    *   **`run_full_analysis(current_snapshot, prev_snapshot)`**: The primary new public method. It will take two snapshots as input and perform a complete analysis:
        *   It will calculate the **drift** for every corresponding weight matrix between the two snapshots.
        *   It will calculate the **rank utilization** for key matrices in the `current_snapshot`.
        *   It will run the memory-efficient **embedding geometry** analysis on the embedding tensor.
        *   It will return a single, structured dictionary containing all these results.

#### **3. Refactoring the Asynchronous Workflow in `train.py`**

*   **3.1. Snapshotting:** The main loop will now call `analyzer.get_model_state_snapshot()` to create a full dictionary snapshot, not just the embedding tensor.
*   **3.2. Asynchronous Task:** The `run_analysis_async` function will be simplified. Its only job will be to call `analyzer.run_full_analysis()` with the provided snapshots.
*   **3.3. Reporting:** The `analysis_done_callback` function will be rewritten to intelligently parse the new, rich, nested dictionary of results and print a comprehensive, formatted report to the console and `wandb`.
*   **3.4. Robustness:** We will add system memory checks using `psutil` before dispatching the analysis to prevent starting a memory-intensive task when the system is already under pressure.

---
---

### **Part 2: Step-by-Step Implementation Plan**

This is a complete, self-contained guide to refactor your codebase.

#### **Phase 1: Fully Replace `analyzer.py`**

The changes required are significant. To ensure correctness and fix the memory bug, **replace the entire content of your existing `analyzer.py` file** with the following code. It contains the new generic architecture, the bug fix for geometry analysis, and improved reporting.

```python
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

```

#### **Phase 2: Refactor `train.py` for Generic, Robust Analysis**

This phase updates the training script to use the new, powerful analyzer and adds robustness checks.

*   **Step 2.1: Add `psutil` Import**
    *   At the top of `train.py`, add an import for checking system resources. You may need to install it (`pip install psutil`).
        ```python
        import psutil
        ```
*   **Step 2.2: Replace Asynchronous Task and Callback Functions**
    *   In `train.py`, **delete your existing `run_analysis_async` and `analysis_done_callback` functions.**
    *   **Replace them** with the following new versions, which are designed to work with the refactored analyzer and its rich output dictionary.

    ```python
    # <<< REPLACE THE OLD ASYNC HELPERS in train.py WITH THIS CODE >>>

    def run_full_analysis_async(analyzer, current_snapshot, prev_snapshot, iter_num):
        """
        The new async task function. It calls the main analysis method of the analyzer.
        """
        print(f"(Async Analysis @ iter {iter_num}) Starting full model analysis job...")
        results = analyzer.run_full_analysis(current_snapshot, prev_snapshot)
        results['iter_num'] = iter_num # Tag results with the iteration number
        print(f"(Async Analysis @ iter {iter_num}) Job finished.")
        return results

    def analysis_done_callback(future):
        """
        The new callback, rewritten to handle the rich, nested results dictionary.
        """
        try:
            results = future.result()
            iter_num = results['iter_num']
            
            print(f"\n--- ASYNC ANALYSIS RESULTS FOR ITERATION {iter_num} ---")
            
            # --- Report Geometry & Rank Results ---
            if 'geometry' in results:
                geo = results['geometry']
                if 'embeddings' in geo and geo['embeddings']:
                    emb_geo = geo['embeddings']
                    avg_neighbors = emb_geo['local_density']['average_neighborhood_size']
                    mean_sim = emb_geo['global_sparsity']['mean_similarity']
                    print(f"  [Embeddings Geometry] Avg Neighbors: {avg_neighbors:.2f} | Mean Similarity: {mean_sim:.4f}")
                    if wandb_log:
                        wandb.log({
                            "analysis/embed/geom_avg_neighbors": avg_neighbors,
                            "analysis/embed/geom_mean_sim": mean_sim
                        }, step=iter_num)
                
                if 'ffn_ranks' in geo:
                    # Report rank for the first, middle, and last FFN layers for brevity
                    num_layers = len(geo['ffn_ranks'])
                    layers_to_log = {0, num_layers // 2, num_layers - 1}
                    for i in layers_to_log:
                        rank_info = geo['ffn_ranks'].get(f'layer_{i}')
                        if rank_info:
                            util = rank_info['utilization']
                            print(f"  [FFN Rank L{i}] Utilization: {util:.2%}")
                            if wandb_log: wandb.log({f"analysis/ffn/rank_util_L{i}": util}, step=iter_num)

            # --- Report Drift Results ---
            if 'drift' in results:
                drift = results['drift']
                if 'embeddings' in drift and drift['embeddings']:
                    emb_drift = drift['embeddings']['avg_cosine_similarity']
                    print(f"  [Embeddings Drift] Avg Cosine Sim: {emb_drift:.4f}")
                    if wandb_log: wandb.log({"analysis/embed/drift_avg_sim": emb_drift}, step=iter_num)

                # Report drift for the first FFN layer for brevity
                ffn0_drift = drift.get('ffn.0.c_fc.weight')
                if ffn0_drift:
                    ffn_drift_sim = ffn0_drift['avg_cosine_similarity']
                    print(f"  [FFN L0 Drift] Avg Cosine Sim: {ffn_drift_sim:.4f}")
                    if wandb_log: wandb.log({"analysis/ffn/drift_avg_sim_L0": ffn_drift_sim}, step=iter_num)

            print("--- END OF ASYNC ANALYSIS RESULTS ---\n")

        except Exception as e:
            print(f"\n--- ERROR in analysis_done_callback: {e} ---\n")
            import traceback
            traceback.print_exc()

    # <<< END OF REPLACEMENT CODE FOR train.py >>>
    ```

*   **Step 2.3: Update the Main Loop Dispatch Logic**
    *   In the main `while True:` loop, find the block where you dispatch the analysis job (`if iter_num % eval_interval == 0 and master_process:`).
    *   **Replace your entire existing dispatch logic block** with this new, more robust version.

    ```python
    # <<< REPLACE THE DISPATCH LOGIC in train.py WITH THIS CODE >>>

    # --- NEW ROBUST ASYNCHRONOUS ANALYSIS DISPATCH LOGIC ---
    # Check system memory before dispatching analysis to prevent OOM
    memory_info = psutil.virtual_memory()
    if memory_info.percent > 90.0:
        print(f"WARNING: Skipping async analysis due to high system memory usage ({memory_info.percent:.1f}%)")
    else:
        print(f"Dispatching async analysis for iter {iter_num}...")
        try:
            # 1. Create a full snapshot of the model state on CPU.
            current_snapshot = analyzer.get_model_state_snapshot()

            # 2. Submit the new, generic analysis task to the executor.
            future = executor.submit(
                run_full_analysis_async,
                analyzer,
                current_snapshot,
                prev_embeddings, # Will be None on the first run.
                iter_num
            )
            future.add_done_callback(analysis_done_callback)

            # 4. CRITICAL: Update state for the next analysis cycle.
            prev_embeddings = current_snapshot
            print("Async analysis job dispatched. Training continues.")

        except Exception as dispatch_error:
            print(f"ERROR dispatching async analysis for iter {iter_num}: {dispatch_error}")

    # --- END OF NEW LOGIC ---
    ```

#### **Phase 3: Final Shutdown**

*   **Step 3.1: Verify Clean Shutdown**
    *   Ensure the code at the very end of `train.py` correctly shuts down the executor. This code should already be present from the previous version and remains correct:
    ```python
    if master_process:
        print("Training finished. Shutting down analysis executor (waiting for any pending jobs)...")
        executor.shutdown(wait=True)
        # ... rest of cleanup
    ```

This completes the refactoring. Your framework now has a powerful, generic, and memory-safe analysis capability that can provide deep insights into the behavior of all major components of your model as it trains.