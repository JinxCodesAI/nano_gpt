

### **Part 1: Detailed Functional Specification**

This document outlines the functional requirements for new, advanced analysis features to be added to the model training and analysis framework. The primary objective is to provide deep, automated, and intrinsic insights into the quality and dynamics of the token embedding layer during training without interrupting the main GPU training loop.

#### **1. High-Level Goal & System Architecture**

We will implement two analysis tools to evaluate the quality of the model's "embedding layer." The embedding layer acts as the model's internal dictionary, where each word or token is represented by a high-dimensional vector. The quality of this dictionary is fundamental to the model's overall performance.

These analyses are computationally intensive. To avoid slowing down model training on the GPU, they will be executed **asynchronously on the CPU**.

The system architecture will be as follows:
1.  **Snapshot:** During a regular evaluation step, the main training script (`train.py`) will take a snapshot of the current embedding layer's weights and copy them to CPU memory.
2.  **Dispatch:** It will hand off this snapshot to a background worker thread managed by a thread pool.
3.  **Continue Training:** The main script will immediately continue the next training iteration on the GPU without waiting.
4.  **Analyze in Background:** The worker thread will execute the analysis functions (defined in `analyzer.py`) using the CPU snapshot.
5.  **Report Results:** Once the analysis is complete, the worker thread will trigger a callback function to print a formatted summary to the console and log the key metrics to the monitoring tool (Weights & Biases), tagged with the correct training step number.

#### **2. Feature 1: Semantic Drift Analysis**

*   **2.1. Purpose (The "Why"):**
    This feature answers the question: **"How much are the learned meanings of words changing as the model trains?"**
    Early in training, we expect word meanings (their vectors) to change significantly. As the model converges, these meanings should stabilize. A sudden, large change late in training could indicate learning instability. This metric provides a powerful diagnostic tool.

*   **2.2. The Technical Challenge & Solution:**
    A simple vector subtraction (`new_vector - old_vector`) is misleading. The entire dictionary of vectors can rotate in space between checkpoints without changing the relative meanings of words. To get a true measure of change, we must first mathematically rotate the old dictionary to best align with the new one. This standard alignment technique is called **Orthogonal Procrustes Analysis**.

*   **2.3. Functional Details:**
    *   **Location:** The logic will be implemented in a method named `measure_semantic_drift` within the `ModelAnalyzer` class in the `analyzer.py` file.
    *   **Inputs:**
        *   `prev_embedding_weights`: A PyTorch tensor on the CPU containing the embedding weights from a previous analysis cycle.
    *   **Output:** A dictionary containing the analysis results.
        ```json
        {
          "average_cosine_similarity": float,
          "average_euclidean_distance": float,
          "most_drifted_tokens": [ (token_id, cosine_sim), ... ],
          "least_drifted_tokens": [ (token_id, cosine_sim), ... ]
        }
        ```

#### **3. Feature 2: Embedding Geometry Analysis**

*   **3.1. Purpose (The "Why"):**
    This feature assesses the *structure* of the model's internal dictionary. A well-trained dictionary should follow a **"Galaxy Model"**:
    *   **Local Clusters (Galaxies):** Words with similar meanings (e.g., "king," "queen," "prince") should form tight clusters.
    *   **Global Sparsity (The Universe):** These clusters should be spread far apart from each other in the vast, high-dimensional space.
    A common training failure is **"representation collapse,"** where all words get crammed into one small region, making them indistinguishable. This analysis detects that failure.

*   **3.2. The Technical Challenge & Solution:**
    To check this structure, we must calculate the similarity of every word to every other word. This creates a massive matrix (`vocab_size` x `vocab_size`) that would consume too much memory if computed at once. The solution is to perform this calculation in **chunks**, processing a small number of rows at a time to keep memory usage low.

*   **3.3. Functional Details:**
    *   **Location:** The logic will be implemented in a method named `analyze_embedding_geometry` within the `ModelAnalyzer` class in `analyzer.py`.
    *   **Inputs:**
        *   `embedding_weights`: A PyTorch tensor on the CPU containing the current embedding weights to be analyzed.
    *   **Output:** A dictionary containing the geometry metrics.
        ```json
        {
          "local_density": {
            "average_neighborhood_size": float,
            "threshold": float
          },
          "global_sparsity": {
            "histogram_counts": [ int, ... ],
            "histogram_bins": [ float, ... ],
            "mean_similarity": float,
            "std_similarity": float
          }
        }
        ```

---
---

### **Part 2: Step-by-Step Implementation Plan**

This is a self-contained, developer-focused guide to implementing the features described above. It provides all necessary code and instructions for a proficient Python developer.

#### **Phase 1: Implement the Analysis Logic in `analyzer.py`**

This phase involves adding the complete code for the two new analysis methods to the `ModelAnalyzer` class.

*   **Step 1.1: Locate the File**
    *   Open the file `analyzer.py`.

*   **Step 1.2: Add New Methods to `ModelAnalyzer`**
    *   Copy and paste the following two complete methods inside the `ModelAnalyzer` class definition in `analyzer.py`.

```python
    # <<< START OF CODE TO ADD TO analyzer.py >>>

    @torch.no_grad()
    def measure_semantic_drift(self, prev_embedding_weights, top_k=50):
        """
        Measures the semantic drift of the embedding layer against a previous state
        using Orthogonal Procrustes alignment. This method is designed to run on CPU tensors.

        Args:
            prev_embedding_weights (torch.Tensor): The saved CPU embedding weight tensor
                                                   from a previous checkpoint.
            top_k (int): The number of most and least drifted tokens to report.

        Returns:
            A dictionary containing drift metrics or None if an error occurs.
        """
        try:
            # --- Step 1: Get the current embedding weights from the live model ---
            # This is the only part that touches the live model object.
            embedding_layer = self.model.transformer.wte
            if hasattr(embedding_layer, 'main_weight'):
                current_embedding_weights = embedding_layer.main_weight.weight.clone().detach()
            else:
                current_embedding_weights = embedding_layer.weight.clone().detach()

            # --- Step 2: Prepare Tensors for Analysis ---
            # Ensure both tensors are on the same device and are float32 for SVD stability.
            # The background task will pass prev_embedding_weights already on CPU.
            device = current_embedding_weights.device
            X = prev_embedding_weights.to(device, dtype=torch.float32)
            Y = current_embedding_weights.to(device, dtype=torch.float32)

            if X.shape != Y.shape:
                print(f"Warning: Cannot measure drift, embedding shapes mismatch. Prev: {X.shape}, Curr: {Y.shape}")
                return None

            # --- Step 3: Orthogonal Procrustes Alignment ---
            # This finds the optimal rotation to align the old space (X) with the new space (Y).
            # 3a: Center the matrices by subtracting the mean of all vectors.
            X_centered = X - X.mean(dim=0, keepdim=True)
            Y_centered = Y - Y.mean(dim=0, keepdim=True)
            
            # 3b: Compute the covariance matrix M = Y_centered^T @ X_centered
            M = Y_centered.T @ X_centered

            # 3c: Perform Singular Value Decomposition (SVD) on M to get its core components.
            U, _, Vh = torch.linalg.svd(M)

            # 3d: Compute the optimal rotation matrix R.
            R = Vh.T @ U.T

            # --- Step 4: Align and Compare ---
            # Apply the rotation to the original previous embedding space.
            X_aligned = X @ R

            # Calculate cosine similarity: measures change in direction (meaning).
            cosine_sims = F.cosine_similarity(X_aligned, Y, dim=1)
            
            # Calculate Euclidean distance: measures change in position in the aligned space.
            euclidean_dists = torch.linalg.norm(X_aligned - Y, dim=1)

            # --- Step 5: Aggregate and Report Results ---
            # Sort by cosine similarity to find most and least changed tokens.
            sorted_sims, indices = torch.sort(cosine_sims)

            most_drifted_tokens = [(idx.item(), sim.item()) for idx, sim in zip(indices[:top_k], sorted_sims[:top_k])]
            least_drifted_tokens = [(idx.item(), sim.item()) for idx, sim in zip(indices[-top_k:], sorted_sims[-top_k:])]

            return {
                'average_cosine_similarity': cosine_sims.mean().item(),
                'average_euclidean_distance': euclidean_dists.mean().item(),
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
        try:
            # --- Step 1: Prepare Weights ---
            # The input is already a CPU tensor, so we just ensure it's float32.
            weights = embedding_weights.clone().float()

            # L2 normalize the vectors. After this, a matrix multiplication (A @ B.T)
            # is equivalent to calculating the cosine similarity between vectors in A and B.
            weights_norm = F.normalize(weights, p=2, dim=1)
            vocab_size, _ = weights_norm.shape

            # --- Step 2: Calculate Metrics in Chunks to Conserve Memory ---
            total_neighbors = 0
            all_sim_values = []

            print(f"(Async Geo Analysis) Analyzing {vocab_size} tokens (chunk size: {chunk_size})...")
            for i in range(0, vocab_size, chunk_size):
                chunk_end = min(i + chunk_size, vocab_size)
                chunk = weights_norm[i:chunk_end, :]

                # Calculate similarity for this chunk against the whole vocabulary.
                sim_matrix_chunk = torch.matmul(chunk, weights_norm.T)

                # Avoid counting self-similarity (which is always 1.0) by setting
                # the diagonal elements to a low value before counting neighbors.
                for j in range(chunk.shape[0]):
                    global_idx = i + j
                    sim_matrix_chunk[j, global_idx] = -1.0

                # Metric 1: Count "close neighbors" for this chunk.
                total_neighbors += (sim_matrix_chunk > threshold).sum().item()

                # Metric 2: Collect all similarity values for the final histogram.
                all_sim_values.append(sim_matrix_chunk.flatten())

            # --- Step 3: Finalize and Return Results ---
            full_sim_distribution = torch.cat(all_sim_values)
            
            avg_neighborhood_size = total_neighbors / vocab_size
            hist = torch.histogram(full_sim_distribution, bins=100, range=(-0.5, 1.0))

            return {
                'local_density': {
                    'average_neighborhood_size': avg_neighborhood_size,
                    'threshold': threshold
                },
                'global_sparsity': {
                    'histogram_counts': hist.hist.tolist(),
                    'histogram_bins': hist.bin_edges.tolist(),
                    'mean_similarity': full_sim_distribution.mean().item(),
                    'std_similarity': full_sim_distribution.std().item()
                }
            }

        except Exception as e:
            print(f"Warning: Could not analyze embedding geometry. Error: {e}")
            return None

    # <<< END OF CODE TO ADD TO analyzer.py >>>
```

#### **Phase 2: Implement the Asynchronous Orchestrator in `train.py`**

This phase involves setting up the background processing framework within the main training script.

*   **Step 2.1: Add Imports and Global Initializations**
    *   At the top of `train.py`, add the import for threading:
        ```python
        import concurrent.futures
        ```
    *   Before the `while True:` training loop, initialize the thread pool executor and the state variable that will store the embedding snapshot for drift analysis.
        ```python
        # Place this code block before the "while True:" line in train.py
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        prev_embeddings = None # This will store the CPU snapshot for semantic drift
        ```

*   **Step 2.2: Define the Asynchronous Task and Callback Functions**
    *   In `train.py`, after the `get_lr` function definition and before the `while True:` loop, add the following two helper functions. They manage the background work and reporting.

    ```python
    # <<< START OF CODE TO ADD TO train.py >>>

    def run_analysis_async(analyzer, current_embeddings, prev_embeddings, iter_num):
        """
        This function is executed by the background thread. It runs all analyses
        and returns a single dictionary containing all results.
        """
        print(f"(Async Analysis @ iter {iter_num}) Starting job in background...")
        results = {'iter_num': iter_num}
        
        # Run semantic drift if we have a previous state to compare against.
        if prev_embeddings is not None:
            # We call the analyzer method on the live model instance, but pass it the CPU tensor.
            results['drift'] = analyzer.measure_semantic_drift(prev_embeddings)
        
        # Run embedding geometry analysis on the current snapshot.
        results['geometry'] = analyzer.analyze_embedding_geometry(current_embeddings)
        
        print(f"(Async Analysis @ iter {iter_num}) Job finished.")
        return results

    def analysis_done_callback(future):
        """
        This function is automatically called when the background job is done.
        It handles formatting and reporting the results.
        """
        try:
            # .result() retrieves the return value from run_analysis_async or raises an exception
            results = future.result()
            iter_num = results['iter_num']
            
            print(f"\n--- ASYNC ANALYSIS RESULTS FOR ITERATION {iter_num} ---")
            
            # Safely retrieve and report Drift Results
            drift_results = results.get('drift')
            if drift_results:
                avg_sim = drift_results['average_cosine_similarity']
                print(f"  [Drift] Average Cosine Similarity vs Prev: {avg_sim:.4f}")
                if wandb_log:
                    wandb.log({"analysis/drift_cosine_sim": avg_sim}, step=iter_num)

            # Safely retrieve and report Geometry Results
            geometry_results = results.get('geometry')
            if geometry_results:
                avg_neighbors = geometry_results['local_density']['average_neighborhood_size']
                mean_sim = geometry_results['global_sparsity']['mean_similarity']
                print(f"  [Geometry] Avg Neighbors (Galaxy Size): {avg_neighbors:.2f}")
                print(f"  [Geometry] Mean Similarity (Universe Sparsity): {mean_sim:.4f}")
                if wandb_log:
                    wandb.log({
                        "analysis/geom_avg_neighbors": avg_neighbors,
                        "analysis/geom_mean_similarity": mean_sim,
                    }, step=iter_num)

            print("--- END OF ASYNC ANALYSIS RESULTS ---\n")

        except Exception as e:
            # This will catch and print any errors that occurred in the background thread
            print(f"\n--- ERROR DURING ASYNC ANALYSIS ---\n{e}\n---------------------------------\n")

    # <<< END OF CODE TO ADD TO train.py >>>
    ```

*   **Step 2.3: Modify the Main Loop to Dispatch Analysis Jobs**
    *   Find the main training loop (`while True:`). Inside it, locate the evaluation block that starts with `if iter_num % eval_interval == 0:`.
    *   Within the `if master_process:` block of that evaluation step, after the existing code that prints validation loss, add the following logic to dispatch the analysis job.

    ```python
    # Location: In train.py, inside the `while True:` loop, inside the `if iter_num % eval_interval == 0:` block
    if master_process:
        # ... (keep the existing code for printing loss, logging to wandb, and saving checkpoints) ...
        
        # --- NEW ASYNCHRONOUS ANALYSIS DISPATCH LOGIC ---
        print(f"Dispatching async analysis for iter {iter_num}...")
        
        # 1. Take a snapshot of the current embedding weights and copy it to the CPU.
        #    This isolates the analysis from the live GPU model, preventing blocking.
        wte_layer = raw_model.transformer.wte
        current_embeddings_snapshot = (wte_layer.main_weight.weight.clone().detach().cpu()
                                       if hasattr(wte_layer, 'main_weight')
                                       else wte_layer.weight.clone().detach().cpu())

        # 2. Submit the analysis function to the background executor.
        #    The `analyzer` instance is passed to the task.
        future = executor.submit(
            run_analysis_async,
            analyzer,
            current_embeddings_snapshot,
            prev_embeddings, # Will be None on the first run; handled inside the task.
            iter_num
        )
        
        # 3. Attach the callback function that will handle reporting when the job is done.
        future.add_done_callback(analysis_done_callback)

        # 4. CRITICAL: Update the state variable with the current snapshot for the *next* analysis cycle.
        prev_embeddings = current_embeddings_snapshot
        
        # The main loop continues immediately without waiting for the analysis to finish.
        print("Async analysis job dispatched. Training continues without blocking.")
        # --- END OF NEW LOGIC ---
    ```

#### **Phase 3: Ensure Clean Shutdown of the Background Thread**

*   **Step 3.1: Add the Shutdown Command**
    *   Go to the very end of the `train.py` script, after the `while True:` loop has finished.
    *   Modify the final block to explicitly shut down the executor. This ensures the program waits for any pending analysis to complete before exiting.

    ```python
    # Location: End of train.py script
    if master_process:
        print("Training finished. Shutting down analysis executor (waiting for any pending jobs)...")
        executor.shutdown(wait=True) # wait=True is important for a clean exit
        training_logger.close()
    
    if ddp:
        destroy_process_group()
    ```