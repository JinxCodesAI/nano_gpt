
# Batch Creation Improvements

### **Part 1: Detailed Functional Specification**

This document specifies the design for a high-performance, stateful `BatchManager` that intelligently serves training batches to a model according to a dynamic curriculum, optimizing for both training stability and data fidelity.

#### **1. High-Level Goal & System Architecture**

The goal is to replace the naive data loader with a sophisticated manager that implements a **curriculum learning** strategy for token presentation. This is achieved by actively managing a large buffer of candidate batches and serving the "best" one at each step, where "best" is defined by a dynamically changing target distribution.

The new architecture will be asynchronous and highly efficient:
1.  **Approximated Corpus Analysis:** On the first run, the manager will quickly estimate the corpus's natural token distribution by sampling a large, configurable number of tokens from across all data shards. This result is cached for all subsequent runs.
2.  **Persistent Candidate Buffer:** The manager will maintain a large, persistent buffer of candidate batches in memory.
3.  **Asynchronous Buffer Management:** A dedicated background thread will be responsible for continuously:
    *   **Refilling:** Reading new data chunks sequentially from the dataset.
    *   **Scoring:** Evaluating new candidates against the current target distribution.
    *   **Culling:** Removing the lowest-scoring candidates to maintain a buffer of high-quality "novel" batches.
4.  **Batch Retrieval (`get_next_batch`):** This call will become extremely fast. It will simply pop the highest-scoring batch from the pre-managed buffer.
5.  **Curriculum Control:** The target distribution will be controlled by an `alpha` parameter, which blends a uniform distribution (for stability) with the corpus distribution (for fidelity). The annealing of this `alpha` will be managed via explicit operations in the main training script's schedule.

#### **2. Core Components of `BatchManager`**

*   **2.1. `__init__(...)` - Initialization:**
    *   Takes a new parameter `starting_estimation_token_count` to control the size of the initial corpus analysis.
    *   Initializes state variables: `corpus_distribution`, `served_token_counts`, etc.
    *   Initializes a large `candidate_buffer` (e.g., storing 1000-2000 batches).
    *   Starts a **dedicated background thread** for managing the candidate buffer.

*   **2.2. `_get_corpus_distribution()` - Fast Corpus Analysis:**
    *   Replaces the full scan.
    *   Calculates how many tokens to read from each shard based on `starting_estimation_token_count`.
    *   Reads the prefix from each shard, aggregates the token counts, and computes the distribution.
    *   Caches the result to a file for instant loading on future runs.

*   **2.3. `_buffer_management_worker()` - The Asynchronous Heart:**
    *   This function runs continuously in a background thread.
    *   It contains a deterministic double loop to iterate through every shard file exactly once per "epoch."
    *   **Refill Logic:** It reads data sequentially, creates new candidate batches, and adds them to a temporary queue.
    *   **Score & Cull Logic:** Periodically, it will lock the main candidate buffer, add the new batches from its queue, re-score the *entire buffer* (this is efficient as it only happens when the target distribution changes), sort the buffer by score, and trim it down to its maximum size by removing the worst candidates.

*   **2.4. `update_target_distribution(alpha)` - Curriculum Control:**
    *   This is the public method called by the main training script.
    *   It recomputes the `target_distribution` based on the new `alpha`.
    *   **Crucially, it will signal the background thread** via a `threading.Event` to trigger a re-scoring of the entire candidate buffer, as the definition of a "good" batch has now changed.

*   **2.5. `get_next_batch()` - Fast Batch Retrieval:**
    *   This method becomes very lightweight.
    *   It will wait briefly if the buffer is empty.
    *   It will acquire a lock, `pop` the highest-scoring batch from the buffer (which is kept sorted by the worker), release the lock, and return the batch.
    *   It also updates the `served_token_counts` state.

*   **2.6. `shutdown()` - Graceful Exit:**
    *   A new method to signal the background worker thread to terminate cleanly and wait for it to finish.

---
---

### **Part 2: Step-by-Step Implementation Plan**

This is a complete, self-contained guide to replacing your current `BatchManager` with the new high-performance version.

#### **Phase 1: Implement the High-Performance `BatchManager`**

This phase involves creating the new class with its asynchronous worker and optimized logic. You will replace your entire existing `BatchManager` class with this new code.

*   **Step 1.1: New Imports for `train.py`**
    *   At the top of your `train.py` script (or in a new `data_loader.py` file), you will need these imports.
        ```python
        import torch
        import numpy as np
        import os
        import random
        import threading
        import time
        from collections import deque
        ```

*   **Step 1.2: The Complete `BatchManager` V2 Code**
    *   **Delete your entire existing `BatchManager` class.**
    *   **Replace it** with the following complete, self-contained implementation.

```python
# <<< PASTE THIS ENTIRE CLASS into train.py (replacing the old one) or a new data_loader.py file >>>

class BatchManager:
    def __init__(self, data_dir, shard_filenames, vocab_size, batch_size, block_size, device, device_type,
                 starting_estimation_token_count=100_000_000, buffer_size=2000):
        print("Initializing High-Performance BatchManager (V2)...")
        self.data_dir = data_dir
        self.shard_filenames = shard_filenames
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.device_type = device_type
        self.buffer_size = buffer_size

        # 1. Approximate or load the corpus token distribution
        self.corpus_distribution = self._get_corpus_distribution(starting_estimation_token_count)
        self.uniform_distribution = torch.ones(self.vocab_size, dtype=torch.float32) / self.vocab_size

        # 2. Initialize state for tracking served tokens
        self.served_token_counts = torch.zeros(self.vocab_size, dtype=torch.float64)
        self.total_tokens_served = 0

        # 3. Thread-safe candidate buffer and control variables for the worker
        self.candidate_buffer = deque()
        self.buffer_lock = threading.Lock()
        self.rescore_event = threading.Event()
        self.shutdown_event = threading.Event()

        # 4. Initialize the curriculum and target distribution
        self.alpha = 1.0
        self.target_distribution = self.uniform_distribution.clone()
        
        # 5. Start the background worker thread
        self.worker_thread = threading.Thread(target=self._buffer_management_worker, daemon=True)
        self.worker_thread.start()
        print("BatchManager initialized and background worker started.")

    def _get_corpus_distribution(self, estimation_tokens):
        """Calculates an approximate token distribution from a sample of the dataset and caches it."""
        cache_path = os.path.join(self.data_dir, f'corpus_dist_approx_{estimation_tokens}.pt')
        if os.path.exists(cache_path):
            print(f"Loading cached approximate corpus distribution from {cache_path}")
            return torch.load(cache_path)

        print(f"Approximating corpus distribution from {estimation_tokens:,} tokens...")
        total_counts = torch.zeros(self.vocab_size, dtype=torch.int64)
        tokens_per_shard = estimation_tokens // len(self.shard_filenames)

        for shard_name in self.shard_filenames:
            shard_path = os.path.join(self.data_dir, shard_name)
            data = np.memmap(shard_path, dtype=np.uint16, mode='r')
            if len(data) > tokens_per_shard:
                sample = data[:tokens_per_shard]
                shard_counts = torch.from_numpy(np.bincount(sample, minlength=self.vocab_size))
                total_counts += shard_counts
        
        distribution = total_counts.float() / total_counts.sum()
        print(f"Saving approximate corpus distribution to {cache_path}")
        torch.save(distribution, cache_path)
        return distribution

    def _buffer_management_worker(self):
        """
        Runs in a background thread to continuously read, score, and manage the candidate buffer.
        """
        shard_cycle = iter(self.shard_filenames * 1000) # Loop over the dataset many times
        
        while not self.shutdown_event.is_set():
            # --- Phase 1: Refill the buffer if it has space ---
            if len(self.candidate_buffer) < self.buffer_size:
                try:
                    shard_name = next(shard_cycle)
                    shard_path = os.path.join(self.data_dir, shard_name)
                    data = np.memmap(shard_path, dtype=np.uint16, mode='r')

                    # Create multiple batches from a large sequential chunk for I/O efficiency
                    num_batches_to_create = 50
                    chunk_size = num_batches_to_create * self.batch_size * self.block_size
                    start_idx = random.randint(0, max(0, len(data) - chunk_size))
                    chunk = data[start_idx : start_idx + chunk_size]
                    
                    new_candidates = []
                    # Create non-overlapping batches from the chunk
                    for i in range(0, len(chunk), self.batch_size * self.block_size):
                        if i + self.batch_size * self.block_size + 1 > len(chunk): continue
                        x = torch.from_numpy(chunk[i : i + self.batch_size * self.block_size].astype(np.int64)).view(self.batch_size, self.block_size)
                        y = torch.from_numpy(chunk[i+1 : i+1 + self.batch_size * self.block_size].astype(np.int64)).view(self.batch_size, self.block_size)
                        new_candidates.append({'x': x, 'y': y, 'score': -1.0})
                    
                    with self.buffer_lock:
                        self.candidate_buffer.extend(new_candidates)

                except StopIteration:
                    print("Worker has finished all shard cycles.")
                    break
                except Exception as e:
                    print(f"Error in buffer refill worker: {e}")
                    time.sleep(1)

            # --- Phase 2: Re-score and sort the entire buffer if signaled ---
            if self.rescore_event.is_set():
                with self.buffer_lock:
                    print("(Async Worker) Re-scoring candidate buffer...")
                    served_dist = (self.served_token_counts / (self.total_tokens_served + 1e-9)).to(torch.float32)
                    
                    temp_list = list(self.candidate_buffer)
                    for batch_data in temp_list:
                        tokens, counts = torch.unique(batch_data['x'], return_counts=True)
                        neglect_score = self.target_distribution[tokens] / (served_dist[tokens] + 1e-9)
                        batch_data['score'] = (neglect_score * counts).sum().item()
                    
                    # Sort the buffer by score (highest first) and trim excess
                    temp_list.sort(key=lambda b: b['score'], reverse=True)
                    self.candidate_buffer = deque(temp_list[:self.buffer_size])
                    
                    self.rescore_event.clear() # Mark re-scoring as done
                    print(f"(Async Worker) Buffer re-scored. Size: {len(self.candidate_buffer)}")

            time.sleep(0.1) # Prevent busy-looping, yield to other threads

    def update_target_distribution(self, alpha):
        """Updates the target distribution and signals the worker to re-score."""
        print(f"Updating batch manager alpha to {alpha:.3f}")
        self.alpha = alpha
        # Blend corpus and uniform distributions
        self.target_distribution = (1 - alpha) * self.corpus_distribution + alpha * self.uniform_distribution
        self.rescore_event.set() # Signal the worker thread to re-score all candidates

    def get_next_batch(self):
        """Waits for and retrieves the highest-scoring batch from the buffer."""
        # Wait for the buffer to be populated, especially at the start
        while not self.candidate_buffer:
            print("Main thread is waiting for the batch buffer to fill...")
            time.sleep(0.5)
            if self.shutdown_event.is_set(): return None, None

        with self.buffer_lock:
            # The buffer is kept sorted by the worker, so the best is always at the front
            best_batch_data = self.candidate_buffer.popleft()

        best_x, best_y = best_batch_data['x'], best_batch_data['y']

        # Update the state of served tokens
        unique_tokens, counts = torch.unique(best_x, return_counts=True)
        self.served_token_counts[unique_tokens] += counts.to(self.served_token_counts.dtype)
        self.total_tokens_served += best_x.numel()

        # Move the chosen batch to the correct GPU/CPU device
        if self.device_type == 'cuda':
            best_x = best_x.pin_memory().to(self.device, non_blocking=True)
            best_y = best_y.pin_memory().to(self.device, non_blocking=True)
        else:
            best_x, best_y = best_x.to(self.device), best_y.to(self.device)
            
        return best_x, best_y

    def shutdown(self):
        """Signals the background worker to stop and waits for it to exit."""
        print("Shutting down BatchManager background worker...")
        self.shutdown_event.set()
        self.worker_thread.join(timeout=5)
        print("BatchManager shut down.")

```

#### **Phase 2: Integrate `BatchManager` V2 into `train.py`**

This involves instantiating the new manager and updating the training loop to use it correctly.

*   **Step 2.1: Instantiate the New Manager**
    *   In `train.py`, before the main `while True:` loop, **delete the entire old `get_batch` function.**
    *   In its place, create an instance of your new `BatchManager`. Make sure to pass the new `starting_estimation_token_count` parameter.

    ```python
    # <<< DELETE THE OLD get_batch FUNCTION in train.py >>>

    # <<< ADD THIS CODE BEFORE THE `while True:` LOOP >>>
    
    # Initialize the BatchManager for the training set
    batch_manager = BatchManager(
        data_dir=data_dir,
        shard_filenames=train_shard_filenames,
        vocab_size=meta_vocab_size, # This must be loaded from meta.pkl
        batch_size=batch_size,
        block_size=block_size,
        device=device,
        device_type=device_type,
        starting_estimation_token_count=100_000_000 # ~100M tokens for approximation
    )

    # We still need a simple way to get validation batches
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    def get_val_batch():
        # ... (this helper function remains the same as in the previous version)
    ```

*   **Step 2.2: Implement the `set_curriculum_alpha` Operation**
    *   Find your `execute_operation` function in `train.py`.
    *   Add a new case to the `if/elif` block for non-architectural operations. This is the explicit hook for controlling the curriculum.

    ```python
    # <<< ADD THIS `elif` BLOCK TO THE `execute_operation` FUNCTION in train.py >>>
    
    # --- Inside the `else` block for non-architectural operations ---
    # ... (after other cases like 'set_lr') ...
    elif op_name == 'set_curriculum_alpha':
        if master_process: print(f"Setting curriculum alpha: {op_value}")
        # This calls the method in our new BatchManager to update the target and trigger a re-score
        batch_manager.update_target_distribution(op_value)
    ```

*   **Step 2.3: Use the New Operation in Your Scaling Schedule**
    *   You now control the curriculum from your `scaling_schedule.yml` file. This makes your experiments reproducible and easy to configure. A typical schedule would look like this:

    ```yaml
    # In your scaling_schedule.yml
    - name: set_curriculum_alpha
      value: 1.0 # Start with fully uniform
      desc: "Initialize with uniform target distribution"
      trigger_loss: 99.0 # Trigger immediately at the start
      max_wait_iters: 1
      reevaluate: false

    - name: set_curriculum_alpha
      value: 0.5
      desc: "Start annealing towards natural distribution"
      trigger_loss: 4.5 # Example loss threshold
      max_wait_iters: 20000
      reevaluate: false
    
    - name: set_curriculum_alpha
      value: 0.0
      desc: "Switch to fully natural distribution"
      trigger_loss: 3.5
      max_wait_iters: 40000
      reevaluate: false
    ```

*   **Step 2.4: Update Batch Retrieval Calls in the Main Loop**
    *   Ensure all calls to `get_batch('train')` are replaced with `batch_manager.get_next_batch()`. The calls inside `estimate_loss` and the main training step should be updated as specified in the previous version.

*   **Step 2.5: Ensure Clean Shutdown at the End of Training**
    *   Go to the very end of `train.py`, after the `while True:` loop.
    *   Add the call to `batch_manager.shutdown()` to ensure the background thread is properly closed.

    ```python
    # <<< AT THE VERY END OF train.py >>>
    if master_process:
        batch_manager.shutdown() # Add this line to stop the background worker
        print("Training finished. Shutting down analysis executor...")
        executor.shutdown(wait=True)
        training_logger.close()
    
    if ddp:
        destroy_process_group()
    ```