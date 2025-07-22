Of course. Here is the detailed functional specification and implementation plan for the "Shrunken Vocabulary Training" feature.

---

# Dynamic Vocabulary Resizing for Efficient Training

This document describes the "Shrunken Vocabulary Training" feature, a memory-efficient strategy implemented in the training orchestrator.

## Overview

This feature enables training to begin on a model with a physically smaller vocabulary, significantly reducing the memory footprint of the embedding and language model head layers. The orchestrator can then trigger an architectural operation to grow the model to its full vocabulary size mid-training, followed by a controlled fine-tuning phase to stabilize the newly introduced token embeddings. This entire process is designed to be transparent to the core model logic and largely automated through the scaling schedule.

## Key Features

1.  **Memory Efficiency**: Training starts with an architecturally smaller model (`shrunken_vocab_size`), providing substantial GPU memory savings.
2.  **Pipeline Integration**: The vocabulary re-mapping is handled within the `train.py` pipeline, keeping the `model.py` implementation "vanilla" and unaware of the mapping process.
3.  **Dynamic Growth**: A new `resize_vocabulary` architectural operation grows the model from its shrunken size to its full vocabulary size in a function-preserving manner.
4.  **Controlled Fine-Tuning**: A new `set_embedding_finetune_mode` operation allows for multi-stage fine-tuning by selectively freezing/unfreezing the model backbone.
5.  **Honest Validation Metrics**: A "Core Accuracy" metric is introduced to provide a true measure of model performance, complementing the standard validation loss.

## Functional Specification

### 1. Configuration (`train.py` & Config Files)

The system will be controlled by three new top-level configuration parameters:

*   `shrunken_vocab_size` (int, optional): The size of the active vocabulary to be used during the initial training phase. If `None` or not set, the feature is disabled and the model trains with `vocab_size`.
*   `vocab_remapping_file` (str, optional): The path to a `.pt` file containing the pre-computed vocabulary remapping tensor. This is **required** if `shrunken_vocab_size` is set.
*   `RARE_TOKEN_ID` (int, optional): The token ID within the shrunken vocabulary that all out-of-vocabulary tokens will be mapped to. This is **required** if `shrunken_vocab_size` is set.

### 2. Data Pipeline (`train.py`)

*   If `shrunken_vocab_size` is active, the training script will load the `remapping_vector` from the specified file and move it to the training `device`.
*   In a DDP environment, the master process will load the tensor, which will then be broadcast to all other processes.
*   Before each training step (inside the `for micro_step in ...` loop), the input `X` and target `Y` tensors fetched by `get_batch` will be re-indexed using the on-device `remapping_vector`.
*   The `get_batch` function itself will remain unchanged, always returning tensors with original, full-vocabulary token IDs.

### 3. Model Instantiation (`train.py`)

*   When `shrunken_vocab_size` is set, the `GPT` model will be instantiated with `vocab_size` set to the smaller, shrunken value. This ensures the `wte` and `lm_head` layers are physically small, achieving the desired memory savings.

### 4. New Orchestration Operations

The orchestrator will support three new operations:

*   **`resize_vocabulary`**: An architectural operation that grows the model to its full vocabulary size.
    *   **`value`**: A list `[source_token_id, noise_std]`. The `source_token_id` is the ID *within the shrunken vocabulary* (e.g., the `RARE_TOKEN_ID`) to use as a base for initializing new embeddings. `noise_std` is the standard deviation for the symmetry-breaking noise.
    *   **Action**: Replaces the `wte` and `lm_head` layers with new, full-sized versions. Copies the old embeddings and initializes the new embeddings based on the source token and noise. Forces an optimizer re-creation and state transfer.
*   **`set_embedding_finetune_mode`**: An architectural operation that controls which model parameters are trainable.
    *   **`value`**: A boolean. `True` freezes the model backbone (all layers except `wte` and `lm_head`). `False` unfreezes all trainable parameters.
    *   **Action**: Modifies a state flag in the model. Forces an optimizer re-creation to respect the new set of trainable parameters.
*   **`disable_vocab_remapping`**: A standard operation to deactivate the vocabulary mapping.
    *   **`value`**: `null` or `None`.
    *   **Action**: Sets the `use_shrunken_vocab` flag to `False`. The data pipeline will stop re-mapping `X` and `Y` tensors. This should be called immediately after `resize_vocabulary`.

### 5. Validation Metrics (`train.py`)

*   The `estimate_loss` function will be updated.
*   When vocabulary re-mapping is active, it will calculate two metrics for the validation set:
    1.  **`val_loss`**: The standard cross-entropy loss, calculated on the shrunken vocabulary space. This is used for system mechanics like checkpointing.
    2.  **`val_core_acc`**: A "Core Accuracy" metric. This measures the model's prediction accuracy *only on the tokens that were part of the original core vocabulary*, ignoring predictions for the `RARE_TOKEN_ID`. This provides an honest measure of the model's language learning progress.

## Implementation Plan

### Step 1: Pre-computation Script (`scripts/create_remapping_tensor.py`)

*   **Create a new script** that takes the full vocabulary size, core vocabulary size, a rare token ID, and an output path as arguments.
*   The script will generate a `remapping_vector` tensor of shape `(full_vocab_size,)`.
*   For each token ID `i`:
    *   If `i` is in the core vocabulary, `remapping_vector[i] = i`.
    *   If `i` is a rare token, `remapping_vector[i] = RARE_TOKEN_ID`.
*   Save the resulting tensor using `torch.save()`.

### Step 2: Update Configuration (`train.py`)

*   **In `train.py`, add the new global configuration variables**: `shrunken_vocab_size`, `vocab_remapping_file`, and `RARE_TOKEN_ID`, initializing them to `None`.
*   Ensure these new variables are added to `config_keys` so they can be overridden by `configurator.py`.

### Step 3: Update Model Instantiation Logic (`train.py`)

*   **In `train.py`, modify the "model init" block**:
    *   Create a new variable `active_vocab_size`.
    *   If `shrunken_vocab_size` is set, set `active_vocab_size = shrunken_vocab_size`.
    *   Otherwise, set `active_vocab_size = meta_vocab_size`.
    *   Pass this `active_vocab_size` to the `GPTConfig` when creating the model: `model_args['vocab_size'] = active_vocab_size`.

### Step 4: Implement Data Pipeline Mapping (`train.py`)

*   **At the top of `train.py`**, after device setup, add logic to load the `remapping_vector` if `shrunken_vocab_size` is set. Handle DDP broadcasting as discussed.
*   **In the main training loop**, modify the `for micro_step in ...` block. Add the GPU-side mapping of `X` and `Y` before the `model(X, Y)` call. This should be controlled by a new global boolean flag, e.g., `remapping_active = True`.

### Step 5: Implement New Model Methods (`model.py`)

*   **Add `resize_vocabulary(self, new_vocab_size, source_token_id, noise_std)` method to the `GPT` class**:
    *   Implement the logic to create new `wte` and `lm_head` layers.
    *   Copy old weights to the new layers.
    *   Initialize new token embeddings using the source token and added noise.
    *   Update `self.config.vocab_size` to the new full size.
*   **Add `embedding_finetune_mode` flag to `GPT.__init__`**: Initialize `self.embedding_finetune_mode = False`.
*   **Modify `GPT.configure_optimizers`**:
    *   Add a conditional block at the top: `if self.embedding_finetune_mode:`.
    *   Inside the block, filter `param_dict` to only include parameters whose names contain `'wte'` or `'lm_head'`.
    *   The rest of the function will then operate on this smaller set of parameters.

### Step 6: Implement New Operations (`train.py`)

*   **In `execute_operation`**:
    *   Add `resize_vocabulary` and `set_embedding_finetune_mode` to the `architectural_ops` list.
    *   Add `elif` blocks to handle these new operations. The `resize_vocabulary` block will call the new model method. `set_embedding_finetune_mode` will set the flag on the `unwrapped_model`.
    *   Add a new `disable_vocab_remapping` operation that sets the global `remapping_active` flag to `False`. This is a standard (non-architectural) operation.

### Step 7: Update Validation Metrics (`train.py`)

*   **Modify `estimate_loss`**:
    *   It should accept the `model` as an argument.
    *   Inside the `for split in ...` loop, initialize `core_token_correct = 0` and `core_token_total = 0`.
    *   If `remapping_active` is `True`:
        *   Remap the `Y` tensor to `remapped_Y`.
        *   Get predictions (`preds`) from the model's shrunken logits.
        *   Create a boolean mask `is_core_token = (remapped_Y != RARE_TOKEN_ID)`.
        *   Calculate `core_token_correct += ((preds == remapped_Y) & is_core_token).sum()`.
        *   Calculate `core_token_total += is_core_token.sum()`.
    *   After the loop, calculate `out[f'{split}_core_acc'] = core_token_correct / core_token_total`.
*   **Update logging calls** (`print`, `wandb.log`) to include the new `val_core_acc` metric.

This comprehensive plan covers all aspects of the feature, from data preparation to model architecture and training orchestration, while ensuring the final evaluation is robust and meaningful.