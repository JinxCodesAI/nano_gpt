# Development Plan: Aligning `train_refactored.py` with `train.py`

## 1. Introduction

**Objective:** This document outlines the necessary steps to resolve the functional discrepancies between the original `train.py` and the non-functional `train_refactored.py`. The goal is to restore the correct, intended behavior of the training script while preserving the improved modular architecture of the refactored version.

**Audience:** This plan is intended for a proficient Python developer who has access to the codebase but may not be familiar with its intricacies. Each task includes a reference to the problem described in `discrepancies_report.md` and a clear, actionable solution.

**Strategy:** The fixes are ordered from most critical (bugs causing incorrect results) to minor functional alignments. Each task is designed to be a self-contained unit of work.

---

## 2. Task Breakdown

### Task 1: Correct Evaluation Loss Reporting

**Problem Reference:** `discrepancies_report.md`, Section 3.1. The script incorrectly logs the validation loss as the training loss.

*   **Sub-task 1.1: Modify `estimate_loss` Function**
    *   **File:** `training/evaluation.py`
    *   **Action:** The current `estimate_loss` function in the refactored code likely only computes and returns the validation loss. Modify it to calculate loss for both `'train'` and `'val'` splits, just as it is done in the original `train.py`.
    *   **Guidance:**
        1.  The function should accept the `model`, a data-fetching function, evaluation iterations, and other necessary parameters.
        2.  It should loop through two splits: `['train', 'val']`.
        3.  Inside the loop, it should fetch the appropriate data batch for the split and calculate the loss.
        4.  The function must return a dictionary containing the mean loss for each split, e.g., `{'train': train_loss, 'val': val_loss}`.

*   **Sub-task 1.2: Update Logging in Main Loop**
    *   **File:** `train_refactored.py`
    *   **Action:** Update the evaluation section in the main `while` loop to correctly access and log the separate training and validation losses returned from the updated `estimate_loss` function.
    *   **Guidance:**
        1.  Find the line: `print(f"step {iter_num}: train loss {val_loss:.4f}, val loss {val_loss:.4f}, ...")`.
        2.  Modify it to use the dictionary returned by `estimate_loss`. The new line should look like: `print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, ...")`.
        3.  Ensure that the `wandb` logging section also uses `losses['train']` and `losses['val']` correctly.

### Task 2: Fix DDP Gradient Accumulation

**Problem Reference:** `discrepancies_report.md`, Section 3.2. The script fails to adjust `gradient_accumulation_steps` for distributed training.

*   **Sub-task 2.1: Reinstate DDP Adjustment**
    *   **File:** `train_refactored.py`
    *   **Action:** After the DDP setup block, add the logic to divide `gradient_accumulation_steps` by the `ddp_world_size`.
    *   **Guidance:**
        1.  Locate the section where `ddp`, `ddp_rank`, and `ddp_world_size` are initialized.
        2.  Immediately after this block, insert the following code from the original `train.py`:
            ```python
            if ddp:
                config.gradient_accumulation_steps //= ddp_world_size
            ```
        3.  It's good practice to add an assertion to ensure the division is clean: `assert config.gradient_accumulation_steps % ddp_world_size == 0`.

### Task 3: Correct "Tokens per Iteration" Calculation

**Problem Reference:** `discrepancies_report.md`, Section 3.3. The informational log message for "tokens per iteration" is inaccurate.

*   **Sub-task 3.1: Update Throughput Logging**
    *   **File:** `train_refactored.py`
    *   **Action:** Correct the formula used in the `print` statement that displays the number of tokens processed per iteration.
    *   **Guidance:**
        1.  Find the line: `print(f"tokens per iteration will be: {config.batch_size * config.block_size}")`.
        2.  Modify it to include all relevant factors: `gradient_accumulation_steps` and `ddp_world_size`. The corrected line should be:
            ```python
            tokens_per_iter = config.gradient_accumulation_steps * ddp_world_size * config.batch_size * config.block_size
            print(f"tokens per iteration will be: {tokens_per_iter:,}")
            ```
        3.  Note that `ddp_world_size` will be `1` in a non-DDP run, so this single formula works for both cases.

### Task 4: Align Asynchronous Analysis and Scheduler Flow

**Problem Reference:** `discrepancies_report.md`, Sections 3.4 and 3.5. The refactored script has a different and potentially buggy analysis callback. The `TrainingScheduler` also introduces an extra, unintended evaluation step.

*   **Sub-task 4.1: Consolidate Evaluation Logic**
    *   **File:** `training/scheduler.py`
    *   **Action:** Remove the `estimate_loss` call from the `check_and_execute_operations` method within the `TrainingScheduler` class. The scheduler's role should be to check conditions based on a *provided* loss value, not to trigger a new evaluation itself.
    *   **Guidance:**
        1.  The `check_and_execute_operations` method should accept `current_val_loss` as an argument.
        2.  Remove any lines from that method that call `estimate_loss()`. The validation loss should be passed in from the main training loop's evaluation step.

*   **Sub-task 4.2: Align Analysis Callback**
    *   **File:** `train_refactored.py`
    *   **Action:** Replace the logic within `simple_callback` to precisely match the parsing and printing format of the `analysis_done_callback` function from the original `train.py`.
    *   **Guidance:**
        1.  Carefully review the structure of the `results` dictionary produced by the analysis task.
        2.  The callback should correctly parse nested keys like `geometry['embeddings']['global_sparsity']`.
        3.  Format the print statements to be identical to those in the original script to ensure consistent logging output. This is critical for comparing runs.

### Task 5: Minor Functional Alignments

**Problem Reference:** `discrepancies_report.md`, Section 4. These are minor differences that should be aligned for consistency.

*   **Sub-task 5.1: Standardize Data Loader Method Name**
    *   **File:** `training/utils.py` (or wherever `BatchManager` is defined)
    *   **Action:** Rename the `get_batch()` method in the `BatchManager` class to `get_next_batch()`.
    *   **Guidance:**
        1.  Perform a simple find-and-replace for all usages of this method within `train_refactored.py` to ensure the call sites are updated.

---

## 3. Validation and Testing

After implementing these changes, the developer should perform the following checks:

1.  **Run a Short Training Session:** Execute `train_refactored.py` for a few hundred iterations.
2.  **Check the Logs:**
    *   Verify that the "train loss" and "val loss" reported at evaluation intervals are now different and plausible.
    *   Confirm that the "tokens per iteration" log message at startup displays the correct, larger number.
    *   Check the output from the asynchronous analysis callback to ensure it is formatted correctly and not throwing errors.
3.  **(Optional) DDP Test:** If a multi-GPU environment is available, run the script with DDP to confirm that the gradient accumulation fix works as expected and the training does not hang or error out.

By following this plan, `train_refactored.py` should be restored to full functionality while retaining its superior modular structure.
