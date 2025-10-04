# **Technical Specification: Hierarchical U-Net Transformer (HUT)**

## 1.0 Executive Summary & Project Goal
*(This section remains the same as Version 1.0)*

## 2.0 Detailed Model Architecture



### 2.1. Hierarchical Representation & Configuration
The model operates on a pyramid of tensor representations. The structure of this pyramid is not hard-coded but is defined by a configuration parameter.

* **`config.pooling_ratios`**: A list of integers defining the pooling factor at each downsampling stage. For the target application, a proposed configuration is `[8, 8, 4, 4]`, which defines the following five levels:
    * **L0:** Base character sequence (Length `L`). Example: 1024.
    * **L1:** `L0 -> L1` (8:1 pooling). Length `L/8`. Example: 128.
    * **L2:** `L1 -> L2` (8:1 pooling). Length `L/64`. Example: 16.
    * **L3:** `L2 -> L3` (4:1 pooling). Length `L/256`. Example: 4.
    * **L4 (Bottleneck):** `L3 -> L4` (4:1 pooling). Length `L/1024`. Example: 1.

### 2.2. Foundational Component: The Transformer Block
The primary computational unit is a standard **pre-norm Transformer Block** with bidirectional attention, consisting of:
1.  Layer Normalization.
2.  Multi-Head Self-Attention (MHA) with 2D-aware RoPE (see 2.3).
3.  Residual Connection with Dropout.
4.  Layer Normalization.
5.  A 2-layer Feed-Forward Network (FFN) with GELU activation.
6.  Residual Connection with Dropout.

### 2.3. Positional Encoding Scheme (Modular Design)
The positional encoding must be 2D-aware, encoding both `(level, position_in_level)`. This component must be designed modularly to allow for empirical testing of different strategies. The two primary variants to be implemented and tested are:

* **Option A: 2D Rotary Positional Embedding (RoPE)**
    * **Mechanism:** This is a parameter-free approach. The model's embedding dimension, `n_embd`, is conceptually split. The first half of the features are rotated based on the `level` index, and the second half are rotated based on the `position_in_level` index. This requires a custom implementation of the RoPE application logic but supports a fully recursive, depth-flexible architecture.
    * **Storage:** No learnable parameters are stored. Sin/Cos caches are generated on the fly or pre-computed.

* **Option B: Hybrid (Learnable Level Embedding + 1D RoPE)**
    * **Mechanism:** This approach separates the two dimensions. A learnable embedding vector is used to represent the discrete `level`, while the standard 1D RoPE is used to represent the continuous `position_in_level`.
    * **Storage:** A learnable `nn.Embedding` layer (`level_embeddings`) is stored as part of the main `HierarchicalUNetTransformer` module. In the forward pass, the appropriate level embedding is added to the token representations for that segment *before* the 1D RoPE is applied within the attention mechanism. This approach fixes the maximum number of levels at initialization.


### 2.4. Encoder Path (Downsampling)
The encoder path comprises a series of `DownsamplingBlock` modules. Crucially, each block generates the next level of the hierarchy (`Level_{i+1}`) by explicitly conditioning on **all** preceding levels (`Levels_{0...i}`), ensuring maximum information retention.

#### 2.4.1. Downsampling Block Data Flow
The process to generate `Level_{i+1}` is as follows:

1.  **Input Assembly:** The primary input to the `i`-th downsampling process is the output of the previous level, `Level_i`.
2.  **Context Upsampling & Concatenation:** To explicitly condition on all lower levels, every preceding level (`L_0` through `L_{i-1}`) is upsampled via nearest-neighbor interpolation (`F.interpolate`) to match the sequence length of `Level_i`. These upsampled tensors are then concatenated with the `Level_i` tensor along the embedding dimension.
3.  **Feature Projection:** This wide, concatenated tensor is passed through a dedicated linear layer (`nn.Linear`) to project it back to the model's standard embedding dimension, `n_embd`. This step merges the rich, multi-scale information into a single, unified representation.
4.  **Local Processing:** The resulting merged tensor is processed by a Transformer Block using a **block-diagonal attention mask** (as specified in 2.4.2) to refine the representation locally.
5.  **Learnable Pooling & Output:** The output of the local processing is passed through a **1D strided convolution** (`nn.Conv1d`) to reduce its sequence length, producing the `Level_{i+1}` representation.

* **Outputs:** The block outputs two tensors: the result of the local processing (**pre-pooling**), which serves as the **skip connection**, and the final **pooled** tensor, which is the input for the next downsampling stage.

#### 2.4.2. Local Attention Window vs. Pooling Ratio
To maintain a strict, non-overlapping hierarchical structure, the **window size for local self-attention must be equal to the pooling ratio.**

* **Mechanism:** For a level with a pooling ratio of 8 (e.g., L1 -> L2), the self-attention mechanism within the Transformer block is constrained to operate only within distinct, non-overlapping blocks of 8 tokens.
* **Attention Mask:** This is implemented via a **block-diagonal attention mask**, preventing information from "leaking" between the chunks that are about to be summarized.
* **Note:** For Version 1, a strict non-overlapping design is specified.

#### 2.4.3. Implementation & Optimization (FlashAttention)
The use of a block-diagonal attention mask constitutes a form of **sparse attention**, which allows for significant computational and memory optimizations (reducing complexity from `O(T²)` to `O(T * W)`).

* **Efficient Implementation:** This is best realized using modern, I/O-aware attention algorithms like **FlashAttention**. The most efficient method is the **"reshaping trick"**:
    1.  The input tensor is reshaped from `(batch_size, sequence_length, n_embd)` into `(batch_size * num_blocks, window_size, n_embd)`.
    2.  A standard, highly-optimized dense attention operation is performed on this new, large batch of small sequences.
    3.  The output tensor is reshaped back to its original format.


### 2.4. Encoder Path (Downsampling)

The best place to put this information is directly within **Section 2.4.1 (Local Attention Window vs. Pooling Ratio)**. This is the most logical location because it immediately follows the description of the block-diagonal attention mask, explaining *how* to implement that specific mask efficiently.

Here is the updated section, with the new information added as a sub-section `2.4.1.1`.

***

### 2.4. Encoder Path (Downsampling)

#### 2.4.1. Local Attention Window vs. Pooling Ratio
To maintain a strict, non-overlapping hierarchical structure, the **window size for local self-attention must be equal to the pooling ratio.**

* **Mechanism:** For a level with a pooling ratio of 8 (e.g., L1 -> L2), the input sequence of length 128 is processed. The self-attention mechanism is constrained to operate only within distinct blocks of 8 tokens.
* **Attention Mask:** As you correctly intuited, this is implemented via a **block-diagonal attention mask**. For an input of 128 tokens and a window/ratio of 8, the attention mask would be a 128x128 matrix composed of 16 contiguous 8x8 blocks of ones along the diagonal, with all other positions set to zero. This prevents information from "leaking" between the blocks that are about to be summarized.
* **Note:** Treating `window_size` as an independent hyperparameter larger than the ratio would create overlapping windows, a different architectural choice that could also be explored. For V1, a strict non-overlapping design is specified.

#### 2.4.1.1. Implementation & Optimization (FlashAttention)
The use of a block-diagonal attention mask constitutes a form of **sparse attention**, which allows for significant computational and memory optimizations.

* **Complexity Reduction:** Standard self-attention has a complexity that scales quadratically with sequence length `T`, i.e., `O(T²)`. By using block-diagonal attention with a window size `W`, the complexity is reduced to scale **linearly** with sequence length, `O(T * W)`.

* **Efficient Implementation:** This optimization is realized using modern, I/O-aware attention algorithms like **FlashAttention**. The most efficient and standard method to implement this is the **"reshaping trick"**:
    1.  The input tensor, with shape `(batch_size, sequence_length, n_embd)`, is reshaped into `(batch_size * num_blocks, window_size, n_embd)`.
    2.  A standard, highly-optimized dense attention operation (e.g., PyTorch's `scaled_dot_product_attention`, which has a FlashAttention backend) is performed on this new, large batch of small sequences. Attention is naturally confined within each window without needing an explicit mask.
    3.  The output tensor is reshaped back to its original format.

This approach perfectly maps the problem to GPU hardware, leveraging optimized kernels for maximum speed and memory efficiency.

#### 2.4.2. Downsampling Block Data Flow
You asked for a clearer example of the two outputs from a `DownsamplingBlock`.

* **Input:** Let's say the block receives a tensor `x` of shape `(B, 256, C)` representing Level 1.
* **Step 1: Local Processing:** `x` is passed through the internal Transformer Block. The output, `processed_x`, still has the shape `(B, 256, C)`. This tensor is enriched with local contextual information.
    * **`processed_x` is the first output, designated as the skip connection.**
* **Step 2: Learnable Pooling:** `processed_x` is then passed through the `Conv1d` pooling layer (`kernel_size=8, stride=8`). The output, `pooled_x`, now has a shape of `(B, 32, C)`.
    * **`pooled_x` is the second output, which becomes the input for the next level in the hierarchy.**

### 2.5. Bottleneck
The bottleneck consists of one or more standard bidirectional Transformer Blocks that operate on the final `L4` representation. This is where the embedding size may be optionally increased to better capture the global information context.

### 2.6. Decoder Path (Upsampling)
The decoder path is a mirror of the encoder path, using a series of `UpsamplingBlock` modules.

* **Module:** `UpsamplingBlock(in_channels, out_channels, ratio)`
    * **Input:** The tensor from the level above (`Level_{i+1}`) and the corresponding skip connection tensor from the encoder path (`Level_i`).
    * **Operations:**
        1.  **Learnable Upsampling:** A **1D transposed convolution** (`nn.ConvTranspose1d`) with `kernel_size=ratio` and `stride=ratio` increases the sequence length of the input tensor.
        2.  **Skip Connection Integration:** The upsampled tensor is concatenated with the skip connection tensor along the embedding dimension.
        3.  **Feature Projection:** A linear layer projects the concatenated tensor back to `n_embd`.
        4.  **Local Processing:** The merged tensor is processed by a bidirectional Transformer Block.
    * **Output:** The refined `Level_i` representation, which is passed to the next upsampling block.

### 2.7. Output Head
The final output of the decoder path (the reconstructed Level 0) is passed through a final `LayerNorm` and a linear layer tied to the character embedding weights to produce character logits.


---
## 3.0 Training Strategy

### 3.1. Primary Objective: Masked Language Modeling (MLM)
The model is trained end-to-end via MLM on the character-level sequence (L0). A subset of input tokens is masked, and the model's final output is trained to predict the original tokens at those positions using a standard Cross-Entropy Loss.

### 3.2. Auxiliary Objective: Multi-Level Reconstruction Loss

#### 3.2.1. Index Alignment
You are correct to scrutinize the indexing for the skip connections. The relationship is symmetric. The output of the `i`-th `DownsamplingBlock` is used by the `i`-th `UpsamplingBlock` from the end.

* **Example (4 Levels, indices 0 to 3):**
    * `down_blocks[0]` produces `skip_0`. This is used by `up_blocks[3]` (the first upsampling step).
    * `down_blocks[1]` produces `skip_1`. This is used by `up_blocks[2]`.
    * `down_blocks[2]` produces `skip_2`. This is used by `up_blocks[1]`.
    * `down_blocks[3]` produces `skip_3`. This is used by `up_blocks[0]` (the last upsampling step).
* **General Formula:** The skip connection from `down_blocks[i]` is consumed by `up_blocks[num_levels - 1 - i]`. The specification is correct as written.

### 3.3. Loss Weighting: Learned Uncertainty
To balance the MLM loss and the multiple auxiliary reconstruction losses, the weights are not hand-tuned. Instead, they are **learned by the model** as a function of its predictive uncertainty for each task.

* **Mechanism:** For each loss term (the final MLM loss and each intermediate reconstruction loss), the model predicts a learnable scalar, `σ²` (variance).
* **Loss Formulation:** The total loss is the sum of individual task losses, where each is calculated as:
    `TaskLoss_i = (0.5 / σ_i²) * ReconstructionLoss_i + 0.5 * log(σ_i²)`
* **Rationale:** This formulation, based on homoscedastic uncertainty, allows the model to automatically down-weigh tasks it finds inherently noisy or difficult (by increasing `σ²`) and up-weigh tasks it is confident about. The `log(σ²)` term acts as a regularizer, preventing the model from simply setting all uncertainties to infinity.

## 4.0 Inference & Generation Procedure

### 4.1. Prompt Initialization
A user-provided prompt is converted into an initial top-level `L4` embedding vector using a pre-trained, auxiliary **Prompt Encoder**. This Prompt Encoder is a small Transformer trained separately to map short text sequences to the latent space of the main HUT Encoder.

### 4.2. Top-Down Plan Generation
The initial `L4` vector is fed into the **decoder path** of the HUT. This single top-down pass deterministically generates a full hierarchy of plan embeddings (`L3` down to `L0`). The final `L0` output from this pass can be used to sample an initial draft of the text.

### 4.3. Guided Iterative Refinement
A diffusion-like loop is performed for a fixed number of steps (`N`) to refine the text.
1.  **Re-masking:** A portion of the current text is masked.
2.  **Full U-Net Pass:** The masked text is passed through the entire HUT to generate new output logits.
3.  **Guidance:** The update is guided by an "elastic string" loss: `Loss = L_reconstruction + λ * L_prior`.
    * `L_reconstruction`: Encourages the model to be self-consistent with the unmasked portions of its own generation.
    * `L_prior`: An MSE loss that penalizes the `L4` embedding from the current pass for drifting from the original `L4` embedding generated by the prompt. `λ` controls guidance strength.
4.  **Resampling:** Masked tokens are re-sampled based on the guided output, improving the text quality.

---
## 5.0 Architectural Variants & Trade-Offs

The following key design decisions present trade-offs and should be investigated during development.

| Feature | **Variant A: Specialized (Fixed Depth)** | **Variant B: Recursive (Flexible Depth)** | **Recommendation for V1** |
| :--- | :--- | :--- | :--- |
| **Parameter Sharing** | Each `Down/Up-samplingBlock` has its own unique set of weights. | A single `DownsamplingBlock` and `UpsamplingBlock` are defined and reused at each level. | **Variant A**. While less parameter-efficient, it allows the model to learn specialized functions for each level of abstraction, which is likely crucial for performance. Flexibility can be explored later. |
| **Embedding Size** | `n_embd` is constant across all levels. | `n_embd` increases towards the bottleneck and decreases on the decoder path. | **Variant A**. Start with a fixed size for simplicity. If loss analysis (see 2.) indicates a bottleneck, explore a variable size in V2. |
| **Pooling Mechanism** | **1D Strided Convolution** (`nn.Conv1d`). | **Attention Pooling** (e.g., a dedicated summary token and a self-attention layer). | **Variant A**. `Conv1D` is computationally cheaper and provides a strong, proven baseline for learnable pooling. |

---
## 6.0 Potential Challenges & Mitigation Strategies

* **Gradient Flow Stability:** The deep, multi-level nature of the model poses a risk of vanishing or exploding gradients.
    * **Mitigation:** Strict adherence to pre-norm architecture (`LayerNorm` before attention/MLP), careful weight initialization, and potentially gradient checkpointing to reduce memory usage and improve stability.
* **Computational Cost:** The explicit multi-level conditioning, especially the upsampling and concatenation, is memory and compute-intensive.
    * **Mitigation:** Begin with a smaller-scale model (fewer levels, smaller embedding size) to validate the architecture. Use mixed-precision training (`AMP`).
* **Implementation Complexity:** The 2D RoPE, custom U-Net blocks, and uncertainty-weighted loss are non-trivial to implement correctly.
    * **Mitigation:** Develop and test each component in isolation with unit tests before integrating them into the full model.
* **Hyperparameter Tuning:** The model has a large number of hyperparameters (pooling ratios, window sizes, guidance strength `λ`, number of levels).
    * **Mitigation:** Focus on a minimal, validated configuration first. A principled approach to hyperparameter optimization (e.g., using a tool like Optuna) should be planned for after the initial prototype is working.


### 6.1. Configuration
* **Vocabulary Size:** `vocab_size = 64`
* **Sequence Length:** `block_size = 1024`
* **Embedding Dimension:** `n_embd = 256`
* **Number of Heads:** `n_head = 4`
* **Pooling Ratios:** `pooling_ratios = [4, 4, 4, 4]` (4 levels, each pooling by 4)
* **Layers per Block:** `n_layer_per_block = 2` (Each `Down/Up-samplingBlock` contains a 2-layer Transformer)

### 6.2. Parameter Calculation (Approximate)
Let `C = n_embd = 256`. The FFN expansion factor is 4. A single Transformer block has `12 * C^2` params.

1.  **Embeddings:**
    * `char_embed`: `64 * 256` = 16,384
    * `level_embed` (Option B): `5 levels * 256` = 1,280
    * **Subtotal:** ~18k

2.  **Encoder Path (4 Downsampling Blocks):**
    * Each `DownsamplingBlock` contains:
        * `2 * Transformer Blocks`: `2 * (12 * 256^2)` ≈ 1.57M params
        * `Conv1d Pool`: `(4 * 256 * 256) + 256` ≈ 0.26M params
        * `Projection Layer` (worst case `5C -> C`): `5 * 256 * 256 + 256` ≈ 0.33M params
    * Total per block ≈ 2.16M
    * **Subtotal (4 blocks):** ~8.64M

3.  **Bottleneck Block:**
    * `2 * Transformer Blocks`: `2 * (12 * 256^2)` ≈ 1.57M
    * **Subtotal:** ~1.57M

4.  **Decoder Path (4 Upsampling Blocks):**
    * Each `UpsamplingBlock` contains:
        * `ConvTranspose1d Unpool`: `(256 * 256 * 4) + 256` ≈ 0.26M params
        * `Projection Layer` (for skip connection `2C -> C`): `2 * 256 * 256 + 256` ≈ 0.13M params
        * `2 * Transformer Blocks`: `2 * (12 * 256^2)` ≈ 1.57M params
    * Total per block ≈ 1.96M
    * **Subtotal (4 blocks):** ~7.84M

5.  **Final Head:**
    * `final_head`: `256 * 64` = 16,384
    * **Subtotal:** ~16k

6.  **Uncertainty Parameters:**
    * `log_vars`: 5 params (negligible)

**Total Estimated Parameters for "HUT-Tiny":**
`18k + 8.64M + 1.57M + 7.84M + 16k` ≈ **18.1 Million parameters**

---
## 7.0 Hyperparameter Tuning Guide

This table outlines key hyperparameters, their meaning, and symptoms that might suggest tuning them.

| Parameter | Meaning | Symptoms to INCREASE Value | Symptoms to DECREASE Value |
| :--- | :--- | :--- | :--- |
| **`pooling_ratios`** | The list of factors for downsampling at each level. Defines the hierarchy's shape. | Model fails to capture very long-range dependencies. The bottleneck is not abstract enough. | Model loses too much fine-grained detail early. Low-level reconstruction loss is high. |
| **`n_embd`** | The core dimensionality of the model's representations. | Reconstruction loss is high across all levels; model seems to lack capacity to store information. | Model overfits quickly. Memory usage is too high for the target hardware. |
| **`n_head`** | Number of attention heads in each MHA layer. | Model struggles with nuanced syntactic or semantic relationships. | Attention patterns are redundant across heads. Minor performance gains for high computational cost. |
| **`n_layer_per_block`** | The depth of the Transformer within each `Down/Up-samplingBlock`. | The model's representations at a given level feel "shallow." It's not learning complex patterns from its local context. | Model overfits. Training is slow. Gradients become unstable. |
| **`λ` (Guidance Strength)**| The weight of the "elastic string" prior loss during inference. | Generated text quickly drifts from the topic of the initial prompt. | Generated text is repetitive and lacks creativity, sticking too rigidly to the prompt. |
| **`dropout`** | The dropout rate applied for regularization. | Model is overfitting; training loss is low but validation loss is high. | Model is underfitting; it fails to converge and learn the training data effectively. |
## 8.0 Architectural Variants & Trade-Offs

The following key design decisions present trade-offs and should be investigated during development.

| Feature | **Variant A: Specialized (Fixed Depth)** | **Variant B: Recursive (Flexible Depth)** | **Recommendation for V1** |
| :--- | :--- | :--- | :--- |
| **Parameter Sharing** | Each `Down/Up-samplingBlock` has its own unique set of weights. | A single `DownsamplingBlock` and `UpsamplingBlock` are defined and reused at each level. | **Variant A**. While less parameter-efficient, it allows the model to learn specialized functions for each level of abstraction, which is likely crucial for performance. Flexibility can be explored later. |
| **Embedding Size** | `n_embd` is constant across all levels. | `n_embd` increases towards the bottleneck and decreases on the decoder path. | **Variant A**. Start with a fixed size for simplicity. If loss analysis (see 2.) indicates a bottleneck, explore a variable size in V2. |
| **Pooling Mechanism** | **1D Strided Convolution** (`nn.Conv1d`). | **Attention Pooling** (e.g., a dedicated summary token and a self-attention layer). | **Variant A**. `Conv1D` is computationally cheaper and provides a strong, proven baseline for learnable pooling. |

### 8.1. Enabling Variant B with Low-Rank Factorization (LoRF)

The primary challenge of **Variant B (Variable Embedding Size)** is the quadratic scaling of parameters within the wider bottleneck's Transformer blocks. Low-Rank Factorization (LoRF) is the recommended technique to mitigate this, making Variant B a viable and efficient option.

* **Mechanism:** LoRF proposes that the large weight matrices within a Transformer block do not need their full rank to be effective. A large matrix `W` (e.g., in an FFN or attention projection) of size `(input_dim, output_dim)` can be approximated by two smaller, "low-rank" matrices: a compression matrix `W_c` of size `(input_dim, r)` and a decompression matrix `W_d` of size `(r, output_dim)`. The effective operation becomes `x @ W_c @ W_d`.

* **Parameter Reduction:** The number of parameters for this layer is reduced from `input_dim * output_dim` to `(input_dim * r) + (r * output_dim)`, where `r` (the rank) is a small hyperparameter.

* **Application:** In the context of the HUT's bottleneck, LoRF would be applied to the largest linear layers: the expansion and contraction layers of the FFN, and potentially the Q/K/V projection matrices. For example, a 1024-dimensional FFN expansion layer (`1024 -> 4096`) with over 4 million parameters could be factorized with a rank of `r=128`, reducing its parameter count by over 80%.

* **Conclusion:** By using LoRF, the model can achieve the expressive benefits of a larger embedding dimension in the bottleneck while keeping the parameter count manageable. It is the key enabling technology for making Variant B practical.