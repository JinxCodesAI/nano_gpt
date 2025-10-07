import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

from model import GPT, GPTConfig, ModelMode


class MLMInferenceEngine:
    """Handles loading and inference with pre-trained LM/MLM models.

    Note: Uses our GPT in LANGUAGE_MODEL mode and forces a full forward pass
    (by providing targets) to obtain logits for all sequence positions.
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu', verbose: bool = False):
        self.device = device
        self.verbose = verbose

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract model configuration
        if 'model_args' in checkpoint:
            model_args = dict(checkpoint['model_args'])
        else:
            raise ValueError("Checkpoint missing model_args")

        # Filter out deprecated config fields (for backward compatibility with old checkpoints)
        deprecated_fields = {'mode', 'num_token_classes', 'binary_classification'}
        filtered_model_args = {k: v for k, v in model_args.items() if k not in deprecated_fields}

        # Create and load model
        config = GPTConfig(**filtered_model_args)
        self.model = GPT(config)

        # Ensure model is in language modeling mode for inference
        self.model.set_mode(ModelMode.LANGUAGE_MODEL)
        state_dict = checkpoint.get('model', checkpoint)

        # Handle old checkpoints with single head (backward compatibility)
        has_lm_head = any(k.startswith('lm_head.') for k in state_dict.keys())
        has_sequence_head = any(k.startswith('sequence_head.') for k in state_dict.keys())

        if not has_lm_head and has_sequence_head:
            if 'transformer.wte.weight' in state_dict:
                state_dict['lm_head.weight'] = state_dict['transformer.wte.weight'].clone()
        elif has_lm_head and not has_sequence_head:
            n_embd = state_dict['transformer.wte.weight'].shape[1]
            state_dict['sequence_head.base_predictor.weight'] = torch.randn(1, n_embd) * 0.01
            state_dict['sequence_head.base_predictor.bias'] = torch.zeros(1)
            state_dict['sequence_head.log_temperature'] = torch.zeros(1)

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device)
        self.model.eval()

        # Extract vocabulary info from meta
        if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
            # Some training runs store meta under config['meta']
            meta = checkpoint['config'].get('meta', {})
        else:
            meta = checkpoint.get('meta', {})
        if not meta:
            raise ValueError("Checkpoint missing metadata (expected under 'config.meta' or 'meta')")

        self.vocab_size = meta.get('vocab_size')
        self.mask_token_id = meta.get('mask_token_id')
        self.stoi = meta.get('stoi', {})
        self.itos = meta.get('itos', {})

        if self.vocab_size is None or self.mask_token_id is None:
            raise ValueError("Meta must contain 'vocab_size' and 'mask_token_id'")

        if self.verbose:
            print("MLMInferenceEngine initialized:")
            print(f"  Model: {config.n_layer}L/{config.n_head}H/{config.n_embd}D")
            print(f"  Vocab size: {self.vocab_size}")
            print(f"  Mask token ID: {self.mask_token_id}")

    @torch.no_grad()
    def predict_masked_tokens(
        self,
        input_ids: torch.Tensor,
        mask_positions: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Predict tokens for masked positions.

        Args:
            input_ids: Input sequence with [MASK] tokens [batch_size, seq_len]
            mask_positions: Boolean mask indicating positions to predict [batch_size, seq_len]
            temperature: Sampling temperature
            top_k: Top-k sampling (None for full)

        Returns:
            Predicted token IDs for masked positions [batch_size, seq_len]
        """
        device = self.device
        input_ids = input_ids.to(device)
        mask_positions = mask_positions.to(device)

        # Force full sequence logits by providing dummy targets (we discard the loss)
        logits, _ = self.model(input_ids, targets=input_ids)
        # logits: [B, T, V]

        # Select only masked positions across the batch
        masked_logits = logits[mask_positions]  # [N_masked, V]

        if masked_logits.numel() == 0:
            return input_ids.clone().to(input_ids.device)

        # Temperature scaling
        if temperature != 1.0:
            masked_logits = masked_logits / max(temperature, 1e-6)

        # Top-k filtering
        if top_k is not None:
            k = min(top_k, masked_logits.size(-1))
            v, _ = torch.topk(masked_logits, k)
            masked_logits[masked_logits < v[:, [-1]]] = -float('inf')

        # Sample predictions
        probs = F.softmax(masked_logits, dim=-1)
        predicted_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Stitch back into a result tensor
        result = input_ids.clone()
        result[mask_positions] = predicted_tokens.to(result.device)
        return result

