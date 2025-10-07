"""
Utility functions for model inference and critic-guided refinement.
Extracted from interactive_diffusion_explorer.py for reuse.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from contextlib import nullcontext

import torch
from model import GPTConfig, GPT


def load_model_checkpoint(checkpoint_path: str, device: str = 'cuda', dtype: str = 'float16') -> Tuple[GPT, Dict[str, Any]]:
    """
    Load a model checkpoint and return the model and metadata.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        dtype: Data type for model
        
    Returns:
        Tuple of (model, metadata_dict) where metadata contains:
            - vocab_size, block_size, etc from model_args
            - has_critic: bool indicating if critic head is present
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_args' not in checkpoint:
        raise ValueError("Invalid checkpoint: missing 'model_args'")
    
    model_args = checkpoint['model_args']
    
    # Ensure backward compatibility
    if 'attention_type' not in model_args:
        model_args['attention_type'] = 'causal'
    if 'position_encoding' not in model_args:
        model_args['position_encoding'] = 'absolute'

    # Filter out deprecated config fields (for backward compatibility with old checkpoints)
    deprecated_fields = {'mode', 'num_token_classes', 'binary_classification'}
    old_mode = model_args.get('mode', None)
    filtered_model_args = {k: v for k, v in model_args.items() if k not in deprecated_fields}

    # Create model
    gptconf = GPTConfig(**filtered_model_args)
    model = GPT(gptconf)

    # Set mode based on old config if present
    if old_mode:
        if old_mode == 'sequence_scorer' or old_mode == ModelMode.SEQUENCE_SCORER:
            model.set_mode(ModelMode.SEQUENCE_SCORER)
        elif old_mode == 'language_model' or old_mode == ModelMode.LANGUAGE_MODEL:
            model.set_mode(ModelMode.LANGUAGE_MODEL)
    
    # Load weights
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

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

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    
    # Extract metadata
    metadata = {
        'vocab_size': model_args.get('vocab_size'),
        'block_size': model_args.get('block_size'),
        'attention_type': model_args.get('attention_type'),
        'position_encoding': model_args.get('position_encoding'),
        'has_critic': model_args.get('add_critic_head', False) and hasattr(model, 'critic_head'),
        'mask_token_id': model_args.get('mask_token_id'),
        'pad_token_id': model_args.get('pad_token_id'),
        'cls_token_id': model_args.get('cls_token_id'),
        'sep_token_id': model_args.get('sep_token_id'),
    }
    
    return model, metadata


def load_vocabulary(meta_path: str) -> Tuple[Dict[int, str], Dict[str, int], Dict[str, Any]]:
    """
    Load vocabulary and metadata from meta.pkl file.
    
    Args:
        meta_path: Path to meta.pkl file
        
    Returns:
        Tuple of (itos, stoi, meta_dict)
    """
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    itos = meta.get('itos', {})
    stoi = meta.get('stoi', {})
    
    return itos, stoi, meta


def encode_text(text: str, stoi: Dict[str, int]) -> List[int]:
    """Encode text to token IDs using character-level tokenization."""
    return [stoi.get(c, 0) for c in text]


def decode_tokens(tokens: torch.Tensor, itos: Dict[int, str], 
                  mask_token_id: Optional[int] = None,
                  pad_token_id: Optional[int] = None,
                  cls_token_id: Optional[int] = None,
                  sep_token_id: Optional[int] = None) -> str:
    """
    Decode token IDs to text.
    
    Args:
        tokens: Tensor of token IDs
        itos: Index to string mapping
        mask_token_id: ID of mask token (will be rendered as [MASK])
        pad_token_id: ID of pad token (will be rendered as [PAD])
        cls_token_id: ID of CLS token (will be rendered as [CLS])
        sep_token_id: ID of SEP token (will be rendered as [SEP])
        
    Returns:
        Decoded string
    """
    if hasattr(tokens, 'tolist'):
        tokens = tokens.tolist()
    
    result = []
    for token_id in tokens:
        if mask_token_id is not None and token_id == mask_token_id:
            result.append('[MASK]')
        elif cls_token_id is not None and token_id == cls_token_id:
            result.append('[CLS]')
        elif pad_token_id is not None and token_id == pad_token_id:
            result.append('[PAD]')
        elif sep_token_id is not None and token_id == sep_token_id:
            result.append('[SEP]')
        elif token_id < len(itos):
            result.append(itos[token_id])
        else:
            result.append(f'[UNK:{token_id}]')
    return ''.join(result)


def unmask_tokens(model: GPT, tokens: torch.Tensor, mask_token_id: int, 
                 vocab_size: int, temperature: float = 0.8,
                 device: str = 'cuda', dtype: str = 'float16') -> torch.Tensor:
    """
    Unmask all [MASK] tokens in the input using model predictions.
    
    Args:
        model: The GPT model
        tokens: Input tokens (1, seq_len) or (seq_len,)
        mask_token_id: ID of the mask token
        vocab_size: Vocabulary size
        temperature: Sampling temperature
        device: Device to run on
        dtype: Data type
        
    Returns:
        Unmasked tokens (same shape as input)
    """
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    tokens = tokens.to(device)
    
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
    
    with torch.no_grad():
        with ctx:
            dummy_targets = torch.zeros_like(tokens)
            model_output = model(tokens, targets=dummy_targets)
            
            if isinstance(model_output, tuple):
                logits = model_output[0]
            else:
                logits = model_output
            
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            
            # Find masked positions
            masked_positions = (tokens[0] == mask_token_id).nonzero(as_tuple=True)[0]
            
            # Sample predictions for masked positions
            result_tokens = tokens.clone()
            for pos in masked_positions:
                pos_probs = probs[0, pos, :vocab_size-1]  # Exclude mask token
                predicted_token = torch.multinomial(pos_probs, 1).item()
                result_tokens[0, pos] = predicted_token
    
    if squeeze_output:
        result_tokens = result_tokens.squeeze(0)
    
    return result_tokens


def score_tokens_with_critic(model: GPT, tokens: torch.Tensor,
                             device: str = 'cuda', dtype: str = 'float16') -> torch.Tensor:
    """
    Score tokens using the critic head.
    
    Args:
        model: The GPT model with critic head
        tokens: Input tokens (1, seq_len) or (seq_len,)
        device: Device to run on
        dtype: Data type
        
    Returns:
        Critic scores (probabilities 0-1, higher = more likely wrong) same shape as input
    """
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    tokens = tokens.to(device)
    
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
    
    with torch.no_grad():
        with ctx:
            critic_logits = model.critic_scores(tokens)
            critic_probs = torch.sigmoid(critic_logits)
    
    if squeeze_output:
        critic_probs = critic_probs.squeeze(0)
    
    return critic_probs


def remask_worst_tokens(tokens: torch.Tensor, critic_scores: torch.Tensor,
                       mask_token_id: int, threshold_pct: float,
                       content_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remask the worst tokens based on critic scores.
    
    Args:
        tokens: Current tokens (1, seq_len) or (seq_len,)
        critic_scores: Critic scores for each token (same shape as tokens)
        mask_token_id: ID of the mask token
        threshold_pct: Percentage of tokens to remask (0-100)
        content_len: Optional content length (excludes padding from remasking)
        
    Returns:
        Tuple of (remasked_tokens, indices_of_remasked_tokens)
    """
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
        critic_scores = critic_scores.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    if content_len is None:
        content_len = tokens.shape[1]
    
    # Calculate number of tokens to remask
    num_to_remask = int((threshold_pct / 100.0) * content_len)
    num_to_remask = max(1, min(num_to_remask, content_len))
    
    # Select top-k worst tokens (highest critic scores) within content
    content_scores = critic_scores[0, :content_len]
    _, worst_indices = torch.topk(content_scores, k=num_to_remask, largest=True)
    
    # Apply remasking
    remasked_tokens = tokens.clone()
    for idx in worst_indices:
        remasked_tokens[0, idx] = mask_token_id
    
    if squeeze_output:
        remasked_tokens = remasked_tokens.squeeze(0)
    
    return remasked_tokens, worst_indices

