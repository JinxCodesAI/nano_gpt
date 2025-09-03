"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
from torch.nn import functional as F

class ModelMode(Enum):
    """Defines the three operational modes for the transformer model"""
    LANGUAGE_MODEL = "language_model"      # Unmasking/reconstruction (current 'unmasking')
    TOKEN_CLASSIFIER = "token_classifier"  # Per-token multi-class classification (refactored from 'remasking_binary')
    SEQUENCE_SCORER = "sequence_scorer"    # Sequence-level scoring 0-1 (new)

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) as described in:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
    """
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Create frequency bands
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos[position_ids].unsqueeze(1)  # [seq_len, 1, dim]
    sin = sin[position_ids].unsqueeze(1)  # [seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """Causal self-attention with optional flash attention and RoPE"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Rotary positional embeddings
        self.head_dim = config.n_embd // config.n_head
        use_rope = getattr(config, 'use_rope', True)
        if use_rope:
            self.rotary_emb = RotaryPositionalEmbedding(
                self.head_dim, 
                max_position_embeddings=config.block_size,
                device=None  # Will be set when model is moved to device
            )
        else:
            self.rotary_emb = None

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Apply rotary positional embeddings if available
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(v, seq_len=T)
            position_ids = torch.arange(T, device=x.device).unsqueeze(0)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class BidirectionalSelfAttention(nn.Module):
    """Bidirectional self-attention with optional flash attention and RoPE"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Rotary positional embeddings
        self.head_dim = config.n_embd // config.n_head
        use_rope = getattr(config, 'use_rope', True)
        if use_rope:
            self.rotary_emb = RotaryPositionalEmbedding(
                self.head_dim, 
                max_position_embeddings=config.block_size,
                device=None  # Will be set when model is moved to device
            )
        else:
            self.rotary_emb = None

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Apply rotary positional embeddings if available
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(v, seq_len=T)
            position_ids = torch.arange(T, device=x.device).unsqueeze(0)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # bidirectional self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # No masking for bidirectional attention
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)

        # Choose attention type based on config
        attention_type = getattr(config, 'attention_type', 'causal')
        if attention_type == 'bidirectional':
            self.attn = BidirectionalSelfAttention(config)
            print("Using bidirectional attention")
        else:
            self.attn = CausalSelfAttention(config)
            print("Using causal attention")

        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    attention_type: str = 'causal' # 'causal' or 'bidirectional' - type of attention to use
    use_rope: bool = True # Use Rotary Position Embeddings instead of absolute position embeddings
    mode: ModelMode = ModelMode.LANGUAGE_MODEL
    num_token_classes: int = 2  # Number of classes for token classification (flexible, not just binary)
    cls_token_id: int = None  # Special token ID for [CLS] token in sequence scoring

    # Transfer learning support
    freeze_transformer: bool = False  # If True, freeze all transformer weights (feature extraction)
    init_from_checkpoint: str = None  # Path to pretrained checkpoint for transfer learning

    # Dynamic unfreezing support for two-stage training
    unfreeze_at_iteration: int = None  # Iteration at which to unfreeze transformer (None = never unfreeze)
    unfreeze_lr_multiplier: float = 0.1  # Learning rate multiplier when unfreezing (to avoid instability)

    # Backward compatibility
    binary_classification: bool = False  # Legacy parameter for backward compatibility

    def __post_init__(self):
        # Handle backward compatibility
        if self.binary_classification and self.mode == ModelMode.LANGUAGE_MODEL:
            print("WARNING: binary_classification=True detected, converting to TOKEN_CLASSIFIER mode")
            self.mode = ModelMode.TOKEN_CLASSIFIER
            self.num_token_classes = 2

        # Enforce bidirectional attention for classification tasks
        if self.mode in [ModelMode.TOKEN_CLASSIFIER, ModelMode.SEQUENCE_SCORER]:
            if self.attention_type != 'bidirectional':
                print(f"WARNING: {self.mode.value} requires bidirectional attention. Changing from '{self.attention_type}' to 'bidirectional'")
                self.attention_type = 'bidirectional'

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Debug counters for mask token encounters
        self.mask_encounter_counters = {
            'training_calls': 0,
            'validation_calls': 0,
            'masks_seen_training': 0,
            'masks_seen_validation': 0,
            'total_tokens_training': 0,
            'total_tokens_validation': 0
        }

        # Create transformer components - conditionally include position embeddings
        transformer_components = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        )
        
        # Only add absolute position embeddings if not using RoPE
        if not getattr(config, 'use_rope', True):
            transformer_components['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        
        self.transformer = nn.ModuleDict(transformer_components)

        # Create appropriate output head based on mode
        if self.config.mode == ModelMode.LANGUAGE_MODEL:
            # Language modeling: predict next token from vocabulary
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            # Weight tying for language modeling
            # with weight tying when using torch.compile() some warnings get generated:
            # "UserWarning: functional_call was passed multiple values for tied weights.
            # This behavior is deprecated and will be an error in future versions"
            # not 100% sure what this is, so far seems to be harmless. TODO investigate
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        elif self.config.mode == ModelMode.TOKEN_CLASSIFIER:
            # Token-level classification: flexible number of classes per token
            self.lm_head = nn.Linear(config.n_embd, config.num_token_classes, bias=False)
            print(f"Token classifier head: {config.num_token_classes} classes per token")
        elif self.config.mode == ModelMode.SEQUENCE_SCORER:
            # Sequence-level scoring: single continuous score 0-1 from [CLS] token
            self.sequence_head = nn.Sequential(
                nn.Linear(config.n_embd, 1, bias=False),
                nn.Sigmoid()  # Ensure output is between 0 and 1
            )
            # Use much smaller initialization to prevent gradient explosion
            # Initialize weights to small values around 0 for stable sigmoid gradients
            with torch.no_grad():
                self.sequence_head[0].weight.normal_(0.0, 0.01)  # Small std dev
            print("Sequence scorer head: continuous score 0-1 (small init for stability)")
        else:
            raise ValueError(f"Unknown model mode: {self.config.mode}")

        # Initialize all weights AFTER creating heads to preserve sequence head initialization
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Re-initialize sequence head with small weights for stability (after general init)
        if self.config.mode == ModelMode.SEQUENCE_SCORER:
            with torch.no_grad():
                self.sequence_head[0].weight.normal_(0.0, 0.01)  # Small std dev for stability
            print("Re-initialized sequence head with small weights after general initialization")

        # Transfer learning support: load pretrained weights if specified
        if config.init_from_checkpoint is not None and config.init_from_checkpoint != "":
            print(f"Loading pretrained weights from {config.init_from_checkpoint}")
            checkpoint = torch.load(config.init_from_checkpoint, map_location='cpu', weights_only=False)

            # Load transformer weights (excluding heads)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Filter out head weights from pretrained checkpoint and handle _orig_mod prefix
            transformer_state_dict = {}
            for k, v in state_dict.items():
                # Remove _orig_mod prefix if present (from torch.compile)
                clean_key = k
                if k.startswith('_orig_mod.'):
                    clean_key = k[len('_orig_mod.'):]

                # Only include transformer weights, exclude heads
                if (clean_key.startswith('transformer.') and
                    not clean_key.startswith('lm_head') and
                    not clean_key.startswith('sequence_head')):
                    transformer_state_dict[clean_key] = v

            # Load transformer weights
            missing_keys, unexpected_keys = self.load_state_dict(transformer_state_dict, strict=False)
            print(f"Loaded pretrained transformer weights:")
            print(f"  Missing keys: {len(missing_keys)} (expected for new heads)")
            print(f"  Unexpected keys: {len(unexpected_keys)}")

            if missing_keys:
                print(f"  Missing (will be randomly initialized): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"  Unexpected (ignored): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

            # Initialize CLS token embedding with small values for stability
            if hasattr(config, 'mode') and str(config.mode) == 'ModelMode.SEQUENCE_SCORER':
                cls_token_id = 66  # From debug logs
                if cls_token_id < self.transformer.wte.weight.size(0):
                    with torch.no_grad():
                        # Initialize CLS token with very small values
                        self.transformer.wte.weight[cls_token_id].normal_(0.0, 0.001)
                        print(f"Initialized CLS token (ID {cls_token_id}) with small random values")

        # Weight initialization already done before transfer learning - don't repeat
        
        # Transfer learning support: freeze transformer if requested
        if config.freeze_transformer:
            self.freeze_transformer_weights()

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        # DEBUG: Check parameter registration and gradient flow for sequence scoring
        if hasattr(config, 'mode') and str(config.mode) == 'ModelMode.SEQUENCE_SCORER':
            print(f"DEBUG: Sequence head detailed analysis:")
            print(f"  sequence_head type: {type(self.sequence_head)}")
            print(f"  sequence_head modules: {list(self.sequence_head.modules())}")

            total_params = 0
            trainable_params = 0
            for name, param in self.sequence_head.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                print(f"  {name}: shape={param.shape}, numel={param.numel()}, requires_grad={param.requires_grad}")
            print(f"  Total sequence head params: {total_params}, trainable: {trainable_params}")

            # Check the linear layer specifically
            linear_layer = self.sequence_head[0]
            print(f"  Linear layer weight shape: {linear_layer.weight.shape}")
            print(f"  Linear layer weight numel: {linear_layer.weight.numel()}")
            print(f"  Linear layer bias: {linear_layer.bias}")

            # Check CLS token embedding
            cls_token_id = 70  # Updated from debug logs
            if cls_token_id < self.transformer.wte.weight.size(0):
                cls_embedding = self.transformer.wte.weight[cls_token_id]
                cls_norm = cls_embedding.norm().item()
                print(f"  CLS token (ID {cls_token_id}) embedding norm: {cls_norm:.4f}")
            else:
                print(f"  CLS token ID {cls_token_id} is out of vocab range ({self.transformer.wte.weight.size(0)})")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def log_mask_encounter_stats(self, prefix=""):
        """Log mask token encounter statistics for debugging"""
        c = self.mask_encounter_counters
        print(f"{prefix}Mask encounter stats (SEQUENCE_SCORER mode only):")
        print(f"  Training: {c['training_calls']} calls, {c['masks_seen_training']} masks in {c['total_tokens_training']} tokens")
        print(f"  Validation: {c['validation_calls']} calls, {c['masks_seen_validation']} masks in {c['total_tokens_validation']} tokens")

        if c['total_tokens_training'] > 0:
            train_mask_ratio = c['masks_seen_training'] / c['total_tokens_training']
            print(f"  Training mask ratio: {train_mask_ratio:.4f}")
        if c['total_tokens_validation'] > 0:
            val_mask_ratio = c['masks_seen_validation'] / c['total_tokens_validation']
            print(f"  Validation mask ratio: {val_mask_ratio:.4f}")
            if c['masks_seen_validation'] > 0:
                print(f"  WARNING: Validation should have 0 masks in sequence_scoring mode!")

    def reset_mask_encounter_stats(self):
        """Reset mask encounter counters"""
        for key in self.mask_encounter_counters:
            self.mask_encounter_counters[key] = 0

    def freeze_transformer_weights(self):
        """Freeze transformer weights for feature extraction, keep heads trainable"""
        print("Freezing transformer weights for feature extraction")
        for param in self.transformer.parameters():
            param.requires_grad = False
        # Keep head trainable
        if hasattr(self, 'lm_head'):
            for param in self.lm_head.parameters():
                param.requires_grad = True
        if hasattr(self, 'sequence_head'):
            for param in self.sequence_head.parameters():
                param.requires_grad = True

    def unfreeze_transformer_weights(self):
        """Unfreeze all transformer weights for fine-tuning"""
        print("Unfreezing transformer weights for fine-tuning")
        for param in self.transformer.parameters():
            param.requires_grad = True
        # Heads remain trainable
        if hasattr(self, 'lm_head'):
            for param in self.lm_head.parameters():
                param.requires_grad = True
        if hasattr(self, 'sequence_head'):
            for param in self.sequence_head.parameters():
                param.requires_grad = True

    def get_frozen_status(self):
        """Check if transformer weights are currently frozen"""
        if not hasattr(self, 'transformer'):
            return False

        # Check if any transformer parameter requires grad
        for param in self.transformer.parameters():
            if param.requires_grad:
                return False
        return True

    def print_parameter_status(self):
        """Print detailed parameter status for debugging"""
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        transformer_trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)

        head_params = 0
        head_trainable = 0
        if hasattr(self, 'lm_head'):
            head_params += sum(p.numel() for p in self.lm_head.parameters())
            head_trainable += sum(p.numel() for p in self.lm_head.parameters() if p.requires_grad)
        if hasattr(self, 'sequence_head'):
            head_params += sum(p.numel() for p in self.sequence_head.parameters())
            head_trainable += sum(p.numel() for p in self.sequence_head.parameters() if p.requires_grad)

        total_params = transformer_params + head_params
        total_trainable = transformer_trainable + head_trainable

        print(f"Parameter Status:")
        print(f"  Transformer: {transformer_trainable:,}/{transformer_params:,} trainable ({100*transformer_trainable/transformer_params:.1f}%)")
        if head_params > 0:
            print(f"  Head: {head_trainable:,}/{head_params:,} trainable ({100*head_trainable/head_params:.1f}%)")
        print(f"  Total: {total_trainable:,}/{total_params:,} trainable ({100*total_trainable/total_params:.1f}%)")
        print(f"  Frozen status: {'Frozen' if self.get_frozen_status() else 'Unfrozen'}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Track mask token encounters for debugging - ONLY for sequence_scoring mode
        if hasattr(self.config, 'mode') and str(self.config.mode) == 'ModelMode.SEQUENCE_SCORER':
            is_training_mode = self.training
            if is_training_mode:
                self.mask_encounter_counters['training_calls'] += 1
                self.mask_encounter_counters['total_tokens_training'] += b * t
            else:
                self.mask_encounter_counters['validation_calls'] += 1
                self.mask_encounter_counters['total_tokens_validation'] += b * t

            # Count mask tokens - use the actual mask_token_id = 65 from debug log
            mask_token_id = 65  # From debug log: mask_token_id = 65
            mask_count = (idx == mask_token_id).sum().item()
            if mask_count > 0:
                if is_training_mode:
                    self.mask_encounter_counters['masks_seen_training'] += mask_count
                else:
                    self.mask_encounter_counters['masks_seen_validation'] += mask_count

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        
        # Add positional embeddings only if not using RoPE
        if hasattr(self.transformer, 'wpe'):
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            # When using RoPE, no absolute position embeddings are needed
            x = self.transformer.drop(tok_emb)
            
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # Mode-specific forward pass and loss computation
        if self.config.mode == ModelMode.SEQUENCE_SCORER:
            # Extract [CLS] token representation (first token)
            cls_output = x[:, 0, :]  # (batch_size, n_embd)
            logits = self.sequence_head(cls_output).squeeze(-1)  # (batch_size,) with sigmoid applied

            if targets is not None:
                # DEBUG: Check for problematic values before loss computation
                if self.training:
                    if not hasattr(self, '_debug_iter_counter'):
                        self._debug_iter_counter = 0
                    self._debug_iter_counter += 1

                    if self._debug_iter_counter % 20 == 0:  # Log every 20 iterations
                        print(f"DEBUG: Sequence scorer values (iter {self._debug_iter_counter}):")
                        print(f"  CLS output range: [{cls_output.min().item():.4f}, {cls_output.max().item():.4f}]")
                        print(f"  CLS output mean/std: {cls_output.mean().item():.4f} ± {cls_output.std().item():.4f}")
                        print(f"  Raw linear output (pre-sigmoid): [{(self.sequence_head[0](cls_output)).min().item():.4f}, {(self.sequence_head[0](cls_output)).max().item():.4f}]")
                        print(f"  Predictions (post-sigmoid): [{logits.min().item():.6f}, {logits.max().item():.6f}]")
                        print(f"  Targets: [{targets.min().item():.6f}, {targets.max().item():.6f}]")
                        print(f"  Target mean/std: {targets.mean().item():.4f} ± {targets.std().item():.4f}")

                # MSE loss for continuous score prediction (0-1 range)
                loss = F.mse_loss(logits, targets.float())

                if self.training and self._debug_iter_counter % 20 == 0:
                    print(f"  MSE Loss: {loss.item():.6f}")

                # DEBUG: Log sequence scoring details during training (every 200 iterations)
                if self.training and self._debug_iter_counter % 200 == 0:
                        with torch.no_grad():
                            print(f"DEBUG: Sequence scorer forward pass (iter ~{self._debug_iter_counter}):")
                            print(f"  CLS output stats: mean={cls_output.mean().item():.4f}, std={cls_output.std().item():.4f}")
                            print(f"  Predictions (first 5): {logits[:5].tolist()}")
                            print(f"  Targets (first 5): {targets[:5].tolist()}")
                            print(f"  Loss: {loss.item():.6f}")
                            print(f"  Prediction stats: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
                elif not hasattr(self, '_debug_iter_counter'):
                    self._debug_iter_counter = 0
            else:
                loss = None

        elif self.config.mode in [ModelMode.LANGUAGE_MODEL, ModelMode.TOKEN_CLASSIFIER]:
            # Token-level predictions for all positions
            if targets is not None:
                # Training: compute logits for all positions
                logits = self.lm_head(x)
            else:
                # Inference: optimize based on attention type
                if self.config.mode == ModelMode.LANGUAGE_MODEL and getattr(self.config, 'attention_type', 'causal') == 'causal':
                    # Causal language modeling: only need last position for generation
                    logits = self.lm_head(x[:, [-1], :])
                else:
                    # Token classification or bidirectional: need all positions
                    logits = self.lm_head(x)

            if targets is not None:
                if self.config.mode == ModelMode.TOKEN_CLASSIFIER:
                    # Multi-class token classification with flexible number of classes
                    num_classes = self.config.num_token_classes

                    if targets.dim() == 3:
                        # Soft targets (probability distributions)
                        loss = F.cross_entropy(logits.view(-1, num_classes), targets.view(-1, num_classes))
                    else:
                        # Hard targets with optional dynamic class weighting
                        flattened_targets = targets.view(-1)
                        valid_targets = flattened_targets[flattened_targets != -1]

                        if len(valid_targets) > 0 and num_classes > 1:
                            # Dynamic class weighting for imbalanced datasets
                            unique, counts = torch.unique(valid_targets, return_counts=True)
                            n_samples = len(valid_targets)

                            class_weights = torch.zeros(num_classes, device=targets.device, dtype=logits.dtype)
                            for cls, count in zip(unique, counts):
                                if cls < num_classes:  # Ensure class index is valid
                                    class_weights[cls] = n_samples / (num_classes * count)

                            loss = F.cross_entropy(logits.view(-1, num_classes), flattened_targets,
                                                 weight=class_weights, ignore_index=-1)
                        else:
                            loss = F.cross_entropy(logits.view(-1, num_classes), flattened_targets, ignore_index=-1)
                else:
                    # Language modeling loss
                    if targets.dim() == 3:
                        # Soft targets (label smoothing)
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1, logits.size(-1)))
                    else:
                        # Hard targets
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else:
                loss = None
        else:
            raise ValueError(f"Unknown model mode: {self.config.mode}")

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        
        # Only crop position embeddings if they exist (not using RoPE)
        if hasattr(self.transformer, 'wpe'):
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
            
        for block in self.transformer.h:
            # Only causal attention has bias buffer
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
            # Update RoPE max position embeddings if using RoPE
            if hasattr(block.attn, 'rotary_emb') and block.attn.rotary_emb is not None:
                block.attn.rotary_emb.max_position_embeddings = block_size
                block.attn.rotary_emb._set_cos_sin_cache(
                    seq_len=block_size, 
                    device=block.attn.rotary_emb.inv_freq.device,
                    dtype=torch.get_default_dtype()
                )

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
