
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

import time
from timings_singleton import get_global_timer

from sample_utils import build_critic_artifacts_from_logits

class ModelMode(Enum):
    """Defines the operational modes for the transformer model"""
    LANGUAGE_MODEL = "language_model"      # Standard language modeling
    TOKEN_CLASSIFIER = "token_classifier"  # Per-token classification
    SEQUENCE_SCORER = "sequence_scorer"    # Sequence-level 0-1 scoring

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
        position_encoding = getattr(config, 'position_encoding', 'absolute')
        if position_encoding == 'rotary':
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
            # Note: This warning occurs during model init, before logger is available
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, attention_mask=None):
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

        # Build key padding mask for attention (mask out keys where attention_mask==0)
        sdpa_attn_mask = None
        if attention_mask is not None:
            # Boolean mask where True indicates positions to mask out
            key_padding = (attention_mask == 0)  # (B, T)
            sdpa_attn_mask = key_padding[:, None, None, :]  # (B, 1, 1, T)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=sdpa_attn_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # causal mask
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # key padding mask
            if attention_mask is not None:
                key_padding = (attention_mask == 0).view(B, 1, 1, T)
                att = att.masked_fill(key_padding, float('-inf'))
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
        position_encoding = getattr(config, 'position_encoding', 'absolute')
        if position_encoding == 'rotary':
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
            # Note: This warning occurs during model init, before logger is available
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x, attention_mask=None):
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

        # Build key padding mask for attention (mask out keys where attention_mask==0)
        sdpa_attn_mask = None
        if attention_mask is not None:
            key_padding = (attention_mask == 0)  # (B, T)
            sdpa_attn_mask = key_padding[:, None, None, :]  # (B, 1, 1, T)

        # bidirectional self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=sdpa_attn_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # key padding mask
            if attention_mask is not None:
                key_padding = (attention_mask == 0).view(B, 1, 1, T)
                att = att.masked_fill(key_padding, float('-inf'))
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

class ScaledSigmoidHead(nn.Module):
    """
    Linear head with a learnable affine transformation before the sigmoid
    to control the output distribution.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.base_predictor = nn.Linear(input_dim, 1, bias=False)
        # Learnable scale (acts like temperature) and shift for the sigmoid input
        self.A = nn.Parameter(torch.ones(1))
        self.B = nn.Parameter(torch.zeros(1))
        self.temperature = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        logits = self.base_predictor(x)
        # Apply learnable scale and shift to the logits before the sigmoid
        scaled_logits = logits * self.A + self.B
        return torch.sigmoid(scaled_logits)


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)

        # Choose attention type based on config
        attention_type = getattr(config, 'attention_type', 'causal')
        if attention_type == 'bidirectional':
            self.attn = BidirectionalSelfAttention(config)
            # Note: These messages occur during model init, before logger is available
            print("Using bidirectional attention")
        else:
            self.attn = CausalSelfAttention(config)
            # Note: These messages occur during model init, before logger is available
            print("Using causal attention")

        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
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
    ignore_index: int = -100 # Standard PyTorch ignore index for loss computation
    attention_type: str = 'causal' # 'causal' for autoregressive, 'bidirectional' for BERT-style
    position_encoding: str = 'absolute' # 'absolute' or 'rotary'

    # Multi-mode support
    mode: ModelMode = ModelMode.LANGUAGE_MODEL
    num_token_classes: int = 2  # For token classification
    cls_token_id: int = None  # For sequence scoring

    # Transfer learning support
    freeze_transformer: bool = False
    init_from_checkpoint: str = None
    unfreeze_at_iteration: int = None
    unfreeze_lr_multiplier: float = 0.1


    # Optional critic head configuration (LANGUAGE_MODEL multi-task)
    add_critic_head: bool = False
    critic_alpha: float = 0.5
    critic_target_scope: str = 'masked_and_ignore'
    mask_token_id: int = None
    pad_token_id: int = None
    # Critic alpha warmup
    start_critic_iteration: int = 0
    end_critic_iteration: int = 0


    # Backward compatibility
    binary_classification: bool = False  # Legacy support

    def __post_init__(self):
        # Handle backward compatibility
        if self.binary_classification and self.mode == ModelMode.LANGUAGE_MODEL:
            self._log_warning("binary_classification=True detected, converting to TOKEN_CLASSIFIER mode")
            self.mode = ModelMode.TOKEN_CLASSIFIER
            self.num_token_classes = 2

        # Enforce bidirectional attention for classification tasks
        if self.mode in [ModelMode.TOKEN_CLASSIFIER, ModelMode.SEQUENCE_SCORER]:
            if self.attention_type != 'bidirectional':
                self._log_warning(f"{self.mode.value} requires bidirectional attention. Changing from '{self.attention_type}' to 'bidirectional'")
                self.attention_type = 'bidirectional'

    def _log_warning(self, message):
        """Log warning message - will be enhanced when logger is available"""
        print(f"WARNING: {message}")

class GPT(nn.Module):

    def __init__(self, config, logger=None):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.logger = logger

        # Create transformer components - conditionally include position embeddings
        transformer_components = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        )

        # Only add absolute position embeddings if not using RoPE
        if config.position_encoding == 'absolute':
            transformer_components['wpe'] = nn.Embedding(config.block_size, config.n_embd)
            self._log_info("Using absolute position embeddings")
        else:
            self._log_info("Using Rotary Position Embeddings (RoPE)")

        self.transformer = nn.ModuleDict(transformer_components)

        # Create mode-specific output heads
        if self.config.mode == ModelMode.LANGUAGE_MODEL:
            # Existing language modeling head
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            # with weight tying when using torch.compile() some warnings get generated:
            # "UserWarning: functional_call was passed multiple values for tied weights.
            # This behavior is deprecated and will be an error in future versions"
            # not 100% sure what this is, so far seems to be harmless. TODO investigate
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        elif self.config.mode == ModelMode.TOKEN_CLASSIFIER:
            # Token-level classification head
            self.lm_head = nn.Linear(config.n_embd, config.num_token_classes, bias=False)
            self._log_info(f"Token classifier head: {config.num_token_classes} classes per token")
        elif self.config.mode == ModelMode.SEQUENCE_SCORER:
            # Sequence-level scoring head with learnable temperature scaling
            self.sequence_head = ScaledSigmoidHead(config.n_embd)
            # Initialize with small weights for stability
            with torch.no_grad():
                self.sequence_head.base_predictor.weight.normal_(0.0, 0.01)
            self._log_info("Sequence scorer head: ScaledSigmoidHead (0-1)")
        else:
            raise ValueError(f"Unknown model mode: {self.config.mode}")

        # Optional critic head for LANGUAGE_MODEL multi-tasking
        if getattr(self.config, 'add_critic_head', False) and self.config.mode == ModelMode.LANGUAGE_MODEL:
            self.critic_head = nn.Linear(self.config.n_embd, 1, bias=False)
            self._log_info(f"Critic head enabled (alpha={self.config.critic_alpha})")

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Add dedicated CLS embedding parameter (optional, used in SEQUENCE_SCORER)
        if self.config.mode == ModelMode.SEQUENCE_SCORER and self.config.cls_token_id is not None:
            # A standalone parameter so freezing the transformer does not freeze CLS
            self.cls_embedding = nn.Parameter(torch.empty(self.config.n_embd))
            with torch.no_grad():
                torch.nn.init.normal_(self.cls_embedding, mean=0.0, std=0.02)
            self._log_info("Dedicated CLS embedding enabled")

        # Transfer learning: load pretrained weights if specified
        if config.init_from_checkpoint is not None and config.init_from_checkpoint != "":
            self._load_pretrained_checkpoint(config.init_from_checkpoint)

        # Transfer learning: freeze transformer if requested
        if config.freeze_transformer:
            self.freeze_transformer_weights()

        # report number of parameters
        self._log_info("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _log_info(self, message):
        """Log info message using logger if available, otherwise print."""
        if self.logger:
            self.logger.log_info(message)
        else:
            print(message)

    def _load_pretrained_checkpoint(self, checkpoint_path):
        """Load pretrained weights (transformer only), with safe embedding expansion"""
        self._log_info(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        state_dict = checkpoint.get('model', checkpoint)

        # Clean keys (remove torch.compile prefix)
        cleaned_state = {}
        for k, v in state_dict.items():
            clean_key = k.replace('_orig_mod.', '')
            cleaned_state[clean_key] = v

        # Load all transformer weights except the token embedding first
        transformer_state_dict = {}
        for k, v in cleaned_state.items():
            if k.startswith('transformer.') and not k.startswith('lm_head') and not k.startswith('sequence_head'):
                if k == 'transformer.wte.weight':
                    continue  # handle separately due to potential shape mismatch
                transformer_state_dict[k] = v

        missing_keys, unexpected_keys = self.load_state_dict(transformer_state_dict, strict=False)
        self._log_info(f"Loaded transformer (except embeddings): {len(missing_keys)} missing, {len(unexpected_keys)} unexpected")

        # Now handle token embeddings safely
        wte_key = 'transformer.wte.weight'
        if wte_key in cleaned_state:
            ckpt_wte = cleaned_state[wte_key]
            model_wte = self.transformer.wte.weight
            if ckpt_wte.dim() != 2 or model_wte.dim() != 2:
                raise RuntimeError("Unexpected token embedding dimensionality")
            old_vocab, old_dim = ckpt_wte.shape
            new_vocab, new_dim = model_wte.shape
            if old_dim != new_dim:
                raise RuntimeError(f"Embedding dimension mismatch: ckpt {old_dim} vs model {new_dim}")
            if old_vocab <= new_vocab:
                # copy overlapping rows, keep new rows as initialized
                with torch.no_grad():
                    model_wte[:old_vocab].copy_(ckpt_wte)
                added = new_vocab - old_vocab
                if added > 0:
                    self._log_info(f"Extended token embeddings by {added} new token(s)")
            else:
                raise RuntimeError(f"Checkpoint vocab ({old_vocab}) > model vocab ({new_vocab}); cannot load safely")
        else:
            self._log_info("No token embedding found in checkpoint; using initialized embeddings")

    def freeze_transformer_weights(self):
        """Freeze transformer for feature extraction"""
        self._log_info("Freezing transformer weights for feature extraction")
        for param in self.transformer.parameters():
            param.requires_grad = False
        # Keep heads trainable
        if hasattr(self, 'lm_head'):
            for param in self.lm_head.parameters():
                param.requires_grad = True
        if hasattr(self, 'sequence_head'):
            for param in self.sequence_head.parameters():
                param.requires_grad = True
        if hasattr(self, 'critic_head'):
            for param in self.critic_head.parameters():
                param.requires_grad = True


    def unfreeze_transformer_weights(self):
        """Unfreeze transformer for full fine-tuning"""
        self._log_info("Unfreezing transformer weights for fine-tuning")

        for param in self.transformer.parameters():
            param.requires_grad = True


    def extend_optimizer_with_unfrozen(self, optimizer, weight_decay=None, learning_rate=None):
        """After unfreezing transformer, add newly-trainable params to optimizer.
        - If learning_rate is None: infer from existing optimizer groups (median lr or optimizer.defaults['lr']).
        - If weight_decay is None: infer positive WD from existing groups (median over >0 values), else 0.0.
        Mirrors configure_optimizers grouping (dim>=2 -> weight decay; else no decay).
        """
        try:
            # infer LR if not provided
            if learning_rate is None:
                lrs = [pg.get('lr') for pg in optimizer.param_groups if pg.get('lr', None) is not None]
                if lrs:
                    lrs_sorted = sorted(lrs)
                    learning_rate = lrs_sorted[len(lrs_sorted)//2]
                else:
                    learning_rate = optimizer.defaults.get('lr', 0.0)
            # infer WD if not provided
            if weight_decay is None:
                wds = [pg.get('weight_decay', 0.0) for pg in optimizer.param_groups if pg.get('weight_decay', 0.0) > 0.0]
                if wds:
                    wds_sorted = sorted(wds)
                    weight_decay = wds_sorted[len(wds_sorted)//2]
                else:
                    weight_decay = 0.0

            existing = {id(p) for g in optimizer.param_groups for p in g.get('params', [])}
            new_decay, new_nodecay = [], []
            for name, p in self.named_parameters():
                if p.requires_grad and id(p) not in existing:
                    (new_decay if p.dim() >= 2 else new_nodecay).append(p)
            if len(new_decay) > 0:
                optimizer.add_param_group({
                    'params': new_decay,
                    'weight_decay': weight_decay,
                    'lr': learning_rate,
                })
            if len(new_nodecay) > 0:
                optimizer.add_param_group({
                    'params': new_nodecay,
                    'weight_decay': 0.0,
                    'lr': learning_rate,
                })
            if (len(new_decay) + len(new_nodecay)) > 0:
                self._log_info(f"Added {len(new_decay)} decayed and {len(new_nodecay)} non-decayed param tensors to optimizer after unfreeze (lr={learning_rate}, wd={weight_decay})")
        except Exception as e:
            self._log_warning(f"Failed to extend optimizer after unfreeze: {e}")

    def get_frozen_status(self):
        """Check if transformer is frozen"""
        for param in self.transformer.parameters():
            if param.requires_grad:
                return False
        return True

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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _encode_tokens(self, idx, attention_mask=None):
        """Encode input token ids through the transformer trunk up to ln_f.
        Returns hidden states of shape (B, T, n_embd).
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)

        # If dedicated CLS embedding is enabled, swap it in where token == cls_token_id

        if hasattr(self, 'cls_embedding') and self.config.cls_token_id is not None:
            cls_id = int(self.config.cls_token_id)
            with torch.no_grad():
                cls_mask = (idx == cls_id).unsqueeze(-1)  # (b, t, 1)
            tok_emb = torch.where(cls_mask, self.cls_embedding.view(1, 1, -1).expand_as(tok_emb), tok_emb)

        # Add positional embeddings only if not using RoPE
        if hasattr(self.transformer, 'wpe'):
            pos = torch.arange(0, t, dtype=torch.long, device=device)  # (t)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask)
        x = self.transformer.ln_f(x)
        return x

    def forward(self, idx, targets=None, attention_mask=None, loss_modifiers=None):
        # Optional global timing start
        _timer = None
        try:
            _timer = get_global_timer()
        except Exception:
            _timer = None
        if _timer is not None and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        _t0 = time.perf_counter()

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Encode through transformer trunk
        x = self._encode_tokens(idx, attention_mask=attention_mask)

        # Mode-specific output and loss computation
        if self.config.mode == ModelMode.SEQUENCE_SCORER:
            out = self._forward_sequence_scorer(x, targets, loss_modifiers)
        elif self.config.mode == ModelMode.TOKEN_CLASSIFIER:
            out = self._forward_token_classifier(x, targets, loss_modifiers)
        else:  # LANGUAGE_MODEL
            out = self._forward_language_model(x, targets, loss_modifiers, idx=idx)

        # Record forward timing if a global timer is registered
        try:
            if _timer is not None:
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                _timer.record('model.forward', time.perf_counter() - _t0)
        except Exception:
            pass
        return out

    def _forward_sequence_scorer(self, x, targets, loss_modifiers):
        """Sequence scoring forward pass"""
        cls_output = x[:, 0, :]
        logits = self.sequence_head(cls_output).squeeze(-1)
        if targets is not None:

            base_loss = F.mse_loss(logits, targets.float())

            # Apply loss modifiers if available and compatible with sequence scoring
            if loss_modifiers is not None and not loss_modifiers.is_empty():
                # Note: Some modifiers may not be applicable to sequence scoring
                loss = loss_modifiers.modify_loss(
                    logits.unsqueeze(-1), targets, base_loss,
                    model_mode=self.config.mode
                )
            else:
                loss = base_loss
        else:
            loss = None

        return logits, loss

    def _forward_token_classifier(self, x, targets, loss_modifiers):
        """Token classification forward pass"""
        logits = self.lm_head(x)

        if targets is not None:
            num_classes = self.config.num_token_classes
            if targets.dim() == 3:  # Soft targets
                base_loss = F.cross_entropy(logits.view(-1, num_classes), targets.view(-1, num_classes))
            else:  # Hard targets with dynamic weighting
                base_loss = self._compute_weighted_classification_loss(logits, targets, num_classes)

            # Apply loss modifiers if available
            if loss_modifiers is not None and not loss_modifiers.is_empty():
                mask = targets != -1  # Valid token mask
                loss = loss_modifiers.modify_loss(
                    logits, targets, base_loss, mask=mask,
                    ignore_index=-1, model_mode=self.config.mode
                )
            else:
                loss = base_loss
        else:
            loss = None

        return logits, loss

    def _forward_language_model(self, x, targets, loss_modifiers, idx=None):
        """Language modeling forward pass with optional critic head"""
        if targets is not None:
            logits = self.lm_head(x)
            if loss_modifiers is not None and not loss_modifiers.is_empty():
                # Existing loss modifier logic
                flat_logits = logits.view(-1, logits.size(-1))
                flat_targets = targets.view(-1)
                per_position_loss = F.cross_entropy(
                    flat_logits, flat_targets,
                    ignore_index=self.config.ignore_index,
                    reduction='none'
                )
                per_position_loss = per_position_loss.view(targets.size(0), targets.size(1))
                mask = targets != self.config.ignore_index
                base_loss = (per_position_loss * mask.float()).sum() / (mask.float().sum() + 1e-8)
                loss = loss_modifiers.modify_loss(
                    logits, targets, base_loss, mask=mask,
                    per_position_loss=per_position_loss,
                    ignore_index=self.config.ignore_index,
                    model_mode=self.config.mode
                )
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.config.ignore_index)

            # Expose LM loss component for external logging (detach to avoid graph retention)
            try:
                self._last_lm_loss = float(loss.detach().item())
            except Exception:
                self._last_lm_loss = 0.0
            # Default critic component to 0.0; may be overwritten below when enabled
            self._last_critic_loss = 0.0


            # Optional critic loss (multi-task) when enabled
            alpha_eff = self._effective_critic_alpha()
            if getattr(self.config, 'add_critic_head', False) and hasattr(self, 'critic_head') and alpha_eff > 0.0:
                # Build critic artifacts using shared helper
                if idx is None or getattr(self.config, 'mask_token_id', None) is None:
                    raise RuntimeError("critic_target_scope requires idx and mask_token_id; misconfiguration detected")
                artifacts = build_critic_artifacts_from_logits(
                    idx=idx,
                    logits=logits,
                    targets=targets,
                    mask_token_id=int(self.config.mask_token_id),
                    ignore_index=int(self.config.ignore_index),

                    pad_token_id=getattr(self.config, 'pad_token_id', None),
                    scope=getattr(self.config, 'critic_target_scope', 'masked_and_ignore'),
                )
                critic_input = artifacts['critic_input']
                critic_target = artifacts['critic_target']
                critic_valid = artifacts['critic_valid']

                # Encode critic input through the transformer trunk
                h2 = self._encode_tokens(critic_input)
                critic_logits = self.critic_head(h2).squeeze(-1)

                critic_loss_per_pos = F.binary_cross_entropy_with_logits(critic_logits, critic_target, reduction='none')
                denom = (critic_valid.float().sum() + 1e-8)
                critic_loss = (critic_loss_per_pos * critic_valid.float()).sum() / denom
                loss = loss + float(alpha_eff) * critic_loss

                # Expose critic loss component for external logging (after computing critic_loss)
                try:
                    self._last_critic_loss = float(critic_loss.detach().item())
                except Exception:
                    self._last_critic_loss = 0.0

        else:
            # Inference optimization
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
    @torch.no_grad()
    def critic_scores(self, idx, attention_mask=None):
        """Return per-token critic logits (B, T). Requires add_critic_head=True and LANGUAGE_MODEL mode."""
        if not getattr(self.config, 'add_critic_head', False) or not hasattr(self, 'critic_head'):
            raise RuntimeError("critic_head not enabled in config")
        if self.config.mode != ModelMode.LANGUAGE_MODEL:
            raise RuntimeError("critic_scores is only available in LANGUAGE_MODEL mode")
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        # Optional global timing start for critic_scores
        _timer = None
        try:
            _timer = get_global_timer()
        except Exception:
            _timer = None
        if _timer is not None and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        _t0 = time.perf_counter()

        tok_emb = self.transformer.wte(idx)
        if hasattr(self.transformer, 'wpe'):
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:

            x = block(x, attention_mask=attention_mask)
        x = self.transformer.ln_f(x)
        logits = self.critic_head(x).squeeze(-1)
        # Record timing if global timer
        try:
            if _timer is not None:
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                _timer.record('model.critic_scores', time.perf_counter() - _t0)
        except Exception:
            pass
        return logits


    def _compute_weighted_classification_loss(self, logits, targets, num_classes):
        """Compute classification loss with dynamic class weighting for imbalanced datasets"""
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

            loss = F.cross_entropy(
                logits.view(-1, num_classes), flattened_targets,
                weight=class_weights, ignore_index=-1
            )
        else:
            loss = F.cross_entropy(
                logits.view(-1, num_classes), flattened_targets, ignore_index=-1
            )

        return loss

    def _effective_critic_alpha(self) -> float:
        """Compute iteration-based effective critic alpha with linear warmup.
        Trainer should set self._current_iter each iteration (including eval).
        Schedule: 0 until start; linear to base alpha by end; clamp [0, base].
        If end <= start or no iteration provided, return base (no warmup).
        """
        base = float(getattr(self.config, 'critic_alpha', 0.0) or 0.0)
        start = int(getattr(self.config, 'start_critic_iteration', 0) or 0)
        end = int(getattr(self.config, 'end_critic_iteration', 0) or 0)
        it = getattr(self, '_current_iter', None)
        if it is None or end <= start:
            return base
        if it < start:
            return 0.0
        if it >= end:
            return base
        frac = (float(it - start) / max(1.0, float(end - start)))
        return max(0.0, min(base, base * frac))

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
        # Note: This is a class method, so we use print directly as no logger instance is available
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        # Note: This is a class method, so we use print directly as no logger instance is available
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            # Note: This is a class method, so we use print directly as no logger instance is available
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
        self._log_info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        self._log_info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        self._log_info(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size

        # Standard transformer FLOPS
        flops_per_token = 6*N + 12*L*H*Q*T

        # Add RoPE overhead if using rotary position encoding
        if getattr(cfg, 'position_encoding', 'absolute') == 'rotary':
            # RoPE operations: 8 ops per head per sequence position (4 ops each for Q and K)
            # Applied in forward and backward pass: 16 * L * H * Q * T
            rope_flops_per_fwdbwd = 16 * L * H * Q * T
            flops_per_fwdbwd = flops_per_token * T + rope_flops_per_fwdbwd
        else:
            flops_per_fwdbwd = flops_per_token * T

        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 150e12 # A40 GPU bfloat16 peak flops is 150 TFLOPS
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

