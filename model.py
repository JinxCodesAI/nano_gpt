
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
import dataclasses
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

class BidirectionalSelfAttention(nn.Module):
    """Self-attention layer that always operates bidirectionally with RoPE."""

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
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            max_position_embeddings=config.block_size,
            device=None  # Will be set when model is moved to device
        )

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

        # Apply rotary positional embeddings
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


class CrossAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, kv, kv_mask=None):
        if kv is None:
            return x

        B, T, C = x.shape
        H = self.n_head
        D = C // H

        q = self.q(x).view(B, T, H, D).transpose(1, 2)
        k = self.k(kv).view(B, kv.size(1), H, D).transpose(1, 2)
        v = self.v(kv).view(B, kv.size(1), H, D).transpose(1, 2)

        if self.flash:
            attn_mask = None
            if kv_mask is not None:
                attn_mask = (kv_mask == 0)[:, None, None, :]
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(D)
            if kv_mask is not None:
                att = att.masked_fill((kv_mask == 0)[:, None, None, :], float('-inf'))
            att = torch.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(self.dropout(y))


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = BidirectionalSelfAttention(config)

        self.cross = CrossAttention(config)
        self.ln_cross = LayerNorm(config.n_embd, bias=config.bias)

        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None, guidance_h=None, guidance_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        if guidance_h is not None:
            x = x + self.cross(self.ln_cross(x), guidance_h, kv_mask=guidance_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class PlanEncoder(nn.Module):

    def __init__(
        self,
        config,
        K=None,
        depth_factor=None,
        token_embedding=None,
        position_embedding=None,
    ):
        super().__init__()
        self.K = int(K if K is not None else getattr(config, 'plan_tokens', 16))
        if self.K <= 0:
            raise ValueError("PlanEncoder requires K > 0 plan tokens")

        self.plan_emb = nn.Parameter(torch.randn(self.K, config.n_embd) * 0.02)

        self.wte = token_embedding if token_embedding is not None else nn.Embedding(config.vocab_size, config.n_embd)

        position_encoding = getattr(config, 'position_encoding', 'absolute')
        if position_embedding is not None:
            self.wpe = position_embedding
        elif position_encoding == 'absolute':
            self.wpe = nn.Embedding(config.block_size, config.n_embd)
        else:
            self.wpe = None

        depth_factor = depth_factor if depth_factor is not None else getattr(config, 'plan_encoder_depth_factor', 0.5)
        if isinstance(depth_factor, int):
            depth = max(1, int(depth_factor))
        else:
            depth = max(1, int(round(config.n_layer * float(depth_factor))))

        enc_cfg = dataclasses.replace(config, attention_type='bidirectional', n_layer=depth)

        self.ln_in = LayerNorm(enc_cfg.n_embd, bias=enc_cfg.bias)
        self.layers = nn.ModuleList([Block(enc_cfg) for _ in range(depth)])
        self.ln_out = LayerNorm(enc_cfg.n_embd, bias=enc_cfg.bias)

    def embed(self, idx):
        tok = self.wte(idx)
        if self.wpe is not None:
            positions = torch.arange(idx.size(1), device=idx.device, dtype=torch.long)
            tok = tok + self.wpe(positions)
        return tok

    def forward(self, src_emb_or_idx, src_mask=None, already_embedded=False, return_mask=False):
        if already_embedded:
            src = src_emb_or_idx
        else:
            src = self.embed(src_emb_or_idx)

        B = src.size(0)
        plan = self.plan_emb.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([plan, src], dim=1)

        if src_mask is not None:
            plan_mask = torch.ones(B, self.K, dtype=src_mask.dtype, device=src_mask.device)
            attention_mask = torch.cat([plan_mask, src_mask], dim=1)
        else:
            attention_mask = None
            plan_mask = None

        x = self.ln_in(x)
        for block in self.layers:
            x = block(x, attention_mask=attention_mask)
        x = self.ln_out(x)
        plan_states = x[:, : self.K, :]

        if return_mask:
            if plan_mask is None:
                plan_mask = torch.ones(B, self.K, device=x.device, dtype=torch.long)
            return plan_states, plan_mask
        return plan_states

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
    use_guidance: bool = True
    plan_tokens: int = 16
    plan_encoder_depth_factor: float = 0.5
    cond_dropout_prob: float = 0.1

    # Dual-mode support: model has both heads, mode is switchable at runtime
    cls_token_id: int = None  # For sequence scoring mode

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

class GPT(nn.Module):

    def __init__(self, config, logger=None):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.logger = logger

        # Runtime mode switching (default: LANGUAGE_MODEL)
        self._current_mode = ModelMode.LANGUAGE_MODEL

        # Create transformer components. Rotary embeddings are applied inside attention.
        transformer_components = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        )

        self._log_info("Using Rotary Position Embeddings (RoPE)")

        self.transformer = nn.ModuleDict(transformer_components)

        self.plan_encoder = None
        if getattr(self.config, 'use_guidance', False):
            depth_factor = getattr(self.config, 'plan_encoder_depth_factor', 0.5)
            position_embedding = self.transformer['wpe'] if 'wpe' in self.transformer else None
            self.plan_encoder = PlanEncoder(
                self.config,
                K=getattr(self.config, 'plan_tokens', 16),
                depth_factor=depth_factor,
                token_embedding=self.transformer.wte,
                position_embedding=position_embedding,
            )
            self._log_info(f"Plan encoder enabled (K={self.plan_encoder.K}, depth_factor={depth_factor})")

        # Main head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
        # Sequence scoring head
        self.sequence_head = ScaledSigmoidHead(config.n_embd)
        # Initialize with small weights for stability
        with torch.no_grad():
            self.sequence_head.base_predictor.weight.normal_(0.0, 0.01)

        self._log_info("Dual-mode model: LANGUAGE_MODEL head (vocab_size) + SEQUENCE_SCORER head (0-1)")

        # Optional critic head for LANGUAGE_MODEL multi-tasking
        if getattr(self.config, 'add_critic_head', False):
            self.critic_head = nn.Linear(self.config.n_embd, 1, bias=False)
            self._log_info(f"Critic head enabled (alpha={self.config.critic_alpha})")

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Add dedicated CLS embedding parameter (optional, used in SEQUENCE_SCORER mode)
        if self.config.cls_token_id is not None:
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

    def set_mode(self, mode: ModelMode) -> None:
        """
        Switch the model's operational mode at runtime.

        Args:
            mode: ModelMode.LANGUAGE_MODEL or ModelMode.SEQUENCE_SCORER
        """
        if not isinstance(mode, ModelMode):
            raise TypeError(f"mode must be a ModelMode enum, got {type(mode)}")
        self._current_mode = mode

    def get_mode(self) -> ModelMode:
        """Get the current operational mode."""
        return self._current_mode

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
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _build_default_attention_mask(self, idx: torch.Tensor) -> torch.Tensor:
        
        """Build a key-padding mask that follows the attention-mask spec."""
        device = idx.device
        mask = torch.ones_like(idx, dtype=torch.long, device=device)

        pad_token_id = getattr(self.config, 'pad_token_id', None)
        if pad_token_id is not None:
            pad_id = int(pad_token_id)
            mask = mask * (idx == pad_id).long()
       
        mask_token_id = getattr(self.config, 'mask_token_id', None)
        if mask_token_id is not None:
            pad_id = int(mask_token_id)
            mask = mask | mask * (idx == pad_id).long()

        return mask

    def _encode_tokens(self, idx, attention_mask=None, guidance_h=None, guidance_mask=None):
        """Encode input token ids through the transformer trunk up to ln_f.
        Returns hidden states of shape (B, T, n_embd).
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        if attention_mask is None:
            attention_mask = self._build_default_attention_mask(idx)

        tok_emb = self.transformer.wte(idx)

        # If dedicated CLS embedding is enabled, swap it in where token == cls_token_id

        if hasattr(self, 'cls_embedding') and self.config.cls_token_id is not None:
            cls_id = int(self.config.cls_token_id)
            with torch.no_grad():
                cls_mask = (idx == cls_id).unsqueeze(-1)  # (b, t, 1)
            tok_emb = torch.where(cls_mask, self.cls_embedding.view(1, 1, -1).expand_as(tok_emb), tok_emb)

        x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask, guidance_h=guidance_h, guidance_mask=guidance_mask)
        x = self.transformer.ln_f(x)
        return x

    def forward(
        self,
        idx,
        targets=None,
        attention_mask=None,
        loss_modifiers=None,
        guidance_h=None,
        guidance_mask=None,
    ):
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

        # Build default attention mask when none provided
        if attention_mask is None:
            attention_mask = self._build_default_attention_mask(idx)

        # Encode through transformer trunk
        x = self._encode_tokens(
            idx,
            attention_mask=attention_mask,
            guidance_h=guidance_h,
            guidance_mask=guidance_mask,
        )

        # Mode-specific output and loss computation based on current runtime mode
        #print(f"[DEBUG] Model.forward: current_mode={self._current_mode}, idx.shape={idx.shape}, targets.shape={targets.shape if targets is not None else None}, targets.dtype={targets.dtype if targets is not None else None}")
        if self._current_mode == ModelMode.SEQUENCE_SCORER:
            #print(f"[DEBUG] Using SEQUENCE_SCORER forward path")
            out = self._forward_sequence_scorer(x, targets, loss_modifiers)
        else:  # LANGUAGE_MODEL
            #print(f"[DEBUG] Using LANGUAGE_MODEL forward path")
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
                    model_mode=self._current_mode
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
                    model_mode=self._current_mode
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
        """Return per-token critic logits (B, T). Requires add_critic_head=True."""
        if not getattr(self.config, 'add_critic_head', False) or not hasattr(self, 'critic_head'):
            raise RuntimeError("critic_head not enabled in config")
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size

        if attention_mask is None:
            attention_mask = self._build_default_attention_mask(idx)
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

        x = self._encode_tokens(idx, attention_mask=attention_mask)
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

        for block in self.transformer.h:
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

        # Account for RoPE overhead (8 ops per head per sequence position for Q and K,
        # applied in both forward and backward passes)
        rope_flops_per_fwdbwd = 16 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T + rope_flops_per_fwdbwd

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


def custom_mask(b, h, q_idx, kv_idx):
    W = 16  # local window size
    M = 64  # prefix size (0..63 are specials)

    # 1) Root is global both ways
    if q_idx == 0 or kv_idx == 0:
        return True

    # 2) Always allow self to avoid corner cases for specials
    if q_idx == kv_idx:
        return True

    # Helpers for parent mapping
    def parent_L3_of_normal(q):   # q >= 64
        return 4 + ((q - 64) // 16)          # in [4..63]
    def parent_L2_of_normal(q):   # q >= 64
        return 1 + ((q - 64) // 320)         # in [1..3]
    def parent_L2_of_L3(q):       # q in [4..63]
        return 1 + ((q - 4) // 20)           # in [1..3]

    # 3) L2 query: see its children (L3 & normals) + root (handled above)
    if 1 <= q_idx <= 3:
        # children L3 under this L2
        if 4 <= kv_idx <= 63:
            return ((kv_idx - 4) // 20) == (q_idx - 1)
        # children normals under this L2
        if kv_idx >= 64:
            return ((kv_idx - 64) // 320) == (q_idx - 1)
        return False

    # 4) L3 query: see its children (normals) + its parent L2 + root
    if 4 <= q_idx <= 63:
        # parent L2
        if 1 <= kv_idx <= 3:
            return kv_idx == parent_L2_of_L3(q_idx)
        # children normals under this L3
        if kv_idx >= 64:
            return ((kv_idx - 64) // 16) == (q_idx - 4)
        return False

    # 5) Normal token: local window among normals + its parents (L3,L2) + root
    if q_idx >= 64:
        # local window ONLY over normals
        in_window = (kv_idx >= max(64, q_idx - W)) and (kv_idx <= q_idx + W)
        # upward summaries it contributes to
        p3 = parent_L3_of_normal(q_idx)
        p2 = parent_L2_of_normal(q_idx)
        is_up = (kv_idx == p3) or (kv_idx == p2)
        return in_window or is_up

    # default
    return False
