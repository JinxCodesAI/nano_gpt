"""Discrete diffusion Transformer backbone used throughout the repository."""

import math
import inspect
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
from torch.nn import functional as F

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

class LoRALinear(nn.Module):
    """Lightweight adapter that defaults to no-op when disabled."""

    def __init__(self, in_features, out_features, r=8, alpha=16.0, dropout=0.0, enabled=False):
        super().__init__()
        self.enabled = bool(enabled and r > 0)
        if not self.enabled:
            # keep module in the tree without parameters; eases conditional logic
            self.register_buffer("_disabled", torch.tensor(0.0), persistent=False)
            return
        self.r = int(r)
        self.scaling = float(alpha) / float(self.r)
        self.A = nn.Linear(in_features, self.r, bias=False)
        self.B = nn.Linear(self.r, out_features, bias=False)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        if not self.enabled:
            return 0.0
        return self.B(self.A(self.drop(x))) * self.scaling

class FiLMAdapterLR(nn.Module):
    """Low-rank FiLM adapter that maps global context into per-channel scale and shift."""

    def __init__(self, g_dim, c_dim, rank=8, bias=False, scale=1.0):
        super().__init__()
        r = max(1, int(rank))
        self.scale = float(scale)
        self.down = nn.Linear(g_dim, r, bias=bias)
        self.up = nn.Linear(r, 2 * c_dim, bias=bias)

    def forward(self, g):
        gamma_beta = self.up(self.down(g)) * self.scale
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma, beta

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
        
        # Rotary positional embeddings are required for discrete diffusion experiments.
        self.head_dim = config.n_embd // config.n_head
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            max_position_embeddings=config.block_size,
            device=None  # Will be set when model is moved to device
        )

        lora_kwargs = dict(
            r=getattr(config, 'lora_rank', 0),
            alpha=getattr(config, 'lora_alpha', 1.0),
            dropout=getattr(config, 'lora_dropout', 0.0),
            enabled=getattr(config, 'use_lora_attn', False),
        )
        self.lora_qkv = LoRALinear(config.n_embd, 3 * config.n_embd, **lora_kwargs)
        self.lora_out = LoRALinear(config.n_embd, config.n_embd, **lora_kwargs)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv_linear = self.c_attn(x)
        if self.lora_qkv.enabled:
            qkv_linear = qkv_linear + self.lora_qkv(x)
        q, k, v  = qkv_linear.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

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
        proj = self.c_proj(y)
        if self.lora_out.enabled:
            proj = proj + self.lora_out(y)
        y = self.resid_dropout(proj)
        return y

class Encoder(nn.Module):
    """
    Lightweight bidirectional encoder that produces a single global vector via [CLS].
    """

    def __init__(self, config: 'GPTConfig'):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.enc_n_embd)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.enc_n_embd))
        self.drop = nn.Dropout(config.enc_dropout)

        encoder_cfg = replace(
            config,
            n_layer=config.enc_n_layer,
            n_head=config.enc_n_head,
            n_embd=config.enc_n_embd,
            dropout=config.enc_dropout,
            use_lora_attn=config.enc_use_lora_attn,
            use_lora_mlp=config.enc_use_lora_mlp,
            lora_rank=config.enc_lora_rank,
            lora_alpha=config.enc_lora_alpha,
            lora_dropout=config.enc_lora_dropout,
            use_encoder_guidance=False,
            film_rank=0,
        )
        self.h = nn.ModuleList([Block(encoder_cfg) for _ in range(config.enc_n_layer)])
        self.ln_f = LayerNorm(config.enc_n_embd, bias=config.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError("Encoder expects input of shape (batch, sequence_length)")
        b, _ = idx.size()
        tok_emb = self.wte(idx)
        cls_tok = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tok, tok_emb), dim=1)
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return x[:, 0]

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        lora_kwargs = dict(
            r=getattr(config, 'lora_rank', 0),
            alpha=getattr(config, 'lora_alpha', 1.0),
            dropout=getattr(config, 'lora_dropout', 0.0),
            enabled=getattr(config, 'use_lora_mlp', False),
        )
        self.lora_fc = LoRALinear(config.n_embd, 4 * config.n_embd, **lora_kwargs)
        self.lora_proj = LoRALinear(4 * config.n_embd, config.n_embd, **lora_kwargs)

    def forward(self, x):
        fc_out = self.c_fc(x)
        if self.lora_fc.enabled:
            fc_out = fc_out + self.lora_fc(x)
        hidden = self.gelu(fc_out)
        proj_out = self.c_proj(hidden)
        if self.lora_proj.enabled:
            proj_out = proj_out + self.lora_proj(hidden)
        proj_out = self.dropout(proj_out)
        return proj_out

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = BidirectionalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        self.film_attn = None
        self.film_mlp = None
        if getattr(config, 'use_encoder_guidance', False):
            film_rank = int(getattr(config, 'film_rank', 0) or 0)
            if film_rank > 0:
                g_dim = getattr(config, 'enc_n_embd', config.n_embd)
                scale = float(getattr(config, 'guidance_scale', 1.0))
                self.film_attn = FiLMAdapterLR(g_dim, config.n_embd, rank=film_rank, scale=scale)
                self.film_mlp = FiLMAdapterLR(g_dim, config.n_embd, rank=film_rank, scale=scale)

    def forward(self, x, g=None):
        attn_in = self.ln_1(x)
        if self.film_attn is not None and g is not None:
            gamma, beta = self.film_attn(g)
            gamma = gamma.to(attn_in.dtype).unsqueeze(1)
            beta = beta.to(attn_in.dtype).unsqueeze(1)
            attn_in = attn_in * (1 + gamma) + beta
        x = x + self.attn(attn_in)

        mlp_in = self.ln_2(x)
        if self.film_mlp is not None and g is not None:
            gamma, beta = self.film_mlp(g)
            gamma = gamma.to(mlp_in.dtype).unsqueeze(1)
            beta = beta.to(mlp_in.dtype).unsqueeze(1)
            mlp_in = mlp_in * (1 + gamma) + beta
        x = x + self.mlp(mlp_in)
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms. False: a bit better and faster
    ignore_index: int = -100 # Standard PyTorch ignore index for loss computation
    use_lora_attn: bool = False
    use_lora_mlp: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    use_encoder_guidance: bool = False
    enc_n_layer: int = 1
    enc_n_head: int = 4
    enc_n_embd: int = 256
    enc_dropout: float = 0.0
    enc_use_lora_attn: bool = False
    enc_use_lora_mlp: bool = False
    enc_lora_rank: int = 8
    enc_lora_alpha: float = 16.0
    enc_lora_dropout: float = 0.0
    film_rank: int = 0
    guidance_scale: float = 1.0

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Create transformer components
        transformer_components = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        )

        self.transformer = nn.ModuleDict(transformer_components)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        self.encoder = None
        if config.use_encoder_guidance:
            self.encoder = Encoder(config)

        # share projection matrices across blocks when LoRA is active
        share_attn = getattr(config, 'use_lora_attn', False)
        share_mlp = getattr(config, 'use_lora_mlp', False)
        if (share_attn or share_mlp) and len(self.transformer.h) > 1:
            base_block = self.transformer.h[0]
            for block in self.transformer.h[1:]:
                if share_attn:
                    block.attn.c_attn.weight = base_block.attn.c_attn.weight
                    block.attn.c_attn.bias = base_block.attn.c_attn.bias
                    block.attn.c_proj.weight = base_block.attn.c_proj.weight
                    block.attn.c_proj.bias = base_block.attn.c_proj.bias
                if share_mlp:
                    block.mlp.c_fc.weight = base_block.mlp.c_fc.weight
                    block.mlp.c_fc.bias = base_block.mlp.c_fc.bias
                    block.mlp.c_proj.weight = base_block.mlp.c_proj.weight
                    block.mlp.c_proj.bias = base_block.mlp.c_proj.bias

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, g=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)

        g_ctx = None
        if self.encoder is not None:
            if g is None:
                if targets is not None:
                    enc_x = torch.where(targets != self.config.ignore_index, targets, idx)
                else:
                    enc_x = idx
                g_ctx = self.encoder(enc_x)
            else:
                g_ctx = g
            if g_ctx.dim() != 2 or g_ctx.size(0) != b:
                raise ValueError("Global context g must be of shape (batch, enc_n_embd)")

        for block in self.transformer.h:
            x = block(x, g_ctx)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.config.ignore_index)
        else:
            loss = None

        return logits, loss

    @torch.no_grad()
    def encode(self, enc_x: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("Encoder guidance is disabled in config.")
        return self.encoder(enc_x)

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        
        for block in self.transformer.h:
            block.attn.rotary_emb.max_position_embeddings = block_size
            block.attn.rotary_emb._set_cos_sin_cache(
                seq_len=block_size,
                device=block.attn.rotary_emb.inv_freq.device,
                dtype=torch.get_default_dtype()
            )

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
        flops_promised = 150e12 # A40 GPU bfloat16 peak flops is 150 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
