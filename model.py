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
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LoRAEmbedding(nn.Module):
    """LoRA adapter for embedding layers"""
    
    def __init__(self, vocab_size, n_embd, rank, alpha):
        super().__init__()
        self.main_weight = nn.Embedding(vocab_size, n_embd)
        self.lora_A = nn.Embedding(vocab_size, rank) if rank > 0 else None
        self.lora_B = nn.Linear(rank, n_embd, bias=False) if rank > 0 else None
        self.rank = rank
        self.alpha = alpha
        
        if self.rank > 0:
            self.main_weight.requires_grad_(False)
            nn.init.normal_(self.lora_A.weight, std=0.02)
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, idx):
        main_output = self.main_weight(idx)
        if self.rank == 0 or self.lora_A is None or self.lora_B is None:
            return main_output
        lora_output = self.lora_B(self.lora_A(idx))
        return main_output + (self.alpha / self.rank) * lora_output

    def merge_and_reset(self):
        if self.rank == 0 or self.lora_A is None or self.lora_B is None:
            return
        # Correct matrix multiplication order: A @ B
        lora_update = self.lora_A.weight @ self.lora_B.weight.T
        # The result is already (vocab_size, n_embd), so no transpose is needed.
        self.main_weight.weight.data += lora_update * (self.alpha / self.rank)
        # Reset LoRA weights
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

class LoRALinear(nn.Module):
    """LoRA adapter for linear layers"""
    
    def __init__(self, in_features, out_features, rank, alpha, bias=True):
        super().__init__()
        self.main_weight = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A = nn.Linear(in_features, rank, bias=False) if rank > 0 else None
        self.lora_B = nn.Linear(rank, out_features, bias=False) if rank > 0 else None
        self.rank = rank
        self.alpha = alpha
        
        if self.rank > 0:
            self.main_weight.requires_grad_(False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        main_output = self.main_weight(x)
        if self.rank == 0 or self.lora_A is None or self.lora_B is None:
            return main_output
        lora_output = self.lora_B(self.lora_A(x))
        return main_output + (self.alpha / self.rank) * lora_output

    def merge_and_reset(self):
        if self.rank == 0 or self.lora_A is None or self.lora_B is None:
            return
        # Calculate W_0 + B @ A
        self.main_weight.weight.data += self.lora_B.weight @ self.lora_A.weight * (self.alpha / self.rank)
        # Reset LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

class RotaryPositionalEmbedding(nn.Module):
    """ Rotary Position Embedding (RoPE) implementation """
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        print("Using rotary embedings")
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin values for all positions
        self.max_seq_len_cached = 0
        self._precompute_freqs(max_position_embeddings)
    
    def _precompute_freqs(self, seq_len):
        """Precompute cos and sin values for all positions up to seq_len"""
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
            freqs = torch.outer(t, self.inv_freq)
            # Different from paper, but it uses a different permutation in parts of the final model
            freqs = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer('cos_cached', freqs.cos(), persistent=False)
            self.register_buffer('sin_cached', freqs.sin(), persistent=False)
    
    def rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q, k, seq_len=None):
        """Apply rotary position embedding to query and key tensors."""
        if seq_len is None:
            seq_len = q.shape[-2]
        
        # Ensure we have precomputed values for this sequence length
        self._precompute_freqs(seq_len)
        
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # Reshape for broadcasting: (seq_len, dim) -> (1, 1, seq_len, dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Apply rotary embeddings
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
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

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # FIX: Dynamically choose between LoRALinear and nn.Linear for the QKV projection
        if config.attn_lora_rank > 0:
            self.c_attn = LoRALinear(config.n_embd, 3 * config.n_embd,
                                     rank=config.attn_lora_rank,
                                     alpha=config.lora_alpha,
                                     bias=config.bias)
        else:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection (typically not adapted with LoRA, but could be)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # rotary embeddings
        self.use_rotary_embeddings = config.use_rotary_embeddings
        if self.use_rotary_embeddings:
            self.rotary_emb = RotaryPositionalEmbedding(
                config.n_embd // config.n_head,
                max_position_embeddings=config.rotary_max_position_embeddings,
                base=config.rotary_base
            )
        
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

        # apply rotary embeddings if enabled
        if self.use_rotary_embeddings:
            q, k = self.rotary_emb(q, k, seq_len=T)

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

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Use configurable hidden dimension, default to 4 * n_embd for backward compatibility
        hidden_dim = config.n_hidden if config.n_hidden is not None else 4 * config.n_embd
        self.c_fc    = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
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
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 57664 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
    # feed forward network parameters
    n_hidden: int = None # feed forward hidden dimension, defaults to 4 * n_embd if None
    
    # rotary embedding parameters
    use_rotary_embeddings: bool = False
    rotary_base: float = 10000.0
    rotary_max_position_embeddings: int = 2048
    
    # LoRA parameters
    embedding_mode: str = 'standard' # 'standard' or 'lora'
    embedding_rank: int = 0 # rank for embedding LoRA, 0 disables
    attn_lora_rank: int = 0 # rank for attention LoRA, 0 disables
    lora_alpha: float = 1.0 # scaling factor for LoRA layers

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # FIX: Dynamically choose the embedding layer type based on the config
        if config.embedding_mode == 'lora' and config.embedding_rank > 0:
            wte_module = LoRAEmbedding(config.vocab_size, config.n_embd,
                                       rank=config.embedding_rank,
                                       alpha=config.lora_alpha)
        else:
            wte_module = nn.Embedding(config.vocab_size, config.n_embd)

        # Build transformer components
        transformer_dict = dict(
            wte = wte_module,  # Use the dynamically chosen module
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        )
        
        # Only add position embeddings if not using rotary embeddings
        if not config.use_rotary_embeddings:
            transformer_dict['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        
        self.transformer = nn.ModuleDict(transformer_dict)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        if isinstance(self.transformer.wte, LoRAEmbedding):
            # For LoRA embeddings, tie the main_weight with lm_head
            self.transformer.wte.main_weight.weight = self.lm_head.weight
            # CRITICAL FIX: After tying, explicitly freeze the lm_head as well,
            # since it now shares the same (supposedly frozen) weight tensor.
            self.lm_head.requires_grad_(False)
        else:
            # Standard weight tying for regular embeddings
            self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.use_rotary_embeddings:
            # Only subtract position embeddings if they exist (not using rotary embeddings)
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def get_detailed_param_count(self):
        """
        Return a detailed parameter count broken down by component type.
        This version distinguishes between total and trainable parameters and correctly
        handles shared weights (weight tying) to avoid double-counting.
        """
        param_counts = {
            'total': {'total': 0, 'trainable': 0},
            'token_embeddings': {'total': 0, 'trainable': 0},
            'position_embeddings': {'total': 0, 'trainable': 0},
            'attention_layers': {'total': 0, 'trainable': 0},
            'feed_forward_layers': {'total': 0, 'trainable': 0},
            'layer_norms': {'total': 0, 'trainable': 0},
            'final_layer_norm': {'total': 0, 'trainable': 0},
            'language_model_head': {'total': 0, 'trainable': 0},
        }
        
        # Use a set to keep track of parameter IDs that have already been counted
        counted_param_ids = set()

        # Helper function to update counts for a module's parameters
        def _update_counts(module, component_name):
            for p in module.parameters():
                # Only count a parameter if its ID hasn't been seen before
                if id(p) not in counted_param_ids:
                    total_params = p.numel()
                    trainable_params = p.numel() if p.requires_grad else 0
                    
                    param_counts[component_name]['total'] += total_params
                    param_counts[component_name]['trainable'] += trainable_params
                    
                    # Add the parameter's ID to the set of counted parameters
                    counted_param_ids.add(id(p))

        # --- Count parameters component by component ---
        
        # IMPORTANT: The order matters due to weight tying.
        # We count the language_model_head first.
        _update_counts(self.lm_head, 'language_model_head')
        
        # Now count the token embeddings. If weights are tied, the main embedding
        # parameter will already be in counted_param_ids and will be skipped.
        _update_counts(self.transformer.wte, 'token_embeddings')

        # Position embeddings (if they exist)
        if not self.config.use_rotary_embeddings and hasattr(self.transformer, 'wpe'):
            _update_counts(self.transformer.wpe, 'position_embeddings')

        # Iterate through all blocks for attention, MLP, and layer norms
        for block in self.transformer.h:
            _update_counts(block.attn, 'attention_layers')
            _update_counts(block.mlp, 'feed_forward_layers')
            _update_counts(block.ln_1, 'layer_norms')
            _update_counts(block.ln_2, 'layer_norms')

        # Final layer norm
        _update_counts(self.transformer.ln_f, 'final_layer_norm')
        
        # --- Final Aggregation ---
        # Now, sum up the component counts to get the grand total.
        # This is now correct because double-counting has been prevented.
        for component in param_counts:
            if component != 'total':
                param_counts['total']['total'] += param_counts[component]['total']
                param_counts['total']['trainable'] += param_counts[component]['trainable']

        return param_counts

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
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        
        # Handle position embeddings based on rotary embedding setting
        if self.config.use_rotary_embeddings:
            # No position embeddings when using rotary embeddings
            x = self.transformer.drop(tok_emb)
        else:
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
            
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        
        # Only crop position embeddings if they exist (not using rotary embeddings)
        if not self.config.use_rotary_embeddings:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

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
        flops_promised = 121e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
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

    def stack_layers(self, num_copies):
        """
        Stack layers by duplicating the existing layers num_copies times.
        This implements the layer stacking operation for dynamic model growth.
        """
        if num_copies <= 1:
            return
        
        print(f"Stacking layers: current depth {self.config.n_layer}, creating {self.config.n_layer * num_copies} total layers.")
        original_layers = self.transformer.h
        new_layers = nn.ModuleList()
        
        for _ in range(num_copies):
            for layer in original_layers:
                new_layers.append(copy.deepcopy(layer))
        
        self.transformer.h = new_layers
        self.config.n_layer = len(new_layers)
        print(f"Model now has {self.config.n_layer} layers.")

    def widen_mlp(self, scale_factor):
        """
        Widen MLP layers using Net2WiderNet algorithm with noise injection.
        This implements the MLP widening operation for dynamic model growth.
        """
        if scale_factor <= 1:
            return
            
        print(f"Widening MLP layers by a factor of {scale_factor}.")
        new_hidden_dim = 0
        
        for block in self.transformer.h:
            mlp = block.mlp
            w_fc, b_fc, w_proj = mlp.c_fc.weight, mlp.c_fc.bias, mlp.c_proj.weight
            
            # Get the correct device from an existing parameter
            device = w_fc.device
            
            original_hidden_dim = w_fc.shape[0]
            new_hidden_dim = int(original_hidden_dim * scale_factor)

            # Create new layers on the CPU
            new_c_fc = nn.Linear(self.config.n_embd, new_hidden_dim, bias=self.config.bias)
            new_c_proj = nn.Linear(new_hidden_dim, self.config.n_embd, bias=self.config.bias)
            
            # Net2WiderNet mapping, created on the correct device
            mapping = torch.randint(0, original_hidden_dim, (new_hidden_dim,), device=device)
            mapping[:original_hidden_dim] = torch.arange(original_hidden_dim, device=device)
            
            # === FIX STARTS HERE ===
            # Move the new layers to the correct device BEFORE operating on their weights.
            new_c_fc = new_c_fc.to(device)
            new_c_proj = new_c_proj.to(device)
            # === FIX ENDS HERE ===
            
            # Now all tensors (new_c_fc.weight, w_fc, mapping) are on the same device.
            new_c_fc.weight.data.copy_(w_fc.data[mapping])
            if b_fc is not None:
                new_c_fc.bias.data.copy_(b_fc.data[mapping])
            
            # Break symmetry for new neurons by adding small noise
            if new_hidden_dim > original_hidden_dim:
                noise = torch.randn_like(new_c_fc.weight.data[original_hidden_dim:]) * 1e-4
                new_c_fc.weight.data[original_hidden_dim:] += noise
            
            # replication_factors tensor, created on the correct device
            replication_factors = torch.zeros(original_hidden_dim, device=device)
            for i in range(original_hidden_dim):
                replication_factors[i] = (mapping == i).sum()
            
            # All tensors (new_c_proj.weight, w_proj, mapping, replication_factors) are now on the same device.
            new_c_proj.weight.data.copy_(w_proj.data[:, mapping])
            new_c_proj.weight.data /= replication_factors[mapping].view(1, -1)
            
            # Assign the new, fully-formed, and device-correct layers
            mlp.c_fc = new_c_fc
            mlp.c_proj = new_c_proj
            
        self.config.n_hidden = new_hidden_dim
        print(f"MLP hidden dimension widened to {new_hidden_dim}.")

    def resize_lora_rank(self, new_rank):
        """
        Function-preserving resize of the attention LoRA rank.
        Merges existing adapter, then creates a new one with the specified rank.
        """
        new_rank = int(new_rank)  # Ensure rank is an integer
        print(f"Resizing attention LoRA rank to {new_rank}.")
        self.config.attn_lora_rank = new_rank
        device = self.lm_head.weight.device

        for block in self.transformer.h:
            # Check if the attention projection is a LoRA layer
            if not isinstance(block.attn.c_attn, LoRALinear):
                print("Warning: c_attn is not a LoRALinear layer. Skipping resize.")
                continue
            
            # 1. Merge existing knowledge into the main weight
            block.attn.c_attn.merge_and_reset()
            
            # 2. Create a new LoRA layer with the new rank
            new_c_attn = LoRALinear(
                in_features=self.config.n_embd,
                out_features=3 * self.config.n_embd,
                rank=new_rank,
                alpha=self.config.lora_alpha,
                bias=self.config.bias
            )
            
            # 3. Copy the merged main weights from the old layer to the new one
            new_c_attn.main_weight.load_state_dict(block.attn.c_attn.main_weight.state_dict())
            
            # 4. Replace the old layer with the new, resized layer
            block.attn.c_attn = new_c_attn.to(device)
        
    def resize_embedding_rank(self, new_rank):
        """
        Function-preserving resize of the embedding LoRA rank.
        Merges existing adapter, then creates a new one with the specified rank.
        """
        if not isinstance(self.transformer.wte, LoRAEmbedding):
            print("Warning: wte is not a LoRAEmbedding layer. Skipping resize.")
            return
            
        new_rank = int(new_rank)  # Ensure rank is an integer
        print(f"Resizing embedding LoRA rank to {new_rank}.")
        self.config.embedding_rank = new_rank
        device = self.lm_head.weight.device
        
        # 1. Merge existing knowledge
        self.transformer.wte.merge_and_reset()
        
        # 2. Create new module with the new rank
        new_wte = LoRAEmbedding(
            vocab_size=self.config.vocab_size,
            n_embd=self.config.n_embd,
            rank=new_rank,
            alpha=self.config.lora_alpha
        )
        
        # 3. Copy merged main weights
        new_wte.main_weight.load_state_dict(self.transformer.wte.main_weight.state_dict())
        
        # 4. Replace module and re-tie weights to the language model head
        self.transformer.wte = new_wte.to(device)
        self.transformer.wte.main_weight.weight = self.lm_head.weight
        # Re-freeze the head after re-tying to maintain parameter efficiency
        self.lm_head.requires_grad_(False)

    def merge_lora_weights(self):
        """
        Merge LoRA weights into the main weights and reset LoRA adapters.
        """
        print("Merging LoRA weights into main weights...")
        
        # Merge embedding LoRA if it exists
        if hasattr(self.transformer.wte, 'merge_and_reset'):
            self.transformer.wte.merge_and_reset()
            
        # Merge attention LoRA weights in all blocks
        for block in self.transformer.h:
            if hasattr(block.attn.c_attn, 'merge_and_reset'):
                block.attn.c_attn.merge_and_reset()
            if hasattr(block.attn.c_proj, 'merge_and_reset'):
                block.attn.c_proj.merge_and_reset()
        
        print("LoRA weights merged and reset.")
