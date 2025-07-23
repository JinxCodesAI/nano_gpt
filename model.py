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
        # Always create this buffer as we need it for return_attention=True even with flash attention
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, return_attention=False):
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
        att_scores = None # Initialize as None
        if self.flash:
            # Problem: Flash Attention doesn't return attention scores by default.
            # We must use the manual path when analysis is needed.
            if return_attention:
                # Fallback to manual implementation for analysis
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att_scores = att # Capture scores
                y = att @ v
            else:
                # Default fast path
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att_scores = att # Capture scores
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        if return_attention:
            return y, att_scores
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

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_output, attn_scores = self.attn(self.ln_1(x), return_attention=True)
            x = x + attn_output
            x = x + self.mlp(self.ln_2(x))
            return x, attn_scores
        else:
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
        self.embedding_finetune_mode = False # Flag for embedding fine-tuning mode

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

    def forward(self, idx, targets=None, return_attention=False):
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

        all_attention_scores = []
        for block in self.transformer.h:
            if return_attention:
                x, attn_scores = block(x, return_attention=True)
                all_attention_scores.append(attn_scores)
            else:
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

        # Return the collected scores if requested
        if return_attention:
            return logits, loss, all_attention_scores

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

        # If in embedding finetune mode, only optimize the embedding and lm_head layers
        if self.embedding_finetune_mode:
            print("Optimizer configured for embedding fine-tuning mode.")
            param_dict = {pn: p for pn, p in param_dict.items() if 'wte' in pn or 'lm_head' in pn}
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
        flops_promised = 121e12 # L4 GPU bfloat16 peak flops is 312 TFLOPS
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

    def stack_layers(self, layer_map):
        """
        Reconstructs the transformer stack based on an explicit layer map.
        The map is a list of source indices. e.g., [0, 0, 1] creates a 3-layer model
        with two copies of the original layer 0 and one copy of the original layer 1.

        Throws a ValueError if any source index is out of bounds.
        """
        print(f"Re-stacking layers based on map: {layer_map}. New depth will be {len(layer_map)}.")
        original_n_layer = self.config.n_layer

        # --- Validation First ---
        if not layer_map: # Cannot stack to zero layers this way
            raise ValueError("Layer map cannot be empty.")
        if min(layer_map) < 0:
            raise ValueError(f"Invalid layer map: negative index {min(layer_map)} is not allowed.")
        if max(layer_map) >= original_n_layer:
            raise ValueError(f"Invalid layer map: index {max(layer_map)} is out of bounds for current model with {original_n_layer} layers.")

        # Deepcopy original layers to use as a clean source palette
        original_layers = copy.deepcopy(self.transformer.h)
        new_layers = nn.ModuleList()

        # Build the new stack layer by layer
        for source_idx in layer_map:
            new_layers.append(copy.deepcopy(original_layers[source_idx]))

        self.transformer.h = new_layers
        self.config.n_layer = len(new_layers)
        print(f"Model now has {self.config.n_layer} layers.")

    def widen_mlp(self, new_hidden_dim):
        """
        Widens MLP layers to a new absolute dimension using Net2WiderNet.
        Throws a ValueError if the new dimension is not larger than the current one.
        """
        print(f"Widening MLP hidden dimension to {new_hidden_dim}.")

        # Validation is the first step
        original_hidden_dim = self.config.n_hidden if self.config.n_hidden is not None else 4 * self.config.n_embd
        if not new_hidden_dim > original_hidden_dim:
            raise ValueError(f"New hidden dimension ({new_hidden_dim}) must be greater than the original ({original_hidden_dim}).")

        for block in self.transformer.h:
            mlp = block.mlp
            w_fc, b_fc, w_proj = mlp.c_fc.weight, mlp.c_fc.bias, mlp.c_proj.weight
            device = w_fc.device

            new_c_fc = nn.Linear(self.config.n_embd, new_hidden_dim, bias=self.config.bias).to(device)
            new_c_proj = nn.Linear(new_hidden_dim, self.config.n_embd, bias=self.config.bias).to(device)

            # Net2WiderNet mapping
            mapping = torch.randint(0, original_hidden_dim, (new_hidden_dim,), device=device)
            mapping[:original_hidden_dim] = torch.arange(original_hidden_dim, device=device)

            # Copy weights for the first part of the mapping
            new_c_fc.weight.data.copy_(w_fc.data[mapping])
            if b_fc is not None:
                new_c_fc.bias.data.copy_(b_fc.data[mapping])

            # Add noise to break symmetry for the new neurons
            noise = torch.randn_like(new_c_fc.weight.data[original_hidden_dim:]) * 1e-4
            new_c_fc.weight.data[original_hidden_dim:] += noise

            # Calculate replication factors for adjusting the output layer
            replication_factors = torch.zeros(original_hidden_dim, device=device)
            for i in range(original_hidden_dim):
                replication_factors[i] = (mapping == i).sum()

            # Copy and scale the output projection weights
            new_c_proj.weight.data.copy_(w_proj.data[:, mapping])
            new_c_proj.weight.data /= replication_factors[mapping].view(1, -1)

            mlp.c_fc = new_c_fc
            mlp.c_proj = new_c_proj

        self.config.n_hidden = new_hidden_dim
        print(f"MLP hidden dimension successfully widened to {new_hidden_dim}.")

    def resize_lora_rank(self, new_rank):
        """
        Legacy function for backward compatibility. 
        Use set_layer_lora_rank("attn.X", rank) for specific layers.
        """
        print("Warning: resize_lora_rank is deprecated. Use set_layer_lora_rank('attn.X', rank) instead.")
        new_rank = int(new_rank)
        self.config.attn_lora_rank = new_rank
        
        for i in range(self.config.n_layer):
            self.set_layer_lora_rank(f"attn.{i}", new_rank)
        
    def resize_embedding_rank(self, new_rank):
        """
        Legacy function for backward compatibility.
        Use set_layer_lora_rank("wte", rank) instead.
        """
        print("Warning: resize_embedding_rank is deprecated. Use set_layer_lora_rank('wte', rank) instead.")
        new_rank = int(new_rank)
        self.config.embedding_rank = new_rank
        self.set_layer_lora_rank("wte", new_rank)

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

    @torch.no_grad()
    def get_merged_state_dict(self):
        """
        Returns a state_dict with all LoRA weights merged into their main weights,
        ready for saving a universal checkpoint.
        This method is decorated with @torch.no_grad() to prevent gradient tracking.
        """
        # Create a new state_dict to populate
        final_sd = {}

        # Get a fresh copy of the model's current state_dict
        source_sd = self.state_dict()

        for key, value in source_sd.items():
            # This is a LoRA layer's main weight, it will be handled by the merge logic below. Skip it.
            if 'main_weight' in key:
                continue

            # These are the LoRA-specific weights. We are merging them, so we don't include them in the final dict.
            if 'lora_A' in key or 'lora_B' in key:
                continue

            # It's a standard parameter, so copy it directly.
            final_sd[key] = value

        # Now, explicitly find LoRA modules and perform the merge, adding the result to our final_sd
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear) and module.rank > 0:
                # The key for the final dict should be the standard layer name
                key = f"{name}.weight"
                # Get the bias from the main_weight linear layer
                bias_key = f"{name}.bias"

                # Calculate the merged weight
                lora_update = module.lora_B.weight @ module.lora_A.weight * (module.alpha / module.rank)
                merged_weight = module.main_weight.weight.data + lora_update
                final_sd[key] = merged_weight

                # Copy the bias if it exists
                if module.main_weight.bias is not None:
                    final_sd[bias_key] = module.main_weight.bias.data

            elif isinstance(module, LoRAEmbedding) and module.rank > 0:
                key = f"{name}.weight"

                # Calculate the merged weight for LoRAEmbedding
                # Fixed: Removed erroneous .T at the end - the result is already (vocab_size, n_embd)
                lora_update = module.lora_A.weight @ module.lora_B.weight.T
                merged_weight = module.main_weight.weight.data + lora_update * (module.alpha / module.rank)
                final_sd[key] = merged_weight

        return final_sd

    def resize_vocabulary(self, new_vocab_size, source_token_id, noise_std=0.01):
        """
        Grows the vocabulary size of the model in a function-preserving way.
        New token embeddings are initialized from a source token plus noise.
        """
        print(f"Resizing vocabulary from {self.config.vocab_size} to {new_vocab_size}.")
        if new_vocab_size <= self.config.vocab_size:
            raise ValueError("New vocabulary size must be larger than the current one.")

        old_wte = self.transformer.wte
        old_lm_head = self.lm_head
        device = old_wte.weight.device

        # Create new, larger layers
        self.transformer.wte = nn.Embedding(new_vocab_size, self.config.n_embd).to(device)
        self.lm_head = nn.Linear(self.config.n_embd, new_vocab_size, bias=False).to(device)

        # Copy old weights
        self.transformer.wte.weight.data[:self.config.vocab_size, :] = old_wte.weight.data
        self.lm_head.weight.data[:self.config.vocab_size, :] = old_lm_head.weight.data

        # Initialize new weights from the source token
        source_embedding = old_wte.weight.data[source_token_id, :]
        source_lm_head = old_lm_head.weight.data[source_token_id, :]

        # Add noise to break symmetry
        noise_embedding = torch.randn(new_vocab_size - self.config.vocab_size, self.config.n_embd, device=device) * noise_std
        noise_lm_head = torch.randn(new_vocab_size - self.config.vocab_size, self.config.n_embd, device=device) * noise_std

        self.transformer.wte.weight.data[self.config.vocab_size:, :] = source_embedding + noise_embedding
        self.lm_head.weight.data[self.config.vocab_size:, :] = source_lm_head + noise_lm_head

        # Update config and re-tie weights
        self.config.vocab_size = new_vocab_size
        self.transformer.wte.weight = self.lm_head.weight

        print("Vocabulary resized successfully.")

    def set_embedding_freeze_mode(self, enabled: bool):
        """
        Legacy function for backward compatibility.
        Use freeze_layer("wte") and freeze_layer("lm_head") instead.
        """
        print("Warning: set_embedding_freeze_mode is deprecated. Use freeze_layer/unfreeze_layer instead.")
        self.embedding_finetune_mode = enabled

        if enabled:
            # Freeze all layers except embeddings
            for i in range(self.config.n_layer):
                self.freeze_layer(f"attn.{i}")
                self.freeze_layer(f"mlp.{i}")
                self.freeze_layer(f"ln_1.{i}")
                self.freeze_layer(f"ln_2.{i}")
            self.freeze_layer("ln_f")
        else:
            # Unfreeze all layers
            for i in range(self.config.n_layer):
                self.unfreeze_layer(f"attn.{i}")
                self.unfreeze_layer(f"mlp.{i}")
                self.unfreeze_layer(f"ln_1.{i}")
                self.unfreeze_layer(f"ln_2.{i}")
            self.unfreeze_layer("ln_f")
            self.unfreeze_layer("wte")
            self.unfreeze_layer("lm_head")

    def set_embedding_finetune_mode(self, enabled: bool):
        """
        Legacy function for backward compatibility.
        Use freeze_layer/unfreeze_layer for specific control.
        """
        print("Warning: set_embedding_finetune_mode is deprecated. Use freeze_layer/unfreeze_layer instead.")
        self.embedding_finetune_mode = enabled

        if enabled:
            # Freeze all layers except embeddings and lm_head
            for i in range(self.config.n_layer):
                self.freeze_layer(f"attn.{i}")
                self.freeze_layer(f"mlp.{i}")
                self.freeze_layer(f"ln_1.{i}")
                self.freeze_layer(f"ln_2.{i}")
            self.freeze_layer("ln_f")
        else:
            # Unfreeze all layers
            for i in range(self.config.n_layer):
                self.unfreeze_layer(f"attn.{i}")
                self.unfreeze_layer(f"mlp.{i}")
                self.unfreeze_layer(f"ln_1.{i}")
                self.unfreeze_layer(f"ln_2.{i}")
            self.unfreeze_layer("ln_f")
            self.unfreeze_layer("wte")
            self.unfreeze_layer("lm_head")

    def freeze_layer(self, layer_name: str):
        """
        Freeze a specific layer by name. Supports patterns like:
        - "attn.2" for attention in layer 2 (0-based)
        - "mlp.0" for MLP in layer 0
        - "ln_1.5" for first LayerNorm in layer 5
        - "ln_2.3" for second LayerNorm in layer 3
        - "wte" for token embeddings
        - "wpe" for position embeddings (if not using rotary)
        - "ln_f" for final layer norm
        - "lm_head" for language model head
        """
        if '.' in layer_name:
            component, layer_idx = layer_name.split('.', 1)
            try:
                layer_idx = int(layer_idx)
            except ValueError:
                print(f"Invalid layer index in '{layer_name}'. Expected format: component.layer_index")
                return
            
            if layer_idx >= self.config.n_layer:
                print(f"Layer index {layer_idx} out of range. Model has {self.config.n_layer} layers.")
                return
            
            if component == "attn":
                for param in self.transformer.h[layer_idx].attn.parameters():
                    param.requires_grad_(False)
                print(f"Frozen attention layer {layer_idx}")
            elif component == "mlp":
                for param in self.transformer.h[layer_idx].mlp.parameters():
                    param.requires_grad_(False)
                print(f"Frozen MLP layer {layer_idx}")
            elif component == "ln_1":
                for param in self.transformer.h[layer_idx].ln_1.parameters():
                    param.requires_grad_(False)
                print(f"Frozen first LayerNorm in layer {layer_idx}")
            elif component == "ln_2":
                for param in self.transformer.h[layer_idx].ln_2.parameters():
                    param.requires_grad_(False)
                print(f"Frozen second LayerNorm in layer {layer_idx}")
            else:
                print(f"Unknown component '{component}'. Valid components: attn, mlp, ln_1, ln_2")
        else:
            # Handle global components
            if layer_name == "wte":
                for param in self.transformer.wte.parameters():
                    param.requires_grad_(False)
                print("Frozen token embeddings")
            elif layer_name == "wpe" and not self.config.use_rotary_embeddings:
                for param in self.transformer.wpe.parameters():
                    param.requires_grad_(False)
                print("Frozen position embeddings")
            elif layer_name == "ln_f":
                for param in self.transformer.ln_f.parameters():
                    param.requires_grad_(False)
                print("Frozen final layer norm")
            elif layer_name == "lm_head":
                for param in self.lm_head.parameters():
                    param.requires_grad_(False)
                print("Frozen language model head")
            else:
                print(f"Unknown layer '{layer_name}'. Valid global layers: wte, wpe, ln_f, lm_head")

    def unfreeze_layer(self, layer_name: str):
        """
        Unfreeze a specific layer by name. Uses same naming convention as freeze_layer.
        """
        if '.' in layer_name:
            component, layer_idx = layer_name.split('.', 1)
            try:
                layer_idx = int(layer_idx)
            except ValueError:
                print(f"Invalid layer index in '{layer_name}'. Expected format: component.layer_index")
                return
            
            if layer_idx >= self.config.n_layer:
                print(f"Layer index {layer_idx} out of range. Model has {self.config.n_layer} layers.")
                return
            
            if component == "attn":
                for param in self.transformer.h[layer_idx].attn.parameters():
                    param.requires_grad_(True)
                print(f"Unfrozen attention layer {layer_idx}")
            elif component == "mlp":
                for param in self.transformer.h[layer_idx].mlp.parameters():
                    param.requires_grad_(True)
                print(f"Unfrozen MLP layer {layer_idx}")
            elif component == "ln_1":
                for param in self.transformer.h[layer_idx].ln_1.parameters():
                    param.requires_grad_(True)
                print(f"Unfrozen first LayerNorm in layer {layer_idx}")
            elif component == "ln_2":
                for param in self.transformer.h[layer_idx].ln_2.parameters():
                    param.requires_grad_(True)
                print(f"Unfrozen second LayerNorm in layer {layer_idx}")
            else:
                print(f"Unknown component '{component}'. Valid components: attn, mlp, ln_1, ln_2")
        else:
            # Handle global components
            if layer_name == "wte":
                for param in self.transformer.wte.parameters():
                    param.requires_grad_(True)
                print("Unfrozen token embeddings")
            elif layer_name == "wpe" and not self.config.use_rotary_embeddings:
                for param in self.transformer.wpe.parameters():
                    param.requires_grad_(True)
                print("Unfrozen position embeddings")
            elif layer_name == "ln_f":
                for param in self.transformer.ln_f.parameters():
                    param.requires_grad_(True)
                print("Unfrozen final layer norm")
            elif layer_name == "lm_head":
                for param in self.lm_head.parameters():
                    param.requires_grad_(True)
                print("Unfrozen language model head")
            else:
                print(f"Unknown layer '{layer_name}'. Valid global layers: wte, wpe, ln_f, lm_head")

    def set_layer_lora_rank(self, layer_name: str, rank: int):
        """
        Set LoRA rank for a specific layer. Supports patterns like:
        - "attn.2" for attention in layer 2 (0-based)
        - "wte" for token embeddings
        """
        rank = int(rank)
        
        if '.' in layer_name:
            component, layer_idx = layer_name.split('.', 1)
            try:
                layer_idx = int(layer_idx)
            except ValueError:
                print(f"Invalid layer index in '{layer_name}'. Expected format: component.layer_index")
                return
            
            if layer_idx >= self.config.n_layer:
                print(f"Layer index {layer_idx} out of range. Model has {self.config.n_layer} layers.")
                return
            
            if component == "attn":
                block = self.transformer.h[layer_idx]
                device = block.attn.c_attn.weight.device if hasattr(block.attn.c_attn, 'weight') else next(block.attn.c_attn.parameters()).device
                
                # Merge existing LoRA weights if it's already a LoRA layer
                if isinstance(block.attn.c_attn, LoRALinear):
                    block.attn.c_attn.merge_and_reset()
                
                # Create new LoRA layer with specified rank
                if rank > 0:
                    new_c_attn = LoRALinear(
                        in_features=self.config.n_embd,
                        out_features=3 * self.config.n_embd,
                        rank=rank,
                        alpha=self.config.lora_alpha,
                        bias=self.config.bias
                    )
                    
                    # Copy weights from existing layer
                    if isinstance(block.attn.c_attn, LoRALinear):
                        new_c_attn.main_weight.load_state_dict(block.attn.c_attn.main_weight.state_dict())
                    else:
                        new_c_attn.main_weight.weight.data.copy_(block.attn.c_attn.weight.data)
                        if block.attn.c_attn.bias is not None:
                            new_c_attn.main_weight.bias.data.copy_(block.attn.c_attn.bias.data)
                    
                    block.attn.c_attn = new_c_attn.to(device)
                    print(f"Set attention layer {layer_idx} LoRA rank to {rank}")
                else:
                    # Convert to regular Linear layer
                    if isinstance(block.attn.c_attn, LoRALinear):
                        new_c_attn = nn.Linear(
                            self.config.n_embd, 
                            3 * self.config.n_embd, 
                            bias=self.config.bias
                        )
                        new_c_attn.weight.data.copy_(block.attn.c_attn.main_weight.weight.data)
                        if block.attn.c_attn.main_weight.bias is not None:
                            new_c_attn.bias.data.copy_(block.attn.c_attn.main_weight.bias.data)
                        block.attn.c_attn = new_c_attn.to(device)
                    print(f"Disabled LoRA for attention layer {layer_idx}")
            else:
                print(f"LoRA only supported for 'attn' component, not '{component}'")
        else:
            if layer_name == "wte":
                device = self.transformer.wte.weight.device
                
                # Merge existing LoRA weights if it's already a LoRA layer
                if isinstance(self.transformer.wte, LoRAEmbedding):
                    self.transformer.wte.merge_and_reset()
                
                if rank > 0:
                    new_wte = LoRAEmbedding(
                        vocab_size=self.config.vocab_size,
                        n_embd=self.config.n_embd,
                        rank=rank,
                        alpha=self.config.lora_alpha
                    )
                    
                    # Copy weights from existing layer
                    if isinstance(self.transformer.wte, LoRAEmbedding):
                        new_wte.main_weight.load_state_dict(self.transformer.wte.main_weight.state_dict())
                    else:
                        new_wte.main_weight.weight.data.copy_(self.transformer.wte.weight.data)
                    
                    self.transformer.wte = new_wte.to(device)
                    # Re-tie weights with lm_head
                    self.transformer.wte.main_weight.weight = self.lm_head.weight
                    self.lm_head.requires_grad_(False)
                    print(f"Set token embeddings LoRA rank to {rank}")
                else:
                    # Convert to regular Embedding layer
                    if isinstance(self.transformer.wte, LoRAEmbedding):
                        new_wte = nn.Embedding(self.config.vocab_size, self.config.n_embd)
                        new_wte.weight.data.copy_(self.transformer.wte.main_weight.weight.data)
                        self.transformer.wte = new_wte.to(device)
                        # Re-tie weights with lm_head
                        self.transformer.wte.weight = self.lm_head.weight
                    print("Disabled LoRA for token embeddings")
            else:
                print(f"LoRA only supported for 'wte' and 'attn.X' layers, not '{layer_name}'")
