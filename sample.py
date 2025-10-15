"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from sampling_utils import apply_re_noise, compute_noise_ratio

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-char-random-replacement' # ignored if init_from is not 'resume'
ckpt_name = 'ckpt_MLM_1000.pt'
start = "\nWhere is the king?" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 900 # number of tokens generated in each sample
max_iterations = 50 # maximum number of diffusion iterations per sample
fix_prompt_during_diffusion = True # keep conditioning text fixed at every iteration when True
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
seed = 42
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

# --- Re-noise schedule knobs ---
noise_schedule = 'cosine'   # 'linear' or 'cosine' (anything else -> no re-noise)
noise_start    = 0.20       # fraction of positions to randomize at iter 0 (e.g., 0.05–0.30)
noise_end      = 0.00       # fraction at the last iteration (usually 0.0)
avoid_ids      = []         # optional list of token IDs to never inject (e.g., PAD)

exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

vocab_size = getattr(model.config, 'vocab_size', None)
if vocab_size is None:
    vocab_size = model.get_input_embeddings().weight.size(0)
pad_id = getattr(model.config, 'pad_token_id', None)
extra_avoid = [] if pad_id is None else [pad_id]
avoid = list({int(i) for i in (list(avoid_ids) + extra_avoid) if i is not None})
avoid_ids_tuple = tuple(avoid)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

GREEN = "\033[92m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
prompt = torch.tensor(start_ids, dtype=torch.long, device=device)
initial_length = prompt.size(0)
block_size = model.config.block_size
if initial_length > block_size:
    raise ValueError(f"Prompt is longer ({initial_length}) than model block size ({block_size}).")

# run diffusion-style generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            seq_length = block_size
            max_token_pos = min(seq_length, initial_length + max_new_tokens)

            x = torch.zeros((1, seq_length), dtype=torch.long, device=device)
            x[0, :initial_length] = prompt

            prev_decoded = None
            iteration = 0
            while iteration < max_iterations:
                # Execute a full forward pass on the current sequence to obtain token logits.
                # We pass the entire padded sequence so the model can condition on every position
                # (prompt + generated tokens) before proposing replacements for the active window.
                
                dupa = x[0, :max_token_pos].tolist()
                print(f"Iteration {iteration}, sample {k}")
                decoded = decode(dupa)

                colored_chars = []
                for idx, char in enumerate(decoded):
                    if prev_decoded is not None and idx < len(prev_decoded) and char == prev_decoded[idx]:
                        colored_chars.append(f"{GREEN}{char}{RESET}")
                    else:
                        colored_chars.append(f"{ORANGE}{char}{RESET}")
                print("".join(colored_chars))
                prev_decoded = decoded

                beta_s = compute_noise_ratio(iteration, max_iterations, noise_schedule, noise_start, noise_end)
                x = apply_re_noise(
                    x=x,
                    max_token_pos=max_token_pos,
                    initial_length=initial_length,
                    vocab_size=vocab_size,
                    ratio=beta_s,
                    device=device,
                    avoid_ids=avoid_ids_tuple,
                )
                logits, _ = model(x)

                # Convert logits to probabilities for every token position. The softmax is taken
                # across the vocabulary dimension, producing a categorical distribution that we can
                # sample from to perform discrete diffusion updates.
                probs = torch.softmax(logits, dim=-1)

                # Slice out the probability distributions for only the active portion of the sequence
                # (prompt length + allowable new tokens). We cast to float32 to keep multinomial sampling
                # numerically stable even when the model is running in lower precision (e.g., bf16).
                active_probs = probs[0, :max_token_pos, :].to(dtype=torch.float)

                # Draw a token for every active position via multinomial sampling. We first collapse the
                # sequence dimension so multinomial can treat each position independently, then restore
                # the original shape to align with the sequence layout.
                sampled = torch.multinomial(active_probs.view(-1, active_probs.size(-1)), 1)
                sampled = sampled.view(1, -1)

                # Write the sampled tokens back into the working sequence window, overwriting any previous
                # proposals. Optionally restore the original prompt tokens so the conditioning text stays
                # fixed throughout diffusion updates when the flag is enabled.
                x[:, :max_token_pos] = sampled
                if fix_prompt_during_diffusion:
                    x[0, :initial_length] = prompt

                # Zero out any positions beyond the active window so that the next iteration continues to
                # treat them as padding (i.e., not part of the diffusion process yet).
                if max_token_pos < seq_length:
                    x[:, max_token_pos:] = 0

                iteration += 1
                if iteration >= max_iterations:
                    break

            final_tokens = x[0, :max_token_pos].tolist()
            print(decode(final_tokens))
            print('---------------')
