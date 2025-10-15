"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-char-random-replacement' # ignored if init_from is not 'resume'
ckpt_name = 'new_hope_2_9000.pt'
start = "\nPOMPEY:\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 900 # number of tokens generated in each sample
max_iterations = 20 # maximum number of diffusion iterations per sample
fix_prompt_during_diffusion = True # keep conditioning text fixed at every iteration when True
temperature = 1.5 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
seed = 42
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
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

            if temperature <= 0:
                raise ValueError("temperature must be greater than zero to perform sampling.")

            prev_decoded = None
            total_log_likelihood = 0.0  # Track cumulative log likelihood over diffusion steps.
            iteration = 0
            while iteration < max_iterations:
                # Execute a full forward pass on the current sequence to obtain token logits.
                # We pass the entire padded sequence so the model can condition on every position
                # (prompt + generated tokens) before proposing replacements for the active window.
                logits, _ = model(x)
                logits = logits / temperature

                # Convert logits to log-probabilities for every token position, then exponentiate
                # to obtain probabilities for sampling.
                log_probs = torch.log_softmax(logits, dim=-1)
                probs = log_probs.exp()

                # Slice out both probability and log-probability distributions for the active portion
                # of the sequence. Cast to float32 for stable multinomial sampling and log accumulation.
                active_log_probs = log_probs[0, :max_token_pos, :].to(dtype=torch.float)
                active_probs = probs[0, :max_token_pos, :].to(dtype=torch.float)

                # Draw a token for every active position via multinomial sampling. We first collapse the
                # sequence dimension so multinomial can treat each position independently, then restore
                # the original shape to align with the sequence layout. Accumulate the log likelihood of
                # the selected tokens to report overall generation confidence.
                flat_probs = active_probs.view(-1, active_probs.size(-1))
                sampled_indices = torch.multinomial(flat_probs, 1)

                flat_log_probs = active_log_probs.view(-1, active_log_probs.size(-1))
                iteration_log_likelihood = flat_log_probs.gather(1, sampled_indices).mean().item()
                total_log_likelihood += iteration_log_likelihood

                sampled = sampled_indices.view(1, -1)

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

                dupa = x[0, :max_token_pos].tolist()
                decoded = decode(dupa)


                colored_chars = []
                green_count = 0
                orange_count = 0
                for idx, char in enumerate(decoded):
                    if prev_decoded is not None and idx < len(prev_decoded) and char == prev_decoded[idx]:
                        colored_chars.append(f"{GREEN}{char}{RESET}")
                        green_count+=1
                    else:
                        colored_chars.append(f"{ORANGE}{char}{RESET}")
                        orange_count+=1
                print(f"Iteration {iteration}, sample {k} | cumulative log likelihood: {total_log_likelihood:.4f} | changed {orange_count}/{green_count+orange_count}")
                print("".join(colored_chars))
                prev_decoded = decoded

                iteration += 1
                if iteration >= max_iterations:
                    break

            final_tokens = x[0, :max_token_pos].tolist()
            print(f"Total log likelihood for sample {k}: {total_log_likelihood:.4f}")
            print(decode(final_tokens))
            print('---------------')
