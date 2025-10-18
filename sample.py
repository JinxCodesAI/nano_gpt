"""Sample from a trained model with pluggable diffusion transformations."""
import torch
from display import DiffusionDisplay
from model_setup import ModelSetup
from sampling_pipeline import (
    DiffusionConfig,
    DiffusionPipeline,
    DisplayTransformation,
    EditScheduleTransformation,
    NoiseScheduleTransformation,
)

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-char-random-replacement' # ignored if init_from is not 'resume'
ckpt_name = 'new_hope_7_5250.pt'
start = "POMPEY:\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 900 # number of tokens generated in each sample
max_iterations = 20 # maximum number of diffusion iterations per sample
fix_prompt_during_diffusion = True # keep conditioning text fixed at every iteration when True
temperature = 1.5 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
seed = 123
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

# --- Re-noise schedule knobs ---
noise_schedule = 'cosine'   # 'linear' or 'cosine' (anything else -> no re-noise)
noise_start    = 0.0       # fraction of positions to randomize at iter 0 (e.g., 0.05-0.30)
noise_end      = 0.0       # fraction at the last iteration (usually 0.0)

# --- Structural edit knobs ---
# edit_schedule: interpolation curve for mapping iteration -> edit ratios. Accepts "linear" or
#                "cosine", defaults to cosine decay. Any other value disables edits.
edit_schedule        = 'cosine'
# insert_ratio_start/end: fraction of the currently active (non-prompt) length that we try to fill
#                         with new insertions at the beginning/end of diffusion. Values in the
#                         0.00–0.10 range tend to preserve coherence without overwhelming updates.
insert_ratio_start   = 0.0
insert_ratio_end     = 0.00
# delete_ratio_start/end: fraction of the active region eligible for deletion early/late in the
#                         schedule. Keep these below ~0.10 to avoid deleting large spans at once.
delete_ratio_start   = 0.0
delete_ratio_end     = 0.00
# delete_margin: probability buffer applied before considering a removal beneficial. Larger margins
#                make deletions rarer; values around 0.01–0.05 generally work well.
delete_margin        = 0.02
# delete_lambda: weight for the second-order deletion heuristic that looks one token ahead. Raising
#                this emphasizes deletions that also improve the following token. Typical range is
#                0.1–0.5.
delete_lambda        = 0.30
# length_target_mode: optional policy that nudges the sequence toward a target length. "none" leaves
#                     lengths unconstrained, while "to_max_new" gradually pushes toward
#                     initial_length + max_new_tokens.
length_target_mode   = 'none'
# cooldown_distance: radius of edit suppression in tokens. Recently edited positions are skipped
#                    for this many steps on each side to avoid thrashing. 0 disables cooldown.
cooldown_distance    = 5

exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
setup = ModelSetup(
    init_from=init_from,
    out_dir=out_dir,
    ckpt_name=ckpt_name,
    device=device,
    dtype=dtype,
    compile_model=compile,
    start=start,
)
model = setup.model
decode = setup.decode
space_token_id = setup.space_token_id
prompt = setup.prompt
initial_length = setup.initial_length
block_size = setup.block_size
ctx = setup.autocast_context()

config = DiffusionConfig(
    block_size=block_size,
    initial_length=initial_length,
    max_iterations=max_iterations,
    max_new_tokens=max_new_tokens,
    prompt=prompt,
    space_token_id=space_token_id,
    temperature=temperature,
    device=device,
    fix_prompt_during_diffusion=fix_prompt_during_diffusion,
)

transformations = [
    DisplayTransformation(lambda: DiffusionDisplay(decode)),
    EditScheduleTransformation(
        edit_schedule,
        insert_ratio_start=insert_ratio_start,
        insert_ratio_end=insert_ratio_end,
        delete_ratio_start=delete_ratio_start,
        delete_ratio_end=delete_ratio_end,
        delete_margin=delete_margin,
        delete_lambda=delete_lambda,
        cooldown_distance=cooldown_distance,
        length_target_mode=length_target_mode,
    ),
    NoiseScheduleTransformation(
        noise_schedule,
        start=noise_start,
        end=noise_end,
    ),
]

pipeline = DiffusionPipeline(
    model=model,
    config=config,
    transformations=transformations,
)

with torch.no_grad():
    with ctx:
        final_tokens_list = pipeline.run_samples(num_samples)
        if not any(isinstance(t, DisplayTransformation) for t in transformations):
            for tokens in final_tokens_list:
                print(decode(tokens))
                print("---------------")
