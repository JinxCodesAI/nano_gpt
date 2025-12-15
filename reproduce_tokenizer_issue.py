
import os
import torch
import pickle
from model_setup import ModelSetup

# Mocking the checkpoint to force ModelSetup to load cosmopedia meta
# We need a dummy checkpoint file structure
# However, ModelSetup loads the checkpoint file. 
# We can just instantiate ModelSetup with a config that points to the right place if we have a real checkpoint.
# The user mentioned '2_ckpt_MLM_9250.pt' in 'out-cosmopedia'.
# Let's see if that file exists.

def reproduce():
    # We can perform a simpler check by manually inspecting how proper tokenizers works vs current implementation
    # But to use ModelSetup we need the checkpoint.
    
    # Let's try to verify if the checkpoint exists
    ckpt_path = os.path.join('out-cosmopedia', '2_ckpt_MLM_9250.pt')
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}, cannot fully verify ModelSetup without it.")
        return

    setup = ModelSetup(
        init_from='resume',
        out_dir='out-cosmopedia',
        ckpt_name='2_ckpt_MLM_9250.pt',
        device='cpu',
        dtype='float32',
        compile_model=False,
        start="Logical implication is"
    )

    encoded = setup.encode("Logical implication is")
    print(f"Encoded IDs: {encoded}")
    
    # Expected: "L" (48), "og" (418), "ical" (434), etc.
    # Current bad: 48, 83, 75, 77, 71, 69, 80...
    
    expected_partial = [48, 418, 434]
    
    # Check if we see the bad pattern (char level)
    # L(48), o(83), g(75)
    bad_pattern = [48, 83, 75]
    
    if encoded[:3] == bad_pattern:
        print("FAIL: Detected character-level encoding (naive).")
    elif encoded[:len(expected_partial)] == expected_partial:
        print("SUCCESS: Detected correct BPE encoding.")
    else:
        print(f"UNKNOWN: Got {encoded[:5]}...")

if __name__ == "__main__":
    reproduce()
