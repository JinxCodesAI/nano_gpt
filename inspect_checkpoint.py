#!/usr/bin/env python3

# Simple checkpoint inspector without complex dependencies
import pickle
import sys
import os

def inspect_checkpoint():
    # Check the latest checkpoint
    ckpt_path = 'out/ckpt_unmasking_7500.pt'
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    print(f"Inspecting: {ckpt_path}")
    
    try:
        # Try to load without torch
        with open(ckpt_path, 'rb') as f:
            # Read the file header to see if it's a torch file
            header = f.read(100)
            print(f"File header (first 100 bytes): {header[:50]}...")
            
            # Try to examine structure without full torch load
            f.seek(0)
            try:
                # This might give us some insight into the structure
                import struct
                # Torch files start with a magic number
                magic = f.read(8)
                print(f"Magic bytes: {magic}")
            except Exception as e:
                print(f"Could not read magic: {e}")
        
    except Exception as e:
        print(f"Error reading checkpoint: {e}")
    
    # Let's also try to examine with the available Python
    print("\nTrying to find Python executable...")
    import subprocess
    
    # Try different python commands
    python_cmds = [ 'python', '/usr/bin/python', 'python3','/usr/bin/python3']
    
    for cmd in python_cmds:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"Found Python: {cmd} - {result.stdout.strip()}")
                
                # Try to run a simple torch test
                test_script = '''
import torch
ckpt = torch.load("out/ckpt_unmasking_7000.pt", map_location="cpu", weights_only=False)
print("Checkpoint keys:", list(ckpt.keys()))
if "training_context" in ckpt:
    ctx = ckpt["training_context"]
    print("Training context:", ctx)
else:
    print("NO training_context found!")
'''
                
                try:
                    result = subprocess.run([cmd, '-c', test_script], 
                                          capture_output=True, text=True, timeout=10, cwd='.')
                    print("Torch test output:")
                    print(result.stdout)
                    if result.stderr:
                        print("Errors:")
                        print(result.stderr)
                    break
                except subprocess.TimeoutExpired:
                    print(f"Timeout with {cmd}")
                except Exception as e:
                    print(f"Failed with {cmd}: {e}")
        except:
            continue

if __name__ == "__main__":
    inspect_checkpoint()