#!/usr/bin/env python3
"""
Test script to verify LoRA integration fixes in model.py
This script tests that LoRA modules are properly instantiated when configured
and that the model can perform forward passes correctly.
"""

import torch
import sys
import os

# Add the current directory to the path so we can import model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import GPTConfig, GPT, LoRALinear, LoRAEmbedding

def test_standard_model():
    """Test that standard model (no LoRA) works as before"""
    print("Testing standard model (no LoRA)...")
    
    config = GPTConfig(
        block_size=128,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=False,
        # LoRA disabled
        embedding_mode='standard',
        embedding_rank=0,
        attn_lora_rank=0,
        lora_alpha=1.0
    )
    
    model = GPT(config)
    
    # Check that standard modules are used
    assert isinstance(model.transformer.wte, torch.nn.Embedding), "Expected standard Embedding"
    assert isinstance(model.transformer.h[0].attn.c_attn, torch.nn.Linear), "Expected standard Linear"
    
    # Test forward pass with targets (training mode - returns full sequence)
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, loss = model(input_ids, targets)

    assert logits.shape == (batch_size, seq_len, config.vocab_size), f"Unexpected logits shape: {logits.shape}"
    assert loss is not None, "Expected loss to be computed"
    print("✓ Standard model test passed")

def test_lora_embedding_model():
    """Test model with LoRA embedding enabled"""
    print("Testing model with LoRA embedding...")
    
    config = GPTConfig(
        block_size=128,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=False,
        # LoRA embedding enabled
        embedding_mode='lora',
        embedding_rank=16,  # n_embd / 4
        attn_lora_rank=0,   # attention LoRA disabled
        lora_alpha=1.0
    )
    
    model = GPT(config)
    
    # Check that LoRA embedding is used
    assert isinstance(model.transformer.wte, LoRAEmbedding), "Expected LoRAEmbedding"
    assert model.transformer.wte.rank == 16, f"Expected rank 16, got {model.transformer.wte.rank}"
    assert isinstance(model.transformer.h[0].attn.c_attn, torch.nn.Linear), "Expected standard Linear for attention"
    
    # Test forward pass with targets (training mode - returns full sequence)
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, loss = model(input_ids, targets)

    assert logits.shape == (batch_size, seq_len, config.vocab_size), f"Unexpected logits shape: {logits.shape}"
    assert loss is not None, "Expected loss to be computed"
    print("✓ LoRA embedding model test passed")

def test_lora_attention_model():
    """Test model with LoRA attention enabled"""
    print("Testing model with LoRA attention...")
    
    config = GPTConfig(
        block_size=128,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=False,
        # LoRA attention enabled
        embedding_mode='standard',
        embedding_rank=0,
        attn_lora_rank=16,  # n_embd / 4
        lora_alpha=2.0
    )
    
    model = GPT(config)
    
    # Check that LoRA attention is used
    assert isinstance(model.transformer.wte, torch.nn.Embedding), "Expected standard Embedding"
    assert isinstance(model.transformer.h[0].attn.c_attn, LoRALinear), "Expected LoRALinear for attention"
    assert model.transformer.h[0].attn.c_attn.rank == 16, f"Expected rank 16, got {model.transformer.h[0].attn.c_attn.rank}"
    assert model.transformer.h[0].attn.c_attn.alpha == 2.0, f"Expected alpha 2.0, got {model.transformer.h[0].attn.c_attn.alpha}"
    
    # Test forward pass with targets (training mode - returns full sequence)
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, loss = model(input_ids, targets)

    assert logits.shape == (batch_size, seq_len, config.vocab_size), f"Unexpected logits shape: {logits.shape}"
    assert loss is not None, "Expected loss to be computed"
    print("✓ LoRA attention model test passed")

def test_full_lora_model():
    """Test model with both LoRA embedding and attention enabled"""
    print("Testing model with full LoRA (embedding + attention)...")
    
    config = GPTConfig(
        block_size=128,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=False,
        # Both LoRA modes enabled
        embedding_mode='lora',
        embedding_rank=8,   # n_embd / 8
        attn_lora_rank=16,  # n_embd / 4
        lora_alpha=1.5
    )
    
    model = GPT(config)
    
    # Check that both LoRA modules are used
    assert isinstance(model.transformer.wte, LoRAEmbedding), "Expected LoRAEmbedding"
    assert model.transformer.wte.rank == 8, f"Expected embedding rank 8, got {model.transformer.wte.rank}"
    assert isinstance(model.transformer.h[0].attn.c_attn, LoRALinear), "Expected LoRALinear for attention"
    assert model.transformer.h[0].attn.c_attn.rank == 16, f"Expected attention rank 16, got {model.transformer.h[0].attn.c_attn.rank}"
    
    # Test forward pass with targets (training mode - returns full sequence)
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, loss = model(input_ids, targets)

    assert logits.shape == (batch_size, seq_len, config.vocab_size), f"Unexpected logits shape: {logits.shape}"
    assert loss is not None, "Expected loss to be computed"
    print("✓ Full LoRA model test passed")

def test_weight_tying():
    """Test that weight tying works correctly for both standard and LoRA embeddings"""
    print("Testing weight tying...")
    
    # Test standard embedding weight tying
    config_standard = GPTConfig(
        block_size=128, vocab_size=1000, n_layer=1, n_head=4, n_embd=64,
        embedding_mode='standard', embedding_rank=0, attn_lora_rank=0
    )
    model_standard = GPT(config_standard)
    assert model_standard.transformer.wte.weight is model_standard.lm_head.weight, "Standard weight tying failed"
    
    # Test LoRA embedding weight tying
    config_lora = GPTConfig(
        block_size=128, vocab_size=1000, n_layer=1, n_head=4, n_embd=64,
        embedding_mode='lora', embedding_rank=16, attn_lora_rank=0
    )
    model_lora = GPT(config_lora)
    assert model_lora.transformer.wte.main_weight.weight is model_lora.lm_head.weight, "LoRA weight tying failed"
    
    print("✓ Weight tying test passed")

def main():
    """Run all tests"""
    print("Running LoRA integration tests...\n")
    
    try:
        test_standard_model()
        test_lora_embedding_model()
        test_lora_attention_model()
        test_full_lora_model()
        test_weight_tying()
        
        print("\n🎉 All tests passed! LoRA integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
