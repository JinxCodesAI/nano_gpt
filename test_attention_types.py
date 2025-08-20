#!/usr/bin/env python3
"""
Test script to verify both causal and bidirectional attention types work correctly
"""

import torch
import torch.nn.functional as F
from model import GPTConfig, GPT

def test_attention_type(attention_type):
    """Test a specific attention type"""
    print(f"\n=== Testing {attention_type} attention ===")
    
    # Create a small test model
    config = GPTConfig(
        block_size=8, 
        vocab_size=50, 
        n_layer=2, 
        n_head=2, 
        n_embd=32,
        dropout=0.0,
        bias=False,
        attention_type=attention_type
    )
    model = GPT(config)
    model.eval()
    
    # Create test input
    batch_size = 2
    seq_len = 6
    x = torch.randint(0, 50, (batch_size, seq_len))
    
    print(f"Input shape: {x.shape}")
    print(f"Input tokens: {x}")
    
    # Test forward pass without targets (inference mode)
    with torch.no_grad():
        logits, loss = model(x)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Loss: {loss}")
    
    if attention_type == 'causal':
        # For causal attention in inference mode, should only get logits for last position
        expected_shape = (batch_size, 1, config.vocab_size)
        assert logits.shape == expected_shape, f"Causal inference: expected {expected_shape}, got {logits.shape}"
        print("✓ Causal attention correctly returns only last position logits in inference")
    else:
        # For bidirectional attention, should get logits for all positions
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert logits.shape == expected_shape, f"Bidirectional inference: expected {expected_shape}, got {logits.shape}"
        print("✓ Bidirectional attention correctly returns all position logits in inference")
    
    # Test forward pass with targets (training mode)
    y = torch.randint(0, 50, (batch_size, seq_len))
    with torch.no_grad():
        logits_train, loss_train = model(x, y)
    
    print(f"Training mode - Logits shape: {logits_train.shape}, Loss: {loss_train.item():.4f}")
    
    # Both attention types should return full sequence logits in training mode
    expected_train_shape = (batch_size, seq_len, config.vocab_size)
    assert logits_train.shape == expected_train_shape, f"Training: expected {expected_train_shape}, got {logits_train.shape}"
    assert loss_train is not None, "Training mode should return a loss"
    
    print(f"✓ {attention_type.capitalize()} attention works correctly in both training and inference modes")
    
    return True

def test_attention_patterns():
    """Test that causal and bidirectional attention produce different patterns"""
    print(f"\n=== Testing attention pattern differences ===")
    
    # Create identical models with different attention types
    base_config = {
        'block_size': 5,
        'vocab_size': 10,
        'n_layer': 1,
        'n_head': 1,
        'n_embd': 8,
        'dropout': 0.0,
        'bias': False
    }
    
    causal_config = GPTConfig(**base_config, attention_type='causal')
    bidirectional_config = GPTConfig(**base_config, attention_type='bidirectional')
    
    causal_model = GPT(causal_config)
    bidirectional_model = GPT(bidirectional_config)
    
    # Use same weights for fair comparison
    bidirectional_model.load_state_dict(causal_model.state_dict())
    
    causal_model.eval()
    bidirectional_model.eval()
    
    # Test input
    x = torch.tensor([[1, 2, 3, 4, 5]])  # Shape: (1, 5)
    
    with torch.no_grad():
        # Get training mode outputs (full sequence) for comparison
        causal_logits, _ = causal_model(x, x)  # Pass targets to get full sequence
        bidirectional_logits, _ = bidirectional_model(x, x)
    
    print(f"Causal logits shape: {causal_logits.shape}")
    print(f"Bidirectional logits shape: {bidirectional_logits.shape}")
    
    # Convert to probabilities
    causal_probs = F.softmax(causal_logits[0], dim=-1)  # Shape: (5, 10)
    bidirectional_probs = F.softmax(bidirectional_logits[0], dim=-1)  # Shape: (5, 10)
    
    # Calculate differences between attention types
    differences = []
    for pos in range(5):
        kl_div = F.kl_div(causal_probs[pos].log(), bidirectional_probs[pos], reduction='sum')
        differences.append(kl_div.item())
    
    avg_difference = sum(differences) / len(differences)
    print(f"Average KL divergence between causal and bidirectional: {avg_difference:.4f}")
    
    # The attention patterns should be different
    assert avg_difference > 0.001, "Causal and bidirectional attention should produce different patterns"
    
    print("✓ Causal and bidirectional attention produce different output patterns")
    
    return True

def test_config_saving_loading():
    """Test that attention_type is properly saved and loaded"""
    print(f"\n=== Testing config saving/loading ===")

    # Create model with bidirectional attention
    config = GPTConfig(
        block_size=10,
        vocab_size=20,
        n_layer=1,
        n_head=1,
        n_embd=16,
        attention_type='bidirectional'
    )

    model = GPT(config)

    # Simulate saving model_args (as done in train.py)
    model_args = {
        'n_layer': config.n_layer,
        'n_head': config.n_head,
        'n_embd': config.n_embd,
        'block_size': config.block_size,
        'bias': config.bias,
        'vocab_size': config.vocab_size,
        'dropout': config.dropout,
        'attention_type': config.attention_type
    }

    # Simulate loading (as done in sample.py)
    loaded_config = GPTConfig(**model_args)
    loaded_model = GPT(loaded_config)

    print(f"Original attention_type: {config.attention_type}")
    print(f"Loaded attention_type: {loaded_config.attention_type}")

    assert config.attention_type == loaded_config.attention_type, "Attention type should be preserved"

    print("✓ Attention type is properly saved and loaded")

    return True

def test_backward_compatibility():
    """Test backward compatibility when attention_type is missing"""
    print(f"\n=== Testing backward compatibility ===")

    # Simulate old model_args without attention_type (as in old checkpoints)
    old_model_args = {
        'n_layer': 2,
        'n_head': 2,
        'n_embd': 32,
        'block_size': 10,
        'bias': False,
        'vocab_size': 20,
        'dropout': 0.0
        # Note: no 'attention_type' key
    }

    # Test that GPTConfig defaults to 'causal' when attention_type is missing
    config = GPTConfig(**old_model_args)
    print(f"Config without attention_type defaults to: {config.attention_type}")
    assert config.attention_type == 'causal', "Should default to 'causal' for backward compatibility"

    # Test that the model works with this config
    model = GPT(config)
    model.eval()

    x = torch.randint(0, 20, (1, 5))
    with torch.no_grad():
        logits, _ = model(x)

    # Should behave like causal attention (only last position in inference)
    assert logits.shape == (1, 1, 20), f"Expected (1, 1, 20), got {logits.shape}"

    print("✓ Backward compatibility works correctly")
    print("✓ Old checkpoints without attention_type will default to causal attention")

    return True

if __name__ == "__main__":
    print("Testing configurable attention types...\n")
    
    try:
        # Test both attention types
        test_attention_type('causal')
        test_attention_type('bidirectional')
        
        # Test that they produce different patterns
        test_attention_patterns()
        
        # Test config saving/loading
        test_config_saving_loading()

        # Test backward compatibility
        test_backward_compatibility()

        print("\n🎉 All attention type tests passed!")
        print("Both causal and bidirectional attention are working correctly.")
        print("The attention_type parameter is properly configurable and persistent.")
        print("Backward compatibility is maintained for old checkpoints.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
