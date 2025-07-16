"""
Basic functionality tests for dual-mode GPT implementation.
These tests are designed to run on CPU to ensure compatibility without CUDA.
"""

import torch
import pytest
from model import GPT, GPTConfig


def test_gptconfig_mode_parameter():
    """Test GPTConfig mode parameter validation"""
    # Test default mode
    config = GPTConfig()
    assert config.mode == 'generator'
    
    # Test valid modes
    config_gen = GPTConfig(mode='generator')
    assert config_gen.mode == 'generator'
    
    config_reward = GPTConfig(mode='reward')
    assert config_reward.mode == 'reward'


def test_gpt_mode_validation():
    """Test GPT class mode validation in constructor"""
    # Test valid generator mode
    config_gen = GPTConfig(mode='generator', vocab_size=100, block_size=64)
    model_gen = GPT(config_gen)
    assert model_gen.config.mode == 'generator'
    assert hasattr(model_gen, 'lm_head')
    assert not hasattr(model_gen, 'reward_head')
    
    # Test valid reward mode
    config_reward = GPTConfig(mode='reward', vocab_size=100, block_size=64)
    model_reward = GPT(config_reward)
    assert model_reward.config.mode == 'reward'
    assert hasattr(model_reward, 'reward_head')
    assert not hasattr(model_reward, 'lm_head')
    
    # Test invalid mode
    config_invalid = GPTConfig(mode='invalid', vocab_size=100, block_size=64)
    with pytest.raises(AssertionError, match="mode must be 'generator' or 'reward'"):
        GPT(config_invalid)


def test_generator_mode_forward_pass():
    """Test forward pass output shapes for generator mode"""
    config = GPTConfig(
        mode='generator',
        vocab_size=100,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=128
    )
    model = GPT(config)
    model.eval()
    
    batch_size = 2
    seq_len = 32
    
    # Test forward pass without targets (inference)
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(idx)
    
    # Should return logits for last position only during inference
    assert logits.shape == (batch_size, 1, config.vocab_size)
    assert loss is None
    
    # Test forward pass with targets (training)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(idx, targets)
    
    # Should return full sequence logits during training
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is not None
    assert isinstance(loss.item(), float)


def test_reward_mode_forward_pass():
    """Test forward pass output shapes for reward mode"""
    config = GPTConfig(
        mode='reward',
        vocab_size=100,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=128
    )
    model = GPT(config)
    model.eval()
    
    batch_size = 2
    seq_len = 32
    
    # Test forward pass without targets
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    probabilities, loss = model(idx)
    
    # Should return probability distribution over [natural, synthetic]
    assert probabilities.shape == (batch_size, 2)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(batch_size), atol=1e-6)
    assert loss is None
    
    # Test forward pass with targets
    targets = torch.rand(batch_size, 2)
    targets = targets / targets.sum(dim=1, keepdim=True)  # Normalize to probabilities
    probabilities, loss = model(idx, targets)
    
    assert probabilities.shape == (batch_size, 2)
    assert loss is not None
    assert isinstance(loss.item(), float)


def test_reward_head_architecture():
    """Test reward head architecture components"""
    config = GPTConfig(
        mode='reward',
        vocab_size=100,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=128
    )
    model = GPT(config)
    
    # Check reward head structure
    assert len(model.reward_head) == 4  # Linear, ReLU, Linear, Softmax
    assert isinstance(model.reward_head[0], torch.nn.Linear)
    assert isinstance(model.reward_head[1], torch.nn.ReLU)
    assert isinstance(model.reward_head[2], torch.nn.Linear)
    assert isinstance(model.reward_head[3], torch.nn.Softmax)
    
    # Check dimensions with default hidden size
    assert model.reward_head[0].in_features == config.n_embd
    assert model.reward_head[0].out_features == config.reward_head_hidden_dim
    assert model.reward_head[2].in_features == config.reward_head_hidden_dim
    assert model.reward_head[2].out_features == 2


def test_configurable_reward_head_hidden_dim():
    """Test that reward head hidden dimension is configurable"""
    custom_hidden_dim = 512
    config = GPTConfig(
        mode='reward',
        vocab_size=100,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=128,
        reward_head_hidden_dim=custom_hidden_dim
    )
    model = GPT(config)
    
    # Check that custom hidden dimension is used
    assert model.reward_head[0].out_features == custom_hidden_dim
    assert model.reward_head[2].in_features == custom_hidden_dim
    assert config.reward_head_hidden_dim == custom_hidden_dim


def test_shared_trunk_weights():
    """Test that transformer trunk is properly shared between modes"""
    base_params = dict(vocab_size=100, block_size=64, n_layer=2, n_head=4, n_embd=128)
    
    # Create generator model
    config_gen = GPTConfig(**base_params, mode='generator')
    model_gen = GPT(config_gen)
    
    # Create reward model
    config_reward = GPTConfig(**base_params, mode='reward')
    model_reward = GPT(config_reward)
    
    # Both should have identical transformer components
    assert model_gen.transformer.keys() == model_reward.transformer.keys()
    
    # Check that transformer components have same structure
    for key in model_gen.transformer.keys():
        if key == 'h':  # transformer blocks
            assert len(model_gen.transformer[key]) == len(model_reward.transformer[key])
        else:
            assert type(model_gen.transformer[key]) == type(model_reward.transformer[key])


def test_parameter_counting():
    """Test that parameter counting works correctly for both modes"""
    base_params = dict(
        vocab_size=1000,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=128
    )
    
    # Test generator mode
    config_gen = GPTConfig(**base_params, mode='generator')
    model_gen = GPT(config_gen)
    gen_params = model_gen.get_num_params()
    assert gen_params > 0
    
    # Test reward mode
    config_reward = GPTConfig(**base_params, mode='reward')
    model_reward = GPT(config_reward)
    reward_params = model_reward.get_num_params()
    assert reward_params > 0
    
    # Both models should have reasonable parameter counts
    # The exact relationship depends on vocab_size vs reward_head size
    # Just verify both are positive and different
    assert gen_params != reward_params


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    test_gptconfig_mode_parameter()
    test_gpt_mode_validation()
    test_generator_mode_forward_pass()
    test_reward_mode_forward_pass()
    test_reward_head_architecture()
    test_configurable_reward_head_hidden_dim()
    test_shared_trunk_weights()
    test_parameter_counting()
    print("All tests passed!")


def test_backward_compatibility():
    """Test that existing generator functionality remains unchanged"""
    # Test that default config creates generator mode
    config = GPTConfig(vocab_size=100, block_size=64, n_layer=2, n_head=4, n_embd=128)
    model = GPT(config)
    
    # Should default to generator mode
    assert model.config.mode == 'generator'
    assert hasattr(model, 'lm_head')
    assert not hasattr(model, 'reward_head')
    
    # Should work exactly like before
    batch_size = 2
    seq_len = 32
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass should work as before
    logits, loss = model(idx, targets)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is not None
    
    # Inference should work as before
    logits, loss = model(idx)
    assert logits.shape == (batch_size, 1, config.vocab_size)
    assert loss is None


def test_generate_method_compatibility():
    """Test that generate method still works for generator mode"""
    config = GPTConfig(
        mode='generator',
        vocab_size=100,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=128
    )
    model = GPT(config)
    model.eval()
    
    # Test generation
    start_tokens = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(start_tokens, max_new_tokens=5)
    
    # Should generate 5 additional tokens
    assert generated.shape == (1, 15)
    assert torch.all(generated[:, :10] == start_tokens)


def test_detailed_param_count_compatibility():
    """Test that detailed parameter counting works for both modes"""
    config_gen = GPTConfig(
        mode='generator',
        vocab_size=100,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=128
    )
    model_gen = GPT(config_gen)
    
    # Should work without errors
    detailed_params = model_gen.get_detailed_param_count()
    assert isinstance(detailed_params, dict)
    assert 'total' in detailed_params
    assert detailed_params['total'] > 0
    
    # Test reward mode
    config_reward = GPTConfig(
        mode='reward',
        vocab_size=100,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=128
    )
    model_reward = GPT(config_reward)
    
    # Should also work without errors
    detailed_params_reward = model_reward.get_detailed_param_count()
    assert isinstance(detailed_params_reward, dict)
    assert 'total' in detailed_params_reward
    assert detailed_params_reward['total'] > 0


def test_rotary_embeddings_compatibility():
    """Test that rotary embeddings work with both modes"""
    config_gen = GPTConfig(
        mode='generator',
        vocab_size=100,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=128,
        use_rotary_embeddings=True
    )
    model_gen = GPT(config_gen)
    
    # Should not have position embeddings
    assert 'wpe' not in model_gen.transformer
    
    # Forward pass should work
    idx = torch.randint(0, config_gen.vocab_size, (2, 32))
    logits, loss = model_gen(idx)
    assert logits.shape == (2, 1, config_gen.vocab_size)
    
    # Test reward mode with rotary embeddings
    config_reward = GPTConfig(
        mode='reward',
        vocab_size=100,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=128,
        use_rotary_embeddings=True
    )
    model_reward = GPT(config_reward)
    
    # Should not have position embeddings
    assert 'wpe' not in model_reward.transformer
    
    # Forward pass should work
    probabilities, loss = model_reward(idx)
    assert probabilities.shape == (2, 2)


if __name__ == "__main__":
    # Run all tests manually if pytest is not available
    test_gptconfig_mode_parameter()
    test_gpt_mode_validation()
    test_generator_mode_forward_pass()
    test_reward_mode_forward_pass()
    test_reward_head_architecture()
    test_shared_trunk_weights()
    test_parameter_counting()
    test_backward_compatibility()
    test_generate_method_compatibility()
    test_detailed_param_count_compatibility()
    test_rotary_embeddings_compatibility()
    print("All tests passed!")