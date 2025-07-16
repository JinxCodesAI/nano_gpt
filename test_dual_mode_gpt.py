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
    
    # Check dimensions
    assert model.reward_head[0].in_features == config.n_embd
    assert model.reward_head[0].out_features == 256
    assert model.reward_head[2].in_features == 256
    assert model.reward_head[2].out_features == 2


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
    config = GPTConfig(
        vocab_size=100,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=128
    )
    
    # Test generator mode
    config_gen = GPTConfig(**config.__dict__, mode='generator')
    model_gen = GPT(config_gen)
    gen_params = model_gen.get_num_params()
    assert gen_params > 0
    
    # Test reward mode
    config_reward = GPTConfig(**config.__dict__, mode='reward')
    model_reward = GPT(config_reward)
    reward_params = model_reward.get_num_params()
    assert reward_params > 0
    
    # Reward model should have fewer parameters due to smaller head
    # (reward head: n_embd*256 + 256*2 vs lm_head: n_embd*vocab_size)
    expected_diff = config.n_embd * config.vocab_size - (config.n_embd * 256 + 256 * 2)
    assert abs((gen_params - reward_params) - expected_diff) < 100  # Allow small tolerance


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    test_gptconfig_mode_parameter()
    test_gpt_mode_validation()
    test_generator_mode_forward_pass()
    test_reward_mode_forward_pass()
    test_reward_head_architecture()
    test_shared_trunk_weights()
    test_parameter_counting()
    print("All tests passed!")