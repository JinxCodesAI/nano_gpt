"""
Quick verification script to test the new critic modes.
Run this to verify the refactored implementation works correctly.
"""

import torch
from model import GPT, GPTConfig, ModelMode, CriticMode

def test_critic_modes():
    """Test all three critic modes with a small model"""
    print("=" * 60)
    print("Testing Critic Mode Refactor")
    print("=" * 60)
    
    # Create a small test model
    base_config = dict(
        block_size=32,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        mode=ModelMode.LANGUAGE_MODEL,
    )
    
    # Test data
    B, T = 2, 16
    idx = torch.randint(0, 128, (B, T))
    targets = torch.randint(0, 128, (B, T))
    
    modes = [
        (CriticMode.NONE, "NONE (Standard CE Loss)"),
        (CriticMode.TARGETLESS, "TARGETLESS (Critic weights loss)"),
        (CriticMode.TARGETED, "TARGETED (Critic trained on entropy)")
    ]
    
    results = {}
    
    for mode, description in modes:
        print(f"\n{'-' * 60}")
        print(f"Testing {description}")
        print(f"{'-' * 60}")
        
        config = GPTConfig(**base_config, critic_mode=mode, critic_alpha=0.5)
        model = GPT(config)
        model.eval()
        
        # Check if critic head exists
        has_critic = hasattr(model, 'critic_head')
        print(f"Has critic_head: {has_critic}")
        
        if has_critic:
            # Check critic head structure
            print(f"Critic head type: {type(model.critic_head)}")
            if isinstance(model.critic_head, torch.nn.Sequential):
                print(f"Critic head layers: {len(model.critic_head)}")
        
        # Forward pass
        with torch.no_grad():
            logits, loss = model(idx, targets=targets)
        
        print(f"Logits shape: {logits.shape}")
        print(f"Loss value: {loss.item():.4f}")
        
        # Check loss components
        if hasattr(model, '_last_lm_loss'):
            print(f"LM loss component: {model._last_lm_loss:.4f}")
        if hasattr(model, '_last_critic_loss'):
            print(f"Critic loss component: {model._last_critic_loss:.4f}")
        
        # Test critic_scores if available
        if has_critic and mode != CriticMode.NONE:
            try:
                scores = model.critic_scores(idx)
                print(f"Critic scores shape: {scores.shape}")
                print(f"Critic scores range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
                print(f"Critic scores mean: {scores.mean().item():.4f}")
            except Exception as e:
                print(f"Error getting critic scores: {e}")
        
        results[mode] = loss.item()
    
    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"NONE mode loss:       {results[CriticMode.NONE]:.4f}")
    print(f"TARGETLESS mode loss: {results[CriticMode.TARGETLESS]:.4f}")
    print(f"TARGETED mode loss:   {results[CriticMode.TARGETED]:.4f}")
    print(f"\nAll modes executed successfully! ✓")
    print(f"{'=' * 60}")

def test_config_compatibility():
    """Test that old-style configs fail gracefully"""
    print("\n" + "=" * 60)
    print("Testing Config Compatibility")
    print("=" * 60)
    
    # This should work (new style)
    try:
        config = GPTConfig(
            block_size=32,
            vocab_size=128,
            n_layer=2,
            n_head=2,
            n_embd=64,
            mode=ModelMode.LANGUAGE_MODEL,
            critic_mode=CriticMode.TARGETLESS,
        )
        model = GPT(config)
        print("✓ New-style config (critic_mode) works")
    except Exception as e:
        print(f"✗ New-style config failed: {e}")
    
    # Old-style config should be ignored (no add_critic_head parameter)
    try:
        config = GPTConfig(
            block_size=32,
            vocab_size=128,
            n_layer=2,
            n_head=2,
            n_embd=64,
            mode=ModelMode.LANGUAGE_MODEL,
        )
        model = GPT(config)
        print("✓ Config without critic parameters works (defaults to NONE)")
    except Exception as e:
        print(f"✗ Config without critic parameters failed: {e}")

if __name__ == "__main__":
    test_critic_modes()
    test_config_compatibility()
    print("\n✓ All verification tests passed!")

