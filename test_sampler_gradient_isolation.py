"""
Test that sampler head training is truly isolated from main model.
Verifies that gradients from sampler loss do NOT flow to:
- Transformer layers
- Token embeddings (wte)
- Position embeddings (wpe)
"""
import torch
from model import GPTConfig, GPT, ModelMode


def test_sampler_gradient_isolation():
    """
    Test that sampler loss gradients do NOT affect transformer or embeddings.
    
    This is critical for the auxiliary network design.
    """
    print("Testing sampler gradient isolation...")
    
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
        vocab_size=100,
        mode=ModelMode.LANGUAGE_MODEL,
        attention_type='bidirectional',
        add_sampler_head=True,
        mask_token_id=99,
        start_sampler_iteration=0
    )
    model = GPT(config)
    model.train()
    model._current_iter = 1000  # Enable sampler
    
    # Create training batch
    batch_size = 2
    seq_len = 16
    idx = torch.randint(0, 98, (batch_size, seq_len))
    targets = torch.randint(0, 98, (batch_size, seq_len))
    
    # Mask some positions
    mask_prob = torch.rand(batch_size, seq_len) < 0.3
    idx[mask_prob] = 99
    
    # Store initial parameters
    initial_wte = model.transformer.wte.weight.data.clone()
    initial_wpe = model.transformer.wpe.weight.data.clone()
    initial_transformer_block0_ln1 = model.transformer.h[0].ln_1.weight.data.clone()
    
    # Forward pass to get sampler loss only
    logits, loss = model(idx, targets)
    
    # Get sampler loss component
    sampler_loss = model._last_sampler_loss
    
    if sampler_loss == 0.0:
        print("  FAIL: Sampler loss is 0, cannot test gradient isolation")
        return False
    
    print(f"  Sampler loss: {sampler_loss:.4f}")
    
    # Compute only sampler loss (isolate it from main loss)
    model.zero_grad()
    
    # Re-run forward to get sampler loss
    hidden_states = model._encode_tokens(idx)
    sampler_loss_tensor = model._compute_sampler_loss(hidden_states, idx, targets)
    
    if sampler_loss_tensor is None:
        print("  FAIL: Could not compute sampler loss")
        return False
    
    # Backward pass with only sampler loss
    sampler_loss_tensor.backward()
    
    # Check gradients
    print("\n  Checking gradients...")
    
    # 1. Sampler head SHOULD have gradients
    sampler_has_grad = False
    for name, param in model.sampler_head.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            sampler_has_grad = True
            print(f"    ✓ sampler_head.{name}: has gradients (expected)")
            break
    
    if not sampler_has_grad:
        print("    ✗ FAIL: Sampler head has no gradients!")
        return False
    
    # 2. Transformer layers should NOT have gradients
    transformer_has_grad = False
    for name, param in model.transformer.h.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            transformer_has_grad = True
            print(f"    ✗ FAIL: transformer.h.{name} has gradients (should be isolated!)")
            break
    
    if not transformer_has_grad:
        print(f"    ✓ Transformer layers: no gradients (expected)")
    else:
        return False
    
    # 3. Token embeddings (wte) should NOT have gradients
    if model.transformer.wte.weight.grad is not None and model.transformer.wte.weight.grad.abs().sum() > 0:
        print(f"    ✗ FAIL: wte (token embeddings) has gradients (should be isolated!)")
        return False
    else:
        print(f"    ✓ Token embeddings (wte): no gradients (expected)")
    
    # 4. Position embeddings (wpe) should NOT have gradients
    if model.transformer.wpe.weight.grad is not None and model.transformer.wpe.weight.grad.abs().sum() > 0:
        print(f"    ✗ FAIL: wpe (position embeddings) has gradients (should be isolated!)")
        return False
    else:
        print(f"    ✓ Position embeddings (wpe): no gradients (expected)")
    
    # 5. LM head should NOT have gradients (sampler has its own output_head)
    if model.lm_head.weight.grad is not None and model.lm_head.weight.grad.abs().sum() > 0:
        print(f"    ✗ FAIL: lm_head has gradients (should use sampler's own output_head!)")
        return False
    else:
        print(f"    ✓ LM head: no gradients (expected - sampler has own output_head)")
    
    print("\n  ✓ PASS: Sampler training is completely isolated from main model")
    return True


def test_main_loss_affects_transformer():
    """
    Sanity check: verify that main LM loss DOES affect transformer.
    This ensures our test methodology is correct.
    """
    print("\nSanity check: Main loss should affect transformer...")
    
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
        vocab_size=100,
        mode=ModelMode.LANGUAGE_MODEL,
        attention_type='bidirectional',
        add_sampler_head=False,  # No sampler for this test
        mask_token_id=99
    )
    model = GPT(config)
    model.train()
    
    # Create training batch
    batch_size = 2
    seq_len = 16
    idx = torch.randint(0, 98, (batch_size, seq_len))
    targets = torch.randint(0, 98, (batch_size, seq_len))
    
    # Forward + backward
    logits, loss = model(idx, targets)
    model.zero_grad()
    loss.backward()
    
    # Check that transformer HAS gradients
    transformer_has_grad = False
    for name, param in model.transformer.h.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            transformer_has_grad = True
            break
    
    if transformer_has_grad:
        print("  ✓ PASS: Main loss correctly affects transformer (sanity check passed)")
        return True
    else:
        print("  ✗ FAIL: Main loss doesn't affect transformer (test methodology broken!)")
        return False


def test_combined_loss_isolation():
    """
    Test that when both main and sampler losses are combined,
    sampler gradients still don't affect transformer.
    """
    print("\nTesting combined loss (main + sampler) isolation...")
    
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
        vocab_size=100,
        mode=ModelMode.LANGUAGE_MODEL,
        attention_type='bidirectional',
        add_sampler_head=True,
        mask_token_id=99,
        start_sampler_iteration=0
    )
    model = GPT(config)
    model.train()
    model._current_iter = 1000  # Enable sampler
    
    # Create training batch
    batch_size = 2
    seq_len = 16
    idx = torch.randint(0, 98, (batch_size, seq_len))
    targets = torch.randint(0, 98, (batch_size, seq_len))
    
    # Mask some positions
    mask_prob = torch.rand(batch_size, seq_len) < 0.3
    idx[mask_prob] = 99
    
    # Forward pass (includes both main and sampler loss)
    logits, total_loss = model(idx, targets)
    
    main_loss = model._last_lm_loss
    sampler_loss = model._last_sampler_loss
    
    print(f"  Main loss: {main_loss:.4f}")
    print(f"  Sampler loss: {sampler_loss:.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")
    
    # Backward pass with combined loss
    model.zero_grad()
    total_loss.backward()
    
    # Check gradients
    print("\n  Checking gradients after combined backward...")
    
    # 1. Sampler head SHOULD have gradients
    sampler_has_grad = any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in model.sampler_head.parameters()
    )
    
    if sampler_has_grad:
        print(f"    ✓ Sampler head: has gradients (expected)")
    else:
        print(f"    ✗ FAIL: Sampler head has no gradients")
        return False
    
    # 2. Transformer SHOULD have gradients (from main loss)
    transformer_has_grad = any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in model.transformer.h.parameters()
    )
    
    if transformer_has_grad:
        print(f"    ✓ Transformer: has gradients from main loss (expected)")
    else:
        print(f"    ✗ FAIL: Transformer has no gradients (main loss should affect it)")
        return False
    
    # 3. The key test: sampler's output_head should have gradients,
    #    but lm_head should ONLY have gradients from main loss
    #    (we can't easily separate them, but we verified isolation in test 1)
    
    print("\n  ✓ PASS: Combined loss works correctly")
    print("    - Main loss affects transformer ✓")
    print("    - Sampler loss affects sampler head ✓")
    print("    - Sampler uses separate output_head (verified in test 1) ✓")
    return True


def main():
    print("="*60)
    print("SAMPLER GRADIENT ISOLATION TESTS")
    print("="*60)
    print("\nThese tests verify that sampler training is truly auxiliary")
    print("and does NOT affect transformer layers or embeddings.\n")
    
    all_passed = True
    
    all_passed &= test_sampler_gradient_isolation()
    all_passed &= test_main_loss_affects_transformer()
    all_passed &= test_combined_loss_isolation()
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL GRADIENT ISOLATION TESTS PASSED ✓")
        print("\nSampler is truly auxiliary:")
        print("  - Sampler gradients do NOT affect transformer")
        print("  - Sampler gradients do NOT affect embeddings")
        print("  - Sampler has its own separate output_head")
        print("  - Main loss still trains transformer normally")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

