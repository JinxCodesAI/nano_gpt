#!/usr/bin/env python3
"""
Debug sequence head initialization and output distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def test_sequence_head_init():
    print("=" * 60)
    print("TESTING SEQUENCE HEAD INITIALIZATION")
    print("=" * 60)
    
    # Create sequence head like in model.py
    n_embd = 384  # From optimal5 config
    sequence_head = nn.Sequential(
        nn.Linear(n_embd, 1, bias=False),
        nn.Sigmoid()
    )
    
    # Initialize with small weights (current approach)
    torch.nn.init.normal_(sequence_head[0].weight, mean=0.0, std=0.002)
    
    print("Current initialization:")
    print(f"  Weight shape: {sequence_head[0].weight.shape}")
    print(f"  Weight mean: {sequence_head[0].weight.mean().item():.6f}")
    print(f"  Weight std: {sequence_head[0].weight.std().item():.6f}")
    print(f"  Weight range: [{sequence_head[0].weight.min().item():.6f}, {sequence_head[0].weight.max().item():.6f}]")
    
    # Test with dummy CLS representations
    batch_size = 16
    dummy_cls_features = torch.randn(batch_size, n_embd)
    
    print(f"\nTesting with batch_size={batch_size}")
    print(f"Dummy CLS features shape: {dummy_cls_features.shape}")
    print(f"CLS features mean: {dummy_cls_features.mean().item():.4f}")
    print(f"CLS features std: {dummy_cls_features.std().item():.4f}")
    
    # Get model predictions
    with torch.no_grad():
        logits = sequence_head(dummy_cls_features).squeeze(-1)
    
    print(f"\nModel predictions (current init):")
    print(f"  Predictions shape: {logits.shape}")
    print(f"  Predictions mean: {logits.mean().item():.6f}")
    print(f"  Predictions std: {logits.std().item():.6f}")
    print(f"  Predictions range: [{logits.min().item():.6f}, {logits.max().item():.6f}]")
    print(f"  Individual predictions: {logits[:8].tolist()}")
    
    # Test MSE loss with typical masking ratios
    typical_targets = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.4, 0.3, 0.5, 0.4, 0.6, 0.2, 0.7, 0.5, 0.3, 0.4])
    mse_loss = F.mse_loss(logits, typical_targets)
    
    print(f"\nMSE loss test:")
    print(f"  Typical targets: {typical_targets[:8].tolist()}")
    print(f"  Current predictions: {logits[:8].tolist()}")
    print(f"  MSE Loss: {mse_loss.item():.6f}")
    
    # Compare with different initialization
    print(f"\n" + "=" * 40)
    print("TESTING ALTERNATIVE INITIALIZATION")
    print(f"=" * 40)
    
    # Create new head with different initialization
    sequence_head_alt = nn.Sequential(
        nn.Linear(n_embd, 1, bias=False),
        nn.Sigmoid()
    )
    
    # Use standard Xavier/Glorot initialization
    torch.nn.init.xavier_uniform_(sequence_head_alt[0].weight)
    
    print("Alternative initialization (Xavier):")
    print(f"  Weight mean: {sequence_head_alt[0].weight.mean().item():.6f}")
    print(f"  Weight std: {sequence_head_alt[0].weight.std().item():.6f}")
    print(f"  Weight range: [{sequence_head_alt[0].weight.min().item():.6f}, {sequence_head_alt[0].weight.max().item():.6f}]")
    
    with torch.no_grad():
        logits_alt = sequence_head_alt(dummy_cls_features).squeeze(-1)
    
    print(f"\nModel predictions (Xavier init):")
    print(f"  Predictions mean: {logits_alt.mean().item():.6f}")
    print(f"  Predictions std: {logits_alt.std().item():.6f}")
    print(f"  Predictions range: [{logits_alt.min().item():.6f}, {logits_alt.max().item():.6f}]")
    
    mse_loss_alt = F.mse_loss(logits_alt, typical_targets)
    print(f"  MSE Loss: {mse_loss_alt.item():.6f}")
    
    # Test with even larger initialization
    sequence_head_large = nn.Sequential(
        nn.Linear(n_embd, 1, bias=False),
        nn.Sigmoid()
    )
    
    # Larger std to get more diverse initial predictions
    torch.nn.init.normal_(sequence_head_large[0].weight, mean=0.0, std=0.1)
    
    print(f"\nLarge initialization (std=0.1):")
    print(f"  Weight std: {sequence_head_large[0].weight.std().item():.6f}")
    
    with torch.no_grad():
        logits_large = sequence_head_large(dummy_cls_features).squeeze(-1)
    
    print(f"  Predictions mean: {logits_large.mean().item():.6f}")
    print(f"  Predictions std: {logits_large.std().item():.6f}")
    print(f"  Predictions range: [{logits_large.min().item():.6f}, {logits_large.max().item():.6f}]")
    
    mse_loss_large = F.mse_loss(logits_large, typical_targets)
    print(f"  MSE Loss: {mse_loss_large.item():.6f}")
    
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print(f"=" * 60)
    print(f"Current (std=0.002):  MSE = {mse_loss.item():.6f}, pred_std = {logits.std().item():.6f}")
    print(f"Xavier initialization: MSE = {mse_loss_alt.item():.6f}, pred_std = {logits_alt.std().item():.6f}")
    print(f"Large (std=0.1):      MSE = {mse_loss_large.item():.6f}, pred_std = {logits_large.std().item():.6f}")
    
    if mse_loss.item() < 0.01:
        print(f"\nWARNING: Current initialization produces very low initial loss!")
        print(f"This might explain why training loss starts so low.")
        print(f"Consider using Xavier initialization or larger std.")

if __name__ == "__main__":
    test_sequence_head_init()