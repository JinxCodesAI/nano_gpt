use_all_stages_for_training = True

unmasking_stages = [
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.15, 'p2_probability': 0.3, 'val_loss_stale_count': 6},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    {'type':'random','max_masked_ratio': 0.5, 'val_loss_stale_count': 10},
    {'type': 'span', 'spans_count': 20, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.3, 'p2_probability': 0.1, 'val_loss_stale_count': 8},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    {'type':'random','max_masked_ratio': 0.2, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.55, 'p1_probability': 0.1, 'p2_probability': 0.6, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.9, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 20},
    {'type': 'line', 'min_ratio': 0.1, 'max_ratio': 0.3, 'val_loss_stale_count': 10}
]

validation_stages = [
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.15, 'p2_probability': 0.3, 'val_loss_stale_count': 6},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    {'type':'random','max_masked_ratio': 0.5, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.3, 'p2_probability': 0.1, 'val_loss_stale_count': 8},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    {'type':'random','max_masked_ratio': 0.2, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.2, 'p1_probability': 0.3, 'p2_probability': 0.0, 'val_loss_stale_count': 2},
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.3, 'p2_probability': 0.0, 'val_loss_stale_count': 4},
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.15, 'p2_probability': 0.3, 'val_loss_stale_count': 6},
    {'type':'sticky','target_masked_ratio': 0.55, 'p1_probability': 0.1, 'p2_probability': 0.6, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.7, 'p1_probability': 0.2, 'p2_probability': 0.4, 'val_loss_stale_count': 15},
    {'type':'sticky','target_masked_ratio': 0.8, 'p1_probability': 0.2, 'p2_probability': 0.4, 'val_loss_stale_count': 20},
    {'type':'sticky','target_masked_ratio': 0.8, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 20},
    {'type':'sticky','target_masked_ratio': 0.9, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 20},
]