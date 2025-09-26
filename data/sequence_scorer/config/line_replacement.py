use_all_stages_for_training = True

unmasking_stages = [
    {'type': 'line', 'min_ratio': 0.1, 'max_ratio': 0.3, 'val_loss_stale_count': 8},
    {'type': 'line', 'min_ratio': 0.2, 'max_ratio': 0.4, 'val_loss_stale_count': 10},
    {'type': 'line', 'min_ratio': 0.3, 'max_ratio': 0.5, 'val_loss_stale_count': 12},
    {'type': 'random', 'max_masked_ratio': 0.2, 'val_loss_stale_count': 8},
    {'type': 'line', 'min_ratio': 0.4, 'max_ratio': 0.6, 'val_loss_stale_count': 15},
]

validation_stages = [
    {'type': 'line', 'min_ratio': 0.1, 'max_ratio': 0.3, 'val_loss_stale_count': 8},
    {'type': 'line', 'min_ratio': 0.2, 'max_ratio': 0.4, 'val_loss_stale_count': 10},
    {'type': 'random', 'max_masked_ratio': 0.3, 'val_loss_stale_count': 10},
    {'type': 'line', 'min_ratio': 0.3, 'max_ratio': 0.5, 'val_loss_stale_count': 12},
]
