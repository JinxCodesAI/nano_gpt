"""Example stage composition for the random replacement dataset."""

# Train on every stage sequentially until the specified validation patience
# (``val_loss_stale_count``) is reached for that stage. This mirrors the
# behaviour of the ``char_diffusion`` dataset configs.
use_all_stages_for_training = True

# A compact but diverse training curriculum that exercises all masking modes.
unmasking_stages = [
    {
        "type": "sticky",
        "target_masked_ratio": 0.25,
        "p1_probability": 0.2,
        "p2_probability": 0.3,
        "val_loss_stale_count": 4,
    },
    {
        "type": "random",
        "max_masked_ratio": 0.4,
        "val_loss_stale_count": 6,
    },
    {
        "type": "span",
        "spans_count": 16,
        "val_loss_stale_count": 6,
    },
]

# Mirror the training schedule during validation so metrics remain comparable.
validation_stages = [
    {
        "type": "sticky",
        "target_masked_ratio": 0.25,
        "p1_probability": 0.2,
        "p2_probability": 0.3,
        "val_loss_stale_count": 4,
    },
    {
        "type": "random",
        "max_masked_ratio": 0.4,
        "val_loss_stale_count": 6,
    },
    {
        "type": "span",
        "spans_count": 16,
        "val_loss_stale_count": 6,
    },
]
