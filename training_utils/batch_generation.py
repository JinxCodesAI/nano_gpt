"""
Simplified batch generation - just delegates to dataset interface
"""
from .dataset_interface import DatasetConfig


def get_batch(split: str, dataset_config: DatasetConfig, iter_num: int, batch_size: int, block_size: int, validation_sample_idx=None):
    """Simplified batch loader - delegates to dataset interface"""
    if split == 'val':
        return dataset_config.get_validation_batch(iter_num, validation_sample_idx or 0)
    else:
        return dataset_config.get_training_batch(iter_num, batch_size, block_size)


# All other functions removed and moved to dataset preparation:
# - get_batch_unmasking (200+ lines) → moved to dataset preparation
# - get_batch_remasking_binary (100+ lines) → moved to dataset preparation  
# - All caching logic → moved to dataset_interface.py
# - All masking logic → moved to dataset preparation
# - All validation set creation → moved to dataset preparation