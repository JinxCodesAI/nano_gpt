"""
Compatibility layer for old train_utils module.
This file ensures that checkpoints saved with references to train_utils can still be loaded.
"""

# Import everything from the new training_utils module
from training_utils import *

# Ensure backward compatibility by making this module available as train_utils
import sys
sys.modules["train_utils"] = sys.modules[__name__]
