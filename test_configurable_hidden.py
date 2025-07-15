#!/usr/bin/env python3
"""
Test script to verify the configurable feed forward hidden dimension works correctly.
"""

import sys
import torch
from model import GPT, GPTConfig

def test_default_behavior():
    """Test that default behavior (n_hidden=None) uses 4*n_embd"""
