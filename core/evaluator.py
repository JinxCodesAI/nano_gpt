"""
Model evaluation functionality.

This module provides the Evaluator class that encapsulates loss estimation
logic extracted from the original estimate_loss() function in train.py.
"""

import torch
from typing import Dict


class Evaluator:
    """
    Handles model evaluation over train/val splits.
    
    Encapsulates the exact logic from the original estimate_loss() function
    to maintain identical behavior while improving modularity.
    """
    
    def __init__(
        self, 
        model, 
        consumer, 
        loss_modifier_pipeline, 
        eval_iters: int,
        ctx,
        device: str
    ):
        """
        Initialize evaluator with required components.
        
        Args:
            model: GPT model to evaluate
            consumer: DatasetConsumer for getting batches
            loss_modifier_pipeline: Loss modifier pipeline (temporarily disabled during eval)
            eval_iters: Number of evaluation iterations per split
            ctx: Context manager for autocast (nullcontext or torch.amp.autocast)
            device: Device string for getting batches (e.g., 'cuda', 'cuda:0', 'cpu')
        """
        self.model = model
        self.consumer = consumer
        self.loss_modifier_pipeline = loss_modifier_pipeline
        self.eval_iters = eval_iters
        self.ctx = ctx
        self.device = device
    
    @torch.no_grad()
    def evaluate(self, splits=None) -> Dict[str, float]:
        """
        Estimate loss over specified splits using many batches.
        
        This function replicates the exact logic from the original estimate_loss()
        function to ensure identical behavior.
        
        Args:
            splits: List of splits to evaluate ['train', 'val']. If None, evaluates both.
            
        Returns:
            Dictionary mapping split names to average loss values
        """
        if splits is None:
            splits = ['train', 'val']
        
        out = {}
        self.model.eval()
        
        # Temporarily disable loss modifiers during evaluation to get comparable baseline metrics
        with self.loss_modifier_pipeline.temporarily_disabled():
            for split in splits:
                losses = torch.zeros(self.eval_iters)
                for k in range(self.eval_iters):
                    X, Y = self.consumer.get_batch(split, self.device)
                    with self.ctx:
                        logits, loss = self.model(X, Y, loss_modifiers=self.loss_modifier_pipeline)
                    losses[k] = loss.item()
                out[split] = losses.mean()
        
        self.model.train()
        return out