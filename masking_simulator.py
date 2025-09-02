#!/usr/bin/env python3
"""
Masking Simulator for Debugging Batch Generation
Simulates the masking behavior without actual training to analyze clustering patterns.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pickle

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training_utils import TrainingContext, apply_target_driven_sticky_masking_gpu, UnmaskingStage
from utils import analyze_clustering

@dataclass
class SimulationConfig:
    """Configuration for masking simulation"""
    batch_size: int
    block_size: int
    
    # Stage-based unmasking parameters
    stages: List[Dict] = None
    mask_token_id: int = 50304
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_stages: List[int] = None
    
    def __post_init__(self):
        if self.stages is None:
            # Default stages for testing
            self.stages = [
                {'target_masked_ratio': 0.2, 'p1_probability': 0.3, 'p2_probability': 0.0},
                {'target_masked_ratio': 0.4, 'p1_probability': 0.3, 'p2_probability': 0.0},
                {'target_masked_ratio': 0.4, 'p1_probability': 0.1, 'p2_probability': 0.5},
                {'target_masked_ratio': 0.6, 'p1_probability': 0.3, 'p2_probability': 0.1},
                {'target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.8},
            ]
        if self.test_stages is None:
            # Test all stages by default
            self.test_stages = list(range(len(self.stages)))

class MaskingSimulator:
    """Simulator for testing masking behavior across training iterations"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.results = {}
        
        print(f"Masking Simulator initialized on {self.device}")
        print(f"Config: {config}")
    
    def create_dummy_data(self) -> torch.Tensor:
        """Create dummy token data for testing"""
        # Create realistic token sequences (0-50256 for GPT-2 vocab)
        vocab_size = 50257
        data = torch.randint(0, vocab_size, 
                           (self.config.batch_size, self.config.block_size), 
                           device=self.device)
        return data
    
    def create_training_context(self, stage_idx: int) -> TrainingContext:
        """Create TrainingContext for specific stage"""
        # Convert stage configs to UnmaskingStage objects
        unmasking_stage_objects = [
            UnmaskingStage(
                target_masked_ratio=stage['target_masked_ratio'],
                p1_probability=stage['p1_probability'],
                p2_probability=stage['p2_probability'],
                val_loss_stale_count=5  # Default stale count
            ) for stage in self.config.stages
        ]
        
        return TrainingContext(
            training_type='unmasking',
            batch_size=self.config.batch_size,
            block_size=self.config.block_size,
            device=str(self.device),
            device_type=self.device.type,
            seed_offset=0,
            data_dir='data/shakespeare_char',  # Default data dir
            meta_vocab_size=65,  # Shakespeare char vocab size
            mask_token_id=self.config.mask_token_id,
            wrong_token_id=66,  # meta_vocab_size + 1
            remask_good_id=67,   # meta_vocab_size + 2
            remask_wrong_id=68,  # meta_vocab_size + 3
            extended_vocab_size=65 + 15,  # meta_vocab_size + 15 (reserve 15 special tokens)
            iter_num=0,  # Not used in stage-based system
            current_stage=stage_idx,
            unmasking_stages=unmasking_stage_objects,
            remasking_corruption_strategy='mixed',  # Default strategy
            remasking_strategy_weights=[0.25, 0.4, 0.25, 0.1],  # Default weights
            eval_iters=20,  # Default eval iters
            warmup_iters=2000,  # Default warmup
            lr_decay_iters=8000,  # Default decay
            learning_rate=1e-4,  # Default lr
            min_lr=1e-5  # Default min lr
        )
    
    def analyze_single_stage(self, stage_idx: int) -> Dict:
        """Analyze masking behavior for a single stage"""
        print(f"\n=== Analyzing Stage {stage_idx} ===")
        
        # Create context and data
        ctx = self.create_training_context(stage_idx)
        x = self.create_dummy_data()
        
        # Get stage configuration
        stage_config = ctx.get_current_stage_config()
        
        # Apply target-driven sticky masking
        masked_x, mask = apply_target_driven_sticky_masking_gpu(
            x,
            stage_config.target_masked_ratio,
            stage_config.p1_probability,
            stage_config.p2_probability,
            ctx.mask_token_id,
            ctx.meta_vocab_size
        )
        
        # Calculate statistics
        mask_ratio = mask.float().mean().item()
        cluster_stats = analyze_clustering(mask)
        
        # Calculate cumulative cluster size distribution
        cluster_size_distribution = self._analyze_cluster_size_distribution(mask)
        
        results = {
            'stage_idx': stage_idx,
            'target_masked_ratio': stage_config.target_masked_ratio,
            'actual_mask_ratio': mask_ratio,
            'p1_probability': stage_config.p1_probability,
            'p2_probability': stage_config.p2_probability,
            'avg_cluster_size': cluster_stats['avg_cluster_size'],
            'max_cluster_size': cluster_stats['max_cluster_size'],
            'num_clusters_per_batch': cluster_stats['num_clusters_per_batch'],
            'cluster_size_distribution': cluster_size_distribution,
            'ratio_accuracy': abs(mask_ratio - stage_config.target_masked_ratio)
        }
        
        # Print detailed analysis
        print(f"Target mask ratio: {stage_config.target_masked_ratio:.4f}")
        print(f"Actual mask ratio: {mask_ratio:.4f} (diff: {results['ratio_accuracy']:.4f})")
        print(f"P1 probability: {stage_config.p1_probability:.3f}")
        print(f"P2 probability: {stage_config.p2_probability:.3f}")
        print(f"Avg cluster size: {cluster_stats['avg_cluster_size']:.3f}")
        print(f"Max cluster size: {cluster_stats['max_cluster_size']}")
        print(f"Clusters per batch: {cluster_stats['num_clusters_per_batch']:.1f}")
        
        return results
    
    def _analyze_cluster_size_distribution(self, mask: torch.Tensor) -> Dict:
        """Analyze cumulative cluster size distribution"""
        # Convert mask to numpy for easier processing
        mask_np = mask.cpu().numpy()
        
        # Find all clusters across all sequences in the batch
        all_cluster_sizes = []
        total_masked_tokens = 0
        
        for batch_idx in range(mask_np.shape[0]):
            sequence_mask = mask_np[batch_idx]
            total_masked_tokens += sequence_mask.sum()
            
            # Find connected components (clusters) in this sequence
            in_cluster = False
            current_cluster_size = 0
            
            for pos in range(len(sequence_mask)):
                if sequence_mask[pos]:  # Position is masked
                    if not in_cluster:
                        # Start of new cluster
                        in_cluster = True
                        current_cluster_size = 1
                    else:
                        # Continue existing cluster
                        current_cluster_size += 1
                else:  # Position is not masked
                    if in_cluster:
                        # End of cluster
                        all_cluster_sizes.append(current_cluster_size)
                        in_cluster = False
                        current_cluster_size = 0
            
            # Handle cluster that extends to end of sequence
            if in_cluster:
                all_cluster_sizes.append(current_cluster_size)
        
        if not all_cluster_sizes or total_masked_tokens == 0:
            return {
                'cumulative_distribution': {},
                'max_cluster_size': 0,
                'total_masked_tokens': 0
            }
        
        # Calculate cumulative distribution
        max_cluster_size = max(all_cluster_sizes)
        cumulative_distribution = {}
        
        for size_threshold in range(1, max_cluster_size + 1):
            # Count tokens in clusters of size <= size_threshold
            tokens_in_small_clusters = sum(
                cluster_size for cluster_size in all_cluster_sizes 
                if cluster_size <= size_threshold
            )
            percentage = (tokens_in_small_clusters / total_masked_tokens) * 100.0
            cumulative_distribution[size_threshold] = percentage
        
        return {
            'cumulative_distribution': cumulative_distribution,
            'max_cluster_size': max_cluster_size,
            'total_masked_tokens': total_masked_tokens
        }
    
    def run_full_simulation(self) -> Dict:
        """Run simulation across all test stages"""
        print("Running full masking simulation...")
        print(f"Testing stages: {self.config.test_stages}")
        
        all_results = []
        
        for stage_idx in self.config.test_stages:
            results = self.analyze_single_stage(stage_idx)
            all_results.append(results)
        
        self.results = {
            'config': self.config,
            'stages': all_results
        }
        
        return self.results
    
    def print_mask_ratio_summary(self):
        """Print clear summary of actual mask ratios"""
        if not self.results:
            print("No results to analyze. Run simulation first.")
            return
        
        stages = self.results['stages']
        
        print("\n" + "="*80)
        print("STAGE-BASED MASK RATIO ANALYSIS")
        print("="*80)
        print(f"{'Stage':>6} | {'Target':>8} | {'Actual':>8} | {'Error':>8} | {'P1':>6} | {'P2':>6} | {'Avg Cluster':>12}")
        print(f"{'':>6} | {'Ratio':>8} | {'Ratio':>8} | {'':>8} | {'':>6} | {'':>6} | {'Size':>12}")
        print("-" * 80)
        
        for r in stages:
            print(f"{r['stage_idx']:6d} | {r['target_masked_ratio']:8.3f} | {r['actual_mask_ratio']:8.3f} | {r['ratio_accuracy']:8.4f} | {r['p1_probability']:6.3f} | {r['p2_probability']:6.3f} | {r['avg_cluster_size']:12.3f}")
        
        print("\nKEY:")
        print("- Target Ratio: Desired masking ratio for the stage")
        print("- Actual Ratio: Real fraction of tokens that are masked")
        print("- Error: Absolute difference between target and actual")
        print("- P1: Cluster initiation probability") 
        print("- P2: Cluster extension probability")
        print("- Avg Cluster Size: Average size of masked clusters")
    
    def diagnose_issues(self) -> Dict:
        """Diagnose potential issues with clustering"""
        print("\n" + "="*60)
        print("STAGE-BASED DIAGNOSTIC ANALYSIS")
        print("="*60)
        
        if not self.results:
            print("No results to analyze. Run simulation first.")
            return {}
        
        stages = self.results['stages']
        
        # Analyze accuracy of target ratios
        ratio_errors = [r['ratio_accuracy'] for r in stages]
        cluster_sizes = [r['avg_cluster_size'] for r in stages]
        
        print(f"\n1. TARGET RATIO ACCURACY:")
        print(f"   Average error: {np.mean(ratio_errors):.4f}")
        print(f"   Max error: {max(ratio_errors):.4f}")
        print(f"   Expected: < 0.05 for good accuracy")
        
        print(f"\n2. CLUSTER SIZE PROGRESSION:")
        print(f"   Smallest avg: {min(cluster_sizes):.3f} -> Largest avg: {max(cluster_sizes):.3f}")
        print(f"   Range: {max(cluster_sizes) - min(cluster_sizes):.3f}")
        
        # Identify issues
        issues = []
        
        # Issue 1: Poor ratio accuracy
        if np.mean(ratio_errors) > 0.05:
            issues.append(f"ISSUE: Poor target ratio accuracy (avg error: {np.mean(ratio_errors):.4f})")
            print(f"\n❌ ISSUE: Poor target ratio accuracy")
        
        # Issue 2: Low cluster sizes
        max_cluster_ever = max(r['max_cluster_size'] for r in stages)
        if max_cluster_ever < 5:
            issues.append(f"ISSUE: Maximum cluster size very small ({max_cluster_ever})")
            print(f"\n❌ ISSUE: Max cluster size across all stages: {max_cluster_ever}")
        
        if not issues:
            print(f"\n✅ No major issues detected!")
        else:
            print(f"\n📋 SUMMARY: {len(issues)} issues found")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
        
        return {
            'issues': issues,
            'stats': {
                'avg_ratio_error': np.mean(ratio_errors),
                'max_ratio_error': max(ratio_errors),
                'cluster_size_range': max(cluster_sizes) - min(cluster_sizes),
                'max_cluster_ever': max_cluster_ever
            }
        }
    
    def plot_results(self, save_path: str = None):
        """Plot simulation results with stage-based parameter information"""
        if not self.results:
            print("No results to plot. Run simulation first.")
            return
        
        stages = self.results['stages']
        stage_indices = [r['stage_idx'] for r in stages]
        config = self.config
        
        # Create figure with more space for parameter box
        fig = plt.figure(figsize=(18, 12))
        
        # Create main plots area and parameter box area
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.4], hspace=0.3, wspace=0.3)
        
        # Plot 1: Target vs Actual mask ratios
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(stage_indices, [r['target_masked_ratio'] for r in stages], 'g-', label='Target Mask Ratio', linewidth=2, marker='s', markersize=4)
        ax1.plot(stage_indices, [r['actual_mask_ratio'] for r in stages], 'b-', label='Actual Mask Ratio', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Stage')
        ax1.set_ylabel('Mask Ratio')
        ax1.set_title('Target vs Actual Mask Ratios')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cluster Size Distribution (for multiple stages)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Plot cumulative distributions for different stages
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
        
        has_data = False
        for i, stage_result in enumerate(stages[::max(1, len(stages)//4)]):  # Sample stages
            cluster_dist = stage_result['cluster_size_distribution']
            
            if cluster_dist['cumulative_distribution']:
                cluster_sizes = sorted(cluster_dist['cumulative_distribution'].keys())
                percentages = [cluster_dist['cumulative_distribution'][size] for size in cluster_sizes]
                
                color_idx = i % len(colors)
                ax2.plot(cluster_sizes, percentages, color=colors[color_idx], linewidth=2, 
                        marker='o', markersize=3, label=f'Stage {stage_result["stage_idx"]} (ratio={stage_result["target_masked_ratio"]:.1f})',
                        alpha=0.8)
                has_data = True
        
        if has_data:
            ax2.set_xlabel('Cluster Size')
            ax2.set_ylabel('Cumulative % of Masked Tokens')
            ax2.set_title('Cumulative Cluster Distribution (Selected Stages)')
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)
            
            # Add reference lines
            ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
            ax2.axhline(y=90, color='gray', linestyle=':', alpha=0.5, label='90%')
            ax2.legend(fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'No cluster data available', 
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('Cumulative Cluster Distribution')
        
        # Plot 3: Number of clusters and ratio accuracy
        ax3 = fig.add_subplot(gs[0, 2])
        ax3_twin = ax3.twinx()
        
        # Clusters on left y-axis
        line1 = ax3.plot(stage_indices, [r['num_clusters_per_batch'] for r in stages], 'purple', linewidth=2, marker='d', markersize=4, label='Clusters/Batch')
        ax3.set_xlabel('Stage')
        ax3.set_ylabel('Clusters per Batch', color='purple')
        ax3.tick_params(axis='y', labelcolor='purple')
        ax3.grid(True, alpha=0.3)
        
        # Ratio accuracy on right y-axis
        line2 = ax3_twin.plot(stage_indices, [r['ratio_accuracy'] for r in stages], 'red', linewidth=2, marker='x', markersize=6, label='Ratio Error')
        ax3_twin.set_ylabel('Target Ratio Error', color='red')
        ax3_twin.tick_params(axis='y', labelcolor='red')
        
        ax3.set_title('Clusters and Target Accuracy')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
        
        # Plot 4: Probabilities
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(stage_indices, [r['p1_probability'] for r in stages], 'cyan', label='P1 (initiation)', linewidth=2, marker='o', markersize=4)
        ax4.plot(stage_indices, [r['p2_probability'] for r in stages], 'magenta', label='P2 (extension)', linewidth=2, marker='s', markersize=4)
        ax4.set_xlabel('Stage')
        ax4.set_ylabel('Probability')
        ax4.set_title('Masking Probabilities by Stage')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Cluster size progression
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(stage_indices, [r['avg_cluster_size'] for r in stages], 'b-', label='Avg Cluster Size', linewidth=2, marker='o', markersize=4)
        ax5.plot(stage_indices, [r['max_cluster_size'] for r in stages], 'r--', label='Max Cluster Size', linewidth=2, marker='s', markersize=4)
        ax5.set_xlabel('Stage')
        ax5.set_ylabel('Cluster Size')
        ax5.set_title('Cluster Size by Stage')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Ratio accuracy scatter
        ax6 = fig.add_subplot(gs[1, 2])
        ratio_errors = [r['ratio_accuracy'] for r in stages]
        target_ratios = [r['target_masked_ratio'] for r in stages]
        ax6.scatter(target_ratios, ratio_errors, c=stage_indices, cmap='viridis', s=80, alpha=0.8)
        ax6.set_xlabel('Target Mask Ratio')
        ax6.set_ylabel('Ratio Error')
        ax6.set_title('Target Accuracy vs Ratio')
        cbar = plt.colorbar(ax6.collections[0], ax=ax6)
        cbar.set_label('Stage Index')
        ax6.grid(True, alpha=0.3)
        
        # Parameter information box
        param_ax = fig.add_subplot(gs[2, :])
        param_ax.axis('off')
        
        # Create parameter info text for stage-based system
        num_stages = len(config.stages)
        param_text = f"""STAGE-BASED SIMULATION PARAMETERS:
Batch Size: {config.batch_size} | Block Size: {config.block_size} | Device: {config.device} | Total Stages: {num_stages}

STAGE CONFIGURATIONS:
"""
        
        # Add first few stages as examples
        for i, stage in enumerate(config.stages[:min(5, len(config.stages))]):
            param_text += f"Stage {i}: ratio={stage['target_masked_ratio']:.1f}, p1={stage['p1_probability']:.1f}, p2={stage['p2_probability']:.1f}"
            if i < min(4, len(config.stages)-1):
                param_text += " | "
            if i == 4 and len(config.stages) > 5:
                param_text += f" | ... (+{len(config.stages)-5} more)"
            
        param_text += f"""

RESULTS SUMMARY:
Avg ratio error: {np.mean([r['ratio_accuracy'] for r in stages]):.4f} | Max ratio error: {max([r['ratio_accuracy'] for r in stages]):.4f}
Min avg cluster: {min([r['avg_cluster_size'] for r in stages]):.3f} | Max avg cluster: {max([r['avg_cluster_size'] for r in stages]):.3f}
Target ratios: {min([r['target_masked_ratio'] for r in stages]):.1f} → {max([r['target_masked_ratio'] for r in stages]):.1f}"""
        
        param_ax.text(0.02, 0.98, param_text, transform=param_ax.transAxes, 
                     verticalalignment='top', horizontalalignment='left',
                     fontsize=9, fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        # Add main title
        fig.suptitle(f'Stage-Based Masking Simulation Results\n{num_stages} Stages: Target Ratios {min([r["target_masked_ratio"] for r in stages]):.1f}-{max([r["target_masked_ratio"] for r in stages]):.1f}', 
                     fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Stage-based plot saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, filepath: str):
        """Save simulation results"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Results saved to {filepath}")

def main():
    """Main execution function"""
    print("Stage-Based Masking Behavior Simulator")
    print("=" * 50)
    
    # Create configuration with stages matching train_run.py
    stages =  [
    {'target_masked_ratio': 0.2, 'p1_probability': 0.3, 'p2_probability': 0.0, 'val_loss_stale_count': 3},
    {'target_masked_ratio': 0.4, 'p1_probability': 0.3, 'p2_probability': 0.0, 'val_loss_stale_count': 3},
    {'target_masked_ratio': 0.4, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 3},
    {'target_masked_ratio': 0.6, 'p1_probability': 0.3, 'p2_probability': 0.1, 'val_loss_stale_count': 6},
    {'target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.8, 'val_loss_stale_count': 6},
    {'target_masked_ratio': 0.7, 'p1_probability': 0.2, 'p2_probability': 0.4, 'val_loss_stale_count': 6},
    {'target_masked_ratio': 0.8, 'p1_probability': 0.2, 'p2_probability': 0.4, 'val_loss_stale_count': 6},
    {'target_masked_ratio': 0.8, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 6},
    {'target_masked_ratio': 0.9, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 6},
]
    
    config = SimulationConfig(
        batch_size=16,  # Match current train_run.py batch_size
        block_size=1024,
        stages=stages
    )
    
    # Create and run simulator
    simulator = MaskingSimulator(config)
    
    # Run full simulation
    results = simulator.run_full_simulation()
    
    # Print mask ratio summary
    simulator.print_mask_ratio_summary()
    
    # Diagnose issues
    diagnosis = simulator.diagnose_issues()
    
    # Plot results
    try:
        simulator.plot_results('stage_based_masking_simulation.png')
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    # Save results
    simulator.save_results('masking_simulation_results.pkl')
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()