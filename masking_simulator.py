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

from train_utils import TrainingContext, apply_gpu_masking_training
from utils import analyze_clustering, analyze_masking_patterns_with_transition

@dataclass
class SimulationConfig:
    """Configuration for masking simulation"""
    batch_size: int
    block_size: int
    
    # Training progression settings
    max_iters: int
    
    # Masking parameters (from train_run.py)
    guaranteed_unmasked_max: float
    guaranteed_unmasked_min: float 
    random_mask_warmup: int
    
    # Sticky masking parameters
    sticky_rounds: int
    sticky_p1_p2_multiplier: float 
    sticky_p1_divisor: float 
    sticky_transition_start: int
    sticky_transition_end: int
    mask_token_id: int = 50304
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_iterations: List[int] = None
    
    def __post_init__(self):
        if self.test_iterations is None:
            # Generate test iterations based on transition points and max_iters
            # Create 20 uniformly spaced samples over max_iters
            self.test_iterations = [int(i * self.max_iters / 19) for i in range(20)]

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
    
    def create_training_context(self, iter_num: int) -> TrainingContext:
        """Create TrainingContext for specific iteration"""
        return TrainingContext(
            training_type='unmasking',
            batch_size=self.config.batch_size,
            block_size=self.config.block_size,
            max_iters=self.config.max_iters,
            device=str(self.device),
            device_type=self.device.type,
            seed_offset=0,
            data_dir='data/shakespeare_char',  # Default data dir
            meta_vocab_size=65,  # Shakespeare char vocab size
            mask_token_id=self.config.mask_token_id,
            wrong_token_id=66,  # meta_vocab_size + 1
            remask_good_id=67,   # meta_vocab_size + 2
            remask_wrong_id=68,  # meta_vocab_size + 3
            extended_vocab_size=69,  # meta_vocab_size + 4
            iter_num=iter_num,
            guaranteed_unmasked_max=self.config.guaranteed_unmasked_max,
            guaranteed_unmasked_min=self.config.guaranteed_unmasked_min,
            random_mask_warmup=self.config.random_mask_warmup,
            noise_max_ratio=0.2,  # Default noise ratio
            sticky_rounds=self.config.sticky_rounds,
            sticky_p1_p2_multiplier=self.config.sticky_p1_p2_multiplier,
            sticky_p1_divisor=self.config.sticky_p1_divisor,
            sticky_transition_start=self.config.sticky_transition_start,
            sticky_transition_end=self.config.sticky_transition_end,
            remasking_corruption_strategy='mixed',  # Default strategy
            remasking_strategy_weights=[0.25, 0.4, 0.25, 0.1],  # Default weights
            eval_iters=20,  # Default eval iters
            warmup_iters=2000,  # Default warmup
            lr_decay_iters=8000,  # Default decay
            learning_rate=1e-4,  # Default lr
            min_lr=1e-5  # Default min lr
        )
    
    def analyze_single_iteration(self, iter_num: int) -> Dict:
        """Analyze masking behavior for a single iteration"""
        print(f"\n=== Analyzing iteration {iter_num} ===")
        
        # Create context and data
        ctx = self.create_training_context(iter_num)
        x = self.create_dummy_data()
        
        # Calculate dynamic guaranteed_unmasked using centralized logic
        current_guaranteed_unmasked = ctx.get_guaranteed_unmasked(iter_num)
        
        # Apply masking (training version)
        masked_x, mask = apply_gpu_masking_training(
            x, iter_num, ctx.mask_token_id, ctx.sticky_rounds, 
            ctx.sticky_p1_p2_multiplier, current_guaranteed_unmasked,
            ctx.sticky_transition_start, ctx.sticky_transition_end,
            ctx.sticky_p1_divisor
        )
        
        # Calculate statistics
        mask_ratio = mask.float().mean().item()
        cluster_stats = analyze_clustering(mask)
        transition_stats = analyze_masking_patterns_with_transition(
            mask, iter_num, ctx.sticky_transition_start, ctx.sticky_transition_end
        )
        
        # Calculate cumulative cluster size distribution
        cluster_size_distribution = self._analyze_cluster_size_distribution(mask)
        
        # Calculate sticky ratio
        if iter_num < ctx.sticky_transition_start:
            sticky_ratio = 0.0
        elif iter_num >= ctx.sticky_transition_end:
            sticky_ratio = 1.0
        else:
            progress = (iter_num - ctx.sticky_transition_start) / \
                      (ctx.sticky_transition_end - ctx.sticky_transition_start)
            sticky_ratio = progress
        
        # Detailed probability analysis
        p1_expected = 1.0 / (ctx.sticky_rounds * ctx.sticky_p1_divisor)  # Average p1
        p2_expected = min(1.0, p1_expected * ctx.sticky_p1_p2_multiplier)
        
        results = {
            'iter_num': iter_num,
            'mask_ratio': mask_ratio,
            'sticky_ratio': sticky_ratio,
            'guaranteed_unmasked': current_guaranteed_unmasked,
            'avg_cluster_size': cluster_stats['avg_cluster_size'],
            'max_cluster_size': cluster_stats['max_cluster_size'],
            'num_clusters_per_batch': cluster_stats['num_clusters_per_batch'],
            'transition_state': transition_stats['transition_state'],
            'p1_expected': p1_expected,
            'p2_expected': p2_expected,
            'sticky_rounds': ctx.sticky_rounds,
            'sticky_p1_divisor': ctx.sticky_p1_divisor,
            'sticky_p1_p2_multiplier': ctx.sticky_p1_p2_multiplier,
            'cluster_size_distribution': cluster_size_distribution
        }
        
        # Print detailed analysis
        print(f"Mask ratio: {mask_ratio:.4f}")
        print(f"Sticky ratio: {sticky_ratio:.4f}")
        print(f"Expected p1: {p1_expected:.6f}, p2: {p2_expected:.6f}")
        print(f"Avg cluster size: {cluster_stats['avg_cluster_size']:.3f}")
        print(f"Max cluster size: {cluster_stats['max_cluster_size']}")
        print(f"Clusters per batch: {cluster_stats['num_clusters_per_batch']:.1f}")
        print(f"Transition state: {transition_stats['transition_state']}")
        
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
        """Run simulation across all test iterations"""
        print("Running full masking simulation...")
        print(f"Testing iterations: {self.config.test_iterations}")
        
        all_results = []
        
        for iter_num in self.config.test_iterations:
            results = self.analyze_single_iteration(iter_num)
            all_results.append(results)
        
        self.results = {
            'config': self.config,
            'iterations': all_results
        }
        
        return self.results
    
    def print_mask_ratio_summary(self):
        """Print clear summary of actual mask ratios"""
        if not self.results:
            print("No results to analyze. Run simulation first.")
            return
        
        iterations = self.results['iterations']
        
        print("\n" + "="*80)
        print("ACTUAL MASK RATIO ANALYSIS")
        print("="*80)
        print(f"{'Iteration':>10} | {'Actual':>10} | {'Expected':>10} | {'Guaranteed':>12} | {'Sticky':>8}")
        print(f"{'':>10} | {'Mask Ratio':>10} | {'Max Mask':>10} | {'Unmasked':>12} | {'Ratio':>8}")
        print("-" * 80)
        
        for r in iterations:
            expected_max = 1.0 - r['guaranteed_unmasked']  # Maximum possible mask ratio for independent masking
            print(f"{r['iter_num']:10d} | {r['mask_ratio']:10.4f} | {expected_max:10.4f} | {r['guaranteed_unmasked']:12.4f} | {r['sticky_ratio']:8.3f}")
        
        print("\nKEY:")
        print("- Actual Mask Ratio: Real fraction of tokens that are masked (mask.float().mean())")
        print("- Expected Max Mask: Maximum possible for independent masking = 1 - guaranteed_unmasked")
        print("- Guaranteed Unmasked: Fraction guaranteed to stay unmasked") 
        print("- Sticky Ratio: Fraction of batch using sticky (vs independent) masking")
    
    def diagnose_issues(self) -> Dict:
        """Diagnose potential issues with clustering"""
        print("\n" + "="*60)
        print("DIAGNOSTIC ANALYSIS")
        print("="*60)
        
        if not self.results:
            print("No results to analyze. Run simulation first.")
            return {}
        
        iterations = self.results['iterations']
        
        # Check progression trends
        mask_ratios = [r['mask_ratio'] for r in iterations]
        cluster_sizes = [r['avg_cluster_size'] for r in iterations]
        sticky_ratios = [r['sticky_ratio'] for r in iterations]
        iter_nums = [r['iter_num'] for r in iterations]
        
        print(f"\n1. MASK RATIO PROGRESSION:")
        print(f"   Start: {mask_ratios[0]:.4f} -> End: {mask_ratios[-1]:.4f}")
        print(f"   Expected: decreasing (more guaranteed_unmasked)")
        
        print(f"\n2. CLUSTER SIZE PROGRESSION:")
        print(f"   Start: {cluster_sizes[0]:.3f} -> End: {cluster_sizes[-1]:.3f}")
        print(f"   Expected: increasing (more sticky masking)")
        
        print(f"\n3. STICKY RATIO PROGRESSION:")
        print(f"   Start: {sticky_ratios[0]:.3f} -> End: {sticky_ratios[-1]:.3f}")
        print(f"   Expected: 0.0 -> 1.0")
        
        # Identify issues
        issues = []
        
        # Issue 1: Cluster size not growing
        if cluster_sizes[-1] <= cluster_sizes[0] + 0.5:
            issues.append("ISSUE: Cluster size not significantly increasing")
            print(f"\n❌ ISSUE: Cluster size barely growing ({cluster_sizes[0]:.3f} -> {cluster_sizes[-1]:.3f})")
        
        # Issue 2: Low absolute cluster sizes
        max_cluster_ever = max(r['max_cluster_size'] for r in iterations)
        if max_cluster_ever < 10:
            issues.append(f"ISSUE: Maximum cluster size very small ({max_cluster_ever})")
            print(f"\n❌ ISSUE: Max cluster size across all iterations: {max_cluster_ever}")
        
        # Issue 3: Probability analysis
        late_iter = iterations[-1]
        if late_iter['p2_expected'] < 0.5:
            issues.append(f"ISSUE: P2 probability too low ({late_iter['p2_expected']:.4f})")
            print(f"\n❌ ISSUE: P2 (neighbor extension) probability too low: {late_iter['p2_expected']:.4f}")
        
        # Issue 4: Too many rounds
        if late_iter['sticky_rounds'] > 15:
            issues.append("ISSUE: Too many sticky rounds may cause interference")
            print(f"\n❌ ISSUE: {late_iter['sticky_rounds']} rounds may cause masking interference")
        
        print(f"\n4. PARAMETER ANALYSIS:")
        last_result = iterations[-1]
        print(f"   P1 (initiation): {last_result['p1_expected']:.6f}")
        print(f"   P2 (extension): {last_result['p2_expected']:.6f}")
        print(f"   Rounds: {last_result['sticky_rounds']}")
        print(f"   P1 divisor: {last_result['sticky_p1_divisor']}")
        print(f"   P1-P2 multiplier: {last_result['sticky_p1_p2_multiplier']}")
        
        if not issues:
            print(f"\n✅ No major issues detected!")
        else:
            print(f"\n📋 SUMMARY: {len(issues)} issues found")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
        
        return {
            'issues': issues,
            'trends': {
                'mask_ratio_change': mask_ratios[-1] - mask_ratios[0],
                'cluster_size_change': cluster_sizes[-1] - cluster_sizes[0],
                'sticky_ratio_change': sticky_ratios[-1] - sticky_ratios[0],
                'max_cluster_ever': max_cluster_ever
            }
        }
    
    def plot_results(self, save_path: str = None):
        """Plot simulation results with parameter information"""
        if not self.results:
            print("No results to plot. Run simulation first.")
            return
        
        iterations = self.results['iterations']
        iter_nums = [r['iter_num'] for r in iterations]
        config = self.config
        
        # Create figure with more space for parameter box
        fig = plt.figure(figsize=(18, 12))
        
        # Create main plots area and parameter box area
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.4], hspace=0.3, wspace=0.3)
        
        # Plot 1: Mask ratio and sticky ratio
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(iter_nums, [r['mask_ratio'] for r in iterations], 'b-', label='Actual Mask Ratio', linewidth=2, marker='o', markersize=4)
        ax1.plot(iter_nums, [r['sticky_ratio'] for r in iterations], 'r--', label='Sticky Ratio', linewidth=2, marker='s', markersize=4)
        # Add guaranteed unmasked for reference
        ax1.plot(iter_nums, [1 - r['guaranteed_unmasked'] for r in iterations], 'g:', label='Expected Max Mask', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Ratio')
        ax1.set_title('Masking Ratios Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cluster Size Distribution (for multiple iterations)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Find iterations at max_iters, 1/2, 1/4, 1/8
        max_iter = max(iter_nums)
        target_iters = [max_iter, max_iter//2, max_iter//4, max_iter//8]
        colors = ['green', 'blue', 'orange', 'red']
        labels = ['max_iters', '1/2 max', '1/4 max', '1/8 max']
        
        has_data = False
        for i, target_iter in enumerate(target_iters):
            # Find closest iteration in results
            closest_iteration = min(iterations, key=lambda x: abs(x['iter_num'] - target_iter))
            cluster_dist = closest_iteration['cluster_size_distribution']
            
            if cluster_dist['cumulative_distribution']:
                cluster_sizes = sorted(cluster_dist['cumulative_distribution'].keys())
                percentages = [cluster_dist['cumulative_distribution'][size] for size in cluster_sizes]
                
                ax2.plot(cluster_sizes, percentages, color=colors[i], linewidth=2, 
                        marker='o', markersize=3, label=f'{labels[i]} (iter {closest_iteration["iter_num"]})',
                        alpha=0.8)
                has_data = True
        
        if has_data:
            ax2.set_xlabel('Cluster Size')
            ax2.set_ylabel('Cumulative % of Masked Tokens')
            ax2.set_title('Cumulative Cluster Distribution (Multiple Iterations)')
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
        
        # Plot 3: Number of clusters and guaranteed unmasked
        ax3 = fig.add_subplot(gs[0, 2])
        ax3_twin = ax3.twinx()
        
        # Clusters on left y-axis
        line1 = ax3.plot(iter_nums, [r['num_clusters_per_batch'] for r in iterations], 'purple', linewidth=2, marker='d', markersize=4, label='Clusters/Batch')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Clusters per Batch', color='purple')
        ax3.tick_params(axis='y', labelcolor='purple')
        ax3.grid(True, alpha=0.3)
        
        # Guaranteed unmasked on right y-axis
        line2 = ax3_twin.plot(iter_nums, [r['guaranteed_unmasked'] for r in iterations], 'brown', linewidth=2, marker='x', markersize=6, label='Guaranteed Unmasked')
        ax3_twin.set_ylabel('Guaranteed Unmasked Ratio', color='brown')
        ax3_twin.tick_params(axis='y', labelcolor='brown')
        
        ax3.set_title('Clusters and Guaranteed Unmasked')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
        
        # Plot 4: Probabilities
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(iter_nums, [r['p1_expected'] for r in iterations], 'cyan', label='P1 (initiation)', linewidth=2, marker='o', markersize=4)
        ax4.plot(iter_nums, [r['p2_expected'] for r in iterations], 'magenta', label='P2 (extension)', linewidth=2, marker='s', markersize=4)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Probability')
        ax4.set_title('Masking Probabilities')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Detailed mask ratio analysis
        ax5 = fig.add_subplot(gs[1, 1])
        # Show actual vs expected mask ratios
        expected_mask_ratios = [0.5 * (1 - r['guaranteed_unmasked']) for r in iterations]  # Expected for independent masking
        ax5.plot(iter_nums, [r['mask_ratio'] for r in iterations], 'b-', label='Actual Mask Ratio', linewidth=2, marker='o', markersize=4)
        ax5.plot(iter_nums, expected_mask_ratios, 'b--', label='Expected (Independent)', linewidth=2, alpha=0.7)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Mask Ratio')
        ax5.set_title('Actual vs Expected Mask Ratios')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Mask ratio trend analysis
        ax6 = fig.add_subplot(gs[1, 2])
        mask_ratios = [r['mask_ratio'] for r in iterations]
        ax6.scatter(iter_nums, mask_ratios, c=iter_nums, cmap='viridis', s=50, alpha=0.8)
        ax6.plot(iter_nums, mask_ratios, 'k-', alpha=0.5, linewidth=1)
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Mask Ratio')
        ax6.set_title('Mask Ratio Progression (Color = Time)')
        cbar = plt.colorbar(ax6.collections[0], ax=ax6)
        cbar.set_label('Iteration')
        ax6.grid(True, alpha=0.3)
        
        # Parameter information box
        param_ax = fig.add_subplot(gs[2, :])
        param_ax.axis('off')
        
        # Create parameter info text
        param_text = f"""SIMULATION PARAMETERS:
Batch Size: {config.batch_size} | Block Size: {config.block_size} | Device: {config.device}

MASKING INTENSITY:
guaranteed_unmasked_max: {config.guaranteed_unmasked_max} | guaranteed_unmasked_min: {config.guaranteed_unmasked_min}
Random Mask Warmup: {config.random_mask_warmup} iterations (then stays at min)
Sticky Transition: {config.sticky_transition_start} → {config.sticky_transition_end} iterations

STICKY CLUSTERING:
sticky_rounds: {config.sticky_rounds} | sticky_p1_divisor: {config.sticky_p1_divisor} | sticky_p1_p2_multiplier: {config.sticky_p1_p2_multiplier}
Expected P1: ~{1.0/(config.sticky_rounds * config.sticky_p1_divisor):.6f} | Expected P2: ~{min(1.0, (1.0/(config.sticky_rounds * config.sticky_p1_divisor)) * config.sticky_p1_p2_multiplier):.6f}

RESULTS SUMMARY:
Initial mask ratio: {iterations[0]['mask_ratio']:.4f} → Final mask ratio: {iterations[-1]['mask_ratio']:.4f} (change: {iterations[-1]['mask_ratio'] - iterations[0]['mask_ratio']:+.4f})
Initial avg cluster: {iterations[0]['avg_cluster_size']:.3f} → Final avg cluster: {iterations[-1]['avg_cluster_size']:.3f} (change: {iterations[-1]['avg_cluster_size'] - iterations[0]['avg_cluster_size']:+.3f})"""
        
        param_ax.text(0.02, 0.98, param_text, transform=param_ax.transAxes, 
                     verticalalignment='top', horizontalalignment='left',
                     fontsize=9, fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        # Add main title
        fig.suptitle(f'Masking Simulation Results\nSticky P1 Divisor: {config.sticky_p1_divisor} | P1-P2 Multiplier: {config.sticky_p1_p2_multiplier} | Rounds: {config.sticky_rounds}', 
                     fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced plot saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, filepath: str):
        """Save simulation results"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Results saved to {filepath}")
    
    def suggest_improvements(self):
        """Suggest parameter improvements based on results"""
        if not self.results:
            print("No results to analyze. Run simulation first.")
            return
        
        print("\n" + "="*60)
        print("IMPROVEMENT SUGGESTIONS")
        print("="*60)
        
        last_result = self.results['iterations'][-1]
        current_cluster_size = last_result['avg_cluster_size']
        
        print(f"\nCurrent average cluster size: {current_cluster_size:.3f}")
        
        if current_cluster_size < 3.0:
            print("\n💡 SUGGESTIONS FOR LARGER CLUSTERS:")
            print("   1. Reduce sticky_p1_divisor from 10.0 to 2.0-5.0")
            print("      → Increases p1 (cluster initiation probability)")
            print("   2. Reduce sticky_rounds from 10 to 5-7")
            print("      → Less interference between rounds")
            print("   3. Consider increasing sticky_p1_p2_multiplier to 15-20")
            print("      → More aggressive cluster growth")
            
            # Calculate suggested parameters
            suggested_divisor = 3.0
            suggested_rounds = 6
            suggested_multiplier = 15.0
            
            new_p1 = 1.0 / (suggested_rounds * suggested_divisor)
            new_p2 = min(1.0, new_p1 * suggested_multiplier)
            
            print(f"\n📋 SUGGESTED PARAMETERS:")
            print(f"   sticky_p1_divisor = {suggested_divisor}")
            print(f"   sticky_rounds = {suggested_rounds}")
            print(f"   sticky_p1_p2_multiplier = {suggested_multiplier}")
            print(f"   → This gives p1≈{new_p1:.4f}, p2≈{new_p2:.4f}")
        else:
            print(f"\n✅ Cluster size looks reasonable ({current_cluster_size:.3f})")

def main():
    """Main execution function"""
    print("Masking Behavior Simulator")
    print("=" * 50)
    
    # Create configuration
    config = SimulationConfig(
        batch_size=16,  # Match current train_run.py batch_size
        block_size=1024,
        max_iters=13000,  # Match current train_run.py max_iters
        # Use your current settings from train_run.py
        sticky_rounds=30,
        sticky_p1_p2_multiplier=20.0,
        sticky_p1_divisor=3.0,
        sticky_transition_start=5000,
        sticky_transition_end=20000,
        guaranteed_unmasked_max=0.8,
        guaranteed_unmasked_min=0.2,
        random_mask_warmup=8000  # Match train_run.py
    )
    
    # Create and run simulator
    simulator = MaskingSimulator(config)
    
    # Run full simulation
    results = simulator.run_full_simulation()
    
    # Print mask ratio summary
    simulator.print_mask_ratio_summary()
    
    # Diagnose issues
    diagnosis = simulator.diagnose_issues()
    
    # Suggest improvements
    simulator.suggest_improvements()
    
    # Plot results
    try:
        simulator.plot_results('masking_simulation.png')
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    # Save results
    simulator.save_results('masking_simulation_results.pkl')
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()