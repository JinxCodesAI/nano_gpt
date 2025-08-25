#!/usr/bin/env python3
"""
Test script to verify the guaranteed_unmasked fixes work correctly
"""

import sys
sys.path.append('.')

from masking_simulator_minimal import MinimalMaskingSimulator

def test_fixes():
    print("🔧 TESTING FIXES FOR GUARANTEED_UNMASKED BEHAVIOR")
    print("=" * 70)
    
    print("\n📋 Test 1: sticky_transition_start=14000 (Independent masking)")
    print("Expected: Mostly independent masking, mask ratio should progress 0.1 -> 0.4")
    
    analyzer = MinimalMaskingSimulator(
        batch_size=64,
        block_size=1024,
        sticky_rounds=10,
        sticky_p1_divisor=10.0,
        sticky_p1_p2_multiplier=10.0,
        sticky_transition_start=14000,  # Very late - mostly independent!
        sticky_transition_end=15000,
        guaranteed_unmasked_max=0.8,    # 80% unmasked = 20% masked max early
        guaranteed_unmasked_min=0.2     # 20% unmasked = 80% masked max late
    )
    
    results = analyzer.run_simulation([0, 2000, 5000, 10000, 13000, 14000, 14500, 15000, 17000, 20000])
    
    print(f"\n📊 Results:")
    print(f"   Early (iter 0):      mask_ratio = {results[0]['mask_ratio']:.4f} (expected ~0.1)")
    print(f"   Mid   (iter 10000):  mask_ratio = {results[3]['mask_ratio']:.4f} (expected ~0.3)")
    print(f"   Late  (iter 20000):  mask_ratio = {results[-1]['mask_ratio']:.4f} (expected ~0.4)")
    
    # Check if progression is correct
    early_mask = results[0]['mask_ratio']
    late_mask = results[-1]['mask_ratio']
    
    if late_mask > early_mask + 0.1:
        print(f"   ✅ PASS: Mask ratio increases over time ({early_mask:.3f} -> {late_mask:.3f})")
    else:
        print(f"   ❌ FAIL: Mask ratio should increase! ({early_mask:.3f} -> {late_mask:.3f})")
    
    if early_mask < 0.15:
        print(f"   ✅ PASS: Early mask ratio is low ({early_mask:.3f})")
    else:
        print(f"   ❌ FAIL: Early mask ratio too high ({early_mask:.3f})")
    
    if late_mask > 0.25:
        print(f"   ✅ PASS: Late mask ratio is higher ({late_mask:.3f})")
    else:
        print(f"   ❌ FAIL: Late mask ratio too low ({late_mask:.3f})")
    
    print(f"\n📋 Test 2: Normal transition settings")
    print("Expected: Should see sticky clustering effects")
    
    analyzer2 = MinimalMaskingSimulator(
        batch_size=64,
        block_size=1024,
        sticky_rounds=6,                # Fixed parameters
        sticky_p1_divisor=3.0,          # More aggressive  
        sticky_p1_p2_multiplier=15.0,   # clustering
        sticky_transition_start=500,
        sticky_transition_end=15000,
        guaranteed_unmasked_max=0.95,   # Fixed progression direction
        guaranteed_unmasked_min=0.6     # 5% -> 40% masked
    )
    
    results2 = analyzer2.run_simulation([0, 1000, 5000, 10000, 15000, 17500])
    
    print(f"\n📊 Results with improved parameters:")
    print(f"   Early cluster size: {results2[0]['avg_cluster_size']:.3f}")
    print(f"   Late  cluster size: {results2[-1]['avg_cluster_size']:.3f}")
    print(f"   Early mask ratio:   {results2[0]['mask_ratio']:.4f}")
    print(f"   Late  mask ratio:   {results2[-1]['mask_ratio']:.4f}")
    
    if results2[-1]['avg_cluster_size'] > results2[0]['avg_cluster_size'] + 1.0:
        print(f"   ✅ PASS: Clustering improves significantly")
    else:
        print(f"   ❌ FAIL: Clustering should improve more")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   The fixes should now give you:")
    print(f"   - Proper mask ratio progression (low -> high)")
    print(f"   - Dynamic guaranteed_unmasked calculation")
    print(f"   - Test iterations based on actual parameters")

if __name__ == "__main__":
    test_fixes()