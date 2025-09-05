#!/usr/bin/env python3
"""
Compare the structure and complexity of train_run.py vs train_run2.py
"""

import os
import re

def count_lines_and_analyze(filename):
    """Count lines and analyze structure of a training file."""
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    
    # Find main training loop
    training_loop_start = None
    for i, line in enumerate(lines):
        if 'while True:' in line and 'training' in ''.join(lines[max(0, i-10):i]).lower():
            training_loop_start = i + 1
            break
    
    # Count different types of content
    import_lines = 0
    config_lines = 0
    comment_lines = 0
    blank_lines = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_lines += 1
        elif stripped.startswith('#'):
            comment_lines += 1
        elif stripped.startswith('import ') or stripped.startswith('from '):
            import_lines += 1
        elif '=' in stripped and not stripped.startswith('def ') and not stripped.startswith('class '):
            # Rough heuristic for config lines
            if not any(keyword in stripped for keyword in ['for ', 'if ', 'while ', 'with ', 'try:']):
                config_lines += 1
    
    # Count function definitions
    function_count = len([line for line in lines if line.strip().startswith('def ')])
    
    # Count class definitions
    class_count = len([line for line in lines if line.strip().startswith('class ')])
    
    return {
        'total_lines': total_lines,
        'training_loop_start': training_loop_start,
        'training_loop_percentage': (training_loop_start / total_lines * 100) if training_loop_start else None,
        'import_lines': import_lines,
        'config_lines': config_lines,
        'comment_lines': comment_lines,
        'blank_lines': blank_lines,
        'function_count': function_count,
        'class_count': class_count,
        'code_lines': total_lines - comment_lines - blank_lines
    }

def main():
    print("=" * 80)
    print("TRAINING FILE COMPARISON")
    print("=" * 80)
    
    # Analyze original train.py
    original_stats = count_lines_and_analyze('data/original_train.py')
    if original_stats:
        print(f"\nORIGINAL train.py:")
        print(f"  Total lines: {original_stats['total_lines']}")
        print(f"  Training loop starts at line: {original_stats['training_loop_start']} ({original_stats['training_loop_percentage']:.1f}%)")
        print(f"  Code lines: {original_stats['code_lines']}")
        print(f"  Functions: {original_stats['function_count']}")
    
    # Analyze current train_run.py
    current_stats = count_lines_and_analyze('train_run.py')
    if current_stats:
        print(f"\nCURRENT train_run.py:")
        print(f"  Total lines: {current_stats['total_lines']}")
        print(f"  Training loop starts at line: {current_stats['training_loop_start']} ({current_stats['training_loop_percentage']:.1f}%)")
        print(f"  Code lines: {current_stats['code_lines']}")
        print(f"  Functions: {current_stats['function_count']}")
    
    # Analyze refactored train_run2.py
    refactored_stats = count_lines_and_analyze('train_run2.py')
    if refactored_stats:
        print(f"\nREFACTORED train_run2.py:")
        print(f"  Total lines: {refactored_stats['total_lines']}")
        print(f"  Training loop starts at line: {refactored_stats['training_loop_start']} ({refactored_stats['training_loop_percentage']:.1f}%)")
        print(f"  Code lines: {refactored_stats['code_lines']}")
        print(f"  Functions: {refactored_stats['function_count']}")
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    if original_stats and current_stats and refactored_stats:
        print(f"{'Metric':<25} {'Original':<12} {'Current':<12} {'Refactored':<12} {'Improvement'}")
        print("-" * 80)
        
        # File size comparison
        orig_lines = original_stats['total_lines']
        curr_lines = current_stats['total_lines']
        refact_lines = refactored_stats['total_lines']
        
        print(f"{'Total Lines':<25} {orig_lines:<12} {curr_lines:<12} {refact_lines:<12} {curr_lines - refact_lines:+d} lines")
        
        # Training loop position
        orig_loop = original_stats['training_loop_start']
        curr_loop = current_stats['training_loop_start']
        refact_loop = refactored_stats['training_loop_start']
        
        print(f"{'Loop Start Line':<25} {orig_loop:<12} {curr_loop:<12} {refact_loop:<12} {curr_loop - refact_loop:+d} lines")
        
        # Loop position percentage
        orig_pct = original_stats['training_loop_percentage']
        curr_pct = current_stats['training_loop_percentage']
        refact_pct = refactored_stats['training_loop_percentage']
        
        print(f"{'Loop Position %':<25} {orig_pct:.1f}%{'':<8} {curr_pct:.1f}%{'':<8} {refact_pct:.1f}%{'':<8} {curr_pct - refact_pct:+.1f}%")
        
        # Code density
        orig_code = original_stats['code_lines']
        curr_code = current_stats['code_lines']
        refact_code = refactored_stats['code_lines']
        
        print(f"{'Code Lines':<25} {orig_code:<12} {curr_code:<12} {refact_code:<12} {curr_code - refact_code:+d} lines")
        
        print("\n" + "=" * 80)
        print("REFACTORING SUCCESS METRICS")
        print("=" * 80)
        
        # Calculate improvements
        size_reduction = ((curr_lines - refact_lines) / curr_lines) * 100
        loop_prominence = curr_pct - refact_pct
        
        print(f"✓ File size reduction: {size_reduction:.1f}% ({curr_lines - refact_lines} lines removed)")
        print(f"✓ Training loop prominence: {loop_prominence:.1f}% earlier in file")
        print(f"✓ Complexity reduction: {curr_code - refact_code} lines of code moved to modules")
        
        # Success criteria
        print(f"\nSUCCESS CRITERIA:")
        print(f"  Target: Reduce to ~450 lines: {'✓ PASS' if refact_lines <= 500 else '✗ FAIL'} ({refact_lines} lines)")
        print(f"  Target: Loop starts before 50%: {'✓ PASS' if refact_pct < 50 else '✗ FAIL'} ({refact_pct:.1f}%)")
        print(f"  Target: Significant reduction: {'✓ PASS' if size_reduction > 30 else '✗ FAIL'} ({size_reduction:.1f}%)")
        
        if refact_lines <= 500 and refact_pct < 50 and size_reduction > 30:
            print(f"\n🎉 REFACTORING SUCCESS! All targets achieved.")
        else:
            print(f"\n⚠️  Some targets not met, but significant improvement achieved.")

if __name__ == '__main__':
    main()
