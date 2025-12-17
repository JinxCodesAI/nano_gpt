import json
import sys
from collections import defaultdict

def main():
    # Path to tokenizer.json
    path = "data/cosmopedia/tokenizer.json"
    if len(sys.argv) > 1:
        path = sys.argv[1]

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return

    model = data.get('model', {})
    vocab = model.get('vocab', {})
    merges = model.get('merges', [])

    if not vocab or not merges:
        print("Error: Invalid tokenizer format (missing vocab or merges)")
        return

    # Track levels of all tokens/intermediate merges
    # Default level is 1 (base tokens)
    token_levels = {}

    # Process merges sequentially
    for merge in merges:
        # Handle list format ["a", "b"] or string format "a b"
        if isinstance(merge, list):
            parts = merge
            combined = "".join(merge)
        elif isinstance(merge, str):
            parts = merge.split(" ")
            combined = "".join(parts)
        else:
            continue

        # Calculate level: max(parent_levels) + 1
        max_level = 0
        for p in parts:
            max_level = max(max_level, token_levels.get(p, 1))
        
        token_levels[combined] = max_level + 1

    # Group final vocabulary by level
    level_counts = defaultdict(int)
    level_samples = defaultdict(list)
    
    for token in vocab.keys():
        lvl = token_levels.get(token, 1)
        level_counts[lvl] += 1
        level_samples[lvl].append(token)

    # Output results
    sorted_levels = sorted(level_counts.keys())
    
    for lvl in sorted_levels:
        count = level_counts[lvl]
        print(f"\nLevel {lvl} ({count})")
        
        # Sort samples alphabetically
        samples = sorted(level_samples[lvl])
        
        # Display all samples
        output_line = ",".join(samples)
        print(output_line)

if __name__ == "__main__":
    main()
