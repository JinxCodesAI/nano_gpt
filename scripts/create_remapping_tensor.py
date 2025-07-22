#!/usr/bin/env python3
"""
Script to create vocabulary remapping tensors for shrunken vocabulary training.

This script converts a JSON token mapping file to a PyTorch tensor for efficient
vocabulary remapping during training. The JSON file maps original token IDs to
shrunken vocabulary IDs, with rare tokens mapped to a special RARE_TOKEN_ID.

Usage:
    python scripts/create_remapping_tensor.py --mapping_json data/fineweb10B/token_mapping_k11199.json --output_path data/vocab_remapping.pt

Arguments:
    --mapping_json: Path to JSON file containing token ID mappings
    --output_path: Path where to save the remapping tensor (.pt file)
"""

import argparse
import torch
import json
import os


def create_remapping_tensor_from_json(mapping_json_path, output_path):
    """
    Create a vocabulary remapping tensor from a JSON mapping file.

    Args:
        mapping_json_path (str): Path to JSON file with token mappings
        output_path (str): Path to save the tensor

    Returns:
        tuple: (remapping_tensor, full_vocab_size, core_vocab_size, rare_token_id)
    """

    # Validation
    if not os.path.exists(mapping_json_path):
        raise FileNotFoundError(f"Mapping JSON file not found: {mapping_json_path}")

    print(f"Loading token mapping from: {mapping_json_path}")

    # Load the JSON mapping
    with open(mapping_json_path, 'r') as f:
        token_mapping = json.load(f)

    # Determine vocabulary sizes and rare token ID
    original_ids = [int(k) for k in token_mapping.keys()]
    mapped_ids = list(token_mapping.values())

    full_vocab_size = max(original_ids) + 1
    core_vocab_size = max(mapped_ids) + 1

    # Find the rare token ID (most frequent mapped ID)
    from collections import Counter
    id_counts = Counter(mapped_ids)
    rare_token_id = id_counts.most_common(1)[0][0]

    print(f"Detected configuration:")
    print(f"  Full vocabulary size: {full_vocab_size}")
    print(f"  Core vocabulary size: {core_vocab_size}")
    print(f"  RARE_TOKEN_ID: {rare_token_id}")
    print(f"  Tokens mapped to RARE_TOKEN_ID: {id_counts[rare_token_id]}")

    # Create the remapping tensor
    remapping_vector = torch.zeros(full_vocab_size, dtype=torch.long)

    # Apply the mappings from JSON
    for orig_id_str, mapped_id in token_mapping.items():
        orig_id = int(orig_id_str)
        remapping_vector[orig_id] = mapped_id

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the tensor
    torch.save(remapping_vector, output_path)

    print(f"Remapping tensor saved to: {output_path}")
    print(f"Tensor shape: {remapping_vector.shape}")

    # Verify the tensor
    unique_mapped = torch.unique(remapping_vector)
    print(f"Unique mapped token IDs: {len(unique_mapped)} (should be {core_vocab_size})")

    print("✓ Remapping tensor creation completed")

    return remapping_vector, full_vocab_size, core_vocab_size, rare_token_id


def main():
    parser = argparse.ArgumentParser(description="Create vocabulary remapping tensor from JSON mapping file")

    parser.add_argument("--mapping_json", type=str, required=True,
                        help="Path to JSON file containing token ID mappings")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path where to save the remapping tensor (.pt file)")

    args = parser.parse_args()

    try:
        remapping_tensor, full_vocab_size, core_vocab_size, rare_token_id = create_remapping_tensor_from_json(
            mapping_json_path=args.mapping_json,
            output_path=args.output_path
        )

        print("✓ Remapping tensor creation completed successfully")
        print(f"Configuration for train.py:")
        print(f"  shrunken_vocab_size = {core_vocab_size}")
        print(f"  vocab_remapping_file = '{args.output_path}'")
        print(f"  RARE_TOKEN_ID = {rare_token_id}")

    except Exception as e:
        print(f"✗ Error creating remapping tensor: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
