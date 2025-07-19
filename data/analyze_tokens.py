
import os
import numpy as np
import sys
import json
from multiprocessing import Pool, cpu_count

# --- Configuration ---
# Set the directory where your FineWeb10B .bin files are located.
# This script assumes they are in a subdirectory named 'fineweb10B'.
DATA_DIR = os.path.join(os.path.dirname(__file__), 'fineweb10B')

# The GPT-2 tokenizer has a vocabulary size of 50257.
VOCAB_SIZE = 57664

# Number of parallel processes to use for file reading.
# Default is the number of CPU cores available.
NUM_PROCESSES = cpu_count()

# K values for the token mappings
K1 = 11199
K2 = 37311


def process_file(file_path):
    """
    Reads a single .bin file and returns the token counts.
    """
    try:
        # The GPT-2 tokenizer uses uint16 for tokens
        tokens = np.fromfile(file_path, dtype=np.uint16)
        unique_tokens, counts = np.unique(tokens, return_counts=True)
        
        # Create a local count array for this file
        file_token_counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
        file_token_counts[unique_tokens] += counts
        
        print(f"Successfully processed: {os.path.basename(file_path)}")
        return file_token_counts
    except Exception as e:
        print(f"Could not process {os.path.basename(file_path)}: {e}")
        return np.zeros(VOCAB_SIZE, dtype=np.int64)


def create_and_save_mapping(sorted_indices, k, output_path):
    """
    Creates a token mapping based on frequency rank and saves it to a JSON file.
    Top K tokens are mapped to their rank (0 to K-1).
    All other tokens are mapped to K-1.
    """
    # Create a dictionary mapping the top K token IDs to their rank.
    # e.g., {most_frequent_token_id: 0, second_most_frequent: 1, ...}
    token_to_rank = {token_id: rank for rank, token_id in enumerate(sorted_indices[:k])}
    
    # The value for all tokens not in the top K will be k-1
    other_value = k - 1
    
    # Create the final mapping for all tokens in the vocabulary
    mapping = {
        i: token_to_rank.get(i, other_value)
        for i in range(VOCAB_SIZE)
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=4)
        print(f"Successfully saved mapping to: {output_path}")
    except Exception as e:
        print(f"Error saving mapping to {output_path}: {e}")


def analyze_token_distribution():
    """
    Counts token occurrences in parallel, calculates distribution statistics,
    and creates token mappings.
    """
    if not os.path.isdir(DATA_DIR):
        print(f"Error: The directory '{DATA_DIR}' was not found.")
        print("Please ensure you have downloaded the data and the path is correct.")
        sys.exit(1)

    # Get a list of all .bin files
    all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.bin')]

    if not all_files:
        print(f"Error: No '.bin' files found in '{DATA_DIR}'.")
        sys.exit(1)

    print(f"Starting token analysis with {NUM_PROCESSES} processes...")

    # Process all files in parallel
    with Pool(NUM_PROCESSES) as pool:
        results = pool.map(process_file, all_files)

    # Aggregate the results from all processes
    token_counts = np.sum(results, axis=0)
    total_tokens = np.sum(token_counts)

    print("\nToken analysis complete.")
    print("---" * 10)

    # --- Statistical Analysis ---
    print("\nCalculating distribution statistics...")

    if total_tokens == 0:
        print("No tokens were processed. Exiting.")
        sys.exit()

    # Sort tokens by frequency in descending order
    sorted_indices = np.argsort(token_counts)[::-1]
    sorted_counts = token_counts[sorted_indices]

    # --- Mapping Creation ---
    print("\nCreating and saving token mappings...")

    # Create and save the first mapping (K1)
    mapping_path_k1 = os.path.join(DATA_DIR, f'token_mapping_k{K1}.json')
    create_and_save_mapping(sorted_indices, K1, mapping_path_k1)

    # Create and save the second mapping (K2)
    mapping_path_k2 = os.path.join(DATA_DIR, f'token_mapping_k{K2}.json')
    create_and_save_mapping(sorted_indices, K2, mapping_path_k2)

    # --- Display Results ---
    # Calculate cumulative sum for display purposes
    cumulative_counts = np.cumsum(sorted_counts)
    tokens_for_90_percent = np.searchsorted(cumulative_counts, 0.90 * total_tokens) + 1
    tokens_for_99_percent = np.searchsorted(cumulative_counts, 0.99 * total_tokens) + 1

    print("\n## Dataset Statistics")
    print("---" * 10)
    print(f"**Total unique tokens found:** {np.count_nonzero(token_counts):,}")
    print(f"**Vocabulary Size:** {VOCAB_SIZE:,}")
    print(f"**Total tokens processed:** {total_tokens:,}")
    print("\n## Token Distribution")
    print("---" * 10)
    print(f"A small subset of the vocabulary accounts for a large portion of the data:")
    print(f"**{tokens_for_90_percent:,}** unique tokens make up **90%** of the dataset.")
    print(f"**{tokens_for_99_percent:,}** unique tokens make up **99%** of the dataset.")
    print("\n## Top 10 Most Frequent Tokens")
    print("---" * 10)
    print(f"{'Rank':<8} {'Token ID':<10} {'Occurrences':<15} {'Percentage':<10}")
    for i in range(10):
        token_id = sorted_indices[i]
        count = sorted_counts[i]
        percentage = (count / total_tokens) * 100
        print(f"{i+1:<8} {token_id:<10} {count:<15,} {percentage:.4f}%")
        
    print(f"{10000:<8} {sorted_indices[9999]:<10} {sorted_counts[9999]:<15,} {(sorted_counts[9999]/total_tokens * 100):.4f}%")
    print(f"{20000:<8} {sorted_indices[19999]:<10} {sorted_counts[19999]:<15,} {(sorted_counts[19999]/total_tokens * 100):.4f}%")
    print(f"{30000:<8} {sorted_indices[29999]:<10} {sorted_counts[29999]:<15,} {(sorted_counts[29999]/total_tokens * 100):.4f}%")
    print(f"{35000:<8} {sorted_indices[34999]:<10} {sorted_counts[34999]:<15,} {(sorted_counts[34999]/total_tokens * 100):.4f}%")
    print("---" * 10)


if __name__ == "__main__":
    analyze_token_distribution()