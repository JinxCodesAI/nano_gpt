import datasets
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from collections import Counter
from tqdm import tqdm

# --- CONFIG ---
# Reduced list for testing speed, or use your full list
CONFIGS = ['web_samples_v1', 'web_samples_v2'] 
VOCAB_SIZE = 4096
BATCH_SIZE = 1000
LIMIT_PER_CONFIG = 40000 

def get_data_iterator():
    """
    Re-usable generator. We need to call this twice:
    Once for training, once for counting.
    """
    for config_name in CONFIGS:
        print(f"--- Loading stream: {config_name} ---")
        try:
            dataset = datasets.load_dataset(
                "HuggingFaceTB/cosmopedia", 
                config_name, 
                split="train", 
                streaming=True
            )
            
            batch = []
            count = 0
            
            for example in dataset:
                text = example.get('text', '')
                if text:
                    batch.append(text)
                
                if len(batch) == BATCH_SIZE:
                    yield batch
                    batch = []
                    count += BATCH_SIZE
                    
                if count >= LIMIT_PER_CONFIG:
                    break
                    
            if batch:
                yield batch
                
        except Exception as e:
            print(f"Skipping {config_name} due to error: {e}")

# ==========================================
# PHASE 1: TRAIN TOKENIZER
# ==========================================
print("\n=== PHASE 1: Training Tokenizer ===")
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
special_tokens = ["[PAD]", "[UNK]", "[SEP]", "[CLS]", "[MASK]"]

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=special_tokens,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True
)

# We consume the generator once here
tokenizer.train_from_iterator(get_data_iterator(), trainer=trainer)
tokenizer.save("cosmopedia_16k_counted.json")
print("Tokenizer trained and saved.")

# ==========================================
# PHASE 2: COUNT TOKENS (The part you wanted)
# ==========================================
print("\n=== PHASE 2: Counting Exact Token Occurrences ===")

token_counts = Counter()
total_tokens_processed = 0

# We reload the generator to iterate over the data again
data_stream = get_data_iterator()

# We use tqdm to show progress
for batch in tqdm(data_stream, desc="Counting tokens"):
    # batch is a list of strings
    # We encode them in batch (much faster)
    encodings = tokenizer.encode_batch(batch)
    
    for encoding in encodings:
        ids = encoding.ids
        token_counts.update(ids)
        total_tokens_processed += len(ids)

# ==========================================
# PHASE 3: REPORT
# ==========================================
print("\n" + "="*40)
print("       EXACT FREQUENCY REPORT       ")
print("="*40)
print(f"Total Tokens Processed: {total_tokens_processed:,}")

# Get the vocab dictionary to map IDs -> Strings
vocab = tokenizer.get_vocab()
# Sort vocab by ID to find the last added tokens
sorted_vocab_by_id = sorted(vocab.items(), key=lambda item: item[1])

print(f"\n--- Rarest Tokens (Last 15 IDs added to Vocab) ---")
print(f"{'ID':<6} | {'Token':<20} | {'Count':<10}")
print("-" * 45)

# Look at the last 15 tokens
for token_str, token_id in sorted_vocab_by_id[-15:]:
    count = token_counts[token_id]
    print(f"{token_id:<6} | {token_str:<20} | {count:<10}")

print("-" * 45)

print(f"\n--- Most Common Tokens (Top 10 excluding special) ---")
# Get top 10 most common from our Counter, filtering out special tokens (ids < 5)
common = [(tokenizer.id_to_token(tid), count) for tid, count in token_counts.most_common(15) if tid > 4][:10]
for token_str, count in common:
    print(f"{token_str:<20}: {count:,}")