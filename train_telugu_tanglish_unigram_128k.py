#!/usr/bin/env python3
"""
Phase 2 + Phase 3 Production Engine:
128k Vocab Unigram Tokenizer for Native Telugu, Dialects, Tanglish, and English.
"""

import os
import time
from tokenizers import (
    Tokenizer, 
    models, 
    trainers, 
    normalizers, 
    pre_tokenizers, 
    decoders
)
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast

# Multi-Corpus Input: Phase 1 Authentic Telugu + Phase 1B Tanglish
CORPUS_FILES = [
    "final_authentic_telugu_corpus.txt",
    "tanglish_dialect_corpus.txt"
]

OUTPUT_DIR = "telugu_tanglish_unigram_128k"
VOCAB_SIZE = 128_000

SPECIAL_TOKENS = [
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
    "<mask conservatism>"
]

def run_128k_pipeline():
    # Verify input corpora exist
    for file_path in CORPUS_FILES:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Corpus '{file_path}' not found! Run generation scripts first.")

    print("=" * 70)
    print(f"🚀 STARTING MULTI-CORPUS 128K UNIGRAM TRAINING ({VOCAB_SIZE:,} VOCAB)")
    print("=" * 70)

    # 1. Initialize Unigram Model Engine
    tokenizer = Tokenizer(models.Unigram())

    # 2. NFC Normalization (Standardizes Telugu matras/viramas)
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC()
    ])

    # 3. Pre-Tokenizer: Word-Level Metaspace + Punctuation
    # Allows full Telugu agglutinated words and Tanglish terms to be trained as single pieces
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(replacement=" "),
        pre_tokenizers.Punctuation()
    ])

    # 4. Decoder Setup
    tokenizer.decoder = decoders.Metaspace(replacement=" ")

    # 5. Trainer Parameters with ByteLevel Initial Alphabet for ZERO OOV
    trainer = trainers.UnigramTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        unk_token="<unk>",
        initial_alphabet=ByteLevel.alphabet(),  # Guarantees complete ASCII + UTF-8 fallback
        max_piece_length=32,                   # Supports long agglutinative Telugu words
        show_progress=True
    )

    # 6. Run Training across both datasets
    start_time = time.time()
    print("⏳ Running entropy-pruning over combined Native Telugu & Tanglish corpora...")
    tokenizer.train(CORPUS_FILES, trainer)
    elapsed = time.time() - start_time
    print(f"✓ Training finished in {elapsed:.2f} seconds.")

    # 7. Save Raw Tokenizer Artifacts
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "tokenizer.json")
    tokenizer.save(json_path)

    # 8. Wrap into Hugging Face Fast Tokenizer format
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=json_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask conservatism>",
        clean_up_tokenization_spaces=True
    )
    fast_tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✓ Model successfully compiled & saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    run_128k_pipeline()