#!/usr/bin/env python3
"""
Phase 2 + Phase 3 Corrected Engine:
Word-Level Pre-Tokenized Unigram Model (64k Vocab)
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
from transformers import PreTrainedTokenizerFast

CORPUS_PATH = "final_authentic_telugu_corpus.txt"
OUTPUT_DIR = "telugu_akshara_unigram_64k"
VOCAB_SIZE = 64_000

SPECIAL_TOKENS = [
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
    "<mask conservatism>"
]

def run_pipeline():
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"Corpus file '{CORPUS_PATH}' missing. Ensure Phase 1 ran successfully.")

    print("=" * 70)
    print("🚀 STARTING CORRECTED UNIGRAM TOKENIZER TRAINING (64K VOCAB)")
    print("=" * 70)

    # 1. Initialize Unigram Engine
    tokenizer = Tokenizer(models.Unigram())

    # 2. NFC Normalization (Standardizes Telugu matra/virama sequences)
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC()
    ])

    # 3. Correct Pre-Tokenizer: Metaspace + Punctuation
    # Allows Unigram to evaluate whole words as pre-token candidates
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(replacement=" "),
        pre_tokenizers.Punctuation()
    ])

    # 4. Decoder Setup
    tokenizer.decoder = decoders.Metaspace(replacement=" ")

    # 5. Unigram Trainer Parameters
    trainer = trainers.UnigramTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        unk_token="<unk>",
        initial_alphabet=[],
        max_piece_length=32,  # Allows whole long Telugu words
        show_progress=True
    )

    # 6. Train Model
    start_time = time.time()
    print("⏳ Running Unigram entropy-pruning over Word blocks...")
    tokenizer.train([CORPUS_PATH], trainer)
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
    print(f"✓ Saved Hugging Face model files to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    run_pipeline()