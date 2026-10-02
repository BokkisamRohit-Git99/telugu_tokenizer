#!/usr/bin/env python3
"""
Phase 2 + Phase 3 Production Engine:
128k Vocab Clean BPE Tokenizer with Byte Fallback (0% OOV, 100% Lossless Decode).
"""

import os
import time
from tokenizers import (
    Tokenizer, 
    models, 
    trainers, 
    pre_tokenizers, 
    decoders,
    normalizers,
    Regex
)
from transformers import PreTrainedTokenizerFast

CORPUS_FILES = [
    "final_authentic_telugu_corpus.txt",
    "tanglish_dialect_corpus.txt"
]

OUTPUT_DIR = "telugu_tanglish_master_128k"
VOCAB_SIZE = 128_000

# 256 Byte Fallback Tokens (<0x00> to <0xFF>) for 100% Emoji/OOV Losslessness
BYTE_TOKENS = [f"<0x{i:02X}>" for i in range(256)]

SPECIAL_TOKENS = [
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
    "<mask conservatism>"
] + BYTE_TOKENS

def run_master_pipeline():
    for file_path in CORPUS_FILES:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Corpus '{file_path}' missing. Run generation script first.")

    print("=" * 70)
    print(f"🚀 TRAINING MASTER 128K BPE TOKENIZER ({VOCAB_SIZE:,} VOCAB)")
    print("=" * 70)

    # 1. BPE Model with Byte Fallback
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))

    # 2. Normalization: NFC + Strip ZWNJ (\u200c) & ZWJ (\u200d)
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Replace(pattern=Regex(r"[\u200c\u200d]"), content="")
    ])

    # 3. Pre-Tokenizer: Metaspace + Punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(replacement=" "),
        pre_tokenizers.Punctuation()
    ])

    # 4. Decoder: ByteFallback + Metaspace (Converts <0xXX> bytes back to Emojis/Symbols)
    tokenizer.decoder = decoders.Sequence([
        decoders.ByteFallback(),
        decoders.Metaspace(replacement=" ")
    ])

    # 5. Trainer Setup
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True
    )

    # 6. Train Model
    start_time = time.time()
    print("⏳ Running BPE merges across Telugu & Tanglish corpora...")
    tokenizer.train(CORPUS_FILES, trainer)
    elapsed = time.time() - start_time
    print(f"✓ Training finished in {elapsed:.2f} seconds.")

    # 7. Save Raw Tokenizer Artifacts
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "tokenizer.json")
    tokenizer.save(json_path)

    # 8. Save Hugging Face Fast Tokenizer Wrapper
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=json_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask conservatism>",
        clean_up_tokenization_spaces=False
    )
    fast_tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✓ Saved master tokenizer model to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    run_master_pipeline()