#!/usr/bin/env python3
"""
Validation Script: Verifies Akshara preservation and fertility rate.
"""

from transformers import AutoTokenizer

TOKENIZER_DIR = "telugu_akshara_unigram_64k"

def test_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    test_sentences = [
        "నమస్కారం! తెలుగు భాషా విభాగానికి స్వాగతం.",
        "స్వాతంత్య్ర దినోత్సవ శుభాకాంక్షలు.",
        "సంస్కృతి, సాహిత్యం, శాస్త్రం మరియు సాంకేతికత.",
        "కృత్రిమ మేధస్సు (Artificial Intelligence) మరియు యంత్ర అభ్యసనం.",
        "హైదరాబాద్ మరియు విశాఖపట్నం తెలంగాణ మరియు ఆంధ్రప్రదేశ్ రాష్ట్రాలలో ముఖ్యమైన నగరాలు."
    ]

    print("=" * 70)
    print("🔍 TOKENIZER QUALITY AUDIT")
    print("=" * 70)

    total_words = 0
    total_tokens = 0

    for i, text in enumerate(test_sentences, 1):
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)

        words = text.split()
        num_words = len(words)
        num_tokens = len(tokens)

        total_words += num_words
        total_tokens += num_tokens

        fertility = num_tokens / num_words if num_words > 0 else 0
        is_exact = (decoded.strip() == text.strip())

        print(f"\n[Sample {i}] {text}")
        print(f"  Tokens ({num_tokens})  : {tokens}")
        print(f"  Fertility Rate : {fertility:.2f} tokens/word")
        print(f"  Exact Decode   : {'✅ PASS' if is_exact else '❌ FAIL'}")

    avg_fertility = total_tokens / total_words if total_words > 0 else 0
    print("\n" + "=" * 70)
    print(f"📊 Aggregate Fertility Rate: {avg_fertility:.2f} tokens/word (Target: 1.05 - 1.25)")
    print("=" * 70)

if __name__ == "__main__":
    test_tokenizer()