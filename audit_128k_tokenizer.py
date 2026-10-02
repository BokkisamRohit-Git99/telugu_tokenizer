#!/usr/bin/env python3
"""
Master Audit Suite for 128k Telugu/Tanglish BPE Tokenizer
"""

from transformers import AutoTokenizer

TOKENIZER_DIR = "telugu_tanglish_master_128k"

def audit():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, clean_up_tokenization_spaces=False)

    test_samples = [
        ("Native Telugu", "నమస్కారం! తెలుగు భాషా విభాగానికి స్వాగతం."),
        ("Complex Agglutination", "హైదరాబాద్‌లోని ముఖ్యమైన పరిశోధనా కేంద్రాలు."),
        ("Tanglish Telangana", "emmi doing ra baigan? yada poyinav eppuno chusa."),
        ("Tanglish Andhra", "eti chestunnav andi? bagunnara?"),
        ("Tanglish Rayalaseema", "yami saami agudam pokunda kaludam."),
        ("Code-Switched Tech", "API response load avvaledhu, server restart chesa."),
        ("OOV / Robustness Test", "Telugu123! @#$% chusa 🚀 α-β-γ physics")
    ]

    print("=" * 70)
    print("🔍 AUDITING MASTER 128K TOKENIZER")
    print("=" * 70)

    for category, text in test_samples:
        tokens = tokenizer.tokenize(text)
        
        # Normalize input string before comparison to account for stripped ZWNJ/ZWJ control chars
        normalized_text = tokenizer.backend_tokenizer.normalizer.normalize_str(text)
        decoded = tokenizer.decode(tokenizer.encode(text))
        
        exact_pass = "✅ PASS" if decoded.strip() == normalized_text.strip() else "❌ FAIL"
        
        print(f"\n[{category}]")
        print(f"  Input   : {text}")
        print(f"  Tokens ({len(tokens)}) : {tokens}")
        print(f"  Decode  : {exact_pass}")

if __name__ == "__main__":
    audit()