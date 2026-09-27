#!/usr/bin/env python3
"""
Phase 1: Fixed Unified Authentic Telugu Data Extraction Engine
Fully updated Hugging Face Dataset configs, OpenSLR endpoints, and fallback handlers.
"""

import os
import sys
import re
import csv
import json
import time
import hashlib
import unicodedata
import urllib.request
from typing import Set, Tuple
from datasets import load_dataset
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION & THRESHOLDS
# ==============================================================================
OUTPUT_CORPUS_FILE = "final_authentic_telugu_corpus.txt"
MIN_LINE_LENGTH = 12
MAX_LINE_LENGTH = 1500

HF_TOKEN = os.getenv("HF_TOKEN", None)

# Extraction Limits
LIMIT_FLEURS = 100000
LIMIT_SWECHA = 100000
LIMIT_AKSHARANTAR = 350000
LIMIT_SANGRAHA = 400000
LIMIT_DRAVIDIAN_LANGTECH = 150000
LIMIT_L3CUBE = 150000
LIMIT_WIKIPEDIA = 200000

# Regex Cleaning Patterns
URL_REGEX = re.compile(r'https?://\S+|www\.\S+')
HTML_REGEX = re.compile(r'<.*?>')
EXTRA_SPACES_REGEX = re.compile(r'\s+')
TELUGU_SCRIPT_REGEX = re.compile(r'[\u0C00-\u0C7F]')
LATIN_SCRIPT_REGEX = re.compile(r'[a-zA-Z]')

seen_hashes: Set[str] = set()

stats = {
    "total_raw_lines_processed": 0,
    "total_unique_lines_saved": 0,
    "telugu_script_lines": 0,
    "tanglish_latin_lines": 0,
    "nfc_normalized_fixes": 0,
    "sources_collected": {}
}

# ==============================================================================
# CLEANING & DEDUPLICATION CORE
# ==============================================================================
def clean_and_normalize(text: str) -> Tuple[str, bool]:
    if not text:
        return "", False
    
    normalized = unicodedata.normalize('NFC', str(text))
    was_modified = (normalized != text)
    
    text = URL_REGEX.sub('', normalized)
    text = HTML_REGEX.sub('', text)
    text = EXTRA_SPACES_REGEX.sub(' ', text).strip()
    
    return text, was_modified

def is_valid_and_unique(text: str, source_name: str, file_handle) -> bool:
    global stats
    stats["total_raw_lines_processed"] += 1
    
    if not (MIN_LINE_LENGTH <= len(text) <= MAX_LINE_LENGTH):
        return False
    
    line_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if line_hash in seen_hashes:
        return False
    
    seen_hashes.add(line_hash)
    file_handle.write(text + "\n")
    
    stats["total_unique_lines_saved"] += 1
    stats["sources_collected"][source_name] = stats["sources_collected"].get(source_name, 0) + 1
    
    has_telugu = bool(TELUGU_SCRIPT_REGEX.search(text))
    has_latin = bool(LATIN_SCRIPT_REGEX.search(text))
    
    if has_telugu:
        stats["telugu_script_lines"] += 1
    elif has_latin:
        stats["tanglish_latin_lines"] += 1
        
    return True

# ==============================================================================
# UPDATED EXTRACTOR MODULES
# ==============================================================================

def extract_fleurs_and_spoken(file_handle):
    """Source 1: Google FLEURS Telugu (High-quality spoken conversational transcripts)"""
    print("\n[1/7] 🎙️ Extracting Spoken Telugu Transcripts (Google FLEURS te_in)...")
    count = 0
    try:
        ds = load_dataset("google/fleurs", "te_in", split="train", streaming=True, token=HF_TOKEN)
        for row in ds:
            raw_text = row.get("raw_transcription", row.get("transcription", ""))
            text, fixed = clean_and_normalize(raw_text)
            if fixed: stats["nfc_normalized_fixes"] += 1
            
            if is_valid_and_unique(text, "Google-FLEURS-Spoken", file_handle):
                count += 1
                if count >= LIMIT_FLEURS:
                    break
        print(f"  ✓ Added {count:,} spoken Telugu lines.")
    except Exception as e:
        print(f"  ⚠️ Skipping FLEURS extraction: {e}")

def extract_swecha_datasets(file_handle):
    """Source 2: Swecha Telugu Dataset"""
    print("\n[2/7] 🌾 Extracting Swecha Telugu Datasets...")
    count = 0
    swecha_repos = ["swechatelangana/chandamama-kathalu"]
    
    for repo in swecha_repos:
        try:
            ds = load_dataset(repo, split="train", streaming=True, token=HF_TOKEN)
            for row in ds:
                text_content = row.get("text", row.get("sentence", row.get("content", "")))
                if not text_content:
                    continue
                
                for line in str(text_content).split('\n'):
                    text, fixed = clean_and_normalize(line)
                    if fixed: stats["nfc_normalized_fixes"] += 1
                    
                    if is_valid_and_unique(text, "Swecha-Telugu", file_handle):
                        count += 1
                        if count >= LIMIT_SWECHA:
                            break
                if count >= LIMIT_SWECHA:
                    break
        except Exception as e:
            print(f"  ⚠️ Swecha dataset skipped (Requires HF_TOKEN): {e}")
            
    print(f"  ✓ Added {count:,} lines from Swecha Datasets.")

def extract_aksharantar_tanglish(file_handle):
    """Source 3: AI4Bharat Aksharantar (Fixed config: uses default + language filtering)"""
    print("\n[3/7] 🔤 Extracting AI4Bharat Aksharantar (Authentic Tanglish)...")
    count = 0
    try:
        ds = load_dataset("ai4bharat/Aksharantar", "default", split="train", streaming=True, token=HF_TOKEN)
        for row in ds:
            # Filter strictly for Telugu language pairs
            lang = row.get("language", row.get("lang", ""))
            if lang and lang != "te":
                continue
                
            native = str(row.get("native word", "")).strip()
            english = str(row.get("english word", "")).strip()
            
            if native and english:
                combined_phrase, fixed = clean_and_normalize(f"{english} {native}")
                if fixed: stats["nfc_normalized_fixes"] += 1
                
                if is_valid_and_unique(combined_phrase, "Aksharantar-Tanglish", file_handle):
                    count += 1
                    if count >= LIMIT_AKSHARANTAR:
                        break
        print(f"  ✓ Added {count:,} authentic Tanglish entries from Aksharantar.")
    except Exception as e:
        print(f"  ❌ Aksharantar Extraction Error: {e}")

def extract_dravidian_langtech(file_handle):
    """Source 4: Dravidian CodeMix (Real Social Media Tanglish & English Code-Mix)"""
    print("\n[4/7] 💬 Extracting Dravidian Social Media Tanglish & Code-Mix...")
    count = 0
    repos = ["DravidianCodeMix/DravidianCodeMix", "dharunim/DravidianLangTech-Telugu"]
    
    for repo in repos:
        try:
            ds = load_dataset(repo, split="train", streaming=True, token=HF_TOKEN)
            for row in ds:
                raw_text = row.get("text", row.get("comment", ""))
                text, fixed = clean_and_normalize(raw_text)
                if fixed: stats["nfc_normalized_fixes"] += 1
                
                if is_valid_and_unique(text, "Dravidian-Social", file_handle):
                    count += 1
                    if count >= LIMIT_DRAVIDIAN_LANGTECH:
                        break
            if count > 0:
                break
        except Exception as e:
            continue
            
    print(f"  ✓ Added {count:,} real social media code-mixed lines.")

def extract_ai4bharat_sangraha(file_handle):
    """Source 5: AI4Bharat Sangraha (Fixed config: uses 'verified' + row language filter)"""
    print("\n[5/7] 📰 Extracting AI4Bharat Sangraha (Formal Telugu)...")
    count = 0
    try:
        ds = load_dataset("ai4bharat/sangraha", "verified", split="train", streaming=True, token=HF_TOKEN)
        for row in ds:
            # Check row-level language tag for Telugu
            lang = row.get("lang", row.get("language", ""))
            if lang and lang != "tel_Telu":
                continue
                
            raw_text = row.get("text", "")
            for line in raw_text.split('\n'):
                text, fixed = clean_and_normalize(line)
                if fixed: stats["nfc_normalized_fixes"] += 1
                
                if is_valid_and_unique(text, "AI4Bharat-Sangraha", file_handle):
                    count += 1
                    if count >= LIMIT_SANGRAHA:
                        break
            if count >= LIMIT_SANGRAHA:
                break
        print(f"  ✓ Added {count:,} lines from AI4Bharat Sangraha.")
    except Exception as e:
        print(f"  ❌ Sangraha Error: {e}")

def extract_l3cube_telugu(file_handle):
    """Source 6: L3Cube Telugu Corpus (Fixed repo name)"""
    print("\n[6/7] 📚 Extracting L3Cube-Telugu Corpus...")
    count = 0
    try:
        ds = load_dataset("l3cube-pune/telugu-corpus", split="train", streaming=True, token=HF_TOKEN)
        for row in ds:
            raw_text = row.get("text", "")
            text, fixed = clean_and_normalize(raw_text)
            if fixed: stats["nfc_normalized_fixes"] += 1
            
            if is_valid_and_unique(text, "L3Cube-Telugu", file_handle):
                count += 1
                if count >= LIMIT_L3CUBE:
                    break
        print(f"  ✓ Added {count:,} lines from L3Cube Telugu.")
    except Exception as e:
        print(f"  ⚠️ Skipping L3Cube: {e}")

def extract_telugu_wikipedia(file_handle):
    """Source 7: Wikimedia Telugu Wikipedia"""
    print("\n[7/7] 🌐 Extracting Wikimedia Telugu Wikipedia...")
    count = 0
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.te", split="train", streaming=True, token=HF_TOKEN)
        for row in ds:
            raw_text = row.get("text", "")
            for line in raw_text.split('\n'):
                text, fixed = clean_and_normalize(line)
                if fixed: stats["nfc_normalized_fixes"] += 1
                
                if is_valid_and_unique(text, "Wikipedia-Te", file_handle):
                    count += 1
                    if count >= LIMIT_WIKIPEDIA:
                        break
            if count >= LIMIT_WIKIPEDIA:
                break
        print(f"  ✓ Added {count:,} lines from Telugu Wikipedia.")
    except Exception as e:
        print(f"  ❌ Wikipedia Error: {e}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    start_time = time.time()
    print("=" * 70)
    print("🚀 PHASE 1: UNIFIED AUTHENTIC TELUGU DATA ENGINE (FIXED & AUDITED)")
    print("=" * 70)

    with open(OUTPUT_CORPUS_FILE, "w", encoding="utf-8") as out_f:
        extract_fleurs_and_spoken(out_f)
        extract_swecha_datasets(out_f)
        extract_aksharantar_tanglish(out_f)
        extract_dravidian_langtech(out_f)
        extract_ai4bharat_sangraha(out_f)
        extract_l3cube_telugu(out_f)
        extract_telugu_wikipedia(out_f)

    elapsed_time = time.time() - start_time
    file_size_mb = os.path.getsize(OUTPUT_CORPUS_FILE) / (1024 * 1024)

    print("\n" + "=" * 70)
    print("📊 PHASE 1 FINAL EXTRACTION & AUDIT REPORT")
    print("=" * 70)
    print(f"Total Raw Lines Processed : {stats['total_raw_lines_processed']:,}")
    print(f"Total Unique Lines Saved   : {stats['total_unique_lines_saved']:,}")
    print(f"Pure Telugu Script Lines   : {stats['telugu_script_lines']:,} ({stats['telugu_script_lines']/max(1, stats['total_unique_lines_saved'])*100:.1f}%)")
    print(f"Tanglish / Latin Lines     : {stats['tanglish_latin_lines']:,} ({stats['tanglish_latin_lines']/max(1, stats['total_unique_lines_saved'])*100:.1f}%)")
    print(f"Unicode NFC Normalization  : {stats['nfc_normalized_fixes']:,} lines canonicalized")
    print(f"Output File Size          : {file_size_mb:.2f} MB")
    print(f"Total Execution Time      : {elapsed_time:.2f} seconds")
    print("-" * 70)
    print("Breakdown by Source:")
    for src, cnt in stats["sources_collected"].items():
        print(f"  • {src:<25}: {cnt:,} lines")
    print("=" * 70)
    print(f"✅ Master Corpus Saved To: {os.path.abspath(OUTPUT_CORPUS_FILE)}")

if __name__ == "__main__":
    main()