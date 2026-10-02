#!/usr/bin/env python3
"""
Production Data Engine - Pure Telugu Corpus Generator
Enforces strict regex file discovery + Unicode character-level script verification.
"""

import os
import sys
import re
import time
import hashlib
import unicodedata
from typing import Set, List
from huggingface_hub import hf_hub_download, HfApi
import pyarrow.parquet as pq
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", None)

OUTPUT_CORPUS_FILE = "final_authentic_telugu_corpus.txt"
CACHE_DIR = os.path.join(os.getcwd(), "hf_parquet_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

LIMITS = {
    "FLEURS": 100_000,
    "CC100": 100_000,
    "Aksharantar": 350_000,
    "SocialMedia": 150_000,
    "Sangraha": 400_000,
    "Glot500": 150_000,
    "Wikipedia": 200_000,
}

URL_REGEX = re.compile(r'https?://\S+|www\.\S+')
HTML_REGEX = re.compile(r'<.*?>')
EXTRA_SPACES_REGEX = re.compile(r'\s+')

# Unicode ranges for strict script validation
TELUGU_SCRIPT = re.compile(r'[\u0C00-\u0C7F]')
OTHER_INDIC_SCRIPTS = re.compile(r'[\u0900-\u0BF9\u0C80-\u0D7F]') # Devanagari, Bengali, Tamil, Kannada, Malayalam, etc.

seen_hashes: Set[str] = set()

# Optional: Reset output file on new run to avoid keeping old contaminated data
if os.path.exists(OUTPUT_CORPUS_FILE):
    print(f"📖 Existing corpus found. Resetting file to ensure 100% pure Telugu...")
    os.remove(OUTPUT_CORPUS_FILE)


def clean_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize('NFC', str(text))
    text = URL_REGEX.sub('', normalized)
    text = HTML_REGEX.sub('', text)
    return EXTRA_SPACES_REGEX.sub(' ', text).strip()


def is_pure_telugu(text: str, min_telugu_ratio: float = 0.70) -> bool:
    """
    Validates that the string contains authentic Telugu script.
    Rejects other Indic scripts and ensures native script dominant presence.
    """
    if not text:
        return False

    # 1. Reject if other Indic scripts are present
    if OTHER_INDIC_SCRIPTS.search(text):
        return False

    # 2. Count total alphabetic characters
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False

    # 3. Count Telugu characters
    telugu_chars = TELUGU_SCRIPT.findall(text)
    if not telugu_chars:
        return False

    # 4. Check Telugu script ratio
    ratio = len(telugu_chars) / len(letters)
    return ratio >= min_telugu_ratio


def auto_discover_and_download(api: HfApi, repo_id: str, strict_regex: str, max_files: int = 10) -> List[str]:
    """Dynamically finds and downloads matching Parquet files using strict regex pattern."""
    downloaded_paths = []
    pattern = re.compile(strict_regex)

    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=HF_TOKEN)
        parquet_files = [f for f in files if f.endswith(".parquet") and pattern.search(f)]

        if not parquet_files:
            print(f"  ⚠️ No files matching pattern '{strict_regex}' found in {repo_id}")
            return []

        parquet_files = parquet_files[:max_files]

        for pf in parquet_files:
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=pf,
                    repo_type="dataset",
                    cache_dir=CACHE_DIR,
                    token=HF_TOKEN
                )
                downloaded_paths.append(local_path)
            except Exception as e:
                print(f"  ⚠️ Error downloading {pf} from {repo_id}: {e}")

    except Exception as e:
        print(f"  ⚠️ Failed to inspect repository {repo_id}: {e}")

    return downloaded_paths


def process_parquet_files(file_paths: List[str], candidate_columns: List[str], max_lines: int, file_handle) -> int:
    """Parses local Parquet files and filters lines with strict Telugu validation."""
    added = 0
    for path in file_paths:
        if not path or not os.path.exists(path):
            continue

        try:
            parquet_file = pq.ParquetFile(path)
            for row_group_idx in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group_idx)
                df = table.to_pandas()

                target_col = next((col for col in candidate_columns if col in df.columns), None)

                # Check for word mapping datasets (e.g. native word column)
                if not target_col and "native word" in df.columns:
                    for val in df["native word"].dropna():
                        cleaned = clean_text(val)
                        if is_pure_telugu(cleaned):
                            h = hashlib.sha256(cleaned.encode('utf-8')).hexdigest()
                            if h not in seen_hashes:
                                seen_hashes.add(h)
                                file_handle.write(cleaned + "\n")
                                added += 1
                                if added >= max_lines:
                                    return added
                    continue

                if not target_col:
                    continue

                for raw_val in df[target_col].dropna():
                    for raw_line in str(raw_val).split('\n'):
                        cleaned = clean_text(raw_line)
                        if len(cleaned) < 5 or len(cleaned) > 1500:
                            continue

                        # Strict Telugu verification step
                        if not is_pure_telugu(cleaned):
                            continue

                        h = hashlib.sha256(cleaned.encode('utf-8')).hexdigest()
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            file_handle.write(cleaned + "\n")
                            added += 1
                            if added >= max_lines:
                                return added
                if added >= max_lines:
                    break
        except Exception as e:
            print(f"  ⚠️ Error parsing Parquet file {os.path.basename(path)}: {e}")

        if added >= max_lines:
            break

    return added


def main():
    start_time = time.time()
    print("=" * 70)
    print("🚀 PRODUCTION DATA ENGINE (STRICT PURE TELUGU)")
    print("=" * 70)

    api = HfApi()

    with open(OUTPUT_CORPUS_FILE, "a", encoding="utf-8") as out_f:

        # [1/7] FLEURS Spoken
        print("\n[1/7] 🎙️ Processing Google FLEURS (te_in)...")
        paths = auto_discover_and_download(api, "google/fleurs", strict_regex=r'te_in/')
        added = process_parquet_files(paths, ["raw_transcription", "transcription", "sentence"], LIMITS["FLEURS"], out_f)
        print(f"  ✓ Added {added:,} pure Telugu spoken lines.")

        # [2/7] Open Telugu Stories / CC100
        print("\n[2/7] 🌾 Processing CC100 (Telugu)...")
        paths = auto_discover_and_download(api, "statmt/cc100", strict_regex=r'(^|/)te(/|\.|\_)')
        added = process_parquet_files(paths, ["text"], LIMITS["CC100"], out_f)
        print(f"  ✓ Added {added:,} story/web lines.")

        # [3/7] Aksharantar Telugu
        print("\n[3/7] 🔤 Processing AI4Bharat Aksharantar (Telugu only)...")
        paths = auto_discover_and_download(api, "eswardivi/Aksharantar", strict_regex=r'(^|/)(te|tel)(/|\_)')
        if not paths:
            paths = auto_discover_and_download(api, "ai4bharat/Aksharantar", strict_regex=r'(^|/)(te|tel)(/|\_)')
        added = process_parquet_files(paths, ["text"], LIMITS["Aksharantar"], out_f)
        print(f"  ✓ Added {added:,} native Telugu entries.")

        # [4/7] Social Media Code-Mix (Telugu only)
        print("\n[4/7] 💬 Processing Dravidian Social Media (Telugu only)...")
        paths = auto_discover_and_download(api, "community-datasets/offenseval_dravidian", strict_regex=r'(^|/)telugu(/|\_)')
        if not paths:
            paths = auto_discover_and_download(api, "dravidianlangtech/hope_edi", strict_regex=r'(^|/)telugu(/|\_)')
        added = process_parquet_files(paths, ["text", "comment", "tweet", "sentence"], LIMITS["SocialMedia"], out_f)
        print(f"  ✓ Added {added:,} social media lines.")

        # [5/7] AI4Bharat Sangraha (Native Telugu Script only: tel_Telu)
        print("\n[5/7] 📰 Processing AI4Bharat Sangraha (tel_Telu script)...")
        paths = auto_discover_and_download(api, "ai4bharat/sangraha", strict_regex=r'(^|/)(tel_Telu|verified/tel)(/|\_)')
        added = process_parquet_files(paths, ["text"], LIMITS["Sangraha"], out_f)
        print(f"  ✓ Added {added:,} formal lines.")

        # [6/7] Glot500 Telugu Corpus
        print("\n[6/7] 📚 Processing Glot500 Telugu Corpus...")
        paths = auto_discover_and_download(api, "cis-lmu/Glot500", strict_regex=r'(^|/)tel_Telu(/|\_)')
        added = process_parquet_files(paths, ["text"], LIMITS["Glot500"], out_f)
        print(f"  ✓ Added {added:,} monolingual lines.")

        # [7/7] Wikimedia Telugu Wikipedia
        print("\n[7/7] 🌐 Processing Wikimedia Telugu Wikipedia (20231101.te)...")
        paths = auto_discover_and_download(api, "wikimedia/wikipedia", strict_regex=r'20231101\.te/')
        added = process_parquet_files(paths, ["text"], LIMITS["Wikipedia"], out_f)
        print(f"  ✓ Added {added:,} Wikipedia lines.")

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(OUTPUT_CORPUS_FILE) / (1024 * 1024) if os.path.exists(OUTPUT_CORPUS_FILE) else 0

    print("\n" + "=" * 70)
    print("✅ PURE TELUGU CORPUS BUILD COMPLETE")
    print("=" * 70)
    print(f"Total Unique Lines Saved : {len(seen_hashes):,}")
    print(f"Corpus File Size         : {file_size_mb:.2f} MB")
    print(f"Total Execution Time     : {elapsed:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()