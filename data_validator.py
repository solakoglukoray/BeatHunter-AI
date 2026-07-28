"""
BeatHunter AI - Data Validator (v2.0)

Changes in v2.0:
- Parallel scanning with multiprocessing — (cpu_count - 2) workers
  Scanning 9000+ files sequentially was slow; now ~4x faster.
- Metadata cleanup — corrupt files are removed from metadata.csv too,
  preventing data_processor.py from crashing on missing/broken files.
- Progress counter — shows live scan progress every 500 files.
"""

import os
import csv
import librosa
import pandas as pd
import config
from multiprocessing import Pool, cpu_count

NUM_WORKERS = max(1, cpu_count() - 2)


def check_file(file_path):
    """
    Attempts to load the first second of an audio file.
    Returns (file_path, True) if intact, (file_path, False) if corrupt.
    """
    try:
        librosa.load(file_path, sr=None, duration=1)
        return (file_path, True)
    except Exception as e:
        return (file_path, False, str(e))


def validate_data():
    print(f"🕵️ BeatHunter AI — Data Validator v2.0")
    print(f"   Parallel workers: {NUM_WORKERS} / {cpu_count()} cores\n")

    # Collect all MP3 paths
    all_files = []
    for root, _, files in os.walk(config.DATASET_ROOT):
        for f in files:
            if f.endswith(".mp3"):
                all_files.append(os.path.join(root, f))

    total_files = len(all_files)
    print(f"📂 Found {total_files:,} MP3 files to scan...\n")

    corrupted_paths = []

    # Parallel integrity check
    with Pool(processes=NUM_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(check_file, all_files, chunksize=8), 1):
            if i % 500 == 0:
                print(f"   ⏳ Scanned {i:,} / {total_files:,}...")
            if not result[1]:
                corrupted_paths.append(result[0])
                reason = result[2] if len(result) > 2 else "Unknown"
                print(f"   ❌ CORRUPT: {os.path.basename(result[0])} — {reason}")

    # Delete corrupt files
    deleted = 0
    corrupt_filenames = set()
    for path in corrupted_paths:
        try:
            corrupt_filenames.add(os.path.basename(path))
            os.remove(path)
            deleted += 1
            print(f"   🗑️  Deleted: {os.path.basename(path)}")
        except Exception as e:
            print(f"   ⚠️  Could not delete {os.path.basename(path)}: {e}")

    # Clean metadata.csv — remove rows referencing deleted files
    if corrupt_filenames and os.path.exists(config.METADATA_FILE):
        df = pd.read_csv(config.METADATA_FILE)
        before = len(df)
        df = df[~df['filename'].isin(corrupt_filenames)]
        df.to_csv(config.METADATA_FILE, index=False)
        removed_rows = before - len(df)
        print(f"\n🧹 Metadata cleaned: {removed_rows} rows removed from metadata.csv")

    # Report
    print("\n" + "-" * 35)
    print("📊 VALIDATION REPORT")
    print(f"   Total Files Scanned:     {total_files:,}")
    print(f"   Corrupted Files Deleted: {deleted}")
    if total_files > 0:
        integrity = ((total_files - deleted) / total_files) * 100
        print(f"   Data Integrity Rate:     {integrity:.1f}%")
    else:
        print("   No files found.")
    print("-" * 35)


if __name__ == "__main__":
    validate_data()