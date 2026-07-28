"""
BeatHunter AI - Audio Signal Processor (v2.0)

Changes in v2.0:
- n_mels: 128 → 256  (matches inference_engine.py)
- fmin=20, fmax=11000 added  (matches inference_engine.py)
- hop_length=512 explicitly set  (matches inference_engine.py)
- Parallel processing with multiprocessing.Pool → ~4x faster on i9-14900HX
- matplotlib backend set to 'Agg' → no GUI popups, faster rendering
- Skip already-processed files → safe to resume if interrupted
"""

import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — faster, no GUI
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time

# --- CONFIGURATION ---
AUDIO_DIR     = "dataset/audio"
METADATA_FILE = "dataset/metadata.csv"
PROCESSED_DIR = "dataset_processed"
IMG_DIR       = os.path.join(PROCESSED_DIR, "images")
PROCESSED_CSV = os.path.join(PROCESSED_DIR, "metadata_processed.csv")

SAMPLE_RATE      = 22050
DURATION         = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# ⚠️ These must match inference_engine.py and model_trainer.py exactly.
N_MELS  = 256
F_MIN   = 20
F_MAX   = 11000
HOP_LEN = 512

# Number of CPU cores to use for parallel processing.
# Using (max - 2) keeps the system responsive while you use the computer.
# i9-14900HX has 24 threads — this uses 22.
NUM_WORKERS = max(1, cpu_count() // 4)  # i9-14900HX: 24 thread → 6 worker


def save_spectrogram(y, sr, save_path):
    """
    Converts an audio segment into a Mel-Spectrogram and saves as PNG.

    v2.0: n_mels=256, fmin=20, fmax=11000, hop_length=512
    These match inference_engine.py exactly — critical for model accuracy.
    """
    mels    = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        hop_length=HOP_LEN
    )
    mels_db = librosa.power_to_db(mels, ref=np.max)

    plt.figure(figsize=(3, 3))
    plt.axis('off')
    librosa.display.specshow(mels_db, sr=sr, hop_length=HOP_LEN, fmin=F_MIN, fmax=F_MAX, cmap='viridis')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_single_track(args):
    """
    Processes a single audio file into spectrogram slices.
    Designed to be called in parallel via multiprocessing.Pool.

    Returns a list of metadata dicts for each successfully saved slice.
    """
    row, audio_dir, img_dir = args
    audio_filename = row['filename']
    audio_path     = os.path.join(audio_dir, audio_filename)
    results        = []

    if not os.path.exists(audio_path):
        print(f"   ⚠️ Not found, skipping: {audio_filename}")
        return results

    try:
        y, sr    = librosa.load(audio_path, sr=SAMPLE_RATE)
        num_segs = int(len(y) / SAMPLES_PER_TRACK)

        for i in range(num_segs):
            start   = i * SAMPLES_PER_TRACK
            slice_y = y[start: start + SAMPLES_PER_TRACK]

            if len(slice_y) != SAMPLES_PER_TRACK:
                continue

            base_name    = os.path.splitext(audio_filename)[0]
            img_filename = f"{base_name}_slice_{i}.png"
            img_path     = os.path.join(img_dir, img_filename)

            # Skip already-processed slices — safe to resume if interrupted
            if os.path.exists(img_path):
                results.append({
                    "filename":    img_filename,
                    "genre":       row['genre'],
                    "artist_type": row['artist_type'],
                    "mood":        row['mood']
                })
                continue

            save_spectrogram(slice_y, sr, img_path)

            results.append({
                "filename":    img_filename,
                "genre":       row['genre'],
                "artist_type": row['artist_type'],
                "mood":        row['mood']
            })


    except Exception as e:
        print(f"   ❌ Error ({audio_filename}): {e}")

    time.sleep(0.3)

    return results


def process_v2_data():
    """
    Main processing function.
    Reads raw audio metadata, processes all files in parallel,
    and saves spectrogram images + updated metadata CSV.
    """
    print("🚀 BeatHunter AI — Audio-to-Spectrogram Processor v2.0")
    print(f"   n_mels={N_MELS} | fmin={F_MIN} | fmax={F_MAX} | hop_length={HOP_LEN}")
    print(f"   Parallel workers: {NUM_WORKERS} / {cpu_count()} available cores\n")

    os.makedirs(IMG_DIR, exist_ok=True)

    df = pd.read_csv(METADATA_FILE).drop_duplicates()
    print(f"📊 Unique tracks to process: {len(df)}\n")

    # Prepare args for parallel pool
    args_list = [
        (row, AUDIO_DIR, IMG_DIR)
        for _, row in df.iterrows()
    ]

    all_results = []

    # Parallel processing — uses NUM_WORKERS CPU cores simultaneously
    # chunksize=4 batches tasks to reduce inter-process overhead
    total_tracks = len(args_list)
    with Pool(processes=NUM_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_track, args_list, chunksize=4), 1):
            all_results.extend(result)
            if i % 10 == 0 or i == total_tracks:
                pct = (i / total_tracks) * 100
                print(f"   📦 {i}/{total_tracks} tracks processed ({pct:.1f}%) — {len(all_results):,} spectrograms so far")
    # Save processed metadata
    df_processed = pd.DataFrame(all_results)
    df_processed.to_csv(PROCESSED_CSV, index=False)

    print("\n✅ ALL CONVERSIONS COMPLETED!")
    print(f"📂 Images: {IMG_DIR}")
    print(f"📄 Metadata: {PROCESSED_CSV}")
    print(f"🧬 Total training samples: {len(df_processed):,}")


if __name__ == "__main__":
    process_v2_data()