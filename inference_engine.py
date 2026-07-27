"""
BeatHunter AI - Inference Engine

Current behaviour:

1. Spectrogram parameters are pinned to what ACTUALLY produced the training images
   (n_mels=128, no fmin/fmax, no explicit cmap). An earlier edit raised these to
   n_mels=256 / fmin=20 / fmax=11000 to match a v2.0 rewrite of data_processor.py,
   but that rewrite was never run against the dataset — so the model was being fed
   an input distribution it had never seen (measured correlation vs. the real
   training images: 0.31 with the raised values, 0.96 with these).
   ⚠️  Do not raise these again without regenerating the dataset AND retraining.

2. Spectrograms are rendered to PNG bytes in RAM via io.BytesIO — no temp file per
   segment — using a standalone matplotlib Figure rather than pyplot, so the render
   path is thread-safe under run_in_executor.

3. Decode/resize mirrors model_trainer.create_dataset.load_image exactly:
   decode_png(channels=3) → tf.image.resize (bilinear) → float32 in [0, 255].

4. The embedding sub-model locates its pooling layer by TYPE, not by name — the
   saved model's layer is 'global_average_pooling2d_1', not 'global_average_pooling2d'.

5. Soft Voting: softmax probability vectors are summed across segments, then argmax.
   A segment 99% confident in "Travis" outweighs three 35% "Drake" segments.

6. FAISS index is read-only during inference. analyze_beat() never rebuilds it;
   rebuild is triggered manually via rebuild_vector_database_if_needed().

7. BPM clamping + half-time correction: results outside 60-200 BPM are doubled or halved.
"""

import io
import os
import sys
import json
import threading
import subprocess
import librosa
import librosa.display
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import scipy.stats as stats
import faiss

# --- CONFIGURATION ---
TEMP_FOLDER   = "temp_inference"
MODEL_PATH    = "models/beathunter_model.keras"
CLASSES_PATH  = "models/classes.json"

PROCESSED_CSV    = "dataset_processed/metadata_processed.csv"
IMG_DIR          = "dataset_processed/images"
FAISS_INDEX_PATH = "models/vector_db.index"
VECTOR_META_PATH = "models/vector_meta.csv"

# ============================================================
# MODULE-LEVEL CACHE
# Model ve FAISS index her request'te diskten yüklenmek yerine
# ilk çağrıda bir kez yüklenir ve bellekte tutulur.
# Thread-safe: run_in_executor çağrıları için Lock kullanılır.
# ============================================================
_cache_lock        = threading.Lock()
_cached_model      = None
_cached_class_names = None
_cached_faiss_index = None
_cached_faiss_meta  = None


def _get_model():
    """Returns the cached Keras model, loading it on first call."""
    global _cached_model, _cached_class_names
    if _cached_model is None:
        with _cache_lock:
            if _cached_model is None:
                print("🔄 Model ilk kez yükleniyor (sonraki istekler anında çalışır)...")
                _cached_model = tf.keras.models.load_model(MODEL_PATH)
                with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
                    _cached_class_names = json.load(f)
                print("✅ Model cache'e alındı.")
    return _cached_model, _cached_class_names


def build_embedding_model(model):
    """
    Builds a sub-model that outputs the pooled feature vector used for FAISS search.

    Locates the pooling layer BY TYPE rather than by name. The name is not stable:
    model_trainer.build_efficientnet_multihead() is called twice in one session
    (phase 1, then phase 2), so Keras uniquifies the second one to
    'global_average_pooling2d_1' — and the saved model is that second one.
    Hardcoding 'global_average_pooling2d' raised ValueError on every request.
    """
    # Forward order: the embedding point is the first top-level pooling layer,
    # i.e. the one directly after the backbone and before the shared dense trunk.
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            return tf.keras.Model(inputs=model.input, outputs=layer.output)

    raise ValueError(
        "No GlobalAveragePooling2D layer found in the model — cannot build the "
        "embedding sub-model. Available layers: "
        f"{[(l.name, type(l).__name__) for l in model.layers]}"
    )


def _get_faiss():
    """Returns the cached FAISS index and metadata, loading on first call."""
    global _cached_faiss_index, _cached_faiss_meta
    if _cached_faiss_index is None:
        with _cache_lock:
            if _cached_faiss_index is None:
                _cached_faiss_index = faiss.read_index(FAISS_INDEX_PATH)
                _cached_faiss_meta  = pd.read_csv(VECTOR_META_PATH)
    return _cached_faiss_index, _cached_faiss_meta


SAMPLE_RATE      = 22050
SEGMENT_DURATION = 3
NUM_SEGMENTS     = 8

# --- SPECTROGRAM PARAMETERS ---
# ⚠️ These mirror the code that ACTUALLY produced the 873K training images on disk
# (the pre-v2.0 data_processor.py, git HEAD), NOT the v2.0 comments in that file.
#
# The v2.0 edit to data_processor.py (n_mels=256, fmin=20, fmax=11000) was never run
# against the dataset — and its "skip already-processed" branch would have silently
# skipped every existing PNG anyway. Feeding the model 256-mel/band-limited input
# measured a 0.31 correlation against the real training images; these values measure 0.96.
#
# Do NOT "upgrade" these without regenerating the dataset AND retraining. See TODO.md 4.2.
N_MELS  = 128
HOP_LEN = 512
IMG_SIZE = (224, 224)   # must match model_trainer.py IMG_SIZE


def extract_bpm_and_key(y, sr):
    """
    Extracts Tempo (BPM) and Musical Key from audio.

    v2.0: start_bpm=120 guides tracker toward Hip-hop tempos.
    Half-time/double-time correction applied, result clamped to [60, 200].
    """
    print("🥁 Analyzing BPM and Musical Key...")

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120)
    bpm = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)

    # Half-time correction
    if bpm < 60:
        bpm *= 2
    elif bpm > 200:
        bpm /= 2

    bpm = max(60, min(200, bpm))

    # Krumhansl-Schmuckler key detection
    chromagram  = librosa.feature.chroma_stft(y=y, sr=sr)
    mean_chroma = np.mean(chromagram, axis=1)

    chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    maj_profile   = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile   = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    maj_correlations = [stats.pearsonr(mean_chroma, np.roll(maj_profile, i))[0] for i in range(12)]
    min_correlations = [stats.pearsonr(mean_chroma, np.roll(min_profile, i))[0] for i in range(12)]

    best_maj_idx = int(np.argmax(maj_correlations))
    best_min_idx = int(np.argmax(min_correlations))

    if maj_correlations[best_maj_idx] > min_correlations[best_min_idx]:
        key = f"{chroma_to_key[best_maj_idx]} Major"
    else:
        key = f"{chroma_to_key[best_min_idx]} Minor"

    return round(bpm), key


def render_spectrogram_png(y_segment, sr):
    """
    Renders a mel-spectrogram to PNG bytes, entirely in RAM.

    This reproduces the exact code path that generated the training images
    (pre-v2.0 data_processor.py): same n_mels, no fmin/fmax, no explicit cmap,
    figsize=(3, 3), axis off, bbox_inches='tight', pad_inches=0.

    ⚠️ Every argument here is load-bearing for train/inference parity:
      - n_mels=128 and no fmin/fmax  → the band the model was trained on
      - NO cmap argument             → librosa auto-selects (passing 'viridis' breaks parity)
      - bbox_inches='tight'          → yields the 232x231 canvas training used

    Uses a standalone Figure (not pyplot) so it is thread-safe under
    run_in_executor — pyplot's global figure state is not.
    """
    mels    = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=N_MELS)
    mels_db = librosa.power_to_db(mels, ref=np.max)

    fig = Figure(figsize=(3, 3))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    librosa.display.specshow(mels_db, sr=sr, hop_length=HOP_LEN, ax=ax)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    return buf.getvalue()


def audio_to_spectrogram_array(y_segment, sr):
    """
    Converts a 3-second audio segment into a model-ready (1, 224, 224, 3) array.

    Decoding and resizing mirror model_trainer.create_dataset.load_image exactly:
    decode_png(channels=3) → tf.image.resize (bilinear) → float32 in [0, 255].
    EfficientNetB3 has built-in rescaling, so no manual normalization here.
    """
    png_bytes = render_spectrogram_png(y_segment, sr)

    img = tf.image.decode_png(png_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)

    return np.expand_dims(img.numpy(), 0)


def predict_with_majority_voting(model, y_full, sr, class_names):
    """
    Extracts NUM_SEGMENTS uniformly from the usable portion of the track
    and determines labels via Soft Voting (summed softmax probabilities).

    v2.0: Replaces Counter-based hard voting with probability accumulation.
    High-confidence segments naturally outweigh uncertain ones.
    """
    samples_per_segment = int(SEGMENT_DURATION * sr)
    total_samples       = len(y_full)

    skip_start   = min(int(30 * sr), total_samples // 4)
    skip_end     = max(total_samples - int(30 * sr), total_samples * 3 // 4)
    usable_audio = y_full[skip_start:skip_end]

    if len(usable_audio) < samples_per_segment:
        usable_audio = y_full

    max_start = len(usable_audio) - samples_per_segment
    starts    = [0] if max_start <= 0 else np.linspace(0, max_start, NUM_SEGMENTS, dtype=int)

    # Soft voting accumulators
    genre_probs  = None
    artist_probs = None
    mood_probs   = None
    all_vectors  = []

    # Build embedding sub-model once outside the loop
    embed_model = build_embedding_model(model)

    print(f"🔍 Analyzing {len(starts)} segments (Soft Voting)...")

    for start in starts:
        segment = usable_audio[start: start + samples_per_segment]
        if len(segment) < samples_per_segment:
            continue

        img_array = audio_to_spectrogram_array(segment, sr)
        preds     = model.predict(img_array, verbose=0)

        # Accumulate softmax probability vectors
        if genre_probs is None:
            genre_probs  = preds[0][0].copy()
            artist_probs = preds[1][0].copy()
            mood_probs   = preds[2][0].copy()
        else:
            genre_probs  += preds[0][0]
            artist_probs += preds[1][0]
            mood_probs   += preds[2][0]

        vec = embed_model.predict(img_array, verbose=0)
        all_vectors.append(vec[0])

    # Winner = argmax of accumulated probability sums
    predicted_genre  = class_names['genre'] [int(np.argmax(genre_probs))]
    predicted_artist = class_names['artist'][int(np.argmax(artist_probs))]
    predicted_mood   = class_names['mood']  [int(np.argmax(mood_probs))]

    mean_vector = np.mean(all_vectors, axis=0, keepdims=True)

    return predicted_genre, predicted_artist, predicted_mood, mean_vector


def download_youtube_audio(song_name):
    """Downloads audio from YouTube using yt-dlp."""
    import yt_dlp
    print(f"🔍 Searching YouTube for: {song_name}")
    os.makedirs(TEMP_FOLDER, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{TEMP_FOLDER}/downloaded_song.%(ext)s',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        'quiet': True, 'noplaylist': True, 'default_search': 'ytsearch1'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([song_name])
    return os.path.join(TEMP_FOLDER, "downloaded_song.mp3")


def separate_audio(input_path):
    """Uses Demucs AI to isolate the instrumental beat from vocals."""
    print("✂️ AI Source Separation in progress (Demucs)...")
    command = (
        f'"{sys.executable}" -m demucs -n htdemucs --two-stems=vocals '
        f'--mp3 --mp3-bitrate 320 "{input_path}" -o "{TEMP_FOLDER}"'
    )
    subprocess.run(command, shell=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    beat_path = os.path.join(TEMP_FOLDER, "htdemucs", base_name, "no_vocals.mp3")
    if os.path.exists(beat_path):
        return beat_path
    raise Exception("Source separation failed.")


def rebuild_vector_database_if_needed():
    """
    Admin-only: Rebuilds the FAISS index if missing or outdated.

    Call this manually after retraining, NOT during inference:
        python -c "from inference_engine import rebuild_vector_database_if_needed; rebuild_vector_database_if_needed()"
    """
    if os.path.exists(FAISS_INDEX_PATH):
        model_time = os.path.getmtime(MODEL_PATH)
        index_time = os.path.getmtime(FAISS_INDEX_PATH)
        if model_time <= index_time:
            print("✅ FAISS index is up-to-date.")
            return
        print("🔄 Model newer than index — rebuilding...")
        os.remove(FAISS_INDEX_PATH)
    else:
        print("🧬 No index found — building from scratch...")

    model = tf.keras.models.load_model(MODEL_PATH)
    embed_model = build_embedding_model(model)

    df        = pd.read_csv(PROCESSED_CSV)
    vectors   = []
    BATCH     = 256
    img_paths = [os.path.join(IMG_DIR, f) for f in df['filename']]

    def load_img_tf(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        return tf.cast(img, tf.float32)  # EfficientNetB3 expects [0, 255]

    ds = (
        tf.data.Dataset.from_tensor_slices(img_paths)
        .map(load_img_tf, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH)
        .prefetch(tf.data.AUTOTUNE)
    )

    total = len(img_paths)
    processed = 0
    for batch in ds:
        vecs = embed_model.predict(batch, verbose=0)
        vectors.extend(vecs)
        processed += len(batch)
        if processed % 500 == 0:
            print(f"   ⏳ {processed:,} / {total:,}")

    vectors_np = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors_np)

    index = faiss.IndexFlatIP(vectors_np.shape[1])
    index.add(vectors_np)

    faiss.write_index(index, FAISS_INDEX_PATH)
    df.to_csv(VECTOR_META_PATH, index=False)

    # Invalidate cache so next request picks up the fresh index
    global _cached_faiss_index, _cached_faiss_meta
    _cached_faiss_index = None
    _cached_faiss_meta  = None

    print("✅ FAISS index rebuilt successfully!")


def analyze_beat(audio_path):
    """
    Main inference controller.

    v2.0: No longer calls build_vector_database().
    Reads the existing FAISS index only. Raises FileNotFoundError if missing.
    """
    os.makedirs(TEMP_FOLDER, exist_ok=True)

    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_INDEX_PATH}'. "
            "Run rebuild_vector_database_if_needed() first."
        )

    model, class_names = _get_model()

    print("🎵 Loading audio...")
    y_full, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    bpm, key = extract_bpm_and_key(y_full, sr)

    predicted_genre, predicted_artist, predicted_mood, mean_vector = \
        predict_with_majority_voting(model, y_full, sr, class_names)

    formatted_output = f"{predicted_artist} Type / {predicted_mood} / {predicted_genre}"

    # Save mid-segment spectrogram for UI (disk write kept only for the display image).
    # Reuses the shared renderer — no pyplot global state, safe under run_in_executor.
    # NOTE: this path is still fixed/shared across concurrent requests — see TODO.md 3.5.
    mid_start = len(y_full) // 2
    mid_seg   = y_full[mid_start: mid_start + int(SEGMENT_DURATION * sr)]
    with open(f"{TEMP_FOLDER}/temp_spec.png", "wb") as f:
        f.write(render_spectrogram_png(mid_seg, sr))

    # Query FAISS (read-only, cached)
    index, db_meta = _get_faiss()

    query_vector = np.array(mean_vector, dtype=np.float32)
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, 10)

    recommendations = []
    seen = set()
    for i, idx in enumerate(indices[0]):
        raw_name = db_meta.iloc[idx]['filename'].split('_slice')[0]
        if raw_name in seen:
            continue
        recommendations.append({
            "id":    raw_name,
            "title": f"{db_meta.iloc[idx]['artist_type']} / {db_meta.iloc[idx]['mood']} / {db_meta.iloc[idx]['genre']}",
            "score": float(distances[0][i] * 100)
        })
        seen.add(raw_name)
        if len(recommendations) == 5:
            break

    return formatted_output, recommendations, int(bpm), str(key)


if __name__ == "__main__":
    song_input = input("🎵 Enter song query: ")
    try:
        mp3_path  = download_youtube_audio(song_input)
        beat_path = separate_audio(mp3_path)
        print("\n🧠 Analyzing...")
        result, recs, bpm, key = analyze_beat(beat_path)
        print(f"\n🎯 {result} | BPM: {bpm} | Key: {key}")
        for i, r in enumerate(recs):
            print(f"  {i+1}. {r['title']} — {r['score']:.1f}%")
    except Exception as e:
        print(f"❌ {e}")