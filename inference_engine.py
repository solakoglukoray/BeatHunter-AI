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
import uuid
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
SPEC_FOLDER   = os.path.join(TEMP_FOLDER, "spectrograms")
MODEL_PATH    = "models/beathunter_model.keras"
CLASSES_PATH  = "models/classes.json"
TAXONOMY_PATH = "models/taxonomy.json"   # artist -> {genre, mood}, for joint decoding

PROCESSED_CSV    = "dataset_processed/metadata_processed.csv"
IMG_DIR          = "dataset_processed/images"
# Track-level index: one row per track (14,235), 1536-dim, ~87 MB.
# Replaces the old slice-level pair (vector_db.index / vector_meta.csv), which was
# built from a different model (d=256) against a different dataset snapshot
# (214,506 rows vs. the current 873,924) and could not be queried at all.
# The old files are left on disk untouched as a fallback.
FAISS_INDEX_PATH = "models/vector_db_track.index"
VECTOR_META_PATH = "models/vector_meta_track.csv"

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
_cached_taxonomy    = None
_cached_embed_model = None


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


def _get_taxonomy(class_names):
    """
    Loads artist -> (genre, mood) and returns it as index arrays aligned to
    class_names, for joint decoding.

    Every one of the 194 artists maps to exactly one genre and one mood in the
    training data — genre and mood are functions of artist, not independent labels.
    The three heads are therefore three noisy views of a single decision, and nothing
    stops them from disagreeing: the model has been observed to emit
    "Philanthrope / Dark / Alternative Hip-Hop" when Philanthrope is
    Lofi Hip-Hop / Melancholic — a triple that does not exist in the data.

    Returns (genre_of_artist, mood_of_artist): int arrays of len(class_names['artist']),
    holding indices into class_names['genre'] and class_names['mood'].
    """
    global _cached_taxonomy
    if _cached_taxonomy is None:
        with _cache_lock:
            if _cached_taxonomy is None:
                with open(TAXONOMY_PATH, 'r', encoding='utf-8') as f:
                    tax = json.load(f)

                g_idx = {g: i for i, g in enumerate(class_names['genre'])}
                m_idx = {m: i for i, m in enumerate(class_names['mood'])}

                missing = [a for a in class_names['artist'] if a not in tax]
                if missing:
                    raise ValueError(
                        f"{len(missing)} artist(s) in classes.json are absent from "
                        f"{TAXONOMY_PATH}: {missing[:5]}. Regenerate the taxonomy from "
                        "dataset_processed/metadata_processed.csv."
                    )

                _cached_taxonomy = (
                    np.array([g_idx[tax[a]['genre']] for a in class_names['artist']], dtype=np.int32),
                    np.array([m_idx[tax[a]['mood']]  for a in class_names['artist']], dtype=np.int32),
                )
    return _cached_taxonomy


def joint_decode(genre_p, artist_p, mood_p, class_names, weights=(1.0, 1.0, 1.0)):
    """
    Picks the artist that best explains all three heads at once, then reads genre and
    mood off the taxonomy:

        score(a) = w_a·log P(artist=a) + w_g·log P(genre=G[a]) + w_m·log P(mood=M[a])

    Because genre and mood are looked up rather than predicted, the returned triple is
    always one the taxonomy actually contains — impossible combinations become
    unrepresentable rather than merely unlikely. The genre and mood heads still get a
    vote: they act as priors over groups of artists sharing that genre/mood.

    Caveat: the three heads share a backbone, so they are not independent evidence and
    summing their log-probs double-counts somewhat. `weights` is exposed for tuning;
    equal weights are the untuned default.

    This is a decode-time fix for a labelling problem (TODO.md 4.3), not a substitute
    for it.
    """
    gi, ai, mi = joint_decode_indices(genre_p, artist_p, mood_p, class_names, weights)
    return class_names['genre'][gi], class_names['artist'][ai], class_names['mood'][mi]


def joint_decode_indices(genre_p, artist_p, mood_p, class_names, weights=(1.0, 1.0, 1.0)):
    """joint_decode(), but returning (genre_idx, artist_idx, mood_idx)."""
    g_of_a, m_of_a = _get_taxonomy(class_names)
    eps = 1e-12
    wa, wg, wm = weights

    scores = (wa * np.log(artist_p + eps)
              + wg * np.log(genre_p + eps)[g_of_a]
              + wm * np.log(mood_p + eps)[m_of_a])

    best = int(np.argmax(scores))
    return int(g_of_a[best]), best, int(m_of_a[best])


# A prediction counts as reliable only if its probability beats a uniform guess by at
# least this factor. Normalising by class count makes the three heads comparable:
# 194 artists, 65 genres, 6 moods have very different uniform baselines.
#
# The currently saved model scores roughly 2.7x (artist), 2.1x (genre) and 1.4x (mood)
# — i.e. barely distinguishable from guessing, which matches its measured accuracy of
# 0.5% / 2.2% / 25.0% on its OWN training images. A working model should sit far above
# this threshold, so it gates today's output without capping a retrained one.
CONFIDENCE_MIN_RATIO = 5.0


def _head_confidence(probs, idx, label):
    p = float(probs[idx])
    return {"label": label, "probability": p, "ratio_vs_chance": p * len(probs)}


def classification_confidence(genre_p, artist_p, mood_p, gi, ai, mi, class_names):
    """
    Per-head confidence for the chosen labels, plus an overall `reliable` flag.

    `ratio_vs_chance` is P(chosen) / (1/n_classes): 1.0 means "no better than guessing".
    Consumers should treat reliable=False as "do not present this as a result".
    """
    heads = {
        "genre":  _head_confidence(genre_p,  gi, class_names['genre'][gi]),
        "artist": _head_confidence(artist_p, ai, class_names['artist'][ai]),
        "mood":   _head_confidence(mood_p,   mi, class_names['mood'][mi]),
    }
    heads["reliable"] = all(h["ratio_vs_chance"] >= CONFIDENCE_MIN_RATIO
                            for h in heads.values() if isinstance(h, dict))
    heads["min_ratio_required"] = CONFIDENCE_MIN_RATIO
    return heads


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


def _get_embedding_model(model):
    """Cached embedding sub-model — rebuilding it per request wastes graph construction."""
    global _cached_embed_model
    if _cached_embed_model is None:
        with _cache_lock:
            if _cached_embed_model is None:
                _cached_embed_model = build_embedding_model(model)
    return _cached_embed_model


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
NUM_SEGMENTS     = 8    # classification (soft voting), skips intro/outro

# Retrieval sampling — MUST equal the slices-per-track the FAISS index was built with.
# The index was built with 4 slices per track at linspace over the full 3s grid.
# Changing this without rebuilding the index silently destroys retrieval quality
# (measured: aligned-4 → median self-match rank 1, aligned-8 → 186).
RETRIEVAL_SEGMENTS = 4

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


# Beat trackers routinely report hip-hop tempos at half or double speed. Both 70 and
# 140 sit inside any sane [60, 200] range, so a range check alone never catches it —
# the fold has to target the tempo band the genre actually lives in.
BPM_RANGE  = (60, 200)      # hard bounds; anything outside is certainly wrong
# Deliberately wide at the top. Folding 70 -> 140 fixes the common half-time miss, but
# a tight upper bound would also fold genuinely fast tempos: drill and jersey sit in the
# 160-175 range, and halving those to ~87 would be a new error introduced to fix an old
# one. Below 78 a doubling is nearly always right; above 180 it is nearly always wrong.
BPM_TARGET = (78, 180)


def _fold_bpm(bpm):
    """
    Folds a tempo into the plausible range by repeated halving/doubling.

    The previous version applied a single *2 or /2 and then clamped, so 25 BPM became
    50 and was then clamped to 60 — a number the tracker never suggested. Doubling
    repeatedly reaches 100 instead.

    Once inside the hard range, a value below the target band is doubled (and above,
    halved) if that lands it inside — this is the 70-vs-140 trap case.
    """
    if bpm <= 0:
        return float(BPM_TARGET[0])

    lo, hi = BPM_RANGE
    while bpm < lo:
        bpm *= 2
    while bpm > hi:
        bpm /= 2

    t_lo, t_hi = BPM_TARGET
    if bpm < t_lo and t_lo <= bpm * 2 <= hi:
        bpm *= 2
    elif bpm > t_hi and lo <= bpm / 2 <= hi:
        bpm /= 2

    return float(bpm)


def extract_bpm_and_key(y, sr):
    """
    Extracts tempo (BPM) and musical key from audio.

    Neither value comes from the model — both are measured from the signal, so they
    stay trustworthy while the classifier does not.
    """
    print("🥁 Analyzing BPM and Musical Key...")

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120)
    bpm = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)
    bpm = _fold_bpm(bpm)

    # Key detection on the HARMONIC component only. chroma_stft on a full mix is
    # dominated by drum transients, which smear energy across every pitch class —
    # on beats, that is most of the signal. Separating harmonics out and using CQT
    # (log-spaced bins, so one bin per semitone) is markedly more reliable.
    y_harmonic  = librosa.effects.harmonic(y)
    chromagram  = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
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
    return segments_to_batch([y_segment], sr)


def segments_to_batch(segments, sr):
    """
    Renders several segments into one (N, 224, 224, 3) batch.

    Callers should build a batch and run a single predict() over it. One call per
    segment costs a full graph invocation each time, and the previous code made two of
    them per segment (classification, then embedding) — 16 invocations for one track
    where 12 images in two calls will do.
    """
    imgs = []
    for seg in segments:
        img = tf.image.decode_png(render_spectrogram_png(seg, sr), channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        imgs.append(tf.cast(img, tf.float32))
    return np.stack([i.numpy() for i in imgs], axis=0)


def compute_retrieval_embedding(model, y_full, sr):
    """
    Builds the FAISS query vector using the SAME segment strategy the index was
    built with: RETRIEVAL_SEGMENTS positions from linspace over the track's full
    3-second grid — intro/outro included, no skipping.

    ⚠️ This must mirror the index builder exactly. Measured self-match rank over
    6 in-index tracks (1 = perfect, ~7,117 = random):

        current classification sampling (skip 30s, 8 segments) : median 4,856
        index-aligned, 4 segments                              : median     1
        index-aligned, 8 segments                              : median   186

    Note the third row: matching the *strategy* is not enough — the segment COUNT
    must match too. Aligned-8 hits the same 4 positions plus 4 others, and the extra
    positions dilute the mean away from the index entry. If the index is ever rebuilt
    with a different slice count, change RETRIEVAL_SEGMENTS to match it.
    """
    spt      = int(SEGMENT_DURATION * sr)
    n_slices = int(len(y_full) / spt)
    if n_slices < 1:
        return None

    idxs = np.unique(np.linspace(0, n_slices - 1, min(RETRIEVAL_SEGMENTS, n_slices)).astype(int))

    segments = [y_full[i * spt:(i + 1) * spt] for i in idxs]
    segments = [s for s in segments if len(s) == spt]
    if not segments:
        return None

    embed_model = _get_embedding_model(model)
    vectors     = embed_model.predict(segments_to_batch(segments, sr), verbose=0)
    return np.mean(vectors, axis=0, keepdims=True)


def predict_with_majority_voting(model, y_full, sr, class_names, joint=True):
    """
    Extracts NUM_SEGMENTS uniformly from the usable portion of the track
    and determines labels via Soft Voting (averaged softmax probabilities).

    Soft voting: probability vectors are summed across segments, then averaged.
    High-confidence segments naturally outweigh uncertain ones.

    joint=True  → the three heads are resolved together via joint_decode(), so the
                  returned triple is always taxonomy-consistent.
    joint=False → independent argmax per head (the old behaviour), kept so the two
                  can be compared.

    Classification deliberately skips the first/last 30s (intro/outro bias) and uses
    NUM_SEGMENTS segments. Retrieval does NOT share this sampling — it needs to match
    the FAISS index instead, so the query vector comes from
    compute_retrieval_embedding(). See that function for the measurements.
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

    segments = [usable_audio[s: s + samples_per_segment] for s in starts]
    segments = [s for s in segments if len(s) == samples_per_segment]
    if not segments:
        raise ValueError("Track too short to extract a single analysis segment.")

    print(f"🔍 Analyzing {len(segments)} segments (Soft Voting)...")

    # One batched call rather than one per segment. Soft voting then averages the
    # softmax rows: a segment 99% sure of "Travis" outweighs three 35% "Drake" ones.
    preds = model.predict(segments_to_batch(segments, sr), verbose=0)
    genre_probs, artist_probs, mood_probs = (p.mean(axis=0) for p in preds)

    if joint:
        gi, ai, mi = joint_decode_indices(genre_probs, artist_probs, mood_probs, class_names)
    else:
        # Independent argmax per head — kept for A/B comparison against joint decoding.
        # Can emit triples the taxonomy does not contain; see joint_decode().
        gi = int(np.argmax(genre_probs))
        ai = int(np.argmax(artist_probs))
        mi = int(np.argmax(mood_probs))

    confidence = classification_confidence(
        genre_probs, artist_probs, mood_probs, gi, ai, mi, class_names)

    return (class_names['genre'][gi], class_names['artist'][ai],
            class_names['mood'][mi], confidence)


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


DEMUCS_TIMEOUT = 900   # 15 min; a stuck Demucs used to hang the worker indefinitely


def separate_audio(input_path, out_folder=None):
    """
    Uses Demucs to isolate the instrumental from the vocals.

    Runs without shell=True — the previous version interpolated a user-supplied path
    into a shell string, and checks neither the exit code nor a timeout, so a Demucs
    crash surfaced later as a confusing "file not found" and a hang blocked the worker
    forever.
    """
    print("✂️ AI Source Separation in progress (Demucs)...")
    out_folder = out_folder or TEMP_FOLDER

    cmd = [sys.executable, "-m", "demucs", "-n", "htdemucs", "--two-stems=vocals",
           "--mp3", "--mp3-bitrate", "320", input_path, "-o", out_folder]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=DEMUCS_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Demucs timed out after {DEMUCS_TIMEOUT}s on {input_path}")

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise RuntimeError(
            f"Demucs failed (exit {proc.returncode}): " + " | ".join(tail))

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    beat_path = os.path.join(out_folder, "htdemucs", base_name, "no_vocals.mp3")
    if not os.path.exists(beat_path):
        raise RuntimeError(f"Demucs reported success but produced no file at {beat_path}")
    return beat_path


def rebuild_vector_database_if_needed():
    """
    DISABLED — this built a SLICE-level index and would corrupt the current one.

    The active index is TRACK-level: one row per track (14,235 x 1536), with metadata
    keyed by a 'track' column. This function produced one row per slice (873,924) with
    a 'filename' column, so running it against the current FAISS_INDEX_PATH /
    VECTOR_META_PATH would overwrite a working index with an incompatible one and
    break analyze_beat's metadata lookup.

    It also cannot run usefully on Windows: TensorFlow has no native Windows GPU
    support past 2.10, so the forward pass measured ~10 img/s here versus ~90 img/s
    under WSL2 with CUDA (5.8 hours vs. 10.7 minutes for the same work).

    To rebuild, use the track-level builder under WSL2 — it samples
    RETRIEVAL_SEGMENTS slices per track, streams per-track running sums (87 MB rather
    than 1.4 GB), checkpoints every 60s, and throttles on GPU temperature.
    See TODO.md 1.1 for the procedure.
    """
    raise NotImplementedError(
        "Slice-level rebuild is disabled — it would overwrite the track-level index "
        "with an incompatible one. Use the WSL2 track-level builder instead (TODO.md 1.1)."
    )


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

    predicted_genre, predicted_artist, predicted_mood, confidence = \
        predict_with_majority_voting(model, y_full, sr, class_names)

    # Retrieval uses its own, index-aligned sampling — NOT the classification segments.
    mean_vector = compute_retrieval_embedding(model, y_full, sr)
    if mean_vector is None:
        raise ValueError("Track too short to build a retrieval embedding.")

    formatted_output = f"{predicted_artist} Type / {predicted_mood} / {predicted_genre}"

    # Mid-track spectrogram for the UI. The filename is per-request: a fixed path meant
    # two concurrent analyses overwrote each other's image, so users could be shown a
    # spectrogram belonging to somebody else's track.
    os.makedirs(SPEC_FOLDER, exist_ok=True)
    spec_path = os.path.join(SPEC_FOLDER, f"spec_{uuid.uuid4().hex[:12]}.png")
    mid_start = len(y_full) // 2
    mid_seg   = y_full[mid_start: mid_start + int(SEGMENT_DURATION * sr)]
    with open(spec_path, "wb") as f:
        f.write(render_spectrogram_png(mid_seg, sr))

    # Query FAISS (read-only, cached)
    index, db_meta = _get_faiss()

    # faiss.normalize_L2 mutates in place and requires float32 + C-contiguous.
    query_vector = np.ascontiguousarray(mean_vector, dtype=np.float32)
    faiss.normalize_L2(query_vector)

    # The index is track-level: one row per track, so every hit is already a distinct
    # track and no de-duplication is needed. (The old slice-level index returned many
    # slices of the same song, which is why it often yielded fewer than 5 results.)
    distances, indices = index.search(query_vector, 5)

    recommendations = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:                     # FAISS returns -1 when fewer than k neighbours exist
            continue
        row = db_meta.iloc[idx]
        recommendations.append({
            "id":    row['track'],
            "title": f"{row['artist_type']} / {row['mood']} / {row['genre']}",
            "score": float(dist * 100)
        })

    # confidence is returned separately so callers can decide whether the genre/artist/
    # mood triple is fit to show. BPM and key are NOT affected — they come from librosa,
    # not the model, and stay trustworthy regardless.
    return formatted_output, recommendations, int(bpm), str(key), confidence, spec_path


if __name__ == "__main__":
    song_input = input("🎵 Enter song query: ")
    try:
        mp3_path  = download_youtube_audio(song_input)
        beat_path = separate_audio(mp3_path)
        print("\n🧠 Analyzing...")
        result, recs, bpm, key, conf, _ = analyze_beat(beat_path)
        print(f"\n🎯 {result} | BPM: {bpm} | Key: {key}")
        for i, r in enumerate(recs):
            print(f"  {i+1}. {r['title']} — {r['score']:.1f}%")
    except Exception as e:
        print(f"❌ {e}")