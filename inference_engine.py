"""
BeatHunter AI - Inference Engine (PANNs embedding path)

Classification and retrieval run on PANNs — a CNN pretrained on AudioSet — projected to
128 dimensions. Measured on 2,847 held-out tracks (GroupShuffleSplit by track,
random_state=42; every figure in this project uses that same split):

    representation             artist top-1   lift    top-5    genre (joint)
    EfficientNetB3 (ImageNet)       ~0.7%     0.8x      —          ~1.3%
    classical MFCC 102-dim           9.45%   16.7x    24.13%      17.32%
    PANNs PCA-128                   12.96%   22.9x    32.03%      24.17%

The lesson across those three rows is domain, not capacity: EfficientNetB3 is the
largest of them and scored below chance because it transfers from photographs, while
PANNs transfers from 2M audio clips. Against the classical baseline the gain is
+3.51pt artist top-1 (4.2 SE), +7.90pt top-5 (6.7 SE), +6.85pt genre (6.4 SE).

Mood is NOT predicted. Three independent measurements agree it is not audio-linked
(Cohen d +0.023 on CNN embeddings, +0.094 on classical features, k-NN 1.4x chance) —
which follows, since it is assigned by hand per artist in config.py and never derived
from sound. Genre comes from joint decoding, which measures better than predicting it
directly (24.17% vs 22.87%) AND cannot emit an artist/genre pair the taxonomy lacks.

BPM and key are measured from the signal by librosa and were never affected by any of
the model problems.

The EfficientNetB3 training path is untouched in model_trainer.py; see TODO.md.
"""

import io
import os
import sys
import json
import uuid
import threading
import subprocess

import numpy as np
import pandas as pd
import joblib
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import scipy.stats as stats

import audio_features as AF

# --- CONFIGURATION ---
TEMP_FOLDER   = "temp_inference"
SPEC_FOLDER   = os.path.join(TEMP_FOLDER, "spectrograms")
MODEL_PATH    = "models/classic_model.joblib"
CATALOG_NPZ   = "models/classic_catalog.npz"
CATALOG_CSV   = "models/classic_catalog.csv"
TAXONOMY_PATH = "models/taxonomy.json"

# Audio is loaded once at PANNs' native 32 kHz and reused for everything.
# Resampling the embedding input to anything else silently degrades it;
# beat tracking and chroma work fine at 32 kHz.
SAMPLE_RATE      = AF.PANNS_SR
SEGMENT_DURATION = 3          # only for the UI spectrogram image

# Spectrogram parameters for the DISPLAY image only — nothing is classified from it.
N_MELS, HOP_LEN = 128, 512

N_RECOMMENDATIONS = 5

_cache_lock    = threading.Lock()
_cached_model  = None
_cached_catalog = None


def _get_model():
    """scaler + artist/genre classifiers + class names, loaded once."""
    global _cached_model
    if _cached_model is None:
        with _cache_lock:
            if _cached_model is None:
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(
                        f"{MODEL_PATH} not found. Build it with: python train_classic.py")
                _cached_model = joblib.load(MODEL_PATH)
                # The catalogue and classifier were built by ONE encoder. If a different
                # one is active now every vector is in the wrong space — and the output
                # would look plausible rather than fail, so this is a hard error.
                if _cached_model["kind"].startswith("panns"):
                    want, have = _cached_model.get("encoder_id"), AF.encoder_id()
                    if want and want != have:
                        raise ValueError(
                            f"model was built with encoder '{want}' but '{have}' is active. "
                            "The catalogue vectors live in a different space — rebuild with "
                            "train_classic.py, or restore the encoder it was built with.")
                    wh, hh = _cached_model.get("encoder_hash"), AF.encoder_hash()
                    if wh and hh and wh != hh:
                        raise ValueError(
                            f"encoder id matches ('{have}') but the weights hash differs "
                            f"({wh} vs {hh}) — the file changed after the model was built.")

                exp = AF.PCA_DIM if _cached_model["kind"].startswith("panns") else AF.CLASSIC_DIM
                if _cached_model["embed_dim"] != exp:
                    raise ValueError(
                        f"model was built at {_cached_model['embed_dim']} dims but "
                        f"audio_features expects {exp} for kind={_cached_model['kind']}. "
                        "Rebuild with train_classic.py.")
    return _cached_model


def _get_catalog():
    """L2-normalised track vectors + their labels, for similarity search."""
    global _cached_catalog
    if _cached_catalog is None:
        with _cache_lock:
            if _cached_catalog is None:
                if not os.path.exists(CATALOG_NPZ):
                    raise FileNotFoundError(
                        f"{CATALOG_NPZ} not found. Build it with: python train_classic.py")
                z = np.load(CATALOG_NPZ, allow_pickle=True)
                meta = pd.read_csv(CATALOG_CSV)
                if len(meta) != z["X"].shape[0]:
                    raise ValueError(
                        f"catalogue rows ({z['X'].shape[0]}) and metadata rows ({len(meta)}) "
                        "disagree — they must come from the same train_classic.py run.")
                _cached_catalog = (z["X"], z["tracks"], meta)
    return _cached_catalog


# ============================================================
# BPM / KEY — measured from the signal, independent of any model
# ============================================================
# Folding lives in audio_features so the BPM shown to the user and the tempo feeding the
# mood energy axis can never diverge. They already had: mood used the raw beat_track
# output and drill measured at the 44th tempo percentile, impossible for a ~140 BPM
# genre. One quantity, one implementation — see audio_features.fold_bpm.
BPM_RANGE  = AF.BPM_RANGE
BPM_TARGET = AF.BPM_TARGET
_fold_bpm  = AF.fold_bpm


def extract_bpm_and_key(y, sr):
    """Tempo and musical key. Neither comes from a model."""
    print("🥁 Analyzing BPM and Musical Key...")
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120)
    bpm = _fold_bpm(float(np.atleast_1d(tempo)[0]))

    # Harmonic component only: chroma over a full mix is dominated by drum transients,
    # which smear energy across every pitch class — on beats that is most of the signal.
    chromagram  = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(y), sr=sr)
    mean_chroma = np.mean(chromagram, axis=1)

    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    maj  = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minr = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    maj_c = [stats.pearsonr(mean_chroma, np.roll(maj,  i))[0] for i in range(12)]
    min_c = [stats.pearsonr(mean_chroma, np.roll(minr, i))[0] for i in range(12)]
    bi, mi = int(np.argmax(maj_c)), int(np.argmax(min_c))
    key = f"{keys[bi]} Major" if maj_c[bi] > min_c[mi] else f"{keys[mi]} Minor"
    return round(bpm), key


# ============================================================
# CLASSIFICATION
# ============================================================
# The bar for showing a single artist as a headline answer is CALIBRATED per model and
# stored in the bundle by train_classic.py — never hardcoded here. Copying a previous
# model's number over would be cargo-culting a measurement: the old classical model's
# 100x bar bought 26.1% precision on its own confidence distribution, while on the PANNs
# model 100x buys 25.9% and 150x is needed for 38.7%.
#
# The current bundle's calibration table (PANNs PCA-128, 2,847 held-out tracks):
#
#     bar     shown   top-1 correct
#   >= 25x    90.8%       13.7%
#   >= 50x    59.4%       17.3%
#   >= 75x    36.6%       20.8%
#   >=100x    21.7%       25.9%
#   >=150x     6.8%       38.7%   <- chosen: first bar clearing 35% precision
#
# Coverage is deliberately low because top-5 carries the useful signal (32.03% against
# 12.96%). Callers should lead with artist_top5 and treat the single label as a bonus
# that usually is not there.
DEFAULT_CONFIDENCE_MIN_RATIO = 150.0


def _confidence_threshold(m):
    return float(m.get("confidence_threshold") or DEFAULT_CONFIDENCE_MIN_RATIO)


def _project(m, y, sr):
    """
    Waveform -> the exact vector the classifier and catalogue were built on.

    Order matters and must match train_classic.py: embed -> PCA -> scale. Both callers
    go through here so the query and the catalogue cannot drift apart — the mistake that
    cost this project months on the spectrogram path.
    """
    if m["kind"].startswith("panns"):
        raw = AF.extract_embedding(y, sr)
    else:
        raw = AF.extract_classical_features(y, sr)
    if raw is None:
        return None
    X = raw.reshape(1, -1)
    if m.get("pca") is not None:
        X = m["pca"].transform(X)
    return m["scaler"].transform(X)


def classify(y, sr):
    """
    Returns (artist, genre, confidence).

    Joint decoding: score(a) = log P(artist=a) + log P(genre=G[a]). Genre is read off
    the taxonomy rather than argmaxed independently, so the pair is always one the data
    contains. Measured better than predicting genre directly (17.32% vs 16.37%).
    """
    m = _get_model()
    feat = _project(m, y, sr)
    if feat is None:
        raise ValueError("Track too short to analyse (under 5 seconds).")
    Xs = feat
    Pa = m["clf_artist"].predict_proba(Xs)[0]
    Pg = m["clf_genre"].predict_proba(Xs)[0]

    artists, genres, g_of_a = m["artists"], m["genres"], m["genre_of_artist"]
    a_cls, g_cls = m["clf_artist"].classes_, m["clf_genre"].classes_

    eps = 1e-12
    logPg = np.full(len(genres), -30.0)
    logPg[g_cls] = np.log(Pg + eps)
    scores = np.log(Pa + eps) + logPg[g_of_a[a_cls]]

    best = int(np.argmax(scores))
    ai = int(a_cls[best])
    gi = int(g_of_a[ai])

    p_artist = float(Pa[best])
    chance_a = 1.0 / len(artists)
    confidence = {
        "artist": {"label": str(artists[ai]), "probability": p_artist,
                   "ratio_vs_chance": p_artist / chance_a},
        "genre":  {"label": str(genres[gi]),
                   "probability": float(Pg[list(g_cls).index(gi)]) if gi in g_cls else 0.0,
                   "ratio_vs_chance": (float(Pg[list(g_cls).index(gi)]) * len(genres)
                                       if gi in g_cls else 0.0)},
        "min_ratio_required": _confidence_threshold(m),
    }
    confidence["reliable"] = p_artist / chance_a >= _confidence_threshold(m)

    # Top-5 is the honest view for this product: top-1 is 9.45% but top-5 is 24.13%.
    top5 = np.argsort(-Pa)[:5]
    confidence["artist_top5"] = [
        {"label": str(artists[int(a_cls[i])]), "probability": float(Pa[i])} for i in top5]

    return str(artists[ai]), str(genres[gi]), confidence


def find_similar(y, sr, k=N_RECOMMENDATIONS):
    """Cosine nearest neighbours over the track catalogue. ~0.04 ms for 14,235 tracks."""
    m = _get_model()
    X, tracks, meta = _get_catalog()

    feat = _project(m, y, sr)
    if feat is None:
        return []
    q = feat[0]
    q = q / (np.linalg.norm(q) + 1e-9)

    sims = X @ q
    order = np.argsort(-sims)[:k]
    return [{
        "id":    str(tracks[i]),
        "title": f"{meta.iloc[i]['artist_type']} / {meta.iloc[i]['genre']}",
        "score": float(sims[i] * 100),
    } for i in order]


# ============================================================
# DISPLAY SPECTROGRAM (UI only — nothing is classified from it)
# ============================================================
def render_spectrogram_png(y_segment, sr):
    """Renders a mel-spectrogram to PNG bytes in RAM. Uses a standalone Figure rather
    than pyplot, whose global state is not safe under run_in_executor."""
    mels_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=N_MELS), ref=np.max)
    fig = Figure(figsize=(3, 3))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    librosa.display.specshow(mels_db, sr=sr, hop_length=HOP_LEN, ax=ax)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    return buf.getvalue()


# ============================================================
# AUDIO ACQUISITION
# ============================================================
def download_youtube_audio(song_name, out_folder=None):
    import yt_dlp
    out_folder = out_folder or TEMP_FOLDER
    os.makedirs(out_folder, exist_ok=True)
    print(f"🔍 Searching YouTube for: {song_name}")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(out_folder, 'downloaded_song.%(ext)s'),
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        'quiet': True, 'noplaylist': True, 'default_search': 'ytsearch1'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([song_name])
    return os.path.join(out_folder, "downloaded_song.mp3")


DEMUCS_TIMEOUT = 900


def separate_audio(input_path, out_folder=None):
    """
    Isolates the instrumental with Demucs.

    No shell=True, exit code checked, timeout applied: the previous version
    interpolated a user-supplied path into a shell string and checked neither, so a
    crash surfaced later as a confusing "file not found" and a hang blocked the worker.
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
        raise RuntimeError(f"Demucs failed (exit {proc.returncode}): " + " | ".join(tail))

    base = os.path.splitext(os.path.basename(input_path))[0]
    beat_path = os.path.join(out_folder, "htdemucs", base, "no_vocals.mp3")
    if not os.path.exists(beat_path):
        raise RuntimeError(f"Demucs reported success but produced no file at {beat_path}")
    return beat_path


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def analyze_beat(audio_path):
    """
    Returns (formatted, recommendations, bpm, key, confidence, spec_path, mood).

    `formatted` is "<artist> Type / <genre>". Mood is NOT part of it and is NOT a
    classifier output: it is measured from the audio on three axes and reported
    separately, because unlike artist/genre it is not a prediction that can be wrong
    about a label — it is a description of the signal. `mood` is None when
    models/mood_calibration.json is absent.
    """
    os.makedirs(TEMP_FOLDER, exist_ok=True)

    print("🎵 Loading audio...")
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=AF.DURATION)

    bpm, key = extract_bpm_and_key(y, sr)

    print("🔍 Classifying...")
    artist, genre, confidence = classify(y, sr)
    recommendations = find_similar(y, sr)

    # Reuses the waveform already in memory (resampled to 22.05 kHz inside) rather than
    # decoding the file again. Costs ~5s, dominated by HPSS.
    print("🎨 Measuring mood...")
    mood = AF.mood_from_audio(y, sr)

    formatted = f"{artist} Type / {genre}"

    # Per-request filename: a fixed path meant two concurrent analyses overwrote each
    # other's image, so a user could be shown someone else's spectrogram.
    os.makedirs(SPEC_FOLDER, exist_ok=True)
    spec_path = os.path.join(SPEC_FOLDER, f"spec_{uuid.uuid4().hex[:12]}.png")
    mid = len(y) // 2
    with open(spec_path, "wb") as f:
        f.write(render_spectrogram_png(y[mid: mid + int(SEGMENT_DURATION * sr)], sr))

    return formatted, recommendations, int(bpm), str(key), confidence, spec_path, mood


if __name__ == "__main__":
    query = input("🎵 Enter song query: ")
    try:
        mp3 = download_youtube_audio(query)
        beat = separate_audio(mp3)
        print("\n🧠 Analyzing...")
        result, recs, bpm, key, conf, _, mood = analyze_beat(beat)
        print(f"\n🎯 {result} | BPM: {bpm} | Key: {key}")
        print(f"   confidence: {conf['artist']['ratio_vs_chance']:.1f}x chance "
              f"(reliable={conf['reliable']})")
        if mood:
            print("   mood: " + ", ".join(
                f"{v['label']} ({v['score']:.2f})" for v in mood.values()))
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r['title']} — {r['score']:.1f}%")
    except Exception as e:
        print(f"❌ {e}")
