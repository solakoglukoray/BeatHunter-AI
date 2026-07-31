"""
BeatHunter AI — audio representation.

SINGLE SOURCE OF TRUTH. The catalogue builder and the query path both call
extract_embedding(); neither may keep its own copy.

That rule is the direct lesson of this project's worst bug: training and inference each
had their own spectrogram code, drifted apart, and fed the model an input distribution
it had never seen. Nobody noticed for months.

WHAT WON, AND BY HOW MUCH
─────────────────────────
Three representations, all measured on the same 2,847 held-out tracks
(GroupShuffleSplit by track, random_state=42):

                            artist top-1   lift    top-5    genre (joint)
  EfficientNetB3 (ImageNet)     ~0.7%      0.8x      —          ~1.3%
  classical MFCC 102-dim         9.45%    16.7x    24.13%      17.32%
  PANNs PCA-128                 12.96%    22.9x    32.03%      24.17%

Against the classical baseline: +3.51pt artist top-1 (4.2 SE), +7.90pt top-5 (6.7 SE),
+6.85pt genre (6.4 SE). Solid, not marginal.

The pattern across all three is domain, not capacity. EfficientNetB3 is the largest
model of the three and scored below chance, because it transfers from photographs.
PANNs is pretrained on AudioSet — 2M audio clips — and transfers.

WHY 128 DIMENSIONS AND NOT 2048
───────────────────────────────
PANNs emits 2048. Projected to 128 it beats its own full-width version on every metric
(2048-dim measured 11.52% / 20.3x / 28.13% / 20.02%) and is sixteen times smaller.
2048 dimensions against ~11k training tracks lets the classifier fit noise; the
projection regularises it. It also beat concatenating PANNs with the classical features.

The PCA is fitted on training tracks only and stored in the model bundle alongside the
scaler, so inference applies exactly the projection training used.
"""

import os
import json
import numpy as np
import librosa

# --- PANNs (primary) ---
PANNS_SR    = 32000       # PANNs is trained at 32 kHz. Not 22.05 kHz — resampling to
                          # anything else silently degrades the embedding.
DURATION    = 30          # seconds analysed from the start of the track
EMBED_DIM   = 2048        # raw Cnn14 output, before the PCA in the model bundle
PCA_DIM     = 128         # what the classifier and catalogue actually use
# Two possible encoders, in priority order:
#
#   1. models/panns_ft_encoder.pt — our Cnn14 fine-tuned on this dataset. With the SAME
#      PCA-128 + LogReg + joint decoding on top it measures artist top-1 15.35% /
#      top-5 37.72% / genre 28.21%, against the stock encoder's 12.96% / 32.03% / 24.17%.
#      Fine-tuning closed 94% of the top-1 gap and 106% of the genre gap to a full
#      end-to-end model, so the gain lives in the REPRESENTATION rather than in a trained
#      head — which is exactly why the classifier architecture below is unchanged.
#
#   2. ~/panns_data/Cnn14_mAP=0.431.pth — stock AudioSet weights. Fallback, and the
#      documented baseline.
#
# ⚠️ The catalogue and the classifier are built against ONE of these, and their vectors
# live in different spaces. Mixing them produces confident nonsense rather than an error,
# so train_classic.py records encoder_id() and a hash of the weights in the model bundle
# and inference_engine verifies both on load. Do not remove that check: this project
# already lost months to precisely this failure on the spectrogram path.
FT_CKPT     = os.environ.get("PANNS_FT_CKPT", "models/panns_ft_encoder.pt")
BASE_CKPT   = os.environ.get(
    "PANNS_CKPT", os.path.expanduser("~/panns_data/Cnn14_mAP=0.431.pth"))

_panns_model = None
_panns_device = None
_encoder_id = None
_encoder_hash = None


def _hash_file(path, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def encoder_id():
    """Short identifier for whichever encoder is active. Recorded in the model bundle."""
    _load_panns()
    return _encoder_id


def encoder_hash():
    _load_panns()
    return _encoder_hash


def feature_cache_path():
    """Cache filename is tied to the encoder, so switching encoders cannot silently
    reuse the other one's embeddings."""
    return ("models/panns_ft_features.npz" if encoder_id().startswith("panns-ft")
            else "models/panns_features.npz")


def _load_panns():
    """Loads Cnn14 once, preferring the fine-tuned encoder. Uses CUDA when available —
    PyTorch, unlike TensorFlow, has working Windows CUDA support, so no WSL2 needed."""
    global _panns_model, _panns_device, _encoder_id, _encoder_hash
    if _panns_model is None:
        import torch
        from panns_inference.models import Cnn14

        if os.path.exists(FT_CKPT):
            ck = torch.load(FT_CKPT, map_location="cpu", weights_only=False)
            state = ck["trunk"]
            _encoder_id = f"panns-ft-e{ck.get('epoch', '?')}"
            _encoder_hash = _hash_file(FT_CKPT)
        elif os.path.exists(BASE_CKPT):
            state = torch.load(BASE_CKPT, map_location="cpu",
                               weights_only=False)["model"]
            _encoder_id = "panns-audioset"
            _encoder_hash = _hash_file(BASE_CKPT)
            print("ℹ️  fine-tuned encoder yok, stock AudioSet agirliklari kullaniliyor "
                  "(top-1 %12.96 vs fine-tuned %15.35)")
        else:
            raise FileNotFoundError(
                f"No encoder found.\n\n"
                f"Preferred: {FT_CKPT} — our fine-tuned encoder (327 MB).\n"
                "  A build artefact, not a download. See README 'PANNs weights'.\n"
                "  Regenerate: python panns_finetune.py  (~2.8 h GPU)\n"
                "              python panns_ft_frozen_experiment.py --extract-encoder\n\n"
                f"Fallback: {BASE_CKPT} — stock AudioSet (327 MB):\n"
                "  mkdir -p ~/panns_data && cd ~/panns_data\n"
                "  curl -L -o 'Cnn14_mAP=0.431.pth' "
                "'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'\n"
                "  curl -L -o class_labels_indices.csv "
                "'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
                "class_labels_indices.csv'")

        m = Cnn14(sample_rate=PANNS_SR, window_size=1024, hop_size=320, mel_bins=64,
                  fmin=50, fmax=14000, classes_num=527)
        m.load_state_dict(state)
        _panns_device = "cuda" if torch.cuda.is_available() else "cpu"
        _panns_model = m.eval().to(_panns_device)
    return _panns_model, _panns_device


def extract_embedding(y, sr=PANNS_SR):
    """
    Raw 2048-dim PANNs clip embedding, or None if the audio is too short (<5s).

    The model bundle's PCA is applied downstream, not here, so the cached embeddings
    stay reusable if the projection is ever refitted.
    """
    if y is None or len(y) < sr * 5:
        return None
    import torch
    model, device = _load_panns()
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(y, dtype=np.float32)[None, :]).to(device)
        emb = model(x)["embedding"].cpu().numpy()[0]
    if emb.shape[0] != EMBED_DIM:
        raise ValueError(f"embedding dim {emb.shape[0]} != EMBED_DIM {EMBED_DIM}")
    return emb.astype(np.float32)


def extract_embeddings_batch(waveforms, device=None):
    """Batched version for catalogue building. Zero-pads to the longest clip."""
    import torch
    model, dev = _load_panns()
    n = max(len(w) for w in waveforms)
    x = np.zeros((len(waveforms), n), dtype=np.float32)
    for i, w in enumerate(waveforms):
        x[i, :len(w)] = w
    with torch.no_grad():
        return model(torch.from_numpy(x).to(dev))["embedding"].cpu().numpy().astype(np.float32)


def load_audio(path, sr=PANNS_SR, duration=DURATION):
    try:
        y, _ = librosa.load(path, sr=sr, duration=duration)
    except Exception:
        return None
    return y


def extract_from_file(path):
    y = load_audio(path)
    return None if y is None else extract_embedding(y)


# ============================================================
# CLASSICAL FEATURES — superseded, kept deliberately
# ============================================================
# This is the representation PANNs replaced (16.7x vs 22.9x on artist). It stays for two
# reasons: it needs neither PyTorch nor 327 MB of weights, so it is the fallback if
# PANNs cannot be loaded; and it is the documented baseline any future representation
# has to beat. Do not delete it to "clean up" — that would throw away the only cheap
# reference point this project has.
CLASSIC_SR  = 22050
CLASSIC_DIM = 102


def extract_classical_features(y, sr=CLASSIC_SR):
    """MFCC + delta, spectral contrast, centroid/rolloff/ZCR/RMS. 102 dims, ~345 ms/track
    on one CPU core."""
    if y is None or len(y) < sr * 5:
        return None
    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    dmfcc    = librosa.feature.delta(mfcc)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    cent     = librosa.feature.spectral_centroid(y=y, sr=sr)
    roll     = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr      = librosa.feature.zero_crossing_rate(y)
    rms      = librosa.feature.rms(y=y)
    feat = np.concatenate([
        mfcc.mean(1), mfcc.std(1), dmfcc.mean(1), dmfcc.std(1),
        contrast.mean(1), contrast.std(1),
        [cent.mean(), cent.std(), roll.mean(), roll.std(),
         zcr.mean(), zcr.std(), rms.mean(), rms.std()],
    ]).astype(np.float32)
    if feat.shape[0] != CLASSIC_DIM:
        raise ValueError(f"classical dim {feat.shape[0]} != {CLASSIC_DIM}")
    return feat


# ============================================================
# MOOD / ENERGY — derived from the audio, on three axes
# ============================================================
# This replaces the old `mood` label, which was assigned by hand per artist in config.py
# and turned out to have no relationship to the sound: Cohen d +0.023 on CNN embeddings,
# +0.094 on classical features, k-NN 1.4x chance. Three independent measurements, same
# answer. It was dropped as a prediction target entirely.
#
# What makes this different is not that it uses librosa — it is that NOTHING here is a
# hand-picked number. Every threshold comes from the corpus distribution (see
# calibrate_mood.py), and a label means "this track sits in the upper/lower third of
# 14,235 tracks on this axis". That is a claim about measurement rather than taste, and
# it can be checked.
#
# WHAT THE FIRST VERSION GOT WRONG (measured on 2,000 tracks, then fixed)
# ----------------------------------------------------------------------
# The first signal set failed its own ear test, and the failures are worth recording
# because each one is a trap the obvious design walks into:
#
#   perc_ratio (HPSS percussive share) had NO discriminative power — every genre landed
#   between 0.47 and 0.54 (Lofi 0.54, Phonk 0.48, Crunk 0.47, Drill 0.47). It was also
#   the most expensive signal in the set (3.4s of 5s per track). Dropped.
#
#   Broadband hiss read as aggression. The 3rd "most aggressive" track in the corpus was
#   a LO-FI beat: zcr 0.98, flatness 0.98, perc_ratio 1.00 — top of the corpus on all
#   three aggression signals — while sitting at rms 0.10 and onset density 0.30. Steady
#   vinyl/tape noise is broadband, crosses zero often, and HPSS calls it percussive. The
#   fix is a signal that separates STEADY noise from SPIKY hits: onset_contrast.
#
#   rms_mean measured mastering, not music. Its range across the corpus is 7x, which is
#   mostly how loud each upload was mastered. Now measured after peak normalisation.
#
#   tempo was unfolded. Drill averaged the 44th tempo percentile, impossible for a
#   ~140 BPM genre — beat_track was locking to the half-time feel. inference_engine
#   already had _fold_bpm for exactly this; mood was not using it, so the app held two
#   different tempo numbers for one track. fold_bpm now lives here and both call it.
#
#   centroid and rolloff correlated at r=0.97 — the same measurement twice, which
#   silently gave it 2/3 of the brightness axis. Only one of the pair is kept.
#
#   HPSS was not needed for minor_lean at all: harmonic-component chroma vs raw-mix
#   chroma correlate at r=0.981 (mean |diff| 0.022 against a 0.159 spread). Dropping it
#   cut extraction from ~5.0s to ~1.6s per track.
#
# Axis membership is therefore NOT chosen by intuition — calibrate_mood.py reports each
# signal's genre discrimination and each axis pair's correlation, and the sets below are
# what survived. If two axes correlate strongly they are one axis wearing two names, and
# the report says so.

MOOD_SR       = 22050     # mood features do not need PANNs' 32 kHz; cheaper here
MOOD_DURATION = 30

MOOD_AXES = {
    "energy":     ("Chill", "Moderate", "Energetic"),
    "brightness": ("Dark", "Balanced", "Bright"),
    "aggression": ("Mellow", "Moderate", "Aggressive"),
}

_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]

# Tempo folding lives HERE, not in inference_engine, because both the BPM shown to the
# user and the mood energy axis need it and they must never disagree. They already did:
# mood used raw beat_track and drill averaged the 44th tempo percentile, which is
# impossible for a ~140 BPM genre — beat_track had locked to the half-time feel. That is
# Trap 1's pattern (one quantity computed two ways, drifting apart unnoticed), so there
# is now exactly one implementation.
BPM_RANGE  = (60, 200)
# Deliberately wide at the top. Folding 70 -> 140 fixes the common half-time miss, but a
# tight upper bound would also fold genuinely fast tempos: drill and jersey sit in the
# 160-175 range, and halving those would be a new error introduced to fix an old one.
BPM_TARGET = (78, 180)


def fold_bpm(bpm):
    """Folds a tempo into the plausible range by repeated halving/doubling."""
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


# Bumping this invalidates cached signals. models/mood_signals.npz stores it, and
# calibrate_mood.py refuses a cache whose version does not match — the whole project's
# oldest bug was code and generated data drifting apart silently (Trap 1), and a feature
# cache is exactly that shape of hazard.
MOOD_SIGNAL_VERSION = 2

MOOD_SIGNAL_NAMES = [
    "rms_norm", "onset_density", "onset_contrast", "tempo", "dynamic_range",
    "centroid_mean", "hf_ratio", "low_ratio", "minor_lean",
    "zcr_mean", "flatness_mean", "flux_mean",
]


def extract_mood_signals(y, sr=MOOD_SR):
    """
    Raw per-track measurements feeding the three axes. Returns a dict, or None if the
    audio is too short. No thresholds applied here — that is calibration's job.

    Peak normalisation up front is load-bearing: without it rms tracks how loud the file
    was mastered (7x spread across this corpus) rather than how energetic the music is.
    """
    if y is None or len(y) < sr * 5:
        return None

    import scipy.stats as _stats

    peak = float(np.max(np.abs(y)))
    if peak < 1e-6:
        return None
    y = y / peak

    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    power = S ** 2
    total = power.sum() + 1e-12

    rms      = librosa.feature.rms(S=S)[0]
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    zcr      = librosa.feature.zero_crossing_rate(y)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]

    onset_env = librosa.onset.onset_strength(S=librosa.power_to_db(power), sr=sr)
    onsets    = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration  = len(y) / sr
    tempo, _  = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=120)

    # THE SIGNAL THAT SEPARATES HITS FROM HISS. Steady broadband noise (vinyl crackle,
    # tape hiss — the whole lo-fi aesthetic) is loud in zcr and flatness but perfectly
    # FLAT in time. Real aggression is spiky: hard transients over relative silence.
    # Dividing by the mean makes it a shape measure, immune to overall level.
    onset_contrast = float(np.std(onset_env) / (np.mean(onset_env) + 1e-9))

    # Crest-like: how far the loud moments sit above the typical one. Compressed,
    # wall-of-sound masters approach 1; music that breathes goes higher.
    dynamic_range = float(np.percentile(rms, 95) / (np.mean(rms) + 1e-9))

    # Explicit bands beat centroid+rolloff, which correlated at r=0.97 — the same
    # measurement twice, quietly taking 2/3 of the brightness axis.
    hf_ratio  = float(power[freqs >= 4000].sum() / total)
    low_ratio = float(power[freqs <= 150].sum() / total)

    # Spectral flux: how much the spectrum churns frame to frame. Steady noise barely
    # moves; busy, percussive music does.
    flux_mean = float(np.mean(np.maximum(0.0, np.diff(S, axis=1)).sum(axis=0)) / (S.shape[0]))

    # Minor/major leaning (Krumhansl-Schmuckler). Measured on the raw mix: harmonic-only
    # chroma agrees at r=0.981 (mean |diff| 0.022 vs a 0.159 spread) and costs 3.4s/track.
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)
    maj = max(_stats.pearsonr(mean_chroma, np.roll(_MAJOR_PROFILE, i))[0] for i in range(12))
    minr = max(_stats.pearsonr(mean_chroma, np.roll(_MINOR_PROFILE, i))[0] for i in range(12))

    return {
        "rms_norm":       float(np.mean(rms)),
        "onset_density":  float(len(onsets) / duration),
        "onset_contrast": onset_contrast,
        "tempo":          fold_bpm(float(np.atleast_1d(tempo)[0])),
        "dynamic_range":  dynamic_range,
        "centroid_mean":  float(np.mean(centroid)),
        "hf_ratio":       hf_ratio,
        "low_ratio":      low_ratio,
        "minor_lean":     float(minr - maj),      # >0 means minor fits better
        "zcr_mean":       float(np.mean(zcr)),
        "flatness_mean":  float(np.mean(flatness)),
        "flux_mean":      flux_mean,
    }


def extract_mood_from_file(path):
    y = load_audio(path, sr=MOOD_SR, duration=MOOD_DURATION)
    return None if y is None else extract_mood_signals(y, MOOD_SR)


# Which signals feed which axis, and in which direction. Weights are equal on purpose:
# no evidence yet justifies preferring one signal over another, and inventing weights
# would be exactly the arbitrariness this replaces.
#
# Membership was searched, not guessed — but only WITHIN signals that legitimately belong
# to each axis. An unconstrained search happily puts spectral centroid on "energy"
# because it separates the genres well, and then the axis stops meaning what its name
# says. An axis has to stay interpretable to be worth showing anyone.
#
# Each axis was validated against its OWN known-genre contrast. Using one
# "mellow vs aggressive" split for all three was a measurement error on the way here:
# drill is aggressive but SPARSE, so it belongs high on aggression and low on energy, and
# a single split cannot express that.
#
# Rejected by measurement, recorded so nobody re-adds them:
#   tempo           +0.01 on the energy contrast. Genuinely useless here — type beats of
#                   every genre cluster around 120-150 BPM, so tempo carries no energy
#                   information in THIS corpus. Still extracted, still folded, still
#                   shown as BPM; just not evidence of energy.
#   onset_contrast  Backwards. I predicted aggressive music would be spikier; it is the
#                   opposite (-0.08 energy, -0.14 aggression). Aggressive beats are dense
#                   and compressed, and it is sparse, breathing music that has contrast.
#   perc_ratio      No separation anywhere (0.47-0.54 across every genre), and the most
#                   expensive signal in the set. Removed entirely.
#   dynamic_range   -0.00 / -0.07. Measures mastering style, not mood.
#   flatness+zcr    The intuitive "noisiness" reading of aggression. It passed the ear
#                   test (Nu Metal / Crunk / Rage on top) but tied energy and aggression
#                   together at r=0.681 — two names for one axis. Rejected on
#                   independence, having been rejected once already for hiss.
#
# aggression = low_ratio + hf_ratio reads as a MIX SHAPE: energy piled at both spectral
# extremes, heavy 808s under hard highs with the mids scooped out. That is a fair physical
# description of how aggressive rap is produced, and it defeats the hiss confound by
# construction rather than by luck — tape hiss is broadband mid/high with no low end, so
# it can only ever score half of this axis.
#
# Measured with these sets (2,000 tracks):
#   aggression  top    Georgia Trap .64, Rage .63, Trap .62, Crunk .60, Chicago Drill .59
#               bottom Lofi .31, Chillhop .32, Trip Hop .35, Neo Soul .40
#   axis corr   energy~brightness .26, energy~aggression .29, brightness~aggression .40
#   the hiss track that was ranked 3rd most aggressive of 2,000 now sits at 346th
#
# Note drill: LOW on energy (UK .34, Brooklyn .36) and HIGH on aggression (Chicago .59).
# That is not a contradiction, it is the point — drill is sparse and hard at the same
# time, and one axis could not have said both.
AXIS_SIGNALS = {
    "energy":     {"onset_density": +1, "rms_norm": +1},
    "brightness": {"centroid_mean": +1, "minor_lean": -1},
    "aggression": {"low_ratio": +1, "hf_ratio": +1},
}


def axis_scores(signals, calibration):
    """
    Turns raw signals into one score per axis, in corpus percentile terms.

    Each signal is mapped to its percentile within the corpus, then the axis is the mean
    of its signals' percentiles (flipping sign where the signal runs the other way). So
    0.5 is "average for this corpus" and 0.9 is "higher than 90% of it" — the score is
    always a statement about the corpus, never an absolute.
    """
    out = {}
    for axis, members in AXIS_SIGNALS.items():
        pcts = []
        for name, direction in members.items():
            edges = calibration["percentiles"].get(name)
            if edges is None or name not in signals:
                continue
            p = float(np.searchsorted(edges, signals[name]) / len(edges))
            pcts.append(p if direction > 0 else 1.0 - p)
        out[axis] = float(np.mean(pcts)) if pcts else 0.5
    return out


def classify_mood(signals, calibration):
    """
    Labels each axis using corpus terciles, and reports the score behind each label.

    Terciles rather than a midpoint split: with a midpoint, tracks sitting near the middle
    flip label on noise. The middle third gets an explicit "Moderate"/"Balanced" label
    instead of being forced to one side.

    ⚠️ The cut points come from the distribution of AXIS SCORES, not from 1/3 and 2/3 of
    the 0-1 range. Those are not the same thing and using the latter was a bug: an axis
    score is the mean of three percentiles, and averaging percentiles concentrates the
    result around 0.5 (central limit), so a fixed 1/3-2/3 cut labels most of the corpus
    "Moderate" and reserves the outer labels for freaks. Brightness is worse still — its
    minor_lean term opposes the other two, so it cannot reach the extremes at all
    (measured: all-signals-high still only scores 0.65). Per-axis empirical cuts fix both.

    The 1/3, 2/3 fallback exists only for a calibration file written before axis_cuts
    existed; it produces lopsided labels, so re-run calibrate_mood.py if you hit it.
    """
    if signals is None or calibration is None:
        return None
    scores = axis_scores(signals, calibration)
    cuts = calibration.get("axis_cuts") or {}
    result = {}
    for axis, score in scores.items():
        lo, hi = cuts.get(axis, (1 / 3, 2 / 3))
        low_name, mid_name, high_name = MOOD_AXES[axis]
        label = low_name if score < lo else (high_name if score >= hi else mid_name)
        result[axis] = {"label": label, "score": round(score, 3)}
    return result


MOOD_CALIB_PATH = "models/mood_calibration.json"
_mood_calib = None
_mood_calib_tried = False


def load_mood_calibration():
    """
    The corpus percentiles, or None if they were never computed.

    None is a supported state, not an error: mood is a display extra, so a missing
    calibration file must degrade to "no mood shown" rather than taking down analysis.
    Produce it with `python calibrate_mood.py`.
    """
    global _mood_calib, _mood_calib_tried
    if _mood_calib_tried:
        return _mood_calib
    _mood_calib_tried = True
    try:
        with open(MOOD_CALIB_PATH, encoding="utf-8") as f:
            _mood_calib = json.load(f)
        print(f"✅ mood calibration ({_mood_calib.get('n_tracks', '?')} tracks)")
    except FileNotFoundError:
        print("ℹ️  no mood calibration — run calibrate_mood.py to enable mood output")
    except Exception as e:
        print(f"⚠️  mood calibration unreadable ({e}) — mood disabled")
    return _mood_calib


def mood_from_audio(y, sr):
    """
    Mood for an already-loaded waveform, resampling to MOOD_SR if needed.

    Used by inference so a request does not decode the file a second time. Returns None
    when calibration is missing or the clip is too short to measure.
    """
    calib = load_mood_calibration()
    if calib is None or y is None:
        return None
    if sr != MOOD_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=MOOD_SR)
    return classify_mood(extract_mood_signals(y, MOOD_SR), calib)
