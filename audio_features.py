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
import numpy as np
import librosa

# --- PANNs (primary) ---
PANNS_SR    = 32000       # PANNs is trained at 32 kHz. Not 22.05 kHz — resampling to
                          # anything else silently degrades the embedding.
DURATION    = 30          # seconds analysed from the start of the track
EMBED_DIM   = 2048        # raw Cnn14 output, before the PCA in the model bundle
PCA_DIM     = 128         # what the classifier and catalogue actually use
PANNS_CKPT  = os.environ.get(
    "PANNS_CKPT", os.path.expanduser("~/panns_data/Cnn14_mAP=0.431.pth"))

_panns_model = None
_panns_device = None


def _load_panns():
    """Loads Cnn14 once. Uses CUDA when available — PyTorch, unlike TensorFlow, has
    working Windows CUDA support, so this needs no WSL2."""
    global _panns_model, _panns_device
    if _panns_model is None:
        import torch
        from panns_inference.models import Cnn14
        if not os.path.exists(PANNS_CKPT):
            raise FileNotFoundError(
                f"PANNs weights not found at {PANNS_CKPT}.\n"
                "Download once (327 MB):\n"
                "  mkdir -p ~/panns_data && cd ~/panns_data\n"
                "  curl -L -o 'Cnn14_mAP=0.431.pth' "
                "'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'\n"
                "  curl -L -o class_labels_indices.csv "
                "'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
                "class_labels_indices.csv'\n"
                "Or point PANNS_CKPT at an existing copy.")
        m = Cnn14(sample_rate=PANNS_SR, window_size=1024, hop_size=320, mel_bins=64,
                  fmin=50, fmax=14000, classes_num=527)
        m.load_state_dict(torch.load(PANNS_CKPT, map_location="cpu",
                                     weights_only=False)["model"])
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
