"""
Measures how often each KIND of label is right, so they can be compared to each other.

WHY NOT "RATIO VS CHANCE" AS THE COMMON UNIT
--------------------------------------------
It is the obvious choice and it is wrong, because the ceiling depends on how many
classes the label has:

    artist      194 classes -> a ratio can reach 194x
    genre        65 classes ->                    65x
    mood axis     3 labels  ->                     3x

Ranking by ratio would mean mood can never be chosen, no matter how certain it is: a
track measured at the 99th aggression percentile scores at most 3x, and loses to an
artist guess sitting at 4x chance that is wrong 85% of the time. The unit has to be
something a user actually cares about, and that is:

    P(this label is correct)

which is directly comparable across all three, and measurable for all three.

HOW EACH ONE IS MEASURED
------------------------
artist, genre   Reliability curves on the canonical held-out split (2,847 tracks,
                GroupShuffleSplit by track, random_state=42). For each confidence bin,
                how often was the top label actually right. This is precision, measured,
                not the classifier's own opinion of itself — those are overconfident.

mood            Mood has no ground truth to be wrong about: it is a measurement of the
                audio, and for the 30 seconds analysed the percentile IS the answer. The
                thing that can be wrong is GENERALISATION — whether the label still holds
                for the rest of the track. So reliability is measured as label stability
                across two disjoint windows (0-30s vs 30-60s) as a function of how far
                the score sits from the nearest tercile cut.

    python calibrate_label_choice.py
"""

import json
import os
import warnings

warnings.filterwarnings("ignore")

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import joblib
import librosa
import numpy as np
import pandas as pd
from multiprocessing import Pool
import audio_features as AF
import corpus

MODEL_PATH  = "models/classic_model.joblib"
TRACK_META  = corpus.TRACK_META
OUT_PATH    = "models/label_reliability.json"

# Bin edges in ratio-vs-chance. Bins, not cumulative thresholds: to estimate P(correct)
# for one track we want the precision of tracks LIKE it, not of everything above a bar.
ARTIST_BINS = [0, 25, 50, 75, 100, 125, 150, 175, 1e9]
GENRE_BINS  = [0, 2, 4, 6, 8, 12, 16, 24, 1e9]
# Distance from the nearest tercile cut, in axis-score units.
MOOD_BINS   = [0.0, 0.02, 0.05, 0.10, 0.15, 0.25, 1.0]
MOOD_SAMPLE = 700


def held_out(tracks):
    """Canonical split — see corpus.held_out_mask."""
    return set(np.asarray(tracks)[corpus.held_out_mask(tracks)])


def curve(ratios, correct, bins, label, fmt="{:.0f}"):
    """P(correct) per confidence bin, with the counts behind each number."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (ratios >= lo) & (ratios < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        p = float(correct[sel].mean())
        se = float(np.sqrt(max(p * (1 - p), 1e-9) / n))
        out.append({"lo": float(lo), "hi": float(min(hi, 1e9)), "n": n,
                    "precision": p, "se": se})
    print(f"\n  {label}")
    print(f"    {'aralik':>18s} {'n':>6s} {'P(dogru)':>10s} {'+-SE':>7s}")
    for r in out:
        hi = "inf" if r["hi"] >= 1e9 else fmt.format(r["hi"])
        print(f"    {fmt.format(r['lo']):>7s} - {hi:>8s} {r['n']:6d} "
              f"{r['precision']*100:9.1f}% {r['se']*100:6.1f}")
    return out


def _mood_windows(path):
    """Mood signals for two disjoint 30s windows. Returns (early, late) or None."""
    out = []
    for offset in (0.0, 30.0):
        try:
            y, _ = librosa.load(path, sr=AF.MOOD_SR, offset=offset,
                                duration=AF.MOOD_DURATION)
        except Exception:
            return None
        if y is None or len(y) < AF.MOOD_SR * 10:
            return None
        s = AF.extract_mood_signals(y, AF.MOOD_SR)
        if s is None:
            return None
        out.append(s)
    return out


def main():
    m = joblib.load(MODEL_PATH)
    cache = AF.feature_cache_path()
    z = np.load(cache, allow_pickle=True)
    X, tracks = z["X"], z["tracks"]
    meta = pd.read_csv(TRACK_META).set_index("track")

    val = held_out(tracks)
    is_val = np.array([t in val for t in tracks])
    print(f"held-out {is_val.sum():,} / {len(tracks):,} parca  (encoder {m['encoder_id']})")

    Xv = X[is_val]
    tv = tracks[is_val]
    if m.get("pca") is not None:
        Xv = m["pca"].transform(Xv)
    Xv = m["scaler"].transform(Xv)

    Pa = m["clf_artist"].predict_proba(Xv)
    Pg = m["clf_genre"].predict_proba(Xv)
    artists, genres = m["artists"], m["genres"]
    g_of_a = m["genre_of_artist"]
    a_cls, g_cls = m["clf_artist"].classes_, m["clf_genre"].classes_

    eps = 1e-12
    logPg = np.full((len(Xv), len(genres)), -30.0)
    logPg[:, g_cls] = np.log(Pg + eps)
    scores = np.log(Pa + eps) + logPg[:, g_of_a[a_cls]]
    best = np.argmax(scores, axis=1)

    ai = a_cls[best]
    gi = g_of_a[ai]
    p_artist = Pa[np.arange(len(Pa)), best]
    g_pos = {int(g): i for i, g in enumerate(g_cls)}
    p_genre = np.array([Pg[i, g_pos[int(g)]] if int(g) in g_pos else 0.0
                        for i, g in enumerate(gi)])

    true_a = meta.loc[pd.Index(tv), "artist_type"].to_numpy()
    true_g = meta.loc[pd.Index(tv), "genre"].to_numpy()
    ok_a = (np.array([artists[i] for i in ai]) == true_a).astype(float)
    ok_g = (np.array([genres[i] for i in gi]) == true_g).astype(float)

    ratio_a = p_artist * len(artists)
    ratio_g = p_genre * len(genres)

    print("\n" + "=" * 74)
    print("1) SANATCI ve GENRE — guven aralilarina gore olculen dogruluk")
    print("=" * 74)
    print(f"  genel: artist top-1 %{ok_a.mean()*100:.2f}   genre joint %{ok_g.mean()*100:.2f}")
    art_curve = curve(ratio_a, ok_a, ARTIST_BINS, "artist (ratio_vs_chance)")
    gen_curve = curve(ratio_g, ok_g, GENRE_BINS, "genre (ratio_vs_chance)")

    # ---- mood: stability of the label across two windows ----
    print("\n" + "=" * 74)
    print("2) MOOD — etiket iki farkli 30sn penceresinde ayni kaliyor mu")
    print("=" * 74)
    calib = AF.load_mood_calibration()
    if calib is None:
        raise SystemExit("models/mood_calibration.json yok — once calibrate_mood.py")

    rng = np.random.default_rng(42)
    pool_tracks = [t for t in tv]
    pick = rng.choice(len(pool_tracks), min(MOOD_SAMPLE, len(pool_tracks)), replace=False)
    paths = [(pool_tracks[i], f"dataset/audio/{pool_tracks[i]}.mp3") for i in pick]
    paths = [(t, p) for t, p in paths if os.path.exists(p)]
    print(f"  {len(paths)} held-out parca, her biri icin 0-30sn ve 30-60sn")

    # Per axis, not pooled. The axes are not equally trustworthy and pooling hides it:
    # energy shifts +0.227 between the two windows (r=0.51) because the first 30s of a
    # type beat is the intro, while aggression barely moves (+0.056, r=0.77).
    per_axis = {ax: {"margin": [], "agree": [], "early": [], "late": []}
                for ax in AF.MOOD_AXES}
    with Pool(12) as pool:
        for res in pool.imap_unordered(_mood_windows, [p for _, p in paths], chunksize=4):
            if res is None:
                continue
            early, late = res
            se, sl = AF.axis_scores(early, calib), AF.axis_scores(late, calib)
            le, ll = AF.classify_mood(early, calib), AF.classify_mood(late, calib)
            for axis in se:
                lo, hi = calib["axis_cuts"][axis]
                per_axis[axis]["margin"].append(min(abs(se[axis] - lo),
                                                    abs(se[axis] - hi)))
                per_axis[axis]["agree"].append(float(le[axis]["label"] ==
                                                     ll[axis]["label"]))
                per_axis[axis]["early"].append(se[axis])
                per_axis[axis]["late"].append(sl[axis])

    mood_out = {}
    all_agree = []
    for axis, d in per_axis.items():
        mg, ag = np.array(d["margin"]), np.array(d["agree"])
        all_agree.append(ag)
        shift = float(np.mean(d["late"]) - np.mean(d["early"]))
        r = float(np.corrcoef(d["early"], d["late"])[0, 1])
        print(f"\n  --- {axis} --- tutarlilik %{ag.mean()*100:.1f}  "
              f"pencere kaymasi {shift:+.3f}  r {r:.3f}")
        mood_out[axis] = {
            "n": int(len(mg)), "overall_agreement": float(ag.mean()),
            "window_shift": shift, "window_r": r,
            "curve": curve(mg, ag, MOOD_BINS, f"{axis} (kesim noktasina uzaklik)",
                           fmt="{:.2f}"),
        }
    print(f"\n  havuzlanmis tutarlilik %{np.concatenate(all_agree).mean()*100:.1f}")

    out = {
        "held_out_n": int(is_val.sum()),
        "encoder_id": m["encoder_id"],
        "artist": {"unit": "ratio_vs_chance", "n_classes": len(artists),
                   "note": "joint-decoded, matching inference_engine.classify",
                   "curve": art_curve},
        "genre":  {"unit": "ratio_vs_chance", "n_classes": len(genres),
                   "curve": gen_curve},
        "mood":   {"unit": "distance_from_cut", "per_axis": mood_out,
                   "note": ("stability across disjoint windows, not ground-truth "
                            "accuracy; inference reads 0-30s, which is the intro")},
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nyazildi: {OUT_PATH}")


if __name__ == "__main__":
    main()
