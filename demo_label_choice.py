"""
Shows the label chooser working on real held-out tracks, before anything is integrated.

Deliberately picks three groups so the logic can be judged rather than trusted:
  A) tracks whose artist clears the 150x bar     — artist should usually win
  B) tracks whose artist is hopeless             — something else should win, or nothing
  C) tracks with an extreme mood axis            — mood should win, and should deserve it

Ground truth is printed alongside, so "the chooser picked artist" can be checked against
"the artist was actually right".

    python demo_label_choice.py            # 4 tracks per group
    python demo_label_choice.py --n 6
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import audio_features as AF
import label_recommender as LR

MODEL_PATH = "models/classic_model.joblib"
TRACK_META = "models/vector_meta_track.csv"
SLICE_META = "dataset_processed/metadata_processed.csv"


def held_out_mask(tracks):
    m = pd.read_csv(SLICE_META)
    m["track"] = m["filename"].str.split("_slice").str[0]
    _, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                   random_state=42).split(m, groups=m["track"]))
    val = set(m.iloc[va]["track"].unique())
    return np.array([t in val for t in tracks])


def build_confidence(m, Xs_row):
    """Reproduces inference_engine.classify's confidence dict from a cached embedding."""
    Pa = m["clf_artist"].predict_proba(Xs_row)[0]
    Pg = m["clf_genre"].predict_proba(Xs_row)[0]
    artists, genres, g_of_a = m["artists"], m["genres"], m["genre_of_artist"]
    a_cls, g_cls = m["clf_artist"].classes_, m["clf_genre"].classes_
    eps = 1e-12
    logPg = np.full(len(genres), -30.0)
    logPg[g_cls] = np.log(Pg + eps)
    scores = np.log(Pa + eps) + logPg[g_of_a[a_cls]]
    best = int(np.argmax(scores))
    ai, gi = int(a_cls[best]), int(g_of_a[a_cls[int(np.argmax(scores))]])
    p_a = float(Pa[best])
    gpos = list(g_cls).index(gi) if gi in g_cls else None
    p_g = float(Pg[gpos]) if gpos is not None else 0.0
    return {
        "artist": {"label": str(artists[ai]), "probability": p_a,
                   "ratio_vs_chance": p_a * len(artists)},
        "genre": {"label": str(genres[gi]), "probability": p_g,
                  "ratio_vs_chance": p_g * len(genres)},
        "reliable": p_a * len(artists) >= float(m.get("confidence_threshold") or 150),
    }


def show(tag, rows):
    print("\n" + "=" * 78)
    print(tag)
    print("=" * 78)
    for r in rows:
        c, mood, truth = r["conf"], r["mood"], r["truth"]
        print(f"\n  {r['track']}   gercek: {truth['artist_type']} / {truth['genre']}")
        print(f"    tahmin: {c['artist']['label']} ({c['artist']['ratio_vs_chance']:.0f}x)"
              f"  |  {c['genre']['label']} ({c['genre']['ratio_vs_chance']:.0f}x)")
        print("    mood: " + ", ".join(
            f"{a}={v['label']} {v['score']:.2f}" for a, v in mood.items()))

        for crit in ("reliability", "information"):
            rec = LR.recommend_label(c, mood, criterion=crit)
            mark = "OK " if rec["qualifies"] else "YOK"
            print(f"    [{crit:11s}] {mark} {rec['reason']}")
            if rec["qualifies"]:
                print(f"                  -> ara: \"{rec['query']}\"")

        rec = LR.recommend_label(c, mood)
        b = rec["best"]
        if b["kind"] == "artist":
            hit = b["label"] == truth["artist_type"]
        elif b["kind"] == "genre":
            hit = b["label"] == truth["genre"]
        else:
            hit = None
        if hit is not None:
            print(f"    secilen etiket gercekten dogru mu: {'EVET' if hit else 'HAYIR'}")


def summarise(rows):
    """
    Does the chooser actually beat the current policy, or does it just feel clever?

    Only artist and genre picks can be scored against ground truth. Mood picks are
    counted separately and honestly: their reliability is a stability measurement, not
    an accuracy, so folding them into one precision number would be inventing evidence.
    """
    print("\n" + "=" * 78)
    print("TOPLU DEGERLENDIRME — tum held-out parcalar")
    print("=" * 78)

    prod_hit = prod_shown = 0
    for r in rows:
        if r["conf"]["reliable"]:
            prod_shown += 1
            prod_hit += r["conf"]["artist"]["label"] == r["truth"]["artist_type"]
    print(f"\n  MEVCUT POLITIKA (150x uzeri sanatciyi goster, yoksa liste)")
    print(f"    tek etiket gosterilen: {prod_shown}/{len(rows)} "
          f"(%{prod_shown/len(rows)*100:.1f})   dogruluk %{prod_hit/max(prod_shown,1)*100:.1f}")

    for crit in ("reliability", "information"):
        kinds = {"artist": [0, 0], "genre": [0, 0], "mood": [0, 0]}
        shown = 0
        for r in rows:
            rec = LR.recommend_label(r["conf"], r["mood"], criterion=crit)
            if not rec or not rec["qualifies"]:
                continue
            shown += 1
            b = rec["best"]
            kinds[b["kind"]][0] += 1
            if b["kind"] == "artist":
                kinds["artist"][1] += b["label"] == r["truth"]["artist_type"]
            elif b["kind"] == "genre":
                kinds["genre"][1] += b["label"] == r["truth"]["genre"]
        print(f"\n  SECICI / {crit}")
        print(f"    tek etiket gosterilen: {shown}/{len(rows)} (%{shown/len(rows)*100:.1f})")
        checkable = kinds["artist"][0] + kinds["genre"][0]
        hits = kinds["artist"][1] + kinds["genre"][1]
        for k, (cnt, hit) in kinds.items():
            if not cnt:
                continue
            if k == "mood":
                print(f"      {k:7s} {cnt:4d} kez secildi   (dogruluk yer gercegiyle "
                      f"olculemez — bkz. not)")
            else:
                print(f"      {k:7s} {cnt:4d} kez secildi   dogru %{hit/cnt*100:.1f}")
        if checkable:
            print(f"    yer gercegiyle olculebilen {checkable} secimde dogruluk "
                  f"%{hits/checkable*100:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--summary", action="store_true",
                    help="ornekleri atla, sadece toplu sayilari goster")
    a = ap.parse_args()

    if LR.load_reliability() is None:
        raise SystemExit("once: python calibrate_label_choice.py")
    calib = AF.load_mood_calibration()
    if calib is None:
        raise SystemExit("once: python calibrate_mood.py")

    m = joblib.load(MODEL_PATH)
    z = np.load(AF.feature_cache_path(), allow_pickle=True)
    X, tracks = z["X"], z["tracks"]
    meta = pd.read_csv(TRACK_META).set_index("track")
    iv = held_out_mask(tracks)
    X, tracks = X[iv], tracks[iv]
    Xs = m["scaler"].transform(m["pca"].transform(X))

    # Mood comes from the cached signals where available, so this demo never re-analyses
    # audio — the same constraint the recommender itself works under.
    zs = np.load("models/mood_signals.npz", allow_pickle=True)
    sig = {t: dict(zip(AF.MOOD_SIGNAL_NAMES, row))
           for t, row in zip(zs["tracks"], zs["X"])}

    rows = []
    for i, t in enumerate(tracks):
        if t not in sig:
            continue
        conf = build_confidence(m, Xs[i:i + 1])
        mood = AF.classify_mood(sig[t], calib)
        rows.append({"track": t, "conf": conf, "mood": mood,
                     "truth": meta.loc[t]})
    print(f"{len(rows)} held-out parca (mood sinyali cache'te olanlar)")

    summarise(rows)
    if a.summary:
        return

    rng = np.random.default_rng(3)

    def pick(sel, k):
        idx = [i for i, r in enumerate(rows) if sel(r)]
        if not idx:
            return []
        take = rng.choice(idx, min(k, len(idx)), replace=False)
        return [rows[i] for i in take]

    show("A) SANATCI 150x BARAJINI GECIYOR", pick(
        lambda r: r["conf"]["artist"]["ratio_vs_chance"] >= 150, a.n))
    show("B) SANATCI UMUTSUZ (<25x)", pick(
        lambda r: r["conf"]["artist"]["ratio_vs_chance"] < 25, a.n))
    show("C) MOOD UC NOKTADA (bir eksen 0.85 uzeri ya da 0.15 alti)", pick(
        lambda r: any(v["score"] >= 0.85 or v["score"] <= 0.15
                      for v in r["mood"].values()), a.n))


if __name__ == "__main__":
    main()
