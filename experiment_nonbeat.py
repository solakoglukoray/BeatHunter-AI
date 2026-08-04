"""
Do the 1,509 non-type-beat tracks help the classifier, or hurt it?

The corpus is meant to be instrumentals but 9.6% of it is released songs, official
videos and karaoke — "Kendrick Lamar - ELEMENT.", "A$AP Rocky - L$D (Official Video)".
They carry vocals, which is extra signal the model can key on. Whether that HELPS is
not obvious: vocals identify an artist strongly, so they might sharpen the artist head
while teaching it a cue that production audio (post-Demucs, instrumental) never has.

DESIGN
------
Both conditions are scored on the SAME test set, and that set is type beats only.
Varying the test set between conditions would move the numbers for reasons unrelated
to the tracks' quality, and scoring on songs measures something production never sees:
a user uploads a beat.

    test        held-out ∩ type-beat        (fixed)
    A: train    everything in train         (current production)
    B: train    train ∩ type-beat           (songs removed)

PCA and the scaler are refit inside each condition. Fitting them once on everything
would leak the removed tracks back into condition B.

Reported separately: how the model does ON the non-type-beat held-out tracks. If it
scores them far higher than real beats, the headline metric has been flattered by
material the product never encounters.

    python experiment_nonbeat.py
"""

import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import artist_match
import audio_features as AF
import corpus

TITLES = "models/video_titles.csv"


def evaluate(Xtr, ytr_a, ytr_g, Xte, yte_a, yte_g, artists, genres, g_of_a, tag):
    """Full production path — PCA -> scale -> LogReg -> joint decode."""
    pca = PCA(n_components=AF.PCA_DIM, random_state=42).fit(Xtr)
    sc = StandardScaler().fit(pca.transform(Xtr))
    Ztr, Zte = sc.transform(pca.transform(Xtr)), sc.transform(pca.transform(Xte))

    ca = LogisticRegression(max_iter=1000, n_jobs=-1).fit(Ztr, ytr_a)
    cg = LogisticRegression(max_iter=1000, n_jobs=-1).fit(Ztr, ytr_g)
    Pa, Pg = ca.predict_proba(Zte), cg.predict_proba(Zte)

    lg = np.full((len(Zte), len(genres)), -30.0)
    lg[:, cg.classes_] = np.log(Pg + 1e-12)
    joint = np.log(Pa + 1e-12) + lg[:, g_of_a[ca.classes_]]
    pick = ca.classes_[joint.argmax(1)]

    top1 = float((pick == yte_a).mean())
    top5 = float(np.mean([yte_a[i] in ca.classes_[np.argsort(-Pa[i])[:5]]
                          for i in range(len(yte_a))]))
    gj = float((g_of_a[pick] == yte_g).mean())
    print(f"  {tag:34s} {top1*100:6.2f}% {top5*100:6.2f}% {gj*100:6.2f}%  "
          f"(train {len(Xtr):,})")
    return top1, top5, gj


def main():
    meta = corpus.load_tracks().set_index("track")
    tax = json.load(open("models/taxonomy.json", encoding="utf-8"))
    z = np.load(AF.feature_cache_path(), allow_pickle=True)
    X, tracks = z["X"], z["tracks"]
    keep = np.isin(tracks, meta.index.to_numpy())
    X, tracks = X[keep], tracks[keep]

    titles = pd.read_csv(TITLES).fillna({"title": ""})
    tmap = dict(zip(titles["track"], titles["title"]))
    has_title = np.array([bool(tmap.get(t, "")) for t in tracks])
    is_beat = np.array([artist_match.looks_like_type_beat(tmap.get(t, ""))
                        for t in tracks])
    # No title -> unknown, not guilty. Kept everywhere; only titles that exist and
    # clearly lack any beat marker are treated as songs.
    is_song = has_title & ~is_beat

    iv = corpus.held_out_mask(tracks)
    print(f"korpus {len(tracks):,}  | baslikli {has_title.sum():,}  "
          f"| sarki {is_song.sum():,} (%{is_song.mean()*100:.1f})")
    print(f"held-out {iv.sum():,}  | bunlarin {int((iv & is_song).sum()):,} tanesi sarki")

    artists = np.sort(meta.loc[tracks, "artist_type"].unique())
    genres = np.sort(meta.loc[tracks, "genre"].unique())
    ai = {a: i for i, a in enumerate(artists)}
    gi = {g: i for i, g in enumerate(genres)}
    ya = meta.loc[tracks, "artist_type"].map(ai).to_numpy()
    yg = meta.loc[tracks, "genre"].map(gi).to_numpy()
    g_of_a = np.array([gi[tax[a]["genre"]] for a in artists])

    # Fixed test set: held-out type beats only.
    te = iv & ~is_song
    print(f"\nTEST SETI (sabit): held-out ∩ type-beat = {te.sum():,} parca")
    print(f"\n  {'kosul':34s} {'top-1':>7s} {'top-5':>7s} {'genre':>7s}")

    trA = ~iv
    a = evaluate(X[trA], ya[trA], yg[trA], X[te], ya[te], yg[te],
                 artists, genres, g_of_a, "A: hepsi (mevcut uretim)")

    trB = ~iv & ~is_song
    b = evaluate(X[trB], ya[trB], yg[trB], X[te], ya[te], yg[te],
                 artists, genres, g_of_a, "B: sarkilar egitimden cikarildi")

    n = int(te.sum())
    print(f"\n  fark (B - A): top-1 {(b[0]-a[0])*100:+.2f}p  "
          f"top-5 {(b[1]-a[1])*100:+.2f}p  genre {(b[2]-a[2])*100:+.2f}p")
    se = (a[0] * (1 - a[0]) / n) ** 0.5
    print(f"  top-1 icin SE ~{se*100:.2f}p -> fark {abs(b[0]-a[0])/se:.1f} SE")

    # Are the songs themselves easier? If so the headline number was flattered.
    tes = iv & is_song
    if tes.sum() > 30:
        print(f"\n  ayrica: modelin SARKILAR uzerindeki performansi "
              f"({tes.sum():,} held-out sarki)")
        evaluate(X[trA], ya[trA], yg[trA], X[tes], ya[tes], yg[tes],
                 artists, genres, g_of_a, "A modeli, sarki test seti")

    np.savez_compressed("models/nonbeat_flags.npz",
                        tracks=tracks, is_song=is_song, has_title=has_title)
    print("\nyazildi: models/nonbeat_flags.npz")


if __name__ == "__main__":
    main()
