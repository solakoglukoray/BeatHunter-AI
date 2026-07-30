"""
PCA boyut taramasi — PANNs embedding'i kac boyuta indirilmeli?

128 > 256 cikmasi boyut seciminin duyarli oldugunu gosterdi, ama o iki nokta
held-out uzerinde olculmustu. Bu script tarama yapar ve bir metodoloji sorununu
duzeltir.

⚠️ NEDEN IKI SPLIT
En iyi boyutu held-out uzerinde secmek, hiperparametreyi test setinde secmektir:
kazananin performansi sistematik olarak sisirilir, cunku 7 adaydan test setinde en
sansli olani seciyoruz. Bu yuzden:

  1. SECIM: train parcalari ikiye bolunur (ic-train / ic-val). Tarama ic-val'de
     yapilir, kazanan orada belirlenir. Held-out'a hic bakilmaz.
  2. RAPOR: kazanan boyut, gercek held-out'ta bir kez olculur. Bu sayi durustur.

Held-out'taki tam tarama da yazdirilir ama SADECE seffaflik icin — secimde
kullanilmaz. Aralarindaki fark, test setinde secim yapmanin ne kadar sisirdigini
dogrudan gosterir.

Ses islemez, GPU kullanmaz — onbelleklenmis 2048-dim embedding'leri yeniden projekte
edip LogReg'i tekrar fit eder.
"""

import os, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split

DIMS = (64, 96, 128, 160, 192, 256, 384)
PANNS_CACHE = "models/panns_features.npz"
TRACK_META  = "models/vector_meta_track.csv"
SLICE_META  = "dataset_processed/metadata_processed.csv"
TAXONOMY    = "models/taxonomy.json"
OUT         = os.path.expanduser("~/bench_out/pca_sweep.json")


def held_out_mask(tracks):
    m = pd.read_csv(SLICE_META)
    m["track"] = m["filename"].str.split("_slice").str[0]
    _, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                   random_state=42).split(m, groups=m["track"]))
    val = set(m.iloc[va]["track"].unique())
    return np.array([t in val for t in tracks])


def fit_and_score(X, ya, yg, tr, te, artists, genres, g_of_a, dim):
    """PCA(dim) -> scale -> LogReg -> joint decode. Everything fitted on `tr` only."""
    pca = PCA(n_components=dim, random_state=42).fit(X[tr])
    Xp = pca.transform(X)
    sc = StandardScaler().fit(Xp[tr])
    Xs = sc.transform(Xp)

    clf_a = LogisticRegression(max_iter=1000, n_jobs=-1).fit(Xs[tr], ya[tr])
    clf_g = LogisticRegression(max_iter=1000, n_jobs=-1).fit(Xs[tr], yg[tr])
    Pa, Pg = clf_a.predict_proba(Xs[te]), clf_g.predict_proba(Xs[te])
    ta, tg = ya[te], yg[te]

    a1 = float((clf_a.classes_[Pa.argmax(1)] == ta).mean())
    a5 = float(np.mean([ta[i] in clf_a.classes_[np.argsort(-Pa[i])[:5]]
                        for i in range(len(ta))]))
    logPg = np.full((len(ta), len(genres)), -30.0)
    logPg[:, clf_g.classes_] = np.log(Pg + 1e-12)
    joint = np.log(Pa + 1e-12) + logPg[:, g_of_a[clf_a.classes_]]
    gj = float((g_of_a[clf_a.classes_[joint.argmax(1)]] == tg).mean())

    return {"dim": dim, "artist_top1": a1, "artist_top5": a5, "genre_joint": gj,
            "variance": float(pca.explained_variance_ratio_.sum())}


def main():
    z = np.load(PANNS_CACHE, allow_pickle=True)
    X, tracks = z["X"], z["tracks"]
    meta = pd.read_csv(TRACK_META).set_index("track")
    taxonomy = json.load(open(TAXONOMY, encoding="utf-8"))

    is_held = held_out_mask(tracks)
    artists = np.sort(meta.loc[tracks, "artist_type"].unique())
    genres  = np.sort(meta.loc[tracks, "genre"].unique())
    a_idx = {a: i for i, a in enumerate(artists)}
    g_idx = {g: i for i, g in enumerate(genres)}
    ya = meta.loc[tracks, "artist_type"].map(a_idx).to_numpy()
    yg = meta.loc[tracks, "genre"].map(g_idx).to_numpy()
    g_of_a = np.array([g_idx[taxonomy[a]["genre"]] for a in artists])

    train_rows = np.where(~is_held)[0]
    held_rows  = np.where(is_held)[0]

    # Selection split, carved out of TRAIN only. Single-member classes cannot be
    # stratified, so they are pinned to the inner-train side.
    counts = pd.Series(ya[train_rows]).value_counts()
    ok = np.isin(ya[train_rows], counts[counts >= 2].index)
    inner_tr, inner_val = train_test_split(train_rows[ok], test_size=0.25,
                                           random_state=7, stratify=ya[train_rows][ok])
    inner_tr = np.concatenate([inner_tr, train_rows[~ok]])

    print(f"{len(X):,} tracks x {X.shape[1]} dims")
    print(f"  selection: inner-train {len(inner_tr):,} / inner-val {len(inner_val):,}")
    print(f"  reporting: held-out {len(held_rows):,}\n")

    print("=" * 78)
    print("1) SELECTION — inner-val (held-out untouched)")
    print("=" * 78)
    print(f"{'dim':>5s} {'variance':>9s} {'artist top1':>12s} {'top5':>8s} {'genre joint':>12s}")
    sel = []
    t0 = time.time()
    for d in DIMS:
        r = fit_and_score(X, ya, yg, inner_tr, inner_val, artists, genres, g_of_a, d)
        sel.append(r)
        print(f"{d:5d} {r['variance']*100:8.1f}% {r['artist_top1']*100:11.2f}% "
              f"{r['artist_top5']*100:7.2f}% {r['genre_joint']*100:11.2f}%", flush=True)
    best = max(sel, key=lambda r: r["artist_top1"])
    print(f"\n  chosen on inner-val: {best['dim']} dims "
          f"(artist top-1 {best['artist_top1']*100:.2f}%)   [{time.time()-t0:.0f}s]")

    print("\n" + "=" * 78)
    print("2) REPORT — held-out, full sweep for transparency")
    print("   Only the chosen dim's row is an honest estimate; the others were not")
    print("   used for selection but reading a maximum off this table would be.")
    print("=" * 78)
    print(f"{'dim':>5s} {'artist top1':>12s} {'top5':>8s} {'genre joint':>12s}   ")
    rep = []
    for d in DIMS:
        r = fit_and_score(X, ya, yg, train_rows, held_rows, artists, genres, g_of_a, d)
        rep.append(r)
        mark = "  <- chosen" if d == best["dim"] else ""
        print(f"{d:5d} {r['artist_top1']*100:11.2f}% {r['artist_top5']*100:7.2f}% "
              f"{r['genre_joint']*100:11.2f}%{mark}", flush=True)

    cur = next(r for r in rep if r["dim"] == 128)
    win = next(r for r in rep if r["dim"] == best["dim"])
    n = len(held_rows)
    se = np.sqrt(cur["artist_top1"] * (1 - cur["artist_top1"]) / n)
    print(f"\ncurrent production is 128 dims: artist {cur['artist_top1']*100:.2f}%")
    print(f"chosen {best['dim']} dims:            artist {win['artist_top1']*100:.2f}%")
    d = (win["artist_top1"] - cur["artist_top1"]) * 100
    print(f"difference {d:+.2f}pt against a ~{se*100*1.41:.2f}pt SE "
          f"-> {abs(d)/(se*100*1.41):.1f} SE")
    if abs(d) / (se * 100 * 1.41) < 2:
        print("  NOT separable at this sample size — treat the two as equivalent")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"selection": sel, "held_out": rep, "chosen": best["dim"]},
              open(OUT, "w"), indent=2)
    print(f"\nresults: {OUT}")


if __name__ == "__main__":
    main()
