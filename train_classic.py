"""
BeatHunter AI — trains the classical-feature classifier and builds the retrieval catalogue.

Replaces the CNN path for classification and similarity. Measured on identical held-out
tracks (GroupShuffleSplit by track, random_state=42 — the same split the CNN was scored
on):

    artist      CNN 0.8x chance   ->   16.7x
    genre       CNN 0.7x chance   ->    9.7x

Runs on CPU in seconds. No GPU, no WSL2.

    python train_classic.py                # uses the cached feature matrix if present
    python train_classic.py --extract      # re-extracts features from audio (~14 min)

Writes:
    models/classic_model.joblib     scaler + artist/genre LogReg + class names
    models/classic_catalog.npz      L2-normalised track vectors for k-NN retrieval
    models/classic_catalog.csv      track -> artist/genre, aligned row-for-row
"""

import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

import audio_features as AF

MODEL_DIR    = "models"
FEATURE_CACHE = os.path.join(MODEL_DIR, "classic_features.npz")
MODEL_PATH   = os.path.join(MODEL_DIR, "classic_model.joblib")
CATALOG_NPZ  = os.path.join(MODEL_DIR, "classic_catalog.npz")
CATALOG_CSV  = os.path.join(MODEL_DIR, "classic_catalog.csv")
TRACK_META   = os.path.join(MODEL_DIR, "vector_meta_track.csv")
SLICE_META   = "dataset_processed/metadata_processed.csv"
TAXONOMY     = os.path.join(MODEL_DIR, "taxonomy.json")


def extract_all(workers=6):
    from multiprocessing import Pool
    meta = pd.read_csv(TRACK_META)
    meta["path"] = meta["track"].map(lambda t: f"dataset/audio/{t}.mp3")
    meta = meta[meta["path"].map(os.path.exists)].reset_index(drop=True)
    print(f"extracting {len(meta):,} tracks with {workers} workers...", flush=True)

    t0, rows, feats = time.time(), [], []
    with Pool(workers) as pool:
        for i, (t, f) in enumerate(pool.imap_unordered(
                _work, list(zip(meta["track"], meta["path"])), chunksize=8), 1):
            if f is not None:
                rows.append(t); feats.append(f)
            if i % 1000 == 0:
                el = time.time() - t0
                print(f"  {i:,}/{len(meta):,}  {el:.0f}s  eta {el/i*(len(meta)-i):.0f}s", flush=True)

    X, tracks = np.stack(feats), np.array(rows)
    np.savez_compressed(FEATURE_CACHE, X=X, tracks=tracks)
    print(f"cached {X.shape} -> {FEATURE_CACHE} ({time.time()-t0:.0f}s)")
    return X, tracks


def _work(args):
    track, path = args
    return track, AF.extract_from_file(path)


def held_out_tracks():
    """The exact val tracks the CNN was scored on, so the numbers stay comparable."""
    m = pd.read_csv(SLICE_META)
    m["track"] = m["filename"].str.split("_slice").str[0]
    _, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                   random_state=42).split(m, groups=m["track"]))
    return set(m.iloc[va]["track"].unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="re-extract features from audio")
    args = ap.parse_args()

    if args.extract or not os.path.exists(FEATURE_CACHE):
        X, tracks = extract_all()
    else:
        z = np.load(FEATURE_CACHE, allow_pickle=True)
        X, tracks = z["X"], z["tracks"]
        print(f"loaded cached features {X.shape} from {FEATURE_CACHE}")

    if X.shape[1] != AF.FEATURE_DIM:
        sys.exit(f"cached features are {X.shape[1]}-dim but audio_features.FEATURE_DIM "
                 f"is {AF.FEATURE_DIM}. Re-run with --extract.")

    meta = pd.read_csv(TRACK_META).set_index("track").loc[pd.Index(tracks)]
    taxonomy = json.load(open(TAXONOMY, encoding="utf-8"))

    val = held_out_tracks()
    is_val = np.array([t in val for t in tracks])
    print(f"train {(~is_val).sum():,} / held-out {is_val.sum():,} tracks\n")

    # Scaler is fit on training tracks only — fitting on everything would leak the
    # held-out distribution into the numbers below.
    scaler = StandardScaler().fit(X[~is_val])
    Xs = scaler.transform(X)

    artists = np.sort(meta["artist_type"].unique())
    genres  = np.sort(meta["genre"].unique())
    a_idx = {a: i for i, a in enumerate(artists)}
    g_idx = {g: i for i, g in enumerate(genres)}
    ya = meta["artist_type"].map(a_idx).to_numpy()
    yg = meta["genre"].map(g_idx).to_numpy()

    print("training...", flush=True)
    t0 = time.time()
    clf_a = LogisticRegression(max_iter=1000, n_jobs=-1).fit(Xs[~is_val], ya[~is_val])
    clf_g = LogisticRegression(max_iter=1000, n_jobs=-1).fit(Xs[~is_val], yg[~is_val])
    print(f"  fitted in {time.time()-t0:.0f}s\n")

    # ── how should genre be produced? measure, do not assume ────────────────────
    Pa = clf_a.predict_proba(Xs[is_val])
    Pg = clf_g.predict_proba(Xs[is_val])
    ta, tg = ya[is_val], yg[is_val]
    ch_a = float((pd.Series(ta).value_counts(normalize=True) ** 2).sum())
    ch_g = float((pd.Series(tg).value_counts(normalize=True) ** 2).sum())

    # artist index -> genre index, from the taxonomy
    g_of_a = np.array([g_idx[taxonomy[a]["genre"]] for a in artists], dtype=int)

    def acc(pred, truth): return float((pred == truth).mean())
    def top5(P, cls, truth):
        return float(np.mean([truth[i] in cls[np.argsort(-P[i])[:5]] for i in range(len(truth))]))

    direct_a = clf_a.classes_[Pa.argmax(1)]
    direct_g = clf_g.classes_[Pg.argmax(1)]
    derived_g = g_of_a[direct_a]

    # joint: pick the artist that best explains both heads, then read genre off it
    eps = 1e-12
    logPa = np.log(Pa + eps)
    logPg_full = np.full((len(ta), len(genres)), -30.0)
    logPg_full[:, clf_g.classes_] = np.log(Pg + eps)
    joint_scores = logPa + logPg_full[:, g_of_a[clf_a.classes_]]
    joint_a = clf_a.classes_[joint_scores.argmax(1)]
    joint_g = g_of_a[joint_a]

    print("=" * 72)
    print(f"{'strategy':28s} {'artist':>9s} {'genre':>9s}   consistent?")
    print("=" * 72)
    print(f"{'direct (independent)':28s} {acc(direct_a,ta)*100:8.2f}% {acc(direct_g,tg)*100:8.2f}%   no")
    print(f"{'artist -> taxonomy genre':28s} {acc(direct_a,ta)*100:8.2f}% {acc(derived_g,tg)*100:8.2f}%   yes")
    print(f"{'joint decode':28s} {acc(joint_a,ta)*100:8.2f}% {acc(joint_g,tg)*100:8.2f}%   yes")
    print(f"\nchance: artist {ch_a*100:.2f}%  genre {ch_g*100:.2f}%")
    print(f"top-5 : artist {top5(Pa, clf_a.classes_, ta)*100:.2f}%  "
          f"genre {top5(Pg, clf_g.classes_, tg)*100:.2f}%")
    print(f"lift  : artist {acc(direct_a,ta)/ch_a:.1f}x  genre {acc(direct_g,tg)/ch_g:.1f}x")
    print("\nCNN on these same held-out tracks: artist 0.8x, genre 0.7x")

    # ── persist ────────────────────────────────────────────────────────────────
    joblib.dump({
        "scaler": scaler, "clf_artist": clf_a, "clf_genre": clf_g,
        "artists": artists, "genres": genres, "genre_of_artist": g_of_a,
        "feature_dim": AF.FEATURE_DIM, "duration": AF.DURATION,
    }, MODEL_PATH)

    Xn = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9)
    np.savez_compressed(CATALOG_NPZ, X=Xn.astype(np.float32), tracks=tracks)
    # Built from `tracks` directly so the CSV is row-for-row aligned with the npz.
    # Reusing meta's index here silently depended on its name surviving .loc[].
    pd.DataFrame({
        "track": tracks,
        "artist_type": meta["artist_type"].to_numpy(),
        "genre": meta["genre"].to_numpy(),
    }).to_csv(CATALOG_CSV, index=False)

    print(f"\nwrote {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1e3:.0f} KB)")
    print(f"wrote {CATALOG_NPZ} ({os.path.getsize(CATALOG_NPZ)/1e6:.1f} MB, {Xn.shape})")
    print(f"wrote {CATALOG_CSV}")


if __name__ == "__main__":
    main()
