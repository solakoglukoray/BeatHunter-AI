"""
BeatHunter AI — trains the classifier and builds the retrieval catalogue.

Pipeline: PANNs embedding (2048) -> PCA (128) -> StandardScaler -> LogisticRegression,
with genre resolved by joint decoding over the taxonomy.

Measured on 2,847 held-out tracks (GroupShuffleSplit by track, random_state=42):

    representation            artist top-1   lift    top-5    genre (joint)
    EfficientNetB3 (ImageNet)      ~0.7%     0.8x      —          ~1.3%
    classical MFCC 102-dim          9.45%   16.7x    24.13%      17.32%
    PANNs PCA-128                  12.96%   22.9x    32.03%      24.17%

    python train_classic.py                 # reuse cached embeddings
    python train_classic.py --extract        # re-embed from audio (~49 min on GPU)
    python train_classic.py --classical      # fall back to MFCC features (no PyTorch)

Writes:
    models/classic_model.joblib     pca + scaler + classifiers + calibrated threshold
    models/classic_catalog.npz      L2-normalised 128-dim track vectors
    models/classic_catalog.csv      track -> artist/genre, row-for-row aligned
"""

import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit

import audio_features as AF

MODEL_DIR    = "models"
# Cache path follows the active encoder, so switching encoders cannot reuse the
# other one's embeddings — see audio_features.feature_cache_path().
def _panns_cache():
    return AF.feature_cache_path()
CLASSIC_CACHE = os.path.join(MODEL_DIR, "classic_features.npz")
MODEL_PATH   = os.path.join(MODEL_DIR, "classic_model.joblib")
CATALOG_NPZ  = os.path.join(MODEL_DIR, "classic_catalog.npz")
CATALOG_CSV  = os.path.join(MODEL_DIR, "classic_catalog.csv")
TRACK_META   = os.path.join(MODEL_DIR, "vector_meta_track.csv")
SLICE_META   = "dataset_processed/metadata_processed.csv"
TAXONOMY     = os.path.join(MODEL_DIR, "taxonomy.json")


def extract_panns(batch=16, save_every=1000):
    """Embeds every track, checkpointing every save_every so a power cut costs minutes
    rather than the whole ~49-minute run."""
    meta = pd.read_csv(TRACK_META)
    meta["path"] = meta["track"].map(lambda t: f"dataset/audio/{t}.mp3")
    meta = meta[meta["path"].map(os.path.exists)].reset_index(drop=True)

    tracks_out, emb_out = [], []
    if os.path.exists(_panns_cache()):
        z = np.load(_panns_cache(), allow_pickle=True)
        tracks_out, emb_out = list(z["tracks"]), list(z["X"])
        print(f"resuming from {len(tracks_out):,} cached embeddings")

    done = set(tracks_out)
    todo = meta[~meta["track"].isin(done)].reset_index(drop=True)
    if len(todo) == 0:
        return np.stack(emb_out).astype(np.float32), np.array(tracks_out)
    print(f"embedding {len(todo):,} of {len(meta):,} tracks...", flush=True)

    buf, buf_t, t0 = [], [], time.time()

    def flush():
        if buf:
            emb_out.extend(AF.extract_embeddings_batch(buf))
            tracks_out.extend(buf_t)
            buf.clear(); buf_t.clear()

    def save():
        tmp = _panns_cache() + ".tmp.npz"
        np.savez_compressed(tmp, X=np.stack(emb_out).astype(np.float32),
                            tracks=np.array(tracks_out))
        os.replace(tmp, _panns_cache())

    for i, (t, p) in enumerate(zip(todo["track"], todo["path"]), 1):
        y = AF.load_audio(p)
        if y is not None and len(y) >= AF.PANNS_SR * 5:
            buf.append(y); buf_t.append(t)
        if len(buf) >= batch:
            flush()
        if i % 500 == 0:
            el = time.time() - t0
            print(f"  {i:,}/{len(todo):,}  {el:.0f}s  eta {el/i*(len(todo)-i):.0f}s", flush=True)
        if i % save_every == 0:
            flush(); save()
    flush(); save()
    print(f"done ({time.time()-t0:.0f}s)")
    return np.stack(emb_out).astype(np.float32), np.array(tracks_out)


def held_out(tracks):
    """The exact val tracks every published figure in this project was measured on."""
    m = pd.read_csv(SLICE_META)
    m["track"] = m["filename"].str.split("_slice").str[0]
    _, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                   random_state=42).split(m, groups=m["track"]))
    val = set(m.iloc[va]["track"].unique())
    return np.array([t in val for t in tracks])


def calibrate_threshold(Pa, truth, classes, n_classes, pick):
    """
    Picks the confidence bar from THIS model's accuracy distribution.

    The previous model's bar (100x) is not reusable: it was chosen against a distribution
    where top-1 was 9.45%, and this one is at 12.96% with a differently shaped confidence
    profile. Copying the number over would be cargo-culting a measurement.

    Target: the lowest bar whose retained predictions are right at least MIN_PRECISION of
    the time. Below that a shown name misleads more often than it helps.

    `pick` is the column index of the artist the pipeline ACTUALLY shows — the joint
    decode, not Pa.argmax. Those differ, and calibrating on argmax while serving the
    joint pick measured a quantity nothing downstream used. Same drift as Trap 1, one
    level up: the number was right about a system we were not running.
    """
    MIN_PRECISION = 0.35
    rows_i = np.arange(len(Pa))
    conf  = Pa[rows_i, pick] * n_classes     # ratio against a uniform guess
    right = classes[pick] == truth

    print("\n  confidence bar   shown   top-1 correct")
    rows, chosen = [], None
    for thr in (3, 10, 25, 50, 75, 100, 150, 200, 300, 400):
        sel = conf >= thr
        if sel.sum() < 30:
            continue
        prec = float(right[sel].mean())
        rows.append({"threshold": thr, "coverage": float(sel.mean()), "precision": prec})
        print(f"    >= {thr:4d}x      {sel.mean()*100:5.1f}%        {prec*100:5.1f}%")
        if chosen is None and prec >= MIN_PRECISION:
            chosen = thr
    if chosen is None:
        chosen = max(r["threshold"] for r in rows)
        print(f"\n  no bar reaches {MIN_PRECISION*100:.0f}% precision — using the highest "
              f"measured ({chosen}x); the UI must lead with top-5")
    else:
        print(f"\n  chosen: {chosen}x (first bar at >= {MIN_PRECISION*100:.0f}% precision)")
    return chosen, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--classical", action="store_true",
                    help="use MFCC features instead of PANNs (no PyTorch needed)")
    args = ap.parse_args()

    if args.classical:
        z = np.load(CLASSIC_CACHE, allow_pickle=True)
        X, tracks, kind, pca_dim = z["X"], z["tracks"], "classical-102", None
        print(f"classical features {X.shape} (fallback path)")
    else:
        if args.extract or not os.path.exists(_panns_cache()):
            X, tracks = extract_panns()
        else:
            z = np.load(_panns_cache(), allow_pickle=True)
            X, tracks = z["X"], z["tracks"]
            print(f"cached PANNs embeddings {X.shape} from {_panns_cache()}")
        if X.shape[1] != AF.EMBED_DIM:
            sys.exit(f"cache is {X.shape[1]}-dim, expected {AF.EMBED_DIM}. Re-run --extract.")
        kind, pca_dim = "panns-pca128", AF.PCA_DIM

    meta = pd.read_csv(TRACK_META).set_index("track")
    taxonomy = json.load(open(TAXONOMY, encoding="utf-8"))
    is_val = held_out(tracks)
    print(f"train {(~is_val).sum():,} / held-out {is_val.sum():,} tracks")

    # PCA and scaler are fitted on TRAINING tracks only — fitting on everything would
    # leak the held-out distribution into every number below.
    pca = None
    if pca_dim:
        t0 = time.time()
        pca = PCA(n_components=pca_dim, random_state=42).fit(X[~is_val])
        X = pca.transform(X)
        print(f"PCA {AF.EMBED_DIM} -> {pca_dim} "
              f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance, {time.time()-t0:.0f}s)")

    scaler = StandardScaler().fit(X[~is_val])
    Xs = scaler.transform(X)

    artists = np.sort(meta.loc[tracks, "artist_type"].unique())
    genres  = np.sort(meta.loc[tracks, "genre"].unique())
    a_idx = {a: i for i, a in enumerate(artists)}
    g_idx = {g: i for i, g in enumerate(genres)}
    ya = meta.loc[tracks, "artist_type"].map(a_idx).to_numpy()
    yg = meta.loc[tracks, "genre"].map(g_idx).to_numpy()

    print("training...", flush=True)
    t0 = time.time()
    clf_a = LogisticRegression(max_iter=1000, n_jobs=-1).fit(Xs[~is_val], ya[~is_val])
    clf_g = LogisticRegression(max_iter=1000, n_jobs=-1).fit(Xs[~is_val], yg[~is_val])
    print(f"  {time.time()-t0:.0f}s")

    Pa = clf_a.predict_proba(Xs[is_val]); Pg = clf_g.predict_proba(Xs[is_val])
    ta, tg = ya[is_val], yg[is_val]
    ch_a = float((pd.Series(ta).value_counts(normalize=True) ** 2).sum())
    ch_g = float((pd.Series(tg).value_counts(normalize=True) ** 2).sum())

    g_of_a = np.array([g_idx[taxonomy[a]["genre"]] for a in artists])
    logPg = np.full((len(ta), len(genres)), -30.0)
    logPg[:, clf_g.classes_] = np.log(Pg + 1e-12)
    joint = np.log(Pa + 1e-12) + logPg[:, g_of_a[clf_a.classes_]]
    joint_a = clf_a.classes_[joint.argmax(1)]

    # top-1 is the JOINT pick, because that is the artist inference_engine.classify
    # returns and the UI shows. Scoring Pa.argmax instead reported a system nobody runs
    # (measured: 15.35% argmax vs 16.16% joint on the same split, +0.81p / 1.2 SE).
    # argmax is still printed alongside, so the two can never silently diverge again.
    a_top1 = float((clf_a.classes_[joint.argmax(1)] == ta).mean())
    a_top1_argmax = float((clf_a.classes_[Pa.argmax(1)] == ta).mean())
    a_top5 = float(np.mean([ta[i] in clf_a.classes_[np.argsort(-Pa[i])[:5]]
                            for i in range(len(ta))]))
    g_direct = float((clf_g.classes_[Pg.argmax(1)] == tg).mean())
    g_joint = float((g_of_a[joint_a] == tg).mean())

    print(f"\n=== {kind} on {is_val.sum():,} held-out tracks ===")
    print(f"  artist  top-1 {a_top1*100:5.2f}%  top-5 {a_top5*100:5.2f}%  "
          f"chance {ch_a*100:.2f}%  -> {a_top1/ch_a:.1f}x")
    print(f"          (top-1 is the joint pick, what inference serves; "
          f"plain argmax would read {a_top1_argmax*100:.2f}%)")
    print(f"  genre   direct {g_direct*100:5.2f}%  joint {g_joint*100:5.2f}%  "
          f"chance {ch_g*100:.2f}%  -> {g_joint/ch_g:.1f}x")
    print("\n  reference: classical MFCC 9.45%/24.13%/17.32%, EfficientNetB3 below chance")

    threshold, table = calibrate_threshold(Pa, ta, clf_a.classes_, len(artists),
                                           joint.argmax(1))

    joblib.dump({
        "kind": kind, "pca": pca, "scaler": scaler,
        # Guards against the catalogue and the query using different encoders.
        "encoder_id": (AF.encoder_id() if kind.startswith("panns") else "classical"),
        "encoder_hash": (AF.encoder_hash() if kind.startswith("panns") else None),
        "clf_artist": clf_a, "clf_genre": clf_g,
        "artists": artists, "genres": genres, "genre_of_artist": g_of_a,
        "embed_dim": X.shape[1], "duration": AF.DURATION,
        "confidence_threshold": threshold, "calibration_table": table,
        "metrics": {"artist_top1": a_top1, "artist_top1_argmax": a_top1_argmax,
                    "artist_top5": a_top5,
                    "genre_joint": g_joint, "n_held_out": int(is_val.sum())},
    }, MODEL_PATH)

    Xn = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9)
    np.savez_compressed(CATALOG_NPZ, X=Xn.astype(np.float32), tracks=tracks)
    pd.DataFrame({"track": tracks,
                  "artist_type": meta.loc[tracks, "artist_type"].to_numpy(),
                  "genre": meta.loc[tracks, "genre"].to_numpy()}).to_csv(CATALOG_CSV, index=False)

    print(f"\nwrote {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1e6:.1f} MB)")
    print(f"wrote {CATALOG_NPZ} ({os.path.getsize(CATALOG_NPZ)/1e6:.1f} MB, {Xn.shape})")


if __name__ == "__main__":
    main()
