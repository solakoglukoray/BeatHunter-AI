"""
DENEY — PANNs (AudioSet) embedding'leri klasik özelliklere karşı.

Hipotez: EfficientNetB3'ün başarısız olmasının sebebi yanlış domain'den transfer
(fotoğraf → spektrogram). PANNs doğru domain'den geliyor (AudioSet, 2M ses klibi) ve
MFCC özetinden daha zengin bir temsil taşımalı.

⚠️ BU BİR DENEY. Production'daki A1 sistemi (`audio_features.py` + `train_classic.py`)
bu dosyadan etkilenmez ve çalışır durumda kalır.

KARŞILAŞTIRMA TASARIMI
──────────────────────
Mevcut production sayıları (artist 16.7x, genre 9.7x) **194 sanatçılı tam veriden**.
Küçük bir alt kümede şans seviyesi tamamen farklıdır (20 sanatçı → %5, 194 → %0.5),
dolayısıyla o sayılarla doğrudan kıyaslamak yanıltıcı olur.

Bu yüzden script, klasik özellikleri **aynı alt küme, aynı split, aynı modellerle**
yeniden ölçer. Tek anlamlı karşılaştırma budur; tam-veri sayıları sadece bağlam için
yazdırılır.

    python panns_features_experiment.py --artists 20 --per-artist 20
"""

import os, sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, librosa, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

PANNS_SR   = 32000        # PANNs is trained at 32 kHz — not the 22.05 kHz used elsewhere
DURATION   = 30
CKPT       = os.path.expanduser("~/panns_data/Cnn14_mAP=0.431.pth")
CLASSIC_NPZ = "models/classic_features.npz"
TRACK_META  = "models/vector_meta_track.csv"
OUT         = os.path.expanduser("~/bench_out/panns_experiment.json")


def load_panns(device):
    from panns_inference.models import Cnn14
    m = Cnn14(sample_rate=PANNS_SR, window_size=1024, hop_size=320, mel_bins=64,
              fmin=50, fmax=14000, classes_num=527)
    m.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["model"])
    return m.eval().to(device)


def panns_embeddings(paths, device, batch=8):
    """2048-dim clip embedding per track. Audio load is the slow part, not the net."""
    model = load_panns(device)
    out, ok, t0 = [], [], time.time()
    buf, buf_idx = [], []

    def flush():
        if not buf:
            return
        n = max(len(w) for w in buf)
        x = np.zeros((len(buf), n), dtype=np.float32)
        for i, w in enumerate(buf):
            x[i, :len(w)] = w
        with torch.no_grad():
            emb = model(torch.from_numpy(x).to(device))["embedding"].cpu().numpy()
        out.append(emb); ok.extend(buf_idx)
        buf.clear(); buf_idx.clear()

    for i, p in enumerate(paths):
        try:
            y, _ = librosa.load(p, sr=PANNS_SR, duration=DURATION)
            if len(y) >= PANNS_SR * 5:
                buf.append(y); buf_idx.append(i)
        except Exception:
            pass
        if len(buf) >= batch:
            flush()
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(paths)}  {el:.0f}s  eta {el/(i+1)*(len(paths)-i-1):.0f}s",
                  flush=True)
    flush()
    return np.concatenate(out), np.array(ok)


def evaluate(X, y_idx, n_classes, label, results):
    """Aynı protokol: z-normalize, stratified %25 test, LogReg + k-NN."""
    tr, te = train_test_split(np.arange(len(y_idx)), test_size=0.25,
                              random_state=42, stratify=y_idx)
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)
    Xn = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9)

    chance = float((pd.Series(y_idx[te]).value_counts(normalize=True) ** 2).sum())

    lr = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Xs[tr], y_idx[tr])
    P = lr.predict_proba(Xs[te])
    lr_top1 = float((lr.classes_[P.argmax(1)] == y_idx[te]).mean())
    lr_top5 = float(np.mean([y_idx[te][i] in lr.classes_[np.argsort(-P[i])[:5]]
                             for i in range(len(te))]))

    S = Xn[te] @ Xn[tr].T
    nn = y_idx[tr][np.argsort(-S, axis=1)[:, :5]]
    knn = float((np.array([np.bincount(r).argmax() for r in nn]) == y_idx[te]).mean())

    print(f"  {label:26s} LogReg top1 %{lr_top1*100:5.2f} top5 %{lr_top5*100:5.2f} "
          f"| k-NN %{knn*100:5.2f} | sans %{chance*100:5.2f} -> {lr_top1/chance:5.1f}x")
    results[label] = {"logreg_top1": lr_top1, "logreg_top5": lr_top5,
                      "knn5": knn, "chance": chance, "lift": lr_top1 / chance,
                      "dim": int(X.shape[1])}


PANNS_CACHE = "models/panns_features.npz"


def extract_full(device, batch=16, save_every=1000):
    """
    All 14,235 tracks, checkpointed. Roughly 32 minutes.

    Partial results are written every save_every tracks and reloaded on restart, so a
    power cut costs minutes rather than the whole run — this laptop travels and has died
    mid-job before.
    """
    meta = pd.read_csv(TRACK_META)
    meta["path"] = meta["track"].map(lambda t: f"dataset/audio/{t}.mp3")
    meta = meta[meta["path"].map(os.path.exists)].reset_index(drop=True)

    done_tracks, done_emb = {}, []
    if os.path.exists(PANNS_CACHE):
        z = np.load(PANNS_CACHE, allow_pickle=True)
        done_tracks = {t: i for i, t in enumerate(z["tracks"])}
        done_emb = list(z["X"])
        print(f"resuming: {len(done_tracks):,} tracks already embedded", flush=True)

    todo = meta[~meta["track"].isin(done_tracks)].reset_index(drop=True)
    print(f"to extract: {len(todo):,} of {len(meta):,}", flush=True)
    if len(todo) == 0:
        z = np.load(PANNS_CACHE, allow_pickle=True)
        return z["X"], z["tracks"]

    model = load_panns(device)
    tracks_out = list(done_tracks.keys())
    emb_out = done_emb
    buf, buf_tracks, t0 = [], [], time.time()

    def flush():
        if not buf:
            return
        n = max(len(w) for w in buf)
        x = np.zeros((len(buf), n), dtype=np.float32)
        for i, w in enumerate(buf):
            x[i, :len(w)] = w
        with torch.no_grad():
            e = model(torch.from_numpy(x).to(device))["embedding"].cpu().numpy()
        emb_out.extend(e); tracks_out.extend(buf_tracks)
        buf.clear(); buf_tracks.clear()

    def save():
        tmp = PANNS_CACHE + ".tmp.npz"
        np.savez_compressed(tmp, X=np.stack(emb_out).astype(np.float32),
                            tracks=np.array(tracks_out))
        os.replace(tmp, PANNS_CACHE)      # atomic

    for i, (t, p) in enumerate(zip(todo["track"], todo["path"]), 1):
        try:
            y, _ = librosa.load(p, sr=PANNS_SR, duration=DURATION)
            if len(y) >= PANNS_SR * 5:
                buf.append(y); buf_tracks.append(t)
        except Exception:
            pass
        if len(buf) >= batch:
            flush()
        if i % 200 == 0:
            el = time.time() - t0
            print(f"  {i:,}/{len(todo):,}  {el:.0f}s  eta {el/i*(len(todo)-i):.0f}s", flush=True)
        if i % save_every == 0:
            flush(); save()
            print(f"  [checkpoint: {len(tracks_out):,} tracks saved]", flush=True)

    flush(); save()
    print(f"extraction done: {len(tracks_out):,} tracks ({time.time()-t0:.0f}s)", flush=True)
    return np.stack(emb_out).astype(np.float32), np.array(tracks_out)


def production_split(tracks):
    """The exact held-out tracks the 16.7x / 9.7x production numbers were measured on."""
    from sklearn.model_selection import GroupShuffleSplit
    sl = pd.read_csv("dataset_processed/metadata_processed.csv")
    sl["track"] = sl["filename"].str.split("_slice").str[0]
    _, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                   random_state=42).split(sl, groups=sl["track"]))
    val = set(sl.iloc[va]["track"].unique())
    return np.array([t in val for t in tracks])


def evaluate_production(X, df, label, results, taxonomy=None):
    """Same protocol as train_classic.py: production split, LogReg, joint-decoded genre."""
    is_val = production_split(df["track"].to_numpy())
    sc = StandardScaler().fit(X[~is_val])
    Xs = sc.transform(X)

    artists = np.sort(df["artist_type"].unique())
    genres = np.sort(df["genre"].unique())
    a_idx = {a: i for i, a in enumerate(artists)}
    g_idx = {g: i for i, g in enumerate(genres)}
    ya = df["artist_type"].map(a_idx).to_numpy()
    yg = df["genre"].map(g_idx).to_numpy()

    clf_a = LogisticRegression(max_iter=1000, n_jobs=-1).fit(Xs[~is_val], ya[~is_val])
    clf_g = LogisticRegression(max_iter=1000, n_jobs=-1).fit(Xs[~is_val], yg[~is_val])
    Pa = clf_a.predict_proba(Xs[is_val]); Pg = clf_g.predict_proba(Xs[is_val])
    ta, tg = ya[is_val], yg[is_val]

    ch_a = float((pd.Series(ta).value_counts(normalize=True) ** 2).sum())
    ch_g = float((pd.Series(tg).value_counts(normalize=True) ** 2).sum())

    a_top1 = float((clf_a.classes_[Pa.argmax(1)] == ta).mean())
    a_top5 = float(np.mean([ta[i] in clf_a.classes_[np.argsort(-Pa[i])[:5]]
                            for i in range(len(ta))]))
    g_top1 = float((clf_g.classes_[Pg.argmax(1)] == tg).mean())

    # joint decoding, as production does it
    g_joint = None
    if taxonomy is not None:
        g_of_a = np.array([g_idx[taxonomy[a]["genre"]] for a in artists])
        logPg = np.full((len(ta), len(genres)), -30.0)
        logPg[:, clf_g.classes_] = np.log(Pg + 1e-12)
        joint = np.log(Pa + 1e-12) + logPg[:, g_of_a[clf_a.classes_]]
        g_joint = float((g_of_a[clf_a.classes_[joint.argmax(1)]] == tg).mean())

    print(f"  {label:22s} artist top1 %{a_top1*100:5.2f} ({a_top1/ch_a:5.1f}x) "
          f"top5 %{a_top5*100:5.2f} | genre %{g_top1*100:5.2f} ({g_top1/ch_g:4.1f}x)"
          + (f" joint %{g_joint*100:5.2f}" if g_joint else ""), flush=True)
    results[label] = {"artist_top1": a_top1, "artist_top5": a_top5,
                      "artist_lift": a_top1 / ch_a, "genre_top1": g_top1,
                      "genre_lift": g_top1 / ch_g, "genre_joint": g_joint,
                      "dim": int(X.shape[1]), "n_val": int(is_val.sum())}


def run_full():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | torch {torch.__version__}\n")

    Xp, tracks = extract_full(device)
    z = np.load(CLASSIC_NPZ, allow_pickle=True)
    cls_map = {t: i for i, t in enumerate(z["tracks"])}

    keep = [i for i, t in enumerate(tracks) if t in cls_map]
    Xp = Xp[keep]; tracks = tracks[keep]
    Xc = z["X"][[cls_map[t] for t in tracks]]

    meta = pd.read_csv(TRACK_META).set_index("track")
    df = pd.DataFrame({"track": tracks,
                       "artist_type": meta.loc[tracks, "artist_type"].to_numpy(),
                       "genre": meta.loc[tracks, "genre"].to_numpy()})
    taxonomy = json.load(open("models/taxonomy.json", encoding="utf-8"))

    print(f"\n{len(df):,} tracks | PANNs {Xp.shape} | classical {Xc.shape}")
    print(f"{df['artist_type'].nunique()} artists, {df['genre'].nunique()} genres\n")
    print("=" * 88)
    print("FULL SCALE — production split (GroupShuffleSplit by track, random_state=42)")
    print("=" * 88)
    results = {}
    evaluate_production(Xc, df, "classical 102-dim", results, taxonomy)
    evaluate_production(Xp, df, "PANNs 2048-dim", results, taxonomy)
    print(f"\nheld-out tracks: {results['PANNs 2048-dim']['n_val']:,}")
    print("production reference: artist 16.7x (top-1 9.45%), genre 9.7x (17.32% joint)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT.replace(".json", "_full.json"), "w"), indent=2)
    print(f"\nresults: {OUT.replace('.json', '_full.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artists", type=int, default=20)
    ap.add_argument("--per-artist", type=int, default=20)
    ap.add_argument("--full", action="store_true",
                    help="all 14,235 tracks, production split, ~32 min (checkpointed)")
    a = ap.parse_args()

    if a.full:
        run_full()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"cihaz: {device} | torch {torch.__version__}\n")

    # --- alt kume: klasik ozellikleri ONBELLEKTE olan parcalardan sec, boylece
    #     iki temsil BIREBIR ayni parcalar uzerinde karsilastirilir ---
    z = np.load(CLASSIC_NPZ, allow_pickle=True)
    Xc_all, tracks_all = z["X"], z["tracks"]
    meta = pd.read_csv(TRACK_META).set_index("track").loc[pd.Index(tracks_all)]

    # Built column-by-column from arrays rather than via reset_index(): .loc[] on a
    # pd.Index drops the index NAME, so reset_index() yields "index" not "track".
    # The same trap already bit train_classic.py once.
    full = pd.DataFrame({
        "track": tracks_all,
        "artist_type": meta["artist_type"].to_numpy(),
        "genre": meta["genre"].to_numpy(),
        "row": np.arange(len(Xc_all)),
    })
    top = full["artist_type"].value_counts().head(a.artists).index
    df = (full[full["artist_type"].isin(top)]
          .groupby("artist_type", group_keys=False).head(a.per_artist).copy())
    df["path"] = df["track"].map(lambda t: f"dataset/audio/{t}.mp3")
    df = df[df["path"].map(os.path.exists)].reset_index(drop=True)

    print(f"alt kume: {len(df)} parca / {df['artist_type'].nunique()} artist "
          f"/ {df['genre'].nunique()} genre\n")

    print("PANNs embedding'leri cikariliyor...")
    t0 = time.time()
    Xp, ok = panns_embeddings(df["path"].tolist(), device)
    df = df.iloc[ok].reset_index(drop=True)
    Xc = Xc_all[df["row"].to_numpy()]
    print(f"PANNs {Xp.shape} | klasik {Xc.shape} | {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/len(df)*1000:.0f} ms/parca)\n")

    results = {}
    for target, col in (("ARTIST", "artist_type"), ("GENRE", "genre")):
        classes, yi = np.unique(df[col].to_numpy(), return_inverse=True)
        print(f"=== {target} ({len(classes)} sinif) ===")
        evaluate(Xc, yi, len(classes), f"klasik 102-dim ({target})", results)
        evaluate(Xp, yi, len(classes), f"PANNs 2048-dim ({target})", results)
        print()

    print("=" * 72)
    print("BAGLAM: tam veride (194 artist, 14,235 parca) klasik yol")
    print("  artist 16.7x sans (top-1 %9.45)  |  genre 9.7x (%17.32)")
    print("  ^ farkli sinif sayisi ve referans kumesi — yukaridakilerle DOGRUDAN")
    print("    kiyaslanamaz; yukaridaki iki satir ayni alt kumede olculdugu icin")
    print("    kiyaslanabilir olan onlar.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nsonuc: {OUT}")


if __name__ == "__main__":
    main()
