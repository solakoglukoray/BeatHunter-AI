"""
DENEY C — fine-tune edilmiş encoder'ı DONUK özellik çıkarıcı olarak kullan.

Soru: fine-tuning'in kazancı encoder TEMSILINDE mi, yoksa üzerine eğitilen HEAD'de mi?

Üç satır aynı held-out parçalarda (2,847 track, GroupShuffleSplit random_state=42):

  A. donuk PANNs (orijinal AudioSet) + PCA-128 + LogReg   top-1 %12.96  top-5 %32.03
  B. fine-tune edilmiş encoder DONUK + PCA-128 + LogReg   <- bu deney ölçüyor
  C. tam fine-tune, kendi head'iyle                        top-1 %15.49  top-5 %40.46

  B ≈ C  ise kazanç TEMSILDE  -> encoder'ı özellik çıkarıcı olarak kullanmak yeter,
                                 mevcut PCA+LogReg mimarisi korunur, inference değişmez
  B ≈ A  ise kazanç HEAD'de   -> temsil aslında iyileşmemiş, sadece sınıflandırıcı
                                 doğrudan ağın içinde eğitildiği için daha iyi

⚠️ Bu bir DENEY. Production artefaktlarına (classic_model.joblib, classic_catalog.npz)
yazmaz; sadece ölçer.

    python panns_ft_frozen_experiment.py --extract-encoder   # best.pt -> ince encoder
    python panns_ft_frozen_experiment.py                     # embed + ölç (~50 dk)
"""

import os, sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, librosa, torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

BEST_CKPT   = "models/panns_finetune_best.pt"
SLIM_ENCODER = "models/panns_ft_encoder.pt"
FT_CACHE    = "models/panns_ft_features.npz"
TRACK_META  = "models/vector_meta_track.csv"
SLICE_META  = "dataset_processed/metadata_processed.csv"
TAXONOMY    = "models/taxonomy.json"
OUT         = os.path.expanduser("~/bench_out/panns_ft_frozen.json")

SR, DURATION, BATCH = 32000, 30, 8


def extract_encoder():
    """
    best.pt (971 MB: model + optimizer + scheduler + RNG) -> just the trunk weights.

    The heads are dropped deliberately: this experiment asks what the ENCODER learned,
    so reusing its heads would answer a different question.
    """
    ck = torch.load(BEST_CKPT, map_location="cpu", weights_only=False)
    trunk = {k[len("trunk."):]: v for k, v in ck["model"].items() if k.startswith("trunk.")}
    heads = [k for k in ck["model"] if not k.startswith("trunk.")]
    torch.save({"trunk": trunk, "epoch": ck["epoch"], "best": ck["best"]}, SLIM_ENCODER)
    print(f"best.pt {os.path.getsize(BEST_CKPT)/1e6:.0f} MB "
          f"-> {SLIM_ENCODER} {os.path.getsize(SLIM_ENCODER)/1e6:.0f} MB")
    print(f"  trunk tensors: {len(trunk)} | dropped heads: {heads}")
    print(f"  from epoch {ck['epoch']}, best artist top-1 {ck['best']*100:.2f}%")


def load_ft_encoder(device):
    from panns_inference.models import Cnn14
    m = Cnn14(sample_rate=SR, window_size=1024, hop_size=320, mel_bins=64,
              fmin=50, fmax=14000, classes_num=527)
    m.load_state_dict(torch.load(SLIM_ENCODER, map_location="cpu",
                                 weights_only=False)["trunk"])
    # eval(): BatchNorm uses the running statistics the fine-tune updated, and the
    # internal spec_augmenter stays off. This is a frozen extractor, not training.
    return m.eval().to(device)


def embed_all(device, save_every=1000):
    meta = pd.read_csv(TRACK_META)
    meta["path"] = meta["track"].map(lambda t: f"dataset/audio/{t}.mp3")
    meta = meta[meta["path"].map(os.path.exists)].reset_index(drop=True)

    tracks_out, emb_out = [], []
    if os.path.exists(FT_CACHE):
        z = np.load(FT_CACHE, allow_pickle=True)
        tracks_out, emb_out = list(z["tracks"]), list(z["X"])
        print(f"resuming from {len(tracks_out):,} cached embeddings")
    todo = meta[~meta["track"].isin(set(tracks_out))].reset_index(drop=True)
    if len(todo) == 0:
        return np.stack(emb_out).astype(np.float32), np.array(tracks_out)

    print(f"embedding {len(todo):,} of {len(meta):,} tracks (30s, bf16)...", flush=True)
    model = load_ft_encoder(device)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    buf, buf_t, t0 = [], [], time.time()

    def flush():
        if not buf:
            return
        n = max(len(w) for w in buf)
        x = np.zeros((len(buf), n), dtype=np.float32)
        for i, w in enumerate(buf):
            x[i, :len(w)] = w
        xt = torch.from_numpy(x).to(device)
        with torch.no_grad():
            if use_bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    e = model(xt)["embedding"]
            else:
                e = model(xt)["embedding"]
        emb_out.extend(e.float().cpu().numpy())
        tracks_out.extend(buf_t)
        buf.clear(); buf_t.clear()

    def save():
        tmp = FT_CACHE + ".tmp.npz"
        np.savez_compressed(tmp, X=np.stack(emb_out).astype(np.float32),
                            tracks=np.array(tracks_out))
        os.replace(tmp, FT_CACHE)

    for i, (t, p) in enumerate(zip(todo["track"], todo["path"]), 1):
        try:
            y, _ = librosa.load(p, sr=SR, duration=DURATION)
            if len(y) >= SR * 5:
                buf.append(y); buf_t.append(t)
        except Exception:
            pass
        if len(buf) >= BATCH:
            flush()
        if i % 500 == 0:
            el = time.time() - t0
            print(f"  {i:,}/{len(todo):,}  {el:.0f}s  eta {el/i*(len(todo)-i):.0f}s", flush=True)
        if i % save_every == 0:
            flush(); save()
    flush(); save()
    print(f"done ({time.time()-t0:.0f}s)")
    return np.stack(emb_out).astype(np.float32), np.array(tracks_out)


def measure(X, tracks, label, results, pca_dim=128):
    """Exactly the production pipeline: PCA -> scale -> LogReg -> joint decode."""
    meta = pd.read_csv(TRACK_META).set_index("track")
    taxonomy = json.load(open(TAXONOMY, encoding="utf-8"))

    sl = pd.read_csv(SLICE_META)
    sl["track"] = sl["filename"].str.split("_slice").str[0]
    _, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                   random_state=42).split(sl, groups=sl["track"]))
    held = set(sl.iloc[va]["track"].unique())
    is_val = np.array([t in held for t in tracks])

    artists = np.sort(meta.loc[tracks, "artist_type"].unique())
    genres = np.sort(meta.loc[tracks, "genre"].unique())
    a_idx = {a: i for i, a in enumerate(artists)}
    g_idx = {g: i for i, g in enumerate(genres)}
    ya = meta.loc[tracks, "artist_type"].map(a_idx).to_numpy()
    yg = meta.loc[tracks, "genre"].map(g_idx).to_numpy()
    g_of_a = np.array([g_idx[taxonomy[a]["genre"]] for a in artists])

    pca = PCA(n_components=pca_dim, random_state=42).fit(X[~is_val])
    Xp = pca.transform(X)
    sc = StandardScaler().fit(Xp[~is_val])
    Xs = sc.transform(Xp)

    clf_a = LogisticRegression(max_iter=1000, n_jobs=-1).fit(Xs[~is_val], ya[~is_val])
    clf_g = LogisticRegression(max_iter=1000, n_jobs=-1).fit(Xs[~is_val], yg[~is_val])
    Pa, Pg = clf_a.predict_proba(Xs[is_val]), clf_g.predict_proba(Xs[is_val])
    ta, tg = ya[is_val], yg[is_val]

    a1 = float((clf_a.classes_[Pa.argmax(1)] == ta).mean())
    a5 = float(np.mean([ta[i] in clf_a.classes_[np.argsort(-Pa[i])[:5]]
                        for i in range(len(ta))]))
    logPg = np.full((len(ta), len(genres)), -30.0)
    logPg[:, clf_g.classes_] = np.log(Pg + 1e-12)
    joint = np.log(Pa + 1e-12) + logPg[:, g_of_a[clf_a.classes_]]
    gj = float((g_of_a[clf_a.classes_[joint.argmax(1)]] == tg).mean())

    print(f"  {label:34s} top-1 %{a1*100:5.2f}  top-5 %{a5*100:5.2f}  genre %{gj*100:5.2f}"
          f"  (var %{pca.explained_variance_ratio_.sum()*100:.1f})")
    results[label] = {"artist_top1": a1, "artist_top5": a5, "genre_joint": gj,
                      "n_val": int(is_val.sum())}
    return Pa, ta, clf_a.classes_, len(artists)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-encoder", action="store_true")
    a = ap.parse_args()

    if a.extract_encoder or not os.path.exists(SLIM_ENCODER):
        extract_encoder()
        if a.extract_encoder:
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")
    Xft, tracks_ft = embed_all(device)

    # Original frozen PANNs embeddings, restricted to the same tracks
    zb = np.load("models/panns_features.npz", allow_pickle=True)
    bmap = {t: i for i, t in enumerate(zb["tracks"])}
    keep = [i for i, t in enumerate(tracks_ft) if t in bmap]
    Xft, tracks = Xft[keep], tracks_ft[keep]
    Xbase = zb["X"][[bmap[t] for t in tracks]]

    print(f"\n{len(tracks):,} tracks | fine-tuned {Xft.shape} | baseline {Xbase.shape}")
    print("=" * 90)
    print("DENEY C — kazanç temsilde mi, head'de mi?")
    print("=" * 90)
    results = {}
    measure(Xbase, tracks, "A. donuk PANNs (AudioSet)", results)
    Pa, ta, cls, n_art = measure(Xft, tracks, "B. fine-tuned encoder, DONUK", results)
    print(f"  {'C. tam fine-tune (kendi head)':34s} top-1 %15.49  top-5 %40.46  genre %27.96")

    print("\n" + "=" * 90)
    b = results["B. fine-tuned encoder, DONUK"]
    A = results["A. donuk PANNs (AudioSet)"]
    n = b["n_val"]
    for name, key, c_val in (("top-1", "artist_top1", 0.1549),
                             ("top-5", "artist_top5", 0.4046),
                             ("genre", "genre_joint", 0.2796)):
        av, bv = A[key], b[key]
        span = c_val - av
        share = (bv - av) / span if abs(span) > 1e-9 else float("nan")
        se = np.sqrt(bv * (1 - bv) / n)
        print(f"  {name}: A %{av*100:5.2f} -> B %{bv*100:5.2f} -> C %{c_val*100:5.2f}"
              f"   B, A-C aralığının %{share*100:.0f}'ini kapatıyor   (SE ~%{se*100:.2f})")

    # Threshold calibration for B, since its confidence distribution is its own
    print("\n  B icin esik kalibrasyonu (eski modellerden kopyalanamaz):")
    conf = Pa.max(1) * n_art
    right = cls[Pa.argmax(1)] == ta
    for thr in (25, 50, 100, 150, 200):
        sel = conf >= thr
        if sel.sum() >= 30:
            print(f"    >= {thr:4d}x   gosterilen %{sel.mean()*100:5.1f}   dogruluk %{right[sel].mean()*100:5.1f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nsonuc: {OUT}")


if __name__ == "__main__":
    main()
