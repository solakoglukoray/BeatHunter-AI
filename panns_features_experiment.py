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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artists", type=int, default=20)
    ap.add_argument("--per-artist", type=int, default=20)
    a = ap.parse_args()

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
