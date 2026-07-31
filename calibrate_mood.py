"""
Mood/energy calibration — derives the axis thresholds from the corpus, then checks them.

WHY CALIBRATION AND NOT CONSTANTS
─────────────────────────────────
The label this replaces was assigned by hand per artist in config.py and had no
relationship to the audio (Cohen d +0.023 / +0.094, k-NN 1.4x chance, three independent
measurements). Writing `if rms > 0.1: "Energetic"` would repeat that mistake in a new
costume: a number nobody measured, defended by nothing.

So every threshold here is a percentile of the actual corpus, and a label means "this
track sits in the upper or lower third of 14,235 tracks on this axis". That is checkable,
and it is honest about being relative — there is no absolute scale for "energetic".

WHAT THIS SCRIPT VERIFIES, NOT JUST COMPUTES
────────────────────────────────────────────
1. Axis independence. If two axes correlate strongly they are one axis with two names.
2. Signal agreement within an axis. If the three signals feeding "energy" disagree with
   each other, averaging them produces mush.
3. Genre sanity. Genre is NOT an input, so if lo-fi lands chill and drill lands
   aggressive, the axes are tracking something real. If they do not, say so.

    python calibrate_mood.py                 # sample, calibrate, verify
    python calibrate_mood.py --tracks 3000
"""

import os, json, time, argparse, warnings
warnings.filterwarnings("ignore")

# One BLAS thread per worker. Without this, 12 processes each spawn 32 threads and spend
# most of their time fighting over cores instead of extracting features.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np, pandas as pd
from multiprocessing import Pool

import audio_features as AF

CALIB_PATH = "models/mood_calibration.json"
CACHE_PATH = "models/mood_signals.npz"
TRACK_META = "models/vector_meta_track.csv"
SIGNALS = AF.MOOD_SIGNAL_NAMES

# Genres whose character is not in dispute, one contrast PER AXIS. They are never
# inputs — they exist only to ask whether an axis lands where a listener would put it.
#
# One shared contrast for all three axes does not work, and finding that out cost a
# round: drill is aggressive but sparse, so a single "mellow vs aggressive" split marks
# it high and then penalises the energy axis for agreeing.
AXIS_CONTRAST = {
    "energy": (["Lofi Hip-Hop", "Chillhop", "Trip Hop", "Neo Soul",
                "Chopped and Screwed", "Cloud Rap", "Jazz Rap"],
               ["Crunk", "Bounce", "Hyphy", "Phonk", "Snap Music", "Chopper",
                "Digicore", "Rage", "Reggaeton"]),
    "brightness": (["Chopped and Screwed", "Cloud Rap", "Memphis Rap", "Horrorcore",
                    "Slab Music", "Road Rap"],
                   ["Digicore", "Pop Rap", "Reggaeton", "Chillhop", "Hipster Hop",
                    "Comedy Hip-Hop"]),
    "aggression": (["Lofi Hip-Hop", "Chillhop", "Jazz Rap", "Neo Soul", "Hip-Hop Soul",
                    "Trip Hop", "G-Funk"],
                   ["Crunk", "Trap Metal", "Nu Metal", "Hardcore Hip-Hop", "Rage",
                    "Horrorcore", "Punk Rap", "Industrial Hip-Hop", "Grime"]),
}


def _work(args):
    track, path = args
    return track, AF.extract_mood_from_file(path)


def collect(n_tracks, workers=12):
    if os.path.exists(CACHE_PATH):
        z = np.load(CACHE_PATH, allow_pickle=True)
        # A cache written by a different signal set is worse than no cache: the columns
        # still line up by position, so it would silently calibrate the new axes against
        # the old measurements. Refuse it instead.
        cached_ver = int(z["version"][()]) if "version" in z else 0
        if cached_ver != AF.MOOD_SIGNAL_VERSION:
            print(f"cache is signal version {cached_ver}, code is "
                  f"{AF.MOOD_SIGNAL_VERSION} — re-extracting from scratch")
        elif len(z["tracks"]) >= n_tracks:
            print(f"cached mood signals for {len(z['tracks']):,} tracks")
            return pd.DataFrame(z["X"], columns=SIGNALS), z["tracks"]

    meta = pd.read_csv(TRACK_META)
    meta["path"] = meta["track"].map(lambda t: f"dataset/audio/{t}.mp3")
    meta = meta[meta["path"].map(os.path.exists)].reset_index(drop=True)
    rng = np.random.default_rng(42)
    sample = meta.iloc[rng.choice(len(meta), min(n_tracks, len(meta)), replace=False)]

    print(f"extracting mood signals for {len(sample):,} tracks ({workers} workers)...",
          flush=True)
    t0, rows, tracks = time.time(), [], []
    with Pool(workers) as pool:
        for i, (t, s) in enumerate(pool.imap_unordered(
                _work, list(zip(sample["track"], sample["path"])), chunksize=4), 1):
            if s is not None:
                rows.append([s[k] for k in SIGNALS]); tracks.append(t)
            if i % 250 == 0:
                el = time.time() - t0
                print(f"  {i:,}/{len(sample):,}  {el:.0f}s  eta {el/i*(len(sample)-i):.0f}s",
                      flush=True)

    X = np.array(rows, dtype=np.float32)
    np.savez_compressed(CACHE_PATH, X=X, tracks=np.array(tracks),
                        version=AF.MOOD_SIGNAL_VERSION,
                        signals=np.array(SIGNALS))
    print(f"done ({time.time()-t0:.0f}s)\n")
    return pd.DataFrame(X, columns=SIGNALS), np.array(tracks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=2000)
    a = ap.parse_args()

    df, tracks = collect(a.tracks)
    meta = pd.read_csv(TRACK_META).set_index("track").loc[pd.Index(tracks)]

    # 101 edges = percentile lookup table per signal. Storing the curve rather than a
    # mean/std keeps it robust to the skew these features have (tempo especially).
    percentiles = {s: np.percentile(df[s], np.arange(0, 101)).tolist() for s in SIGNALS}
    calib = {"percentiles": percentiles, "n_tracks": int(len(df)), "signals": SIGNALS}

    # Pass 1: score the corpus with no cuts, so we can measure where the thirds actually
    # fall. An axis score is a mean of percentiles and therefore clusters around 0.5 —
    # cutting at a literal 1/3 and 2/3 would label almost everything "Moderate".
    _s0 = pd.DataFrame([AF.axis_scores(dict(r), calib) for _, r in df.iterrows()])
    calib["axis_cuts"] = {ax: [float(np.percentile(_s0[ax], 100 / 3)),
                               float(np.percentile(_s0[ax], 200 / 3))]
                          for ax in _s0.columns}

    print("=" * 78)
    print("1) SINYAL DAGILIMLARI (korpus)")
    print("=" * 78)
    print(f"{'sinyal':16s} {'%10':>10s} {'medyan':>10s} {'%90':>10s}")
    for s in SIGNALS:
        q = np.percentile(df[s], [10, 50, 90])
        print(f"{s:16s} {q[0]:10.3f} {q[1]:10.3f} {q[2]:10.3f}")

    scores = _s0

    print("\n" + "=" * 78)
    print("2a) EKSEN KESIM NOKTALARI ve etiket dagilimi")
    print("=" * 78)
    print("  (sabit 1/3-2/3 kullanilsa cogu parca ortada toplanirdi — asagidaki")
    print("   'sabit kesimle' sutunu bunu gosteriyor)")
    for ax, (lo, hi) in calib["axis_cuts"].items():
        low_name, mid_name, high_name = AF.MOOD_AXES[ax]
        v = scores[ax].to_numpy()
        emp = [(v < lo).mean(), ((v >= lo) & (v < hi)).mean(), (v >= hi).mean()]
        fix = [(v < 1/3).mean(), ((v >= 1/3) & (v < 2/3)).mean(), (v >= 2/3).mean()]
        print(f"\n  {ax:11s} kesim [{lo:.3f}, {hi:.3f}]   aralik {v.min():.2f}-{v.max():.2f}")
        print(f"    kalibre : {low_name} %{emp[0]*100:.0f} / {mid_name} %{emp[1]*100:.0f} / {high_name} %{emp[2]*100:.0f}")
        print(f"    sabit   : {low_name} %{fix[0]*100:.0f} / {mid_name} %{fix[1]*100:.0f} / {high_name} %{fix[2]*100:.0f}")

    print("\n" + "=" * 78)
    print("2) EKSEN BAGIMSIZLIGI — yuksek korelasyon = iki ad, tek eksen")
    print("=" * 78)
    corr = scores.corr()
    print(corr.round(3).to_string())
    worst = 0.0
    for i, x in enumerate(corr.columns):
        for j, y in enumerate(corr.columns):
            if i < j:
                worst = max(worst, abs(corr.loc[x, y]))
    verdict = ("BAGIMSIZ" if worst < 0.5 else
               "KISMEN ORTUSUYOR" if worst < 0.8 else "AYNI EKSEN")
    print(f"\n  en yuksek |korelasyon| = {worst:.3f}  ->  {verdict}")

    print("\n" + "=" * 78)
    print("3b) SINYAL AYIRT ETME GUCU — her eksen KENDI tur karsitligina karsi")
    print("=" * 78)
    print("  Kullanilan sinyaller [*] ile isaretli. fark ~0 ise o sinyal hicbir sey")
    print("  soylemiyor — eski perc_ratio boyleydi ve eksende duruyordu.")
    genres = meta["genre"].to_numpy()
    pct = df.rank(pct=True)
    for axis, (lo_g, hi_g) in AXIS_CONTRAST.items():
        lo, hi = np.isin(genres, lo_g), np.isin(genres, hi_g)
        used = AF.AXIS_SIGNALS[axis]
        print(f"\n  {axis}  (dusuk n={lo.sum()}, yuksek n={hi.sum()})")
        rows = sorted(((s, pct[s][hi].mean() - pct[s][lo].mean()) for s in SIGNALS),
                      key=lambda r: -abs(r[1]))
        for s, d in rows:
            mark = "*" if s in used else " "
            flag = "  <-- ayirmiyor" if abs(d) < 0.08 else ""
            print(f"   {mark} {s:16s} {d:+.2f}{flag}")

    print("\n" + "=" * 78)
    print("3c) EKSEN AYRIMI — eksen skoru, kendi karsitliginda")
    print("=" * 78)
    for axis, (lo_g, hi_g) in AXIS_CONTRAST.items():
        lo, hi = np.isin(genres, lo_g), np.isin(genres, hi_g)
        d = scores[axis][hi].mean() - scores[axis][lo].mean()
        flag = ("  TERS!" if d < 0 else "  ZAYIF — tur bu ekseni dogrulamiyor"
                if d < 0.10 else "")
        print(f"  {axis:11s} dusuk {scores[axis][lo].mean():.3f}"
              f"  yuksek {scores[axis][hi].mean():.3f}  ayrim {d:+.3f}{flag}")

    print("\n" + "=" * 78)
    print("3) EKSEN ICI SINYAL UYUMU — dusukse ortalama almak anlamsiz")
    print("=" * 78)
    for axis, members in AF.AXIS_SIGNALS.items():
        cols = [c for c in members if c in df.columns]
        sub = df[cols].copy()
        for c, d in members.items():
            if d < 0 and c in sub:
                sub[c] = -sub[c]
        c2 = sub.corr()
        pairs = [(cols[i], cols[j], c2.iloc[i, j])
                 for i in range(len(cols)) for j in range(len(cols)) if i < j]
        avg = np.mean([abs(p[2]) for p in pairs]) if pairs else float("nan")
        print(f"  {axis:11s} ortalama |r| = {avg:.3f}   " +
              "  ".join(f"{x[:9]}~{y[:9]} {r:+.2f}" for x, y, r in pairs))

    json.dump(calib, open(CALIB_PATH, "w"), indent=2)
    print(f"\nkalibrasyon yazildi: {CALIB_PATH} ({len(df):,} parca)")

    # --- genre sanity: genre is not an input, so agreement here is evidence ---
    print("\n" + "=" * 78)
    print("4) GENRE SAGLAMASI — genre GIRDI DEGIL, uyum tesadufi olmaz")
    print("=" * 78)
    scores["genre"] = meta["genre"].to_numpy()
    interesting = ["Lofi Hip-Hop", "Chillhop", "Jazz Rap", "Neo Soul", "Trip Hop",
                   "Trap", "UK Drill", "Brooklyn Drill", "Memphis Rap", "Phonk",
                   "Trap Metal", "Crunk", "Hardcore Hip-Hop"]
    g = (scores[scores["genre"].isin(interesting)]
         .groupby("genre")[["energy", "brightness", "aggression"]]
         .agg(["mean", "count"]))
    rows = []
    for genre in g.index:
        n = int(g.loc[genre, ("energy", "count")])
        if n < 8:
            continue
        rows.append((genre, n, g.loc[genre, ("energy", "mean")],
                     g.loc[genre, ("brightness", "mean")],
                     g.loc[genre, ("aggression", "mean")]))
    rows.sort(key=lambda r: r[2])
    print(f"{'genre':22s} {'n':>4s} {'energy':>8s} {'bright':>8s} {'aggr':>8s}")
    for genre, n, e, b, ag in rows:
        print(f"{genre:22s} {n:4d} {e:8.3f} {b:8.3f} {ag:8.3f}")
    print("\n  (0.5 = korpus ortalamasi; lo-fi tarafi dusuk, drill/trap tarafi yuksek")
    print("   enerji/agresyon bekleniyor — oyle degilse eksenler bir sey olcmuyor)")

    # The real test is the ear, not the table. Print the extremes BY NAME so they can be
    # opened and listened to — if the top of the "Energetic" list sounds calm, the axis is
    # wrong no matter how clean the correlations look.
    print("\n" + "=" * 78)
    print("5) KULAK TESTI — bu dosyalari acip dinle (dataset/audio/<ad>.mp3)")
    print("=" * 78)
    scores["track"] = tracks
    for axis, (low, _mid, high) in AF.MOOD_AXES.items():
        s = scores.sort_values(axis)
        print(f"\n  --- {axis} ---")
        for tag, part in ((f"en {low.upper()} 3", s.head(3)),
                          (f"en {high.upper()} 3", s.tail(3).iloc[::-1])):
            print(f"   {tag}:")
            for _, r in part.iterrows():
                print(f"     {r[axis]:.3f}  [{r['genre'][:20]:20s}] {r['track'][:52]}")





if __name__ == "__main__":
    main()
