"""
Does the suggested query actually find music that sounds like the track?

The recommender says "search: aggressive Trap type beat". This asks whether a user who
does that hears something like what they started with — measured, not assumed.

THE TWO CONFOUNDS THIS IS BUILT AROUND
--------------------------------------
1. Type beats already sound alike. Any query returns hip-hop instrumentals at 120-150
   BPM, so a decent-looking similarity proves nothing on its own. Floor: the similarity
   between the track and a random catalogue track.

2. Query KIND is not query CORRECTNESS. If genre searches simply return more coherent
   results than mood searches, comparing "our genre pick" against "a random mood query"
   would credit the recommender for something it did not do. So the control keeps the
   template and swaps only the label for a random one of the SAME kind:

       recommended   "aggressive Trap type beat"
       control       "mellow Bounce type beat"      <- same shape, arbitrary words

   The gap between those two is the recommender's actual contribution.

CONTAMINATION
-------------
dataset/ was itself collected with "<artist> <hint> <genre> Type Beat" searches, so a
suggested query can return videos that are already in the corpus — including, sometimes,
the track being tested. That would manufacture similarity out of nothing. Every returned
video id is checked against the corpus and against the query track, and the rates are
reported for both arms rather than quietly dropped.

    python validate_query_suggestions.py --per-kind 8 --results 3
    python validate_query_suggestions.py --report-only        # reuse what is downloaded
"""

import argparse
import json
import os
import random
import shutil
import time
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import yt_dlp
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import GroupShuffleSplit

import audio_features as AF
import config
import inference_engine as IE
import label_recommender as LR

WORK_DIR   = "temp_query_validation"
STATE_PATH = os.path.join(WORK_DIR, "state.json")
TRACK_META = "models/vector_meta_track.csv"
SLICE_META = "dataset_processed/metadata_processed.csv"

# What the information criterion actually picks, measured on held-out tracks. Used to
# weight the per-kind results into one honest overall number — the per-kind samples are
# deliberately equal-sized so each is measurable, which is NOT the real mix.
REAL_MIX = {"artist": 29 / 371, "genre": 133 / 371, "mood": 209 / 371}

MOOD_WORDS = [w for names in AF.MOOD_AXES.values() for w in (names[0], names[2])]


def held_out_mask(tracks):
    m = pd.read_csv(SLICE_META)
    m["track"] = m["filename"].str.split("_slice").str[0]
    _, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                   random_state=42).split(m, groups=m["track"]))
    val = set(m.iloc[va]["track"].unique())
    return np.array([t in val for t in tracks])


def build_confidence(m, row):
    Pa = m["clf_artist"].predict_proba(row)[0]
    Pg = m["clf_genre"].predict_proba(row)[0]
    artists, genres, g_of_a = m["artists"], m["genres"], m["genre_of_artist"]
    a_cls, g_cls = m["clf_artist"].classes_, m["clf_genre"].classes_
    logPg = np.full(len(genres), -30.0)
    logPg[g_cls] = np.log(Pg + 1e-12)
    scores = np.log(Pa + 1e-12) + logPg[g_of_a[a_cls]]
    best = int(np.argmax(scores))
    ai, gi = int(a_cls[best]), int(g_of_a[a_cls[best]])
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


def control_query(rec, rng):
    """Same template, random labels of the same kind. The point of comparison."""
    b = rec["best"]
    if b["kind"] == "artist":
        pool = [t["artist"] for t in config.TARGETS if t["artist"] != b["label"]]
        return f"{rng.choice(pool)} type beat"
    if b["kind"] == "genre":
        pool = sorted({t["genre"] for t in config.TARGETS} - {b["label"]})
        return f"{rng.choice(pool)} type beat"
    # mood: swap the mood word, and the genre too when the suggestion carried one
    word = rng.choice([w for w in MOOD_WORDS if w.lower() != b["label"].lower()]).lower()
    parts = rec["query"].split()
    if len(parts) > 3:                      # "<mood> <genre...> type beat"
        pool = sorted({t["genre"] for t in config.TARGETS})
        return f"{word} {rng.choice(pool)} type beat"
    return f"{word} type beat"


def search(query, n):
    """Top-n YouTube hits for a query — metadata only, no download yet."""
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True,
            "ignoreerrors": True, "extract_flat": True,
            "match_filter": yt_dlp.utils.match_filter_func(
                "duration > 60 & duration < 600")}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch{n * 3}:{query}", download=False)
    except Exception as e:
        print(f"    arama hatasi ({query}): {e}")
        return []
    entries = [e for e in (info or {}).get("entries") or [] if e and e.get("id")]
    return [{"id": e["id"], "title": e.get("title") or "",
             "channel": e.get("channel") or e.get("uploader") or ""}
            for e in entries[:n]]


def download(vid, out_dir):
    """First 45 seconds only — the embedding reads 30s, so the rest is wasted bandwidth."""
    path = os.path.join(out_dir, f"{vid}.mp3")
    if os.path.exists(path):
        return path
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3",
                            "preferredquality": "128"}],
        "download_ranges": yt_dlp.utils.download_range_func(None, [(0, 45)]),
        "force_keyframes_at_cuts": True,
        "noplaylist": True, "ignoreerrors": True,
        "quiet": True, "no_warnings": True, "noprogress": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])
    except Exception:
        return None
    return path if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-kind", type=int, default=8)
    ap.add_argument("--results", type=int, default=3)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--mood-anchor", action="store_true",
                    help="mood parcalarina ucuncu kol: mood + tahmin edilen genre")
    a = ap.parse_args()

    os.makedirs(WORK_DIR, exist_ok=True)
    rng = random.Random(42)

    m = joblib.load("models/classic_model.joblib")
    z = np.load(AF.feature_cache_path(), allow_pickle=True)
    X, tracks = z["X"], z["tracks"]
    meta = pd.read_csv(TRACK_META).set_index("track")
    calib = AF.load_mood_calibration()

    iv = held_out_mask(tracks)
    Xh, th = X[iv], tracks[iv]
    Xs = m["scaler"].transform(m["pca"].transform(Xh))

    zs = np.load("models/mood_signals.npz", allow_pickle=True)
    sig = {t: dict(zip(AF.MOOD_SIGNAL_NAMES, r)) for t, r in zip(zs["tracks"], zs["X"])}

    # Projected corpus vectors, for the random-catalogue floor.
    Xall = m["scaler"].transform(m["pca"].transform(X))
    Xall = Xall / (np.linalg.norm(Xall, axis=1, keepdims=True) + 1e-9)

    pool = []
    for i, t in enumerate(th):
        if t not in sig:
            continue
        conf = build_confidence(m, Xs[i:i + 1])
        mood = AF.classify_mood(sig[t], calib)
        rec = LR.recommend_label(conf, mood)
        if not rec or not rec["qualifies"]:
            continue
        item = {"track": t, "row": i, "kind": rec["best"]["kind"],
                "label": rec["best"]["label"], "query": rec["query"],
                "precision": rec["best"]["precision"],
                "control": control_query(rec, rng),
                "true_artist": meta.loc[t, "artist_type"],
                "true_genre": meta.loc[t, "genre"]}
        # Third arm for mood tracks: the same mood word anchored to the predicted genre.
        # Every mood suggestion in the first run came out bare, because suggest_query only
        # attaches a genre that clears the precision floor — and a genre that clears it
        # gets picked to lead instead. So "aggressive Trap type beat" was a form the
        # recommender could never actually emit, and it was never tested.
        if item["kind"] == "mood":
            item["anchored"] = f"{rec['best']['label'].lower()} {conf['genre']['label']} type beat"
        pool.append(item)

    by_kind = {}
    for p in pool:
        by_kind.setdefault(p["kind"], []).append(p)
    sample = []
    for kind, items in by_kind.items():
        rng.shuffle(items)
        sample.extend(items[:a.per_kind])
    print(f"havuz {len(pool)} parca -> ornek {len(sample)} "
          f"({', '.join(f'{k}:{min(len(v), a.per_kind)}' for k, v in by_kind.items())})")

    state = {}
    if os.path.exists(STATE_PATH):
        state = json.load(open(STATE_PATH, encoding="utf-8"))

    corpus_ids = set(tracks)

    # ---- search + download ----
    if not a.report_only:
        for n, item in enumerate(sample, 1):
            key = item["track"]
            entry = state.get(key) or {"track": key}
            arms = ["query", "control"] + (["anchored"] if item.get("anchored") else [])
            for arm in arms:
                if entry.get(arm + "_hits"):
                    continue
                q = item[arm]
                hits = search(q, a.results)
                entry[arm + "_q"] = q
                entry[arm + "_hits"] = hits
                print(f"  [{n}/{len(sample)}] {arm:7s} \"{q}\" -> {len(hits)} sonuc")
            state[key] = entry
            json.dump(state, open(STATE_PATH, "w", encoding="utf-8"), indent=1)

        jobs = []
        for item in sample:
            e = state[item["track"]]
            for arm in ("query", "control", "anchored"):
                for h in e.get(arm + "_hits") or []:
                    jobs.append(h["id"])
        jobs = list(dict.fromkeys(jobs))
        todo = [v for v in jobs if not os.path.exists(os.path.join(WORK_DIR, f"{v}.mp3"))]
        print(f"\nindirilecek {len(todo)} / {len(jobs)} video (45sn kesit)")
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=6) as ex:
            for _ in ex.map(lambda v: download(v, WORK_DIR), todo):
                done += 1
                if done % 10 == 0:
                    el = time.time() - t0
                    print(f"  {done}/{len(todo)}  {el:.0f}s  "
                          f"eta {el/done*(len(todo)-done):.0f}s", flush=True)

    # ---- embed + compare ----
    print("\nembedding cikariliyor...")
    emb_cache = {}

    def vec(vid):
        if vid in emb_cache:
            return emb_cache[vid]
        p = os.path.join(WORK_DIR, f"{vid}.mp3")
        v = None
        if os.path.exists(p):
            y = AF.load_audio(p, sr=AF.PANNS_SR, duration=AF.DURATION)
            if y is not None:
                proj = IE._project(m, y, AF.PANNS_SR)
                if proj is not None:
                    v = proj[0] / (np.linalg.norm(proj[0]) + 1e-9)
        emb_cache[vid] = v
        return v

    rows = []
    for item in sample:
        e = state.get(item["track"])
        if not e:
            continue
        i = item["row"]
        orig = Xall[np.where(tracks == item["track"])[0][0]]
        rec_floor = float(np.dot(Xall[rng.randrange(len(Xall))], orig))
        for arm in ("query", "control", "anchored"):
            for h in e.get(arm + "_hits") or []:
                v = vec(h["id"])
                if v is None:
                    continue
                rows.append({
                    "track": item["track"], "kind": item["kind"], "arm": arm,
                    "query": e[arm + "_q"], "vid": h["id"], "title": h["title"],
                    "channel": h["channel"],
                    "cos": float(np.dot(v, orig)),
                    "in_corpus": h["id"] in corpus_ids,
                    "is_self": h["id"] == item["track"],
                    "floor": rec_floor,
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(WORK_DIR, "results.csv"), index=False)
    if df.empty:
        raise SystemExit("sonuc yok")

    # random-catalogue floor, many draws rather than one per track
    floor_draws = [float(np.dot(Xall[rng.randrange(len(Xall))],
                                Xall[rng.randrange(len(Xall))])) for _ in range(4000)]
    floor = float(np.mean(floor_draws))

    print("\n" + "=" * 78)
    print("1) BENZERLIK — onerilen sorgu vs ayni bicimde rastgele sorgu")
    print("=" * 78)
    print(f"  taban (rastgele iki katalog parcasi): {floor:+.3f}")
    clean = df[~df["is_self"]]
    print(f"\n  {'tur':8s} {'n':>4s} {'onerilen':>10s} {'kontrol':>10s} {'fark':>8s} {'SE':>7s}")
    for kind in ["artist", "genre", "mood"]:
        sub = clean[clean["kind"] == kind]
        if sub.empty:
            continue
        q = sub[sub["arm"] == "query"]["cos"]
        c = sub[sub["arm"] == "control"]["cos"]
        if len(q) < 2 or len(c) < 2:
            continue
        d = q.mean() - c.mean()
        se = float(np.sqrt(q.var(ddof=1)/len(q) + c.var(ddof=1)/len(c)))
        print(f"  {kind:8s} {len(q):4d} {q.mean():10.3f} {c.mean():10.3f} "
              f"{d:+8.3f} {se:7.3f}   {d/se if se else 0:+.1f} SE")
    q, c = clean[clean.arm == "query"]["cos"], clean[clean.arm == "control"]["cos"]
    d = q.mean() - c.mean()
    se = float(np.sqrt(q.var(ddof=1)/len(q) + c.var(ddof=1)/len(c)))
    print(f"  {'TUMU':8s} {len(q):4d} {q.mean():10.3f} {c.mean():10.3f} "
          f"{d:+8.3f} {se:7.3f}   {d/se if se else 0:+.1f} SE")
    w = sum(REAL_MIX[k] * (clean[(clean.kind == k) & (clean.arm == "query")]["cos"].mean()
                           - clean[(clean.kind == k) & (clean.arm == "control")]["cos"].mean())
            for k in REAL_MIX if not clean[clean.kind == k].empty)
    print(f"\n  gercek karisimla agirliklandirilmis fark: {w:+.3f}")

    print("\n" + "=" * 78)
    print("2) KONTAMINASYON — sonuclar zaten korpusumuzda mi")
    print("=" * 78)
    for arm in ("query", "control"):
        s = df[df["arm"] == arm]
        print(f"  {arm:8s} korpusta {s['in_corpus'].mean()*100:5.1f}%   "
              f"sorgulanan parcanin kendisi {s['is_self'].sum()} kez")

    print("\n" + "=" * 78)
    print("3) GOZLE BAKMAK ICIN — ilk sonuclar")
    print("=" * 78)
    for kind in ["artist", "genre", "mood"]:
        sub = df[df["kind"] == kind]
        if sub.empty:
            continue
        tr = sub["track"].iloc[0]
        item = next(x for x in sample if x["track"] == tr)
        print(f"\n  [{kind}] {tr} — gercek: {item['true_artist']} / {item['true_genre']}")
        for arm in ("query", "control"):
            s = sub[(sub["track"] == tr) & (sub["arm"] == arm)]
            if s.empty:
                continue
            print(f"    {arm} = \"{s['query'].iloc[0]}\"")
            for _, r in s.iterrows():
                tag = " [KORPUSTA]" if r["in_corpus"] else ""
                print(f"       {r['cos']:+.3f}  {r['title'][:58]}")
                print(f"               {r['channel'][:50]}{tag}")


if __name__ == "__main__":
    main()
