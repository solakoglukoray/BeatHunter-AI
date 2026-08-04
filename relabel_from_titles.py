"""
Backfills video titles for the whole corpus and re-labels the tracks they contradict.

The collector recorded what we ASKED FOR, never what came back, so 10.1% of the corpus
carries the wrong artist (measured — see CLAUDE.md trap 9). Those tracks are not
garbage: the audio is a real type beat, and the title usually says whose. So they get
RE-LABELLED, not deleted. Genre follows automatically because corpus.py derives genre
from the artist.

WHAT IS AND IS NOT TOUCHED
--------------------------
    mismatch, exactly one other artist named   -> re-labelled to that artist
    mismatch, several other artists named      -> LEFT ALONE, it is a guess
    consistent                                 -> untouched
    undetermined (26% — title names nobody)    -> untouched; unverified is not dirty
    unavailable (private/deleted)              -> untouched, no title to judge by

Titles are fetched from YouTube's oembed endpoint rather than yt-dlp: 0.02s per video
against ~2.5s, and it separates deleted (404) from private (403) by status code instead
of by parsing an error string.

Matching comes from artist_match.py, the same module the collector filters with and the
auditor measures with. Three copies of "does this title name this artist" would drift.

    python relabel_from_titles.py               # fetch, plan, report — changes nothing
    python relabel_from_titles.py --apply       # write the new labels
"""

import argparse
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

import artist_match
import config

OEMBED = "https://www.youtube.com/oembed"
TITLE_CACHE = "models/video_titles.csv"


def fetch_titles(video_ids, workers=6):
    """id -> title, or "" when the video is gone. Cached so a rerun is free."""
    cached = {}
    if os.path.exists(TITLE_CACHE):
        c = pd.read_csv(TITLE_CACHE).fillna({"title": ""})
        cached = dict(zip(c["track"], c["title"]))
        print(f"{len(cached):,} baslik cache'ten")

    todo = [v for v in video_ids if v not in cached]
    if not todo:
        return cached
    print(f"{len(todo):,} baslik cekiliyor ({workers} paralel)...", flush=True)

    session = requests.Session()
    codes = Counter()

    def one(vid):
        for attempt in range(3):
            try:
                r = session.get(OEMBED, params={
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "format": "json"}, timeout=10)
                if r.status_code == 200:
                    return vid, r.json().get("title", ""), 200
                if r.status_code in (401, 403, 404):     # gone or private: settled
                    return vid, "", r.status_code
                time.sleep(1 + attempt)                  # 429/5xx: back off, retry
            except Exception:
                time.sleep(1 + attempt)
        return vid, "", -1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (vid, title, code) in enumerate(ex.map(one, todo), 1):
            cached[vid] = title
            codes[code] += 1
            if i % 2000 == 0:
                el = time.time() - t0
                print(f"  {i:,}/{len(todo):,}  {el:.0f}s  "
                      f"eta {el/i*(len(todo)-i):.0f}s", flush=True)

    print(f"  durum kodlari: {dict(codes)}")
    if codes[-1] > len(todo) * 0.2:
        print(f"  ⚠️  {codes[-1]:,} istek hic cevap alamadi — plan eksik olabilir")

    pd.DataFrame({"track": list(cached), "title": list(cached.values())}).to_csv(
        TITLE_CACHE, index=False)
    print(f"  yazildi: {TITLE_CACHE}")
    return cached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    df = pd.read_csv(config.METADATA_FILE)
    df["track"] = df["filename"].str.replace(r"\.mp3$", "", regex=True)
    artists = [t["artist"] for t in config.TARGETS]
    tax = {t["artist"]: t["genre"] for t in config.TARGETS}
    pats = artist_match.build_patterns(artists)

    titles = fetch_titles(df["track"].tolist(), a.workers)
    df["title"] = df["track"].map(titles).fillna("")

    plan, counts = [], Counter()
    for r in df.itertuples():
        if not r.title:
            counts["no_title"] += 1
            continue
        v = artist_match.verdict(r.title, r.artist_type, pats)
        counts[v] += 1
        if v != "mismatch":
            continue
        # Released songs, official videos and karaoke tracks are a DIFFERENT kind of
        # contamination (9.6% of the corpus). Re-labelling one just moves the problem
        # into another class, so they are left where they are and counted instead.
        if not artist_match.looks_like_type_beat(r.title):
            counts["not_a_beat"] += 1
            continue
        named = [x for x in artist_match.names_in(r.title, pats)
                 if x != r.artist_type and not (artist_match.aliases(x)
                                                & artist_match.AMBIGUOUS)]
        if len(named) == 1:
            plan.append({"track": r.track, "title": r.title,
                         "from_artist": r.artist_type, "from_genre": r.genre,
                         "to_artist": named[0], "to_genre": tax[named[0]]})
        else:
            counts["ambiguous"] += 1

    print("\n" + "=" * 76)
    print("KORPUS DURUMU")
    print("=" * 76)
    for k in ("consistent", "mismatch", "undetermined", "no_title"):
        print(f"  {k:14s} {counts[k]:6,}  %{counts[k]/len(df)*100:5.1f}")
    print(f"  {'  bunlardan':14s} {counts['ambiguous']:6,}  birden fazla isim "
          f"aniyor -> DOKUNULMUYOR")

    p = pd.DataFrame(plan)
    if p.empty:
        print("\nyeniden etiketlenecek parca yok")
        return 0

    print(f"\n  YENIDEN ETIKETLENECEK: {len(p):,} parca "
          f"(korpusun %{len(p)/len(df)*100:.1f}'i)")
    same = (p["from_genre"] == p["to_genre"]).sum()
    print(f"    genre ayni kalan : {same:,}")
    print(f"    genre degisen    : {len(p)-same:,}")

    print("\n" + "=" * 76)
    print("SANATCI BASINA ETKI (net degisim, en buyuk 20)")
    print("=" * 76)
    before = df["artist_type"].value_counts()
    delta = (p["to_artist"].value_counts().reindex(before.index).fillna(0)
             - p["from_artist"].value_counts().reindex(before.index).fillna(0))
    after = before + delta
    imp = pd.DataFrame({"once": before, "sonra": after.astype(int),
                        "fark": delta.astype(int)})
    imp = imp[imp["fark"] != 0].sort_values("fark")
    print(f"  {'sanatci':24s} {'once':>6s} {'sonra':>6s} {'fark':>6s}")
    for art, r in pd.concat([imp.head(10), imp.tail(10)]).iterrows():
        print(f"  {art:24s} {r.once:6d} {r.sonra:6d} {r.fark:+6d}")

    risky = imp[imp["sonra"] < 30]
    if len(risky):
        print(f"\n  ⚠️  30 parcanin altina dusen {len(risky)} sanatci:")
        for art, r in risky.iterrows():
            print(f"      {art:24s} {r.once} -> {r.sonra}")

    gafter = df.assign(g=df["artist_type"].map(tax)).groupby("g").size()
    print(f"\n  sinif dengesizligi: once {before.max()/before.min():.1f}x  "
          f"-> sonra {after.max()/max(after.min(),1):.1f}x")

    print("\n" + "=" * 76)
    print("ORNEKLER (ilk 15)")
    print("=" * 76)
    for r in p.head(15).itertuples():
        print(f"\n  {r.from_artist} [{r.from_genre}]  ->  {r.to_artist} [{r.to_genre}]")
        print(f"    {r.title[:70]}")

    p.to_csv("relabel_plan.csv", index=False)
    print(f"\ntam plan: relabel_plan.csv ({len(p):,} satir)")

    if not a.apply:
        print("\n(dry-run — hicbir sey degismedi; uygulamak icin --apply)")
        return 0

    backup = config.METADATA_FILE + f".before_relabel_{int(time.time())}"
    shutil.copy2(config.METADATA_FILE, backup)
    fix = dict(zip(p["track"], p["to_artist"]))
    df["artist_type"] = df.apply(
        lambda r: fix.get(r["track"], r["artist_type"]), axis=1)
    df["genre"] = df["artist_type"].map(tax).fillna(df["genre"])
    df[["filename", "genre", "artist_type", "search_hint", "title"]].to_csv(
        config.METADATA_FILE, index=False)
    print(f"\n✅ {len(p):,} parca yeniden etiketlendi")
    print(f"   yedek: {backup}")
    print("   sonraki: python corpus.py && python train_classic.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
