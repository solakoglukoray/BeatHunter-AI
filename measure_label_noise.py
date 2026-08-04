"""
How often does the video title contradict the label the collector assigned?

dataset/ was built by searching "<artist> <hint> <genre> Type Beat" and keeping the top
results. YouTube does not promise the results match the query, and nothing ever checked
them: the collector stores filename, genre, artist_type and the search hint — never the
title. So the label says what we ASKED FOR, not what we GOT.

A concrete case: h7NBEcRgBos is labelled NWA / Gangsta Rap and its title is
"[FREE] Eminem x Dr. Dre Old School Hip Hop Type Beat". Eminem and Dr. Dre are both
targets in their own right, so that track trains the artist head to call an
Eminem x Dr. Dre beat "NWA".

WHAT COUNTS AS A MISMATCH
-------------------------
Conservative and checkable: the title names a DIFFERENT taxonomy artist and does not
name the assigned one. Four buckets:

    consistent    title contains the assigned artist
    mismatch      title contains another taxonomy artist, not the assigned one
    undetermined  title names no taxonomy artist — the title cannot settle it
    unavailable   video is private or deleted

Undetermined is not "fine". It is unknown, and it is reported separately rather than
folded into either side.

NAME MATCHING
-------------
Word boundaries, punctuation stripped ("N.W.A" -> "nwa", "A$AP" -> "aap"). Names that
are ordinary English words or too short to be safe — Future, Common, Juvenile, Turbo,
Helluva, OZ, SL — can CONFIRM a label but never trigger a mismatch, because "future
bass" in a title is not evidence of the artist Future. That asymmetry biases the
measurement DOWN, which is the right direction for a claim about how bad things are.

    python measure_label_noise.py --sample 800
"""

import argparse
import random
import re
import time
import unicodedata
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yt_dlp

import config
import corpus

# Ordinary words or too-short tokens: may confirm, never accuse.
AMBIGUOUS = {"future", "common", "juvenile", "turbo", "helluva", "oz", "sl",
             "rage", "dave", "nines", "cardo", "twista", "bounce", "glaive"}

OPTS = {"quiet": True, "no_warnings": True, "skip_download": True,
        "extract_flat": True, "ignoreerrors": False}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    # Apostrophes are DELETED, not turned into spaces. "Pi'erre" must normalise to
    # "pierre" so it matches a title that writes it "Pierre Bourne" — otherwise the
    # matcher accuses a correctly-labelled track. That was 1 of the first 12 flagged.
    s = s.replace("'", "").replace("’", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


# Producers and rappers are routinely written without their handle's prefix — a Mustard
# beat is titled "DR DRE/MUSTARD TYPE BEAT", never "DJ Mustard". Without the shortened
# form the matcher called that a mismatch too. Only forms of 4+ characters are kept, so
# "Lil Baby" does not start matching the word "baby".
_PREFIXES = ("dj ", "lil ", "young ", "big ", "mc ", "the ")


def aliases(name):
    forms = {norm(name)}
    n = norm(name)
    for p in _PREFIXES:
        if n.startswith(p) and len(n[len(p):]) >= 4:
            forms.add(n[len(p):])
    return {f for f in forms if f}


class Blocked(RuntimeError):
    """YouTube is refusing us, which is not the same as a video being gone."""


def title_of(vid):
    """
    Returns (vid, title) or a marker. BLOCKED is kept distinct from UNAVAILABLE.

    The first version lumped every exception into "unavailable" and reported 84% of the
    corpus as deleted. It was not: 652 of 675 failures were "Sign in to confirm you're
    not a bot". Treating a refusal as a data point about the data is exactly the mistake
    the collector's failure guard exists to prevent, repeated in the tool built to audit
    the collector.
    """
    try:
        with yt_dlp.YoutubeDL(OPTS) as y:
            info = y.extract_info(f"https://www.youtube.com/watch?v={vid}",
                                  download=False)
        return vid, (info or {}).get("title") or ""
    except Exception as e:
        msg = str(e).lower()
        if "not a bot" in msg or "sign in to confirm" in msg or "429" in msg:
            return vid, "__BLOCKED__"
        if any(k in msg for k in ("private", "unavailable", "removed", "terminated",
                                  "has been closed", "does not exist")):
            return vid, "__UNAVAILABLE__"
        return vid, "__ERROR__"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=800)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    meta = corpus.load_tracks()
    artists = sorted({t["artist"] for t in config.TARGETS})
    pats = {art: [re.compile(rf"\b{re.escape(f)}\b") for f in aliases(art)]
            for art in artists}

    rng = np.random.default_rng(42)
    sample = meta.iloc[rng.choice(len(meta), min(a.sample, len(meta)), replace=False)]
    print(f"{len(sample):,} parca ornekleniyor ({len(meta):,} icinden), "
          f"{a.workers} paralel...")

    # Sequential with a pause, like the collector. Six parallel threads with no delay
    # is what triggered the block that ruined the first run; a title lookup is cheap
    # enough that patience costs little and impatience costs the whole measurement.
    titles, consecutive_blocks = {}, 0
    for i, vid in enumerate(sample["track"], 1):
        _, title = title_of(vid)
        titles[vid] = title
        consecutive_blocks = consecutive_blocks + 1 if title == "__BLOCKED__" else 0
        if consecutive_blocks >= 8:
            print(f"\n  🛑 {consecutive_blocks} ardisik blok — YouTube bizi "
                  f"reddediyor. {i-consecutive_blocks} parca olculdu, gerisi atlandi.")
            break
        if i % 50 == 0:
            blocked = sum(1 for t in titles.values() if t == "__BLOCKED__")
            print(f"  {i}/{len(sample)}  (blok: {blocked})", flush=True)
        time.sleep(random.uniform(1.5, 3.0))

    rows = []
    for r in sample.itertuples():
        if r.track not in titles:
            continue                     # never attempted (run stopped early)
        title = titles[r.track]
        if title in ("__UNAVAILABLE__", "__ERROR__", "__BLOCKED__"):
            verdict = {"__BLOCKED__": "blocked", "__ERROR__": "error"}.get(
                title, "unavailable")
            rows.append({"track": r.track, "verdict": verdict, "scope": "",
                         "assigned": r.artist_type, "genre": r.genre,
                         "title": title, "found": "", "found_genres": ""})
            continue
        nt = norm(title)
        hit = lambda art: any(p.search(nt) for p in pats[art])
        self_hit = hit(r.artist_type)
        others = [x for x in artists
                  if x != r.artist_type
                  and not (aliases(x) & AMBIGUOUS)
                  and hit(x)]
        verdict = ("consistent" if self_hit else
                   "mismatch" if others else "undetermined")
        # A mismatch inside the assigned genre is a different animal from one across it.
        # "Polo G labelled, Central Cee in the title" is two drill artists — the audio
        # still teaches the genre head the right thing and the artist head a near
        # neighbour. "Southside labelled, Meek Mill in the title" crosses genres and
        # teaches both heads something false. Grouping them would hide that.
        tax_genre = {t["artist"]: t["genre"] for t in config.TARGETS}
        scope = ""
        if verdict == "mismatch":
            same = any(tax_genre.get(o) == r.genre for o in others)
            scope = "tur-ici" if same else "tur-disi"
        rows.append({"track": r.track, "verdict": verdict, "scope": scope,
                     "assigned": r.artist_type, "genre": r.genre,
                     "title": title, "found": ", ".join(others),
                     "found_genres": ", ".join(sorted({tax_genre.get(o, "?")
                                                       for o in others}))})

    df = pd.DataFrame(rows)
    df.to_csv("label_noise_sample.csv", index=False)

    n = len(df)
    checkable = df[df.verdict.isin(["consistent", "mismatch"])]
    print("\n" + "=" * 74)
    print("BASLIK vs ETIKET")
    print("=" * 74)
    for v in ("consistent", "mismatch", "undetermined", "unavailable", "blocked",
              "error"):
        c = int((df.verdict == v).sum())
        print(f"  {v:14s} {c:5d}  %{c/n*100:5.1f}")
    if len(checkable):
        mm = int((checkable.verdict == "mismatch").sum())
        p = mm / len(checkable)
        se = (p * (1 - p) / len(checkable)) ** 0.5
        print(f"\n  Baslikla KARAR VERILEBILEN {len(checkable)} parcada uyusmazlik: "
              f"%{p*100:.1f} (±{se*100:.1f})")
        print(f"  Tum ornek uzerinden alt sinir: %{mm/n*100:.1f} "
              f"(undetermined'lar temiz sayilirsa)")

        bad = df[df.verdict == "mismatch"]
        if len(bad):
            print("\n  UYUSMAZLIGIN TURU:")
            for scope in ("tur-ici", "tur-disi"):
                c = int((bad.scope == scope).sum())
                print(f"    {scope:9s} {c:4d}  (uyusmazliklarin %{c/len(bad)*100:.0f}'i"
                      f", karar verilebilenlerin %{c/len(checkable)*100:.1f}'i)")
            print("    tur-ici = baslikta gecen sanatci ayni genre'de: genre etiketi"
                  " dogru kalir, artist etiketi komsuya kayar")

    bad = df[df.verdict == "mismatch"]
    if len(bad):
        print("\n" + "=" * 74)
        print("ORNEKLER")
        print("=" * 74)
        for r in bad.head(12).itertuples():
            print(f"\n  etiket : {r.assigned}  [{r.genre}]")
            print(f"  baslik : {r.title[:72]}")
            print(f"  gecen  : {r.found}")

        print("\n" + "=" * 74)
        print("EN COK ETKILENEN HEDEFLER")
        print("=" * 74)
        g = (df[df.verdict.isin(["consistent", "mismatch"])]
             .groupby("assigned")["verdict"]
             .agg(n="size", mm=lambda s: (s == "mismatch").sum()))
        g = g[g.n >= 3].assign(rate=lambda x: x.mm / x.n).sort_values("rate",
                                                                     ascending=False)
        for art, r in g.head(12).iterrows():
            print(f"  %{r.rate*100:5.1f}  {int(r.mm)}/{int(r.n)}  {art}")

    print("\nayrinti: label_noise_sample.csv")


if __name__ == "__main__":
    main()
