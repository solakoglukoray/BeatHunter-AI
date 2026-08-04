"""
The corpus and its train/held-out split — ONE definition, imported by everyone.

WHY THIS EXISTS
---------------
models/vector_meta_track.csv was read by eight scripts and written by none. It was a
leftover from the FAISS era, and after the taxonomy change it still described the old
14,235-track corpus while dataset/ held 16,169. Every downstream measurement would have
run on stale data and reported success.

The split was worse. held_out() derived it from dataset_processed/metadata_processed.csv
— spectrogram slices of the OLD corpus. Tracks collected since are absent from that file,
so `track in val` was False for all of them and every new track landed in training. A
213-class model would have been evaluated on a held-out set containing only the original
194 artists, and the headline accuracy would have described a model nobody was running.

Same shape of bug as Trap 1 and as the argmax/joint mismatch: a number measured on
something other than the thing in production. So the corpus list and the split now come
from here, are derived from dataset/metadata.csv (the only file collection actually
writes), and nothing caches a second copy.

    python corpus.py          # rebuild models/vector_meta_track.csv and report
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config

TRACK_META = os.path.join("models", "vector_meta_track.csv")

# The split seed every published figure in this project uses. Changing it invalidates
# comparability with everything already measured.
SPLIT_SEED = 42
VAL_FRACTION = 0.2


def rebuild_track_meta(verbose=True):
    """
    Regenerates models/vector_meta_track.csv from dataset/metadata.csv.

    Only rows whose audio is actually on disk are kept — a metadata row without a file
    would make every consumer's length checks disagree.
    """
    df = pd.read_csv(config.METADATA_FILE)
    df = df.rename(columns={"mood": "search_hint"})
    df["track"] = df["filename"].str.replace(r"\.mp3$", "", regex=True)
    df["path"] = df["track"].map(lambda t: os.path.join(config.AUDIO_DIR, f"{t}.mp3"))
    present = df["path"].map(os.path.exists)
    dropped = int((~present).sum())
    df = df[present]

    # Genre is DERIVED from the artist, never trusted from the CSV. Each row records the
    # genre as it stood when that file was downloaded, so renaming a genre in the
    # taxonomy silently splits the corpus in two: after Mexican Drill became Corridos
    # Tumbados, 225 rows still carried the old string and joint decoding died looking up
    # a genre the data had never heard of. In this taxonomy genre is a function of
    # artist, so computing it removes the possibility of them disagreeing.
    if "title" not in df.columns:
        df["title"] = ""
    df["title"] = df["title"].fillna("")

    tax_genre = {t["artist"]: t["genre"] for t in config.TARGETS}
    derived = df["artist_type"].map(tax_genre)
    changed = int((derived.notna() & (derived != df["genre"])).sum())
    df["genre"] = derived.fillna(df["genre"])
    if changed and verbose:
        print(f"  {changed:,} satirin genre'si taksonomiden guncellendi")

    out = df[["track", "genre", "artist_type", "search_hint", "title"]]
    os.makedirs(os.path.dirname(TRACK_META), exist_ok=True)
    out.to_csv(TRACK_META, index=False)

    if verbose:
        print(f"{TRACK_META}: {len(out):,} parca, "
              f"{out.artist_type.nunique()} sanatci, {out.genre.nunique()} genre")
        if dropped:
            print(f"  ⚠️  {dropped} metadata satiri diskte dosyasi olmadigi icin atlandi")
    return out


TAXONOMY = os.path.join("models", "taxonomy.json")


def rebuild_taxonomy(verbose=True):
    """
    Regenerates models/taxonomy.json from config.TARGETS.

    Read by six scripts, written by none, and after the taxonomy change it still listed
    194 artists — 33 of the new ones absent, 14 removed ones still present. joint
    decoding maps artist -> genre through this file, so a stale copy silently decodes
    into a taxonomy that no longer exists.

    The old format carried a "mood" key. Nothing ever read it, and the name was a lie
    twice over (it was the scraper hint, and mood is now measured from audio), so it is
    emitted as search_hint.
    """
    import json
    tax = {t["artist"]: {"genre": t["genre"], "search_hint": t["search_hint"]}
           for t in config.TARGETS}
    os.makedirs(os.path.dirname(TAXONOMY), exist_ok=True)
    with open(TAXONOMY, "w", encoding="utf-8") as f:
        json.dump(tax, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"{TAXONOMY}: {len(tax)} sanatci, "
              f"{len({v['genre'] for v in tax.values()})} genre")
    return tax


def load_tracks(rebuild_if_stale=True):
    """The corpus as the pipeline sees it, rebuilt when dataset/ has moved on."""
    if rebuild_if_stale and os.path.exists(config.METADATA_FILE):
        fresh = (not os.path.exists(TRACK_META) or
                 os.path.getmtime(config.METADATA_FILE) > os.path.getmtime(TRACK_META))
        if fresh:
            print("ℹ️  vector_meta_track.csv dataset/metadata.csv'den eski — yeniden uretiliyor")
            return rebuild_track_meta()
    return pd.read_csv(TRACK_META)


def held_out_mask(tracks):
    """
    Boolean mask marking the held-out tracks, for any ordering of track ids.

    Grouped by track, which at one row per track is a plain 80/20 split — the grouping
    is kept because it states the intent: never let one song appear on both sides. That
    was a real bug once (model_trainer split at slice level and memorised songs).
    """
    meta = load_tracks()
    all_tracks = meta["track"].to_numpy()
    _, va = next(GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION,
                                   random_state=SPLIT_SEED).split(
                                       all_tracks, groups=all_tracks))
    val = set(all_tracks[va])
    return np.array([t in val for t in tracks])


if __name__ == "__main__":
    meta = rebuild_track_meta()
    rebuild_taxonomy()
    mask = held_out_mask(meta["track"].to_numpy())
    print(f"\nsplit (seed {SPLIT_SEED}): train {(~mask).sum():,} / held-out {mask.sum():,}")
    va = meta[mask]
    print(f"held-out icinde: {va.artist_type.nunique()} sanatci, {va.genre.nunique()} genre")
    missing = set(meta.artist_type) - set(va.artist_type)
    if missing:
        print(f"⚠️  held-out'ta hic ornegi olmayan {len(missing)} sanatci: "
              f"{sorted(missing)[:5]}")
