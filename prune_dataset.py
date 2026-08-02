"""
Moves audio for artists no longer in the taxonomy out of the active dataset.

MOVES, never deletes. dataset/ is not in git — 1,056 tracks represent hours of
collection and there is no way to get an identical set back, so a wrong prune would be
unrecoverable. Everything lands in dataset/pruned/ with a manifest, and --restore puts
it back.

The metadata CSV is rewritten to match, because the CSV is what data_collector reads to
decide what is already collected and what train_classic reads to build labels. Leaving
rows for files that are no longer there is exactly the code/data drift this project has
been bitten by before.

    python prune_dataset.py --dry-run
    python prune_dataset.py
    python prune_dataset.py --restore
"""

import argparse
import csv
import os
import shutil

import pandas as pd

import config

PRUNE_DIR      = os.path.join(config.DATASET_ROOT, "pruned")
PRUNE_AUDIO    = os.path.join(PRUNE_DIR, "audio")
PRUNE_MANIFEST = os.path.join(PRUNE_DIR, "manifest.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true")
    a = ap.parse_args()

    if a.restore:
        return restore()

    df = pd.read_csv(config.METADATA_FILE)
    keep_artists = {t["artist"] for t in config.TARGETS}
    drop = df[~df["artist_type"].isin(keep_artists)]

    if drop.empty:
        print("taksonomi disinda parca yok — yapilacak bir sey yok")
        return 0

    by_artist = drop.groupby(["artist_type", "genre"]).size()
    print(f"taksonomi disinda {len(drop):,} parca, {drop['artist_type'].nunique()} sanatci")
    for (art, gen), n in by_artist.sort_values(ascending=False).items():
        print(f"   {n:4d}  {art:24s} [{gen}]")

    on_disk = sum(os.path.exists(os.path.join(config.AUDIO_DIR, f))
                  for f in drop["filename"])
    print(f"\ndiskte bulunan: {on_disk}/{len(drop)}")
    print(f"kalacak: {len(df) - len(drop):,} parca")

    if a.dry_run:
        print("\n(dry-run — hicbir sey tasinmadi)")
        return 0

    os.makedirs(PRUNE_AUDIO, exist_ok=True)
    moved = 0
    for f in drop["filename"]:
        src = os.path.join(config.AUDIO_DIR, f)
        if os.path.exists(src):
            shutil.move(src, os.path.join(PRUNE_AUDIO, f))
            moved += 1

    # Append to the manifest so repeated prunes accumulate rather than overwrite.
    header = not os.path.exists(PRUNE_MANIFEST)
    drop.to_csv(PRUNE_MANIFEST, mode="a", header=header, index=False)

    kept = df[df["artist_type"].isin(keep_artists)]
    kept.to_csv(config.METADATA_FILE, index=False)

    print(f"\ntasinan dosya : {moved}  -> {PRUNE_AUDIO}")
    print(f"manifest      : {PRUNE_MANIFEST}")
    print(f"metadata      : {len(df):,} -> {len(kept):,} satir")
    return 0


def restore():
    if not os.path.exists(PRUNE_MANIFEST):
        print("manifest yok — geri yuklenecek bir sey bulunamadi")
        return 1
    man = pd.read_csv(PRUNE_MANIFEST)
    moved = 0
    for f in man["filename"]:
        src = os.path.join(PRUNE_AUDIO, f)
        if os.path.exists(src):
            shutil.move(src, os.path.join(config.AUDIO_DIR, f))
            moved += 1
    df = pd.read_csv(config.METADATA_FILE)
    pd.concat([df, man]).drop_duplicates("filename").to_csv(
        config.METADATA_FILE, index=False)
    os.remove(PRUNE_MANIFEST)
    print(f"geri yuklendi: {moved} dosya")
    print("⚠️  config.py'yi de geri almayi unutma — taksonomi hala kirpilmis halde")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
