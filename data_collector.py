"""
BeatHunter AI - Autonomous Data Collector (v2.0)

Changes in v2.0:
─────────────────────────────────────────────────────────────
1. Duplicate Detection (Audio Fingerprinting)
   - MD5 hash computed for every downloaded file.
   - If the same audio appears under a different YouTube ID or search,
     it is deleted immediately and the slot is retried with the next result.
   - Global hash registry persists across all targets in one run.

2. Guaranteed Quota per Target
   - Old behavior: max_downloads counted errors/skips, real file count was lower.
   - New behavior: we keep trying until exactly FILES_PER_TARGET VALID files
     are collected per target, pulling from a 5x candidate pool.

3. Corrupt / Too-Short File Guard
   - Files under 60 seconds after download are deleted and retried.
   - Files that fail to open (corrupt MP3) are deleted and retried.

4. YouTube ID Registry
   - Tracks all downloaded YouTube IDs globally.
   - Prevents re-downloading the same video even across different targets.

5. Resume Support
   - On startup, loads existing metadata CSV and rebuilds hash + ID registries.
   - Safe to stop and restart — won't reprocess already-collected files.
─────────────────────────────────────────────────────────────
"""

import os
import csv
import hashlib
import yt_dlp
import config

# ============================================================
# GLOBAL REGISTRIES
# Populated at startup from existing metadata (resume support).
# ============================================================
DOWNLOADED_HASHES = set()   # MD5 hashes of all valid audio files
DOWNLOADED_IDS    = set()   # YouTube video IDs already collected

# Candidate multiplier — search this many results to guarantee quota
# e.g. FILES_PER_TARGET=75, we search up to 75*5=375 candidates
CANDIDATE_MULTIPLIER = 5


def compute_md5(filepath, chunk_size=8192):
    """Computes MD5 hash of a file to detect duplicates regardless of filename."""
    h = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def initialize_dataset():
    """
    Creates dataset directories and metadata CSV if they don't exist.
    If metadata already exists, rebuilds hash + ID registries for resume support.
    """
    os.makedirs(config.AUDIO_DIR, exist_ok=True)

    if not os.path.exists(config.METADATA_FILE):
        with open(config.METADATA_FILE, mode='w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(["filename", "genre", "artist_type", "mood"])
        print(f"📄 Metadata initialized: {config.METADATA_FILE}")
    else:
        # Resume: rebuild registries from existing metadata
        print(f"📄 Existing metadata found — loading registries for resume...")
        with open(config.METADATA_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filepath = os.path.join(config.AUDIO_DIR, row['filename'])
                if os.path.exists(filepath):
                    file_hash = compute_md5(filepath)
                    if file_hash:
                        DOWNLOADED_HASHES.add(file_hash)
                    # Extract YouTube ID from filename (format: {yt_id}.mp3)
                    yt_id = os.path.splitext(row['filename'])[0]
                    DOWNLOADED_IDS.add(yt_id)

        print(f"   ✅ Loaded {len(DOWNLOADED_HASHES)} hashes, {len(DOWNLOADED_IDS)} IDs from existing data.\n")


def append_metadata(filename, target):
    """Appends a single validated file entry to the metadata CSV."""
    with open(config.METADATA_FILE, mode='a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            filename,
            target['genre'],
            target['artist'],
            target['mood']
        ])


def get_audio_duration(filepath):
    """
    Returns duration in seconds using mutagen (fast, no full decode needed).
    Falls back to None if file is unreadable.
    """
    try:
        from mutagen.mp3 import MP3
        audio = MP3(filepath)
        return audio.info.length
    except Exception:
        return None


def fetch_target_audio(target):
    """
    Downloads exactly FILES_PER_TARGET valid, unique audio files for one target.

    Strategy:
    - Search for CANDIDATE_MULTIPLIER * FILES_PER_TARGET candidates at once.
    - Iterate through results one by one.
    - For each: check ID registry → download → check duration → check hash.
    - If valid: keep it, write metadata, increment counter.
    - If invalid (duplicate/corrupt/short): delete and try next candidate.
    - Stop when FILES_PER_TARGET valid files are collected.
    """
    
    if os.path.exists(config.METADATA_FILE):
        import pandas as pd
        df_existing = pd.read_csv(config.METADATA_FILE)
        already_done = len(df_existing[df_existing['artist_type'] == target['artist']])
        if already_done >= config.FILES_PER_TARGET:
            print(f"\n⏭️  [{target['artist']}] already complete ({already_done} files). Skipping.")
            return

    search_term   = f"{target['artist']} {target['mood']} {target['genre']} Type Beat"
    candidate_cap = config.FILES_PER_TARGET * CANDIDATE_MULTIPLIER

    print(f"\n🔍 Mining: {search_term}")
    print(f"   Target: {config.FILES_PER_TARGET} valid files | Candidate pool: {candidate_cap}")

    # Step 1: Collect candidate video metadata (no download yet)
    info_opts = {
        'quiet':          True,
        'no_warnings':    True,
        'noplaylist':     True,
        'ignoreerrors':   True,
        'extract_flat':   True,  # Fast — only fetches metadata, not audio
        'match_filter':   yt_dlp.utils.match_filter_func("duration > 60 & duration < 600"),
    }

    search_query = f"ytsearch{candidate_cap}:{search_term}"
    candidates   = []

    with yt_dlp.YoutubeDL(info_opts) as ydl:
        try:
            info = ydl.extract_info(search_query, download=False)
            if info and 'entries' in info:
                candidates = [e for e in info['entries'] if e and e.get('id')]
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            return

    print(f"   📋 Found {len(candidates)} candidates.")

    # Step 2: Download candidates one by one until quota is met
    valid_count = 0

    for candidate in candidates:
        if valid_count >= config.FILES_PER_TARGET:
            break

        yt_id = candidate.get('id')
        if not yt_id:
            continue

        # Skip already-downloaded YouTube IDs (global registry)
        if yt_id in DOWNLOADED_IDS:
            print(f"   ⏭️  Skipping duplicate ID: {yt_id}")
            continue

        expected_mp3 = os.path.join(config.AUDIO_DIR, f"{yt_id}.mp3")

        # Download this single video
        dl_opts = {
            'format':      'bestaudio/best',
            'outtmpl':     os.path.join(config.AUDIO_DIR, '%(id)s.%(ext)s'),
            'postprocessors': [{
                'key':              'FFmpegExtractAudio',
                'preferredcodec':   'mp3',
                'preferredquality': '192',
            }],
            'noplaylist':   True,
            'ignoreerrors': True,
            'quiet':        True,
            'no_warnings':  True,
        }

        try:
            with yt_dlp.YoutubeDL(dl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={yt_id}"])
        except Exception as e:
            print(f"   ❌ Download error ({yt_id}): {e}")
            continue

        # Verify file exists after download
        if not os.path.exists(expected_mp3):
            print(f"   ⚠️  File missing after download: {yt_id}")
            continue

        # Duration check — skip files under 60 seconds
        duration = get_audio_duration(expected_mp3)
        if duration is None or duration < 60:
            print(f"   🗑️  Too short or unreadable ({duration}s): {yt_id}")
            os.remove(expected_mp3)
            continue

        # Duplicate audio check via MD5 hash
        file_hash = compute_md5(expected_mp3)
        if file_hash is None:
            print(f"   🗑️  Could not hash (corrupt?): {yt_id}")
            os.remove(expected_mp3)
            continue

        if file_hash in DOWNLOADED_HASHES:
            print(f"   🗑️  Duplicate audio detected (hash match): {yt_id}")
            os.remove(expected_mp3)
            continue

        # ✅ Valid file — register and record
        DOWNLOADED_HASHES.add(file_hash)
        DOWNLOADED_IDS.add(yt_id)

        filename = f"{yt_id}.mp3"
        append_metadata(filename, target)
        valid_count += 1

        print(f"   ✅ [{valid_count}/{config.FILES_PER_TARGET}] Saved: {filename} ({duration:.0f}s)")

    if valid_count < config.FILES_PER_TARGET:
        print(f"   ⚠️  Only collected {valid_count}/{config.FILES_PER_TARGET} — "
              f"candidate pool exhausted. Consider increasing CANDIDATE_MULTIPLIER.")
    else:
        print(f"   🎯 Quota met: {valid_count} valid files for [{target['artist']}]")


def main():
    """Main execution entry point for the Data Mining Pipeline."""
    print("🚀 BeatHunter AI — Data Collection Pipeline v2.0")
    print(f"   Target per category: {config.FILES_PER_TARGET} files")
    print(f"   Candidate multiplier: {CANDIDATE_MULTIPLIER}x")
    print(f"   Total targets: {len(config.TARGETS)}\n")

    initialize_dataset()

    for i, target in enumerate(config.TARGETS, 1):
        print(f"\n[{i}/{len(config.TARGETS)}]", end="")
        fetch_target_audio(target)

    print("\n\n🎉 DATA COLLECTION PIPELINE COMPLETED!")
    print(f"   Hash registry size: {len(DOWNLOADED_HASHES)}")
    print(f"   Unique videos collected: {len(DOWNLOADED_IDS)}")


if __name__ == "__main__":
    main()