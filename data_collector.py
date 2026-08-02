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

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import Counter, deque

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

# ============================================================
# FAILURE GUARD (added 2026-08-02)
# ============================================================
# The v2.0 collector ran with ignoreerrors=True and printed failures without counting
# them. Targets #98-#109 of the taxonomy came out empty or near-empty in one contiguous
# block — Memphis Rap, Stoner Rap, Slab Music, Snap Music — and the run reported success.
# 1,112 tracks were missing and nothing said so.
#
# It was not a supply problem: those artists have 50-60 usable results on YouTube today
# (measured). Something broke mid-run, almost certainly throttling, and every error was
# swallowed one at a time until the window passed.
#
# So the distinction that matters here is NOT error vs success. Duplicates and short
# files are supposed to happen — they are the filter working. Download and search
# failures are infrastructure. Only the latter trips the breaker; counting rejections
# as failures would halt healthy runs.
REJECTIONS   = {"duplicate_hash", "duplicate_id", "too_short", "unhashable"}
FAILURES     = {"download_error", "missing_after_download", "search_failed"}

MAX_CONSECUTIVE_FAILURES = 12    # a throttled run fails every attempt, not some
FAILURE_WINDOW           = 30    # rolling window of download attempts
MAX_WINDOW_FAILURE_RATE  = 0.5
MAX_EMPTY_TARGETS        = 2     # targets that yielded nothing despite having candidates

# Pause between downloads. Back-to-back requests with no gap are the bot signature, and
# on 2026-08-02 YouTube escalated from scattered 403s to "Sign in to confirm you're not
# a bot" partway through a repair run — which halts collection entirely, so the delay
# buys far more than it costs. Randomised because a fixed interval is itself a pattern.
DOWNLOAD_DELAY = (2.0, 4.0)

LOG_PATH = "collection_log.json"


class CollectionHalted(RuntimeError):
    """Raised when the run looks broken rather than merely unlucky."""


class Stats:
    """Counts outcomes and stops the run when the pattern says something is wrong."""

    def __init__(self):
        self.outcomes = Counter()
        self.per_target = {}
        self.recent = deque(maxlen=FAILURE_WINDOW)
        self.consecutive_failures = 0
        self.empty_targets = []
        self.started = time.time()

    def record(self, target, outcome):
        self.outcomes[outcome] += 1
        self.per_target.setdefault(target, Counter())[outcome] += 1
        if outcome in REJECTIONS:
            return                       # normal filtering, not evidence of breakage
        is_fail = outcome in FAILURES
        self.recent.append(is_fail)
        self.consecutive_failures = self.consecutive_failures + 1 if is_fail else 0
        self._check()

    def _check(self):
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            raise CollectionHalted(
                f"{self.consecutive_failures} ardisik indirme hatasi. Bu tekil kotu "
                f"videolara benzemiyor — buyuk olasilikla YouTube throttling ya da ag "
                f"kesintisi. Sessizce devam etmek {MAX_EMPTY_TARGETS} hedefi bos "
                f"birakan hatanin ta kendisiydi.")
        if len(self.recent) == FAILURE_WINDOW:
            rate = sum(self.recent) / len(self.recent)
            if rate > MAX_WINDOW_FAILURE_RATE:
                raise CollectionHalted(
                    f"son {FAILURE_WINDOW} denemenin %{rate*100:.0f}'i basarisiz "
                    f"(esik %{MAX_WINDOW_FAILURE_RATE*100:.0f}).")

    def target_finished(self, target, valid, had_candidates):
        if valid == 0 and had_candidates:
            self.empty_targets.append(target)
            if len(self.empty_targets) >= MAX_EMPTY_TARGETS:
                raise CollectionHalted(
                    f"{len(self.empty_targets)} hedef aday bulundugu halde sifir parca "
                    f"verdi: {', '.join(self.empty_targets)}. Tam olarak bu desen "
                    f"#98-#109 blogunu yok etti.")

    def summary(self):
        fails = sum(self.outcomes[k] for k in FAILURES)
        rejects = sum(self.outcomes[k] for k in REJECTIONS)
        ok = self.outcomes["saved"]
        total = ok + fails + rejects
        return {"saved": ok, "failures": fails, "rejections": rejects,
                "attempts": total,
                "failure_rate": (fails / total) if total else 0.0,
                "outcomes": dict(self.outcomes),
                "empty_targets": self.empty_targets,
                "elapsed_s": round(time.time() - self.started, 1),
                "per_target": {k: dict(v) for k, v in self.per_target.items()}}

    def report(self):
        s = self.summary()
        print("\n" + "=" * 62)
        print("TOPLAMA OZETI")
        print("=" * 62)
        print(f"  kaydedilen      {s['saved']:6d}")
        print(f"  elenen (normal) {s['rejections']:6d}   "
              f"{ {k: v for k, v in s['outcomes'].items() if k in REJECTIONS} }")
        print(f"  HATA            {s['failures']:6d}   "
              f"{ {k: v for k, v in s['outcomes'].items() if k in FAILURES} }")
        print(f"  hata orani      %{s['failure_rate']*100:.1f}")
        print(f"  sure            {s['elapsed_s']:.0f}s")
        if s["empty_targets"]:
            print(f"  ⚠️  bos kalan hedefler: {s['empty_targets']}")
        return s


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
            # 4th column is the scraper's query word, NOT a label — see config.TARGETS.
            csv.writer(f).writerow(["filename", "genre", "artist_type", "search_hint"])
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
            target['search_hint']
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


def fetch_target_audio(target, stats, simple_query=False, delay=True):
    """
    Tops one target up to FILES_PER_TARGET valid, unique audio files.

    Strategy:
    - Search for CANDIDATE_MULTIPLIER * FILES_PER_TARGET candidates at once.
    - Iterate through results one by one.
    - For each: check ID registry → download → check duration → check hash.
    - If valid: keep it, write metadata, increment counter.
    - If invalid (duplicate/corrupt/short): delete and try next candidate.
    - Stop when the target has FILES_PER_TARGET files in total.

    `simple_query` drops the search_hint and genre words. The full string is narrow, and
    for a few targets that is the whole problem: "Hollywood Undead Aggressive Crunkcore
    Type Beat" returns 12 results where "Hollywood Undead Type Beat" returns 60.
    """
    needed = config.FILES_PER_TARGET
    if os.path.exists(config.METADATA_FILE):
        import pandas as pd
        df_existing = pd.read_csv(config.METADATA_FILE)
        already_done = len(df_existing[df_existing['artist_type'] == target['artist']])
        if already_done >= config.FILES_PER_TARGET:
            print(f"\n⏭️  [{target['artist']}] already complete ({already_done} files). Skipping.")
            return
        # Top up rather than restart: a partially-collected target keeps what it has.
        needed = config.FILES_PER_TARGET - already_done

    if simple_query:
        search_term = f"{target['artist']} Type Beat"
    else:
        search_term = (f"{target['artist']} {target['search_hint']} "
                       f"{target['genre']} Type Beat")
    candidate_cap = config.FILES_PER_TARGET * CANDIDATE_MULTIPLIER

    print(f"\n🔍 Mining: {search_term}")
    print(f"   Need: {needed} more valid files | Candidate pool: {candidate_cap}")

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
            stats.record(target['artist'], "search_failed")
            return

    if not candidates:
        # An empty result set is itself a failure signal. The old code returned here
        # with no record at all, which is how whole targets vanished quietly.
        print(f"   ❌ Search returned nothing for: {search_term}")
        stats.record(target['artist'], "search_failed")
        return

    print(f"   📋 Found {len(candidates)} candidates.")

    # Step 2: Download candidates one by one until quota is met
    valid_count = 0

    for candidate in candidates:
        if valid_count >= needed:
            break

        yt_id = candidate.get('id')
        if not yt_id:
            continue

        # Skip already-downloaded YouTube IDs (global registry)
        if yt_id in DOWNLOADED_IDS:
            print(f"   ⏭️  Skipping duplicate ID: {yt_id}")
            stats.record(target['artist'], "duplicate_id")
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
            # ignoreerrors was True. yt-dlp then handled the failure internally and
            # returned normally, so the except below never fired and a failed download
            # was indistinguishable from a successful one until the file check.
            'ignoreerrors': False,
            'quiet':        True,
            'no_warnings':  True,
            'noprogress':   True,
        }

        if delay:
            time.sleep(random.uniform(*DOWNLOAD_DELAY))

        try:
            with yt_dlp.YoutubeDL(dl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={yt_id}"])
        except Exception as e:
            print(f"   ❌ Download error ({yt_id}): {str(e)[:120]}")
            stats.record(target['artist'], "download_error")
            continue

        # Verify file exists after download
        if not os.path.exists(expected_mp3):
            print(f"   ⚠️  File missing after download: {yt_id}")
            stats.record(target['artist'], "missing_after_download")
            continue

        # Duration check — skip files under 60 seconds
        duration = get_audio_duration(expected_mp3)
        if duration is None or duration < 60:
            print(f"   🗑️  Too short or unreadable ({duration}s): {yt_id}")
            os.remove(expected_mp3)
            stats.record(target['artist'], "too_short")
            continue

        # Duplicate audio check via MD5 hash
        file_hash = compute_md5(expected_mp3)
        if file_hash is None:
            print(f"   🗑️  Could not hash (corrupt?): {yt_id}")
            os.remove(expected_mp3)
            stats.record(target['artist'], "unhashable")
            continue

        if file_hash in DOWNLOADED_HASHES:
            print(f"   🗑️  Duplicate audio detected (hash match): {yt_id}")
            os.remove(expected_mp3)
            stats.record(target['artist'], "duplicate_hash")
            continue

        # ✅ Valid file — register and record
        DOWNLOADED_HASHES.add(file_hash)
        DOWNLOADED_IDS.add(yt_id)

        filename = f"{yt_id}.mp3"
        append_metadata(filename, target)
        valid_count += 1
        stats.record(target['artist'], "saved")

        print(f"   ✅ [{valid_count}/{needed}] Saved: {filename} ({duration:.0f}s)")

    if valid_count < needed:
        print(f"   ⚠️  Only collected {valid_count}/{needed} — "
              f"candidate pool exhausted. Consider increasing CANDIDATE_MULTIPLIER.")
    else:
        print(f"   🎯 Quota met: {valid_count} valid files for [{target['artist']}]")
    stats.target_finished(target['artist'], valid_count, bool(candidates))


def targets_below_quota():
    """Targets that do not have FILES_PER_TARGET files yet, shortfall first."""
    import pandas as pd
    counts = {}
    if os.path.exists(config.METADATA_FILE):
        df = pd.read_csv(config.METADATA_FILE)
        counts = df.groupby("artist_type").size().to_dict()
    short = [(config.FILES_PER_TARGET - counts.get(t["artist"], 0), t)
             for t in config.TARGETS
             if counts.get(t["artist"], 0) < config.FILES_PER_TARGET]
    short.sort(key=lambda r: -r[0])
    return short


def main():
    ap = argparse.ArgumentParser(description="BeatHunter data collector")
    ap.add_argument("--repair", action="store_true",
                    help="sadece kotanin altindaki hedefleri topla")
    ap.add_argument("--only", nargs="+", metavar="ARTIST",
                    help="yalnizca bu sanatcilari topla")
    ap.add_argument("--simple-query", action="store_true",
                    help="arama terimini sadeleştir (<artist> Type Beat)")
    ap.add_argument("--no-delay", action="store_true",
                    help="indirmeler arasi gecikmeyi kapat (blok riskini artirir)")
    ap.add_argument("--dry-run", action="store_true",
                    help="ne yapilacagini yazdir, hicbir sey indirme")
    args = ap.parse_args()

    if args.repair:
        short = targets_below_quota()
        targets = [t for _, t in short]
        total_missing = sum(n for n, _ in short)
        print("🔧 ONARIM MODU")
        print(f"   kotanin altinda {len(targets)} hedef, toplam {total_missing:,} eksik parca")
        for n, t in short:
            print(f"     {n:3d} eksik  {t['artist']:24s} [{t['genre']}]")
    elif args.only:
        wanted = {a.lower() for a in args.only}
        targets = [t for t in config.TARGETS if t["artist"].lower() in wanted]
        missing = wanted - {t["artist"].lower() for t in targets}
        if missing:
            raise SystemExit(f"taksonomide yok: {sorted(missing)}")
    else:
        targets = list(config.TARGETS)

    print("\n🚀 BeatHunter AI — Data Collection Pipeline v2.1")
    print(f"   Target per category: {config.FILES_PER_TARGET} files")
    print(f"   Candidate multiplier: {CANDIDATE_MULTIPLIER}x")
    print(f"   Targets this run: {len(targets)}")
    print(f"   Download delay: "
          f"{'kapali' if args.no_delay else f'{DOWNLOAD_DELAY[0]:.0f}-{DOWNLOAD_DELAY[1]:.0f}s'}")
    print(f"   Failure guard: {MAX_CONSECUTIVE_FAILURES} ardisik / "
          f"%{MAX_WINDOW_FAILURE_RATE*100:.0f} pencere / "
          f"{MAX_EMPTY_TARGETS} bos hedef\n")

    if args.dry_run:
        print("(dry-run — hicbir sey indirilmedi)")
        return 0

    initialize_dataset()
    stats = Stats()
    halted = None

    try:
        for i, target in enumerate(targets, 1):
            print(f"\n[{i}/{len(targets)}]", end="")
            fetch_target_audio(target, stats, simple_query=args.simple_query,
                               delay=not args.no_delay)
    except CollectionHalted as e:
        halted = str(e)
        print(f"\n\n🛑 TOPLAMA DURDURULDU: {e}")
    except KeyboardInterrupt:
        halted = "kullanici kesintisi"
        print("\n\n⏸️  kullanici tarafindan durduruldu")

    summary = stats.report()
    summary["halted"] = halted
    # Written whether or not the run finished — a halted run's numbers are the
    # interesting ones, and the old collector kept none of this.
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  log: {LOG_PATH}")

    if halted:
        print("\n  ⚠️  Bu koşu EKSİK. Sorunu çözüp --repair ile devam et; "
              "toplanan parçalar korunuyor.")
        return 1
    print("\n🎉 DATA COLLECTION COMPLETED")
    print(f"   Hash registry size: {len(DOWNLOADED_HASHES)}")
    print(f"   Unique videos collected: {len(DOWNLOADED_IDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())