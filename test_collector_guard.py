"""
Proves data_collector's failure guard fires — and, just as important, that it does NOT
fire on normal filtering. A breaker that halts healthy runs would be worse than none.

The guard exists because taxonomy targets #98-#109 came out empty in one contiguous
block while the collector printed success and exited 0. It has since caught a real
YouTube bot-block mid-run (2026-08-02), stopping after 12 consecutive failures instead
of quietly finishing with 432 tracks missing.

The two negative cases matter as much as the positive ones. Duplicates and short files
are the filter doing its job; a guard that counted them would fire on every healthy run
and teach everyone to ignore it.

    python test_collector_guard.py
"""
import sys

import data_collector as DC

ok = True


def case(name, fn, should_raise):
    global ok
    try:
        fn()
        raised = False
        msg = ""
    except DC.CollectionHalted as e:
        raised, msg = True, str(e)[:70]
    good = raised == should_raise
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {name:46s} "
          f"{'durdu' if raised else 'devam etti'}")
    if raised:
        print(f"        -> {msg}...")


# 1. consecutive download failures
def consec():
    s = DC.Stats()
    for i in range(DC.MAX_CONSECUTIVE_FAILURES):
        s.record("A", "download_error")


case(f"{DC.MAX_CONSECUTIVE_FAILURES} ardisik indirme hatasi", consec, True)


# 2. one short of the limit must NOT trip
def almost():
    s = DC.Stats()
    for i in range(DC.MAX_CONSECUTIVE_FAILURES - 1):
        s.record("A", "download_error")


case(f"{DC.MAX_CONSECUTIVE_FAILURES-1} ardisik (esigin bir altinda)", almost, False)


# 3. a success in between resets the counter
def interleaved():
    s = DC.Stats()
    for i in range(40):
        s.record("A", "download_error" if i % 2 else "saved")


case("basari ile bolunmus hatalar (sayac sifirlanmali)", interleaved, False)


# 4. rolling window rate
def window():
    s = DC.Stats()
    # 60% failure over the window, never 12 in a row
    pattern = ["download_error", "download_error", "saved"] * 15
    for o in pattern:
        s.record("A", o)


case("pencere hata orani esigi asiyor", window, True)


# 5. rejections must never trip it — this is the important negative
def rejections():
    s = DC.Stats()
    for i in range(500):
        s.record("A", "duplicate_hash" if i % 2 else "too_short")


case("500 eleme (duplicate/too_short) — tripmemeli", rejections, False)


# 6. empty targets
def empties():
    s = DC.Stats()
    for i in range(DC.MAX_EMPTY_TARGETS):
        s.target_finished(f"artist{i}", 0, True)


case(f"{DC.MAX_EMPTY_TARGETS} hedef aday varken bos dondu", empties, True)


# 7. empty target with NO candidates is a supply issue, not a breakage
def no_candidates():
    s = DC.Stats()
    for i in range(5):
        s.target_finished(f"artist{i}", 0, False)


case("aday yokken bos donen hedefler (arz sorunu)", no_candidates, False)


# ---------------------------------------------------------------------------
# Transient-vs-permanent classification.
#
# Only network faults may be retried. Retrying a bot-block or a 403 hammers a door
# YouTube has already closed and earns a longer ban; retrying "video unavailable"
# is pure waste. Getting this boundary wrong in either direction is expensive, and
# it is one string-match away at all times.
# ---------------------------------------------------------------------------
print("\n  --- gecici mi kalici mi ---")
CLASSIFY = [
    ("getaddrinfo failed", True),
    ("TransportError('[Errno 11001] getaddrinfo failed')", True),
    ("Connection reset by peer", True),
    ("Read timed out", True),
    ("HTTP Error 503: Service Unavailable", True),
    ("HTTP Error 500: Internal Server Error", True),
    ("Sign in to confirm you're not a bot", False),
    ("HTTP Error 403: Forbidden", False),
    ("This video is not available", False),
    ("Sign in to confirm your age", False),
    ("Private video", False),
]
for msg, want in CLASSIFY:
    got = DC._is_transient(Exception(msg))
    good = got == want
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  "
          f"{'tekrar dene' if got else 'tekrar deneme':14s} <- {msg[:46]}")

print("\n" + ("TUM TESTLER GECTI" if ok else "!!! BASARISIZ TEST VAR"))
sys.exit(0 if ok else 1)
