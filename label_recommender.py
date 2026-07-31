"""
Picks which label — artist, genre, or a mood axis — is the most trustworthy thing to
say about a track, and says why.

Consumes what analyze_beat already returns. Nothing here re-analyses audio.

THE UNIT PROBLEM
----------------
The three signals arrive in incompatible units: artist and genre come as "ratio vs
chance", mood as a corpus percentile. The tempting move is to put everything in ratio
terms. It does not work, because the ceiling depends on how many classes there are:

    artist      194 classes -> a ratio can reach 194x
    genre        65 classes ->                    65x
    mood axis     3 labels  ->                     3x

Under that unit a certain mood loses to a hopeless artist guess, always. So the common
unit is the one the user actually cares about:

    P(this label is correct)

measured for all three on the same held-out split by calibrate_label_choice.py. For
artist and genre that is precision against ground truth. Mood has no ground truth to be
wrong about — it is a measurement — so its number is the probability the label survives
being measured on a different part of the track.

RELIABILITY IS NOT USEFULNESS
-----------------------------
Ranking purely by P(correct) has a bias nobody should walk into by accident: a 3-way
label is easy to get right, so mood tends to win. But "Aggressive type beat" is a poor
YouTube query and "Young Chop type beat" is a good one. So both criteria are computed:

    reliability   P(correct)                      — how likely the label is true
    information   P(correct) x log2(n_choices)    — how much it narrows the search

`criterion` selects which one drives the pick. They disagree often, and the disagreement
is the interesting part rather than a defect — see demo_label_choice.py.
"""

import json
import math
import os

RELIABILITY_PATH = "models/label_reliability.json"

# Below this, no single label is worth leading with and the caller should show the
# top-5 candidate list instead. Set at the artist classifier's own overall top-1
# accuracy: a label that is not beating "just show the list" has not earned the spot.
MIN_USEFUL_PRECISION = 0.35

_cache = None
_tried = False


def load_reliability(path=RELIABILITY_PATH):
    """Measured curves, or None if they were never computed."""
    global _cache, _tried
    if _tried:
        return _cache
    _tried = True
    try:
        with open(path, encoding="utf-8") as f:
            _cache = json.load(f)
    except FileNotFoundError:
        print("ℹ️  no models/label_reliability.json — run calibrate_label_choice.py")
    except Exception as e:
        print(f"⚠️  label reliability unreadable ({e})")
    return _cache


def _lookup(curve, value):
    """P(correct) for the bin this value falls into, plus the sample size behind it."""
    for b in curve:
        if b["lo"] <= value < b["hi"]:
            return b["precision"], b["n"], b["se"]
    if curve:
        b = curve[-1] if value >= curve[-1]["lo"] else curve[0]
        return b["precision"], b["n"], b["se"]
    return None, 0, None


def candidates(confidence, mood, reliability=None):
    """
    Every label we could lead with, each scored in the common unit.

    Returns a list of dicts, unsorted. `confidence` and `mood` are exactly what
    analyze_beat returns; either may be missing pieces and this still works.
    """
    rel = reliability if reliability is not None else load_reliability()
    if rel is None:
        return []

    out = []

    conf = confidence or {}
    for kind in ("artist", "genre"):
        c = conf.get(kind) or {}
        ratio = c.get("ratio_vs_chance")
        label = c.get("label")
        if ratio is None or not label:
            continue
        spec = rel.get(kind) or {}
        p, n, se = _lookup(spec.get("curve") or [], float(ratio))
        if p is None:
            continue
        choices = spec.get("n_classes") or 2
        out.append({
            "kind": kind, "axis": None, "label": label,
            "precision": p, "n": n, "se": se,
            "choices": choices,
            "information": p * math.log2(choices),
            "evidence": f"{float(ratio):.0f}x sansin uzerinde",
        })

    axes = (rel.get("mood") or {}).get("per_axis") or {}
    middles = _middle_labels()
    for axis, val in (mood or {}).items():
        spec = axes.get(axis)
        if not spec:
            continue
        # The middle tercile is never a recommendation. "Moderate" is a statement that
        # the track has no character on this axis, so leading with it is useless however
        # reliable it is — and it is not even reliable, since sitting mid-scale means
        # sitting near a cut. Without this the chooser recommended "balanced G-Funk type
        # beat" for a track whose artist had been correctly identified at 161x chance.
        if val["label"] == middles.get(axis):
            continue
        score = float(val["score"])
        # Distance from the nearest tercile cut — the same quantity the curve was
        # measured against. Cuts live in the mood calibration, but the reliability file
        # records the margin bins, so recover the margin from the score directly.
        cuts = spec.get("cuts")
        if cuts is None:
            cuts = _axis_cuts(axis)
        margin = min(abs(score - cuts[0]), abs(score - cuts[1])) if cuts else 0.0
        p, n, se = _lookup(spec.get("curve") or [], margin)
        if p is None:
            continue
        out.append({
            "kind": "mood", "axis": axis, "label": val["label"],
            "precision": p, "n": n, "se": se,
            "choices": 3,
            "information": p * math.log2(3),
            "evidence": f"korpus yuzdeligi {score:.2f}, kesim noktasindan {margin:.2f} uzakta",
        })

    return out


def _middle_labels():
    """The 'no character on this axis' label per axis, e.g. Moderate / Balanced."""
    try:
        import audio_features as AF
        return {ax: names[1] for ax, names in AF.MOOD_AXES.items()}
    except Exception:
        return {}


def _axis_cuts(axis):
    """Tercile cuts for an axis, read from the mood calibration."""
    try:
        import audio_features as AF
        calib = AF.load_mood_calibration()
        return (calib or {}).get("axis_cuts", {}).get(axis)
    except Exception:
        return None


DEFAULT_CRITERION = "information"


def recommend_label(confidence, mood, reliability=None, criterion=DEFAULT_CRITERION):
    """
    The single label most worth showing, with the reason, or None if nothing qualifies.

    criterion: "information" ranks by P(correct) x log2(n_choices); "reliability" ranks
    by P(correct) alone.

    "information" is the default because ranking on reliability alone degenerates.
    Measured on 384 held-out tracks it chose a mood word 349 times out of 371 — a 3-way
    label is simply easier to get right than a 194-way one, so the mechanism collapses
    into saying "Mellow" about everything. Under "information" the same tracks split
    29 artist / 133 genre / 209 mood, and the labels that can be checked against ground
    truth are right 42.6% of the time versus 43.6% for the old 150x-artist-only policy —
    while covering 162 tracks with a specific label instead of 39.
    """
    cands = candidates(confidence, mood, reliability)
    if not cands:
        return None

    key = "precision" if criterion == "reliability" else "information"

    # Filter FIRST, then rank. Ranking first and checking the winner's precision
    # afterwards threw away perfectly good labels: under the information criterion a
    # low-precision artist guess can top the ranking on cardinality alone, and the whole
    # recommendation was then discarded even when a trustworthy genre sat right below it.
    # The question is "among labels worth saying, which is most useful", in that order.
    trusted = [c for c in cands if c["precision"] >= MIN_USEFUL_PRECISION]
    qualifies = bool(trusted)
    ranked = sorted(trusted or cands, key=lambda c: -c[key])
    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None

    name = best["label"] if best["kind"] != "mood" else f"{best['label']} ({best['axis']})"
    if qualifies:
        reason = (f"{name} — olculen dogruluk %{best['precision']*100:.0f} "
                  f"({best['evidence']}, n={best['n']})")
        if runner:
            rname = (runner["label"] if runner["kind"] != "mood"
                     else f"{runner['label']} ({runner['axis']})")
            reason += (f"; sonraki en iyi {rname} %{runner['precision']*100:.0f}")
    else:
        reason = (f"guvenilir tek etiket yok — en iyisi {name} bile sadece "
                  f"%{best['precision']*100:.0f}; aday listesini goster")

    return {
        "qualifies": qualifies,
        "best": best,
        "ranked": ranked,
        "reason": reason,
        "query": suggest_query(best, confidence, ranked) if qualifies else None,
    }


def suggest_query(best, confidence=None, ranked=None):
    """
    What to actually type into YouTube, given the winning label.

    A mood word alone is a thin query, so it is paired with the genre — but ONLY if the
    genre earned its place. Pairing unconditionally produced "dark Chillhop type beat"
    for a track whose genre had been guessed at 3x chance: the query looked specific
    while quietly carrying a label the chooser had just rejected as unreliable.
    """
    if best["kind"] in ("artist", "genre"):
        return f"{best['label']} type beat"

    genre = None
    for c in (ranked or []):
        if c["kind"] == "genre" and c["precision"] >= MIN_USEFUL_PRECISION:
            genre = c["label"]
            break
    word = best["label"].lower()
    return f"{word} {genre} type beat" if genre else f"{word} type beat"
