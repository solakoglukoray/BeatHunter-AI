"""
Builds the "search this on YouTube" suggestion, and says how much it is worth.

Consumes what analyze_beat already returns. Nothing here re-analyses audio.

WHAT THIS OPTIMISES, AND WHY IT CHANGED
---------------------------------------
The first version ranked labels by P(this label is correct). That was the wrong target,
and validate_query_suggestions.py is what showed it: being right and being a useful
search term are different things.

Measured against a control arm that keeps the query template and swaps in a RANDOM label
of the same kind, so the comparison is "did we pick the right label", not "is this kind
of query any good". Two random catalogue tracks sit at ~0.00 cosine — that is the floor.

    query kind    2026-08-01 corpus      2026-08-04 corpus (current)
    artist          +0.147 (5.7 SE)        +0.114 (4.7 SE)
    genre           +0.109 (3.8 SE)        +0.052 (2.6 SE)
    mood            +0.018 (1.1 SE)        not a candidate

Both fell after the taxonomy change, genre by half. The corpus grew from 194 to 213
artists and 65 to 60 genres, so a genre label now covers more ground and pins down the
sound less. Artist still clears its control by a wide margin, which is why the criterion
below weights it more heavily than it did.

MOOD IS NOT A QUERY
-------------------
Mood is measured correctly — that was validated separately, and it still ships as
character information in the UI. It just does not work as a search term: "chill type
beat" genuinely returns chill beats, but chill beats span every genre, so none of them
resemble the specific track you started from. Anchoring it to the predicted genre helped
("chill Southern Hip-Hop type beat", +0.045, 2.8 SE) yet stayed well under artist and
genre, so mood is not a query candidate at all here.

NO CONFIDENCE FLOOR ON THE QUERY
--------------------------------
A query is always built, even when the label is below the old 150x display bar, because
the measurement says a low-confidence label still beats a random one: the genre arm ran
at 49% precision and still cleared its control by 3.8 SE. Whether to SHOW a name as a
claim is a separate question, still governed by confidence["reliable"].

Note what the split by correctness says about why artist queries hold up:

    artist, label correct   +0.165        genre, label correct   +0.176
    artist, label wrong     +0.126        genre, label wrong     +0.078

A wrong artist prediction is usually a stylistically adjacent artist, so the query still
lands in the right neighbourhood. A wrong genre is a bigger miss.
"""

import json
import os

RELIABILITY_PATH = "models/label_reliability.json"

# Measured uplift per unit of P(correct): the arm's measured uplift divided by the mean
# P(correct) of the tracks it was measured on (artist +0.147 at 0.529, genre +0.109 at
# 0.491). Two anchors, linear between them — at P(correct) -> 0 the label carries no
# information and the query degenerates to the control arm, which is uplift 0 by
# definition. No functional form is invented beyond that straight line.
#
# ⚠️ Both arms were sampled where the old criterion chose them: artist around 184x,
# genre around 42x. Far below that this extrapolates, and it will overstate a hopeless
# guess. Re-run validate_query_suggestions.py if the classifier changes.
QUERY_UPLIFT_PER_PRECISION = {
    "artist": 0.114 / 0.304,     # 0.375
    "genre":  0.052 / 0.295,     # 0.176
}

# This weighting was inert when it was written: on the old curves, query_utility and
# reliability picked the same label for all 2,847 held-out tracks, and the note here said
# so. It stopped being inert after the taxonomy change — on the 2026-08-04 corpus the two
# criteria now disagree on a third of tracks (artist chosen 154 times vs 86), because the
# retrained curves moved precision values into the band where P_artist > 0.80 * P_genre
# but P_artist < P_genre.
#
# Which is the right call is NOT settled by that: reliability picks the likelier-correct
# label and scores 27.3% against query_utility's 25.3% on ground truth. query_utility is
# the default anyway, because the measured thing it optimises — how close the returned
# beats actually sound — is what a search is for, and an artist query beat its control by
# +0.147 against genre's +0.109. Re-run validate_query_suggestions.py before trusting
# that trade-off again; the uplift constants below predate this corpus.

DEFAULT_CRITERION = "query_utility"

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


def candidates(confidence, mood=None, reliability=None):
    """
    Labels that could carry the query, each scored in expected cosine uplift.

    `mood` is accepted and ignored — kept in the signature so callers do not have to
    care, and so the omission is visible here rather than looking like an oversight.
    Mood ships to the UI on its own path.
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
        out.append({
            "kind": kind, "label": label,
            "precision": p, "n": n, "se": se,
            "ratio_vs_chance": float(ratio),
            "expected_uplift": p * QUERY_UPLIFT_PER_PRECISION[kind],
            "evidence": f"{float(ratio):.0f}x sansin uzerinde",
        })
    return out


def recommend_label(confidence, mood=None, reliability=None,
                    criterion=DEFAULT_CRITERION):
    """
    The query worth suggesting, with the reason, or None if nothing is available.

    criterion: "query_utility" ranks by expected cosine uplift — P(correct) scaled by
    that kind's measured value as a search term. "reliability" ranks by P(correct)
    alone, which is kept only so the two can be compared; it prefers genre almost
    always, because genre is likelier to be right while being a vaguer thing to search.
    """
    cands = candidates(confidence, mood, reliability)
    if not cands:
        return None

    key = "expected_uplift" if criterion == "query_utility" else "precision"
    ranked = sorted(cands, key=lambda c: -c[key])
    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None

    reason = (f"{best['label']} ({best['kind']}) — olculen dogruluk "
              f"%{best['precision']*100:.0f} ({best['evidence']}, n={best['n']}), "
              f"beklenen benzerlik kazanci {best['expected_uplift']:+.3f}")
    if runner:
        reason += (f"; alternatif {runner['label']} ({runner['kind']}) "
                   f"{runner['expected_uplift']:+.3f}")

    return {
        # A query is always built — see the module docstring. `reliable` on the
        # confidence dict still governs whether a NAME is presented as a claim.
        "qualifies": True,
        "best": best,
        "ranked": ranked,
        "reason": reason,
        "query": suggest_query(best),
    }


def suggest_query(best):
    """What to actually type into YouTube."""
    return f"{best['label']} type beat"
