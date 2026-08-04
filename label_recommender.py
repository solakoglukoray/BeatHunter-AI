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

    query kind    08-01 corpus     08-04 taxonomy    08-04 re-labelled (current)
    artist          +0.147            +0.114             +0.047 (2.2 SE)
    genre           +0.109            +0.052             +0.076 (3.1 SE)
    mood            +0.018            not a candidate

The ranking INVERTED at the last step: genre now beats artist. Re-labelling from video
titles moved 1,171 tracks onto the artists the titles actually named, which concentrated
them on popular names — Drake 75 -> 138, J. Cole 75 -> 141. A search for a household
name returns an enormous, varied pool, so its results resemble any particular track less
than a narrower producer's do. Better labels, weaker queries.

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

On the 2026-08-01 corpus, splitting the uplift by whether the label was right showed a
wrong artist still earning +0.126 against a wrong genre's +0.078 — the classifier
confuses stylistically adjacent artists, so the query lands in the right neighbourhood
anyway. That split has NOT been re-measured since the re-labelling, and the headline
numbers moved enough that it should not be quoted as current.
"""

import json
import os

RELIABILITY_PATH = "models/label_reliability.json"

# Measured uplift per unit of P(correct): the arm's measured uplift divided by the mean
# P(correct) of the tracks it was measured on (artist +0.047 at 0.169, genre +0.076 at
# 0.337). Two anchors, linear between them — at P(correct) -> 0 the label carries no
# information and the query degenerates to the control arm, which is uplift 0 by
# definition. No functional form is invented beyond that straight line.
#
# ⚠️ Each arm was sampled only where the criterion already chose it, so far below those
# operating points this extrapolates and will overstate a hopeless guess. Re-run
# validate_query_suggestions.py whenever the classifier or the labels change — both did
# on 2026-08-04, and the artist/genre ranking inverted.
QUERY_UPLIFT_PER_PRECISION = {
    "artist": 0.047 / 0.169,     # 0.278
    "genre":  0.076 / 0.337,     # 0.226
}

# The two criteria disagree, and which one is right is NOT settled by ground truth alone.
# On the current corpus (506 held-out tracks with mood signals):
#
#     query_utility   artist 147 / genre 359   labels right 25.1%
#     reliability     artist  81 / genre 425   labels right 28.1%
#
# reliability wins on label accuracy by 3 points and always will — it optimises exactly
# that. query_utility stays the default because a search is not a quiz: what matters is
# whether the beats that come back sound like the track, and that is the thing measured
# by downloading them. Revisit this if the uplift gap between artist and genre closes
# further; it has already inverted once.

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
