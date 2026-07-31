"""
Builds the "search this on YouTube" suggestion, and says how much it is worth.

Consumes what analyze_beat already returns. Nothing here re-analyses audio.

WHAT THIS OPTIMISES, AND WHY IT CHANGED
---------------------------------------
The first version ranked labels by P(this label is correct). That was the wrong target,
and validate_query_suggestions.py is what showed it: being right and being a useful
search term are different things.

Measured on 45 held-out tracks, 275 downloaded YouTube results, against a control arm
that keeps the query template and swaps in a RANDOM label of the same kind (so the
comparison is "did we pick the right label", not "is this kind of query any good"):

    query kind    suggested   control    uplift
    artist          +0.136     -0.011    +0.147   (5.7 SE)
    genre           +0.107     -0.002    +0.109   (3.8 SE)
    mood            +0.011     -0.008    +0.018   (1.1 SE)  <- indistinguishable

Floor for reference: two random catalogue tracks sit at -0.001 cosine.

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
    "artist": 0.147 / 0.529,     # 0.278
    "genre":  0.109 / 0.491,     # 0.222
}

# ⚠️ HONEST NOTE: this weighting currently changes NOTHING. Ranking by expected uplift
# prefers artist when P_artist > 0.80 * P_genre; ranking by precision alone prefers it
# when P_artist > P_genre. Across all 2,847 held-out tracks not one lands in that band,
# because the reliability curves are coarse bins and their discrete values never fall
# there — the two criteria agree on 100% of tracks. What actually changed behaviour was
# dropping mood as a candidate and removing the confidence floor.
# It is kept because it is the correct question to be asking, it is grounded in
# measurement rather than intuition, and it will start to bite the moment the curves get
# finer or the classifier shifts. Do not read the criterion name as evidence that
# something clever is happening today.

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
