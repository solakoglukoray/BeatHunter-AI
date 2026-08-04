"""
Finding taxonomy artist names inside a YouTube title — ONE implementation.

Used by data_collector (to reject a result before downloading it) and by
measure_label_noise (to audit what was already collected). Those two must agree: an
auditor with a different matcher than the collector reports on a system nobody runs,
which is the drift this project keeps getting caught by.

WHY IT MATTERS
--------------
dataset/ was built by searching "<artist> <hint> <genre> Type Beat" and keeping the top
results, and nothing ever checked what came back. Measured on 700 sampled tracks:

    consistent    60.4%   title names the assigned artist
    mismatch      10.1%   title names a DIFFERENT taxonomy artist
    undetermined  26.4%   title names no taxonomy artist
    unavailable    3.0%

That is 14.4% (±1.6) of the tracks a title can adjudicate. The clearest case was
h7NBEcRgBos: labelled NWA, titled "Eminem x Dr. Dre Old School Hip Hop Type Beat".
"""

import re
import unicodedata

# Ordinary English words or tokens too short to be safe. These may CONFIRM a label but
# never accuse one — "future bass" in a title is not evidence of the artist Future.
# The asymmetry biases both the filter and the audit toward leniency, on purpose.
AMBIGUOUS = {"future", "common", "juvenile", "turbo", "helluva", "oz", "sl",
             "rage", "dave", "nines", "cardo", "twista", "bounce", "glaive"}

# Handles are routinely written without their prefix: a Mustard beat is titled
# "DR DRE/MUSTARD TYPE BEAT", never "DJ Mustard". Stripped forms shorter than 4
# characters are dropped so "Lil Baby" does not start matching the word "baby".
_PREFIXES = ("dj ", "lil ", "young ", "big ", "mc ", "the ")


def norm(s):
    """Lowercase ASCII, apostrophes deleted, everything else non-alphanumeric spaced."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    # Deleted, not spaced: "Pi'erre" must become "pierre" to match a title spelling it
    # "Pierre Bourne". Spacing it instead accused a correctly-labelled track.
    s = s.replace("'", "").replace("’", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def aliases(name):
    """Every spelling of an artist worth matching."""
    n = norm(name)
    forms = {n}
    for p in _PREFIXES:
        if n.startswith(p) and len(n[len(p):]) >= 4:
            forms.add(n[len(p):])
    return {f for f in forms if f}


def build_patterns(artists):
    """Precompiled word-boundary patterns per artist — build once, reuse per title."""
    return {a: [re.compile(rf"\b{re.escape(f)}\b") for f in aliases(a)] for a in artists}


def names_in(title, patterns):
    """Which artists the title mentions."""
    nt = norm(title)
    return [a for a, pats in patterns.items() if any(p.search(nt) for p in pats)]


def verdict(title, assigned, patterns):
    """
    "consistent" | "mismatch" | "undetermined" for one title against one label.

    undetermined is the common case (26% of the corpus) and is NOT treated as a
    mismatch: most type beat titles describe a vibe rather than naming anyone, and
    rejecting those would throw away a quarter of every search for no evidence.
    """
    nt = norm(title)
    if any(p.search(nt) for p in patterns.get(assigned, [])):
        return "consistent"
    others = [a for a, pats in patterns.items()
              if a != assigned
              and not (aliases(a) & AMBIGUOUS)
              and any(p.search(nt) for p in pats)]
    return "mismatch" if others else "undetermined"
