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


# Members and alternate names for targets that are groups or go by more than one name.
#
# Without this the matcher accuses correct labels: "ANDRE 3000 X ERYKAH BADU TYPE BEAT"
# against an Outkast label looks like a mismatch, but Andre 3000 IS Outkast. It hit 44
# of 1,307 planned re-labels (3.4%) — every one of which would have corrupted a row that
# was already right. Same trap in the collector, which would have thrown the result away.
MEMBERS = {
    "Outkast":        ["andre 3000", "andre3000", "big boi", "3 stacks"],
    "NWA":            ["eazy e", "ice cube", "dr dre", "mc ren", "dj yella", "n w a"],
    "Wu-Tang Clan":   ["rza", "gza", "method man", "raekwon", "ghostface",
                       "ol dirty bastard", "odb", "inspectah deck", "wu tang"],
    "Migos":          ["quavo", "offset", "takeoff"],
    "The Roots":      ["black thought", "questlove"],
    "Mobb Deep":      ["prodigy", "havoc"],
    "D12":            ["proof", "bizarre"],
    "Three 6 Mafia":  ["dj paul", "juicy j", "project pat", "lord infamous"],
    "A Tribe Called Quest": ["q tip", "phife dawg", "atcq"],
    "The Soulquarians": ["dangelo", "j dilla", "questlove", "erykah badu", "common"],
    "Digable Planets": ["butterfly", "ladybug mecca"],
    "BROCKHAMPTON":   ["kevin abstract"],
    "Rage Against the Machine": ["zack de la rocha", "tom morello", "ratm"],
    "Ying Yang Twins": ["dlow", "kaine"],
    "Onyx":           ["fredro starr", "sticky fingaz"],
    "Death Grips":    ["mc ride"],
}
def _own_target_names():
    """Normalised names of every artist that has its own label in the taxonomy."""
    try:
        import config
        return {norm(t["artist"]) for t in config.TARGETS}
    except Exception:
        return set()


# A member only stands in for the group when they do NOT have their own label.
#
# Half of the list above collides with the taxonomy: Dr. Dre, Ice Cube and Eazy-E are
# each separate targets as well as NWA members, RZA and Raekwon as well as Wu-Tang,
# Questlove as well as The Roots. Counting those as the group makes the group swallow
# its own members' classes — a "Dr. Dre type beat" would validate an NWA label forever,
# which is worse than the mismatch this map was added to prevent. When the member has
# their own target, naming them is evidence FOR that target, not for the group.
_OWN = _own_target_names()
_MEMBER_PATTERNS = {
    a: [re.compile(rf"\b{re.escape(m)}\b") for m in ms if norm(m) not in _OWN]
    for a, ms in MEMBERS.items()
}


def names_member_of(title, artist):
    """True when the title names someone who IS the assigned act and has no label of
    their own."""
    nt = norm(title)
    return any(p.search(nt) for p in _MEMBER_PATTERNS.get(artist, []))


# A corpus of instrumentals should not contain released songs, official videos or
# karaoke tracks — 9.6% of it does. These markers are what a type beat upload almost
# always carries in its title.
_BEAT_MARKERS = ("type beat", "instrumental", "prod ", "prod.", "beat)", "[free",
                 "(free", "free)", "beats", "loop kit", "sample pack", "beat |",
                 "beat -")


def looks_like_type_beat(title):
    t = str(title).lower()
    return any(m in t for m in _BEAT_MARKERS)


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
    # A group's member counts as the group. Checked before the accusation, not after.
    if names_member_of(title, assigned):
        return "consistent"
    others = [a for a, pats in patterns.items()
              if a != assigned
              and not (aliases(a) & AMBIGUOUS)
              and any(p.search(nt) for p in pats)]
    return "mismatch" if others else "undetermined"
