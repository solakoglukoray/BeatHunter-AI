"""
BeatHunter AI - Configuration and Taxonomy Definition (v2.0)

Changes in v2.0:
- FILES_PER_TARGET: 50 → 75 (~570K spectrograms total after preprocessing)
- Added genres: Afro Drill, Mexican Drill, Plugg, Pluggnb, Chillhop,
  Horrorcore, Crunkcore, Snap Music, Slab Music, Hip-Hop Soul,
  Country Rap, Chicano Rap, Political Hip-Hop
- All single-artist genres now have minimum 3 artists
- Scott Storch removed from Stoner Rap (incorrect assignment)
- Kendrick Lamar corrected: moved to Conscious Hip-Hop (was West Coast)
- Duplicate detection + thin genre detection in integrity check
"""

import os

# --- DIRECTORY CONFIGURATION ---
DATASET_ROOT  = "dataset"
AUDIO_DIR     = os.path.join(DATASET_ROOT, "audio")
METADATA_FILE = os.path.join(DATASET_ROOT, "metadata.csv")

# Increased from 50 → 75
# Estimated total: ~135 artists x 75 videos x ~58 segments = ~587,000 spectrograms
FILES_PER_TARGET = 75

# ============================================================
# CURATED MASTER TAXONOMY v2.0
#
# Rules enforced:
# - Every genre has minimum 3 artists
# - No artist appears more than once
# - Artists selected for strong YouTube "Type Beat" search signal
#
# "search_hint" WAS CALLED "mood" AND WAS NOT A MOOD (renamed 2026-07-30)
# ----------------------------------------------------------------------
# It is a hand-written word that data_collector.py pastes into the YouTube query
# ("Metro Boomin Dark Trap Type Beat"). For a while it was ALSO used as a third
# classification target, which was a mistake: nobody ever measured whether it matched
# the audio, and when finally measured it did not — Cohen d +0.023 (CNN embedding),
# +0.094 (MFCC), k-NN 1.4x chance. Three independent tests, all negative.
#
# It is kept, under an honest name, for exactly one reason: it is part of how the
# corpus was collected, so deleting it would make dataset/ unreproducible. It is not
# a label, it is not predicted, and nothing downstream may treat it as ground truth.
#
# Real mood now comes from the audio: audio_features.classify_mood() scores three
# axes (energy / brightness / aggression) against corpus percentiles measured by
# calibrate_mood.py. See that file for the axis-independence and genre checks.
# ============================================================
TARGETS = [

    # ================================================================
    # ALTERNATIVE & EXPERIMENTAL
    # ================================================================
    {"artist": "Tyler, The Creator",        "search_hint": "Chill",       "genre": "Alternative Hip-Hop"},
    {"artist": "Outkast",                   "search_hint": "Energetic",   "genre": "Alternative Hip-Hop"},
    {"artist": "Pharrell Williams",         "search_hint": "Energetic",   "genre": "Alternative Hip-Hop"},
    {"artist": "JPEGMAFIA",                 "search_hint": "Aggressive",  "genre": "Experimental Hip-Hop"},
    {"artist": "Death Grips",               "search_hint": "Dark",        "genre": "Experimental Hip-Hop"},
    {"artist": "clipping.",                 "search_hint": "Dark",        "genre": "Experimental Hip-Hop"},
    {"artist": "Childish Gambino",          "search_hint": "Chill",       "genre": "Hipster Hop"},
    {"artist": "Kid Cudi",                  "search_hint": "Melancholic", "genre": "Hipster Hop"},
    {"artist": "Emile Haynie",              "search_hint": "Chill",       "genre": "Hipster Hop"},

    # ================================================================
    # BOOM BAP
    # ================================================================
    {"artist": "DJ Premier",                "search_hint": "Soulful",     "genre": "Boom Bap"},
    {"artist": "Nas",                       "search_hint": "Soulful",     "genre": "Boom Bap"},
    {"artist": "Wu-Tang Clan",              "search_hint": "Dark",        "genre": "Boom Bap"},
    {"artist": "Pete Rock",                 "search_hint": "Soulful",     "genre": "Boom Bap"},

    # ================================================================
    # BOUNCE & BRITISH
    # ================================================================
    {"artist": "Mannie Fresh",              "search_hint": "Energetic",   "genre": "Bounce"},
    {"artist": "Big Freedia",               "search_hint": "Energetic",   "genre": "Bounce"},
    {"artist": "Juvenile",                  "search_hint": "Energetic",   "genre": "Bounce"},
    {"artist": "Dave",                      "search_hint": "Melancholic", "genre": "British Hip-Hop"},
    {"artist": "Little Simz",               "search_hint": "Energetic",   "genre": "British Hip-Hop"},
    {"artist": "Inflo",                     "search_hint": "Soulful",     "genre": "British Hip-Hop"},
    {"artist": "Giggs",                     "search_hint": "Dark",        "genre": "Road Rap"},
    {"artist": "Nines",                     "search_hint": "Dark",        "genre": "Road Rap"},
    {"artist": "Carns Hill",                "search_hint": "Dark",        "genre": "Road Rap"},

    # ================================================================
    # CHOPPED & CHOPPER
    # ================================================================
    {"artist": "DJ Screw",                  "search_hint": "Chill",       "genre": "Chopped and Screwed"},
    {"artist": "Paul Wall",                 "search_hint": "Chill",       "genre": "Chopped and Screwed"},
    {"artist": "Z-Ro",                      "search_hint": "Dark",        "genre": "Chopped and Screwed"},
    {"artist": "Tech N9ne",                 "search_hint": "Energetic",   "genre": "Chopper"},
    {"artist": "Twista",                    "search_hint": "Energetic",   "genre": "Chopper"},
    {"artist": "Busta Rhymes",              "search_hint": "Energetic",   "genre": "Chopper"},

    # ================================================================
    # CLOUD RAP
    # ================================================================
    {"artist": "Clams Casino",              "search_hint": "Chill",       "genre": "Cloud Rap"},
    {"artist": "A$AP Rocky",                "search_hint": "Chill",       "genre": "Cloud Rap"},
    {"artist": "Yung Lean",                 "search_hint": "Melancholic", "genre": "Cloud Rap"},

    # ================================================================
    # COMEDY HIP-HOP  (was single-artist — now 3)
    # ================================================================
    {"artist": "Lil Dicky",                 "search_hint": "Energetic",   "genre": "Comedy Hip-Hop"},
    {"artist": "The Lonely Island",         "search_hint": "Energetic",   "genre": "Comedy Hip-Hop"},
    {"artist": "Weird Al Yankovic",         "search_hint": "Energetic",   "genre": "Comedy Hip-Hop"},

    # ================================================================
    # CRUNK & CRUNKCORE
    # ================================================================
    {"artist": "Lil Jon",                   "search_hint": "Aggressive",  "genre": "Crunk"},
    {"artist": "Ying Yang Twins",           "search_hint": "Aggressive",  "genre": "Crunk"},
    {"artist": "Lil Scrappy",               "search_hint": "Aggressive",  "genre": "Crunk"},
    {"artist": "Brokencyde",                "search_hint": "Aggressive",  "genre": "Crunkcore"},
    {"artist": "3OH!3",                     "search_hint": "Aggressive",  "genre": "Crunkcore"},
    {"artist": "Hollywood Undead",          "search_hint": "Aggressive",  "genre": "Crunkcore"},

    # ================================================================
    # DIGICORE  (was single-artist — now 3)
    # ================================================================
    {"artist": "dltzk",                     "search_hint": "Energetic",   "genre": "Digicore"},
    {"artist": "glaive",                    "search_hint": "Energetic",   "genre": "Digicore"},
    {"artist": "ericdoa",                   "search_hint": "Energetic",   "genre": "Digicore"},

    # ================================================================
    # DRILL
    # ================================================================
    {"artist": "Chief Keef",                "search_hint": "Aggressive",  "genre": "Chicago Drill"},
    {"artist": "King Von",                  "search_hint": "Aggressive",  "genre": "Chicago Drill"},
    {"artist": "Young Chop",                "search_hint": "Aggressive",  "genre": "Chicago Drill"},
    {"artist": "Pop Smoke",                 "search_hint": "Dark",        "genre": "Brooklyn Drill"},
    {"artist": "808Melo",                   "search_hint": "Dark",        "genre": "Brooklyn Drill"},
    {"artist": "Fivio Foreign",             "search_hint": "Dark",        "genre": "Brooklyn Drill"},
    {"artist": "Central Cee",               "search_hint": "Dark",        "genre": "UK Drill"},
    {"artist": "Headie One",                "search_hint": "Dark",        "genre": "UK Drill"},
    {"artist": "Ghosty",                    "search_hint": "Dark",        "genre": "UK Drill"},
    {"artist": "Natanael Cano",             "search_hint": "Dark",        "genre": "Mexican Drill"},
    {"artist": "Junior H",                  "search_hint": "Dark",        "genre": "Mexican Drill"},
    {"artist": "Peso Pluma",                "search_hint": "Aggressive",  "genre": "Mexican Drill"},
    {"artist": "Rondodasosa",               "search_hint": "Dark",        "genre": "Afro Drill"},
    {"artist": "SL",                        "search_hint": "Aggressive",  "genre": "Afro Drill"},
    {"artist": "Unknown T",                 "search_hint": "Dark",        "genre": "Afro Drill"},

    # ================================================================
    # EAST & WEST COAST
    # ================================================================
    {"artist": "Jay-Z",                     "search_hint": "Soulful",     "genre": "East Coast Hip-Hop"},
    {"artist": "The Notorious B.I.G.",      "search_hint": "Soulful",     "genre": "East Coast Hip-Hop"},
    {"artist": "RZA",                       "search_hint": "Dark",        "genre": "East Coast Hip-Hop"},
    {"artist": "2Pac",                      "search_hint": "Soulful",     "genre": "West Coast Hip-Hop"},
    {"artist": "Dr. Dre",                   "search_hint": "Chill",       "genre": "West Coast Hip-Hop"},
    {"artist": "DJ Quik",                   "search_hint": "Chill",       "genre": "West Coast Hip-Hop"},
    {"artist": "MF DOOM",                   "search_hint": "Soulful",     "genre": "Underground Hip-Hop"},
    {"artist": "Madlib",                    "search_hint": "Soulful",     "genre": "Underground Hip-Hop"},
    {"artist": "Aesop Rock",                "search_hint": "Dark",        "genre": "Underground Hip-Hop"},

    # ================================================================
    # G-FUNK
    # ================================================================
    {"artist": "Snoop Dogg",                "search_hint": "Chill",       "genre": "G-Funk"},
    {"artist": "Nate Dogg",                 "search_hint": "Chill",       "genre": "G-Funk"},
    {"artist": "Warren G",                  "search_hint": "Chill",       "genre": "G-Funk"},

    # ================================================================
    # GANGSTA & HARDCORE
    # ================================================================
    {"artist": "Mobb Deep",                 "search_hint": "Dark",        "genre": "Hardcore Hip-Hop"},
    {"artist": "Onyx",                      "search_hint": "Aggressive",  "genre": "Hardcore Hip-Hop"},
    {"artist": "Snowgoons",                 "search_hint": "Dark",        "genre": "Hardcore Hip-Hop"},
    {"artist": "Ice Cube",                  "search_hint": "Aggressive",  "genre": "Gangsta Rap"},
    {"artist": "NWA",                       "search_hint": "Aggressive",  "genre": "Gangsta Rap"},
    {"artist": "Eazy-E",                    "search_hint": "Aggressive",  "genre": "Gangsta Rap"},
    {"artist": "The Alchemist",             "search_hint": "Soulful",     "genre": "Mafioso Rap"},
    {"artist": "Raekwon",                   "search_hint": "Dark",        "genre": "Mafioso Rap"},
    {"artist": "Kool G Rap",                "search_hint": "Dark",        "genre": "Mafioso Rap"},

    # ================================================================
    # HORRORCORE  (NEW)
    # ================================================================
    {"artist": "Insane Clown Posse",        "search_hint": "Dark",        "genre": "Horrorcore"},
    {"artist": "Gravediggaz",               "search_hint": "Dark",        "genre": "Horrorcore"},
    {"artist": "Brotha Lynch Hung",         "search_hint": "Dark",        "genre": "Horrorcore"},

    # ================================================================
    # HYPHY  (was single-artist — now 3)
    # ================================================================
    {"artist": "E-40",                      "search_hint": "Energetic",   "genre": "Hyphy"},
    {"artist": "Mac Dre",                   "search_hint": "Energetic",   "genre": "Hyphy"},
    {"artist": "Keak da Sneak",             "search_hint": "Energetic",   "genre": "Hyphy"},

    # ================================================================
    # LOFI & CHILL  (expanded + new subgenres)
    # ================================================================
    {"artist": "Nujabes",                   "search_hint": "Chill",       "genre": "Lofi Hip-Hop"},
    {"artist": "Jinsang",                   "search_hint": "Melancholic", "genre": "Lofi Hip-Hop"},
    {"artist": "Idealism",                  "search_hint": "Chill",       "genre": "Lofi Hip-Hop"},
    {"artist": "Philanthrope",              "search_hint": "Melancholic", "genre": "Lofi Hip-Hop"},
    {"artist": "Birocratic",                "search_hint": "Chill",       "genre": "Chillhop"},
    {"artist": "Tomppabeats",               "search_hint": "Chill",       "genre": "Chillhop"},
    {"artist": "Kupla",                     "search_hint": "Melancholic", "genre": "Chillhop"},
    {"artist": "Mexikodro",                 "search_hint": "Chill",       "genre": "Plugg"},
    {"artist": "Diego Money",               "search_hint": "Chill",       "genre": "Plugg"},
    {"artist": "Tony Shhnow",               "search_hint": "Chill",       "genre": "Plugg"},
    {"artist": "Goyxrd",                    "search_hint": "Melancholic", "genre": "Pluggnb"},
    {"artist": "Summrs",                    "search_hint": "Melancholic", "genre": "Pluggnb"},
    {"artist": "Autumn!",                   "search_hint": "Melancholic", "genre": "Pluggnb"},

    # ================================================================
    # MEMPHIS & SOUTHERN
    # ================================================================
    {"artist": "Three 6 Mafia",             "search_hint": "Dark",        "genre": "Memphis Rap"},
    {"artist": "DJ Paul",                   "search_hint": "Dark",        "genre": "Memphis Rap"},
    {"artist": "Project Pat",               "search_hint": "Dark",        "genre": "Memphis Rap"},
    {"artist": "Cypress Hill",              "search_hint": "Chill",       "genre": "Stoner Rap"},
    {"artist": "Redman",                    "search_hint": "Chill",       "genre": "Stoner Rap"},
    {"artist": "Method Man",                "search_hint": "Chill",       "genre": "Stoner Rap"},

    # ================================================================
    # SLAB MUSIC  (NEW)
    # ================================================================
    {"artist": "Lil Keke",                  "search_hint": "Chill",       "genre": "Slab Music"},
    {"artist": "Trae tha Truth",            "search_hint": "Dark",        "genre": "Slab Music"},
    {"artist": "Killa Kyleon",              "search_hint": "Dark",        "genre": "Slab Music"},

    # ================================================================
    # SNAP MUSIC  (NEW)
    # ================================================================
    {"artist": "D4L",                       "search_hint": "Energetic",   "genre": "Snap Music"},
    {"artist": "Dem Franchize Boyz",        "search_hint": "Energetic",   "genre": "Snap Music"},
    {"artist": "Mr. Collipark",             "search_hint": "Energetic",   "genre": "Snap Music"},

    # ================================================================
    # MODERN TRAP, RAGE & PHONK
    # ================================================================
    {"artist": "Playboi Carti",             "search_hint": "Energetic",   "genre": "Mumble Rap"},
    {"artist": "Lil Pump",                  "search_hint": "Energetic",   "genre": "Mumble Rap"},
    {"artist": "Pi'erre Bourne",            "search_hint": "Energetic",   "genre": "Mumble Rap"},
    {"artist": "Metro Boomin",              "search_hint": "Dark",        "genre": "Trap"},
    {"artist": "Future",                    "search_hint": "Dark",        "genre": "Trap"},
    {"artist": "Migos",                     "search_hint": "Dark",        "genre": "Trap"},
    {"artist": "21 Savage",                 "search_hint": "Dark",        "genre": "Trap"},
    {"artist": "Bad Bunny",                 "search_hint": "Energetic",   "genre": "Latin Trap"},
    {"artist": "Tainy",                     "search_hint": "Energetic",   "genre": "Latin Trap"},
    {"artist": "Anuel AA",                  "search_hint": "Energetic",   "genre": "Latin Trap"},
    {"artist": "Kordhell",                  "search_hint": "Aggressive",  "genre": "Phonk"},
    {"artist": "Soudiere",                  "search_hint": "Aggressive",  "genre": "Phonk"},
    {"artist": "Ghostface Playa",           "search_hint": "Aggressive",  "genre": "Phonk"},
    {"artist": "Yeat",                      "search_hint": "Aggressive",  "genre": "Rage"},
    {"artist": "F1lthy",                    "search_hint": "Aggressive",  "genre": "Rage"},
    {"artist": "Ken Carson",                "search_hint": "Aggressive",  "genre": "Rage"},

    # ================================================================
    # CONSCIOUS & PROGRESSIVE
    # ================================================================
    {"artist": "Kendrick Lamar",            "search_hint": "Soulful",     "genre": "Conscious Hip-Hop"},
    {"artist": "9th Wonder",                "search_hint": "Soulful",     "genre": "Conscious Hip-Hop"},
    {"artist": "Common",                    "search_hint": "Soulful",     "genre": "Conscious Hip-Hop"},
    {"artist": "Kanye West",                "search_hint": "Soulful",     "genre": "Progressive Rap"},
    {"artist": "Lupe Fiasco",               "search_hint": "Soulful",     "genre": "Progressive Rap"},
    {"artist": "J. Cole",                   "search_hint": "Soulful",     "genre": "Progressive Rap"},
    {"artist": "T.I.",                      "search_hint": "Soulful",     "genre": "Southern Hip-Hop"},
    {"artist": "Organized Noize",           "search_hint": "Soulful",     "genre": "Southern Hip-Hop"},
    {"artist": "UGK",                       "search_hint": "Soulful",     "genre": "Southern Hip-Hop"},

    # ================================================================
    # POLITICAL HIP-HOP  (NEW)
    # ================================================================
    {"artist": "Public Enemy",              "search_hint": "Aggressive",  "genre": "Political Hip-Hop"},
    {"artist": "dead prez",                 "search_hint": "Aggressive",  "genre": "Political Hip-Hop"},
    {"artist": "Immortal Technique",        "search_hint": "Dark",        "genre": "Political Hip-Hop"},

    # ================================================================
    # CHICANO RAP  (NEW)
    # ================================================================
    {"artist": "Kid Frost",                 "search_hint": "Aggressive",  "genre": "Chicano Rap"},
    {"artist": "Lil Rob",                   "search_hint": "Chill",       "genre": "Chicano Rap"},
    {"artist": "B-Real",                    "search_hint": "Aggressive",  "genre": "Chicano Rap"},

    # ================================================================
    # FUSION GENRES
    # ================================================================
    {"artist": "Juice WRLD",                "search_hint": "Melancholic", "genre": "Emo Rap"},
    {"artist": "Lil Peep",                  "search_hint": "Melancholic", "genre": "Emo Rap"},
    {"artist": "XXXTentacion",              "search_hint": "Melancholic", "genre": "Emo Rap"},
    {"artist": "Mary J. Blige",             "search_hint": "Soulful",     "genre": "Hip-Hop Soul"},
    {"artist": "Erykah Badu",               "search_hint": "Chill",       "genre": "Hip-Hop Soul"},
    {"artist": "Puff Daddy",                "search_hint": "Soulful",     "genre": "Hip-Hop Soul"},
    {"artist": "The Soulquarians",          "search_hint": "Soulful",     "genre": "Neo Soul"},
    {"artist": "D'Angelo",                  "search_hint": "Chill",       "genre": "Neo Soul"},
    {"artist": "Lauryn Hill",               "search_hint": "Soulful",     "genre": "Neo Soul"},
    {"artist": "El-P",                      "search_hint": "Aggressive",  "genre": "Industrial Hip-Hop"},
    {"artist": "Nine Inch Nails",           "search_hint": "Dark",        "genre": "Industrial Hip-Hop"},
    {"artist": "Ho99o9",                    "search_hint": "Aggressive",  "genre": "Industrial Hip-Hop"},
    {"artist": "A Tribe Called Quest",      "search_hint": "Chill",       "genre": "Jazz Rap"},
    {"artist": "Digable Planets",           "search_hint": "Chill",       "genre": "Jazz Rap"},
    {"artist": "Q-Tip",                     "search_hint": "Chill",       "genre": "Jazz Rap"},
    {"artist": "Drake",                     "search_hint": "Chill",       "genre": "Pop Rap"},
    {"artist": "Post Malone",               "search_hint": "Chill",       "genre": "Pop Rap"},
    {"artist": "Macklemore",                "search_hint": "Energetic",   "genre": "Pop Rap"},
    {"artist": "Travis Scott",              "search_hint": "Dark",        "genre": "Psychedelic Rap"},
    {"artist": "Flying Lotus",              "search_hint": "Dark",        "genre": "Psychedelic Rap"},
    {"artist": "BROCKHAMPTON",              "search_hint": "Energetic",   "genre": "Psychedelic Rap"},
    {"artist": "Machine Gun Kelly",         "search_hint": "Energetic",   "genre": "Punk Rap"},
    {"artist": "slowthai",                  "search_hint": "Aggressive",  "genre": "Punk Rap"},
    {"artist": "Fever 333",                 "search_hint": "Aggressive",  "genre": "Punk Rap"},
    {"artist": "Linkin Park",               "search_hint": "Aggressive",  "genre": "Rap Rock"},
    {"artist": "Rage Against the Machine",  "search_hint": "Aggressive",  "genre": "Rap Rock"},
    {"artist": "Run-DMC",                   "search_hint": "Energetic",   "genre": "Rap Rock"},
    {"artist": "Limp Bizkit",               "search_hint": "Aggressive",  "genre": "Nu Metal"},
    {"artist": "Korn",                      "search_hint": "Dark",        "genre": "Nu Metal"},
    {"artist": "Slipknot",                  "search_hint": "Aggressive",  "genre": "Nu Metal"},
    {"artist": "Ghostemane",                "search_hint": "Dark",        "genre": "Trap Metal"},
    {"artist": "Ronny J",                   "search_hint": "Aggressive",  "genre": "Trap Metal"},
    {"artist": "Scarlxrd",                  "search_hint": "Aggressive",  "genre": "Trap Metal"},
    {"artist": "Massive Attack",            "search_hint": "Melancholic", "genre": "Trip Hop"},
    {"artist": "Portishead",                "search_hint": "Melancholic", "genre": "Trip Hop"},
    {"artist": "Tricky",                    "search_hint": "Dark",        "genre": "Trip Hop"},
    {"artist": "Lil Nas X",                 "search_hint": "Energetic",   "genre": "Country Rap"},
    {"artist": "Bubba Sparxxx",             "search_hint": "Energetic",   "genre": "Country Rap"},
    {"artist": "Colt Ford",                 "search_hint": "Energetic",   "genre": "Country Rap"},

    # ================================================================
    # ELECTRONIC & REGIONAL
    # ================================================================
    {"artist": "Skepta",                    "search_hint": "Aggressive",  "genre": "Grime"},
    {"artist": "Wiley",                     "search_hint": "Energetic",   "genre": "Grime"},
    {"artist": "Stormzy",                   "search_hint": "Aggressive",  "genre": "Grime"},
    {"artist": "Daddy Yankee",              "search_hint": "Energetic",   "genre": "Reggaeton"},
    {"artist": "J Balvin",                  "search_hint": "Energetic",   "genre": "Reggaeton"},
    {"artist": "Luny Tunes",                "search_hint": "Energetic",   "genre": "Reggaeton"},
    {"artist": "Questlove",                 "search_hint": "Soulful",     "genre": "Philadelphia Hip-Hop"},
    {"artist": "Meek Mill",                 "search_hint": "Aggressive",  "genre": "Philadelphia Hip-Hop"},
    {"artist": "The Roots",                 "search_hint": "Soulful",     "genre": "Philadelphia Hip-Hop"},
    {"artist": "Eminem",                    "search_hint": "Aggressive",  "genre": "Detroit Hip-Hop"},
    {"artist": "Big Sean",                  "search_hint": "Energetic",   "genre": "Detroit Hip-Hop"},
    {"artist": "D12",                       "search_hint": "Aggressive",  "genre": "Detroit Hip-Hop"},
    {"artist": "Zaytoven",                  "search_hint": "Energetic",   "genre": "Georgia Trap"},
    {"artist": "Southside",                 "search_hint": "Dark",        "genre": "Georgia Trap"},
    {"artist": "Gucci Mane",                "search_hint": "Dark",        "genre": "Georgia Trap"},
    {"artist": "DJ Muggs",                  "search_hint": "Dark",        "genre": "Latino Hip-Hop"},
    {"artist": "Snow Tha Product",          "search_hint": "Aggressive",  "genre": "Latino Hip-Hop"},
    {"artist": "Delinquent Habits",         "search_hint": "Energetic",   "genre": "Latino Hip-Hop"},
    {"artist": "J Dilla",                   "search_hint": "Soulful",     "genre": "Instrumental Hip-Hop"},
    {"artist": "DJ Shadow",                 "search_hint": "Dark",        "genre": "Instrumental Hip-Hop"},
    {"artist": "Blockhead",                 "search_hint": "Chill",       "genre": "Instrumental Hip-Hop"},
]

# ============================================================
# AUTO-GENERATED CLASS LISTS (used by model trainer)
# ============================================================
UNIQUE_GENRES  = sorted(list(set(t["genre"]  for t in TARGETS)))
UNIQUE_ARTISTS = sorted(list(set(t["artist"] for t in TARGETS)))
# No UNIQUE_MOODS: search_hint is a scraper query term, not a class list. See above.


# ============================================================
# DATA INTEGRITY CHECK
# Run: python config.py
# ============================================================
if __name__ == "__main__":
    from collections import Counter

    print("=" * 55)
    print("BeatHunter AI — Taxonomy Integrity Report v2.0")
    print("=" * 55)

    artist_counts = Counter(t["artist"] for t in TARGETS)
    duplicates    = {a: c for a, c in artist_counts.items() if c > 1}

    if duplicates:
        print("\n⚠️  WARNING — Duplicate artists found:")
        for a, c in duplicates.items():
            print(f"   {a}: {c} times")
    else:
        print("\n✅ No duplicate artists found.")

    genre_counts = Counter(t["genre"] for t in TARGETS)
    thin_genres  = {g: c for g, c in genre_counts.items() if c < 3}

    if thin_genres:
        print("\n⚠️  WARNING — Genres with fewer than 3 artists:")
        for g, c in thin_genres.items():
            print(f"   {g}: {c} artist(s)")
    else:
        print("✅ All genres have 3+ artists.")

    print(f"\n📦 Summary:")
    print(f"   Total Targets:   {len(TARGETS)}")
    print(f"   Unique Genres:   {len(UNIQUE_GENRES)}")
    print(f"   Unique Artists:  {len(UNIQUE_ARTISTS)}")
    total_videos = len(TARGETS) * FILES_PER_TARGET
    total_specs  = total_videos * 58
    print(f"\n💾 Estimated Dataset Size:")
    print(f"   {len(TARGETS)} artists × {FILES_PER_TARGET} videos = {total_videos:,} videos")
    print(f"   {total_videos:,} × ~58 segments = ~{total_specs:,} spectrograms")