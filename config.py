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
# - Mood reflects dominant sonic character of the artist's beats
# - Artists selected for strong YouTube "Type Beat" search signal
# ============================================================
TARGETS = [

    # ================================================================
    # ALTERNATIVE & EXPERIMENTAL
    # ================================================================
    {"artist": "Tyler, The Creator",    "mood": "Chill",        "genre": "Alternative Hip-Hop"},
    {"artist": "Outkast",               "mood": "Energetic",    "genre": "Alternative Hip-Hop"},
    {"artist": "Pharrell Williams",     "mood": "Energetic",    "genre": "Alternative Hip-Hop"},
    {"artist": "JPEGMAFIA",             "mood": "Aggressive",   "genre": "Experimental Hip-Hop"},
    {"artist": "Death Grips",           "mood": "Dark",         "genre": "Experimental Hip-Hop"},
    {"artist": "clipping.",             "mood": "Dark",         "genre": "Experimental Hip-Hop"},
    {"artist": "Childish Gambino",      "mood": "Chill",        "genre": "Hipster Hop"},
    {"artist": "Kid Cudi",              "mood": "Melancholic",  "genre": "Hipster Hop"},
    {"artist": "Emile Haynie",          "mood": "Chill",        "genre": "Hipster Hop"},

    # ================================================================
    # BOOM BAP
    # ================================================================
    {"artist": "DJ Premier",            "mood": "Soulful",      "genre": "Boom Bap"},
    {"artist": "Nas",                   "mood": "Soulful",      "genre": "Boom Bap"},
    {"artist": "Wu-Tang Clan",          "mood": "Dark",         "genre": "Boom Bap"},
    {"artist": "Pete Rock",             "mood": "Soulful",      "genre": "Boom Bap"},

    # ================================================================
    # BOUNCE & BRITISH
    # ================================================================
    {"artist": "Mannie Fresh",          "mood": "Energetic",    "genre": "Bounce"},
    {"artist": "Big Freedia",           "mood": "Energetic",    "genre": "Bounce"},
    {"artist": "Juvenile",              "mood": "Energetic",    "genre": "Bounce"},
    {"artist": "Dave",                  "mood": "Melancholic",  "genre": "British Hip-Hop"},
    {"artist": "Little Simz",           "mood": "Energetic",    "genre": "British Hip-Hop"},
    {"artist": "Inflo",                 "mood": "Soulful",      "genre": "British Hip-Hop"},
    {"artist": "Giggs",                 "mood": "Dark",         "genre": "Road Rap"},
    {"artist": "Nines",                 "mood": "Dark",         "genre": "Road Rap"},
    {"artist": "Carns Hill",            "mood": "Dark",         "genre": "Road Rap"},

    # ================================================================
    # CHOPPED & CHOPPER
    # ================================================================
    {"artist": "DJ Screw",              "mood": "Chill",        "genre": "Chopped and Screwed"},
    {"artist": "Paul Wall",             "mood": "Chill",        "genre": "Chopped and Screwed"},
    {"artist": "Z-Ro",                  "mood": "Dark",         "genre": "Chopped and Screwed"},
    {"artist": "Tech N9ne",             "mood": "Energetic",    "genre": "Chopper"},
    {"artist": "Twista",                "mood": "Energetic",    "genre": "Chopper"},
    {"artist": "Busta Rhymes",          "mood": "Energetic",    "genre": "Chopper"},

    # ================================================================
    # CLOUD RAP
    # ================================================================
    {"artist": "Clams Casino",          "mood": "Chill",        "genre": "Cloud Rap"},
    {"artist": "A$AP Rocky",            "mood": "Chill",        "genre": "Cloud Rap"},
    {"artist": "Yung Lean",             "mood": "Melancholic",  "genre": "Cloud Rap"},

    # ================================================================
    # COMEDY HIP-HOP  (was single-artist — now 3)
    # ================================================================
    {"artist": "Lil Dicky",             "mood": "Energetic",    "genre": "Comedy Hip-Hop"},
    {"artist": "The Lonely Island",     "mood": "Energetic",    "genre": "Comedy Hip-Hop"},
    {"artist": "Weird Al Yankovic",     "mood": "Energetic",    "genre": "Comedy Hip-Hop"},

    # ================================================================
    # CRUNK & CRUNKCORE
    # ================================================================
    {"artist": "Lil Jon",               "mood": "Aggressive",   "genre": "Crunk"},
    {"artist": "Ying Yang Twins",       "mood": "Aggressive",   "genre": "Crunk"},
    {"artist": "Lil Scrappy",           "mood": "Aggressive",   "genre": "Crunk"},
    {"artist": "Brokencyde",            "mood": "Aggressive",   "genre": "Crunkcore"},
    {"artist": "3OH!3",                 "mood": "Aggressive",   "genre": "Crunkcore"},
    {"artist": "Hollywood Undead",      "mood": "Aggressive",   "genre": "Crunkcore"},

    # ================================================================
    # DIGICORE  (was single-artist — now 3)
    # ================================================================
    {"artist": "dltzk",                 "mood": "Energetic",    "genre": "Digicore"},
    {"artist": "glaive",                "mood": "Energetic",    "genre": "Digicore"},
    {"artist": "ericdoa",               "mood": "Energetic",    "genre": "Digicore"},

    # ================================================================
    # DRILL
    # ================================================================
    {"artist": "Chief Keef",            "mood": "Aggressive",   "genre": "Chicago Drill"},
    {"artist": "King Von",              "mood": "Aggressive",   "genre": "Chicago Drill"},
    {"artist": "Young Chop",            "mood": "Aggressive",   "genre": "Chicago Drill"},
    {"artist": "Pop Smoke",             "mood": "Dark",         "genre": "Brooklyn Drill"},
    {"artist": "808Melo",               "mood": "Dark",         "genre": "Brooklyn Drill"},
    {"artist": "Fivio Foreign",         "mood": "Dark",         "genre": "Brooklyn Drill"},
    {"artist": "Central Cee",           "mood": "Dark",         "genre": "UK Drill"},
    {"artist": "Headie One",            "mood": "Dark",         "genre": "UK Drill"},
    {"artist": "Ghosty",                "mood": "Dark",         "genre": "UK Drill"},
    {"artist": "Natanael Cano",         "mood": "Dark",         "genre": "Mexican Drill"},
    {"artist": "Junior H",              "mood": "Dark",         "genre": "Mexican Drill"},
    {"artist": "Peso Pluma",            "mood": "Aggressive",   "genre": "Mexican Drill"},
    {"artist": "Rondodasosa",           "mood": "Dark",         "genre": "Afro Drill"},
    {"artist": "SL",                    "mood": "Aggressive",   "genre": "Afro Drill"},
    {"artist": "Unknown T",             "mood": "Dark",         "genre": "Afro Drill"},

    # ================================================================
    # EAST & WEST COAST
    # ================================================================
    {"artist": "Jay-Z",                 "mood": "Soulful",      "genre": "East Coast Hip-Hop"},
    {"artist": "The Notorious B.I.G.",  "mood": "Soulful",      "genre": "East Coast Hip-Hop"},
    {"artist": "RZA",                   "mood": "Dark",         "genre": "East Coast Hip-Hop"},
    {"artist": "2Pac",                  "mood": "Soulful",      "genre": "West Coast Hip-Hop"},
    {"artist": "Dr. Dre",               "mood": "Chill",        "genre": "West Coast Hip-Hop"},
    {"artist": "DJ Quik",               "mood": "Chill",        "genre": "West Coast Hip-Hop"},
    {"artist": "MF DOOM",               "mood": "Soulful",      "genre": "Underground Hip-Hop"},
    {"artist": "Madlib",                "mood": "Soulful",      "genre": "Underground Hip-Hop"},
    {"artist": "Aesop Rock",            "mood": "Dark",         "genre": "Underground Hip-Hop"},

    # ================================================================
    # G-FUNK
    # ================================================================
    {"artist": "Snoop Dogg",            "mood": "Chill",        "genre": "G-Funk"},
    {"artist": "Nate Dogg",             "mood": "Chill",        "genre": "G-Funk"},
    {"artist": "Warren G",              "mood": "Chill",        "genre": "G-Funk"},

    # ================================================================
    # GANGSTA & HARDCORE
    # ================================================================
    {"artist": "Mobb Deep",             "mood": "Dark",         "genre": "Hardcore Hip-Hop"},
    {"artist": "Onyx",                  "mood": "Aggressive",   "genre": "Hardcore Hip-Hop"},
    {"artist": "Snowgoons",             "mood": "Dark",         "genre": "Hardcore Hip-Hop"},
    {"artist": "Ice Cube",              "mood": "Aggressive",   "genre": "Gangsta Rap"},
    {"artist": "NWA",                   "mood": "Aggressive",   "genre": "Gangsta Rap"},
    {"artist": "Eazy-E",                "mood": "Aggressive",   "genre": "Gangsta Rap"},
    {"artist": "The Alchemist",         "mood": "Soulful",      "genre": "Mafioso Rap"},
    {"artist": "Raekwon",               "mood": "Dark",         "genre": "Mafioso Rap"},
    {"artist": "Kool G Rap",            "mood": "Dark",         "genre": "Mafioso Rap"},

    # ================================================================
    # HORRORCORE  (NEW)
    # ================================================================
    {"artist": "Insane Clown Posse",    "mood": "Dark",         "genre": "Horrorcore"},
    {"artist": "Gravediggaz",           "mood": "Dark",         "genre": "Horrorcore"},
    {"artist": "Brotha Lynch Hung",     "mood": "Dark",         "genre": "Horrorcore"},

    # ================================================================
    # HYPHY  (was single-artist — now 3)
    # ================================================================
    {"artist": "E-40",                  "mood": "Energetic",    "genre": "Hyphy"},
    {"artist": "Mac Dre",               "mood": "Energetic",    "genre": "Hyphy"},
    {"artist": "Keak da Sneak",         "mood": "Energetic",    "genre": "Hyphy"},

    # ================================================================
    # LOFI & CHILL  (expanded + new subgenres)
    # ================================================================
    {"artist": "Nujabes",               "mood": "Chill",        "genre": "Lofi Hip-Hop"},
    {"artist": "Jinsang",               "mood": "Melancholic",  "genre": "Lofi Hip-Hop"},
    {"artist": "Idealism",              "mood": "Chill",        "genre": "Lofi Hip-Hop"},
    {"artist": "Philanthrope",          "mood": "Melancholic",  "genre": "Lofi Hip-Hop"},
    {"artist": "Birocratic",            "mood": "Chill",        "genre": "Chillhop"},
    {"artist": "Tomppabeats",           "mood": "Chill",        "genre": "Chillhop"},
    {"artist": "Kupla",                 "mood": "Melancholic",  "genre": "Chillhop"},
    {"artist": "Mexikodro",             "mood": "Chill",        "genre": "Plugg"},
    {"artist": "Diego Money",           "mood": "Chill",        "genre": "Plugg"},
    {"artist": "Tony Shhnow",           "mood": "Chill",        "genre": "Plugg"},
    {"artist": "Goyxrd",                "mood": "Melancholic",  "genre": "Pluggnb"},
    {"artist": "Summrs",                "mood": "Melancholic",  "genre": "Pluggnb"},
    {"artist": "Autumn!",               "mood": "Melancholic",  "genre": "Pluggnb"},

    # ================================================================
    # MEMPHIS & SOUTHERN
    # ================================================================
    {"artist": "Three 6 Mafia",         "mood": "Dark",         "genre": "Memphis Rap"},
    {"artist": "DJ Paul",               "mood": "Dark",         "genre": "Memphis Rap"},
    {"artist": "Project Pat",           "mood": "Dark",         "genre": "Memphis Rap"},
    {"artist": "Cypress Hill",          "mood": "Chill",        "genre": "Stoner Rap"},
    {"artist": "Redman",                "mood": "Chill",        "genre": "Stoner Rap"},
    {"artist": "Method Man",            "mood": "Chill",        "genre": "Stoner Rap"},

    # ================================================================
    # SLAB MUSIC  (NEW)
    # ================================================================
    {"artist": "Lil Keke",              "mood": "Chill",        "genre": "Slab Music"},
    {"artist": "Trae tha Truth",        "mood": "Dark",         "genre": "Slab Music"},
    {"artist": "Killa Kyleon",          "mood": "Dark",         "genre": "Slab Music"},

    # ================================================================
    # SNAP MUSIC  (NEW)
    # ================================================================
    {"artist": "D4L",                   "mood": "Energetic",    "genre": "Snap Music"},
    {"artist": "Dem Franchize Boyz",    "mood": "Energetic",    "genre": "Snap Music"},
    {"artist": "Mr. Collipark",         "mood": "Energetic",    "genre": "Snap Music"},

    # ================================================================
    # MODERN TRAP, RAGE & PHONK
    # ================================================================
    {"artist": "Playboi Carti",         "mood": "Energetic",    "genre": "Mumble Rap"},
    {"artist": "Lil Pump",              "mood": "Energetic",    "genre": "Mumble Rap"},
    {"artist": "Pi'erre Bourne",        "mood": "Energetic",    "genre": "Mumble Rap"},
    {"artist": "Metro Boomin",          "mood": "Dark",         "genre": "Trap"},
    {"artist": "Future",                "mood": "Dark",         "genre": "Trap"},
    {"artist": "Migos",                 "mood": "Dark",         "genre": "Trap"},
    {"artist": "21 Savage",             "mood": "Dark",         "genre": "Trap"},
    {"artist": "Bad Bunny",             "mood": "Energetic",    "genre": "Latin Trap"},
    {"artist": "Tainy",                 "mood": "Energetic",    "genre": "Latin Trap"},
    {"artist": "Anuel AA",              "mood": "Energetic",    "genre": "Latin Trap"},
    {"artist": "Kordhell",              "mood": "Aggressive",   "genre": "Phonk"},
    {"artist": "Soudiere",              "mood": "Aggressive",   "genre": "Phonk"},
    {"artist": "Ghostface Playa",       "mood": "Aggressive",   "genre": "Phonk"},
    {"artist": "Yeat",                  "mood": "Aggressive",   "genre": "Rage"},
    {"artist": "F1lthy",                "mood": "Aggressive",   "genre": "Rage"},
    {"artist": "Ken Carson",            "mood": "Aggressive",   "genre": "Rage"},

    # ================================================================
    # CONSCIOUS & PROGRESSIVE
    # ================================================================
    {"artist": "Kendrick Lamar",        "mood": "Soulful",      "genre": "Conscious Hip-Hop"},
    {"artist": "9th Wonder",            "mood": "Soulful",      "genre": "Conscious Hip-Hop"},
    {"artist": "Common",                "mood": "Soulful",      "genre": "Conscious Hip-Hop"},
    {"artist": "Kanye West",            "mood": "Soulful",      "genre": "Progressive Rap"},
    {"artist": "Lupe Fiasco",           "mood": "Soulful",      "genre": "Progressive Rap"},
    {"artist": "J. Cole",               "mood": "Soulful",      "genre": "Progressive Rap"},
    {"artist": "T.I.",                  "mood": "Soulful",      "genre": "Southern Hip-Hop"},
    {"artist": "Organized Noize",       "mood": "Soulful",      "genre": "Southern Hip-Hop"},
    {"artist": "UGK",                   "mood": "Soulful",      "genre": "Southern Hip-Hop"},

    # ================================================================
    # POLITICAL HIP-HOP  (NEW)
    # ================================================================
    {"artist": "Public Enemy",          "mood": "Aggressive",   "genre": "Political Hip-Hop"},
    {"artist": "dead prez",             "mood": "Aggressive",   "genre": "Political Hip-Hop"},
    {"artist": "Immortal Technique",    "mood": "Dark",         "genre": "Political Hip-Hop"},

    # ================================================================
    # CHICANO RAP  (NEW)
    # ================================================================
    {"artist": "Kid Frost",             "mood": "Aggressive",   "genre": "Chicano Rap"},
    {"artist": "Lil Rob",               "mood": "Chill",        "genre": "Chicano Rap"},
    {"artist": "B-Real",                "mood": "Aggressive",   "genre": "Chicano Rap"},

    # ================================================================
    # FUSION GENRES
    # ================================================================
    {"artist": "Juice WRLD",            "mood": "Melancholic",  "genre": "Emo Rap"},
    {"artist": "Lil Peep",              "mood": "Melancholic",  "genre": "Emo Rap"},
    {"artist": "XXXTentacion",          "mood": "Melancholic",  "genre": "Emo Rap"},
    {"artist": "Mary J. Blige",         "mood": "Soulful",      "genre": "Hip-Hop Soul"},
    {"artist": "Erykah Badu",           "mood": "Chill",        "genre": "Hip-Hop Soul"},
    {"artist": "Puff Daddy",            "mood": "Soulful",      "genre": "Hip-Hop Soul"},
    {"artist": "The Soulquarians",      "mood": "Soulful",      "genre": "Neo Soul"},
    {"artist": "D'Angelo",              "mood": "Chill",        "genre": "Neo Soul"},
    {"artist": "Lauryn Hill",           "mood": "Soulful",      "genre": "Neo Soul"},
    {"artist": "El-P",                  "mood": "Aggressive",   "genre": "Industrial Hip-Hop"},
    {"artist": "Nine Inch Nails",       "mood": "Dark",         "genre": "Industrial Hip-Hop"},
    {"artist": "Ho99o9",                "mood": "Aggressive",   "genre": "Industrial Hip-Hop"},
    {"artist": "A Tribe Called Quest",  "mood": "Chill",        "genre": "Jazz Rap"},
    {"artist": "Digable Planets",       "mood": "Chill",        "genre": "Jazz Rap"},
    {"artist": "Q-Tip",                 "mood": "Chill",        "genre": "Jazz Rap"},
    {"artist": "Drake",                 "mood": "Chill",        "genre": "Pop Rap"},
    {"artist": "Post Malone",           "mood": "Chill",        "genre": "Pop Rap"},
    {"artist": "Macklemore",            "mood": "Energetic",    "genre": "Pop Rap"},
    {"artist": "Travis Scott",          "mood": "Dark",         "genre": "Psychedelic Rap"},
    {"artist": "Flying Lotus",          "mood": "Dark",         "genre": "Psychedelic Rap"},
    {"artist": "BROCKHAMPTON",          "mood": "Energetic",    "genre": "Psychedelic Rap"},
    {"artist": "Machine Gun Kelly",     "mood": "Energetic",    "genre": "Punk Rap"},
    {"artist": "slowthai",              "mood": "Aggressive",   "genre": "Punk Rap"},
    {"artist": "Fever 333",             "mood": "Aggressive",   "genre": "Punk Rap"},
    {"artist": "Linkin Park",           "mood": "Aggressive",   "genre": "Rap Rock"},
    {"artist": "Rage Against the Machine", "mood": "Aggressive","genre": "Rap Rock"},
    {"artist": "Run-DMC",               "mood": "Energetic",    "genre": "Rap Rock"},
    {"artist": "Limp Bizkit",           "mood": "Aggressive",   "genre": "Nu Metal"},
    {"artist": "Korn",                  "mood": "Dark",         "genre": "Nu Metal"},
    {"artist": "Slipknot",              "mood": "Aggressive",   "genre": "Nu Metal"},
    {"artist": "Ghostemane",            "mood": "Dark",         "genre": "Trap Metal"},
    {"artist": "Ronny J",               "mood": "Aggressive",   "genre": "Trap Metal"},
    {"artist": "Scarlxrd",              "mood": "Aggressive",   "genre": "Trap Metal"},
    {"artist": "Massive Attack",        "mood": "Melancholic",  "genre": "Trip Hop"},
    {"artist": "Portishead",            "mood": "Melancholic",  "genre": "Trip Hop"},
    {"artist": "Tricky",                "mood": "Dark",         "genre": "Trip Hop"},
    {"artist": "Lil Nas X",             "mood": "Energetic",    "genre": "Country Rap"},
    {"artist": "Bubba Sparxxx",         "mood": "Energetic",    "genre": "Country Rap"},
    {"artist": "Colt Ford",             "mood": "Energetic",    "genre": "Country Rap"},

    # ================================================================
    # ELECTRONIC & REGIONAL
    # ================================================================
    {"artist": "Skepta",                "mood": "Aggressive",   "genre": "Grime"},
    {"artist": "Wiley",                 "mood": "Energetic",    "genre": "Grime"},
    {"artist": "Stormzy",               "mood": "Aggressive",   "genre": "Grime"},
    {"artist": "Daddy Yankee",          "mood": "Energetic",    "genre": "Reggaeton"},
    {"artist": "J Balvin",              "mood": "Energetic",    "genre": "Reggaeton"},
    {"artist": "Luny Tunes",            "mood": "Energetic",    "genre": "Reggaeton"},
    {"artist": "Questlove",             "mood": "Soulful",      "genre": "Philadelphia Hip-Hop"},
    {"artist": "Meek Mill",             "mood": "Aggressive",   "genre": "Philadelphia Hip-Hop"},
    {"artist": "The Roots",             "mood": "Soulful",      "genre": "Philadelphia Hip-Hop"},
    {"artist": "Eminem",                "mood": "Aggressive",   "genre": "Detroit Hip-Hop"},
    {"artist": "Big Sean",              "mood": "Energetic",    "genre": "Detroit Hip-Hop"},
    {"artist": "D12",                   "mood": "Aggressive",   "genre": "Detroit Hip-Hop"},
    {"artist": "Zaytoven",              "mood": "Energetic",    "genre": "Georgia Trap"},
    {"artist": "Southside",             "mood": "Dark",         "genre": "Georgia Trap"},
    {"artist": "Gucci Mane",            "mood": "Dark",         "genre": "Georgia Trap"},
    {"artist": "DJ Muggs",              "mood": "Dark",         "genre": "Latino Hip-Hop"},
    {"artist": "Snow Tha Product",      "mood": "Aggressive",   "genre": "Latino Hip-Hop"},
    {"artist": "Delinquent Habits",     "mood": "Energetic",    "genre": "Latino Hip-Hop"},
    {"artist": "J Dilla",               "mood": "Soulful",      "genre": "Instrumental Hip-Hop"},
    {"artist": "DJ Shadow",             "mood": "Dark",         "genre": "Instrumental Hip-Hop"},
    {"artist": "Blockhead",             "mood": "Chill",        "genre": "Instrumental Hip-Hop"},
]

# ============================================================
# AUTO-GENERATED CLASS LISTS (used by model trainer)
# ============================================================
UNIQUE_GENRES  = sorted(list(set(t["genre"]  for t in TARGETS)))
UNIQUE_ARTISTS = sorted(list(set(t["artist"] for t in TARGETS)))
UNIQUE_MOODS   = sorted(list(set(t["mood"]   for t in TARGETS)))


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

    mood_counts = Counter(t["mood"] for t in TARGETS)
    print("\n📊 Mood Distribution:")
    for mood, count in sorted(mood_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 2)
        print(f"   {mood:<15} {count:>3}  {bar}")

    print(f"\n📦 Summary:")
    print(f"   Total Targets:   {len(TARGETS)}")
    print(f"   Unique Genres:   {len(UNIQUE_GENRES)}")
    print(f"   Unique Artists:  {len(UNIQUE_ARTISTS)}")
    print(f"   Unique Moods:    {len(UNIQUE_MOODS)}")
    total_videos = len(TARGETS) * FILES_PER_TARGET
    total_specs  = total_videos * 58
    print(f"\n💾 Estimated Dataset Size:")
    print(f"   {len(TARGETS)} artists × {FILES_PER_TARGET} videos = {total_videos:,} videos")
    print(f"   {total_videos:,} × ~58 segments = ~{total_specs:,} spectrograms")