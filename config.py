"""
BeatHunter AI - Configuration and Taxonomy Definition
This module defines the global paths, hyperparameters, and the core taxonomy 
(Sonic Brand / Mood / Genre) used for both data mining and model training.
"""

import os

# --- DIRECTORY CONFIGURATION ---
DATASET_ROOT = "dataset"
AUDIO_DIR = os.path.join(DATASET_ROOT, "audio")
METADATA_FILE = os.path.join(DATASET_ROOT, "metadata.csv")

# Target number of audio samples (YouTube videos) to scrape per category
FILES_PER_TARGET = 50

# ============================================================
# CURATED MASTER TAXONOMY (THE "GOLDEN LIST")
# 
# Adjustments applied to prevent multi-label collision/overfitting:
# - Dr. Dre: Consolidated to 'G-Funk' (removed duplicates from West Coast/Gangsta Rap).
# - Kendrick Lamar: Moved exclusively to 'Conscious Hip-Hop'.
# - J Dilla: Maintained only in 'Instrumental Hip-Hop'.
# - Havoc: Removed to prevent sonic collision with Mobb Deep.
# - Busta Rhymes: Mood corrected from 'Chill' to 'Energetic'.
# - Soudiere: Mood aligned with Kordhell as 'Aggressive' (Standard Phonk mood).
# ============================================================
TARGETS = [
    # --- ALTERNATIVE & EXPERIMENTAL ---
    {"artist": "Tyler, The Creator",    "mood": "Chill",        "genre": "Alternative Hip-Hop"},
    {"artist": "Outkast",               "mood": "Energetic",    "genre": "Alternative Hip-Hop"},
    {"artist": "Pharrell Williams",     "mood": "Energetic",    "genre": "Alternative Hip-Hop"},
    {"artist": "JPEGMAFIA",             "mood": "Aggressive",   "genre": "Experimental Hip-Hop"},
    {"artist": "Death Grips",           "mood": "Dark",         "genre": "Experimental Hip-Hop"},
    {"artist": "Childish Gambino",      "mood": "Chill",        "genre": "Hipster Hop"},

    # --- BOOM BAP ---
    {"artist": "DJ Premier",            "mood": "Soulful",      "genre": "Boom Bap"},
    {"artist": "Nas",                   "mood": "Soulful",      "genre": "Boom Bap"},
    {"artist": "Wu-Tang Clan",          "mood": "Dark",         "genre": "Boom Bap"},

    # --- BOUNCE & BRITISH ---
    {"artist": "Mannie Fresh",          "mood": "Energetic",    "genre": "Bounce"},
    {"artist": "Dave",                  "mood": "Melancholic",  "genre": "British Hip-Hop"},
    {"artist": "Giggs",                 "mood": "Dark",         "genre": "Road Rap"},
    {"artist": "Carns Hill",            "mood": "Dark",         "genre": "Road Rap"},

    # --- CHOPPED & CHOPPER ---
    {"artist": "DJ Screw",              "mood": "Chill",        "genre": "Chopped and Screwed"},
    {"artist": "Tech N9ne",             "mood": "Energetic",    "genre": "Chopper"},
    {"artist": "Twista",                "mood": "Energetic",    "genre": "Chopper"},
    {"artist": "Busta Rhymes",          "mood": "Energetic",    "genre": "Chopper"},

    # --- CLOUD RAP ---
    {"artist": "Clams Casino",          "mood": "Chill",        "genre": "Cloud Rap"},
    {"artist": "A$AP Rocky",            "mood": "Chill",        "genre": "Cloud Rap"},

    # --- NICHE SUBGENRES ---
    {"artist": "Lil Dicky",             "mood": "Energetic",    "genre": "Comedy Hip-Hop"},
    {"artist": "Lil Jon",               "mood": "Aggressive",   "genre": "Crunk"},
    {"artist": "dltzk",                 "mood": "Energetic",    "genre": "Digicore"},

    # --- DRILL ---
    {"artist": "Chief Keef",            "mood": "Aggressive",   "genre": "Chicago Drill"},
    {"artist": "King Von",              "mood": "Aggressive",   "genre": "Chicago Drill"},
    {"artist": "Young Chop",            "mood": "Aggressive",   "genre": "Chicago Drill"},
    {"artist": "Pop Smoke",             "mood": "Dark",         "genre": "Brooklyn Drill"},
    {"artist": "808Melo",               "mood": "Dark",         "genre": "Brooklyn Drill"},
    {"artist": "Central Cee",           "mood": "Dark",         "genre": "UK Drill"},
    {"artist": "Ghosty",                "mood": "Dark",         "genre": "UK Drill"},

    # --- EAST & WEST COAST ---
    {"artist": "Jay-Z",                 "mood": "Soulful",      "genre": "East Coast Hip-Hop"},
    {"artist": "The Notorious B.I.G.",  "mood": "Soulful",      "genre": "East Coast Hip-Hop"},
    {"artist": "RZA",                   "mood": "Dark",         "genre": "East Coast Hip-Hop"},
    {"artist": "2Pac",                  "mood": "Soulful",      "genre": "West Coast Hip-Hop"},
    {"artist": "Kendrick Lamar",        "mood": "Soulful",      "genre": "West Coast Hip-Hop"}, 
    {"artist": "MF DOOM",               "mood": "Soulful",      "genre": "Underground Hip-Hop"},
    {"artist": "Madlib",                "mood": "Soulful",      "genre": "Underground Hip-Hop"},

    # --- G-FUNK ---
    {"artist": "Dr. Dre",               "mood": "Chill",        "genre": "G-Funk"},
    {"artist": "Snoop Dogg",            "mood": "Chill",        "genre": "G-Funk"},
    {"artist": "Nate Dogg",             "mood": "Energetic",    "genre": "G-Funk"},

    # --- HARDCORE & GANGSTA ---
    {"artist": "Mobb Deep",             "mood": "Dark",         "genre": "Hardcore Hip-Hop"},
    {"artist": "Ice Cube",              "mood": "Aggressive",   "genre": "Gangsta Rap"},
    {"artist": "NWA",                   "mood": "Aggressive",   "genre": "Gangsta Rap"},
    {"artist": "Eazy-E",                "mood": "Aggressive",   "genre": "Gangsta Rap"},
    {"artist": "The Alchemist",         "mood": "Soulful",      "genre": "Mafioso Rap"},
    {"artist": "Raekwon",               "mood": "Dark",         "genre": "Mafioso Rap"},

    # --- MEMPHIS, SOUTHERN & LOFI ---
    {"artist": "Three 6 Mafia",         "mood": "Dark",         "genre": "Memphis Rap"},
    {"artist": "DJ Paul",               "mood": "Dark",         "genre": "Memphis Rap"},
    {"artist": "Rick Rock",             "mood": "Energetic",    "genre": "Hyphy"},
    {"artist": "J Dilla",               "mood": "Soulful",      "genre": "Instrumental Hip-Hop"},
    {"artist": "Cypress Hill",          "mood": "Chill",        "genre": "Stoner Rap"},
    {"artist": "Redman",                "mood": "Chill",        "genre": "Stoner Rap"},
    {"artist": "Method Man",            "mood": "Chill",        "genre": "Stoner Rap"},
    {"artist": "Scott Storch",          "mood": "Chill",        "genre": "Stoner Rap"},
    {"artist": "Nujabes",               "mood": "Chill",        "genre": "Lofi Hip-Hop"},
    {"artist": "Jinsang",               "mood": "Melancholic",  "genre": "Lofi Hip-Hop"},

    # --- MODERN TRAP, RAGE & PHONK ---
    {"artist": "Playboi Carti",         "mood": "Energetic",    "genre": "Mumble Rap"},
    {"artist": "Lil Pump",              "mood": "Energetic",    "genre": "Mumble Rap"},
    {"artist": "Pi'erre Bourne",        "mood": "Energetic",    "genre": "Mumble Rap"},
    {"artist": "Metro Boomin",          "mood": "Dark",         "genre": "Trap"},
    {"artist": "Future",                "mood": "Dark",         "genre": "Trap"},
    {"artist": "Migos",                 "mood": "Dark",         "genre": "Trap"},
    {"artist": "21 Savage",             "mood": "Dark",         "genre": "Trap"},
    {"artist": "Bad Bunny",             "mood": "Energetic",    "genre": "Latin Trap"},
    {"artist": "Tainy",                 "mood": "Energetic",    "genre": "Latin Trap"},
    {"artist": "Kordhell",              "mood": "Aggressive",   "genre": "Phonk"},
    {"artist": "Soudiere",              "mood": "Aggressive",   "genre": "Phonk"},
    {"artist": "Yeat",                  "mood": "Aggressive",   "genre": "Rage"},
    {"artist": "F1lthy",                "mood": "Aggressive",   "genre": "Rage"},

    # --- CONSCIOUS & PROGRESSIVE ---
    {"artist": "9th Wonder",            "mood": "Soulful",      "genre": "Conscious Hip-Hop"},
    {"artist": "Kanye West",            "mood": "Soulful",      "genre": "Progressive Rap"},
    {"artist": "T.I.",                  "mood": "Soulful",      "genre": "Southern Hip-Hop"},
    {"artist": "Organized Noize",       "mood": "Soulful",      "genre": "Southern Hip-Hop"},

    # --- FUSION GENRES ---
    {"artist": "Juice WRLD",            "mood": "Melancholic",  "genre": "Emo Rap"},
    {"artist": "Lil Peep",              "mood": "Melancholic",  "genre": "Emo Rap"},
    {"artist": "Erykah Badu",           "mood": "Chill",        "genre": "Neo Soul"},
    {"artist": "The Soulquarians",      "mood": "Soulful",      "genre": "Neo Soul"},
    {"artist": "El-P",                  "mood": "Aggressive",   "genre": "Industrial Hip-Hop"},
    {"artist": "A Tribe Called Quest",  "mood": "Chill",        "genre": "Jazz Rap"},
    {"artist": "Drake",                 "mood": "Chill",        "genre": "Pop Rap"},
    {"artist": "Post Malone",           "mood": "Chill",        "genre": "Pop Rap"},
    {"artist": "Travis Scott",          "mood": "Chill",        "genre": "Psychedelic Rap"},
    {"artist": "Mike Dean",             "mood": "Dark",         "genre": "Psychedelic Rap"},
    {"artist": "Machine Gun Kelly",     "mood": "Energetic",    "genre": "Punk Rap"},
    {"artist": "Linkin Park",           "mood": "Aggressive",   "genre": "Rap Rock"},
    {"artist": "Limp Bizkit",           "mood": "Aggressive",   "genre": "Nu Metal"},
    {"artist": "Ghostemane",            "mood": "Dark",         "genre": "Trap Metal"},
    {"artist": "Ronny J",               "mood": "Aggressive",   "genre": "Trap Metal"},
    {"artist": "Massive Attack",        "mood": "Melancholic",  "genre": "Trip Hop"},

    # --- ELECTRONIC & REGIONAL ---
    {"artist": "Skepta",                "mood": "Aggressive",   "genre": "Grime"},
    {"artist": "Wiley",                 "mood": "Energetic",    "genre": "Grime"},
    {"artist": "Daddy Yankee",          "mood": "Energetic",    "genre": "Reggaeton"},
    {"artist": "Luny Tunes",            "mood": "Energetic",    "genre": "Reggaeton"},
    {"artist": "Questlove",             "mood": "Soulful",      "genre": "Philadelphia Hip-Hop"},
    {"artist": "Eminem",                "mood": "Aggressive",   "genre": "Detroit Hip-Hop"},
    {"artist": "D12",                   "mood": "Aggressive",   "genre": "Detroit Hip-Hop"},
    {"artist": "Zaytoven",              "mood": "Energetic",    "genre": "Georgia Trap"},
    {"artist": "DJ Muggs",              "mood": "Energetic",    "genre": "Latino Hip-Hop"},
]

# Extract unique classes for the model's output layers
UNIQUE_GENRES = sorted(list(set(t["genre"] for t in TARGETS)))
UNIQUE_ARTISTS = sorted(list(set(t["artist"] for t in TARGETS)))
UNIQUE_MOODS = sorted(list(set(t["mood"] for t in TARGETS)))


# ============================================================
# DATA INTEGRITY CHECK: Catch duplicate artists
# ============================================================
if __name__ == "__main__":
    from collections import Counter
    artist_counts = Counter(t["artist"] for t in TARGETS)
    duplicates = {a: c for a, c in artist_counts.items() if c > 1}
    
    if duplicates:
        print("⚠️ WARNING — Duplicate artists found in taxonomy:")
        for a, c in duplicates.items():
            print(f"   {a}: {c} times")
    else:
        print("✅ Taxonomy is clean! No duplicate artists found.")
        
    print(f"\n📊 Total Targets: {len(TARGETS)}")
    print(f"   Unique Genres: {len(UNIQUE_GENRES)}")
    print(f"   Unique Artists: {len(UNIQUE_ARTISTS)}")
    print(f"   Unique Moods: {len(UNIQUE_MOODS)}")