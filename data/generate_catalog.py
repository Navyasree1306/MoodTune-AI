"""
generate_catalog.py
--------------------
Builds data/songs_catalog.csv — a local demo music catalog with
audio-feature-style attributes (valence, energy, tempo, danceability,
acousticness), used by the recommendation engine.

WHY A LOCAL DATASET INSTEAD OF A LIVE SPOTIFY API CALL?
Spotify deprecated its /recommendations and /audio-features endpoints for
all new developer apps in Nov 2024, and tightened the Web API further in
Feb 2026. Any new capstone project depending on those endpoints for audio
features will get a 403. Using a local catalog keeps the tool fully
functional, offline-friendly, and demo-safe. In a real deployment you'd
swap this module out for a licensed catalog / your own Spotify library
export (see README "Real-World Deployment" section).

Track and artist names below are synthetic (generated from word banks),
not real songs — this avoids attributing fabricated audio-feature values
to real artists.
"""

import random
import csv
import os

random.seed(42)

GENRE_PROFILES = {
    # genre: (valence_range, energy_range, tempo_range, danceability_range, acousticness_range)
    "Pop":        ((0.55, 0.9), (0.55, 0.85), (100, 130), (0.6, 0.9), (0.05, 0.3)),
    "EDM":        ((0.5, 0.95), (0.75, 1.0), (120, 150), (0.65, 0.95), (0.0, 0.1)),
    "Hip-Hop":    ((0.4, 0.8), (0.55, 0.9), (80, 115),  (0.65, 0.95), (0.05, 0.3)),
    "Rock":       ((0.35, 0.75), (0.65, 0.95), (110, 150), (0.4, 0.7), (0.05, 0.25)),
    "Metal":      ((0.15, 0.45), (0.85, 1.0), (120, 170), (0.3, 0.55), (0.0, 0.1)),
    "Lo-fi":      ((0.35, 0.6), (0.15, 0.4), (65, 90),   (0.4, 0.65), (0.4, 0.75)),
    "Ambient":    ((0.25, 0.5), (0.05, 0.25), (50, 80),  (0.2, 0.4),  (0.5, 0.9)),
    "Classical":  ((0.2, 0.6), (0.1, 0.45), (55, 110),   (0.15, 0.4), (0.75, 0.98)),
    "Jazz":       ((0.4, 0.7), (0.25, 0.55), (70, 120),  (0.35, 0.6), (0.4, 0.75)),
    "Acoustic":   ((0.3, 0.65), (0.15, 0.4), (70, 110),  (0.3, 0.55), (0.55, 0.9)),
    "Indie":      ((0.35, 0.7), (0.35, 0.65), (85, 125), (0.4, 0.65), (0.25, 0.6)),
    "Bollywood":  ((0.5, 0.9), (0.5, 0.85), (90, 135),   (0.55, 0.9), (0.1, 0.4)),
    "Romance":    ((0.45, 0.75), (0.2, 0.45), (65, 100), (0.3, 0.55), (0.35, 0.7)),
    "Devotional": ((0.4, 0.7), (0.1, 0.3), (55, 85),     (0.15, 0.35), (0.55, 0.9)),
}

ADJECTIVES = ["Golden", "Midnight", "Silent", "Electric", "Velvet", "Broken", "Neon",
              "Wild", "Quiet", "Distant", "Restless", "Faded", "Bright", "Hollow",
              "Endless", "Radiant", "Lonesome", "Fearless", "Gentle", "Fierce",
              "Amber", "Crimson", "Hazy", "Drifting", "Burning", "Frozen", "Free"]
NOUNS = ["Horizon", "Highway", "Echoes", "Skyline", "Reverie", "Static", "Bloom",
         "Signal", "Shadows", "Tides", "Embers", "Circuit", "Garden", "Voyage",
         "Fireflies", "Aftermath", "Wavelength", "Paradise", "Rebellion", "Serenade",
         "Compass", "Mirage", "Anthem", "Lullaby", "Momentum", "Constellation"]
FIRST_NAMES = ["Nova", "Aria", "Kai", "Rhea", "Ezra", "Mira", "Leo", "Sana", "Theo",
               "Ishaan", "Ananya", "Jude", "Iris", "Rohan", "Priya", "Kabir", "Noor",
               "Devika", "Arjun", "Selene"]
LAST_NAMES = ["Rey", "Fields", "Marsh", "Kapoor", "Vance", "Ray", "Sinclair", "Bloom",
              "Mehta", "Cross", "Lane", "Rios", "Chandra", "Wolfe", "Sen", "Hart"]

MOOD_TAGS_BY_GENRE = {
    "Pop": ["happy", "energetic"], "EDM": ["energetic", "happy"],
    "Hip-Hop": ["energetic", "focused"], "Rock": ["angry", "energetic"],
    "Metal": ["angry", "energetic"], "Lo-fi": ["calm", "focused"],
    "Ambient": ["calm", "sad"], "Classical": ["calm", "focused"],
    "Jazz": ["calm", "nostalgic"], "Acoustic": ["sad", "nostalgic"],
    "Indie": ["nostalgic", "calm"], "Bollywood": ["happy", "romantic"],
    "Romance": ["romantic", "calm"], "Devotional": ["calm", "nostalgic"],
}

LANGUAGES = ["English", "Hindi", "Telugu", "Tamil", "Punjabi", "Korean", "Spanish"]

# Rough per-genre language weighting so the catalog feels plausible
# (e.g. Bollywood skews Hindi/Punjabi, Devotional skews Hindi/Telugu/Tamil)
# rather than every genre having a uniform random language mix.
GENRE_LANGUAGE_WEIGHTS = {
    "Bollywood":  ["Hindi", "Hindi", "Punjabi", "Hindi", "English"],
    "Devotional": ["Hindi", "Telugu", "Tamil", "Hindi", "Hindi"],
    "Romance":    ["Hindi", "Telugu", "Tamil", "English", "Korean"],
    "Pop":        ["English", "English", "Korean", "Spanish", "Telugu"],
    "EDM":        ["English", "Korean", "English", "Spanish"],
    "Hip-Hop":    ["English", "English", "Punjabi", "Tamil"],
    "Rock":       ["English", "English", "Telugu", "Tamil"],
    "Metal":      ["English", "English", "Tamil"],
    "Lo-fi":      ["English", "Korean", "English", "Telugu"],
    "Ambient":    ["English", "English", "Korean"],
    "Classical":  ["Hindi", "Telugu", "Tamil", "English"],
    "Jazz":       ["English", "English", "Spanish"],
    "Acoustic":   ["English", "Telugu", "Tamil", "English"],
    "Indie":      ["English", "English", "Korean", "Tamil"],
}


def pick_language(genre):
    options = GENRE_LANGUAGE_WEIGHTS.get(genre, LANGUAGES)
    return random.choice(options)


def make_name(used):
    for _ in range(50):
        name = f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)}"
        if name not in used:
            used.add(name)
            return name
    return f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)} {random.randint(1,999)}"


def make_artist():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def generate(n_per_genre=9):
    rows = []
    used_names = set()
    for genre, (val_r, en_r, tempo_r, dance_r, acoustic_r) in GENRE_PROFILES.items():
        for _ in range(n_per_genre):
            track = make_name(used_names)
            artist = make_artist()
            valence = round(random.uniform(*val_r), 2)
            energy = round(random.uniform(*en_r), 2)
            tempo = round(random.uniform(*tempo_r), 1)
            dance = round(random.uniform(*dance_r), 2)
            acoustic = round(random.uniform(*acoustic_r), 2)
            language = pick_language(genre)
            mood_tags = ";".join(MOOD_TAGS_BY_GENRE[genre])
            rows.append([track, artist, genre, valence, energy, tempo, dance, acoustic, language, mood_tags])
    random.shuffle(rows)
    return rows


def main():
    rows = generate()
    out_path = os.path.join(os.path.dirname(__file__), "songs_catalog.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_name", "artist", "genre", "valence", "energy",
                          "tempo", "danceability", "acousticness", "language", "mood_tags"])
        writer.writerows(rows)
    print(f"Generated {len(rows)} tracks -> {out_path}")


if __name__ == "__main__":
    main()
