"""
decision_engine.py
-------------------
MODULE MAPPING: AI for Research & Smart Decision Making

Given a detected mood (+ optional context like activity/time of day),
this module builds a *target audio-feature profile* using a weighted
decision matrix — the "research and reasoning" step before any music
is chosen. This mirrors real decision-support systems: define criteria,
weigh them, produce a scored target state.
"""

# Base target profile per mood: (valence, energy, tempo, danceability, acousticness)
MOOD_TARGET_PROFILES = {
    "happy":     {"valence": 0.80, "energy": 0.70, "tempo": 115, "danceability": 0.75, "acousticness": 0.20},
    "sad":       {"valence": 0.25, "energy": 0.20, "tempo": 75,  "danceability": 0.35, "acousticness": 0.65},
    "energetic": {"valence": 0.70, "energy": 0.90, "tempo": 135, "danceability": 0.80, "acousticness": 0.05},
    "calm":      {"valence": 0.45, "energy": 0.15, "tempo": 65,  "danceability": 0.30, "acousticness": 0.80},
    "angry":     {"valence": 0.25, "energy": 0.90, "tempo": 140, "danceability": 0.45, "acousticness": 0.05},
    "romantic":  {"valence": 0.60, "energy": 0.30, "tempo": 80,  "danceability": 0.40, "acousticness": 0.55},
    "focused":   {"valence": 0.45, "energy": 0.30, "tempo": 80,  "danceability": 0.35, "acousticness": 0.55},
    "nostalgic": {"valence": 0.45, "energy": 0.30, "tempo": 90,  "danceability": 0.40, "acousticness": 0.60},
    # feel-good, danceable, high-tempo pop energy - distinct from "happy"
    # (contentment) by being specifically dance/party-oriented
    "upbeat":    {"valence": 0.88, "energy": 0.75, "tempo": 122, "danceability": 0.85, "acousticness": 0.10},
    # dreamy, confident, fantasy/main-character energy - between romantic
    # and upbeat: high valence, moderate energy, dreamy pop feel
    "delulu":    {"valence": 0.75, "energy": 0.55, "tempo": 106, "danceability": 0.65, "acousticness": 0.25},
}

# Context modifiers slightly nudge the target profile — this is the
# "smart decision making" layer: same mood, different context, different output.
CONTEXT_MODIFIERS = {
    "workout":   {"energy": +0.15, "tempo": +20, "danceability": +0.10},
    "study":     {"energy": -0.15, "acousticness": +0.15, "tempo": -10},
    "sleep":     {"energy": -0.25, "acousticness": +0.20, "tempo": -20, "valence": -0.05},
    "commute":   {"energy": +0.05},
    "party":     {"danceability": +0.15, "energy": +0.10, "tempo": +10},
    "none":      {},
}

FEATURE_KEYS = ["valence", "energy", "tempo", "danceability", "acousticness"]


def build_target_profile(mood: str, context: str = "none") -> dict:
    """Builds the target profile the recommender will search for."""
    base = dict(MOOD_TARGET_PROFILES.get(mood, MOOD_TARGET_PROFILES["calm"]))
    mods = CONTEXT_MODIFIERS.get(context, {})
    for key, delta in mods.items():
        if key == "tempo":
            base[key] = max(40, min(200, base[key] + delta))
        else:
            base[key] = max(0.0, min(1.0, round(base[key] + delta, 2)))
    return base


def explain_decision_matrix(mood: str, context: str, profile: dict) -> str:
    """Human-readable trace of the decision reasoning (for transparency/report)."""
    lines = [
        f"Base target profile for mood '{mood}': {MOOD_TARGET_PROFILES.get(mood)}",
    ]
    if context != "none" and context in CONTEXT_MODIFIERS and CONTEXT_MODIFIERS[context]:
        lines.append(f"Context '{context}' applied modifiers: {CONTEXT_MODIFIERS[context]}")
    lines.append(f"Final target profile: {profile}")
    return "\n".join(lines)
