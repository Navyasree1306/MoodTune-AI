"""
music_matcher.py
-----------------
MODULE MAPPING: AI Problem Solving & Innovation

Core problem-solving step: given a target audio-feature profile and a
catalog of songs, find the best matches. Uses a weighted normalized
distance score (lower = better match) across valence, energy, tempo,
danceability, acousticness — a lightweight, explainable alternative to
a black-box recommender, deliberately chosen so every recommendation
can be justified (ties into the Ethics module's transparency goal).
"""

import pandas as pd

FEATURE_WEIGHTS = {
    "valence": 1.0,
    "energy": 1.0,
    "tempo": 0.6,       # tempo has a wider numeric range, so weighted down
    "danceability": 0.8,
    "acousticness": 0.8,
}

TEMPO_MAX = 200.0  # for normalization


def _load_catalog(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _distance(row, target: dict) -> float:
    d = 0.0
    for feat, weight in FEATURE_WEIGHTS.items():
        target_val = target[feat]
        row_val = row[feat]
        if feat == "tempo":
            target_val = target_val / TEMPO_MAX
            row_val = row_val / TEMPO_MAX
        d += weight * (row_val - target_val) ** 2
    return d ** 0.5


def recommend(csv_path: str, target_profile: dict, top_n: int = 5, diversity: bool = True,
              language: str = None) -> tuple:
    """
    Returns (recommendations_df, language_relaxed) where language_relaxed
    is True if the requested language filter had to be dropped because it
    didn't have enough matching tracks (an autonomous agent decision,
    logged by the caller rather than failing silently).
    """
    df = _load_catalog(csv_path)
    language_relaxed = False

    if language and language.lower() != "any":
        filtered = df[df["language"].str.lower() == language.lower()]
        if len(filtered) >= max(top_n, 3):
            df = filtered
        else:
            # Not enough tracks in that language to give a good result -
            # fall back to the full catalog rather than returning a thin
            # or empty recommendation list.
            language_relaxed = True

    df = df.copy()
    df["match_score"] = df.apply(lambda r: _distance(r, target_profile), axis=1)
    df = df.sort_values("match_score")

    if not diversity:
        return df.head(top_n).reset_index(drop=True), language_relaxed

    selected = []
    genre_counts = {}
    for _, row in df.iterrows():
        g = row["genre"]
        if genre_counts.get(g, 0) >= 2:
            continue
        selected.append(row)
        genre_counts[g] = genre_counts.get(g, 0) + 1
        if len(selected) >= top_n:
            break

    return pd.DataFrame(selected).reset_index(drop=True), language_relaxed
