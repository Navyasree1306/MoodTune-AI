"""
ethics_checker.py
------------------
MODULE MAPPING: Ethics & Emerging Tech

Every AI system that ranks/filters content risks two things:
1. Bias — systematically over/under-representing certain genres or artists.
2. Opacity — the user can't tell *why* they got these results.

This module produces a short, honest report addressing both, run
automatically at the end of every session (not something the user has
to ask for). This is intentionally simple and rule-based — the point
of a capstone ethics layer is that it's auditable, not "AI-graded."
"""

import pandas as pd


def diversity_report(recommendations: pd.DataFrame) -> dict:
    genre_counts = recommendations["genre"].value_counts().to_dict()
    artist_counts = recommendations["artist"].value_counts().to_dict()
    max_share = max(genre_counts.values()) / len(recommendations) if len(recommendations) else 0
    return {
        "genre_distribution": genre_counts,
        "artist_distribution": artist_counts,
        "max_single_genre_share": round(max_share, 2),
        "diversity_flag": "OK" if max_share <= 0.6 else "LOW DIVERSITY - one genre dominates",
    }


def catalog_bias_report(full_catalog: pd.DataFrame) -> dict:
    """Checks whether the underlying catalog itself is skewed (a bias
    check on the data source, not just the output — important because
    a 'fair' algorithm on a skewed dataset still produces skewed results)."""
    genre_counts = full_catalog["genre"].value_counts()
    return {
        "total_tracks": len(full_catalog),
        "genres_covered": len(genre_counts),
        "min_genre_count": int(genre_counts.min()),
        "max_genre_count": int(genre_counts.max()),
        "balanced": bool(genre_counts.max() - genre_counts.min() <= 3),
    }


def transparency_note() -> str:
    return (
        "TRANSPARENCY & PRIVACY NOTE:\n"
        "- Mood is inferred from the text you typed in this session using a small "
        "trained classifier; it is a probabilistic estimate, not a diagnosis.\n"
        "- Recommendations come from a local demo catalog matched by audio-feature "
        "similarity (valence, energy, tempo, danceability, acousticness) - not from "
        "tracking your listening history or personal data.\n"
        "- No data from this session is sent anywhere or stored beyond the local "
        "report file generated on your own machine, unless you explicitly enable "
        "the optional LLM explanation feature (which sends only the mood label and "
        "song list, never personal identifiers, to the LLM API).\n"
        "- Limitation: mood classification accuracy is bounded by the small bundled "
        "training set; ambiguous or mixed-emotion text may be misclassified - the "
        "confidence score is shown so you can judge how much to trust it."
    )
