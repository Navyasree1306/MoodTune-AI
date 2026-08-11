"""
mlops/monitor.py
-----------------
MLOps Week 3-4: Basic drift monitoring.

Real question this answers: is the text people are actually sending to
/predict starting to look different from the text the model was
TRAINED on? If so, accuracy silently degrades even though the API
keeps returning 200 OK - this is one of the most common ways ML
systems fail quietly in production.

This is a deliberately simple, from-scratch drift check (vocabulary
overlap + average confidence trend) rather than a heavyweight library,
so every part of it is auditable - the same design philosophy as
ethics_checker.py in the base project. (Evidently AI is a good
mention in your README as "what I'd reach for at larger scale.")

Run with:
    python mlops/monitor.py
"""

import os
import csv
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mood_detector import TRAINING_DATA

PREDICTION_LOG = os.path.join(os.path.dirname(__file__), "prediction_log.csv")
CONFIDENCE_DROP_THRESHOLD = 0.15  # flag if avg confidence drops this much vs baseline


def _tokenize(text: str) -> set:
    return set(text.lower().split())


def training_vocabulary() -> set:
    vocab = set()
    for text, _ in TRAINING_DATA:
        vocab |= _tokenize(text)
    return vocab


def load_predictions():
    if not os.path.exists(PREDICTION_LOG):
        return []
    with open(PREDICTION_LOG, "r") as f:
        return list(csv.DictReader(f))


def check_vocabulary_drift(predictions, train_vocab):
    """How much of the incoming query vocabulary is unseen during training?
    A rising unseen-word ratio over time is an early drift signal."""
    unseen_ratios = []
    for row in predictions:
        query_vocab = _tokenize(row["mood_text"])
        if not query_vocab:
            continue
        unseen = query_vocab - train_vocab
        unseen_ratios.append(len(unseen) / len(query_vocab))
    return unseen_ratios


def check_confidence_trend(predictions):
    confidences = [float(row["confidence"]) for row in predictions]
    if len(confidences) < 10:
        return None, None
    midpoint = len(confidences) // 2
    early_avg = sum(confidences[:midpoint]) / midpoint
    recent_avg = sum(confidences[midpoint:]) / (len(confidences) - midpoint)
    return early_avg, recent_avg


def check_mood_distribution(predictions):
    return Counter(row["predicted_mood"] for row in predictions)


def run_report():
    predictions = load_predictions()
    print("=" * 60)
    print("  MOODTUNE AI - DRIFT MONITORING REPORT")
    print("=" * 60)

    if not predictions:
        print("No predictions logged yet. Hit /predict a few times first,")
        print("then re-run this script.")
        return

    print(f"Total logged predictions: {len(predictions)}\n")

    # 1. Vocabulary drift
    train_vocab = training_vocabulary()
    unseen_ratios = check_vocabulary_drift(predictions, train_vocab)
    if unseen_ratios:
        avg_unseen = sum(unseen_ratios) / len(unseen_ratios)
        print(f"Avg. unseen-vocabulary ratio in incoming queries: {avg_unseen:.1%}")
        if avg_unseen > 0.5:
            print("  -> HIGH: incoming queries use very different language than "
                  "training data. Consider expanding TRAINING_DATA.")
        else:
            print("  -> OK: incoming queries broadly resemble training vocabulary.")

    # 2. Confidence trend
    early_avg, recent_avg = check_confidence_trend(predictions)
    if early_avg is not None:
        print(f"\nAvg. confidence - first half of log: {early_avg:.1%}")
        print(f"Avg. confidence - second half of log: {recent_avg:.1%}")
        if early_avg - recent_avg > CONFIDENCE_DROP_THRESHOLD:
            print("  -> FLAG: confidence has dropped meaningfully over time - "
                  "possible drift, worth investigating recent queries.")
        else:
            print("  -> OK: confidence stable over time.")
    else:
        print("\nNot enough predictions yet for a confidence trend (need 10+).")

    # 3. Mood distribution sanity check
    dist = check_mood_distribution(predictions)
    print(f"\nPredicted mood distribution: {dict(dist)}")

    print("=" * 60)


if __name__ == "__main__":
    run_report()
