"""
evaluate_classifier.py
------------------------
Measures the mood classifier's performance using stratified 5-fold
cross-validation over the bundled training set (appropriate for a small
~64-example dataset, where a single train/test split would be too noisy
to trust). Prints accuracy, per-class F1, and a confusion matrix.

Run: python3 evaluate_classifier.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np

from mood_detector import TRAINING_DATA, MOODS


def main():
    texts = [t for t, _ in TRAINING_DATA]
    labels = [m for _, m in TRAINING_DATA]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    model = LogisticRegression(max_iter=2000, C=8.0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    preds = cross_val_predict(model, X, y, cv=skf)

    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")

    print("=" * 60)
    print("MOOD CLASSIFIER EVALUATION (5-fold stratified cross-validation)")
    print("=" * 60)
    print(f"Dataset size: {len(y)} examples across {len(MOODS)} mood classes")
    print(f"Overall accuracy: {acc:.1%}")
    print(f"Macro-averaged F1: {macro_f1:.3f}")
    print("\nPer-class report:")
    print(classification_report(y, preds, labels=MOODS, zero_division=0))

    print("Confusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y, preds, labels=MOODS)
    header = "        " + " ".join(f"{m[:4]:>6}" for m in MOODS)
    print(header)
    for i, row in enumerate(cm):
        print(f"{MOODS[i][:7]:>7} " + " ".join(f"{v:>6}" for v in row))

    return acc, macro_f1


if __name__ == "__main__":
    main()
