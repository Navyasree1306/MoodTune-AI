"""
mlops/train.py
---------------
MLOps Week 1: Model training with MLflow experiment tracking + persistence.

Before: mood_detector.py trained a fresh TF-IDF + LogisticRegression model
in memory on every single run — fine for a CLI demo, not viable for a
served API (retraining per request is wasteful and non-reproducible).

After: this script trains the same model ONCE, logs the run (params,
metrics, the training data version) to MLflow, evaluates it properly with
stratified k-fold cross-validation, and saves the fitted vectorizer +
classifier as a versioned artifact under mlops/artifacts/. The serving
layer (api.py) loads this artifact instead of retraining.

Run with:
    python mlops/train.py
Then inspect results with:
    mlflow ui --backend-store-uri mlops/mlruns
"""

import os
import sys
import joblib
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mood_detector import TRAINING_DATA, MOODS  # reuse the existing labeled data

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MLRUNS_DIR = os.path.join(os.path.dirname(__file__), "mlruns")
# MLflow's plain-filesystem backend is now in maintenance mode (deprecated) -
# sqlite is the current recommended lightweight local backend.
MLFLOW_DB = os.path.join(os.path.dirname(__file__), "mlflow.db")

# Hyperparameters — logged explicitly so every run is reproducible and comparable
PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 1,
    "stop_words": "english",
    "C": 8.0,
    "max_iter": 2000,
    "cv_folds": 5,
}


def train_and_log():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("moodtune-mood-classifier")

    texts = [t for t, _ in TRAINING_DATA]
    labels = [m for _, m in TRAINING_DATA]

    with mlflow.start_run(run_name="tfidf_logreg_v1") as run:
        # --- Log parameters ---
        mlflow.log_params(PARAMS)
        mlflow.log_param("n_training_examples", len(texts))
        mlflow.log_param("n_classes", len(MOODS))

        # --- Build pipeline pieces ---
        vectorizer = TfidfVectorizer(
            ngram_range=PARAMS["ngram_range"],
            min_df=PARAMS["min_df"],
            stop_words=PARAMS["stop_words"],
        )
        X = vectorizer.fit_transform(texts)
        model = LogisticRegression(max_iter=PARAMS["max_iter"], C=PARAMS["C"])

        # --- Proper evaluation: stratified 5-fold CV, not just train-set accuracy ---
        skf = StratifiedKFold(n_splits=PARAMS["cv_folds"], shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, labels, cv=skf, scoring="accuracy")

        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())
        for i, score in enumerate(cv_scores):
            mlflow.log_metric(f"cv_fold_{i}_accuracy", score)

        # --- Fit final model on all data for deployment ---
        model.fit(X, labels)

        # --- Persist as a single artifact (vectorizer + model bundled together) ---
        bundle = {"vectorizer": vectorizer, "model": model, "moods": MOODS}
        artifact_path = os.path.join(ARTIFACT_DIR, "mood_classifier.joblib")
        joblib.dump(bundle, artifact_path)
        mlflow.log_artifact(artifact_path)

        print(f"CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        print(f"Model artifact saved to: {artifact_path}")
        print(f"MLflow run ID: {run.info.run_id}")

        return artifact_path, cv_scores.mean()


if __name__ == "__main__":
    train_and_log()
