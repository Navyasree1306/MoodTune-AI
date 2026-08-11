"""
mlops/api.py
------------
MLOps Week 1-2: Serving layer.

Before: MoodTune AI only ran as a CLI (main.py) - no way for another
service or a deployed frontend to call it over the network.

After: a FastAPI app exposing a /predict endpoint. It loads the model
artifact trained by train.py ONCE at startup (not per-request), then
runs it through the same agent pipeline (decision engine -> matcher ->
explanation -> ethics check) that the CLI uses, so behavior stays
identical - only the entry point changes.

Run with:
    uvicorn mlops.api:app --reload --port 8000
Then:
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"mood_text": "feeling great today", "context": "workout", "top_n": 3}'
"""

import os
import sys
import time
import csv
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from decision_engine import build_target_profile
from music_matcher import recommend
from ethics_checker import diversity_report

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "mood_classifier.joblib")
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs_catalog.csv")
PREDICTION_LOG = os.path.join(os.path.dirname(__file__), "prediction_log.csv")

_model_bundle = None  # loaded once at startup, not per-request


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_bundle
    if not os.path.exists(ARTIFACT_PATH):
        raise RuntimeError(
            f"No trained model found at {ARTIFACT_PATH}. Run `python mlops/train.py` first."
        )
    _model_bundle = joblib.load(ARTIFACT_PATH)
    print(f"[startup] Loaded model artifact from {ARTIFACT_PATH}")
    yield
    _model_bundle = None


app = FastAPI(title="MoodTune AI - Serving API", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    mood_text: str = Field(..., min_length=1, examples=["feeling great today"])
    context: str = Field(default="none", examples=["workout"])
    top_n: int = Field(default=5, ge=1, le=20)
    language: str = Field(default="any")


class PredictResponse(BaseModel):
    mood: str
    confidence: float
    context: str
    recommendations: list
    diversity_flag: str


def _log_prediction(mood_text: str, mood: str, confidence: float):
    """Minimal prediction logging - feeds mlops/monitor.py for drift checks."""
    file_exists = os.path.exists(PREDICTION_LOG)
    with open(PREDICTION_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "mood_text", "predicted_mood", "confidence"])
        writer.writerow([time.time(), mood_text, mood, confidence])


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    vectorizer = _model_bundle["vectorizer"]
    model = _model_bundle["model"]

    X = vectorizer.transform([req.mood_text])
    probs = model.predict_proba(X)[0]
    classes = model.classes_
    prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}
    mood = max(prob_dict, key=prob_dict.get)
    confidence = prob_dict[mood]

    target_profile = build_target_profile(mood, req.context)
    recs, _ = recommend(CATALOG_PATH, target_profile, top_n=req.top_n, diversity=True, language=req.language)
    div_report = diversity_report(recs)

    _log_prediction(req.mood_text, mood, confidence)

    return PredictResponse(
        mood=mood,
        confidence=round(confidence, 3),
        context=req.context,
        recommendations=recs.to_dict("records"),
        diversity_flag=div_report["diversity_flag"],
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model_bundle is not None}
