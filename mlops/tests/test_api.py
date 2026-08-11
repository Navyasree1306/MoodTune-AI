"""
mlops/tests/test_api.py
------------------------
Basic test coverage for the serving API. This is what GitHub Actions
runs automatically on every push (see .github/workflows/ci.yml) -
if these fail, the pipeline fails and you know before it reaches
production, not after.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from fastapi.testclient import TestClient
from mlops.api import app


@pytest.fixture
def client():
    # Using TestClient as a context manager triggers the lifespan
    # startup/shutdown events, so the model actually loads before tests run.
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_valid_mood(client):
    response = client.post("/predict", json={
        "mood_text": "I feel amazing today everything is going great",
        "context": "none",
        "top_n": 3,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["mood"] in [
        "happy", "sad", "energetic", "calm", "angry",
        "romantic", "focused", "nostalgic", "upbeat", "delulu",
    ]
    assert 0 <= data["confidence"] <= 1
    assert len(data["recommendations"]) <= 3


def test_predict_rejects_empty_mood_text(client):
    response = client.post("/predict", json={"mood_text": "", "top_n": 3})
    assert response.status_code == 422  # pydantic validation error


def test_predict_respects_top_n(client):
    response = client.post("/predict", json={
        "mood_text": "need to hit the gym and pump myself up",
        "context": "workout",
        "top_n": 2,
    })
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) <= 2
