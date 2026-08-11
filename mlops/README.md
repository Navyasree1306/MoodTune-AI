# MoodTune AI — MLOps Layer

This extends the original MoodTune AI CLI capstone with the infrastructure
needed to take it from a research prototype to a served, monitored system.

## What changed and why

| Before | After | Why it matters |
|---|---|---|
| Model retrained from scratch on every CLI run | Trained once via `train.py`, persisted as a versioned artifact, logged to MLflow | Reproducibility — you can point to exactly which run/params produced which model |
| Only usable via CLI (`main.py`) | Served over HTTP via FastAPI (`api.py`) | Any frontend, mobile app, or another service can now call it |
| No automated evaluation | Stratified 5-fold cross-validation logged as MLflow metrics on every training run | Confidence in the reported accuracy (~72.6% CV accuracy), not just a train-set number |
| No tests | `pytest` suite covering health check, prediction shape, validation, and edge cases | Catches breakage before it reaches "production" |
| Manual, ad-hoc runs | GitHub Actions CI — trains, tests, and builds the Docker image on every push | Nothing merges without passing this gate |
| No idea if the model degrades over time | `monitor.py` — vocabulary drift + confidence trend report over logged predictions | Production ML fails silently; this catches it early |
| Not containerized | `Dockerfile` — runs identically anywhere | Deployable to Render/Railway/any container host |

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt -r mlops/requirements-mlops.txt

# 2. Train the model (persists artifact + logs to MLflow)
python mlops/train.py

# 3. Inspect the experiment tracking UI
mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db

# 4. Serve it
uvicorn mlops.api:app --reload --port 8000

# 5. Hit it
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"mood_text": "feeling great today", "context": "workout", "top_n": 3}'

# 6. Run tests
python -m pytest mlops/tests/ -v

# 7. Check for drift after some real traffic
python mlops/monitor.py

# 8. Build + run the containerized version
docker build -t moodtune-ai .
docker run -p 8000:8000 moodtune-ai
```

## Design notes

- **Why sqlite for MLflow, not the plain file backend:** MLflow's filesystem-only
  tracking store is now in maintenance mode / deprecated upstream — sqlite is
  the current recommended lightweight local backend.
- **Why the model trains at Docker build time rather than loading a pre-committed
  artifact:** at this project's scale, baking training into the image keeps the
  Dockerfile self-contained and reproducible from source. At larger scale, the
  next step would be pulling a specific versioned artifact from an MLflow Model
  Registry or S3 bucket instead.
- **Why a hand-rolled drift check instead of Evidently AI:** matches this
  project's existing philosophy (see `ethics_checker.py`) of keeping checks
  simple and fully auditable rather than a black-box dependency. Evidently
  would be the natural next step at larger scale.
