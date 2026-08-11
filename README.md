# MoodTune AI — Agentic Mood-Based Music Recommender
> **Production deployment layer available** — see [`mlops/README.md`](mlops/README.md) for MLflow experiment tracking, FastAPI serving, Docker, CI/CD, and drift monitoring built on top of this project.

A CLI tool that detects your mood from free text, autonomously decides what
kind of music fits it (and your context), matches it against a local song
catalog, explains its reasoning, and runs an ethics/bias self-check —
producing a downloadable session report for every run.

---

## Why this project, and why no live Spotify API?

Spotify deprecated its `/recommendations` and `/audio-features` endpoints
for all new developer apps in November 2024, and tightened the Web API
further in February 2026 (reduced search pagination, removed several
metadata fields). A capstone built directly on those endpoints would break
before it could even be demoed. MoodTune AI instead uses a **local demo
catalog** (`data/songs_catalog.csv`) with audio-feature-style attributes,
so the tool is fully functional offline and isn't at the mercy of a
third-party API's roadmap. Section "Real-World Deployment" below explains
how you'd swap this for a real catalog in production.

---


## Setup

```bash
cd moodtune_ai
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 data/generate_catalog.py   # generates data/songs_catalog.csv (already included, re-run if you want a fresh shuffle)
```

## Usage

**Interactive mode:**
```bash
python3 main.py
```

**Direct mode:**
```bash
python3 main.py --mood "I need to hit the gym and pump myself up" --context workout --topn 5
```

**Available contexts:** `none`, `workout`, `study`, `sleep`, `commute`, `party`

**Available languages:** `any`, `English`, `Hindi`, `Telugu`, `Tamil`, `Punjabi`, `Korean`, `Spanish`

```bash
python3 main.py --mood "I need to hit the gym" --context workout --language Telugu
```

If a language preference doesn't have enough well-matching tracks, the
agent autonomously relaxes the filter rather than returning a thin or
empty result — and tells you it did so, both on screen and in the saved
report (see `agent.py` Step 3).

**Force offline template explanation (skip any LLM call):**
```bash
python3 main.py --mood "feeling nostalgic today" --no-llm
```

**Enable real LLM-generated explanations** (optional — the tool works
perfectly without this):
```bash
export ANTHROPIC_API_KEY="your-key-here"
python3 main.py --mood "feeling nostalgic today"
```

Every run writes a full Markdown report to `reports/session_<timestamp>.md`
— useful to attach directly to your capstone submission or show in a viva.

---

## Architecture

```
User text ──▶ mood_detector.py (ML classifier) ──▶ mood + confidence
                                                        │
                                                        ▼
                                          decision_engine.py (decision matrix)
                                                        │
                                                        ▼
                                          music_matcher.py (weighted scoring
                                          against data/songs_catalog.csv)
                                                        │
                                                        ▼
                                      prompt_engine.py (LLM or template explanation)
                                                        │
                                                        ▼
                                    ethics_checker.py (diversity + transparency)
                                                        │
                                                        ▼
                                  report_generator.py ──▶ reports/session_*.md

agent.py orchestrates all of the above as one autonomous pipeline (see
the "[Agent] Step X/5" trace printed during every run).
```

---

## Real-World Deployment (production migration path)

This CLI is deliberately structured so each module is a swappable piece:

1. **Catalog** — replace `data/songs_catalog.csv` with a licensed music
   catalog, or a user's exported streaming-service library (e.g. Spotify's
   "Download your data" export, which still includes track/artist metadata
   even though the live audio-features API is gone).
2. **Interface** — wrap `agent.py` in a FastAPI app (`POST /recommend`) to
   turn this into a backend service; the CLI logic in `main.py` maps almost
   1:1 onto a REST handler.
3. **Mood detection** — swap `mood_detector.py`'s Logistic Regression for a
   fine-tuned transformer (e.g. DistilBERT) once you have more labeled data
   than the ~64 bundled examples.
4. **Packaging** — add a `setup.py`/`pyproject.toml` to make this
   `pip install`-able as a real CLI command (`moodtune --mood "..."`).

---

## Evaluation

The mood classifier was evaluated with 5-fold stratified cross-validation
over the bundled 84-example training set spanning **10 mood classes**
(happy, sad, energetic, calm, angry, romantic, focused, nostalgic, upbeat,
delulu). A single train/test split would be too noisy to trust at this
size, so cross-validation is used instead. Run it yourself:
`python3 evaluate_classifier.py`

| Metric | Score |
|---|---|
| Overall accuracy | **72.6%** |
| Macro-averaged F1 | **0.727** |

**Reading the result:** `focused`, `nostalgic`, and `delulu` classify very
reliably (F1 ≥ 0.90). `sad` and `romantic` are the weakest spots — both
share overlapping low-energy, emotionally-loaded vocabulary ("feeling,"
"missing," "heavy"), so the classifier sometimes leans toward `romantic`
on ambiguous input. This is a genuine, explainable limitation of a small
bag-of-words dataset, not a bug — and it's exactly why the tool always
shows its confidence score rather than presenting a guess as certain.

**How to improve it:** more labeled examples for `sad`/`romantic`
specifically, or moving to sentence embeddings instead of TF-IDF so
semantically similar but lexically different phrases separate better.

---



- Mood classifier is trained on a small (~64-example) bundled dataset —
  accuracy on ambiguous or mixed-emotion text will be limited. The
  confidence score is always shown so this is visible, not hidden.
- The song catalog is synthetic/demo data, not a real licensed catalog.
- LLM explanations (when enabled) cost API credits and require your own key.

---

