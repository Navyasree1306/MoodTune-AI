"""
report_generator.py
--------------------
MODULE MAPPING: AI and Real-World Deployment

Real deployed AI tools produce artifacts users can keep, share, or audit -
not just console output. This writes a timestamped Markdown report per
session to reports/, documenting every decision the agent made (useful
for a capstone demo/viva: you can literally hand in a session report).
"""

import os
from datetime import datetime

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


def write_report(mood_text: str, result: dict) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORTS_DIR, f"session_{ts}.md")

    recs = result["recommendations"]
    rec_lines = "\n".join(
        f"| {r.track_name} | {r.artist} | {r.genre} | {r.language} | {r.match_score:.3f} |"
        for r in recs.itertuples()
    )

    content = f"""# MoodTune AI - Session Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Input
- Raw mood text: "{mood_text}"
- Listening context: {result['context']}
- Language preference: {result['language']}{' (relaxed - not enough matches in that language, showing all languages)' if result['language_relaxed'] else ''}

## 2. Mood Detection (Foundations of AI)
- Detected mood: **{result['mood']}** (confidence: {result['confidence']:.0%})
- Full probability distribution: {result['all_mood_probs']}
- Low-confidence blending triggered: {result['blended_uncertainty']}

## 3. Decision Engine (AI for Research & Smart Decision Making)
```
{result['decision_trace']}
```

## 4. Recommendations (AI Problem Solving & Innovation)
| Track | Artist | Genre | Language | Match Score (lower=closer) |
|---|---|---|---|---|
{rec_lines}

## 5. Explanation (Applied AI & Prompts)
Generated via: **{result['explanation_method']}**

> {result['explanation']}

## 6. Ethics & Bias Self-Check (Ethics & Emerging Tech)
- Recommendation genre distribution: {result['diversity_report']['genre_distribution']}
- Max single-genre share: {result['diversity_report']['max_single_genre_share']}
- Diversity flag: {result['diversity_report']['diversity_flag']}
- Underlying catalog balance check: {result['catalog_bias_report']}

{result['transparency_note']}

## 7. Agent Reasoning Trace (Exploring Agentic AI)
{chr(10).join(f"- {step}" for step in result['agent_trace'])}

## 8. Deployment Notes (AI and Real-World Deployment)
This session ran as a local CLI tool. In production this would be
packaged as a pip-installable CLI or wrapped in a lightweight API
(FastAPI) with the local catalog replaced by a licensed music catalog
or a user's own streaming-service library export. See README.md.
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
