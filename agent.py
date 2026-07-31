"""
agent.py
--------
MODULE MAPPING: Exploring Agentic AI

This is what makes MoodTune AI "agentic" rather than a single ML call:
it autonomously runs a multi-step pipeline (perceive -> decide -> act ->
explain -> self-check), makes a decision under uncertainty (low mood
confidence), and logs its own reasoning trace step by step, the way an
agent loop would in a larger LangChain/AutoGen-style system - just
built here from first principles so every step is visible and auditable.
"""

import os
from mood_detector import MoodDetector
from decision_engine import build_target_profile, explain_decision_matrix
from music_matcher import recommend
from ethics_checker import diversity_report, catalog_bias_report, transparency_note
from prompt_engine import generate_explanation

CONFIDENCE_THRESHOLD = 0.35
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "data", "songs_catalog.csv")


class MoodTuneAgent:
    def __init__(self):
        self.detector = MoodDetector()
        self.trace = []

    def _log(self, msg):
        self.trace.append(msg)
        print(f"[Agent] {msg}")

    def run(self, mood_text: str, context: str = "none", top_n: int = 5, use_llm: bool = True,
            language: str = None) -> dict:
        self.trace = []

        # Step 1: Perceive - detect mood from raw text
        self._log("Step 1/5 - Interpreting mood from your input...")
        mood, confidence, all_probs = self.detector.detect(mood_text)

        # Agentic decision point: low confidence -> autonomously blend
        # the top-2 moods instead of committing to a single uncertain label.
        blended = False
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        if confidence < CONFIDENCE_THRESHOLD and len(sorted_probs) > 1:
            second_mood, second_conf = sorted_probs[1]
            self._log(
                f"Confidence low ({confidence:.0%}) - blending top mood '{mood}' "
                f"with second candidate '{second_mood}' ({second_conf:.0%}) instead of guessing."
            )
            mood_for_profile = mood  # primary still drives the profile
            blended = True
        else:
            mood_for_profile = mood
            self._log(f"Detected mood: '{mood}' (confidence {confidence:.0%})")

        # Step 2: Research/decide - build target audio profile
        self._log("Step 2/5 - Building target audio-feature profile (decision matrix)...")
        target_profile = build_target_profile(mood_for_profile, context)
        decision_trace = explain_decision_matrix(mood_for_profile, context, target_profile)

        # Step 3: Act - match against catalog
        lang_txt = language if language and language.lower() != "any" else "any language"
        self._log(f"Step 3/5 - Scoring catalog against target profile (top {top_n}, genre-diverse, language: {lang_txt})...")
        recs, language_relaxed = recommend(CATALOG_PATH, target_profile, top_n=top_n, diversity=True, language=language)
        if language_relaxed:
            self._log(
                f"Not enough '{language}' tracks matched well - autonomously relaxed the "
                f"language filter to avoid returning a thin/empty result."
            )

        # Step 4: Explain - prompt-engineered LLM call or template fallback
        self._log("Step 4/5 - Generating explanation (LLM if available, else template)...")
        songs_list = recs.to_dict("records")
        explanation, method = generate_explanation(mood, confidence, context, songs_list, use_llm=use_llm)
        self._log(f"Explanation generated via: {method}")

        # Step 5: Self-check - ethics/diversity audit before returning results
        self._log("Step 5/5 - Running ethics & diversity self-check on results...")
        div_report = diversity_report(recs)
        import pandas as pd
        full_catalog = pd.read_csv(CATALOG_PATH)
        bias_report = catalog_bias_report(full_catalog)
        if div_report["diversity_flag"] != "OK":
            self._log(f"Self-check flagged: {div_report['diversity_flag']}")
        else:
            self._log("Self-check passed: recommendation set is genre-diverse.")

        return {
            "mood": mood,
            "confidence": confidence,
            "all_mood_probs": all_probs,
            "blended_uncertainty": blended,
            "context": context,
            "language": language if language else "any",
            "language_relaxed": language_relaxed,
            "target_profile": target_profile,
            "decision_trace": decision_trace,
            "recommendations": recs,
            "explanation": explanation,
            "explanation_method": method,
            "diversity_report": div_report,
            "catalog_bias_report": bias_report,
            "transparency_note": transparency_note(),
            "agent_trace": self.trace,
        }
