"""
main.py
-------
MoodTune AI - Agentic Mood-Based Music Recommender
Capstone project for Infosys Springboard "AI EMPOW(H)ER"

CLI entry point. Run with:
    python main.py                          # interactive mode
    python main.py --mood "text" --context workout --topn 5
    python main.py --mood "text" --no-llm    # force template explanations
"""

import argparse
import sys
from agent import MoodTuneAgent
from report_generator import write_report

VALID_CONTEXTS = ["none", "workout", "study", "sleep", "commute", "party","happy","delulu","up-beat"]


def print_results(mood_text: str, result: dict):
    print("\n" + "=" * 60)
    print("  MOODTUNE AI - RECOMMENDATION RESULTS")
    print("=" * 60)
    print(f"Input: \"{mood_text}\"")
    print(f"Detected mood: {result['mood']}  (confidence: {result['confidence']:.0%})")
    print(f"Context: {result['context']}")
    print("\nTop recommendations:")
    recs = result["recommendations"]
    for i, r in enumerate(recs.itertuples(), 1):
        print(f"  {i}. \"{r.track_name}\" - {r.artist}  [{r.genre}]  (score {r.match_score:.3f})")

    print(f"\nWhy these tracks ({result['explanation_method']}):")
    print(f"  {result['explanation']}")

    print(f"\nEthics/diversity check: {result['diversity_report']['diversity_flag']}")
    print("=" * 60)


def run_session(mood_text: str, context: str, top_n: int, use_llm: bool):
    agent = MoodTuneAgent()
    print("\n[Agent] Starting session...\n")
    result = agent.run(mood_text, context=context, top_n=top_n, use_llm=use_llm)
    print_results(mood_text, result)
    report_path = write_report(mood_text, result)
    print(f"\nFull session report saved to: {report_path}\n")


def interactive_mode():
    print("=" * 60)
    print("  MOODTUNE AI - Agentic Mood-Based Music Recommender")
    print("  (type 'quit' to exit)")
    print("=" * 60)
    while True:
        mood_text = input("\nHow are you feeling? (describe in your own words): ").strip()
        if mood_text.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not mood_text:
            continue
        print(f"Available contexts: {', '.join(VALID_CONTEXTS)}")
        context = input("Context (press enter for 'none'): ").strip().lower() or "none"
        if context not in VALID_CONTEXTS:
            print(f"Unrecognized context '{context}', using 'none'.")
            context = "none"
        run_session(mood_text, context, top_n=5, use_llm=True)


def main():
    parser = argparse.ArgumentParser(description="MoodTune AI - Agentic Mood-Based Music Recommender")
    parser.add_argument("--mood", type=str, help="Describe your mood in free text")
    parser.add_argument("--context", type=str, default="none", choices=VALID_CONTEXTS,
                         help="Listening context (workout/study/sleep/commute/party/none)")
    parser.add_argument("--topn", type=int, default=5, help="Number of tracks to recommend")
    parser.add_argument("--no-llm", action="store_true", help="Force template explanation (skip LLM API call)")
    args = parser.parse_args()

    if args.mood:
        run_session(args.mood, args.context, args.topn, use_llm=not args.no_llm)
    else:
        interactive_mode()


if __name__ == "__main__":
    sys.exit(main())
