"""
prompt_engine.py
------------------
MODULE MAPPING: Applied AI & Prompts

Demonstrates deliberate prompt engineering: role definition, context,
explicit constraints, and output-format instructions - rather than a
bare question. If an ANTHROPIC_API_KEY environment variable is set,
this calls the real Claude API (agentic reasoning step). Otherwise it
falls back to a template-based generator so the tool NEVER breaks in a
demo/offline setting.
"""

import os


def build_prompt(mood: str, confidence: float, context: str, songs: list) -> str:
    """Structured prompt: role, context, constraints, output format."""
    song_lines = "\n".join(
        f"- \"{s['track_name']}\" by {s['artist']} (genre: {s['genre']})" for s in songs
    )
    prompt = f"""You are a warm, concise music-mood assistant inside a capstone demo app.

CONTEXT:
- Detected user mood: {mood} (model confidence: {confidence:.0%})
- Listening context: {context}
- Selected tracks (already chosen by a separate matching algorithm):
{song_lines}

TASK:
Write a short (3-4 sentence) explanation of why this set of tracks fits
the detected mood and context. Be specific about the *feel* (tempo/energy/
mood), not just the genre names.

CONSTRAINTS:
- Do not invent facts about real artists or claim these are real songs by real people.
- Do not use more than 80 words.
- Do not use bullet points - write flowing prose.
- If confidence is below 50%, gently note that the mood read is uncertain.

OUTPUT FORMAT:
Plain text, no headers, no markdown formatting.
"""
    return prompt


def _template_fallback(mood: str, confidence: float, context: str, songs: list) -> str:
    """Used when no LLM API key is configured - keeps the tool fully functional offline."""
    genres = sorted({s["genre"] for s in songs})
    uncertainty = ""
    if confidence < 0.5:
        uncertainty = " Note: the mood read on your input was a bit uncertain, so treat this as a best guess."
    context_txt = f" for your {context} session" if context != "none" else ""
    return (
        f"Based on a '{mood}' mood read{context_txt}, this set leans on {', '.join(genres)} "
        f"tracks chosen to match the energy and tempo typical of that mood. "
        f"The matching algorithm balanced valence, energy, and tempo against your "
        f"target profile rather than just picking songs from one genre.{uncertainty}"
    )


def generate_explanation(mood: str, confidence: float, context: str, songs: list, use_llm: bool = True) -> tuple:
    """
    Returns (explanation_text, method_used) where method_used is
    'llm' or 'template' - logged for transparency in the report.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if use_llm and api_key:
        try:
            import anthropic  # optional dependency, only needed for this path
            client = anthropic.Anthropic(api_key=api_key)
            prompt = build_prompt(mood, confidence, context, songs)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            ).strip()
            if text:
                return text, "llm"
        except Exception as e:
            # Fall through to template - the tool should never crash because
            # of an optional, non-critical LLM call.
            print(f"[Agent] LLM call failed ({e}); falling back to template explanation.")

    return _template_fallback(mood, confidence, context, songs), "template"
