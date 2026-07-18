"""
services/intent_classifier.py — Server-side intent classification.

Classifies every incoming user message into one of four intents using a
lightweight GPT call. The client's message_type field is NEVER used for routing.

Intents:
    concept_question  — User is asking about a DL concept ("What is backprop?")
    wrong_answer      — User submitted an incorrect or partially-correct answer
    out_of_scope      — Query is outside the DL curriculum
    general           — Greeting, follow-up, or ambiguous input
"""

import json
from openai import AsyncOpenAI
from app.config import get_settings

settings = get_settings()
_client = AsyncOpenAI(api_key=settings.openai_api_key)

# ─── Intent prompt (self-contained, no external template file needed) ─────────
# Kept inline here deliberately: it's a single-turn classification call with
# a tight JSON contract — not a generative prompt that needs tuning.

_SYSTEM_PROMPT = """You are an intent classifier for a Deep Learning tutor chatbot.
Classify the user message into EXACTLY ONE of these intents:

- concept_question : User is asking what something is, how it works, or why something happens. Examples: "What is backpropagation?", "How does gradient descent work?", "Why do we use ReLU?"
- wrong_answer     : User is stating something incorrect or is attempting an answer that contains a factual error. Examples: "The gradient always points toward the minimum", "Backprop updates weights during the forward pass"
- out_of_scope     : Query is clearly outside deep learning fundamentals (e.g., asking about transformers, RL, NLP, unrelated topics, or things not in the curriculum)
- general          : Greetings, thank-yous, follow-up clarifications, or genuinely ambiguous messages

Respond with ONLY valid JSON. No explanation, no markdown, no extra text.
Format: {"intent": "<one of the four labels above>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}"""


async def classify(message: str) -> tuple[str, float]:
    """
    Classify a user message into an intent label.

    Args:
        message: The raw user message string.

    Returns:
        Tuple of (intent_label, confidence_score).
        intent_label is one of: concept_question | wrong_answer | out_of_scope | general
    """
    try:
        response = await _client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            temperature=0.0,        # deterministic classification
            max_tokens=120,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)

        intent = parsed.get("intent", "general")
        confidence = float(parsed.get("confidence", 0.7))

        # Validate intent is one of the allowed labels
        valid = {"concept_question", "wrong_answer", "out_of_scope", "general"}
        if intent not in valid:
            intent = "general"

        return intent, confidence

    except Exception as exc:
        # Fail open: treat as general so the conversation doesn't break
        import logging
        logging.getLogger(__name__).warning("Intent classification failed: %s", exc)
        return "general", 0.5
