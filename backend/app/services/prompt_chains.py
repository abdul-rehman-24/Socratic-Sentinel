"""
services/prompt_chains.py — Prompt assembly and GPT call orchestration.

All GPT calls for generative responses go through this module.
Template files in /prompt_templates/ are loaded once at module import.
No prompt text is hardcoded in routers or other services.

Day 3 additions:
    - generate_socratic_question now accepts conversation_history and
      escalation_note, passing history as prior OpenAI messages so
      follow-up questions build on the ongoing dialogue.
    - diagnose_misconception similarly receives history for context.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from openai import AsyncOpenAI

from app.config import get_settings
from app.services.retrieval import RetrievalResult, format_chunks_for_prompt

logger = logging.getLogger(__name__)
settings = get_settings()
_client = AsyncOpenAI(api_key=settings.openai_api_key)

# ─── Load templates at import time ───────────────────────────────────────────
_TEMPLATE_DIR = Path(__file__).parent.parent.parent / "prompt_templates"


def _load(filename: str) -> str:
    path = _TEMPLATE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


_SOCRATIC_TMPL = _load("socratic_question.txt")
_MISCONCEPTION_TMPL = _load("misconception_diagnosis.txt")
_OUT_OF_SCOPE_TMPL = _load("out_of_scope.txt")


# ─── Socratic question generation ────────────────────────────────────────────

async def generate_socratic_question(
    message: str,
    retrieved_chunks: List[RetrievalResult],
    mastery_summary: str = "No prior interaction in this session.",
    conversation_history: Optional[List[dict]] = None,
    escalation_note: str = "",
) -> str:
    """
    Generate a Socratic guiding question using curriculum context + conversation history.

    Args:
        message:              The user's current question.
        retrieved_chunks:     Relevant curriculum context from ChromaDB.
        mastery_summary:      Text description of student's mastery state.
        conversation_history: Prior turns as OpenAI message list [{role, content}].
                              Inserted between system and current user message so
                              GPT has full dialogue context. Max 4 turns (8 messages).
        escalation_note:      Optional difficulty escalation instruction from
                              difficulty_adapter.py. Appended to system prompt.

    Returns:
        The Socratic response string (never a direct answer).
    """
    context = format_chunks_for_prompt(retrieved_chunks)
    
    # Format conversation history for the prompt
    from app.services import session_memory
    # Create a formatted history string from the conversation history messages
    history_str = ""
    if conversation_history:
        history_lines = []
        for msg in conversation_history:
            role_label = "Student" if msg["role"] == "user" else "Tutor"
            history_lines.append(f"{role_label}: {msg['content']}")
        history_str = "\n".join(history_lines)
    
    if not history_str:
        history_str = "(No prior dialogue yet — first turn of the session.)"
    
    system_content = (
        _SOCRATIC_TMPL
        .replace("{context}", context)
        .replace("{mastery_summary}", mastery_summary)
        .replace("{conversation_history}", history_str)
    )
    if escalation_note:
        system_content += escalation_note

    # Build message list: system → [history turns] → current user message
    messages = [{"role": "system", "content": system_content}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": message})

    response = await _client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.7,
        max_tokens=350,
    )
    return (response.choices[0].message.content or "").strip()


# ─── Misconception diagnosis ──────────────────────────────────────────────────

async def diagnose_misconception(
    wrong_answer: str,
    retrieved_chunks: List[RetrievalResult],
    concept: str = "deep learning",
    conversation_history: Optional[List[dict]] = None,
) -> dict:
    """
    Diagnose the specific misconception in a wrong answer, with conversation context.

    Args:
        wrong_answer:         The user's incorrect statement.
        retrieved_chunks:     Curriculum context for the concept being discussed.
        concept:              Name of the concept being discussed.
        conversation_history: Prior turns for dialogue context.

    Returns:
        Dict with keys: misconception_label, misconception_description, guided_response.
    """
    context = format_chunks_for_prompt(retrieved_chunks)
    
    # Format conversation history for the prompt
    history_str = ""
    if conversation_history:
        history_lines = []
        # Include last 2 turns max (4 messages) for misconception — keeps token budget tight
        recent_history = conversation_history[-4:]
        for msg in recent_history:
            role_label = "Student" if msg["role"] == "user" else "Tutor"
            history_lines.append(f"{role_label}: {msg['content']}")
        history_str = "\n".join(history_lines)
    
    if not history_str:
        history_str = "(No prior dialogue context.)"
    
    system_content = (
        _MISCONCEPTION_TMPL
        .replace("{context}", context)
        .replace("{concept}", concept)
        .replace("{wrong_answer}", wrong_answer)
        .replace("{conversation_history}", history_str)
    )

    messages = [{"role": "system", "content": system_content}]
    if conversation_history:
        # Include last 2 turns max (4 messages) for misconception — keeps token budget tight
        messages.extend(conversation_history[-4:])
    messages.append({"role": "user", "content": wrong_answer})

    response = await _client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.3,
        max_tokens=400,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Misconception response was not valid JSON: %s", raw)
        return {
            "misconception_label": "UNKNOWN",
            "misconception_description": "Could not parse misconception.",
            "guided_response": "Interesting — can you walk me through your reasoning step by step?",
        }


# ─── Out-of-scope response ────────────────────────────────────────────────────

async def generate_out_of_scope_response(message: str) -> str:
    """
    Generate a warm, honest out-of-scope reply without attempting to answer.
    Anti-hallucination gate — no history needed since we're refusing to engage.
    """
    system_prompt = _OUT_OF_SCOPE_TMPL.replace("{message}", message)

    response = await _client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        temperature=0.4,
        max_tokens=180,
    )
    return (response.choices[0].message.content or "").strip()
