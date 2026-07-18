"""
services/session_memory.py — Per-session conversation history store.

Stores the last N turns (configurable, default 4) per session_id in-memory.
Same lifecycle and persistence assumptions as knowledge_graph.py:
    - In-memory only, NOT persisted across server restarts (MVP limitation).
    - State resets when the backend process restarts.

Each turn is stored as:
    {
        "role": "user" | "assistant",
        "content": str,
        "intent": str | None,       # classified intent (user turns only)
        "response_type": str | None # response type (assistant turns only)
    }

This module is the single source of truth for conversation history.
Both prompt_chains.py and misconception_engine.py read from it.
"""

from collections import deque
from typing import List, Dict, Deque

# Maximum number of turns (user + assistant pairs) stored per session.
# Each pair = 2 messages. 4 turns = up to 8 messages sent to GPT.
MAX_TURNS = 4

# session_id → deque of message dicts, capped at MAX_TURNS * 2 messages
_histories: Dict[str, Deque[dict]] = {}


def get_history(session_id: str) -> List[dict]:
    """
    Return the conversation history for a session as a list of message dicts.
    Returns an empty list for brand-new sessions.
    """
    return list(_histories.get(session_id, []))


def add_turn(
    session_id: str,
    user_message: str,
    assistant_response: str,
    intent: str = "general",
    response_type: str = "general",
) -> None:
    """
    Append a user+assistant turn pair to the session history.

    Older turns are automatically evicted when MAX_TURNS is exceeded.
    Each call adds exactly 2 messages (user + assistant).
    """
    if session_id not in _histories:
        _histories[session_id] = deque(maxlen=MAX_TURNS * 2)

    hist = _histories[session_id]
    hist.append({"role": "user", "content": user_message, "intent": intent})
    hist.append({
        "role": "assistant",
        "content": assistant_response,
        "response_type": response_type,
    })


def format_history_for_prompt(session_id: str) -> str:
    """
    Format conversation history as a compact string for prompt injection.

    Keeps only the content — strips internal metadata fields (intent, response_type).
    Returns empty string if no history exists (first turn).
    """
    history = get_history(session_id)
    if not history:
        return ""

    lines = []
    for msg in history:
        role_label = "Student" if msg["role"] == "user" else "Tutor"
        # Truncate very long messages to keep token usage bounded
        content = msg["content"]
        if len(content) > 200:
            content = content[:197] + "..."
        lines.append(f"{role_label}: {content}")

    return "\n".join(lines)


def get_openai_history(session_id: str) -> List[dict]:
    """
    Return history formatted as OpenAI message list (role/content dicts).
    Strips internal metadata so it's safe to pass directly to the API.
    """
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in get_history(session_id)
    ]


def clear_session(session_id: str) -> None:
    """Remove all history for a session (e.g. on explicit reset)."""
    _histories.pop(session_id, None)
