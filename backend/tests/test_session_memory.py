"""
tests/test_session_memory.py — Unit tests for session_memory service.

Tests history storage, eviction, formatting, and openai message format.
No OpenAI API calls needed.
"""

import pytest
from app.services import session_memory


def setup_function():
    """Clear all histories before each test to ensure isolation."""
    session_memory._histories.clear()


def test_empty_session_returns_empty():
    assert session_memory.get_history("new-session") == []
    assert session_memory.format_history_for_prompt("new-session") == ""
    assert session_memory.get_openai_history("new-session") == []


def test_add_single_turn():
    session_memory.add_turn("s1", "What is backprop?", "Can you think about...", "concept_question", "socratic_question")
    hist = session_memory.get_history("s1")
    assert len(hist) == 2  # 1 user + 1 assistant
    assert hist[0]["role"] == "user"
    assert hist[0]["content"] == "What is backprop?"
    assert hist[0]["intent"] == "concept_question"
    assert hist[1]["role"] == "assistant"
    assert hist[1]["response_type"] == "socratic_question"


def test_add_multiple_turns_within_limit():
    for i in range(4):
        session_memory.add_turn("s2", f"Q{i}", f"A{i}", "concept_question", "socratic_question")
    hist = session_memory.get_history("s2")
    # 4 pairs = 8 messages, all within MAX_TURNS * 2 = 8
    assert len(hist) == 8


def test_history_evicts_old_turns():
    """Adding 5 pairs when MAX_TURNS=4 should evict the oldest pair."""
    for i in range(5):  # 5 pairs = 10 messages > 8 max
        session_memory.add_turn("s3", f"Q{i}", f"A{i}")
    hist = session_memory.get_history("s3")
    # Should keep last 4 pairs = 8 messages
    assert len(hist) == 8
    # Oldest messages (Q0, A0) should be gone
    assert all("Q0" not in m["content"] for m in hist)
    assert any("Q4" in m["content"] for m in hist)


def test_format_history_for_prompt():
    session_memory.add_turn("s4", "What is backprop?", "Think about what the forward pass needs...", "concept_question", "socratic_question")
    session_memory.add_turn("s4", "It computes activations", "Exactly — and what does it need to save?", "general", "socratic_question")
    formatted = session_memory.format_history_for_prompt("s4")
    assert "Student: What is backprop?" in formatted
    assert "Tutor: Think about" in formatted
    assert "Student: It computes activations" in formatted


def test_format_truncates_long_messages():
    long_msg = "x" * 300
    session_memory.add_turn("s5", long_msg, "short response")
    formatted = session_memory.format_history_for_prompt("s5")
    # Should be truncated to 200 chars + "..."
    assert "..." in formatted
    assert len([line for line in formatted.split("\n") if "Student:" in line][0]) < 220


def test_openai_history_strips_metadata():
    """get_openai_history should return only role+content — no intent or response_type."""
    session_memory.add_turn("s6", "Q1", "A1", "concept_question", "socratic_question")
    oai = session_memory.get_openai_history("s6")
    for msg in oai:
        assert set(msg.keys()) == {"role", "content"}
        assert "intent" not in msg
        assert "response_type" not in msg


def test_clear_session():
    session_memory.add_turn("s7", "Q", "A")
    session_memory.clear_session("s7")
    assert session_memory.get_history("s7") == []
