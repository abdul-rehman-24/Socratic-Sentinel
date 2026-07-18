"""
tests/test_difficulty_adapter.py — Unit tests for difficulty_adapter service.

Tests the escalation threshold, edge traversal, and no-escalation cases.
No OpenAI API calls needed — reads directly from knowledge_graph state.
"""

import pytest
from app.services import knowledge_graph, difficulty_adapter


def setup_function():
    """Reset session graphs before each test."""
    knowledge_graph._session_graphs.clear()
    knowledge_graph._node_histories.clear()


def test_no_escalation_below_threshold():
    """Mastery below 0.60 should NOT trigger escalation."""
    sid = "test-escalate-1"
    g = knowledge_graph._get_or_create(sid)
    g.nodes["backpropagation"]["mastery"] = 0.50  # below 0.60 threshold

    result = difficulty_adapter.get_escalation_target(sid, "backpropagation")
    assert result is None


def test_escalation_above_threshold():
    """Mastery >= 0.60 with an unmastered successor should trigger escalation."""
    sid = "test-escalate-2"
    g = knowledge_graph._get_or_create(sid)
    g.nodes["backpropagation"]["mastery"] = 0.70  # above threshold

    result = difficulty_adapter.get_escalation_target(sid, "backpropagation")
    # backpropagation → gradient_descent (extends) should be returned
    assert result == "gradient_descent"


def test_no_escalation_when_successors_mastered():
    """If all successors are already mastered, no escalation target returned."""
    sid = "test-escalate-3"
    g = knowledge_graph._get_or_create(sid)
    g.nodes["backpropagation"]["mastery"] = 0.80
    # Mark the successor as mastered too
    g.nodes["gradient_descent"]["mastery"] = 1.0
    g.nodes["gradient_descent"]["status"] = "mastered"

    result = difficulty_adapter.get_escalation_target(sid, "backpropagation")
    assert result is None


def test_no_escalation_for_unknown_session():
    """Non-existent session returns None."""
    result = difficulty_adapter.get_escalation_target("nonexistent-session", "backpropagation")
    assert result is None


def test_no_escalation_for_unknown_concept():
    """Non-existent concept in graph returns None."""
    sid = "test-escalate-4"
    knowledge_graph._get_or_create(sid)
    result = difficulty_adapter.get_escalation_target(sid, "made_up_concept")
    assert result is None


def test_escalation_picks_lowest_mastery_successor():
    """When multiple 'extends' successors exist, the lowest mastery one is chosen."""
    sid = "test-escalate-5"
    g = knowledge_graph._get_or_create(sid)
    # gradient_descent → optimizers are the 'extends' chain from backpropagation
    g.nodes["backpropagation"]["mastery"] = 0.75
    # Give gradient_descent some mastery already
    g.nodes["gradient_descent"]["mastery"] = 0.30

    result = difficulty_adapter.get_escalation_target(sid, "backpropagation")
    # gradient_descent is the only 'extends' successor of backpropagation
    assert result == "gradient_descent"


def test_escalation_note_contains_concept_labels():
    """build_escalation_note should mention both concept labels."""
    sid = "test-escalate-6"
    knowledge_graph._get_or_create(sid)
    note = difficulty_adapter.build_escalation_note(sid, "backpropagation", "gradient_descent")
    assert "Backpropagation" in note
    assert "Gradient Descent" in note
    assert "ESCALATION NOTE" in note


def test_threshold_boundary_at_exactly_060():
    """Exactly at 0.60 should trigger escalation (>= threshold)."""
    sid = "test-escalate-7"
    g = knowledge_graph._get_or_create(sid)
    g.nodes["backpropagation"]["mastery"] = 0.60

    result = difficulty_adapter.get_escalation_target(sid, "backpropagation")
    assert result is not None
