"""
tests/test_misconception_engine.py — Unit tests for misconception_engine service.

Mocks prompt_chains.diagnose_misconception to test taxonomy normalization
and MisconceptionResult construction in isolation.
"""

import pytest
from unittest.mock import AsyncMock, patch
from app.services.misconception_engine import diagnose, KNOWN_MISCONCEPTIONS
from app.services.retrieval import RetrievalResult


def _chunks():
    return [RetrievalResult(
        chunk_id="backpropagation#chain-rule",
        source_file="backpropagation.md",
        content="...",
        similarity_score=0.85,
        topic="backpropagation",
    )]


@pytest.mark.asyncio
async def test_known_label_preserved():
    """Known taxonomy labels must be preserved as-is."""
    with patch("app.services.misconception_engine.prompt_chains.diagnose_misconception",
               new=AsyncMock(return_value={
                   "misconception_label": "GRADIENT_DIRECTION_CONFUSION",
                   "misconception_description": "Student thinks gradients point toward minimum.",
                   "guided_response": "Which direction does gradient descent actually move?",
               })):
        result = await diagnose("The gradient points toward the minimum", _chunks())

    assert result.misconception_label == "GRADIENT_DIRECTION_CONFUSION"
    assert result.misconception_label in KNOWN_MISCONCEPTIONS
    assert "minimum" in result.misconception_description.lower()
    assert len(result.guided_response) > 10


@pytest.mark.asyncio
async def test_unknown_label_normalized_to_other():
    """Labels not in taxonomy must be normalized to 'OTHER'."""
    with patch("app.services.misconception_engine.prompt_chains.diagnose_misconception",
               new=AsyncMock(return_value={
                   "misconception_label": "MADE_UP_LABEL",
                   "misconception_description": "Some description.",
                   "guided_response": "A guiding question.",
               })):
        result = await diagnose("some wrong statement", _chunks())

    assert result.misconception_label == "OTHER"


@pytest.mark.asyncio
async def test_missing_fields_use_defaults():
    """Partial GPT response — missing fields must use sensible defaults."""
    with patch("app.services.misconception_engine.prompt_chains.diagnose_misconception",
               new=AsyncMock(return_value={})):
        result = await diagnose("some wrong statement", _chunks())

    assert result.misconception_label == "OTHER"
    assert len(result.misconception_description) > 0
    assert len(result.guided_response) > 0


@pytest.mark.asyncio
async def test_all_taxonomy_labels_valid():
    """Sanity check: all labels in taxonomy should be uppercase strings."""
    for label in KNOWN_MISCONCEPTIONS:
        assert label == label.upper()
        assert len(label) > 0
