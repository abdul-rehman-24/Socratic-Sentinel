"""
tests/test_confidence_scorer.py — Unit tests for confidence_scorer service.

Tests the threshold logic for green/yellow/red classification.
No OpenAI API calls needed.
"""

import pytest
from app.services.retrieval import RetrievalResult
from app.services.confidence_scorer import score
from app.models.chat import ConfidenceLevel


def _chunk(similarity: float, chunk_id: str = "backpropagation#chain-rule") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        source_file="backpropagation.md",
        content="Test content",
        similarity_score=similarity,
        topic="backpropagation",
    )


@pytest.mark.asyncio
async def test_green_confidence():
    """Similarity ≥ 0.75 → GREEN."""
    result = await score([_chunk(0.85), _chunk(0.78)])
    assert result.level == ConfidenceLevel.GREEN
    assert result.score == 0.85
    assert "backpropagation#chain-rule" in result.grounded_in


@pytest.mark.asyncio
async def test_yellow_confidence():
    """Similarity 0.45–0.74 → YELLOW."""
    result = await score([_chunk(0.60)])
    assert result.level == ConfidenceLevel.YELLOW
    assert result.score == 0.60


@pytest.mark.asyncio
async def test_red_confidence():
    """Similarity < 0.45 → RED."""
    result = await score([_chunk(0.30)])
    assert result.level == ConfidenceLevel.RED
    assert result.score == 0.30
    # Score below YELLOW threshold should not be in grounded_in
    assert result.grounded_in == []


@pytest.mark.asyncio
async def test_empty_chunks_returns_red():
    """No chunks retrieved → RED with score 0.0."""
    result = await score([])
    assert result.level == ConfidenceLevel.RED
    assert result.score == 0.0
    assert result.grounded_in == []


@pytest.mark.asyncio
async def test_green_boundary():
    """Exactly at 0.75 threshold → GREEN."""
    result = await score([_chunk(0.75)])
    assert result.level == ConfidenceLevel.GREEN


@pytest.mark.asyncio
async def test_yellow_boundary():
    """Exactly at 0.45 threshold → YELLOW."""
    result = await score([_chunk(0.45)])
    assert result.level == ConfidenceLevel.YELLOW


@pytest.mark.asyncio
async def test_best_score_used():
    """Multiple chunks — highest similarity used for level determination."""
    result = await score([_chunk(0.40), _chunk(0.80, "backpropagation#forward-pass")])
    assert result.level == ConfidenceLevel.GREEN
    assert result.score == 0.80
