"""
tests/test_chat.py — Tests for POST /api/chat endpoint.

Uses pytest-asyncio + httpx AsyncClient with ASGI transport.
OpenAI calls are mocked via monkeypatching so tests run offline.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport
from app.main import app


# ─── Shared mock fixtures ─────────────────────────────────────────────────────

def _make_retrieval_result(similarity: float = 0.85):
    from app.services.retrieval import RetrievalResult
    return [RetrievalResult(
        chunk_id="backpropagation#chain-rule",
        source_file="backpropagation.md",
        content="The chain rule states that dz/dx = (dz/dy)·(dy/dx)...",
        similarity_score=similarity,
        topic="backpropagation",
    )]


# ─── Response shape tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_response_shape():
    """POST /api/chat must return all required fields in the stable contract."""
    with (
        patch("app.services.intent_classifier.classify", new=AsyncMock(return_value=("concept_question", 0.9))),
        patch("app.services.retrieval.retrieve", new=AsyncMock(return_value=_make_retrieval_result())),
        patch("app.services.prompt_chains.generate_socratic_question",
              new=AsyncMock(return_value="What do you already know about the chain rule?")),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/chat", json={
                "session_id": "test-001",
                "message": "What is backpropagation?",
            })

    assert response.status_code == 200
    data = response.json()

    # Top-level required fields
    assert data["session_id"] == "test-001"
    assert isinstance(data["response_text"], str) and len(data["response_text"]) > 0
    assert data["response_type"] in ["socratic_question", "misconception_diagnosis", "out_of_scope", "general"]

    # Confidence object
    conf = data["confidence"]
    assert conf["level"] in ["green", "yellow", "red"]
    assert 0.0 <= conf["score"] <= 1.0
    assert isinstance(conf["grounded_in"], list)
    assert isinstance(conf["explanation"], str)

    # Knowledge graph delta
    delta = data["knowledge_graph_delta"]
    assert isinstance(delta["updated_nodes"], list)
    assert isinstance(delta["updated_edges"], list)

    # Curriculum refs
    assert isinstance(data["curriculum_refs"], list)


@pytest.mark.asyncio
async def test_chat_rejects_empty_message():
    """Empty message must fail Pydantic validation with 422."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/chat", json={
            "session_id": "test-001",
            "message": "",
        })
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_message_type_is_optional():
    """message_type field is optional — request without it must succeed."""
    with (
        patch("app.services.intent_classifier.classify", new=AsyncMock(return_value=("concept_question", 0.9))),
        patch("app.services.retrieval.retrieve", new=AsyncMock(return_value=_make_retrieval_result())),
        patch("app.services.prompt_chains.generate_socratic_question",
              new=AsyncMock(return_value="What does the chain rule tell us?")),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/chat", json={
                "session_id": "test-001",
                "message": "How does backprop work?",
                # No message_type field
            })
    assert response.status_code == 200


# ─── Intent routing tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_out_of_scope_routing():
    """Low similarity score must trigger out_of_scope response type."""
    with (
        patch("app.services.intent_classifier.classify", new=AsyncMock(return_value=("concept_question", 0.8))),
        patch("app.services.retrieval.retrieve", new=AsyncMock(return_value=_make_retrieval_result(similarity=0.10))),
        patch("app.services.prompt_chains.generate_out_of_scope_response",
              new=AsyncMock(return_value="That topic is outside my curriculum.")),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/chat", json={
                "session_id": "test-002",
                "message": "Explain transformer attention mechanisms",
            })

    assert response.status_code == 200
    data = response.json()
    assert data["response_type"] == "out_of_scope"
    assert data["confidence"]["level"] == "red"


@pytest.mark.asyncio
async def test_misconception_routing():
    """wrong_answer intent with good retrieval must trigger misconception_diagnosis."""
    misconception_result = MagicMock()
    misconception_result.misconception_label = "GRADIENT_DIRECTION_CONFUSION"
    misconception_result.misconception_description = "Student thinks gradients point toward the minimum."
    misconception_result.guided_response = "If the gradient tells you the steepest ascent direction, which way would descent go?"

    with (
        patch("app.services.intent_classifier.classify", new=AsyncMock(return_value=("wrong_answer", 0.95))),
        patch("app.services.retrieval.retrieve", new=AsyncMock(return_value=_make_retrieval_result(0.82))),
        patch("app.services.misconception_engine.diagnose", new=AsyncMock(return_value=misconception_result)),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/chat", json={
                "session_id": "test-003",
                "message": "The gradient points toward the minimum of the loss function",
            })

    assert response.status_code == 200
    data = response.json()
    assert data["response_type"] == "misconception_diagnosis"
