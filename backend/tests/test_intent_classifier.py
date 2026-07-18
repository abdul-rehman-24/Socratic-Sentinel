"""
tests/test_intent_classifier.py — Unit tests for intent_classifier service.

Mocks the OpenAI API response to test classification logic in isolation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _mock_openai_response(intent: str, confidence: float = 0.9, reasoning: str = "test"):
    """Build a mock OpenAI chat completion response."""
    import json
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "intent": intent,
        "confidence": confidence,
        "reasoning": reasoning,
    })
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.mark.asyncio
async def test_classifies_concept_question():
    mock_resp = _mock_openai_response("concept_question", 0.95)
    with patch("app.services.intent_classifier._client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        intent, conf = await __import__("app.services.intent_classifier", fromlist=["classify"]).classify(
            "What is backpropagation?"
        )
    assert intent == "concept_question"
    assert conf == 0.95


@pytest.mark.asyncio
async def test_classifies_wrong_answer():
    mock_resp = _mock_openai_response("wrong_answer", 0.88)
    with patch("app.services.intent_classifier._client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        from app.services.intent_classifier import classify
        intent, conf = await classify("The gradient points toward the minimum of the loss")
    assert intent == "wrong_answer"


@pytest.mark.asyncio
async def test_classifies_out_of_scope():
    mock_resp = _mock_openai_response("out_of_scope", 0.92)
    with patch("app.services.intent_classifier._client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        from app.services.intent_classifier import classify
        intent, _ = await classify("Explain transformer self-attention")
    assert intent == "out_of_scope"


@pytest.mark.asyncio
async def test_invalid_label_falls_back_to_general():
    """Unknown intent label from GPT must be normalized to 'general'."""
    mock_resp = _mock_openai_response("TOTALLY_UNKNOWN_LABEL", 0.5)
    with patch("app.services.intent_classifier._client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        from app.services.intent_classifier import classify
        intent, _ = await classify("some message")
    assert intent == "general"


@pytest.mark.asyncio
async def test_api_failure_returns_general():
    """If OpenAI call raises an exception, fail open with 'general'."""
    with patch("app.services.intent_classifier._client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API down"))
        from app.services.intent_classifier import classify
        intent, conf = await classify("some message")
    assert intent == "general"
    assert conf == 0.5
