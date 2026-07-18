"""
tests/test_graph.py — Integration tests for GET /api/graph/{session_id}.

Day 1: Tests that the mock graph endpoint returns react-force-graph-2d compatible data.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_graph_returns_valid_structure():
    """Graph endpoint should return nodes and links arrays."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/graph/test-session-001")

    assert response.status_code == 200
    data = response.json()

    assert data["session_id"] == "test-session-001"
    assert "graph" in data
    assert "nodes" in data["graph"]
    assert "links" in data["graph"]

    # Verify node structure (must be react-force-graph-2d compatible)
    nodes = data["graph"]["nodes"]
    assert len(nodes) > 0
    for node in nodes:
        assert "id" in node
        assert "label" in node
        assert "mastery" in node
        assert 0.0 <= node["mastery"] <= 1.0
        assert node["status"] in ["not_started", "learning", "practiced", "mastered"]

    # Verify link structure
    for link in data["graph"]["links"]:
        assert "source" in link
        assert "target" in link
