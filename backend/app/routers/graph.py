"""
routers/graph.py — Knowledge graph endpoints.

GET /api/graph/{session_id}           — Full session graph (nodes + links)
GET /api/graph/{session_id}/node/{concept_id} — Node detail + interaction history
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from app.models.graph import GraphResponse
from app.services import knowledge_graph

router = APIRouter()


class NodeHistoryEntry(BaseModel):
    intent: str
    response_type: str
    mastery_before: float
    mastery_after: float
    preview: str


class NodeDetailResponse(BaseModel):
    session_id: str
    concept_id: str
    label: str
    mastery: float
    status: str
    content_available: bool
    interaction_count: int
    history: List[NodeHistoryEntry]


@router.get("/graph/{session_id}", response_model=GraphResponse)
async def get_graph(session_id: str) -> GraphResponse:
    """
    Returns the full knowledge graph state for a session.

    For a brand-new session, returns the curriculum baseline with all mastery
    values at 0.0. Mastery values update after each /api/chat call.

    Each node now includes content_available (bool):
      True  → real curriculum content exists (indexed in ChromaDB)
      False → topology node only, no indexed content — retrieval returns out_of_scope

    NOTE: State is in-memory only — resets on server restart (MVP behaviour).
    """
    graph_data = await knowledge_graph.get_session_graph(session_id)
    return GraphResponse(session_id=session_id, graph=graph_data)


@router.get("/graph/{session_id}/node/{concept_id}", response_model=NodeDetailResponse)
async def get_node_detail(session_id: str, concept_id: str) -> NodeDetailResponse:
    """
    Returns detail + interaction history for a specific concept node.

    Used by the click-to-expand panel in the frontend GraphPanel.
    History entries show what happened each time this concept was discussed.
    """
    graph = knowledge_graph._get_or_create(session_id)
    node_data = graph.nodes.get(concept_id, {})
    history_raw = knowledge_graph.get_node_history(session_id, concept_id)

    return NodeDetailResponse(
        session_id=session_id,
        concept_id=concept_id,
        label=node_data.get("label", concept_id),
        mastery=node_data.get("mastery", 0.0),
        status=node_data.get("status", "not_started"),
        content_available=node_data.get("content_available", False),
        interaction_count=len(history_raw),
        history=[NodeHistoryEntry(**h) for h in history_raw],
    )
