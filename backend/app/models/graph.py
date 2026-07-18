"""
graph.py — Pydantic schemas for the /api/graph/{session_id} endpoint.

The graph format is intentionally compatible with react-force-graph-2d's
expected data shape: { nodes: [...], links: [...] }.
No frontend transformation needed — consume directly.

Day 3 additive change: GraphNode now includes content_available (bool).
  True  → real curriculum content exists for this concept in ChromaDB.
  False → node is in the curriculum topology but has no indexed knowledge base
          content yet. Retrieval will fall back to out_of_scope for this topic.
  Frontend: visually dim / use dashed border for content_available=False nodes.
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class GraphNode(BaseModel):
    """
    A concept node in the knowledge graph.
    react-force-graph-2d uses 'id' as the unique key.
    x/y are null until the force simulation assigns positions.
    content_available=False means no curriculum chunks are indexed for this concept.
    """
    id: str
    label: str
    group: Literal["core", "prerequisite", "advanced"] = "core"
    mastery: float = Field(default=0.0, ge=0.0, le=1.0)
    status: Literal["not_started", "learning", "practiced", "mastered"] = "not_started"
    content_available: bool = Field(
        default=True,
        description=(
            "True if real curriculum content exists for this concept. "
            "False = topology node only — retrieval will return out_of_scope."
        )
    )
    x: Optional[float] = None
    y: Optional[float] = None


class GraphLink(BaseModel):
    """
    A directed edge between two concept nodes.
    react-force-graph-2d uses 'source' and 'target' as node IDs.
    """
    source: str
    target: str
    relationship: Literal["prerequisite", "related", "extends"] = "related"


class GraphData(BaseModel):
    """The full graph payload — matches react-force-graph-2d's data prop exactly."""
    nodes: List[GraphNode] = Field(default_factory=list)
    links: List[GraphLink] = Field(default_factory=list)


class GraphResponse(BaseModel):
    """Response body for GET /api/graph/{session_id}."""
    session_id: str
    graph: GraphData

