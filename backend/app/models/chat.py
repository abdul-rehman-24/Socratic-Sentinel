"""
chat.py — Pydantic schemas for the /api/chat endpoint.

These models define the STABLE API contract between the React frontend and
FastAPI backend. Do NOT change field names without updating the frontend client.
"""

from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from enum import Enum


# ─── Enums ───────────────────────────────────────────────────────────────────

class MessageType(str, Enum):
    """
    Internal intent classification labels — produced by intent_classifier.py.
    OUT_OF_SCOPE is server-internal only (not exposed to client via message_type).
    """
    CONCEPT_QUESTION = "concept_question"
    WRONG_ANSWER = "wrong_answer"
    OUT_OF_SCOPE = "out_of_scope"
    GENERAL = "general"


class ResponseType(str, Enum):
    SOCRATIC_QUESTION = "socratic_question"
    MISCONCEPTION_DIAGNOSIS = "misconception_diagnosis"
    OUT_OF_SCOPE = "out_of_scope"
    GENERAL = "general"


class ConfidenceLevel(str, Enum):
    GREEN = "green"    # High grounding: ≥0.75 similarity to curriculum chunks
    YELLOW = "yellow"  # Moderate grounding: 0.45–0.74
    RED = "red"        # Low/no grounding: <0.45 or no curriculum match


# ─── Sub-models ──────────────────────────────────────────────────────────────

class ConfidenceInfo(BaseModel):
    """Grounding indicator attached to every chat response."""
    level: ConfidenceLevel
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score 0–1")
    grounded_in: List[str] = Field(
        default_factory=list,
        description="Curriculum chunk IDs this response is grounded in"
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation of the confidence level"
    )


class GraphNodeDelta(BaseModel):
    """A single node's updated state after a chat turn."""
    id: str
    label: str
    mastery: float = Field(..., ge=0.0, le=1.0)
    status: Literal["not_started", "learning", "practiced", "mastered"]
    mastery_before: float = Field(default=0.0, ge=0.0, le=1.0, description="Mastery before this interaction")


class GraphEdgeDelta(BaseModel):
    """A single edge's updated state after a chat turn."""
    source: str
    target: str
    relationship: str


class KnowledgeGraphDelta(BaseModel):
    """Partial graph update returned with each chat response."""
    updated_nodes: List[GraphNodeDelta] = Field(default_factory=list)
    updated_edges: List[GraphEdgeDelta] = Field(default_factory=list)


# ─── Request / Response ──────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Request body for POST /api/chat.

    CORRECTION (Day 2): message_type is now fully Optional and advisory only.
    The server ALWAYS runs intent_classifier.py to determine the true intent.
    The client field is kept for debugging convenience but never used for routing.
    """
    session_id: str = Field(..., description="UUID identifying the user session")
    message: str = Field(..., min_length=1, max_length=2000)
    message_type: Optional[MessageType] = Field(
        default=None,
        description=(
            "Optional client hint — server always re-classifies via intent_classifier. "
            "Never used for routing decisions."
        )
    )


class ChatResponse(BaseModel):
    """Response body from POST /api/chat."""
    session_id: str
    response_text: str
    response_type: ResponseType
    confidence: ConfidenceInfo
    knowledge_graph_delta: KnowledgeGraphDelta
    curriculum_refs: List[str] = Field(
        default_factory=list,
        description="Knowledge base chunk IDs referenced in the response"
    )
