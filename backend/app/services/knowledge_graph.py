"""
services/knowledge_graph.py — Session-based knowledge graph (in-memory, networkx).

PERSISTENCE NOTE (MVP):
    Graph state is in-memory only, keyed by session_id in _session_graphs dict.
    It is NOT persisted across server restarts — this is intentional for the
    hackathon MVP. Documented in DEV_SETUP.md under "Session State Persistence".

CONTENT AVAILABILITY (Day 3):
    Each node carries a content_available flag.
    True  → backpropagation.md (and future topics) are indexed in ChromaDB.
    False → node exists in the curriculum topology but has no indexed content yet.
            Queries about these nodes will correctly fall back to out_of_scope.
    The frontend visually distinguishes unavailable nodes (reduced opacity).

MASTERY PROGRESSION RULES:
    concept_question engaged   → mastery += 0.15  (student is exploring)
    wrong_answer diagnosed     → mastery -= 0.05  (penalize, but small — still learning)
    concept discussed (general)→ mastery += 0.10
    All changes capped to [0.0, 1.0]

STATUS LABELS (based on mastery):
    0.00        → not_started
    0.01 – 0.39 → learning
    0.40 – 0.79 → practiced
    0.80 – 1.00 → mastered
"""

import logging
from typing import Dict, List, Optional

import networkx as nx

from app.models.graph import GraphData, GraphNode, GraphLink
from app.models.chat import KnowledgeGraphDelta, GraphNodeDelta

logger = logging.getLogger(__name__)

# ─── Topics with real curriculum content indexed in ChromaDB ──────────────────
# Update this set as new .md files are properly ingested.
CONTENT_AVAILABLE_TOPICS = {
    "backpropagation",
    # Add more as content is written and ingested:
    "gradient_descent",
    "optimizers",
}

# ─── Full curriculum topology (session-independent baseline) ──────────────────
_CURRICULUM_NODES: List[dict] = [
    {"id": "perceptron",           "label": "Perceptron",                   "group": "core"},
    {"id": "mlp",                  "label": "Multi-Layer Perceptron",       "group": "core"},
    {"id": "activation_functions", "label": "Activation Functions",          "group": "core"},
    {"id": "chain_rule",           "label": "Chain Rule (Calculus)",         "group": "prerequisite"},
    {"id": "backpropagation",      "label": "Backpropagation",              "group": "core"},
    {"id": "gradient_descent",     "label": "Gradient Descent",             "group": "core"},
    {"id": "optimizers",           "label": "Optimizers (SGD/Adam)",        "group": "core"},
    {"id": "loss_functions",       "label": "Loss Functions",               "group": "core"},
    {"id": "overfitting",          "label": "Overfitting & Regularization", "group": "core"},
    {"id": "cnn_basics",           "label": "CNN Fundamentals",             "group": "advanced"},
    {"id": "conv_layers",          "label": "Convolutional Layers",         "group": "advanced"},
    {"id": "pooling",              "label": "Pooling Layers",               "group": "advanced"},
]

_CURRICULUM_EDGES: List[dict] = [
    {"source": "perceptron",           "target": "mlp",              "relationship": "prerequisite"},
    {"source": "mlp",                  "target": "backpropagation",  "relationship": "prerequisite"},
    {"source": "chain_rule",           "target": "backpropagation",  "relationship": "prerequisite"},
    {"source": "activation_functions", "target": "mlp",              "relationship": "prerequisite"},
    {"source": "backpropagation",      "target": "gradient_descent", "relationship": "extends"},
    {"source": "gradient_descent",     "target": "optimizers",       "relationship": "extends"},
    {"source": "loss_functions",       "target": "backpropagation",  "relationship": "prerequisite"},
    {"source": "mlp",                  "target": "cnn_basics",       "relationship": "prerequisite"},
    {"source": "cnn_basics",           "target": "conv_layers",      "relationship": "extends"},
    {"source": "cnn_basics",           "target": "pooling",          "relationship": "extends"},
    {"source": "backpropagation",      "target": "overfitting",      "relationship": "related"},
]

# ─── Mastery deltas per interaction type ─────────────────────────────────────
_DELTA = {
    "concept_question": +0.15,
    "wrong_answer":     -0.05,
    "general":          +0.10,
}

# ─── Session stores ───────────────────────────────────────────────────────────
_session_graphs: Dict[str, nx.DiGraph] = {}
# Per-node interaction history: session_id → concept_id → list of interaction dicts
_node_histories: Dict[str, Dict[str, List[dict]]] = {}


def _build_curriculum_graph() -> nx.DiGraph:
    """Build a fresh networkx DiGraph from the curriculum topology."""
    g = nx.DiGraph()
    for n in _CURRICULUM_NODES:
        g.add_node(
            n["id"],
            label=n["label"],
            group=n["group"],
            mastery=0.0,
            status="not_started",
            content_available=n["id"] in CONTENT_AVAILABLE_TOPICS,
        )
    for e in _CURRICULUM_EDGES:
        g.add_edge(e["source"], e["target"], relationship=e["relationship"])
    return g


def _mastery_to_status(mastery: float) -> str:
    if mastery <= 0.0:
        return "not_started"
    elif mastery < 0.40:
        return "learning"
    elif mastery < 0.80:
        return "practiced"
    else:
        return "mastered"


def _get_or_create(session_id: str) -> nx.DiGraph:
    if session_id not in _session_graphs:
        _session_graphs[session_id] = _build_curriculum_graph()
        _node_histories[session_id] = {}
        logger.info("Created new session graph for session %s", session_id)
    return _session_graphs[session_id]


# ─── Public API ───────────────────────────────────────────────────────────────

async def update_from_interaction(
    session_id: str,
    concept_ids: List[str],
    intent: str,
    response_type: str = "general",
    response_text: str = "",
) -> KnowledgeGraphDelta:
    """
    Update mastery for concept nodes and record the interaction for click-to-expand.

    Args:
        session_id:    The user's session UUID.
        concept_ids:   Concept node IDs touched in this turn.
        intent:        Classified intent: concept_question | wrong_answer | general.
        response_type: The response type produced (for history display).
        response_text: First 120 chars of the response (for history display).
    """
    g = _get_or_create(session_id)
    delta = float(_DELTA.get(intent, 0.10))
    updated = []

    for concept_id in concept_ids:
        if concept_id not in g.nodes:
            g.add_node(
                concept_id,
                label=concept_id.replace("_", " ").title(),
                group="core",
                mastery=0.0,
                status="not_started",
                content_available=concept_id in CONTENT_AVAILABLE_TOPICS,
            )

        current = g.nodes[concept_id].get("mastery", 0.0)
        new_mastery = round(max(0.0, min(1.0, current + delta)), 4)
        status = _mastery_to_status(new_mastery)

        g.nodes[concept_id]["mastery"] = new_mastery
        g.nodes[concept_id]["status"] = status

        # Record interaction history for this node (click-to-expand)
        if session_id not in _node_histories:
            _node_histories[session_id] = {}
        if concept_id not in _node_histories[session_id]:
            _node_histories[session_id][concept_id] = []

        _node_histories[session_id][concept_id].append({
            "intent": intent,
            "response_type": response_type,
            "mastery_before": round(current, 4),
            "mastery_after": new_mastery,
            "preview": response_text[:120] + ("..." if len(response_text) > 120 else ""),
        })

        updated.append(
            GraphNodeDelta(
                id=concept_id,
                label=g.nodes[concept_id].get("label", concept_id),
                mastery=new_mastery,
                status=status,
                mastery_before=round(current, 4),
            )
        )

    return KnowledgeGraphDelta(updated_nodes=updated, updated_edges=[])


async def get_session_graph(session_id: str) -> GraphData:
    """Serialize a session's networkx graph to GraphData (react-force-graph-2d format)."""
    g = _get_or_create(session_id)
    nodes = [
        GraphNode(
            id=n,
            label=data.get("label", n),
            group=data.get("group", "core"),
            mastery=data.get("mastery", 0.0),
            status=data.get("status", "not_started"),
            content_available=data.get("content_available", False),
        )
        for n, data in g.nodes(data=True)
    ]
    links = [
        GraphLink(source=u, target=v, relationship=data.get("relationship", "related"))
        for u, v, data in g.edges(data=True)
    ]
    return GraphData(nodes=nodes, links=links)


def get_node_history(session_id: str, concept_id: str) -> List[dict]:
    """Return the interaction history list for a specific concept node."""
    return _node_histories.get(session_id, {}).get(concept_id, [])


def get_mastery_summary(session_id: str) -> str:
    """Return a short text summary of the student's mastery state for prompt injection."""
    if session_id not in _session_graphs:
        return "No prior interaction in this session."
    g = _session_graphs[session_id]
    lines = []
    for n, data in g.nodes(data=True):
        mastery = data.get("mastery", 0.0)
        if mastery > 0:
            lines.append(f"  - {data.get('label', n)}: {data.get('status', 'not_started')} ({mastery:.0%})")
    if not lines:
        return "No concepts explored yet in this session."
    return "Student mastery so far:\n" + "\n".join(lines)
