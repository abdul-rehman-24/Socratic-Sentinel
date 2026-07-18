"""
services/difficulty_adapter.py — Adaptive difficulty escalation.

When a student's mastery on a concept exceeds ESCALATION_THRESHOLD (0.6),
the next Socratic question should escalate toward a more advanced adjacent
concept rather than staying on the same node.

Uses the curriculum graph's directed edges to find "successor" concepts
(those connected via 'extends' or 'prerequisite' relationships).

This module is intentionally stateless — it reads from knowledge_graph's
in-memory state and returns a recommendation; it does not write any state.
"""

import logging
from typing import Optional

import networkx as nx

from app.services import knowledge_graph as kg

logger = logging.getLogger(__name__)

# Mastery level above which escalation is triggered
ESCALATION_THRESHOLD = 0.60


def get_escalation_target(session_id: str, current_concept_id: str) -> Optional[str]:
    """
    Check if the student has mastered the current concept sufficiently
    to warrant escalation to a more advanced adjacent concept.

    Args:
        session_id: The current session.
        current_concept_id: The concept ID just discussed.

    Returns:
        The ID of the next concept to escalate to, or None if no escalation needed.
        Returns None if:
          - current mastery is below ESCALATION_THRESHOLD
          - no 'extends' successor exists in the curriculum graph
          - all successors are also already practiced/mastered
    """
    graph = kg._session_graphs.get(session_id)
    if graph is None:
        return None

    if current_concept_id not in graph.nodes:
        return None

    current_mastery = graph.nodes[current_concept_id].get("mastery", 0.0)
    if current_mastery < ESCALATION_THRESHOLD:
        return None  # Not mastered enough to escalate

    # Find successor nodes via 'extends' edges (concept → next level)
    successors = []
    for _, target, edge_data in graph.out_edges(current_concept_id, data=True):
        relationship = edge_data.get("relationship", "")
        if relationship in ("extends", "prerequisite"):
            target_mastery = graph.nodes[target].get("mastery", 0.0)
            target_status = graph.nodes[target].get("status", "not_started")
            # Prefer nodes the student hasn't yet mastered
            if target_status not in ("mastered",):
                successors.append((target, target_mastery))

    if not successors:
        return None

    # Pick the successor with the lowest mastery (least explored next step)
    successors.sort(key=lambda x: x[1])
    chosen = successors[0][0]
    logger.info(
        "[%s] Escalating from %s (mastery=%.0f%%) → %s",
        session_id, current_concept_id, current_mastery * 100, chosen
    )
    return chosen


def build_escalation_note(
    session_id: str,
    current_concept_id: str,
    escalation_target_id: str,
) -> str:
    """
    Build a short instruction string to append to the Socratic prompt
    when difficulty escalation is triggered.

    Args:
        session_id: Current session.
        current_concept_id: The concept being escalated from.
        escalation_target_id: The concept being escalated to.

    Returns:
        A short instruction string for injection into the system prompt.
    """
    graph = kg._session_graphs.get(session_id)
    current_label = current_concept_id
    target_label = escalation_target_id

    if graph:
        current_label = graph.nodes.get(current_concept_id, {}).get("label", current_concept_id)
        target_label = graph.nodes.get(escalation_target_id, {}).get("label", escalation_target_id)

    return (
        f"\n\nESCALATION NOTE: The student has shown good understanding of {current_label}. "
        f"Gently bridge toward {target_label} in your Socratic question — introduce the "
        f"connection between these concepts without explicitly naming the transition."
    )
