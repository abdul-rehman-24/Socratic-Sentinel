"""
services/confidence_scorer.py — Response grounding confidence scoring.

Maps ChromaDB similarity scores to a three-level confidence indicator.

Thresholds (source of truth — must match retrieval.py):
    GREEN  : best_similarity ≥ 0.75  → well-grounded in curriculum
    YELLOW : best_similarity ≥ 0.45  → partial grounding
    RED    : best_similarity <  0.45  → low/no grounding (potential hallucination)
"""

from typing import List
from app.models.chat import ConfidenceInfo, ConfidenceLevel
from app.services.retrieval import RetrievalResult, THRESHOLD_GREEN, THRESHOLD_YELLOW

_EXPLANATIONS = {
    ConfidenceLevel.GREEN: (
        "This response is well-grounded in {n} curriculum chunk(s) "
        "(max similarity {score:.0%}). High confidence it's curriculum-accurate."
    ),
    ConfidenceLevel.YELLOW: (
        "This response is partially grounded ({n} chunk(s), max similarity {score:.0%}). "
        "Some claims may extend slightly beyond the curriculum — verify if unsure."
    ),
    ConfidenceLevel.RED: (
        "Low curriculum grounding (max similarity {score:.0%}). "
        "This response may extend beyond the knowledge base. Treat with caution."
    ),
}


async def score(
    retrieved_chunks: List[RetrievalResult],
) -> ConfidenceInfo:
    """
    Score how well a response is grounded in the curriculum.

    Args:
        retrieved_chunks: The chunks retrieved for the user's query.

    Returns:
        ConfidenceInfo with level, numeric score, grounded_in list, and explanation.
    """
    if not retrieved_chunks:
        return ConfidenceInfo(
            level=ConfidenceLevel.RED,
            score=0.0,
            grounded_in=[],
            explanation=(
                "No curriculum chunks matched this query. Response is outside "
                "the knowledge base — the system should have returned out_of_scope."
            ),
        )

    best = max(c.similarity_score for c in retrieved_chunks)
    # Only include chunks that clear the YELLOW threshold in grounded_in list
    grounded = [c.chunk_id for c in retrieved_chunks if c.similarity_score >= THRESHOLD_YELLOW]

    if best >= THRESHOLD_GREEN:
        level = ConfidenceLevel.GREEN
    elif best >= THRESHOLD_YELLOW:
        level = ConfidenceLevel.YELLOW
    else:
        level = ConfidenceLevel.RED

    explanation = _EXPLANATIONS[level].format(
        n=len(grounded) or len(retrieved_chunks),
        score=best,
    )

    return ConfidenceInfo(
        level=level,
        score=round(best, 4),
        grounded_in=grounded,
        explanation=explanation,
    )
