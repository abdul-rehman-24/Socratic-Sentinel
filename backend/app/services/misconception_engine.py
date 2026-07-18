"""
services/misconception_engine.py — Specific misconception diagnosis coordinator.

Wraps prompt_chains.diagnose_misconception() and applies a structured taxonomy
to ensure diagnosis is named, not freeform.

Taxonomy (also referenced in misconception_diagnosis.txt prompt template):
    GRADIENT_DIRECTION_CONFUSION   — Thinks gradients point toward the minimum
    BACKPROP_VS_FORWARD_PASS       — Confuses when weights are updated
    LEARNING_RATE_SEMANTICS        — Thinks higher LR always = faster convergence
    VANISHING_GRADIENT_CAUSE       — Misattributes vanishing gradient cause
    CNN_PARAMETER_SHARING          — Doesn't understand filter weight sharing
    ACTIVATION_ROLE                — Thinks activations just scale values (no non-linearity)
    SOFTMAX_SCOPE                  — Thinks softmax applies at every layer
    OVERFITTING_MECHANISM          — Confuses memorization with poor generalization
    CHAIN_RULE_APPLICATION         — Errors in applying chain rule across layers
    OTHER                          — Doesn't fit a named category
"""

from dataclasses import dataclass
from typing import List, Optional

from app.services.retrieval import RetrievalResult
from app.services import prompt_chains

KNOWN_MISCONCEPTIONS = {
    "GRADIENT_DIRECTION_CONFUSION",
    "BACKPROP_VS_FORWARD_PASS",
    "LEARNING_RATE_SEMANTICS",
    "VANISHING_GRADIENT_CAUSE",
    "CNN_PARAMETER_SHARING",
    "ACTIVATION_ROLE",
    "SOFTMAX_SCOPE",
    "OVERFITTING_MECHANISM",
    "CHAIN_RULE_APPLICATION",
    "OTHER",
}


@dataclass
class MisconceptionResult:
    """Structured output of the misconception engine."""
    misconception_label: str        # From KNOWN_MISCONCEPTIONS taxonomy
    misconception_description: str  # One-sentence description of the specific flaw
    guided_response: str            # The Socratic question targeting this misconception


async def diagnose(
    wrong_answer: str,
    retrieved_chunks: List[RetrievalResult],
    concept: str = "deep learning",
    conversation_history: Optional[List[dict]] = None,
) -> MisconceptionResult:
    """
    Diagnose the specific misconception in a user's wrong answer.

    Args:
        wrong_answer:         The user's incorrect statement.
        retrieved_chunks:     Curriculum context for the discussed concept.
        concept:              The concept name (e.g., "backpropagation").
        conversation_history: Prior turns for dialogue context (last 2 turns used).

    Returns:
        MisconceptionResult with taxonomy label, description, and Socratic question.
    """
    raw = await prompt_chains.diagnose_misconception(
        wrong_answer=wrong_answer,
        retrieved_chunks=retrieved_chunks,
        concept=concept,
        conversation_history=conversation_history,
    )

    label = raw.get("misconception_label", "OTHER").upper().strip()
    # Normalize to known taxonomy
    if label not in KNOWN_MISCONCEPTIONS:
        label = "OTHER"

    return MisconceptionResult(
        misconception_label=label,
        misconception_description=raw.get(
            "misconception_description",
            "A specific misconception was detected in your reasoning.",
        ),
        guided_response=raw.get(
            "guided_response",
            "Interesting — can you walk me through your reasoning step by step?",
        ),
    )
