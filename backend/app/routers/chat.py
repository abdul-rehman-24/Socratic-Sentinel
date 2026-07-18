"""
routers/chat.py — POST /api/chat — Day 3 pipeline with memory + adaptive difficulty.

End-to-end flow:
    1. classify intent       (intent_classifier)
    2. retrieve context      (retrieval / ChromaDB)
    3. anti-hallucination gate
    4. load conversation history (session_memory)
    5. check adaptive difficulty (difficulty_adapter)
    6. route to:
         concept_question → Socratic question (prompt_chains + history + escalation)
         wrong_answer     → Misconception diagnosis (misconception_engine + history)
         out_of_scope     → Explicit scope-guard (prompt_chains, no history needed)
         general          → Socratic question (fallback)
    7. score confidence  (confidence_scorer)
    8. update knowledge graph (knowledge_graph, with response_text for history)
    9. save turn to session memory (session_memory)
   10. return structured ChatResponse
"""

import logging
from fastapi import APIRouter
from app.models.chat import (
    ChatRequest, ChatResponse, ResponseType,
    ConfidenceInfo, ConfidenceLevel, KnowledgeGraphDelta,
)
from app.services import (
    intent_classifier,
    retrieval,
    prompt_chains,
    misconception_engine,
    confidence_scorer,
    knowledge_graph,
    session_memory,
    difficulty_adapter,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _extract_concept_ids(chunks: list) -> list[str]:
    seen = set()
    ids = []
    for c in chunks:
        cid = c.topic or c.chunk_id.split("#")[0]
        if cid and cid not in seen:
            seen.add(cid)
            ids.append(cid)
    return ids


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint — Day 3 pipeline with conversation memory and adaptive difficulty.

    Anti-hallucination guarantee: if retrieval similarity is below
    THRESHOLD_OUT_OF_SCOPE, the system returns response_type=out_of_scope
    WITHOUT calling GPT for a generative answer.
    """
    session_id = request.session_id
    message = request.message

    # ── 1. Classify intent ────────────────────────────────────────────────────
    intent, _intent_conf = await intent_classifier.classify(message)
    logger.info("[%s] Intent: %s", session_id, intent)

    # ── 2. Retrieve relevant curriculum chunks ────────────────────────────────
    chunks = await retrieval.retrieve(message, top_k=4)
    logger.info("[%s] Retrieved %d chunks, best sim=%.3f",
                session_id, len(chunks), retrieval.best_similarity(chunks))

    # ── 3. Anti-hallucination gate ────────────────────────────────────────────
    if retrieval.is_out_of_scope(chunks) or intent == "out_of_scope":
        intent = "out_of_scope"

    # ── 4. Load conversation history ──────────────────────────────────────────
    conversation_history = session_memory.get_openai_history(session_id)
    logger.info("[%s] History: %d prior messages", session_id, len(conversation_history))

    # ── 5. Check adaptive difficulty (only for concept_question) ──────────────
    escalation_note = ""
    concept_ids = _extract_concept_ids(chunks)

    if intent == "concept_question" and concept_ids:
        primary_concept = concept_ids[0]
        escalation_target = difficulty_adapter.get_escalation_target(session_id, primary_concept)
        if escalation_target:
            escalation_note = difficulty_adapter.build_escalation_note(
                session_id, primary_concept, escalation_target
            )
            logger.info("[%s] Difficulty escalation: %s → %s",
                        session_id, primary_concept, escalation_target)

    # ── 6. Route to appropriate response generator ────────────────────────────
    response_text = ""
    response_type = ResponseType.GENERAL

    if intent == "out_of_scope":
        response_text = await prompt_chains.generate_out_of_scope_response(message)
        response_type = ResponseType.OUT_OF_SCOPE

    elif intent == "wrong_answer":
        concept = chunks[0].topic if chunks else "deep learning"
        result = await misconception_engine.diagnose(
            wrong_answer=message,
            retrieved_chunks=chunks,
            concept=concept,
            conversation_history=conversation_history,
        )
        response_text = (
            f"**Misconception detected: {result.misconception_label.replace('_', ' ').title()}**\n\n"
            f"{result.misconception_description}\n\n"
            f"{result.guided_response}"
        )
        response_type = ResponseType.MISCONCEPTION_DIAGNOSIS

    else:
        mastery_summary = knowledge_graph.get_mastery_summary(session_id)
        response_text = await prompt_chains.generate_socratic_question(
            message=message,
            retrieved_chunks=chunks,
            mastery_summary=mastery_summary,
            conversation_history=conversation_history,
            escalation_note=escalation_note,
        )
        response_type = ResponseType.SOCRATIC_QUESTION

    # ── 7. Score confidence ───────────────────────────────────────────────────
    if intent == "out_of_scope":
        confidence = ConfidenceInfo(
            level=ConfidenceLevel.RED,
            score=retrieval.best_similarity(chunks),
            grounded_in=[],
            explanation="Response is outside the curriculum knowledge base.",
        )
    else:
        confidence = await confidence_scorer.score(chunks)

    # ── 8. Update knowledge graph ─────────────────────────────────────────────
    if concept_ids and intent != "out_of_scope":
        graph_delta = await knowledge_graph.update_from_interaction(
            session_id=session_id,
            concept_ids=concept_ids,
            intent=intent,
            response_type=response_type.value,
            response_text=response_text,
        )
    else:
        graph_delta = KnowledgeGraphDelta()

    # ── 9. Save turn to session memory ────────────────────────────────────────
    session_memory.add_turn(
        session_id=session_id,
        user_message=message,
        assistant_response=response_text,
        intent=intent,
        response_type=response_type.value,
    )

    # ── 10. Return response ───────────────────────────────────────────────────
    return ChatResponse(
        session_id=session_id,
        response_text=response_text,
        response_type=response_type,
        confidence=confidence,
        knowledge_graph_delta=graph_delta,
        curriculum_refs=[c.chunk_id for c in chunks],
    )
