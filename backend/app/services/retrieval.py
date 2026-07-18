"""
services/retrieval.py — ChromaDB-backed RAG retrieval.

On startup (via ingest_kb.py script), curriculum .md files are chunked,
embedded with OpenAI text-embedding-3-small, and stored in a local ChromaDB
collection. This module provides the query interface for the chat pipeline.

Chunk format expected in .md files:
    Each H2 section ("## Chunk: <slug>") is treated as one retrievable chunk.
    The YAML frontmatter provides chunk-level metadata.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ─── Confidence thresholds (documented here as source of truth) ───────────────
# These thresholds map ChromaDB cosine distance → ConfidenceLevel.
# ChromaDB returns DISTANCE (lower = more similar). We convert to similarity:
#   similarity = 1 - distance  (for cosine)
#
# GREEN  : similarity ≥ 0.75  → response well-grounded in curriculum
# YELLOW : similarity ≥ 0.45  → partial grounding, some extrapolation
# RED    : similarity <  0.45  → little/no curriculum match, flag as uncertain
# OUT_OF_SCOPE threshold: if best similarity < 0.35, treat as out-of-scope
THRESHOLD_GREEN = 0.75
THRESHOLD_YELLOW = 0.45
THRESHOLD_OUT_OF_SCOPE = 0.35


@dataclass
class RetrievalResult:
    """A single retrieved curriculum chunk with its similarity score."""
    chunk_id: str           # e.g., "backpropagation#chain-rule"
    source_file: str        # e.g., "backpropagation.md"
    content: str            # Raw text of the chunk
    similarity_score: float # 0.0 – 1.0 (converted from cosine distance)
    topic: str = ""         # e.g., "backpropagation"


# ─── ChromaDB client (lazy init) ─────────────────────────────────────────────
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_collection():
    """Lazy-initialize and return the ChromaDB collection."""
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    oai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name="text-embedding-3-small",
    )

    _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    _collection = _chroma_client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=oai_ef,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


async def retrieve(query: str, top_k: int = 4) -> List[RetrievalResult]:
    """
    Retrieve the top-k most relevant curriculum chunks for a query.

    Args:
        query: The user's message or topic string.
        top_k: Maximum chunks to return (default: 4).

    Returns:
        List of RetrievalResult ordered by descending similarity.
        Returns empty list if collection is empty (ingestion not yet run).
    """
    try:
        collection = _get_collection()
        count = collection.count()
        if count == 0:
            logger.warning("ChromaDB collection is empty — run scripts/ingest_kb.py first")
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[RetrievalResult] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            # Convert cosine distance → similarity score
            similarity = max(0.0, min(1.0, 1.0 - dist))
            chunks.append(
                RetrievalResult(
                    chunk_id=meta.get("chunk_id", "unknown"),
                    source_file=meta.get("source_file", "unknown"),
                    content=doc,
                    similarity_score=round(similarity, 4),
                    topic=meta.get("topic", ""),
                )
            )

        return chunks

    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return []


def format_chunks_for_prompt(chunks: List[RetrievalResult]) -> str:
    """Format retrieved chunks into a context block for prompt injection."""
    if not chunks:
        return "(No relevant curriculum content retrieved.)"
    parts = []
    for r in chunks:
        parts.append(f"[Source: {r.chunk_id}]\n{r.content}")
    return "\n\n---\n\n".join(parts)


def best_similarity(chunks: List[RetrievalResult]) -> float:
    """Return the highest similarity score from a list of chunks (0.0 if empty)."""
    if not chunks:
        return 0.0
    return max(c.similarity_score for c in chunks)


def is_out_of_scope(chunks: List[RetrievalResult]) -> bool:
    """True if no chunk clears the out-of-scope threshold."""
    return best_similarity(chunks) < THRESHOLD_OUT_OF_SCOPE
