"""
scripts/ingest_kb.py — One-time knowledge base ingestion into ChromaDB.

Run this before starting the backend for the first time, or after adding
new .md files to the knowledge_base/ directory.

Usage (from /backend directory with venv active):
    python scripts/ingest_kb.py

What it does:
    1. Reads all .md files in /knowledge_base/
    2. Splits each file on H2 headers ("## Chunk: <slug>")
    3. Embeds each chunk using OpenAI text-embedding-3-small
    4. Upserts into the ChromaDB collection (idempotent — safe to re-run)

Chunk IDs are stable: "{topic}#{slug}" (e.g., "backpropagation#chain-rule")
"""

import os
import re
import sys
from pathlib import Path

# Allow running from /backend directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import chromadb
from chromadb.utils import embedding_functions


def parse_chunks(md_path: Path) -> list[dict]:
    """
    Parse an .md curriculum file into retrievable chunks.

    Each H2 section starting with '## Chunk:' becomes one chunk.
    The frontmatter provides topic metadata.
    """
    text = md_path.read_text(encoding="utf-8")

    # Extract topic from frontmatter id field
    topic_match = re.search(r"^id:\s*(\S+)", text, re.MULTILINE)
    topic = topic_match.group(1) if topic_match else md_path.stem

    # Split on "## Chunk:" headers
    chunk_pattern = re.compile(r"^## Chunk:\s*(.+)$", re.MULTILINE)
    splits = chunk_pattern.split(text)

    chunks = []

    # splits[0] is the preamble/overview (include as its own chunk)
    preamble = splits[0].strip()
    if len(preamble) > 100:  # only if substantial content
        chunks.append({
            "chunk_id": f"{topic}#overview",
            "source_file": md_path.name,
            "topic": topic,
            "content": preamble,
        })

    # splits[1::2] = slug names, splits[2::2] = chunk bodies
    slugs = splits[1::2]
    bodies = splits[2::2]

    for slug, body in zip(slugs, bodies):
        slug_clean = slug.strip().lower().replace(" ", "-")
        chunk_id = f"{topic}#{slug_clean}"
        content = body.strip()

        if len(content) < 50:
            continue  # skip near-empty chunks

        chunks.append({
            "chunk_id": chunk_id,
            "source_file": md_path.name,
            "topic": topic,
            "content": content,
        })

    return chunks


def main():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        print("ERROR: OPENAI_API_KEY not set in .env — cannot embed chunks.")
        sys.exit(1)

    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "curriculum")
    kb_dir = Path(__file__).parent.parent.parent / "knowledge_base"

    if not kb_dir.exists():
        print(f"ERROR: knowledge_base directory not found at {kb_dir}")
        sys.exit(1)

    print(f"📚 Knowledge base directory: {kb_dir}")
    print(f"💾 ChromaDB persist dir:     {chroma_dir}")
    print(f"📦 Collection name:          {collection_name}")
    print()

    # Set up ChromaDB with OpenAI embeddings
    oai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=oai_ef,
        metadata={"hnsw:space": "cosine"},
    )

    md_files = list(kb_dir.glob("*.md"))
    if not md_files:
        print("WARNING: No .md files found in knowledge_base/")
        sys.exit(0)

    total_upserted = 0
    for md_path in sorted(md_files):
        chunks = parse_chunks(md_path)
        if not chunks:
            print(f"  ⚠️  No chunks found in {md_path.name} — skipping")
            continue

        print(f"  📄 {md_path.name} → {len(chunks)} chunk(s)")

        # Filter out placeholder files with minimal content
        real_chunks = [c for c in chunks if len(c["content"]) > 200]
        if not real_chunks:
            print(f"      ⚠️  All chunks too short (placeholder file?) — skipping")
            continue

        collection.upsert(
            ids=[c["chunk_id"] for c in real_chunks],
            documents=[c["content"] for c in real_chunks],
            metadatas=[
                {
                    "chunk_id": c["chunk_id"],
                    "source_file": c["source_file"],
                    "topic": c["topic"],
                }
                for c in real_chunks
            ],
        )

        for c in real_chunks:
            print(f"      ✓ {c['chunk_id']}")

        total_upserted += len(real_chunks)

    print()
    print(f"✅ Ingestion complete — {total_upserted} chunk(s) upserted into '{collection_name}'")
    print(f"   Collection now contains {collection.count()} total documents.")


if __name__ == "__main__":
    main()
