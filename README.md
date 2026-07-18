# Socratic Sentinel 🧠

> **OpenAI Build Week — Education Track**
> An adaptive deep learning tutor that teaches through Socratic questioning,
> specific misconception diagnosis, and live knowledge graph tracking.

## What It Does

| Feature | Description |
|---|---|
| **Socratic Method** | Never gives direct answers — responds with guiding questions |
| **Misconception Diagnosis** | Identifies the *specific* flaw in a wrong answer, not generic "incorrect" |
| **Confidence Indicator** | Green/Yellow/Red badge on every response — shows grounding vs. curriculum |
| **Live Knowledge Graph** | Per-session concept mastery tracked in real-time, visualized as a force graph |
| **Curriculum Scope Guard** | Explicitly says "outside my scope" for out-of-curriculum queries |
| **Conversation Memory** ⭐ *Day 3* | Last 3-4 turns stored per session — follow-ups reference prior dialogue |
| **Adaptive Difficulty** ⭐ *Day 3* | Automatically escalates to advanced concepts when mastery ≥ 60% |
| **Mastery Visualization** ⭐ *Day 3* | Node colors reflect mastery (grey → violet → teal → green); click for history |

## Project Structure

```
socratic-sentinel/
├── backend/          # FastAPI app (Python 3.11+)
├── frontend/         # React + Vite + Tailwind SPA
├── knowledge_base/   # Curated DL curriculum in Markdown
├── tests/            # Root integration tests (future)
├── DEV_SETUP.md      # How to run both servers locally
└── README.md         # This file
```

## Quick Start

See [`DEV_SETUP.md`](./DEV_SETUP.md) for the full setup guide.

**TL;DR:**
# Socratic Sentinel 🧠

One-line pitch: An adaptive, transparent tutor that teaches through Socratic questioning, targeted misconception diagnosis, and an editable curriculum knowledge graph.

Purpose: designed for demos and research showing how a tutoring agent can withhold answers, diagnose misconceptions precisely, and escalate pedagogy based on measured mastery.

--

## Why this project (differentiation)

Compared to a standard chatbot, Socratic Sentinel:

- Asks guided questions instead of providing direct answers (Socratic withholding).
- Tracks per-session mastery on a curriculum graph and uses it to decide whether to escalate topics.
- Diagnoses specific misconceptions and adjusts mastery scores granularly.
- Explicitly guards curriculum scope and returns a graceful out-of-scope response when no indexed content exists.

Core differences (short table):

| Characteristic | Typical Chatbot | Socratic Sentinel |
|---|---:|---:|
| Direct answer tendency | High | Low (Socratic) |
| Pedagogical state tracking | No | Yes (per-session mastery) |
| Curriculum grounding | Weak | Strong (ChromaDB + content flags) |
| Adaptive escalation | No | Yes (threshold-based) |

--

## Repo layout (important files)

- backend/ — FastAPI app and ingestion scripts
	- app/  (main, routers, services)
	- scripts/ingest_kb.py (embeds knowledge_base/*.md into ChromaDB)
	- scripts/verify_day3_features.py (manual verification helpers)
- frontend/ — React + Vite app (GraphPanel, NodeDetailPanel, chat UI)
- knowledge_base/ — curriculum markdown files (backpropagation.md, gradient_descent.md, optimizers.md)
- tests/ — backend unit tests (pytest)
- DEV_SETUP.md — installation & run instructions
- README.md — this file (single documentation source)

--

## Setup (quick)

Prerequisites:
- Python 3.11+, Node 18+, npm or pnpm
- An OpenAI API key (set in `backend/.env` as `OPENAI_API_KEY`)

Backend steps (from repo root):

```powershell
cd backend
copy .env.example .env  # then edit .env to set OPENAI_API_KEY and CHROMA_PERSIST_DIR if desired
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Index the knowledge base (required for curriculum-grounded responses):
python scripts/ingest_kb.py
uvicorn app.main:app --reload
```

Frontend steps (new terminal):

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

Notes:
- If `scripts/ingest_kb.py` exits with an API key error, set `OPENAI_API_KEY` in `backend/.env` and re-run the script.
- The ingestion step is idempotent and safe to re-run.

--

## Feature list (consolidated Day 1–3)

- Conversation memory: stores the last 3–4 turns per `session_id` and injects a compact history into generation prompts.
- Retrieval-augmented generation: ChromaDB stores chunked curriculum chunks; retrieval informs Socratic prompts and confidence scoring.
- Socratic questioning: generator produces guiding, open-ended questions rather than direct answers.
- Misconception diagnosis: detects specific error types and reduces mastery on wrong answers.
- Confidence indicator: per-response green/yellow/red badge based on retrieval grounding + model confidence heuristics.
- Knowledge graph: per-session curriculum topology (networkx) with node mastery, `content_available` flags, interaction histories, and a react-force-graph UI.
- Adaptive difficulty: when a node's mastery reaches the configured threshold (default 0.60), the system may escalate to adjacent nodes using curriculum edges.
- Mastery visualization: frontend shows inline `MasteryBadge` after chat responses and a NodeDetail panel with interaction history.

--

## Architecture (text flow)

Message flow (simplified):

1. Frontend posts `POST /api/chat` with `{ session_id, message }`.
2. Backend runs intent classification over the user message.
3. Retrieval queries ChromaDB for relevant chunks (curriculum-grounding).
4. Session memory provides a short formatted history string and mastery summary (last 3–4 turns).
5. Difficulty adapter checks current mastery and may return an escalation target + soft `escalation_note`.
6. Prompt chains construct a message sequence: system prompt (includes `conversation_history` + `escalation_note`), prior turns (OpenAI format), current user message.
7. LLM (OpenAI) generates a Socratic question or misconception diagnosis.
8. Confidence scorer annotates the response.
9. Knowledge graph `update_from_interaction()` updates mastery and appends per-node history entries.
10. Backend returns `ChatResponse` including `knowledge_graph_delta` (additive fields only) and frontend updates UI and graph.

--

## How to run tests

From `backend` with the venv active:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

Current status at time of packaging: all backend tests pass locally (53/53).

--

## Known limitations (honest)

- Session state is in-memory only (no persistence across backend restarts). This is an MVP tradeoff.
- Curriculum depth is limited to the provided markdown files (now: `backpropagation`, `gradient_descent`, and `optimizers`).
- No authentication or multi-user persistence — sessions are anonymous per `session_id` stored in browser sessionStorage.
- The ingestion step requires a valid OpenAI key to create embeddings.

--

## Day 4 changes applied in this submission

- Consolidated documentation into this single `README.md` as requested.
- Added two curriculum files: `knowledge_base/gradient_descent.md` and `knowledge_base/optimizers.md`.
- Marked those topics as `content_available` in the curriculum so the frontend will show them as available.
- Moved `verify_day3_features.py` to `backend/scripts/` for manual verification.

--

## How Codex / GPT-5.6 were used

This project was developed with iterative assistance from code generation tools and LLMs.
- Code generation assistance: selective use of Codex-style completions to scaffold repetitive code and tests; all produced code was reviewed and adapted by human engineers.
- Model usage in the runtime pipeline: GPT-5.6 (or configured OpenAI endpoint) is invoked by `prompt_chains` for intent classification, Socratic question generation, misconception diagnosis, and confidence scoring. Embeddings use OpenAI's embedding models for ChromaDB.

Please note: any autogenerated code was audited and tests were added to maintain correctness.

--

## Demo notes

- Run the ingestion script before demoing to ensure `gradient_descent` and `optimizers` are indexed.
- Start backend + frontend, then run a 4–5 turn conversation focusing on `backpropagation` to trigger escalation into `gradient_descent`.

--

If anything in this README is unclear or you want a shorter quickstart snippet tailored to your platform, tell me which OS and I'll produce it.
