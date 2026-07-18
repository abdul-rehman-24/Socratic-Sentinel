# Local Development Setup

Two dev servers run in parallel. Open **two terminal windows**.

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.11+ | [python.org](https://python.org) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org) |
| npm | 9+ | Bundled with Node |

---

## Backend (FastAPI — Port 8000)

```powershell
# From repo root
cd backend

# 1. Create & activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1    # Windows PowerShell
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Open .env and set OPENAI_API_KEY=sk-your-key-here

# 4. Start the dev server
uvicorn app.main:app --reload --port 8000
```

**Verify:** Visit [http://localhost:8000/health](http://localhost:8000/health) → `{"status":"ok"}`
**API docs:** [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)

---

## Frontend (Vite React — Port 5173)

```powershell
# From repo root (new terminal window)
cd frontend

# 1. Install dependencies (first time only)
npm install

# 2. Start the dev server
npm run dev
```

**Verify:** Visit [http://localhost:5173](http://localhost:5173)

The Vite dev server proxies all `/api/*` requests to `http://localhost:8000`
(configured in `frontend/vite.config.js`). No CORS issues in development.

---

## API Contract Reference

### POST /api/chat

**Request:**
```json
{
  "session_id": "uuid-string",
  "message": "What is backpropagation?",
  "message_type": "concept_question"
}
```

**Response:**
```json
{
  "session_id": "uuid-string",
  "response_text": "...",
  "response_type": "socratic_question | misconception_diagnosis | out_of_scope | general",
  "confidence": {
    "level": "green | yellow | red",
    "score": 0.92,
    "grounded_in": ["chunk-id-1"],
    "explanation": "..."
  },
  "knowledge_graph_delta": {
    "updated_nodes": [{ "id": "...", "label": "...", "mastery": 0.45, "status": "learning" }],
    "updated_edges": []
  },
  "curriculum_refs": ["backpropagation.md#chain-rule"]
}
```

### GET /api/graph/{session_id}

**Response (react-force-graph-2d compatible):**
```json
{
  "session_id": "uuid-string",
  "graph": {
    "nodes": [{ "id": "...", "label": "...", "group": "core", "mastery": 0.0, "status": "not_started" }],
    "links": [{ "source": "chain_rule", "target": "backpropagation", "relationship": "prerequisite" }]
  }
}
```

---

## Running Tests (Backend)

```powershell
cd backend
pytest -v
```

---

## Knowledge Base Ingestion (Day 2+)

Before starting the backend for the first time, ingest the curriculum into ChromaDB:

```powershell
cd backend
.venv\Scripts\Activate.ps1
python scripts/ingest_kb.py
```

This only needs to run once — ChromaDB persists to `./chroma_data/` on disk.
Re-run if you add/update knowledge_base `.md` files.

---

## Session State Persistence

> **⚠️ MVP Note:** Session knowledge graph state is **in-memory only**, keyed by
> `session_id` in a Python dict. It is **NOT persisted** across server restarts.
>
> If the backend restarts, all session graph state resets to baseline. This is
> intentional for the hackathon MVP. SQLite persistence is planned for Day 4.
>
> ChromaDB vector data (curriculum embeddings) **IS** persisted to disk — only
> per-session mastery state is ephemeral.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: app` | Run uvicorn from the `/backend` directory, not `/backend/app` |
| CORS errors in browser | Ensure backend is on `:8000`; check `CORS_ORIGINS` in `.env` |
| `react-force-graph-2d` crash | Ensure Node 18+; clear `node_modules` and reinstall |
| ChromaDB write error | Check `CHROMA_PERSIST_DIR` path exists and is writable |
| Empty graph after restart | Expected — session state is in-memory. Re-chat to rebuild |
| `ingest_kb.py` fails | Ensure `OPENAI_API_KEY` is set in `.env` before ingesting |
