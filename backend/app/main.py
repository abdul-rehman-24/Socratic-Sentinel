"""
main.py — FastAPI application factory.

Creates the FastAPI app, configures CORS for local frontend dev,
mounts all API routers under /api prefix, and exposes a health check.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import chat, graph

settings = get_settings()

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "Socratic Sentinel — A Transparent, Adaptive Deep Learning Tutor API. "
        "Teaches DL concepts via Socratic questioning, misconception diagnosis, "
        "and real-time knowledge graph tracking."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# Allows the Vite dev server (localhost:5173) to call the API without CORS errors.
origins = [o.strip() for o in settings.cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(graph.router, prefix="/api", tags=["Knowledge Graph"])


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    """Simple liveness probe — confirms the API is running."""
    return {"status": "ok", "version": settings.app_version}
