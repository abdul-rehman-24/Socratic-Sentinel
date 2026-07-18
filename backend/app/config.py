"""
config.py — Application settings loaded from environment variables via pydantic-settings.
All secrets (OpenAI key, etc.) are defined here and never hardcoded elsewhere.
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    # OpenAI
    openai_api_key: str = "sk-placeholder"
    openai_model: str = "gpt-4o"  # swap to gpt-4.1 / gpt-5.6 when available

    # ChromaDB
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "curriculum"

    # Knowledge base
    knowledge_base_dir: str = "../knowledge_base"

    # Session storage (SQLite)
    db_path: str = "./sessions.db"

    # CORS — comma-separated list of allowed origins
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # App
    app_title: str = "Socratic Sentinel API"
    app_version: str = "0.2.0"
    debug: bool = True


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton — import this everywhere instead of instantiating directly."""
    return Settings()
