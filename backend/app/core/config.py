from typing import Any, List, Optional
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Path to the repo root: rag-chatbot/
BASE_DIR = Path(__file__).resolve().parents[3]  # core -> app -> backend -> root


class Settings(BaseSettings):
    """
    Central application configuration.

    Values are loaded from environment variables and optional .env file.
    """

    # --- Environment / app ---
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # --- OpenAI / LLM config ---
    OPENAI_API_KEY: SecretStr | None = None
    OPENAI_API_BASE: str | None = None

    CHAT_MODEL_NAME: str = "gpt-4.1-mini"
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-large"

    # --- Database / Postgres + pgvector ---
    DATABASE_URL: str = (
        "postgresql+psycopg://postgres:junaid%40postgres@localhost:5432/rag_chatbot"
    )
    DB_ECHO: bool = False

    PGVECTOR_SCHEMA: Optional[str] = None
    PGVECTOR_COLLECTION: str = "documents"

    # --- RAG / retrieval defaults ---
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20
    DEFAULT_TEMPERATURE: float = 0.2

    # --- CORS ---
    CORS_ALLOW_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8501",
    ]

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",      # <--- always use rag-chatbot/.env
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("CORS_ALLOW_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        if isinstance(v, list):
            return v
        return ["*"]


settings = Settings()
