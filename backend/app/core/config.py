from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from app.audio.types import TTSResponseFormat
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Path to the repo root: ca/
BASE_DIR = Path(__file__).resolve().parents[3]  # core -> app -> backend -> repo root


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
    DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@localhost:5432/rag_chatbot"
    DB_ECHO: bool = False

    PGVECTOR_SCHEMA: Optional[str] = None
    PGVECTOR_COLLECTION: str = "documents"

    # --- RAG / retrieval defaults ---
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20
    DEFAULT_TEMPERATURE: float = 0.2

    # Chunking defaults (used by ingestion)
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 150

    # --- Upload / ingestion ---
    # where uploaded files are temporarily stored before extraction/chunking
    UPLOAD_TMP_DIR: Path = Field(default=Path(".uploads"))

    # max upload size (enforced during streaming save)
    MAX_UPLOAD_MB: int = 25
    # optional explicit override (bytes). if not set, derived from MAX_UPLOAD_MB.
    MAX_UPLOAD_BYTES: Optional[int] = None

    # --- OCR defaults (PDF only) ---
    OCR_DEFAULT_LANGUAGE: str = "eng"
    OCR_DEFAULT_DPI: int = 200
    OCR_MIN_CHARS: int = 50

    # --- Audio flags ---
    ENABLE_TTS: bool = Field(default=True)
    ENABLE_STT: bool = Field(default=True)

    # Model names (adjust to what you actually use)
    TTS_MODEL_NAME: str = Field(default="gpt-4o-mini-tts")  # or "tts-1"
    STT_MODEL_NAME: str = Field(default="gpt-4o-mini-transcribe")  # or "whisper-1"

    # TTS defaults
    TTS_VOICE: str = Field(default="alloy")
    TTS_FORMAT: TTSResponseFormat = Field(default="mp3")

    # --- CORS ---
    CORS_ALLOW_ORIGINS: str = "http://localhost:3000,http://localhost:8000,http://localhost:8501"

    @property
    def cors_allow_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ALLOW_ORIGINS.split(",") if o.strip()]

    # ---------------------------------------------------------------------
    # Agentic A-version settings (NEW)
    # ---------------------------------------------------------------------

    AGENT_ENABLED: bool = Field(
        default=True,
        description="Master switch to enable the agentic orchestrator for /v1/chat.",
    )

    AGENT_MAX_STEPS: int = Field(
        default=6,
        ge=1,
        description="Maximum number of LLM reasoning iterations the agent may perform.",
    )

    AGENT_MAX_TOOL_CALLS: int = Field(
        default=8,
        ge=0,
        description="Maximum number of total tool calls allowed across the full agent run.",
    )

    TOOLS_ALLOWLIST: str = Field(
        default="*",
        description=(
            "Comma-separated allowlist of tool names the agent may call. "
            'Use "*" (or "ALL") to allow all registered tools.'
        ),
    )

    AGENT_DEBUG_TRACE: bool = Field(
        default=False,
        description="If true, agent trace may be captured/logged even when the request debug=false.",
    )

    @property
    def tools_allowlist_list(self) -> List[str]:
        """
        Normalized allowlist list.
        - "*" or "ALL" => return ["*"] sentinel (caller may treat as allow-all)
        - "a,b,c" => ["a","b","c"]
        """
        raw = self.TOOLS_ALLOWLIST.strip()
        if not raw:
            return []
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if any(p in ("*", "ALL") for p in parts):
            return ["*"]
        return parts

    # ---------------------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("UPLOAD_TMP_DIR", mode="before")
    @classmethod
    def coerce_upload_dir(cls, v: Any) -> Path:
        if isinstance(v, Path):
            return v
        return Path(str(v))

    @model_validator(mode="after")
    def compute_upload_bytes(self) -> "Settings":
        # If MAX_UPLOAD_BYTES isn't provided, compute it from MAX_UPLOAD_MB.
        if self.MAX_UPLOAD_BYTES is None:
            self.MAX_UPLOAD_BYTES = int(self.MAX_UPLOAD_MB) * 1024 * 1024
        return self


settings = Settings()
