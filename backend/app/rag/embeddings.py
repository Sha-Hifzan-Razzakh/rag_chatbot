from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector  # from `langchain-postgres`
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.core.config import settings

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None


def get_embedding_model() -> OpenAIEmbeddings:
    """
    Create the OpenAI embedding model used for documents and queries.
    """
    if settings.OPENAI_API_KEY is None:
        raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file.")

    # Unwrap the SecretStr to a plain string for the client
    api_key = settings.OPENAI_API_KEY.get_secret_value()

    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL_NAME,
        api_key=api_key,
        base_url=settings.OPENAI_API_BASE,
    )

def get_engine() -> Engine:
    """
    Lazily initialize and return a SQLAlchemy engine for Postgres.
    """
    global _engine

    if _engine is None:
        logger.info(
            "Initializing SQLAlchemy engine for DATABASE_URL=%s",
            settings.DATABASE_URL,
        )
        _engine = create_engine(
            settings.DATABASE_URL,
            echo=settings.DB_ECHO,
            pool_pre_ping=True,
        )

    return _engine


@lru_cache(maxsize=1)
def get_vector_store() -> PGVector:
    """
    Lazily initialize and return a PGVector vector store instance.

    This uses:
    - Postgres with the `pgvector` extension
    - LangChain's PGVector wrapper from `langchain-postgres`
    """
    logger.info(
        "Initializing PGVector vector store (collection: %s).",
        settings.PGVECTOR_COLLECTION,
    )

    # Ensure engine is created (if other parts of the app rely on it)
    get_engine()

    embeddings = get_embedding_model()

    # You can pass either a connection string or a connection object.
    connection = settings.DATABASE_URL

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=settings.PGVECTOR_COLLECTION,
        connection=connection,
        use_jsonb=True,
    )

    logger.info("PGVector vector store initialized.")
    return vector_store
