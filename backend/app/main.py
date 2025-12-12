from contextlib import asynccontextmanager
from typing import AsyncIterator

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api.v1 import chat, ingest, health
from app.api.v1 import audio

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan context.

    Use this to initialize shared resources such as:
    - Database connections
    - Vector store clients
    - Caches

    And clean them up on shutdown.
    """
    # Startup logic
    setup_logging()
    logger.info("Starting rag-chatbot backend...")
    logger.info("Environment: %s", settings.ENVIRONMENT)
    logger.info("OpenAI model: %s", settings.CHAT_MODEL_NAME)

    # TODO: initialize DB / vector store here, e.g.:
    # from app.rag.embeddings import init_vector_store
    # await init_vector_store()

    try:
        yield
    finally:
        # Shutdown logic
        logger.info("Shutting down rag-chatbot backend...")
        # TODO: close DB / vector store connections if needed
        # await close_vector_store()
        ...


def create_app() -> FastAPI:
    """
    FastAPI application factory.

    Creates and configures the FastAPI app instance,
    attaches middleware and routers.
    """
    app = FastAPI(
        title="RAG Chatbot Backend",
        description="FastAPI backend for a Retrieval-Augmented Generation chatbot.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routers
    app.include_router(health.router, prefix="/v1", tags=["health"])
    app.include_router(chat.router, prefix="/v1", tags=["chat"])
    app.include_router(ingest.router, prefix="/v1", tags=["ingest"])
    app.include_router(audio.router, prefix="/v1", tags=["audio"])

    return app


app = create_app()
