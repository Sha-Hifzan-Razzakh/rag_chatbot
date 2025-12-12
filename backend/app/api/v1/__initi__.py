"""
Version 1 API routers for the RAG chatbot backend.
"""

from fastapi import APIRouter

from app.api.v1 import chat, health, ingest

router = APIRouter()
router.include_router(health.router, tags=["health"])
router.include_router(chat.router, tags=["chat"])
router.include_router(ingest.router, tags=["ingest"])
