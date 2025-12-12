from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("/health", summary="Health check")
async def health_check() -> dict:
    """
    Basic health check endpoint.

    Later you can expand this to verify:
    - Database connectivity
    - Vector store readiness
    - External API availability (e.g., OpenAI)
    """
    # Placeholder values for now; update when DB/vector checks are implemented.
    return {
        "status": "ok",
        "environment": settings.ENVIRONMENT,
        "vector_store": "unknown",  # e.g. "ready" / "degraded" / "unavailable"
    }
