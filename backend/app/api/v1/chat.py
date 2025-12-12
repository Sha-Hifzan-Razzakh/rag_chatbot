import logging

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.models.schemas import ChatRequest, ChatResponse, IntentLiteral
from app.rag.intent import classify_intent
from app.rag.pipeline import run_rag_pipeline, run_chitchat_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat")


@router.post(
    "",
    response_model=ChatResponse,
    summary="Chat with the RAG chatbot",
    response_description="Chatbot answer with sources and suggested questions.",
)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.

    Flow:
    1. Classify intent: RAG_QA / CHITCHAT / OTHER.
    2. If RAG_QA -> run full RAG pipeline (question rewrite, retrieval, answer generation, suggestions).
    3. If CHITCHAT -> run chitchat pipeline (no retrieval).
    4. If OTHER -> currently treated as RAG_QA or returns a default answer (configurable in future).
    """
    logger.info("Received chat request: question=%r", payload.question)

    try:
        intent: IntentLiteral = await classify_intent(payload.question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to classify intent: %s", exc)
        # Fallback: assume RAG_QA if intent classification fails
        intent = "RAG_QA"

    logger.info("Classified intent: %s", intent)

    # Determine top_k (either request-level or default from config)
    top_k = payload.top_k or settings.DEFAULT_TOP_K
    if top_k > settings.MAX_TOP_K:
        logger.warning(
            "Requested top_k=%s exceeds MAX_TOP_K=%s, capping.",
            top_k,
            settings.MAX_TOP_K,
        )
        top_k = settings.MAX_TOP_K

    try:
        if intent == "CHITCHAT":
            logger.debug("Routing to chitchat pipeline.")
            response = await run_chitchat_pipeline(
                question=payload.question,
                history=payload.history,
                tone=payload.tone or "neutral",
                style=payload.style or "concise",
            )
        elif intent == "RAG_QA":
            logger.debug("Routing to RAG pipeline.")
            response = await run_rag_pipeline(
                question=payload.question,
                history=payload.history,
                tone=payload.tone or "neutral",
                style=payload.style or "concise",
                top_k=top_k,
                namespace=payload.namespace,
            )
        else:
            # For now, treat OTHER the same as RAG_QA.
            logger.debug("Intent OTHER, routing to RAG pipeline (fallback).")
            response = await run_rag_pipeline(
                question=payload.question,
                history=payload.history,
                tone=payload.tone or "neutral",
                style=payload.style or "concise",
                top_k=top_k,
                namespace=payload.namespace,
            )

        # Ensure intent field is populated in the response
        response.intent = intent
        return response

    except HTTPException:
        # Already an HTTPException, just re-raise
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error while handling chat request: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the chat request.",
        ) from exc
