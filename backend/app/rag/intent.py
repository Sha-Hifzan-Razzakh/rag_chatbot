from __future__ import annotations

import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.prompts import INTENT_PROMPT
from app.models.schemas import IntentLiteral

logger = logging.getLogger(__name__)

_intent_llm: Optional[ChatOpenAI] = None


def get_intent_llm() -> ChatOpenAI:
    """
    Lazily initialize and return a ChatOpenAI instance for intent classification.

    Uses a low temperature to keep outputs deterministic and stable.
    """
    global _intent_llm

    if settings.OPENAI_API_KEY is None:
        raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file.")

    api_key = settings.OPENAI_API_KEY.get_secret_value()

    if _intent_llm is None:
        logger.info("Initializing ChatOpenAI LLM for intent classification.")
        _intent_llm = ChatOpenAI(
            model=settings.CHAT_MODEL_NAME,
            temperature=0.0,
            api_key=api_key,
            base_url=settings.OPENAI_API_BASE or None,
        )

    return _intent_llm


def _normalize_intent(raw: str) -> IntentLiteral:
    """
    Normalize raw model output into one of: RAG_QA, CHITCHAT, OTHER.
    """
    text = (raw or "").strip().upper()
    logger.debug("Raw intent model output: %r", text)

    # Be forgiving if the model adds explanations like "Intent: RAG_QA"
    if "RAG_QA" in text or "RAG" in text:
        return "RAG_QA"
    if "CHITCHAT" in text or "CHAT" in text or "SMALLTALK" in text:
        return "CHITCHAT"
    if "OTHER" in text:
        return "OTHER"

    # Fallback: default to RAG_QA (safer for a RAG system)
    logger.warning("Unexpected intent output %r, defaulting to RAG_QA.", text)
    return "RAG_QA"


async def classify_intent(question: str) -> IntentLiteral:
    """
    Classify the user's question into one of: RAG_QA, CHITCHAT, OTHER.

    Uses INTENT_PROMPT and a small LLM call via ChatOpenAI.
    """
    if not question or not question.strip():
        logger.debug("Empty question passed to classify_intent, returning OTHER.")
        return "OTHER"

    llm = get_intent_llm()
    prompt = INTENT_PROMPT.format(question=question)

    logger.info("Classifying intent for question=%r", question)

    # ChatOpenAI accepts a list of messages; for simplicity we send a single user message.
    response = await llm.ainvoke(prompt)  # type: ignore[arg-type]
    raw_text = getattr(response, "content", "") if response is not None else ""

    intent = _normalize_intent(raw_text)
    logger.info("Intent classified as: %s", intent)
    return intent
