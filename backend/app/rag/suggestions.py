from __future__ import annotations

import json
import logging
from typing import List, Optional

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.prompts import SUGGEST_QUESTIONS_PROMPT

logger = logging.getLogger(__name__)

_suggestions_llm: Optional[ChatOpenAI] = None


def get_suggestions_llm() -> ChatOpenAI:
    """
    Lazily initialize and return a ChatOpenAI instance for follow-up suggestions.
    """
    global _suggestions_llm

    if settings.OPENAI_API_KEY is None:
        raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file.")

    api_key = settings.OPENAI_API_KEY.get_secret_value()

    if _suggestions_llm is None:
        logger.info("Initializing ChatOpenAI LLM for suggestion generation.")
        _suggestions_llm = ChatOpenAI(
            model=settings.CHAT_MODEL_NAME,
            temperature=0.5,  # a bit of creativity is fine here
            api_key=api_key,
            base_url=settings.OPENAI_API_BASE or None,
        )

    return _suggestions_llm


def _safe_parse_json_array(raw: str) -> List[str]:
    """
    Try to parse a JSON array of strings from the model output.

    If parsing fails, try some fallbacks:
    - Interpret each non-empty line as a question.
    - Strip leading bullets / numbering.
    """
    raw = (raw or "").strip()
    logger.debug("Raw suggestions model output: %r", raw)

    # First, try strict JSON parsing
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            # Keep only non-empty strings
            return [str(item).strip() for item in data if str(item).strip()]
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON suggestions, falling back to line-based parsing.")

    # Fallback: treat each line as a potential question
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    suggestions: List[str] = []

    for line in lines:
        # Strip leading bullets / numbering like "- ", "* ", "1. "
        cleaned = line.lstrip("-*0123456789. ").strip()
        if cleaned:
            suggestions.append(cleaned)

    return suggestions


async def generate_suggested_questions(
    question: str,
    answer: str,
    context: str,
    max_suggestions: int = 5,
) -> List[str]:
    """
    Generate follow-up questions based on the question, answer, and context.

    Returns a list of up to `max_suggestions` non-empty strings.
    """
    llm = get_suggestions_llm()

    prompt = SUGGEST_QUESTIONS_PROMPT.format(
        question=question,
        answer=answer,
        context=context,
    )

    logger.info("Generating suggested questions for question=%r", question)

    try:
        response = await llm.ainvoke(prompt)  # type: ignore[arg-type]
        raw_text = getattr(response, "content", "") if response is not None else ""
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error while calling LLM for suggestions: %s", exc)
        return []

    suggestions = _safe_parse_json_array(raw_text)

    # Deduplicate while preserving order
    seen = set()
    unique_suggestions: List[str] = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique_suggestions.append(s)

    # Truncate to requested max_suggestions
    final_suggestions = unique_suggestions[:max_suggestions]

    logger.debug("Generated %d suggestions: %r", len(final_suggestions), final_suggestions)
    return final_suggestions
