from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.prompts import (
    ANSWER_PROMPT,
    CONDENSE_QUESTION_PROMPT,
    SELF_CHECK_PROMPT,
)
from app.models.schemas import ChatResponse, Message, Source
from app.rag.retriever import docs_to_context, retrieve_docs
from app.rag.suggestions import generate_suggested_questions

logger = logging.getLogger(__name__)

_answer_llm: Optional[ChatOpenAI] = None
_rewrite_llm: Optional[ChatOpenAI] = None
_self_check_llm: Optional[ChatOpenAI] = None
_chitchat_llm: Optional[ChatOpenAI] = None


CHITCHAT_SYSTEM_PROMPT = """
You are a friendly, concise AI assistant.

You can:
- Greet the user.
- Answer casual questions.
- Engage in light small talk.

You should:
- Be polite and clear.
- Avoid pretending to have access to private data or documents.
- If the user asks about specific internal docs or knowledge bases, gently explain
  that you can only answer general questions in chitchat mode.
""".strip()


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _require_api_key() -> str:
    if settings.OPENAI_API_KEY is None:
        raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file.")
    return settings.OPENAI_API_KEY.get_secret_value()


def get_answer_llm() -> ChatOpenAI:
    global _answer_llm
    api_key = _require_api_key()
    if _answer_llm is None:
        logger.info("Initializing ChatOpenAI LLM for RAG answers.")
        _answer_llm = ChatOpenAI(
            model=settings.CHAT_MODEL_NAME,
            temperature=settings.DEFAULT_TEMPERATURE,
            api_key=api_key,
            base_url=settings.OPENAI_API_BASE or None,
        )
    return _answer_llm


def get_rewrite_llm() -> ChatOpenAI:
    global _rewrite_llm
    api_key = _require_api_key()
    if _rewrite_llm is None:
        logger.info("Initializing ChatOpenAI LLM for question rewriting.")
        _rewrite_llm = ChatOpenAI(
            model=settings.CHAT_MODEL_NAME,
            temperature=0.0,  # deterministic rewrites
            api_key=api_key,
            base_url=settings.OPENAI_API_BASE or None,
        )
    return _rewrite_llm


def get_self_check_llm() -> ChatOpenAI:
    global _self_check_llm
    api_key = _require_api_key()
    if _self_check_llm is None:
        logger.info("Initializing ChatOpenAI LLM for self-check.")
        _self_check_llm = ChatOpenAI(
            model=settings.CHAT_MODEL_NAME,
            temperature=settings.DEFAULT_TEMPERATURE,
            api_key=api_key,
            base_url=settings.OPENAI_API_BASE or None,
        )
    return _self_check_llm


def get_chitchat_llm() -> ChatOpenAI:
    global _chitchat_llm
    api_key = _require_api_key()
    if _chitchat_llm is None:
        logger.info("Initializing ChatOpenAI LLM for chitchat.")
        _chitchat_llm = ChatOpenAI(
            model=settings.CHAT_MODEL_NAME,
            temperature=0.7,  # more open-ended
            api_key=api_key,
            base_url=settings.OPENAI_API_BASE or None,
        )
    return _chitchat_llm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_history_for_rewrite(history: List[Message]) -> str:
    """
    Turn chat history into a simple text transcript for question rewriting.
    """
    lines: List[str] = []
    for msg in history:
        prefix = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{prefix}: {msg.content}")
    return "\n".join(lines)


def _format_history_for_chitchat(history: List[Message]) -> str:
    """
    Turn chat history into a transcript for chitchat.
    """
    return _format_history_for_rewrite(history)


async def _rewrite_question_if_needed(
    history: List[Message],
    question: str,
) -> str:
    """
    If there is history, rewrite latest question into a standalone question.
    """
    if not history:
        return question

    llm = get_rewrite_llm()
    chat_history_text = _format_history_for_rewrite(history)

    prompt = CONDENSE_QUESTION_PROMPT.format(
        chat_history=chat_history_text,
        question=question,
    )

    logger.info("Rewriting question based on chat history.")
    try:
        response = await llm.ainvoke(prompt)  # type: ignore[arg-type]
        standalone = getattr(response, "content", "") if response is not None else ""
        standalone = standalone.strip() or question
        logger.debug("Standalone question: %r", standalone)
        return standalone
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error while rewriting question: %s", exc)
        return question


async def _generate_answer_from_context(
    standalone_question: str,
    context_text: str,
    tone: str,
    style: str,
) -> str:
    """
    Generate an answer using the RAG ANSWER_PROMPT and context.
    """
    llm = get_answer_llm()

    prompt = ANSWER_PROMPT.format(
        context=context_text or "No relevant context found.",
        question=standalone_question,
        tone=tone,
        style=style,
    )

    logger.info("Generating RAG answer.")
    response = await llm.ainvoke(prompt)  # type: ignore[arg-type]
    answer = getattr(response, "content", "") if response is not None else ""
    return answer.strip()


async def _self_check_answer(
    question: str,
    context_text: str,
    draft_answer: str,
) -> str:
    """
    Optionally refine an answer using a self-check prompt.
    """
    if not context_text:
        return draft_answer

    llm = get_self_check_llm()
    prompt = SELF_CHECK_PROMPT.format(
        question=question,
        context=context_text,
        draft_answer=draft_answer,
    )

    logger.info("Running self-check on draft answer.")
    try:
        response = await llm.ainvoke(prompt)  # type: ignore[arg-type]
        refined = getattr(response, "content", "") if response is not None else ""
        refined = refined.strip()
        return refined or draft_answer
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error during self-check: %s", exc)
        return draft_answer


def _build_sources_from_docs(docs) -> List[Source]:
    """
    Convert retrieved Documents to API Source objects.
    """
    sources: List[Source] = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        source_id = str(meta.get("id") or meta.get("chunk_id") or meta.get("source") or f"chunk-{idx}")
        title = meta.get("title") or meta.get("filename")
        snippet = (doc.page_content or "")[:300].strip()
        sources.append(
            Source(
                id=source_id,
                title=title,
                snippet=snippet,
                metadata=meta,
            )
        )
    return sources


# ---------------------------------------------------------------------------
# Public pipelines
# ---------------------------------------------------------------------------


async def run_rag_pipeline(
    question: str,
    history: List[Message],
    tone: str,
    style: str,
    top_k: int,
    namespace: Optional[str] = None,
    use_self_check: bool = False,
    conversation_id: Optional[str] = None,
) -> ChatResponse:
    """
    Full RAG pipeline:
    1) rewrite question (optional)
    2) retrieve docs
    3) answer from context
    4) optional self-check
    5) suggested next questions
    """
    convo_id = conversation_id or str(uuid.uuid4())

    logger.info("Running RAG pipeline (top_k=%d, namespace=%r).", top_k, namespace)

    standalone_question = await _rewrite_question_if_needed(history, question)

    docs = await retrieve_docs(
        query=standalone_question,
        top_k=top_k,
        namespace=namespace,
    )

    context_text = docs_to_context(docs)
    draft_answer = await _generate_answer_from_context(
        standalone_question=standalone_question,
        context_text=context_text,
        tone=tone,
        style=style,
    )

    if use_self_check:
        answer = await _self_check_answer(
            question=standalone_question,
            context_text=context_text,
            draft_answer=draft_answer,
        )
    else:
        answer = draft_answer

    suggestions = await generate_suggested_questions(
        question=question,
        answer=answer,
        context=context_text,
    )

    sources = _build_sources_from_docs(docs)

    return ChatResponse(
        conversation_id=convo_id,
        trace=None,
        answer=answer,
        sources=sources,
        suggested_questions=suggestions,
        intent="RAG_QA",
        answer_audio_b64=None,
    )


async def run_chitchat_pipeline(
    question: str,
    history: List[Message],
    tone: str,
    style: str,
    conversation_id: Optional[str] = None,
) -> ChatResponse:
    """
    Chitchat pipeline (no retrieval).
    """
    convo_id = conversation_id or str(uuid.uuid4())

    llm = get_chitchat_llm()
    history_text = _format_history_for_chitchat(history)

    full_prompt = f"""{CHITCHAT_SYSTEM_PROMPT}

Conversation so far:
{history_text}

User message:
{question}

Assistant reply (tone={tone}, style={style}):
"""

    logger.info("Running chitchat pipeline.")
    response = await llm.ainvoke(full_prompt)  # type: ignore[arg-type]
    answer = getattr(response, "content", "") if response is not None else ""
    answer = answer.strip()

    return ChatResponse(
        conversation_id=convo_id,
        trace=None,
        answer=answer,
        sources=[],
        suggested_questions=[],
        intent="CHITCHAT",
        answer_audio_b64=None,
    )


# ---------------------------------------------------------------------------
# OPTIONAL: Tool-friendly entrypoint for rag_tools.py
# ---------------------------------------------------------------------------

async def answer_with_rag(
    question: str,
    history: Optional[List[Dict[str, Any]]] = None,
    tone: Optional[str] = None,
    style: Optional[str] = None,
    top_k: Optional[int] = None,
    namespace: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Tool-friendly "retrieve + answer" endpoint.

    Returns stable JSON:
      {
        "answer": str,
        "sources": [ {id,title,snippet,metadata}, ... ],
        "suggested_questions": [str, ...]
      }

    Notes:
    - filters is accepted for future compatibility; legacy retrieve_docs doesn't use it.
      If you add metadata filtering later, wire it through.
    """
    # Convert history dicts to Message models (best-effort)
    msg_history: List[Message] = []
    for m in (history or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role and content is not None:
            msg_history.append(Message(role=role, content=str(content)))

    k = int(top_k or settings.DEFAULT_TOP_K)
    if k > settings.MAX_TOP_K:
        k = int(settings.MAX_TOP_K)

    resp = await run_rag_pipeline(
        question=question,
        history=msg_history,
        tone=tone or "neutral",
        style=style or "concise",
        top_k=k,
        namespace=namespace,
        use_self_check=False,
        conversation_id=str(uuid.uuid4()),
    )

    # Convert Source models to dict
    sources = [
        {
            "id": s.id,
            "title": s.title,
            "snippet": s.snippet,
            "metadata": s.metadata,
        }
        for s in resp.sources
    ]

    return {
        "answer": resp.answer,
        "sources": sources,
        "suggested_questions": resp.suggested_questions,
        "intent": resp.intent,
    }
