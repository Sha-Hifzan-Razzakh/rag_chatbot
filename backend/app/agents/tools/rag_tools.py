"""
RAG tool wrappers for the agent orchestrator.

Tools provided (A-version):
- search_docs: retrieve relevant document chunks/snippets from the vector store
- answer_with_rag: (optional) end-to-end "retrieve + answer" if your rag/pipeline exposes it

This file is intentionally defensive about imports/signatures, since your rag layer
may evolve. Prefer passing dependencies through ToolContext["rag"] from orchestrator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.agents.tools.registry import ToolContext, ToolRegistry

JsonDict = Dict[str, Any]


# -----------------------------
# Public registration
# -----------------------------


def register_rag_tools(registry: ToolRegistry) -> None:
    """
    Register RAG tools into the ToolRegistry.

    Call from your chat endpoint wiring (or a central tool loader).
    """

    registry.register(
        name="search_docs",
        description=(
            "Search the document knowledge base for relevant snippets. "
            "Returns a list of results with ids, titles, snippets, metadata, and optional scores."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "top_k": {"type": "integer", "description": "Number of results to return.", "minimum": 1},
                "namespace": {"type": "string", "description": "Optional namespace/tenant identifier."},
                "filters": {
                    "type": "object",
                    "description": "Optional metadata filters (implementation-dependent).",
                    "additionalProperties": True,
                },
                "truncate_chars": {
                    "type": "integer",
                    "description": "Max characters for each snippet (truncate if longer). Default 500.",
                    "minimum": 50,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=_tool_search_docs,
    )

    # Optional: only register if pipeline exists / is importable
    if _pipeline_available():
        registry.register(
            name="answer_with_rag",
            description=(
                "Answer a question using RAG end-to-end (retrieve + generate). "
                "Returns answer plus sources if supported by your pipeline."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "User question to answer."},
                    "history": {
                        "type": "array",
                        "description": "Optional prior chat messages.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                    },
                    "tone": {"type": "string", "description": "Optional tone hint."},
                    "style": {"type": "string", "description": "Optional style hint."},
                    "top_k": {"type": "integer", "description": "Number of docs to retrieve.", "minimum": 1},
                    "namespace": {"type": "string", "description": "Optional namespace/tenant identifier."},
                    "filters": {
                        "type": "object",
                        "description": "Optional metadata filters (implementation-dependent).",
                        "additionalProperties": True,
                    },
                },
                "required": ["question"],
                "additionalProperties": False,
            },
            handler=_tool_answer_with_rag,
        )


# -----------------------------
# Tool handlers
# -----------------------------


async def _tool_search_docs(args: JsonDict, context: ToolContext) -> JsonDict:
    query: str = str(args.get("query", "")).strip()
    if not query:
        return {"query": query, "results": [], "error": "query must be a non-empty string"}

    top_k: Optional[int] = args.get("top_k")
    namespace: Optional[str] = args.get("namespace")
    filters: Optional[dict] = args.get("filters")
    truncate_chars: int = int(args.get("truncate_chars") or 500)

    retriever = _get_retriever(context)

    # Call patterns we try (in order):
    # 1) retriever.search_docs(query=..., top_k=..., namespace=..., filters=...)
    # 2) retriever.search(query=..., top_k=..., namespace=..., filters=...)
    # 3) module-level app.rag.retriever.search_docs(...)
    # 4) module-level app.rag.retriever.search(...)
    raw = await _call_retriever(
        retriever=retriever,
        query=query,
        top_k=top_k,
        namespace=namespace,
        filters=filters,
    )

    results = _normalize_search_results(raw, truncate_chars=truncate_chars)
    return {
        "query": query,
        "top_k": top_k,
        "namespace": namespace,
        "results": results,
    }


async def _tool_answer_with_rag(args: JsonDict, context: ToolContext) -> JsonDict:
    question: str = str(args.get("question", "")).strip()
    if not question:
        return {"question": question, "error": "question must be a non-empty string"}

    history = args.get("history") or []
    tone: Optional[str] = args.get("tone")
    style: Optional[str] = args.get("style")
    top_k: Optional[int] = args.get("top_k")
    namespace: Optional[str] = args.get("namespace")
    filters: Optional[dict] = args.get("filters")

    pipeline = _get_pipeline(context)

    # Call patterns we try (in order):
    # 1) pipeline.answer_with_rag(...)
    # 2) pipeline.run(...)
    # 3) module-level app.rag.pipeline.answer_with_rag(...)
    # 4) module-level app.rag.pipeline.run(...)
    raw = await _call_pipeline(
        pipeline=pipeline,
        question=question,
        history=history,
        tone=tone,
        style=style,
        top_k=top_k,
        namespace=namespace,
        filters=filters,
    )

    # We don't assume exact pipeline output shape. Try to normalize.
    answer, sources = _extract_answer_and_sources(raw)
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "raw": raw if isinstance(raw, (dict, list)) else {"value": raw},
    }


# -----------------------------
# Dependency resolution
# -----------------------------


def _get_retriever(context: ToolContext) -> Any:
    rag = context.get("rag")
    if isinstance(rag, dict) and rag.get("retriever") is not None:
        return rag["retriever"]
    return None


def _get_pipeline(context: ToolContext) -> Any:
    rag = context.get("rag")
    if isinstance(rag, dict) and rag.get("pipeline") is not None:
        return rag["pipeline"]
    return None


def _pipeline_available() -> bool:
    try:
        from app.rag import pipeline as _p  # noqa: F401

        return True
    except Exception:
        return False


# -----------------------------
# Calling helpers (defensive)
# -----------------------------


async def _call_retriever(
    retriever: Any,
    *,
    query: str,
    top_k: Optional[int],
    namespace: Optional[str],
    filters: Optional[dict],
) -> Any:
    # Object methods first
    if retriever is not None:
        if hasattr(retriever, "search_docs"):
            fn = getattr(retriever, "search_docs")
            return await _maybe_await(fn, query=query, top_k=top_k, namespace=namespace, filters=filters)
        if hasattr(retriever, "search"):
            fn = getattr(retriever, "search")
            return await _maybe_await(fn, query=query, top_k=top_k, namespace=namespace, filters=filters)

    # Fallback to module-level functions
    from app.rag import retriever as retriever_mod

    if hasattr(retriever_mod, "search_docs"):
        fn = getattr(retriever_mod, "search_docs")
        return await _maybe_await(fn, query=query, top_k=top_k, namespace=namespace, filters=filters)

    if hasattr(retriever_mod, "search"):
        fn = getattr(retriever_mod, "search")
        return await _maybe_await(fn, query=query, top_k=top_k, namespace=namespace, filters=filters)

    raise RuntimeError("No compatible retriever function found (expected search_docs/search).")


async def _call_pipeline(
    pipeline: Any,
    *,
    question: str,
    history: list,
    tone: Optional[str],
    style: Optional[str],
    top_k: Optional[int],
    namespace: Optional[str],
    filters: Optional[dict],
) -> Any:
    # Object methods first
    if pipeline is not None:
        if hasattr(pipeline, "answer_with_rag"):
            fn = getattr(pipeline, "answer_with_rag")
            return await _maybe_await(
                fn,
                question=question,
                history=history,
                tone=tone,
                style=style,
                top_k=top_k,
                namespace=namespace,
                filters=filters,
            )
        if hasattr(pipeline, "run"):
            fn = getattr(pipeline, "run")
            return await _maybe_await(
                fn,
                question=question,
                history=history,
                tone=tone,
                style=style,
                top_k=top_k,
                namespace=namespace,
                filters=filters,
            )

    # Fallback to module-level functions
    from app.rag import pipeline as pipeline_mod

    if hasattr(pipeline_mod, "answer_with_rag"):
        fn = getattr(pipeline_mod, "answer_with_rag")
        return await _maybe_await(
            fn,
            question=question,
            history=history,
            tone=tone,
            style=style,
            top_k=top_k,
            namespace=namespace,
            filters=filters,
        )

    if hasattr(pipeline_mod, "run"):
        fn = getattr(pipeline_mod, "run")
        return await _maybe_await(
            fn,
            question=question,
            history=history,
            tone=tone,
            style=style,
            top_k=top_k,
            namespace=namespace,
            filters=filters,
        )

    raise RuntimeError("No compatible pipeline function found (expected answer_with_rag/run).")


async def _maybe_await(fn: Any, **kwargs: Any) -> Any:
    out = fn(**{k: v for k, v in kwargs.items() if v is not None})
    # coroutine?
    if hasattr(out, "__await__"):
        return await out
    return out


# -----------------------------
# Normalization helpers
# -----------------------------


def _normalize_search_results(raw: Any, *, truncate_chars: int) -> List[JsonDict]:
    """
    Produce stable tool-friendly JSON.

    Output shape:
      [
        {
          "id": str,
          "title": str | None,
          "snippet": str | None,
          "metadata": dict,
          "score": float | None
        },
        ...
      ]
    """
    items = _coerce_list(raw)

    normalized: List[JsonDict] = []
    for item in items:
        if isinstance(item, dict):
            _id = item.get("id") or item.get("chunk_id") or item.get("document_id") or item.get("source_id")
            title = item.get("title")
            snippet = item.get("snippet") or item.get("text") or item.get("content")
            metadata = item.get("metadata") or {}
            score = item.get("score") or item.get("similarity") or item.get("distance")
        else:
            # Unknown object - try attribute access
            _id = getattr(item, "id", None) or getattr(item, "chunk_id", None)
            title = getattr(item, "title", None)
            snippet = getattr(item, "snippet", None) or getattr(item, "text", None) or getattr(item, "content", None)
            metadata = getattr(item, "metadata", None) or {}
            score = getattr(item, "score", None)

        snippet = _truncate(snippet, truncate_chars)

        score_f: Optional[float] = None
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            score_f = float(score)

        normalized.append(
            {
                "id": str(_id) if _id is not None else "",
                "title": str(title) if title is not None else None,
                "snippet": snippet,
                "metadata": metadata if isinstance(metadata, dict) else {},
                "score": score_f,
            }
        )


    return normalized


def _extract_answer_and_sources(raw: Any) -> Tuple[str, List[JsonDict]]:
    """
    Best-effort extraction from whatever your pipeline returns.
    """
    if isinstance(raw, dict):
        answer = raw.get("answer") or raw.get("final") or raw.get("text") or ""
        sources = raw.get("sources") or raw.get("context") or []
        if isinstance(sources, list):
            src_out: List[JsonDict] = []
            for s in sources:
                if isinstance(s, dict):
                    src_out.append(s)
                else:
                    src_out.append({"value": str(s)})
            return str(answer), src_out
        return str(answer), [{"value": str(sources)}]

    return str(raw), []


def _coerce_list(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        for key in ("results", "items", "documents", "matches", "chunks"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        return [raw]

    return [raw]


def _truncate(text: Any, limit: int) -> Optional[str]:
    if text is None:
        return None
    s = str(text)
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)
