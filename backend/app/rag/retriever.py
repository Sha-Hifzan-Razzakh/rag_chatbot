from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, TypedDict

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.rag.embeddings import get_vector_store

logger = logging.getLogger(__name__)


# -----------------------------
# Tool-friendly result schema
# -----------------------------

class SearchDocResult(TypedDict, total=False):
    id: str
    title: Optional[str]
    snippet: Optional[str]
    metadata: Dict[str, Any]
    score: Optional[float]


# -----------------------------
# Chunking helpers
# -----------------------------

def _get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter for chunking documents.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=int(getattr(settings, "DEFAULT_CHUNK_SIZE", 1000)),
        chunk_overlap=int(getattr(settings, "DEFAULT_CHUNK_OVERLAP", 150)),
        separators=["\n\n", "\n", " ", ""],
    )


def chunk_text(
    text: str,
    base_metadata: Optional[Dict[str, Any]] = None,
    namespace: Optional[str] = None,
) -> List[Document]:
    """
    Split raw text into LangChain Document chunks with metadata attached.
    """
    if base_metadata is None:
        base_metadata = {}

    metadata = dict(base_metadata)
    if namespace is not None:
        metadata["namespace"] = namespace

    splitter = _get_text_splitter()
    chunks = splitter.split_text(text)

    docs = [
        Document(
            page_content=chunk,
            metadata=metadata,
        )
        for chunk in chunks
    ]

    logger.debug(
        "chunk_text: produced %d chunks (namespace=%r, base_metadata_keys=%s)",
        len(docs),
        namespace,
        list(base_metadata.keys()),
    )

    return docs


# -----------------------------
# Vector store writing
# -----------------------------

async def add_documents(
    docs: List[Document],
    namespace: Optional[str] = None,
) -> int:
    """
    Add a list of Document objects to the PGVector store.
    """
    if not docs:
        logger.warning("add_documents called with empty docs list.")
        return 0

    # Ensure namespace metadata is present on each document
    if namespace is not None:
        for doc in docs:
            doc.metadata = {**(doc.metadata or {}), "namespace": namespace}

    vector_store = get_vector_store()

    logger.info(
        "Adding %d documents to vector store (collection=%s, namespace=%r).",
        len(docs),
        settings.PGVECTOR_COLLECTION,
        namespace,
    )

    vector_store.add_documents(docs)
    return len(docs)


# -----------------------------
# Legacy retrieval (kept)
# -----------------------------

async def retrieve_docs(
    query: str,
    top_k: int,
    namespace: Optional[str] = None,
) -> List[Document]:
    """
    Retrieve the most relevant documents from the vector store for a query.
    (Legacy pipeline helper.)
    """
    vector_store = get_vector_store()

    search_filter: Dict[str, Any] = {}
    if namespace is not None:
        search_filter["namespace"] = namespace

    logger.info(
        "Retrieving documents for query=%r (top_k=%d, namespace=%r).",
        query,
        top_k,
        namespace,
    )

    docs: List[Document] = vector_store.similarity_search(
        query=query,
        k=top_k,
        filter=search_filter or None,
    )

    logger.debug("Retrieved %d documents for query=%r.", len(docs), query)
    return docs


def docs_to_context(docs: List[Document]) -> str:
    """
    Convert a list of Documents into a single context string.
    (Legacy pipeline helper.)
    """
    if not docs:
        return ""

    parts: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        meta_str = ", ".join(
            f"{k}={v}" for k, v in (doc.metadata or {}).items() if v is not None
        )
        header = f"[Source {idx}] {meta_str}".strip()
        if header:
            parts.append(header)
        parts.append(doc.page_content)
        parts.append("")

    return "\n".join(parts)


# -----------------------------
# NEW: Tool-friendly search API
# -----------------------------

async def search_docs(
    query: str,
    top_k: Optional[int] = None,
    namespace: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    truncate_chars: int = 500,
) -> Dict[str, Any]:
    """
    Tool-friendly retrieval that returns stable JSON.

    Returns:
      {
        "query": "...",
        "top_k": int,
        "namespace": str|None,
        "results": [ {id,title,snippet,metadata,score}, ... ]
      }
    """
    vector_store = get_vector_store()

    k = int(top_k or settings.DEFAULT_TOP_K)
    if k > settings.MAX_TOP_K:
        k = int(settings.MAX_TOP_K)

    search_filter: Dict[str, Any] = {}
    if namespace is not None:
        search_filter["namespace"] = namespace
    if filters:
        # merge without clobbering namespace unless explicitly provided
        search_filter.update({k: v for k, v in filters.items() if v is not None})

    logger.info(
        "search_docs: query=%r (k=%d, namespace=%r, filters_keys=%s)",
        query,
        k,
        namespace,
        list((filters or {}).keys()),
    )

    # Prefer similarity_search_with_score if available (gives scores)
    results: List[SearchDocResult] = []

    if hasattr(vector_store, "similarity_search_with_score"):
        pairs = vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=search_filter or None,
        )
        for doc, score in pairs:
            results.append(_doc_to_result(doc, score=score, truncate_chars=truncate_chars))
    else:
        docs: List[Document] = vector_store.similarity_search(
            query=query,
            k=k,
            filter=search_filter or None,
        )
        for doc in docs:
            results.append(_doc_to_result(doc, score=None, truncate_chars=truncate_chars))

    return {
        "query": query,
        "top_k": k,
        "namespace": namespace,
        "results": results,
    }


def _doc_to_result(doc: Document, *, score: Optional[float], truncate_chars: int) -> SearchDocResult:
    md = dict(doc.metadata or {})

    # best-effort id: prefer chunk_id/doc_id fields if present
    _id = md.get("chunk_id") or md.get("id") or md.get("document_id") or md.get("source_id") or str(uuid.uuid4())
    title = md.get("title") or md.get("filename") or md.get("file_name")

    text = doc.page_content or ""
    snippet = text if len(text) <= truncate_chars else (text[: max(0, truncate_chars - 1)] + "…")

    out: SearchDocResult = {
        "id": str(_id),
        "title": str(title) if title is not None else None,
        "snippet": snippet,
        "metadata": md,
        "score": float(score) if isinstance(score, (int, float)) else None,
    }
    return out
