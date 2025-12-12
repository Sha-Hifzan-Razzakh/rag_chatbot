from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.rag.embeddings import get_vector_store

logger = logging.getLogger(__name__)


def _get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter for chunking documents.

    You can later move these parameters to config if needed.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )


def chunk_text(
    text: str,
    base_metadata: Optional[Dict[str, Any]] = None,
    namespace: Optional[str] = None,
) -> List[Document]:
    """
    Split raw text into LangChain Document chunks with metadata attached.

    :param text: Raw text to chunk.
    :param base_metadata: Metadata to apply to each chunk (e.g., filename, source).
    :param namespace: Optional namespace / knowledge base identifier.
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


async def add_documents(
    docs: List[Document],
    namespace: Optional[str] = None,
) -> int:
    """
    Add a list of Document objects to the PGVector store.

    :param docs: List of LangChain Document objects.
    :param namespace: Optional namespace (stored as metadata on each doc).
    :return: Number of chunks actually added.
    """
    if not docs:
        logger.warning("add_documents called with empty docs list.")
        return 0

    # Ensure namespace metadata is present on each document
    if namespace is not None:
        for doc in docs:
            doc.metadata = {**doc.metadata, "namespace": namespace}

    vector_store = get_vector_store()

    logger.info(
        "Adding %d documents to vector store (collection=%s, namespace=%r).",
        len(docs),
        settings.PGVECTOR_COLLECTION,
        namespace,
    )

    # PGVector from langchain-postgres uses .add_documents
    vector_store.add_documents(docs)

    return len(docs)


async def retrieve_docs(
    query: str,
    top_k: int,
    namespace: Optional[str] = None,
) -> List[Document]:
    """
    Retrieve the most relevant documents from the vector store for a query.

    :param query: User question (ideally standalone).
    :param top_k: Number of documents to retrieve.
    :param namespace: Optional namespace filter.
    """
    vector_store = get_vector_store()

    # Build search filter
    search_filter: Dict[str, Any] = {}
    if namespace is not None:
        search_filter["namespace"] = namespace

    logger.info(
        "Retrieving documents for query=%r (top_k=%d, namespace=%r).",
        query,
        top_k,
        namespace,
    )

    # similarity_search supports a 'filter' argument in most LC vector stores
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

    This is what we send into the ANSWER_PROMPT / LLM.
    """
    if not docs:
        return ""

    parts: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        meta_str = ", ".join(
            f"{k}={v}" for k, v in doc.metadata.items() if v is not None
        )
        header = f"[Source {idx}] {meta_str}".strip()
        if header:
            parts.append(header)
        parts.append(doc.page_content)
        parts.append("")  # blank line between docs

    return "\n".join(parts)
