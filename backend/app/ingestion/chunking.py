from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class ChunkingStats:
    num_chunks: int
    total_chars: int


def build_text_splitter(
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> RecursiveCharacterTextSplitter:
    """
    Default splitter for mixed doc types. Adjust in Settings as needed.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def chunk_text(
    text: str,
    *,
    base_metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Chunk plain text into LangChain Documents.
    """
    base_metadata = base_metadata or {}
    splitter = build_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents([text], metadatas=[base_metadata])
    return docs


def chunk_pdf_pages(
    pages: List[Dict[str, Any]],
    *,
    base_metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Chunk a PDF using per-page extraction results (better metadata + traceability).
    Each chunk carries page_number + method (text/ocr).
    """
    base_metadata = base_metadata or {}
    splitter = build_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    docs: List[Document] = []

    for page in pages:
        page_text = (page.get("text") or "").strip()
        if not page_text:
            continue

        page_no = page.get("page_number")
        method = page.get("method")

        md = dict(base_metadata)
        if page_no is not None:
            md["page_number"] = int(page_no)
        if method:
            md["extract_method"] = method

        # Split each page separately so page_number stays accurate per chunk
        page_docs = splitter.create_documents([page_text], metadatas=[md])
        docs.extend(page_docs)

    return docs


def get_chunking_stats(docs: List[Document]) -> ChunkingStats:
    total_chars = sum(len(d.page_content or "") for d in docs)
    return ChunkingStats(num_chunks=len(docs), total_chars=total_chars)
