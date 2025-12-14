from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.core.config import settings
from app.ingestion.chunking import chunk_pdf_pages, chunk_text as ingest_chunk_text
from app.ingestion.extractors import extract_document
from app.ingestion.file_store import delete_file, save_upload_file
from app.ingestion.preprocess import normalize_text
from app.models.schemas import FileIngestResult
from app.rag.retriever import add_documents
from fastapi import UploadFile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestFileParams:
    """
    Parameters controlling ingestion behavior.

    These mirror your /ingest/file Form fields, but can be used internally too.
    """
    namespace: Optional[str] = None

    # OCR controls (PDF only)
    enable_ocr: bool = False
    ocr_language: str = settings.OCR_DEFAULT_LANGUAGE
    ocr_dpi: int = settings.OCR_DEFAULT_DPI
    ocr_min_chars: int = settings.OCR_MIN_CHARS

    # Chunking controls
    chunk_size: int = settings.DEFAULT_CHUNK_SIZE
    chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP

    # PDF safety cap
    max_pages: Optional[int] = None


async def ingest_single_upload_file(
    upload: UploadFile,
    *,
    params: IngestFileParams,
    base_metadata: Optional[Dict[str, Any]] = None,
) -> FileIngestResult:
    """
    End-to-end ingestion for ONE UploadFile:
      save -> extract -> preprocess -> chunk -> add to vector db -> cleanup

    Returns a FileIngestResult describing the outcome for this file.

    NOTE:
      - This function performs the vector-store write (add_documents) itself.
      - If you prefer batch-writing for speed, use `prepare_documents_from_upload(...)`
        and call add_documents once for the whole batch (as your ingest.py does).
    """
    filename = upload.filename or "upload"
    content_type = getattr(upload, "content_type", None)

    try:
        saved = await save_upload_file(
            upload,
            dest_dir=settings.UPLOAD_TMP_DIR,
            max_bytes=int(settings.MAX_UPLOAD_BYTES or (settings.MAX_UPLOAD_MB * 1024 * 1024)),
        )
    except ValueError as e:
        return FileIngestResult(
            filename=filename,
            content_type=content_type,
            status="error",
            reason=str(e),
            num_chunks=0,
        )
    except Exception as e:  # noqa: BLE001
        return FileIngestResult(
            filename=filename,
            content_type=content_type,
            status="error",
            reason=f"Failed to save upload: {e}",
            num_chunks=0,
        )

    try:
        extraction = extract_document(
            saved.saved_path,
            original_filename=saved.original_filename,
            content_type=saved.content_type,
            enable_ocr=params.enable_ocr,
            ocr_language=params.ocr_language,
            ocr_dpi=params.ocr_dpi,
            ocr_min_chars=params.ocr_min_chars,
            max_pages=params.max_pages,
        )

        extracted_text = normalize_text(extraction.text, collapse_blank_lines=True)
        if not extracted_text.strip():
            return FileIngestResult(
                filename=saved.original_filename,
                content_type=saved.content_type,
                size_bytes=saved.size_bytes,
                status="skipped",
                reason="No extractable text found (possibly scanned PDF without OCR or empty doc).",
                num_pages=extraction.doc_metadata.get("num_pages"),
                ocr_used=extraction.doc_metadata.get("ocr_used"),
                num_chunks=0,
                char_count=0,
            )

        md: Dict[str, Any] = {}
        if base_metadata:
            md.update(base_metadata)
        md.update(dict(extraction.doc_metadata))
        md.update(
            {
                "original_filename": saved.original_filename,
                "content_type": saved.content_type,
                "size_bytes": saved.size_bytes,
            }
        )
        if params.namespace:
            md["namespace"] = params.namespace

        if extraction.pages:
            docs = chunk_pdf_pages(
                extraction.pages,
                base_metadata=md,
                chunk_size=params.chunk_size,
                chunk_overlap=params.chunk_overlap,
            )
        else:
            docs = ingest_chunk_text(
                extracted_text,
                base_metadata=md,
                chunk_size=params.chunk_size,
                chunk_overlap=params.chunk_overlap,
            )

        if not docs:
            return FileIngestResult(
                filename=saved.original_filename,
                content_type=saved.content_type,
                size_bytes=saved.size_bytes,
                status="skipped",
                reason="Chunking produced 0 chunks.",
                num_pages=extraction.doc_metadata.get("num_pages"),
                ocr_used=extraction.doc_metadata.get("ocr_used"),
                num_chunks=0,
                char_count=len(extracted_text),
            )

        # Persist immediately (single-file mode)
        await add_documents(docs, namespace=params.namespace)

        return FileIngestResult(
            filename=saved.original_filename,
            content_type=saved.content_type,
            size_bytes=saved.size_bytes,
            status="ok",
            reason=None,
            num_chunks=len(docs),
            char_count=len(extracted_text),
            num_pages=extraction.doc_metadata.get("num_pages"),
            ocr_used=extraction.doc_metadata.get("ocr_used"),
        )

    except Exception as e:  # noqa: BLE001
        logger.exception("Ingest failed for '%s': %s", filename, e)
        return FileIngestResult(
            filename=filename,
            content_type=content_type,
            status="error",
            reason=str(e),
            num_chunks=0,
        )
    finally:
        try:
            delete_file(saved.saved_path)
        except Exception:
            pass
