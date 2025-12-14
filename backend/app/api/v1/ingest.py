from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.ingestion.chunking import chunk_pdf_pages, chunk_text as ingest_chunk_text
from app.ingestion.extractors import extract_document
from app.ingestion.file_store import delete_file, save_upload_file
from app.ingestion.preprocess import normalize_text
from app.ingestion.validators import validate_upload
from app.models.schemas import (
    FileIngestResult,
    IngestFilesResponse,
    IngestResponse,
    IngestTextRequest,
)
from app.rag.retriever import add_documents, chunk_text as rag_chunk_text

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest")


@router.post(
    "/text",
    response_model=IngestResponse,
    summary="Ingest raw text documents",
    response_description="Number of documents and chunks ingested.",
)
async def ingest_text_endpoint(payload: IngestTextRequest) -> IngestResponse:
    logger.info("Ingesting %d text documents.", len(payload.texts))

    try:
        all_docs = []
        for text in payload.texts:
            docs = rag_chunk_text(
                text=text,
                base_metadata=payload.metadata,
                namespace=payload.namespace,
            )
            all_docs.extend(docs)

        num_chunks = await add_documents(all_docs, namespace=payload.namespace)

        return IngestResponse(
            status="ok",
            num_documents=len(payload.texts),
            num_chunks=num_chunks,
            namespace=payload.namespace,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error during text ingestion: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while ingesting text documents.",
        ) from exc


@router.post(
    "/file",
    response_model=IngestFilesResponse,
    summary="Ingest uploaded files (PDF/DOCX/TXT/MD + optional OCR for PDFs)",
    response_description="Per-file results and total chunks ingested from uploaded files.",
)
async def ingest_file_endpoint(
    files: List[UploadFile] = File(...),
    namespace: Optional[str] = Form(None),
    enable_ocr: bool = Form(False),
    ocr_language: str = Form(settings.OCR_DEFAULT_LANGUAGE),
    ocr_dpi: int = Form(settings.OCR_DEFAULT_DPI),
    ocr_min_chars: int = Form(settings.OCR_MIN_CHARS),
    chunk_size: int = Form(settings.DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(settings.DEFAULT_CHUNK_OVERLAP),
    max_pages: Optional[int] = Form(None),
) -> IngestFilesResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    logger.info(
        "Ingesting %d files. namespace=%s enable_ocr=%s chunk_size=%d overlap=%d",
        len(files),
        namespace,
        enable_ocr,
        chunk_size,
        chunk_overlap,
    )

    results: List[FileIngestResult] = []
    all_docs = []
    ingested_docs_count = 0

    try:
        for upload in files:
            filename = upload.filename or "upload"
            content_type = getattr(upload, "content_type", None)

            # 1) Validate
            vr = validate_upload(upload)
            if not vr.ok:
                results.append(
                    FileIngestResult(
                        filename=filename,
                        content_type=content_type,
                        status="skipped",
                        reason=vr.error,
                        num_chunks=0,
                    )
                )
                continue

            # 2) Save (enforces max upload size while streaming)
            try:
                saved = await save_upload_file(
                    upload,
                    dest_dir=settings.UPLOAD_TMP_DIR,
                    max_bytes=int(settings.MAX_UPLOAD_BYTES or (settings.MAX_UPLOAD_MB * 1024 * 1024)),
                )
            except ValueError as e:
                results.append(
                    FileIngestResult(
                        filename=filename,
                        content_type=content_type,
                        status="error",
                        reason=str(e),
                        num_chunks=0,
                    )
                )
                continue
            except Exception as e:  # noqa: BLE001
                results.append(
                    FileIngestResult(
                        filename=filename,
                        content_type=content_type,
                        status="error",
                        reason=f"Failed to save upload: {e}",
                        num_chunks=0,
                    )
                )
                continue

            try:
                # 3) Extract
                extraction = extract_document(
                    saved.saved_path,
                    original_filename=saved.original_filename,
                    content_type=saved.content_type,
                    enable_ocr=enable_ocr,
                    ocr_language=ocr_language,
                    ocr_dpi=ocr_dpi,
                    ocr_min_chars=ocr_min_chars,
                    max_pages=max_pages,
                )

                # 4) Preprocess
                extracted_text = normalize_text(extraction.text, collapse_blank_lines=True)

                if not extracted_text.strip():
                    results.append(
                        FileIngestResult(
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
                    )
                    continue

                # 5) Metadata for chunks
                base_metadata = dict(extraction.doc_metadata)
                base_metadata.update(
                    {
                        "original_filename": saved.original_filename,
                        "content_type": saved.content_type,
                        "size_bytes": saved.size_bytes,
                    }
                )
                if namespace:
                    base_metadata["namespace"] = namespace

                # 6) Chunk
                if extraction.pages:
                    docs = chunk_pdf_pages(
                        extraction.pages,
                        base_metadata=base_metadata,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                else:
                    docs = ingest_chunk_text(
                        extracted_text,
                        base_metadata=base_metadata,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                if not docs:
                    results.append(
                        FileIngestResult(
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
                    )
                    continue

                all_docs.extend(docs)
                ingested_docs_count += 1

                results.append(
                    FileIngestResult(
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
                )

            except Exception as e:  # noqa: BLE001
                logger.exception("Extraction/chunking failed for '%s': %s", filename, e)
                results.append(
                    FileIngestResult(
                        filename=filename,
                        content_type=content_type,
                        size_bytes=getattr(saved, "size_bytes", None),
                        status="error",
                        reason=str(e),
                        num_chunks=0,
                    )
                )
            finally:
                # 7) Cleanup temp file
                try:
                    delete_file(saved.saved_path)
                except Exception:
                    pass

        if not all_docs:
            return IngestFilesResponse(
                status="ok",
                num_documents=0,
                num_chunks=0,
                namespace=namespace,
                results=results,
            )

        # 8) Persist to vector store
        num_chunks_total = await add_documents(all_docs, namespace=namespace)

        return IngestFilesResponse(
            status="ok",
            num_documents=ingested_docs_count,
            num_chunks=num_chunks_total,
            namespace=namespace,
            results=results,
        )

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error during file ingestion: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while ingesting files.",
        ) from exc
