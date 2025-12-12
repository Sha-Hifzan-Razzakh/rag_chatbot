import logging
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import IngestResponse, IngestTextRequest
from app.rag.retriever import add_documents, chunk_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest")


@router.post(
    "/text",
    response_model=IngestResponse,
    summary="Ingest raw text documents",
    response_description="Number of documents and chunks ingested.",
)
async def ingest_text_endpoint(payload: IngestTextRequest) -> IngestResponse:
    """
    Ingest plain text documents into the vector store.

    Steps:
    1. Chunk each text into smaller pieces.
    2. Add chunks as documents to the vector store (with shared metadata).
    """
    logger.info("Ingesting %d text documents.", len(payload.texts))

    try:
        all_docs = []
        for text in payload.texts:
            docs = chunk_text(
                text=text,
                base_metadata=payload.metadata,
                namespace=payload.namespace,
            )
            all_docs.extend(docs)

        logger.debug("Created %d chunks from %d documents.", len(all_docs), len(payload.texts))

        num_chunks = await add_documents(all_docs, namespace=payload.namespace)

        return IngestResponse(
            status="ok",
            num_documents=len(payload.texts),
            num_chunks=num_chunks,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error during text ingestion: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while ingesting text documents.",
        ) from exc


@router.post(
    "/file",
    response_model=IngestResponse,
    summary="Ingest uploaded files",
    response_description="Number of documents and chunks ingested from uploaded files.",
)
async def ingest_file_endpoint(
    files: List[UploadFile] = File(...),
) -> IngestResponse:
    """
    Ingest uploaded files (basic version).

    Currently supports:
    - .txt / .md: read as UTF-8 text.

    PDF / DOCX parsing can be added later using libraries like `pypdf` or `python-docx`.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    logger.info("Ingesting %d files.", len(files))

    num_documents = 0
    all_docs = []

    try:
        for file in files:
            filename = file.filename or "unnamed"
            content_type = file.content_type or "application/octet-stream"

            logger.debug("Processing file: %s (content_type=%s)", filename, content_type)

            # Basic support: text/markdown and plain text files
            if filename.endswith((".txt", ".md")) or content_type.startswith("text/"):
                raw_bytes = await file.read()
                text = raw_bytes.decode("utf-8", errors="ignore")

                base_metadata = {
                    "filename": filename,
                    "content_type": content_type,
                }

                docs = chunk_text(text=text, base_metadata=base_metadata, namespace=None)
                all_docs.extend(docs)
                num_documents += 1

            else:
                # Placeholder for future PDF/DOCX support
                logger.warning(
                    "Unsupported file type for now (skipping): %s (content_type=%s)",
                    filename,
                    content_type,
                )
                continue

        if not all_docs:
            raise HTTPException(
                status_code=400,
                detail="No supported file types found to ingest.",
            )

        num_chunks = await add_documents(all_docs, namespace=None)

        return IngestResponse(
            status="ok",
            num_documents=num_documents,
            num_chunks=num_chunks,
        )

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error during file ingestion: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while ingesting files.",
        ) from exc
