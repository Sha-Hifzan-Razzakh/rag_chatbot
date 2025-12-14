from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pypdf import PdfReader

from ..ocr import ocr_pdf

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    """
    Normalize extracted PDF text:
    - remove null bytes
    - normalize whitespace
    - trim
    """
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _should_ocr(extracted_text: str, min_chars: int) -> bool:
    return len((extracted_text or "").strip()) < min_chars


def load_pdf(
    file_path: Union[str, Path],
    *,
    max_pages: Optional[int] = None,
    clean: bool = True,
    enable_ocr: bool = False,
    ocr_language: str = "eng",
    ocr_dpi: int = 200,
    ocr_min_chars: int = 50,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extract text from a PDF file, with optional OCR fallback.

    Returns:
        full_text: str
        doc_metadata: dict
        pages: list[dict] where each item contains:
            - page_number (1-based)
            - text
            - char_count
            - method: "text" | "ocr"

    Notes:
      - For scanned/image-only PDFs, `pypdf` often returns empty text.
        OCR fallback can recover text if enabled and system deps are installed.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF '{path.name}': {e}") from e

    # Handle encrypted PDFs (best-effort with empty password)
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")  # type: ignore[attr-defined]
        except Exception as e:
            raise RuntimeError(f"PDF is encrypted and cannot be decrypted: {path.name}") from e

    raw_meta = reader.metadata or {}
    num_pages = len(reader.pages)
    limit = min(num_pages, max_pages) if max_pages is not None else num_pages

    doc_metadata: Dict[str, Any] = {
        "source_path": str(path),
        "file_name": path.name,
        "num_pages": num_pages,
        "title": getattr(raw_meta, "title", None) if hasattr(raw_meta, "title") else raw_meta.get("/Title"),
        "author": getattr(raw_meta, "author", None) if hasattr(raw_meta, "author") else raw_meta.get("/Author"),
        "subject": getattr(raw_meta, "subject", None) if hasattr(raw_meta, "subject") else raw_meta.get("/Subject"),
        "creator": getattr(raw_meta, "creator", None) if hasattr(raw_meta, "creator") else raw_meta.get("/Creator"),
        "producer": getattr(raw_meta, "producer", None) if hasattr(raw_meta, "producer") else raw_meta.get("/Producer"),
        "creation_date": raw_meta.get("/CreationDate") if isinstance(raw_meta, dict) else None,
        "mod_date": raw_meta.get("/ModDate") if isinstance(raw_meta, dict) else None,
        "ocr_used": False,
    }

    pages: List[Dict[str, Any]] = []
    page_texts: List[str] = []

    # ---- Pass 1: native text extraction ----
    for i in range(limit):
        try:
            page = reader.pages[i]
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning("Failed extracting text from %s page %d: %s", path.name, i + 1, e)
            text = ""

        if clean:
            text = _clean_text(text)

        pages.append(
            {
                "page_number": i + 1,
                "text": text,
                "char_count": len(text),
                "method": "text",
            }
        )
        if text:
            page_texts.append(text)

    full_text = "\n\n".join(page_texts).strip()

    # ---- Pass 2: OCR fallback (only if enabled and needed) ----
    if enable_ocr and _should_ocr(full_text, ocr_min_chars):
        logger.info("Low/no extracted text in '%s' -> OCR fallback enabled", path.name)

        ocr_text, ocr_pages = ocr_pdf(
            path,
            max_pages=limit,
            language=ocr_language,
            dpi=ocr_dpi,
            clean=clean,
        )

        if ocr_text.strip():
            full_text = ocr_text
            pages = ocr_pages
            doc_metadata["ocr_used"] = True
            doc_metadata["ocr_language"] = ocr_language
            doc_metadata["ocr_dpi"] = ocr_dpi
        else:
            doc_metadata["ocr_used"] = True
            doc_metadata["ocr_warning"] = "OCR ran but produced no text"

    return full_text, doc_metadata, pages
