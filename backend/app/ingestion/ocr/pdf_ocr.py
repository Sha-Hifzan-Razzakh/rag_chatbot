from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ocr_pdf(
    pdf_path: Path,
    *,
    max_pages: int,
    language: str = "eng",
    dpi: int = 200,
    clean: bool = True,
) -> Tuple[str, List[Dict]]:
    """
    OCR a PDF by rendering pages to images (pdf2image) and extracting text (pytesseract).

    Returns:
      full_text: concatenated OCR text
      pages: list of {page_number, text, char_count, method="ocr"}
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception as e:
        raise RuntimeError(
            "OCR deps missing. Install: pytesseract, pdf2image, Pillow; "
            "and OS packages: tesseract-ocr, poppler-utils."
        ) from e

    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=1,
        last_page=max_pages,
    )

    pages: List[Dict] = []
    out_texts: List[str] = []

    for idx, img in enumerate(images):
        page_no = idx + 1
        try:
            text = pytesseract.image_to_string(img, lang=language) or ""
        except Exception as e:
            logger.warning("OCR failed for %s page %d: %s", pdf_path.name, page_no, e)
            text = ""

        if clean:
            text = _clean_text(text)

        pages.append(
            {"page_number": page_no, "text": text, "char_count": len(text), "method": "ocr"}
        )
        if text:
            out_texts.append(text)

    return "\n\n".join(out_texts).strip(), pages
