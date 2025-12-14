from __future__ import annotations

import re
from typing import Optional


def normalize_text(
    text: str,
    *,
    remove_null_bytes: bool = True,
    collapse_whitespace: bool = False,
    collapse_blank_lines: bool = True,
    strip: bool = True,
) -> str:
    """
    Basic cleanup for extracted text.

    - remove_null_bytes: removes \\x00 which sometimes appears in PDF extraction/OCR
    - collapse_whitespace: converts all whitespace (incl newlines) into single spaces
      (usually NOT desired for RAG unless your docs are very messy)
    - collapse_blank_lines: reduces multiple blank lines to max 2 newlines
    """
    if not text:
        return ""

    out = text

    if remove_null_bytes:
        out = out.replace("\x00", " ")

    # Normalize line endings
    out = out.replace("\r\n", "\n").replace("\r", "\n")

    if collapse_blank_lines:
        # 3+ newlines -> 2 newlines
        out = re.sub(r"\n{3,}", "\n\n", out)

    if collapse_whitespace:
        out = re.sub(r"\s+", " ", out)

    if strip:
        out = out.strip()

    return out


def maybe_trim(
    text: str,
    *,
    max_chars: Optional[int] = None,
) -> str:
    """
    Optional safety guard: trim very large extracted text before chunking.
    (You can prefer max_pages for PDFs instead; this is a backup.)
    """
    if not text:
        return ""
    if max_chars is None or max_chars <= 0:
        return text
    return text[:max_chars]
