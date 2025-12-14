from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

from fastapi import UploadFile


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    error: Optional[str] = None


DEFAULT_ALLOWED_EXTENSIONS: Set[str] = {".pdf", ".txt", ".md", ".docx"}

# MIME types are helpful but not always reliable (browsers/clients vary)
DEFAULT_ALLOWED_MIME_TYPES: Set[str] = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _normalize_ext(ext: str) -> str:
    ext = ext.strip().lower()
    return ext if ext.startswith(".") else f".{ext}"


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_upload(
    upload: UploadFile,
    *,
    allowed_extensions: Iterable[str] = DEFAULT_ALLOWED_EXTENSIONS,
    allowed_mime_types: Iterable[str] = DEFAULT_ALLOWED_MIME_TYPES,
) -> ValidationResult:
    """
    Validate filename + content_type for an UploadFile.

    Size is enforced during streaming save (see file_store.save_upload_file),
    because UploadFile doesn't reliably expose full size beforehand.
    """
    filename = (upload.filename or "").strip()
    if not filename:
        return ValidationResult(ok=False, error="Missing filename.")

    ext = get_extension(filename)
    allowed_exts = {_normalize_ext(e) for e in allowed_extensions}
    if ext not in allowed_exts:
        return ValidationResult(
            ok=False,
            error=f"Unsupported file extension '{ext}'. Allowed: {sorted(allowed_exts)}",
        )

    content_type = (getattr(upload, "content_type", None) or "").strip().lower()
    allowed_mimes = {m.strip().lower() for m in allowed_mime_types}

    # If client provides a content_type, validate it; if missing, rely on extension.
    if content_type and content_type not in allowed_mimes:
        return ValidationResult(
            ok=False,
            error=f"Unsupported content type '{content_type}'. Allowed: {sorted(allowed_mimes)}",
        )

    return ValidationResult(ok=True)


def split_allowed(
    allowed_extensions: Iterable[str],
    allowed_mime_types: Iterable[str],
) -> Tuple[Set[str], Set[str]]:
    """
    Utility for converting env/settings lists into normalized sets.
    """
    exts = {_normalize_ext(e) for e in allowed_extensions}
    mimes = {m.strip().lower() for m in allowed_mime_types}
    return exts, mimes
