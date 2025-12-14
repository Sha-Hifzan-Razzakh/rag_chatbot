from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import UploadFile


_FILENAME_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass(frozen=True)
class SavedFile:
    original_filename: str
    content_type: Optional[str]
    saved_path: Path
    size_bytes: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """
    Keep it filesystem-safe and predictable.
    """
    name = filename.strip().replace("\\", "/").split("/")[-1]  # strip any path parts
    name = _FILENAME_SAFE.sub("_", name)
    name = name.strip("._") or "upload"
    return name


def unique_path(dest_dir: Path, filename: str) -> Path:
    """
    Avoid collisions by adding a short UUID suffix.
    """
    safe = sanitize_filename(filename)
    stem = Path(safe).stem
    suffix = Path(safe).suffix
    return dest_dir / f"{stem}_{uuid4().hex[:10]}{suffix}"


async def save_upload_file(
    upload: UploadFile,
    *,
    dest_dir: Path,
    max_bytes: int,
    chunk_size: int = 1024 * 1024,  # 1MB
) -> SavedFile:
    """
    Save a FastAPI UploadFile to disk with a size limit.

    Args:
        upload: FastAPI UploadFile (multipart/form-data)
        dest_dir: directory to write to
        max_bytes: max allowed size in bytes
        chunk_size: streaming chunk size

    Returns:
        SavedFile metadata

    Raises:
        ValueError if size exceeds max_bytes
    """
    ensure_dir(dest_dir)

    out_path = unique_path(dest_dir, upload.filename or "upload")
    size = 0

    # NOTE: UploadFile.read() is async; we stream it to disk.
    try:
        with open(out_path, "wb") as f:
            while True:
                chunk = await upload.read(chunk_size)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"File too large (>{max_bytes} bytes).")
                f.write(chunk)
    except Exception:
        # best-effort cleanup on error
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        raise
    finally:
        # reset internal file pointer not needed after save; close upload handle
        try:
            await upload.close()
        except Exception:
            pass

    return SavedFile(
        original_filename=upload.filename or "upload",
        content_type=getattr(upload, "content_type", None),
        saved_path=out_path,
        size_bytes=size,
    )


def delete_file(path: Path) -> None:
    """
    Best-effort delete. Use after processing ingestion.
    """
    try:
        if path.exists():
            path.unlink()
    except Exception:
        # intentionally swallow; caller can log if desired
        pass


def bytes_from_mb(mb: int) -> int:
    return mb * 1024 * 1024


def default_upload_dir(project_root: Optional[Path] = None) -> Path:
    """
    Optional convenience: choose a default temp upload dir.
    You can replace this later with Settings.UPLOAD_TMP_DIR.
    """
    if project_root is None:
        project_root = Path(os.getcwd())
    return project_root / ".uploads"
