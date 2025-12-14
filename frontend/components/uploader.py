from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st


def render_uploader(
    *,
    backend_base_url: str,
    default_namespace: str = "",
    show_advanced: bool = True,
    key_prefix: str = "doc_uploader",
) -> Optional[Dict[str, Any]]:
    """
    Renders a document uploader (PDF/DOCX/TXT/MD) and calls backend /v1/ingest/file.

    Args:
        backend_base_url: e.g. "http://localhost:8000/v1"
    Returns:
        Parsed JSON response dict (IngestFilesResponse) if ingestion succeeded, else None.
    """
    st.subheader("📄 Document upload")

    files = st.file_uploader(
        "Upload documents to add to your knowledge base",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key=f"{key_prefix}__files",
    )

    namespace = st.text_input(
        "Namespace (optional)",
        value=default_namespace,
        placeholder="e.g. company_kb",
        key=f"{key_prefix}__namespace",
    )

    col1, col2 = st.columns(2)
    with col1:
        enable_ocr = st.toggle("Enable OCR for scanned PDFs", value=False, key=f"{key_prefix}__enable_ocr")
    with col2:
        ocr_language = st.text_input("OCR language", value="eng", disabled=not enable_ocr, key=f"{key_prefix}__ocr_lang")

    # Defaults match backend Form defaults
    chunk_size = 1000
    chunk_overlap = 150
    ocr_dpi = 200
    ocr_min_chars = 50
    max_pages: Optional[int] = None

    if show_advanced:
        with st.expander("Advanced settings", expanded=False):
            chunk_size = st.slider(
                "Chunk size",
                min_value=300,
                max_value=3000,
                value=1000,
                step=50,
                key=f"{key_prefix}__chunk_size",
            )
            chunk_overlap = st.slider(
                "Chunk overlap",
                min_value=0,
                max_value=500,
                value=150,
                step=10,
                key=f"{key_prefix}__chunk_overlap",
            )

            st.markdown("**OCR (PDF only)**")
            ocr_dpi = st.slider(
                "OCR DPI",
                min_value=100,
                max_value=400,
                value=200,
                step=25,
                disabled=not enable_ocr,
                key=f"{key_prefix}__ocr_dpi",
            )
            ocr_min_chars = st.slider(
                "OCR fallback threshold (min extracted chars)",
                min_value=0,
                max_value=500,
                value=50,
                step=10,
                disabled=not enable_ocr,
                key=f"{key_prefix}__ocr_min_chars",
            )
            max_pages_val = st.number_input(
                "Max PDF pages to process (optional)",
                min_value=0,
                value=0,
                step=1,
                help="Set 0 for unlimited. Useful to cap very large PDFs.",
                key=f"{key_prefix}__max_pages",
            )
            max_pages = None if max_pages_val == 0 else int(max_pages_val)

    ingest_btn = st.button(
        "Ingest documents",
        type="primary",
        disabled=not files,
        use_container_width=True,
        key=f"{key_prefix}__ingest_btn",
    )

    if not ingest_btn:
        return None

    if not files:
        st.warning("Please select at least one file.")
        return None

    ingest_url = f"{backend_base_url.rstrip('/')}/ingest/file"

    multipart_files: List[Tuple[str, Tuple[str, bytes, str]]] = []
    for f in files:
        file_bytes = f.getvalue()
        content_type = f.type or "application/octet-stream"
        multipart_files.append(("files", (f.name, file_bytes, content_type)))

    form_data: Dict[str, Any] = {
        "namespace": namespace or "",
        "enable_ocr": "true" if enable_ocr else "false",
        "ocr_language": ocr_language or "eng",
        "ocr_dpi": str(int(ocr_dpi)),
        "ocr_min_chars": str(int(ocr_min_chars)),
        "chunk_size": str(int(chunk_size)),
        "chunk_overlap": str(int(chunk_overlap)),
    }
    if max_pages is not None:
        form_data["max_pages"] = str(int(max_pages))

    with st.spinner("Uploading and ingesting…"):
        try:
            resp = requests.post(
                ingest_url,
                files=multipart_files,
                data=form_data,
                timeout=180,
            )
        except requests.RequestException as e:
            st.error(f"Upload failed: {e}")
            return None

    if resp.status_code >= 400:
        try:
            err = resp.json()
        except Exception:
            err = {"detail": resp.text}
        st.error(f"Ingestion failed ({resp.status_code}): {err}")
        return None

    try:
        payload = resp.json()
    except json.JSONDecodeError:
        st.error("Backend returned non-JSON response.")
        st.code(resp.text)
        return None

    _render_ingest_results(payload)
    return payload


def _render_ingest_results(payload: Dict[str, Any]) -> None:
    status = payload.get("status", "unknown")
    num_docs = payload.get("num_documents", 0)
    num_chunks = payload.get("num_chunks", 0)
    namespace = payload.get("namespace")

    st.success(f"Ingestion complete: status={status}, documents={num_docs}, chunks={num_chunks}")
    if namespace:
        st.caption(f"Namespace: `{namespace}`")

    results = payload.get("results") or []
    if not results:
        return

    rows = []
    for r in results:
        rows.append(
            {
                "filename": r.get("filename"),
                "status": r.get("status"),
                "chunks": r.get("num_chunks", 0),
                "pages": r.get("num_pages"),
                "ocr": r.get("ocr_used"),
                "reason": r.get("reason"),
            }
        )

    st.dataframe(rows, use_container_width=True)

    problems = [r for r in results if r.get("status") in ("error", "skipped")]
    if problems:
        with st.expander("Details (skipped/errors)"):
            for r in problems:
                st.markdown(f"- **{r.get('filename')}** → `{r.get('status')}`: {r.get('reason')}")
