import os
from typing import Any, Dict, Optional

import streamlit as st
from dotenv import load_dotenv
from .stt import stt_widget

# NEW: document uploader
try:
    from .uploader import render_uploader  # type: ignore
except Exception:  # noqa: BLE001
    def render_uploader(*, backend_base_url: str, default_namespace: str = "", show_advanced: bool = True, key_prefix: str = "uploader"):
        st.info("components/uploader.py not found yet. Create it to enable document upload ingestion.")
        return None


def render_sidebar(backend_url: str) -> Dict[str, Any]:
    """
    Render the sidebar controls and return the current configuration.

    Returns a dict with keys:
      - tone: str
      - style: str
      - top_k: int
      - namespace: Optional[str]
      - ingest_text: str
      - ingest_button_clicked: bool
      - tts_enabled/use_tts/tts_mode
      - debug: bool                  (NEW: agent trace)
      - reset_conversation: bool     (NEW: reset conversation_id + messages)
    """
    with st.sidebar:
        st.title("RAG Chatbot Settings")

        # Backend info (read-only, real URL is handled in streamlit_app.py)
        st.markdown("#### Backend")
        st.text_input(
            "Backend URL",
            value=backend_url,
            disabled=True,
            help="Configure via BACKEND_URL env var.",
            key="backend_url_display",
        )

        st.markdown("#### Answer controls")
        tone = st.selectbox(
            "Tone",
            options=["neutral", "formal", "casual"],
            index=0,
            key="tone_select",
        )
        style = st.selectbox(
            "Style",
            options=["concise", "detailed", "bullet_points"],
            index=0,
            key="style_select",
        )
        top_k = st.slider(
            "Top K (number of context chunks)",
            min_value=1,
            max_value=20,
            value=5,
            key="top_k_slider",
        )

        namespace_input = st.text_input(
            "Namespace (optional)",
            value="",
            key="namespace_input",
            placeholder="e.g. 'kb-1' or leave blank for default",
        )
        namespace: Optional[str] = namespace_input or None

        st.divider()

        # --- Agent (NEW) ---
        st.markdown("#### Agent")
        debug = st.checkbox(
            "Show agent trace (debug)",
            value=st.session_state.get("debug", False),
            key="agent_debug_trace",
            help="When enabled, backend will return an execution trace (tool calls, stop reason, etc.).",
        )
        # keep state consistent for streamlit_app.py to read
        st.session_state["debug"] = bool(debug)

        st.markdown("---")

        # --- TTS (kept) ---
        tts_enabled = st.checkbox("🔊 Read answers aloud (TTS)", key="use_tts", value=False)

        tts_mode = "inline_base64"
        if tts_enabled:
            tts_mode = st.selectbox(
                "TTS mode",
                ["inline_base64", "separate_endpoint"],
                index=0,
                key="tts_mode",
            )

        st.divider()

        # --- STT (kept) ---
        stt_widget(backend_url=backend_url)

        st.markdown("---")

        # --- Document upload ingestion (kept) ---
        st.markdown("#### Ingest documents")
        render_uploader(
            backend_base_url=f"{backend_url}/v1",
            default_namespace=namespace or "",
            show_advanced=False,
            key_prefix="sidebar_uploader",
        )

        st.markdown("---")

        # --- Existing: Ingest text ---
        st.markdown("#### Ingest text")
        ingest_text = st.text_area(
            "Text to ingest into the knowledge base",
            key="ingest_text_area",
            height=150,
            placeholder="Paste some text here and click 'Ingest text' to add it to the RAG KB.",
        )
        ingest_button_clicked = st.button("Ingest text", key="ingest_button")

        st.markdown("---")

        # Reset conversation (NEW: clears chat + conversation_id)
        reset_conversation = st.button("Reset conversation", key="reset_conversation_button")
        if reset_conversation:
            st.session_state["messages"] = []
            st.session_state["conversation_id"] = None
            st.rerun()

        # Keep the old button too (optional/back-compat)
        if st.button("Clear chat", key="clear_chat_button"):
            st.session_state["messages"] = []
            st.session_state["conversation_id"] = None
            st.rerun()

        # Optional debug toggles
        st.markdown("#### Debug (optional)")
        st.checkbox(
            "Show raw backend responses (not implemented yet)",
            value=False,
            key="debug_show_raw",
        )

    return {
        "tone": tone,
        "style": style,
        "top_k": top_k,
        "namespace": namespace,
        "ingest_text": ingest_text,
        "ingest_button_clicked": ingest_button_clicked,
        "use_tts": st.session_state.get("use_tts", False),
        "tts_enabled": tts_enabled,
        "tts_mode": tts_mode,

        # NEW
        "debug": bool(debug),
        "reset_conversation": bool(reset_conversation),
    }
