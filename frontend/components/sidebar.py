import os
from typing import Any, Dict, Optional

import streamlit as st
from dotenv import load_dotenv
from .stt import stt_widget



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

        tts_enabled = st.checkbox("🔊 Read answers aloud (TTS)", key="use_tts", value=False)
        st.divider()

        stt_widget(backend_url=backend_url)

        st.markdown("---")
        st.markdown("#### Ingest text")
        ingest_text = st.text_area(
            "Text to ingest into the knowledge base",
            key="ingest_text_area",
            height=150,
            placeholder="Paste some text here and click 'Ingest text' to add it to the RAG KB.",
        )
        ingest_button_clicked = st.button("Ingest text", key="ingest_button")

        st.markdown("---")
        if st.button("Clear chat", key="clear_chat_button"):
            st.session_state["messages"] = []

        # Optional debug toggles (not currently used in main app, but handy later)
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
    }

