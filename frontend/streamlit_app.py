import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment / config
# ---------------------------------------------------------------------------

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{BACKEND_URL}/v1/chat"
INGEST_TEXT_ENDPOINT = f"{BACKEND_URL}/v1/ingest/text"
HEALTH_ENDPOINT = f"{BACKEND_URL}/v1/health"


# ---------------------------------------------------------------------------
# Optional component imports with fallback
# ---------------------------------------------------------------------------

try:
    from components.chat_ui import render_chat  # type: ignore
except Exception:  # noqa: BLE001
    def render_chat(messages: List[Dict[str, Any]]) -> None:
        """
        Fallback chat renderer if components.chat_ui is not implemented yet.

        Expects messages as a list of dicts:
        {
          "role": "user" | "assistant",
          "content": str,
          "intent": Optional[str],
          "sources": Optional[list],
          "suggested_questions": Optional[list[str]],
        }
        """
        for message in messages:
            role = message.get("role", "assistant")
            content = message.get("content", "")
            intent = message.get("intent")
            sources = message.get("sources") or []
            suggested = message.get("suggested_questions") or []

            with st.chat_message(role):
                if intent and role == "assistant":
                    st.caption(f"Intent: {intent}")
                st.markdown(content)

                # Show sources under assistant messages
                if role == "assistant" and sources:
                    with st.expander("Sources", expanded=False):
                        for idx, src in enumerate(sources, start=1):
                            title = src.get("title") or f"Source {idx}"
                            snippet = src.get("snippet") or ""
                            meta = src.get("metadata") or {}
                            st.markdown(f"**{title}**")
                            if snippet:
                                st.write(snippet)
                            if meta:
                                with st.expander("Metadata", expanded=False):
                                    st.json(meta)

                # Show suggested questions as info chips (buttons handled outside)
                if role == "assistant" and suggested:
                    st.markdown("**Suggested questions:**")
                    for q in suggested:
                        st.markdown(f"- {q}")


try:
    from components.sidebar import render_sidebar  # type: ignore
except Exception:  # noqa: BLE001
    def render_sidebar() -> Dict[str, Any]:
        """
        Fallback sidebar renderer if components.sidebar is not implemented yet.

        Returns a dict with:
        - tone
        - style
        - top_k
        - namespace
        - ingest_text
        - ingest_button_clicked
        """
        with st.sidebar:
            st.title("RAG Chatbot Settings")

            st.markdown("#### Backend")
            st.text_input("Backend URL", value=BACKEND_URL, key="backend_url", disabled=True)

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
            namespace = st.text_input(
                "Namespace (optional)",
                value="",
                key="namespace_input",
                placeholder="e.g. 'kb-1' or leave blank",
            )

            st.markdown("---")
            st.markdown("#### Ingest text")
            ingest_text = st.text_area(
                "Text to ingest (will be split into chunks)",
                key="ingest_text_area",
                height=150,
            )
            ingest_button_clicked = st.button("Ingest text", key="ingest_button")

            st.markdown("---")
            if st.button("Clear chat", key="clear_chat_button"):
                st.session_state["messages"] = []

        return {
            "tone": tone,
            "style": style,
            "top_k": top_k,
            "namespace": namespace or None,
            "ingest_text": ingest_text,
            "ingest_button_clicked": ingest_button_clicked,
        }


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "queued_question" not in st.session_state:
        st.session_state["queued_question"] = None


# ---------------------------------------------------------------------------
# Backend calls
# ---------------------------------------------------------------------------


def call_health() -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None


def call_ingest_text(text: str, namespace: Optional[str]) -> Optional[Dict[str, Any]]:
    try:
        payload = {
            "texts": [text],
            "namespace": namespace,
            "metadata": {},
        }
        resp = requests.post(INGEST_TEXT_ENDPOINT, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Ingest failed: {resp.status_code} - {resp.text}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Ingest error: {exc}")
    return None


def call_chat(
    question: str,
    history: List[Dict[str, Any]],
    tone: str,
    style: str,
    top_k: int,
    namespace: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Call the backend /v1/chat endpoint.

    history is a list of messages in the UI format; we map it down to
    [{role, content}, ...] for the API.
    """
    api_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
        if msg.get("role") in ("user", "assistant")
    ]

    payload = {
        "question": question,
        "history": api_history,
        "tone": tone,
        "style": style,
        "top_k": top_k,
        "namespace": namespace,
    }

    try:
        resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Chat failed: {resp.status_code} - {resp.text}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Chat error: {exc}")
    return None


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide",
    )

    init_session_state()

    st.title("🤖 RAG Chatbot")
    st.caption("Backend: FastAPI + LangChain + Postgres (pgvector) | Frontend: Streamlit")

    # Health indicator
    health = call_health()
    cols = st.columns(3)
    with cols[0]:
        if health:
            st.success("Backend: healthy ✅")
        else:
            st.error("Backend: unreachable ❌")
    with cols[1]:
        st.write(f"Backend URL: `{BACKEND_URL}`")
    with cols[2]:
        if health:
            st.write(f"Environment: `{health.get('environment', 'unknown')}`")

    # Sidebar settings & ingestion
    sidebar_state = render_sidebar()
    tone = sidebar_state["tone"]
    style = sidebar_state["style"]
    top_k = sidebar_state["top_k"]
    namespace = sidebar_state["namespace"]

    if sidebar_state["ingest_button_clicked"] and sidebar_state["ingest_text"].strip():
        with st.spinner("Ingesting text into vector store..."):
            ingest_result = call_ingest_text(
                text=sidebar_state["ingest_text"],
                namespace=namespace,
            )
        if ingest_result:
            st.success(
                f"Ingested {ingest_result.get('num_documents', 0)} document(s), "
                f"{ingest_result.get('num_chunks', 0)} chunk(s)."
            )

    st.markdown("---")

    # Display chat history
    render_chat(st.session_state["messages"])

    # Process queued suggested question if any
    queued_question = st.session_state.pop("queued_question", None)

    # Chat input
    user_input = st.chat_input("Ask a question about your documents or just chat...")

    # Suggested question clicked?
    question_to_send: Optional[str] = None
    if user_input:
        question_to_send = user_input
    elif queued_question:
        question_to_send = queued_question

    # If we have a question to send, do the round-trip
    if question_to_send:
        # Append user message
        st.session_state["messages"].append(
            {
                "role": "user",
                "content": question_to_send,
            }
        )

        with st.chat_message("user"):
            st.markdown(question_to_send)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = call_chat(
                    question=question_to_send,
                    history=st.session_state["messages"][:-1],
                    tone=tone,
                    style=style,
                    top_k=top_k,
                    namespace=namespace,
                )

            if response is None:
                st.error("No response from backend.")
                return

            answer = response.get("answer", "")
            sources = response.get("sources", [])
            suggested = response.get("suggested_questions", [])
            intent = response.get("intent", "RAG_QA")

            # Show answer in the live assistant bubble
            st.markdown(answer)

            if sources:
                with st.expander("Sources", expanded=False):
                    for idx, src in enumerate(sources, start=1):
                        title = src.get("title") or f"Source {idx}"
                        snippet = src.get("snippet") or ""
                        meta = src.get("metadata") or {}
                        st.markdown(f"**{title}**")
                        if snippet:
                            st.write(snippet)
                        if meta:
                            with st.expander("Metadata", expanded=False):
                                st.json(meta)

            # Suggested questions as buttons (click queues them for the next run)
            if suggested:
                st.markdown("**Suggested questions:**")
                for i, q in enumerate(suggested):
                    if st.button(q, key=f"suggested_{len(st.session_state['messages'])}_{i}"):
                        st.session_state["queued_question"] = q
                        st.rerun()

            # Append to history for future render
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "intent": intent,
                    "sources": sources,
                    "suggested_questions": suggested,
                }
            )


if __name__ == "__main__":
    main()
