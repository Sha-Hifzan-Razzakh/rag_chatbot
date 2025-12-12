from typing import Any, Dict, List

import streamlit as st


def render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return

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


def render_suggested_questions(suggested_questions: List[str]) -> None:
    if not suggested_questions:
        return

    st.markdown("**Suggested questions:**")
    # The actual buttons for suggested questions are handled in streamlit_app.py,
    # so here we just display them as a list.
    for q in suggested_questions:
        st.markdown(f"- {q}")


def render_chat(messages: List[Dict[str, Any]]) -> None:
    """
    Render chat history using Streamlit's chat UI.

    Expected message format:
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

            if role == "assistant":
                render_sources(sources)
                render_suggested_questions(suggested)
