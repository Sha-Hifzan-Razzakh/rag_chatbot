import pytest
from httpx import AsyncClient
from httpx import ASGITransport

from app.core.config import settings
from app.main import app


@pytest.mark.anyio
async def test_chat_endpoint_rag_returns_conversation_id(monkeypatch):
    """
    Ensures /v1/chat returns the new required fields:
    - conversation_id (always)
    - trace (optional; not present unless debug=true)
    """
    # Force legacy path for this test (simple + deterministic)
    monkeypatch.setattr(settings, "AGENT_ENABLED", False)

    # Mock intent classifier -> RAG_QA
    async def _mock_classify_intent(_q: str):
        return "RAG_QA"

    monkeypatch.setattr("app.rag.intent.classify_intent", _mock_classify_intent)

    # Mock RAG pipeline response
    async def _mock_run_rag_pipeline(**kwargs):
        from app.models.schemas import ChatResponse, Source

        return ChatResponse(
            conversation_id=kwargs.get("conversation_id") or "test-convo",
            trace=None,
            answer="mock answer",
            sources=[
                Source(id="S1", title="Doc 1", snippet="snippet", metadata={"page": 1}),
            ],
            suggested_questions=["q1", "q2"],
            intent="RAG_QA",
            answer_audio_b64=None,
        )

    monkeypatch.setattr("app.rag.pipeline.run_rag_pipeline", _mock_run_rag_pipeline)

    payload = {
        "question": "What is in the docs?",
        "history": [],
        "tone": "neutral",
        "style": "concise",
        "top_k": 3,
        "namespace": None,
        "return_audio": False,
        "debug": False,
        "conversation_id": "conv-123",
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/v1/chat", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    assert data["conversation_id"] == "conv-123"
    assert data["answer"] == "mock answer"
    assert data["intent"] == "RAG_QA"
    assert isinstance(data["sources"], list)
    assert data.get("trace") is None


@pytest.mark.anyio
async def test_chat_endpoint_chitchat_returns_conversation_id(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_ENABLED", False)

    async def _mock_classify_intent(_q: str):
        return "CHITCHAT"

    monkeypatch.setattr("app.rag.intent.classify_intent", _mock_classify_intent)

    async def _mock_run_chitchat_pipeline(**kwargs):
        from app.models.schemas import ChatResponse

        return ChatResponse(
            conversation_id=kwargs.get("conversation_id") or "test-convo",
            trace=None,
            answer="hello!",
            sources=[],
            suggested_questions=[],
            intent="CHITCHAT",
            answer_audio_b64=None,
        )

    monkeypatch.setattr("app.rag.pipeline.run_chitchat_pipeline", _mock_run_chitchat_pipeline)

    payload = {
        "question": "hi",
        "history": [],
        "tone": "neutral",
        "style": "concise",
        "return_audio": False,
        "debug": False,
        "conversation_id": "conv-999",
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/v1/chat", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    assert data["conversation_id"] == "conv-999"
    assert data["answer"] == "hello!"
    assert data["intent"] == "CHITCHAT"
    assert data["sources"] == []
    assert data.get("trace") is None
