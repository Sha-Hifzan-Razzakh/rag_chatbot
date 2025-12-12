import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import ChatResponse


client = TestClient(app)


@pytest.mark.asyncio
async def test_chat_endpoint_chitchat(monkeypatch):
    # Import here so we can monkeypatch the functions used by the endpoint.
    import app.rag.intent as intent_module
    import app.rag.pipeline as pipeline_module

    async def fake_classify_intent(question: str) -> str:
        return "CHITCHAT"

    async def fake_chitchat_pipeline(question, history, tone, style) -> ChatResponse:
        return ChatResponse(
            answer="Hello from fake chitchat pipeline!",
            sources=[],
            suggested_questions=[],
            intent="CHITCHAT",
        )

    # Apply monkeypatches
    monkeypatch.setattr(intent_module, "classify_intent", fake_classify_intent)
    monkeypatch.setattr(pipeline_module, "run_chitchat_pipeline", fake_chitchat_pipeline)

    payload = {
        "question": "Hi there!",
        "history": [],
        "tone": "casual",
        "style": "concise",
    }

    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["answer"] == "Hello from fake chitchat pipeline!"
    assert data["intent"] == "CHITCHAT"
    assert isinstance(data["sources"], list)
    assert isinstance(data["suggested_questions"], list)
