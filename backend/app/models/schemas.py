from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# --- Shared types ---


RoleLiteral = Literal["user", "assistant"]
IntentLiteral = Literal["RAG_QA", "CHITCHAT", "OTHER"]


class Message(BaseModel):
    """
    Single message in the chat history.
    """

    role: RoleLiteral = Field(
        ...,
        description='Message role, e.g. "user" or "assistant".',
    )
    content: str = Field(
        ...,
        description="Plain text content of the message.",
    )


# --- Chat endpoint models ---


class ChatRequest(BaseModel):
    """
    Request body for /v1/chat endpoint.
    """

    question: str = Field(
        ...,
        description="The current user question or message.",
    )
    history: List[Message] = Field(
        default_factory=list,
        description="Previous messages in the conversation (user + assistant).",
    )
    tone: Optional[str] = Field(
        default="neutral",
        description='Optional tone hint, e.g. "neutral", "formal", "casual".',
    )
    style: Optional[str] = Field(
        default="concise",
        description='Optional style hint, e.g. "concise", "detailed", "bullet_points".',
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of context documents to retrieve; falls back to DEFAULT_TOP_K.",
        ge=1,
    )
    namespace: Optional[str] = Field(
        default=None,
        description="Optional namespace / knowledge base identifier for multi-tenant setups.",
    )


class Source(BaseModel):
    """
    Single retrieved source snippet used to answer a question.
    """

    id: str = Field(
        ...,
        description="Identifier of the source (e.g. document ID or chunk ID).",
    )
    title: Optional[str] = Field(
        default=None,
        description="Human-readable title of the source document, if available.",
    )
    snippet: Optional[str] = Field(
        default=None,
        description="Short snippet of the source text relevant to the answer.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (file path, page number, etc.).",
    )


class ChatResponse(BaseModel):
    """
    Response body for /v1/chat endpoint.
    """

    answer: str = Field(
        ...,
        description="Final answer text returned by the assistant.",
    )
    sources: List[Source] = Field(
        default_factory=list,
        description="List of source snippets that were used for this answer.",
    )
    suggested_questions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions for better exploration.",
    )
    intent: IntentLiteral = Field(
        ...,
        description='Detected query intent, e.g. "RAG_QA", "CHITCHAT", or "OTHER".',
    )


# --- Ingestion endpoint models ---


class IngestTextRequest(BaseModel):
    """
    Request body for /v1/ingest/text endpoint.
    """

    texts: List[str] = Field(
        ...,
        description="Raw text documents to ingest into the vector store.",
        min_length=1,
    )
    namespace: Optional[str] = Field(
        default=None,
        description="Optional namespace / knowledge base identifier.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata applied to all ingested chunks from these texts.",
    )


class IngestResponse(BaseModel):
    """
    Generic ingestion response for text or files.
    """

    status: str = Field(
        ...,
        description='Status string, usually "ok" on success.',
    )
    num_documents: int = Field(
        ...,
        description="Number of documents ingested.",
    )
    num_chunks: int = Field(
        ...,
        description="Total number of chunks stored in the vector DB.",
    )
