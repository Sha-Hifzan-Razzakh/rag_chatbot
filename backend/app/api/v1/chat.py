import base64
import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence, cast

from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolUnionParam,
)

from app.agents.orchestrator import AgentOrchestrator, OrchestratorLLM
from app.agents.policies import AgentPolicies
from app.agents.tools.registry import ToolContext, ToolRegistry
from app.agents.tools.rag_tools import register_rag_tools
from app.audio.tts import synthesize_speech
from app.core.config import settings
from app.core.logging_config import get_agent_logger
from app.models.schemas import ChatRequest, ChatResponse, IntentLiteral, Message, Source
from app.rag.intent import classify_intent
from app.rag.pipeline import run_chitchat_pipeline, run_rag_pipeline

# module-level logger (no request context here)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat")


# ----------------------------
# Minimal OpenAI LLM adapter
# ----------------------------

class OpenAILLM(OrchestratorLLM):
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY.get_secret_value() if settings.OPENAI_API_KEY else None,
            base_url=settings.OPENAI_API_BASE or None,
        )

    async def chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        messages_param = cast(Sequence[ChatCompletionMessageParam], messages)

        create_kwargs: Dict[str, Any] = {
            "model": settings.CHAT_MODEL_NAME,
            "messages": messages_param,
            "temperature": temperature if temperature is not None else settings.DEFAULT_TEMPERATURE,
        }

        # Only pass tools/tool_choice when present (Pylance-safe)
        if tools is not None:
            create_kwargs["tools"] = cast(Sequence[ChatCompletionToolUnionParam], tools)
        if tool_choice is not None:
            create_kwargs["tool_choice"] = cast(ChatCompletionToolChoiceOptionParam, tool_choice)

        resp = await self._client.chat.completions.create(**create_kwargs)

        choice0 = resp.choices[0]
        msg = choice0.message

        tool_calls: List[Dict[str, Any]] = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            if getattr(tc, "type", None) == "function" and getattr(tc, "function", None) is not None:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        usage: Dict[str, Any] = {}
        u = getattr(resp, "usage", None)
        if u is not None:
            usage = {
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            }

        return {
            "message": {
                "role": msg.role,
                "content": msg.content,
                "tool_calls": tool_calls,
            },
            "usage": usage,
            "stop_reason": getattr(choice0, "finish_reason", None),
        }


def _extract_sources_from_trace(trace: Optional[list]) -> List[Source]:
    if not trace:
        return []

    sources: List[Source] = []

    for step in trace:
        tool_result = getattr(step, "tool_result", None)
        if not tool_result:
            continue

        tool_name = getattr(tool_result, "name", None)
        output = getattr(tool_result, "output", None)

        if tool_name == "search_docs" and isinstance(output, dict):
            results = output.get("results") or []
            for r in results:
                if not isinstance(r, dict):
                    continue
                md = r.get("metadata")
                metadata = md if isinstance(md, dict) else {}
                sources.append(
                    Source(
                        id=str(r.get("id", "")),
                        title=r.get("title"),
                        snippet=r.get("snippet"),
                        metadata=metadata,
                    )
                )

        if tool_name == "answer_with_rag" and isinstance(output, dict):
            raw_sources = output.get("sources") or []
            if isinstance(raw_sources, list):
                for s in raw_sources:
                    if not isinstance(s, dict):
                        continue
                    md = s.get("metadata")
                    metadata = md if isinstance(md, dict) else {}
                    sources.append(
                        Source(
                            id=str(s.get("id", s.get("source_id", "")) or ""),
                            title=s.get("title"),
                            snippet=s.get("snippet") or s.get("text"),
                            metadata=metadata,
                        )
                    )

    deduped: List[Source] = []
    seen = set()
    for s in sources:
        key = (s.id, s.snippet)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    return deduped


@router.post(
    "",
    response_model=ChatResponse,
    summary="Chat with the RAG chatbot",
    response_description="Chatbot answer with sources and suggested questions.",
)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    # conversation id (required in response now)
    conversation_id = payload.conversation_id or str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    # request-scoped logger with context
    req_logger = get_agent_logger(__name__, conversation_id=conversation_id, request_id=request_id)

    req_logger.info(
        "Received chat request: conversation_id=%r question=%r",
        conversation_id,
        payload.question,
    )

    # classify intent (kept from old flow)
    try:
        intent: IntentLiteral = await classify_intent(payload.question)
    except Exception as exc:  # noqa: BLE001
        req_logger.exception("Failed to classify intent: %s", exc)
        intent = "RAG_QA"

    req_logger.info("Classified intent: %s", intent)

    # Determine top_k (either request-level or default from config)
    top_k = payload.top_k or settings.DEFAULT_TOP_K
    if top_k > settings.MAX_TOP_K:
        req_logger.warning(
            "Requested top_k=%s exceeds MAX_TOP_K=%s, capping.",
            top_k,
            settings.MAX_TOP_K,
        )
        top_k = settings.MAX_TOP_K

    try:
        # ----------------------------
        # Agentic path (A-version)
        # ----------------------------
        if settings.AGENT_ENABLED:
            registry = ToolRegistry(allowlist=getattr(settings, "tools_allowlist_list", None))
            register_rag_tools(registry)

            tool_choice = "none" if intent == "CHITCHAT" else "auto"
            policies = AgentPolicies.from_settings(tool_choice=tool_choice)

            llm = OpenAILLM()
            orchestrator = AgentOrchestrator(llm=llm, registry=registry, policies=policies)

            augmented_history: List[Message] = list(payload.history or [])
            if intent != "CHITCHAT":
                augmented_history.append(
                    Message(
                        role="system",
                        content=(
                            f"Retrieval defaults for tool use: top_k={top_k}, "
                            f"namespace={payload.namespace or ''}. "
                            "When calling search_docs, use these unless the user specifies otherwise."
                        ),
                    )
                )

            tool_ctx = cast(
                ToolContext,
                {
                    "conversation_id": conversation_id,
                    "request_id": request_id,
                    "logger": req_logger,
                    "settings": settings,
                    "rag": {},
                },
            )

            agent_result = await orchestrator.run(
                question=payload.question,
                history=augmented_history,
                conversation_id=conversation_id,
                debug=payload.debug,
                context=tool_ctx,
                tone=payload.tone or "neutral",
                style=payload.style or "concise",
                temperature=settings.DEFAULT_TEMPERATURE,
            )

            answer_text = (agent_result.message.content or "").strip()
            trace = agent_result.trace if payload.debug else None
            sources = _extract_sources_from_trace(trace)

            response = ChatResponse(
                conversation_id=agent_result.conversation_id,
                trace=trace,
                answer=answer_text,
                sources=sources,
                suggested_questions=[],
                intent=intent,
                answer_audio_b64=None,
            )

        # ----------------------------
        # Legacy path (agent disabled)
        # ----------------------------
        else:
            if intent == "CHITCHAT":
                req_logger.debug("Routing to chitchat pipeline.")
                response = await run_chitchat_pipeline(
                    question=payload.question,
                    history=payload.history,
                    tone=payload.tone or "neutral",
                    style=payload.style or "concise",
                )
            else:
                req_logger.debug("Routing to RAG pipeline.")
                response = await run_rag_pipeline(
                    question=payload.question,
                    history=payload.history,
                    tone=payload.tone or "neutral",
                    style=payload.style or "concise",
                    top_k=top_k,
                    namespace=payload.namespace,
                )

            response.conversation_id = conversation_id
            response.trace = None
            response.intent = intent

        # ----------------------------
        # Optional TTS
        # ----------------------------
        req_logger.info(
            "TTS flags: return_audio=%s ENABLE_TTS=%s",
            payload.return_audio,
            settings.ENABLE_TTS,
        )

        if payload.return_audio and settings.ENABLE_TTS:
            try:
                audio_bytes = synthesize_speech(
                    text=response.answer,
                    voice=settings.TTS_VOICE,
                    format=settings.TTS_FORMAT,
                )
                response.answer_audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            except Exception as tts_exc:  # noqa: BLE001
                req_logger.exception("TTS failed (continuing without audio): %s", tts_exc)
                response.answer_audio_b64 = None

        return response

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        req_logger.exception("Error while handling chat request: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the chat request.",
        ) from exc
