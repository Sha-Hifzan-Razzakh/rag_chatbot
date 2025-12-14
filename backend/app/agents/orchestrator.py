"""
Agent orchestrator (B-version, drop-in replacement).

What this improves vs A-version:
- Correct message ordering for tool calling:
    assistant(tool_calls) -> tool(result) -> assistant(next turn)
- More robust tool-call extraction:
    supports OpenAI-style tool_calls and legacy function_call
- Keeps the same "minimal + dependency-light" philosophy
- Preserves A-version behavior of stopping early on tool errors (you can relax that later)

It expects your existing project types:
  - app.models.schemas: AgentResult, AgentStopReason, AgentTraceStep, ChatMessage, ToolCall, ToolResult, Message
  - app.agents.tools.registry: ToolContext, ToolRegistry, ToolExecutionError, ToolNotAllowedError
  - app.agents.policies: AgentPolicies
  - app.core.prompts: AGENT_SYSTEM_PROMPT
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

from app.models.schemas import Message
from app.agents.policies import AgentPolicies
from app.agents.tools.registry import (
    ToolContext,
    ToolRegistry,
    ToolExecutionError,
    ToolNotAllowedError,
)
from app.core.prompts import AGENT_SYSTEM_PROMPT
from app.models.schemas import (
    AgentResult,
    AgentStopReason,
    AgentTraceStep,
    ChatMessage,
    ToolCall,
    ToolResult,
)

JsonDict = Dict[str, Any]


class OrchestratorLLM:
    """
    Minimal interface for the injected LLM adapter.
    """

    async def chat(
        self,
        *,
        messages: List[JsonDict],
        tools: Optional[List[JsonDict]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> JsonDict:
        raise NotImplementedError


class AgentOrchestrator:
    def __init__(
        self,
        *,
        llm: OrchestratorLLM,
        registry: ToolRegistry,
        policies: AgentPolicies,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._policies = policies.clamp()

    async def run(
        self,
        *,
        question: str,
        history: Optional[List[ChatMessage]] = None,
        conversation_id: Optional[str] = None,
        debug: bool = False,
        context: Optional[ToolContext] = None,
        tone: Optional[str] = None,
        style: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> AgentResult:
        """
        Run the agent loop and return AgentResult.

        - question: current user question
        - history: list of prior messages (user/assistant/tool) if any
        - conversation_id: optionally provided by caller
        - debug: whether to include trace
        - context: passed to tools (logger/settings/rag/etc.)
        """
        convo_id = conversation_id or str(uuid.uuid4())
        ctx = cast(ToolContext, dict(context or {}))
        ctx.setdefault("conversation_id", convo_id)

        trace_enabled = bool(debug) or bool(self._policies.debug_trace_default)
        trace: List[AgentTraceStep] = []

        tool_calls_made: List[ToolCall] = []
        tool_call_count = 0

        # Build message list for the LLM
        messages: List[JsonDict] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        ]

        # Optional tone/style nudge (kept light)
        if tone or style:
            messages.append(
                {
                    "role": "system",
                    "content": f"Tone: {tone or 'neutral'}\nStyle: {style or 'concise'}",
                }
            )

        if history:
            for m in history:
                messages.append(_chatmessage_to_dict(m))

        messages.append({"role": "user", "content": question})

        # Tool specs for function calling
        tool_specs = self._registry.list_tool_specs()

        stop = AgentStopReason(reason="completed", detail=None)
        usage_agg: Dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for step_idx in range(1, self._policies.max_steps + 1):
            step_t0 = time.perf_counter()
            try:
                llm_resp: JsonDict = await self._llm.chat(
                    messages=messages,
                    tools=tool_specs if tool_specs else None,
                    tool_choice=self._policies.tool_choice,
                    temperature=temperature,
                )
            except Exception as e:
                stop = AgentStopReason(reason="llm_error", detail=str(e))
                if trace_enabled:
                    trace.append(
                        AgentTraceStep(
                            step=step_idx,
                            type="error",
                            input={"messages_count": len(messages)},
                            output={"error": str(e)},
                            latency_ms=_ms(step_t0),
                            timestamp=datetime.utcnow(),
                        )
                    )
                return _final_result(
                    conversation_id=convo_id,
                    final_text="Sorry—something went wrong while generating a response.",
                    stop=stop,
                    trace=trace if trace_enabled else None,
                    tool_calls=tool_calls_made,
                    usage=usage_agg,
                )

            _merge_usage(usage_agg, llm_resp.get("usage"))

            assistant_msg, tool_calls = _extract_assistant_message_and_tool_calls(llm_resp)

            if trace_enabled:
                trace.append(
                    AgentTraceStep(
                        step=step_idx,
                        type="llm",
                        input={"messages_count": len(messages), "tools_count": len(tool_specs)},
                        output={
                            "assistant_content": assistant_msg.get("content"),
                            "tool_calls_count": len(tool_calls),
                            "stop_reason": llm_resp.get("stop_reason") or llm_resp.get("finish_reason"),
                        },
                        tokens_in=_safe_int(cast(dict, llm_resp.get("usage") or {}).get("prompt_tokens")),
                        tokens_out=_safe_int(cast(dict, llm_resp.get("usage") or {}).get("completion_tokens")),
                        latency_ms=_ms(step_t0),
                        timestamp=datetime.utcnow(),
                    )
                )

            # No tool calls -> done
            if not tool_calls:
                final_text = str(assistant_msg.get("content") or "").strip()
                if not final_text:
                    final_text = "I don’t have an answer yet."
                return _final_result(
                    conversation_id=convo_id,
                    final_text=final_text,
                    stop=stop,
                    trace=trace if trace_enabled else None,
                    tool_calls=tool_calls_made,
                    usage=usage_agg,
                )

            # IMPORTANT: append the assistant tool-call message BEFORE tool results
            # (common function-calling ordering expectation)
            messages.append(assistant_msg)

            # Execute tool calls (respect max_tool_calls)
            for tc in tool_calls:
                if tool_call_count >= self._policies.max_tool_calls:
                    stop = AgentStopReason(reason="max_tool_calls", detail=f"limit={self._policies.max_tool_calls}")
                    return _final_result(
                        conversation_id=convo_id,
                        final_text="I reached the tool-call limit while working on this. Please rephrase or try again.",
                        stop=stop,
                        trace=trace if trace_enabled else None,
                        tool_calls=tool_calls_made,
                        usage=usage_agg,
                    )

                tool_call_count += 1
                tool_calls_made.append(tc)

                tool_t0 = time.perf_counter()
                tool_result_model: ToolResult

                try:
                    out = await self._registry.call(tc.name, tc.arguments, ctx)
                    tool_result_model = ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        output=out,
                        error=None,
                        latency_ms=_ms(tool_t0),
                    )
                except ToolNotAllowedError as e:
                    stop = AgentStopReason(reason="blocked_tool", detail=str(e))
                    tool_result_model = ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        output=None,
                        error=str(e),
                        latency_ms=_ms(tool_t0),
                    )
                except ToolExecutionError as e:
                    stop = AgentStopReason(reason="tool_error", detail=str(e))
                    tool_result_model = ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        output=None,
                        error=str(e),
                        latency_ms=_ms(tool_t0),
                    )

                if trace_enabled:
                    trace.append(
                        AgentTraceStep(
                            step=step_idx,
                            type="tool",
                            input={"tool": tc.name, "arguments": tc.arguments},
                            output={
                                "output": _sanitize_tool_output(tool_result_model.output),
                                "error": tool_result_model.error,
                            },
                            tool_call=tc,
                            tool_result=tool_result_model,
                            latency_ms=tool_result_model.latency_ms,
                            timestamp=datetime.utcnow(),
                        )
                    )

                # Add tool result message for next LLM turn
                messages.append(_tool_result_to_message(tool_result_model))

                # A-version behavior preserved: stop early on tool failure
                if tool_result_model.error:
                    return _final_result(
                        conversation_id=convo_id,
                        final_text="I ran into an issue while using a tool and couldn’t complete the request.",
                        stop=stop,
                        trace=trace if trace_enabled else None,
                        tool_calls=tool_calls_made,
                        usage=usage_agg,
                    )

            # Continue loop; next LLM call will see assistant(tool_calls) + tool(results)

        # Hit max_steps
        stop = AgentStopReason(reason="max_steps", detail=f"limit={self._policies.max_steps}")
        if trace_enabled:
            trace.append(
                AgentTraceStep(
                    step=self._policies.max_steps,
                    type="stop",
                    input=None,
                    output={"reason": stop.reason, "detail": stop.detail},
                    latency_ms=None,
                    timestamp=datetime.utcnow(),
                )
            )
        return _final_result(
            conversation_id=convo_id,
            final_text="I reached the step limit while working on this. Please try a simpler question or add details.",
            stop=stop,
            trace=trace if trace_enabled else None,
            tool_calls=tool_calls_made,
            usage=usage_agg,
        )


# -----------------------------
# Helpers
# -----------------------------


def _final_result(
    *,
    conversation_id: str,
    final_text: str,
    stop: AgentStopReason,
    trace: Optional[List[AgentTraceStep]],
    tool_calls: List[ToolCall],
    usage: Dict[str, Any],
) -> AgentResult:
    return AgentResult(
        conversation_id=conversation_id,
        message=ChatMessage(role="assistant", content=final_text),
        stop=stop,
        trace=trace,
        tool_calls=tool_calls or None,
        usage=usage or None,
    )


def _chatmessage_to_dict(m: ChatMessage) -> JsonDict:
    d: JsonDict = {"role": m.role, "content": m.content}
    if getattr(m, "name", None):
        d["name"] = m.name
    if getattr(m, "tool_call_id", None):
        d["tool_call_id"] = m.tool_call_id
    if getattr(m, "tool_name", None):
        d["tool_name"] = m.tool_name
    if getattr(m, "metadata", None):
        d["metadata"] = m.metadata
    return d


def _tool_result_to_message(tr: ToolResult) -> JsonDict:
    # Tool messages should be role="tool" and linked with tool_call_id.
    content: Any = tr.output
    if not isinstance(content, str):
        try:
            content = json.dumps(content, ensure_ascii=False)
        except Exception:
            content = str(content)

    return {
        "role": "tool",
        "content": content,
        "tool_call_id": tr.tool_call_id,
        "name": tr.name,  # some APIs use "name" to indicate tool name
    }


def _extract_assistant_message_and_tool_calls(llm_resp: Any) -> Tuple[JsonDict, List[ToolCall]]:
    """
    Normalize different provider response shapes into:
      - assistant message dict {role, content, tool_calls?}
      - list[ToolCall]

    Supports:
      - OpenAI tool_calls:
          message.tool_calls = [{id,type,function:{name,arguments}}, ...]
      - Legacy OpenAI function_call:
          message.function_call = {name, arguments}
      - Direct message dicts or Message pydantic models
    """
    msg_any: Any = None

    if isinstance(llm_resp, Mapping):
        msg_any = llm_resp.get("message")

        # OpenAI-like non-streaming response: {"choices":[{"message":{...}}]}
        if msg_any is None and "choices" in llm_resp:
            try:
                msg_any = cast(list, llm_resp["choices"])[0].get("message")
            except Exception:
                msg_any = None

    if isinstance(msg_any, Message):
        msg: JsonDict = msg_any.model_dump(exclude_none=True)
    elif isinstance(msg_any, Mapping):
        msg = dict(msg_any)
    else:
        msg = {"role": "assistant", "content": str(llm_resp)}

    msg.setdefault("role", "assistant")

    tool_calls: List[ToolCall] = []

    # 1) OpenAI "tool_calls" list
    tool_calls_any = msg.get("tool_calls")
    if isinstance(tool_calls_any, list):
        for tcr_any in tool_calls_any:
            if not isinstance(tcr_any, Mapping):
                continue
            tcr = cast(Mapping[str, Any], tcr_any)

            try:
                tc_id = tcr.get("id") or str(uuid.uuid4())
                fn_any = tcr.get("function")
                fn = cast(Mapping[str, Any], fn_any) if isinstance(fn_any, Mapping) else {}

                name = fn.get("name") or tcr.get("name")
                args_raw = fn.get("arguments") if fn else tcr.get("arguments")
                args = _parse_json_args(args_raw)

                if name:
                    tool_calls.append(ToolCall(id=str(tc_id), name=str(name), arguments=args))
            except Exception:
                continue

    # 2) Legacy OpenAI "function_call" single call
    #    message.function_call = { "name": "...", "arguments": "..." }
    fn_call_any = msg.get("function_call")
    if not tool_calls and isinstance(fn_call_any, Mapping):
        try:
            name = fn_call_any.get("name")
            args = _parse_json_args(fn_call_any.get("arguments"))
            if name:
                tool_calls.append(ToolCall(id=str(uuid.uuid4()), name=str(name), arguments=args))
                # For continuity, mirror it into tool_calls field (optional)
                msg.setdefault("tool_calls", [])
        except Exception:
            pass

    return msg, tool_calls


def _parse_json_args(args_raw: Any) -> Dict[str, Any]:
    if args_raw is None:
        return {}
    if isinstance(args_raw, dict):
        return args_raw
    if isinstance(args_raw, str):
        s = args_raw.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            return {"_raw": s}
    return {"_raw": str(args_raw)}


def _merge_usage(dst: Dict[str, Any], usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = usage.get(k)
        if isinstance(v, int):
            dst[k] = int(dst.get(k, 0)) + v
    if dst.get("total_tokens", 0) == 0:
        pt = int(dst.get("prompt_tokens", 0) or 0)
        ct = int(dst.get("completion_tokens", 0) or 0)
        dst["total_tokens"] = pt + ct


def _sanitize_tool_output(out: Any, limit: int = 2000) -> Any:
    """
    Keep trace payloads small (avoid dumping huge results).
    (Does NOT change what is sent back to the LLM—only the trace.)
    """
    if out is None:
        return None
    if isinstance(out, (int, float, bool)):
        return out
    if isinstance(out, str):
        return out if len(out) <= limit else out[: limit - 1] + "…"
    try:
        s = json.dumps(out, ensure_ascii=False)
        if len(s) <= limit:
            return out
        return {"_truncated": True, "preview": s[: limit - 1] + "…"}
    except Exception:
        s = str(out)
        return s if len(s) <= limit else s[: limit - 1] + "…"


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 2)
