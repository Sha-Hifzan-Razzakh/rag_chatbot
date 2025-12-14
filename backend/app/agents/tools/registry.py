"""
Tool registry for the agentic orchestration layer.

Responsibilities:
- Register tool handlers with metadata (name/description/JSON schema params)
- Produce tool specs for LLM function calling
- Enforce allowlist and dispatch tool calls
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set, TypedDict, Union


JsonDict = Dict[str, Any]
ToolHandler = Union[
    Callable[[JsonDict, "ToolContext"], Any],
    Callable[[JsonDict, "ToolContext"], Awaitable[Any]],
]


class ToolContext(TypedDict, total=False):
    """
    Context passed to tools at runtime.

    Keep this lightweight and JSON-serializable where possible.
    Orchestrator can add fields like:
      - conversation_id: str
      - request_id: str
      - user_id: str
      - logger: logging.Logger
      - rag: prebuilt RAG dependencies (retriever, embedder, etc.)
      - settings: settings object/dict
    """
    conversation_id: str
    request_id: str
    user_id: str
    logger: Any
    settings: Any
    rag: Any


@dataclass(frozen=True)
class ToolSpec:
    """
    Metadata describing a tool for function calling.
    """
    name: str
    description: str
    parameters: JsonDict  # JSON Schema object


@dataclass
class ToolRegistration:
    spec: ToolSpec
    handler: ToolHandler


class ToolNotFoundError(Exception):
    pass


class ToolNotAllowedError(Exception):
    pass


class ToolExecutionError(Exception):
    pass


class ToolRegistry:
    def __init__(self, allowlist: Optional[Sequence[str]] = None) -> None:
        # allowlist semantics:
        # - None: read from settings if available; else allow all
        # - ["*"] or ["ALL"]: allow all
        # - otherwise: exact tool-name match
        self._tools: Dict[str, ToolRegistration] = {}
        self._explicit_allowlist = list(allowlist) if allowlist is not None else None

    # ---------------------------
    # Registration
    # ---------------------------

    def register(
        self,
        *,
        name: str,
        description: str,
        parameters: JsonDict,
        handler: ToolHandler,
        overwrite: bool = False,
    ) -> None:
        if not overwrite and name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")
        self._tools[name] = ToolRegistration(
            spec=ToolSpec(name=name, description=description, parameters=parameters),
            handler=handler,
        )

    def names(self) -> List[str]:
        return sorted(self._tools.keys())

    # ---------------------------
    # Tool specs for LLM
    # ---------------------------

    def list_tool_specs(self) -> List[JsonDict]:
        """
        Return OpenAI-style function/tool specs:
        [
          {
            "type": "function",
            "function": {
              "name": "...",
              "description": "...",
              "parameters": {...json schema...}
            }
          },
          ...
        ]
        """
        allowed = self._compute_allowset()
        specs: List[JsonDict] = []
        for name in self.names():
            if allowed is not None and name not in allowed:
                continue
            reg = self._tools[name]
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": reg.spec.name,
                        "description": reg.spec.description,
                        "parameters": reg.spec.parameters,
                    },
                }
            )
        return specs

    # ---------------------------
    # Dispatch
    # ---------------------------

    async def call(self, tool_name: str, args: JsonDict, context: ToolContext) -> Any:
        """
        Execute a tool by name with args + context.

        Raises:
          ToolNotFoundError, ToolNotAllowedError, ToolExecutionError
        """
        reg = self._tools.get(tool_name)
        if reg is None:
            raise ToolNotFoundError(f"Unknown tool '{tool_name}'")

        allowed = self._compute_allowset(context=context)
        if allowed is not None and tool_name not in allowed:
            raise ToolNotAllowedError(f"Tool '{tool_name}' is not allowed")

        logger = context.get("logger")
        t0 = time.perf_counter()
        try:
            result = reg.handler(args, context)  # may be awaitable
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except ToolNotAllowedError:
            raise
        except ToolNotFoundError:
            raise
        except Exception as e:
            raise ToolExecutionError(f"Tool '{tool_name}' failed: {e}") from e
        finally:
            if logger:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                try:
                    logger.info(
                        "tool_call",
                        extra={
                            "tool_name": tool_name,
                            "latency_ms": round(dt_ms, 2),
                            "conversation_id": context.get("conversation_id"),
                            "request_id": context.get("request_id"),
                        },
                    )
                except Exception:
                    # never let logging break tool execution
                    pass

    # ---------------------------
    # Allowlist handling
    # ---------------------------

    def _compute_allowset(self, context: Optional[ToolContext] = None) -> Optional[Set[str]]:
        """
        Returns:
          - None => allow all registered tools
          - Set[str] => explicit allowlist
        """
        allowlist = self._explicit_allowlist

        # If not explicitly provided, attempt to read from settings.
        if allowlist is None and context is not None:
            settings = context.get("settings")
            allowlist = _read_allowlist_from_settings(settings)

        if allowlist is None:
            # default: allow all
            return None

        normalized = [str(x).strip() for x in allowlist if str(x).strip()]
        if not normalized:
            return None

        if any(x in ("*", "ALL") for x in normalized):
            return None

        return set(normalized)


def _read_allowlist_from_settings(settings: Any) -> Optional[List[str]]:
    """
    Tries to read TOOLS_ALLOWLIST from a typical settings object.
    Supports:
      - attribute access: settings.TOOLS_ALLOWLIST
      - dict access: settings["TOOLS_ALLOWLIST"]
    """
    if settings is None:
        return None
    # attribute
    if hasattr(settings, "TOOLS_ALLOWLIST"):
        value = getattr(settings, "TOOLS_ALLOWLIST")
        return _coerce_allowlist(value)
    # dict-like
    if isinstance(settings, dict) and "TOOLS_ALLOWLIST" in settings:
        return _coerce_allowlist(settings.get("TOOLS_ALLOWLIST"))
    return None


def _coerce_allowlist(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        # allow comma-separated: "search_docs,answer_with_rag"
        parts = [p.strip() for p in value.split(",")]
        return [p for p in parts if p]
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    # unknown type -> ignore
    return None
