"""
System / utility tools for the agent orchestrator.

These are optional but very useful for:
- debugging the tool loop
- writing deterministic unit tests for the agent
- basic health/time checks without touching business logic
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from app.agents.tools.registry import ToolContext, ToolRegistry

JsonDict = Dict[str, Any]


def register_system_tools(registry: ToolRegistry) -> None:
    """
    Register system tools into the ToolRegistry.
    """

    registry.register(
        name="health",
        description="Return a simple health payload indicating the service is running.",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        handler=_tool_health,
    )

    registry.register(
        name="time_now",
        description="Return the current server time in ISO 8601 UTC format.",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        handler=_tool_time_now,
    )

    registry.register(
        name="debug_echo",
        description="Echo back the provided arguments for debugging and tests.",
        parameters={
            "type": "object",
            "properties": {
                "value": {"description": "Any JSON value to echo back."},
            },
            "required": ["value"],
            "additionalProperties": True,
        },
        handler=_tool_debug_echo,
    )


async def _tool_health(args: JsonDict, context: ToolContext) -> JsonDict:
    # If caller put extra info into context, include safe bits.
    conv_id = context.get("conversation_id")
    req_id = context.get("request_id")
    return {
        "status": "ok",
        "service": "backend",
        "conversation_id": conv_id,
        "request_id": req_id,
    }


async def _tool_time_now(args: JsonDict, context: ToolContext) -> JsonDict:
    now = datetime.now(timezone.utc).isoformat()
    return {"utc_now": now}


async def _tool_debug_echo(args: JsonDict, context: ToolContext) -> JsonDict:
    return {"echo": args.get("value")}
