"""
Agent policy definitions.

This module centralizes safety/limits knobs for the agent loop:
- max_steps
- max_tool_calls
- tool choice behavior (auto/none/required)
- allowlist (optional; normally enforced by ToolRegistry)

Defaults are read from app.core.config.settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from app.core.config import settings


ToolChoiceLiteral = Literal["auto", "none", "required"]


@dataclass(frozen=True)
class AgentPolicies:
    """
    Policies controlling the agent loop.

    Notes:
    - tool_choice is meant to map to your LLM client/tool-calling behavior:
        * "auto"     => model decides when to call tools
        * "none"     => disallow tool calling at the LLM layer (registry still exists)
        * "required" => force tool calling (rare; useful for strict RAG flows)
    - tools_allowlist is duplicated here for convenience; enforcement is done in ToolRegistry.
    """

    max_steps: int
    max_tool_calls: int
    tool_choice: ToolChoiceLiteral
    tools_allowlist: Optional[List[str]] = None
    debug_trace_default: bool = False

    @classmethod
    def from_settings(
        cls,
        *,
        tool_choice: ToolChoiceLiteral = "auto",
        tools_allowlist: Optional[List[str]] = None,
    ) -> "AgentPolicies":
        """
        Construct policies from Settings, with optional overrides.
        """
        allowlist = tools_allowlist
        if allowlist is None:
            # Prefer normalized list property if available
            allowlist = getattr(settings, "tools_allowlist_list", None) or None

        return cls(
            max_steps=int(getattr(settings, "AGENT_MAX_STEPS", 6)),
            max_tool_calls=int(getattr(settings, "AGENT_MAX_TOOL_CALLS", 8)),
            tool_choice=tool_choice,
            tools_allowlist=allowlist,
            debug_trace_default=bool(getattr(settings, "AGENT_DEBUG_TRACE", False)),
        )

    def clamp(self) -> "AgentPolicies":
        """
        Return a safe clamped copy (prevents accidental 0/negative settings).
        """
        ms = max(1, int(self.max_steps))
        mt = max(0, int(self.max_tool_calls))
        return AgentPolicies(
            max_steps=ms,
            max_tool_calls=mt,
            tool_choice=self.tool_choice,
            tools_allowlist=self.tools_allowlist,
            debug_trace_default=self.debug_trace_default,
        )
