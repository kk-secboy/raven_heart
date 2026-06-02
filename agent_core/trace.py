"""Trace records for prompt, tool, and cost observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptTrace:
    run_id: str
    turn_id: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ToolTrace:
    run_id: str
    turn_id: str
    call_id: str
    tool_name: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CostTrace:
    run_id: str
    turn_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class CapabilityTrace:
    run_id: str = ""
    turn_id: str = ""
    actions: dict[str, Any] = field(default_factory=dict)
    tools: dict[str, Any] = field(default_factory=dict)
    skills: dict[str, Any] = field(default_factory=dict)
    mcp: dict[str, Any] = field(default_factory=dict)
    catalog: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-capability-trace/v1",
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "actions": self.actions,
            "tools": self.tools,
            "skills": self.skills,
            "mcp": self.mcp,
            "catalog": self.catalog,
        }

