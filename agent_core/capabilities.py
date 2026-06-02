"""Unified capability catalog for prompt injection and replay audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.actions import ActionRegistry
from agent_core.mcp import MCPCenter
from agent_core.skills import SkillsContext
from agent_core.tools import ToolRuntimePort


@dataclass(frozen=True)
class CapabilityCatalog:
    actions: ActionRegistry | None = None
    tools: ToolRuntimePort | None = None
    skills: SkillsContext | None = None
    mcp: MCPCenter | None = None
    tool_token_budget: int = 1600
    skill_token_budget: int = 1200
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-capability-catalog/v1",
            "actions": self.actions.manifest() if self.actions is not None else {},
            "tools": _tool_manifest(self.tools),
            "skills": self.skills.snapshot() if self.skills is not None else {},
            "mcp": self.mcp.manifest() if self.mcp is not None else {},
            "metadata": self.metadata,
        }

    def render_prompt(self, *, include_skills: bool = False) -> str:
        parts: list[str] = []
        if self.actions is not None:
            actions = self.actions.render_actions()
            if actions:
                parts.append("[action_inventory]\n" + actions)
        if self.tools is not None:
            inventory = _render_tool_inventory(self.tools, max_tokens=self.tool_token_budget)
            if inventory:
                parts.append("[tool_inventory]\n" + inventory)
        if self.mcp is not None:
            mcp_servers = self._render_mcp_servers()
            if mcp_servers:
                parts.append("[mcp_servers]\n" + mcp_servers)
        if include_skills and self.skills is not None:
            skills = self.skills.render_stable(available_token_budget=self.skill_token_budget)
            if skills:
                parts.append("[skills_context]\n" + skills)
        return "\n\n".join(parts)

    def _render_mcp_servers(self) -> str:
        if self.mcp is None:
            return ""
        lines = []
        for server in self.mcp.servers(include_disabled=True):
            state = self.mcp.state(server.name)
            status = state.status if state is not None else "registered"
            tool_count = state.tool_count if state is not None else 0
            suffix = f" error={state.last_error}" if state is not None and state.last_error else ""
            lines.append(
                f"- {server.name} transport={server.transport} "
                f"status={status} tools={tool_count}{suffix}"
            )
        return "\n".join(lines)


def _render_tool_inventory(tools: ToolRuntimePort, *, max_tokens: int) -> str:
    render = getattr(tools, "render_inventory", None)
    if callable(render):
        return str(render(max_tokens=max_tokens))
    specs = tools.specs()
    return "\n".join(f"- {spec.name}: {spec.description}".rstrip() for spec in specs)


def _tool_manifest(tools: ToolRuntimePort | None) -> dict[str, Any]:
    if tools is None:
        return {}
    manifest = getattr(tools, "manifest", None)
    if callable(manifest):
        return dict(manifest())
    return {
        "schema_version": "agent-core-tool-runtime/v1",
        "tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "aliases": list(spec.aliases),
                "tags": list(spec.tags),
                "enabled": spec.enabled,
                "metadata": spec.metadata,
            }
            for spec in tools.specs()
        ],
    }

