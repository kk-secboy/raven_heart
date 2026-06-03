"""Unified capability catalog for prompt injection and replay audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agent_core.actions import ActionRegistry, ActionSpec
from agent_core.mcp import MCPCenter, MCPPromptSpec, MCPResourceSpec
from agent_core.search import SearchDocument, rank_documents
from agent_core.skills import SkillsContext
from agent_core.tools import ToolRuntimePort, ToolSpec, tool_manifest_item


CapabilityKind = Literal[
    "action",
    "tool",
    "skill",
    "mcp_tool",
    "mcp_resource",
    "mcp_prompt",
    "mcp_server",
]


@dataclass(frozen=True)
class CapabilityQuery:
    query: str = ""
    limit: int = 12
    tags: tuple[str, ...] = ()
    include_actions: bool = True
    include_tools: bool = True
    include_skills: bool = True
    include_mcp: bool = True
    include_mcp_resources: bool = True
    include_mcp_prompts: bool = True

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-capability-query/v1",
            "query": self.query,
            "limit": self.limit,
            "tags": list(self.tags),
            "include_actions": self.include_actions,
            "include_tools": self.include_tools,
            "include_skills": self.include_skills,
            "include_mcp": self.include_mcp,
            "include_mcp_resources": self.include_mcp_resources,
            "include_mcp_prompts": self.include_mcp_prompts,
        }


@dataclass(frozen=True)
class CapabilityMatch:
    kind: CapabilityKind
    name: str
    description: str = ""
    source: str = ""
    score: float = 0.0
    tags: tuple[str, ...] = ()
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "score": self.score,
            "tags": list(self.tags),
            "enabled": self.enabled,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CapabilityDiscoveryResult:
    query: CapabilityQuery
    matches: tuple[CapabilityMatch, ...] = ()
    omitted_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for match in self.matches:
            counts[match.kind] = counts.get(match.kind, 0) + 1
        return {
            "schema_version": "agent-core-capability-discovery/v1",
            "query": self.query.manifest(),
            "match_count": len(self.matches),
            "omitted_count": self.omitted_count,
            "counts": counts,
            "matches": [match.manifest() for match in self.matches],
            "metadata": dict(self.metadata),
        }

    def render_prompt(self) -> str:
        lines = ["[capability_discovery]"]
        if not self.matches:
            lines.append("  (none)")
            return "\n".join(lines)
        for match in self.matches:
            source = f" source={match.source}" if match.source else ""
            tags = f" tags={','.join(match.tags)}" if match.tags else ""
            lines.append(
                f"- {match.kind}:{match.name}{source}{tags}: {match.description}".rstrip()
            )
        if self.omitted_count:
            lines.append(f"  ... and {self.omitted_count} more matches.")
        return "\n".join(lines)


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

    def discover(
        self,
        query: CapabilityQuery | str,
        *,
        limit: int | None = None,
    ) -> CapabilityDiscoveryResult:
        request = query if isinstance(query, CapabilityQuery) else CapabilityQuery(query=str(query))
        if limit is not None:
            request = CapabilityQuery(
                query=request.query,
                limit=limit,
                tags=request.tags,
                include_actions=request.include_actions,
                include_tools=request.include_tools,
                include_skills=request.include_skills,
                include_mcp=request.include_mcp,
                include_mcp_resources=request.include_mcp_resources,
                include_mcp_prompts=request.include_mcp_prompts,
            )
        matches: list[CapabilityMatch] = []
        if request.include_actions and self.actions is not None:
            matches.extend(_action_matches(self.actions, request))
        if request.include_tools and self.tools is not None:
            matches.extend(_tool_matches(self.tools, request, kind="tool", source="tools"))
        if request.include_skills and self.skills is not None:
            matches.extend(_skill_matches(self.skills, request))
        if request.include_mcp and self.mcp is not None:
            matches.extend(_mcp_matches(self.mcp, request))
        matches = sorted(
            matches,
            key=lambda item: (-item.score, item.kind, item.name),
        )
        limit_value = max(0, int(request.limit))
        visible = tuple(matches[:limit_value]) if limit_value else ()
        return CapabilityDiscoveryResult(
            query=request,
            matches=visible,
            omitted_count=max(0, len(matches) - len(visible)),
            metadata={"catalog": dict(self.metadata)},
        )

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


def _action_matches(actions: ActionRegistry, query: CapabilityQuery) -> tuple[CapabilityMatch, ...]:
    ranked = _rank_items(
        actions.specs(),
        query=query,
        text=lambda item: " ".join((item.name, item.description)),
        name=lambda item: item.name,
    )
    return tuple(
        CapabilityMatch(
            kind="action",
            name=spec.name,
            description=spec.description,
            score=_score(index),
            enabled=True,
            metadata=_action_manifest(spec),
        )
        for index, spec in enumerate(ranked)
    )


def _tool_matches(
    tools: ToolRuntimePort,
    query: CapabilityQuery,
    *,
    kind: CapabilityKind,
    source: str,
) -> tuple[CapabilityMatch, ...]:
    specs = tuple(spec for spec in tools.specs() if spec.enabled and _tags_match(spec.tags, query.tags))
    ranked = _rank_items(
        specs,
        query=query,
        text=lambda item: " ".join((item.name, item.description, " ".join(item.tags))),
        name=lambda item: item.name,
    )
    return tuple(
        CapabilityMatch(
            kind=kind,
            name=spec.name,
            description=spec.description,
            source=_tool_source(spec, source),
            score=_score(index),
            tags=spec.tags,
            enabled=spec.enabled,
            metadata=tool_manifest_item(spec),
        )
        for index, spec in enumerate(ranked)
    )


def _skill_matches(skills: SkillsContext, query: CapabilityQuery) -> tuple[CapabilityMatch, ...]:
    candidates = tuple(
        skill for skill in skills.registry.list() if _tags_match(skill.tags, query.tags)
    )
    ranked = _rank_items(
        candidates,
        query=query,
        text=lambda item: " ".join((item.name, item.description, " ".join(item.tags), item.compatibility)),
        name=lambda item: item.name,
    )
    loaded = {skill.name for skill in skills.loaded()}
    return tuple(
        CapabilityMatch(
            kind="skill",
            name=skill.name,
            description=skill.description,
            source="skills",
            score=_score(index),
            tags=skill.tags,
            enabled=not skill.disable_model_invocation,
            metadata={**skill.manifest(), "loaded": skill.name in loaded},
        )
        for index, skill in enumerate(ranked)
    )


def _mcp_matches(mcp: MCPCenter, query: CapabilityQuery) -> tuple[CapabilityMatch, ...]:
    matches: list[CapabilityMatch] = []
    matches.extend(_tool_matches(mcp, query, kind="mcp_tool", source="mcp"))
    matches.extend(_mcp_server_matches(mcp, query))
    if query.include_mcp_resources:
        matches.extend(_mcp_resource_matches(mcp.resources(), query))
    if query.include_mcp_prompts:
        matches.extend(_mcp_prompt_matches(mcp.prompts(), query))
    return tuple(matches)


def _mcp_server_matches(mcp: MCPCenter, query: CapabilityQuery) -> tuple[CapabilityMatch, ...]:
    servers = tuple(server for server in mcp.servers(include_disabled=True) if _tags_match(server.tags, query.tags))
    ranked = _rank_items(
        servers,
        query=query,
        text=lambda item: " ".join((item.name, item.transport, " ".join(item.tags), str(item.metadata))),
        name=lambda item: item.name,
    )
    matches = []
    for index, server in enumerate(ranked):
        state = mcp.state(server.name)
        matches.append(
            CapabilityMatch(
                kind="mcp_server",
                name=server.name,
                description=f"MCP server transport={server.transport}",
                source="mcp",
                score=_score(index),
                tags=server.tags,
                enabled=server.enabled,
                metadata={
                    "transport": server.transport,
                    "enabled": server.enabled,
                    "state": state.manifest() if state is not None else {},
                    "metadata": dict(server.metadata),
                },
            )
        )
    return tuple(matches)


def _mcp_resource_matches(resources: tuple[MCPResourceSpec, ...], query: CapabilityQuery) -> tuple[CapabilityMatch, ...]:
    candidates = tuple(resource for resource in resources if _tags_match(resource.tags, query.tags))
    ranked = _rank_items(
        candidates,
        query=query,
        text=lambda item: " ".join(
            (item.uri, item.name, item.description, item.mime_type, " ".join(item.tags))
        ),
        name=lambda item: item.uri,
    )
    return tuple(
        CapabilityMatch(
            kind="mcp_resource",
            name=resource.uri,
            description=resource.description or resource.name,
            source=resource.server_name,
            score=_score(index),
            tags=resource.tags,
            metadata=resource.manifest(),
        )
        for index, resource in enumerate(ranked)
    )


def _mcp_prompt_matches(prompts: tuple[MCPPromptSpec, ...], query: CapabilityQuery) -> tuple[CapabilityMatch, ...]:
    candidates = tuple(prompt for prompt in prompts if _tags_match(prompt.tags, query.tags))
    ranked = _rank_items(
        candidates,
        query=query,
        text=lambda item: " ".join((item.name, item.description, " ".join(item.tags), item.server_name)),
        name=lambda item: item.prompt_id(),
    )
    return tuple(
        CapabilityMatch(
            kind="mcp_prompt",
            name=prompt.prompt_id(),
            description=prompt.description,
            source=prompt.server_name,
            score=_score(index),
            tags=prompt.tags,
            metadata=prompt.manifest(),
        )
        for index, prompt in enumerate(ranked)
    )


def _rank_items(
    items: tuple[Any, ...],
    *,
    query: CapabilityQuery,
    text: Any,
    name: Any,
) -> tuple[Any, ...]:
    if not query.query.strip():
        return items
    documents = tuple(
        SearchDocument(
            item=item,
            text=str(text(item)),
            name=str(name(item)),
        )
        for item in items
    )
    return tuple(rank_documents(query.query, documents, limit=max(len(items), query.limit)))


def _tags_match(item_tags: tuple[str, ...], requested_tags: tuple[str, ...]) -> bool:
    if not requested_tags:
        return True
    own_tags = {tag.casefold() for tag in item_tags}
    return all(tag.casefold() in own_tags for tag in requested_tags)


def _score(index: int) -> float:
    return 1.0 / (index + 1)


def _action_manifest(spec: ActionSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "parameters_schema": spec.parameters_schema,
        "terminal": spec.terminal,
    }


def _tool_source(spec: ToolSpec, default: str) -> str:
    tool_center = spec.metadata.get("tool_center") if isinstance(spec.metadata, dict) else None
    if isinstance(tool_center, dict):
        return str(tool_center.get("mount") or default)
    mcp = spec.metadata.get("mcp") if isinstance(spec.metadata, dict) else None
    if isinstance(mcp, dict):
        return str(mcp.get("server_name") or default)
    return default


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

