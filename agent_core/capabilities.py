"""Unified capability catalog for prompt injection and replay audit."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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


_FIXED_INVENTORY_TOOL_PRIORITY = (
    "search_capabilities",
    "search_tools",
    "load_capability",
    "load_skill_resource",
    "query_mcp_servers",
    "query_mcp_tools",
    "read_file",
    "write_file",
    "modify_file",
    "find_file",
    "grep",
    "tree",
    "bash",
    "cmd",
    "exec",
    "web_search",
    "do_http_request",
    "batch_do_http_request",
)

_FIXED_INVENTORY_ACTION_PRIORITY = (
    "finish",
    "call_tool",
    "search_capabilities",
    "search_tools",
    "load_capability",
    "query_mcp_tools",
)


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
class CapabilityRecallRequest:
    """Internal turn-time capability recall request."""

    query: str = ""
    limit: int = 12
    perception_terms: tuple[str, ...] = ()
    recent_tools: tuple[str, ...] = ()
    failed_tools: tuple[str, ...] = ()
    scenario_whitelist: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_query(self) -> str:
        parts = [self.query, " ".join(self.perception_terms)]
        return " ".join(part for part in parts if str(part).strip()).strip()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-capability-recall-request/v1",
            "query": self.query,
            "limit": self.limit,
            "perception_terms": list(self.perception_terms),
            "recent_tools": list(self.recent_tools),
            "failed_tools": list(self.failed_tools),
            "scenario_whitelist": list(self.scenario_whitelist),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CapabilityRecallResult:
    """Turn-time capability recommendations suitable for prompt injection."""

    request: CapabilityRecallRequest
    matches: tuple[CapabilityMatch, ...] = ()
    omitted_count: int = 0
    failed_tools: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for match in self.matches:
            counts[match.kind] = counts.get(match.kind, 0) + 1
        return {
            "schema_version": "agent-core-capability-discovery/v1",
            "enabled": True,
            "recall_schema_version": "agent-core-capability-recall/v1",
            "request": self.request.manifest(),
            "query": {
                "query": self.request.normalized_query(),
                "limit": self.request.limit,
            },
            "match_count": len(self.matches),
            "omitted_count": self.omitted_count,
            "counts": counts,
            "failed_tools": list(self.failed_tools),
            "matches": [match.manifest() for match in self.matches],
            "metadata": dict(self.metadata),
        }

    def render_prompt(self) -> str:
        lines = ["[capability_recall]"]
        if self.matches:
            lines.append("Recommended capabilities for the next turn:")
            for match in self.matches:
                tags = f" tags={','.join(match.tags)}" if match.tags else ""
                source = f" source={match.source}" if match.source else ""
                lines.append(
                    f"- {match.kind}:{match.name}{source}{tags}: {match.description}".rstrip()
                )
                hint = _capability_use_hint(match)
                if hint:
                    lines.append(f"  use: {hint}")
        else:
            lines.append("(no targeted capability matches)")
        if self.failed_tools:
            lines.append("Recent tool failures to route around:")
            for tool_name in self.failed_tools:
                lines.append(f"- {tool_name}")
        if self.omitted_count:
            lines.append(
                f"{self.omitted_count} additional matches omitted; use capability search if needed."
            )
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

    def recall(self, request: CapabilityRecallRequest | str) -> CapabilityRecallResult:
        recall_request = (
            request if isinstance(request, CapabilityRecallRequest) else CapabilityRecallRequest(query=str(request))
        )
        normalized_query = recall_request.normalized_query()
        discovery = self.discover(
            CapabilityQuery(
                query=normalized_query,
                limit=max(recall_request.limit * 3, recall_request.limit, 1),
            )
        )
        whitelist = {
            item.casefold()
            for item in recall_request.scenario_whitelist
            if str(item).strip()
        }
        failed = {
            item.casefold()
            for item in recall_request.failed_tools
            if str(item).strip()
        }
        recent = {
            item.casefold()
            for item in recall_request.recent_tools
            if str(item).strip()
        }
        scored: list[tuple[float, str, CapabilityMatch]] = []
        for index, match in enumerate(discovery.matches):
            if whitelist and match.kind in {"tool", "mcp_tool"} and match.name.casefold() not in whitelist:
                continue
            score = float(match.score or 0.0) + 1.0 / (index + 1)
            if match.name.casefold() in recent:
                score += 0.2
            if match.name.casefold() in failed:
                score -= 0.35
            boosted = replace(match, score=round(score, 6))
            scored.append((score, boosted.name, boosted))
        scored.sort(key=lambda item: (-item[0], item[1]))
        limit = max(0, int(recall_request.limit))
        visible = tuple(item for _, _, item in scored[:limit]) if limit else ()
        return CapabilityRecallResult(
            request=recall_request,
            matches=visible,
            omitted_count=max(0, len(scored) - len(visible) + discovery.omitted_count),
            failed_tools=tuple(recall_request.failed_tools),
            metadata={
                "strategy": "bm25_keyword_recent_failed_reroute",
                "catalog": dict(self.metadata),
                "discovery": discovery.manifest(),
                "recent_tools": list(recall_request.recent_tools),
                "failed_tools": list(recall_request.failed_tools),
                "scenario_whitelist": list(recall_request.scenario_whitelist),
            },
        )

    def render_prompt(self, *, include_skills: bool = False) -> str:
        parts: list[str] = []
        if self.actions is not None:
            actions = _render_fixed_action_inventory(self.actions, max_actions=6)
            if actions:
                parts.append("[action_inventory]\n" + actions)
        if self.tools is not None:
            inventory = _render_fixed_tool_inventory(self.tools, max_tokens=self.tool_token_budget)
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


def _render_fixed_action_inventory(actions: ActionRegistry, *, max_actions: int) -> str:
    specs = tuple(actions.specs())
    if not specs:
        return ""
    priority = {name.casefold(): index for index, name in enumerate(_FIXED_INVENTORY_ACTION_PRIORITY)}
    fixed = [spec for spec in specs if spec.name.casefold() in priority]
    if fixed:
        ordered = sorted(fixed, key=lambda item: (priority[item.name.casefold()], item.name.casefold()))
    else:
        ordered = sorted(specs, key=lambda item: item.name.casefold())
    visible = ordered[: max(1, int(max_actions))]
    omitted = max(0, len(specs) - len(visible))
    lines = [_format_fixed_action_inventory_line(spec) for spec in visible]
    if omitted:
        lines.append(f"... and {omitted} more actions. Use search_capabilities/search_tools if needed.")
    return "\n".join(lines)


def _format_fixed_action_inventory_line(spec: ActionSpec) -> str:
    terminal = " terminal=true" if spec.terminal else ""
    return f"- {spec.name}{terminal}: {spec.description}".rstrip()


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


def _render_fixed_tool_inventory(tools: ToolRuntimePort, *, max_tokens: int) -> str:
    specs = tuple(spec for spec in tools.specs() if spec.enabled)
    if not specs:
        return ""
    fixed_names = {name.casefold(): index for index, name in enumerate(_FIXED_INVENTORY_TOOL_PRIORITY)}
    fixed: list[ToolSpec] = []
    remaining: list[ToolSpec] = []
    for spec in specs:
        if _is_fixed_inventory_tool(spec, fixed_names):
            fixed.append(spec)
        else:
            remaining.append(spec)
    fixed_sorted = sorted(
        fixed,
        key=lambda item: (
            fixed_names.get(item.name.casefold(), len(fixed_names)),
            item.name.casefold(),
        ),
    )
    if fixed_sorted:
        ordered = tuple(fixed_sorted)
        dynamic_omitted = len(remaining)
    elif len(specs) <= 20:
        ordered = tuple(sorted(remaining, key=lambda item: item.name.casefold()))
        dynamic_omitted = 0
    else:
        ordered = ()
        dynamic_omitted = len(remaining)
    lines: list[str] = []
    used = 0
    budget = max(0, int(max_tokens)) * 4
    omitted = dynamic_omitted
    for spec in ordered:
        line = _format_fixed_tool_inventory_line(spec)
        line_bytes = len((line + "\n").encode("utf-8"))
        if lines and budget and used + line_bytes > budget:
            omitted += 1
            continue
        if not lines and budget and line_bytes > budget:
            lines.append(line)
            used += line_bytes
            omitted += max(0, len(ordered) - 1)
            break
        lines.append(line)
        used += line_bytes
    if omitted:
        lines.append(
            f"... and {omitted} more tools. Use capability_recall/search_capabilities for dynamic tools."
        )
    return "\n".join(lines)


def _is_fixed_inventory_tool(spec: ToolSpec, priority_names: dict[str, int]) -> bool:
    if spec.name.casefold() in priority_names:
        return True
    metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
    if bool(metadata.get("fixed_inventory") or metadata.get("core_inventory")):
        return True
    tags = {str(tag).casefold() for tag in spec.tags}
    return bool(tags.intersection({"core", "fixed", "stable", "base"}))


def _format_fixed_tool_inventory_line(spec: ToolSpec) -> str:
    tags = f" tags={','.join(spec.tags)}" if spec.tags else ""
    aliases = f" aliases={','.join(spec.aliases)}" if spec.aliases else ""
    description = f": {spec.description}" if spec.description else ""
    return f"- {spec.name}{tags}{aliases}{description}".rstrip()


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


def _capability_use_hint(match: CapabilityMatch) -> str:
    if match.kind in {"tool", "mcp_tool"}:
        return f"call_tool with tool_name={match.name}"
    if match.kind == "skill":
        return f"load_capability or load_skill name={match.name}"
    if match.kind == "mcp_server":
        return f"query_mcp_tools after refreshing server={match.name}"
    if match.kind in {"mcp_resource", "mcp_prompt"}:
        return "request MCP context material or knowledge recall for this item"
    if match.kind == "action":
        return f"emit action {match.name}"
    return ""

