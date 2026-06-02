"""Tool protocol types for the agent core."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from agent_core.prompt import estimate_tokens
from agent_core.search import SearchDocument, rank_documents


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ""
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolInvocation:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: dict[str, Any] = field(default_factory=dict)

    def replay_key(self) -> str:
        payload = {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "metadata_replay_scope": self.metadata.get("replay_scope", ""),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool_name: str
    status: str = "completed"
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "completed" and not self.error

    @property
    def content_bytes(self) -> int:
        return len(self.content.encode("utf-8"))

    def compact(self, max_content_bytes: int) -> "ToolResult":
        """Return a prompt-safe result while preserving provenance metadata."""

        if max_content_bytes <= 0 or self.content_bytes <= max_content_bytes:
            return self
        raw = self.content.encode("utf-8")
        keep = max(1, max_content_bytes)
        prefix = raw[: keep // 2].decode("utf-8", errors="ignore")
        suffix = raw[-(keep - len(prefix.encode("utf-8"))):].decode("utf-8", errors="ignore")
        notice = (
            f"\n\n[tool_result_compacted original_bytes={len(raw)} "
            f"sha256={hashlib.sha256(raw).hexdigest()}]"
        )
        compacted = f"{prefix}\n...[truncated]...\n{suffix}{notice}"
        metadata = {
            **self.metadata,
            "compacted": True,
            "original_content_bytes": len(raw),
            "original_content_sha256": hashlib.sha256(raw).hexdigest(),
        }
        return ToolResult(
            call_id=self.call_id,
            tool_name=self.tool_name,
            status=self.status,
            content=compacted,
            data=dict(self.data),
            error=self.error,
            metadata=metadata,
        )


@dataclass(frozen=True)
class ToolInventorySelection:
    visible_tools: tuple[ToolSpec, ...]
    omitted_count: int = 0

    @property
    def has_more(self) -> bool:
        return self.omitted_count > 0

    def manifest(self) -> dict[str, Any]:
        return {
            "visible_tools": [tool_manifest_item(spec) for spec in self.visible_tools],
            "omitted_count": self.omitted_count,
            "has_more": self.has_more,
        }


@dataclass(frozen=True)
class ToolRuntimeMount:
    name: str
    runtime: "ToolRuntimePort"
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


class ToolRuntimePort(Protocol):
    async def invoke(self, invocation: ToolInvocation) -> ToolResult:
        """Invoke one tool through the runtime-specific boundary."""

    def specs(self) -> tuple[ToolSpec, ...]:
        """Return visible tools."""
        return ()


ToolHandler = Callable[[ToolInvocation], Awaitable[ToolResult]]
ToolFunction = Callable[..., Any]


class ToolRegistry(ToolRuntimePort):
    """In-process tool registry for core tests and lightweight agents."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}
        self._aliases: dict[str, str] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        name = spec.name.strip()
        if not name:
            raise ValueError("tool name is required")
        if name in self._specs:
            raise ValueError(f"tool already registered: {name}")
        self._specs[name] = spec
        self._handlers[name] = handler
        for alias in spec.aliases:
            normalized = alias.strip()
            if normalized:
                self._aliases[normalized] = name

    def register_function(
        self,
        func: ToolFunction | None = None,
        *,
        name: str | None = None,
        description: str = "",
        aliases: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        enabled: bool = True,
    ) -> ToolFunction:
        """Register a Python function as a tool.

        The generated schema is intentionally small and deterministic. Runtime
        adapters can replace it with richer schemas later.
        """

        def decorator(inner: ToolFunction) -> ToolFunction:
            spec = ToolSpec(
                name=name or inner.__name__,
                description=description or (inspect.getdoc(inner) or ""),
                parameters_schema=schema_from_callable(inner),
                aliases=aliases,
                tags=tags,
                enabled=enabled,
            )

            async def handler(invocation: ToolInvocation) -> ToolResult:
                try:
                    value = inner(**invocation.arguments)
                    if inspect.isawaitable(value):
                        value = await value
                except Exception as exc:
                    return ToolResult(
                        call_id=invocation.call_id,
                        tool_name=invocation.tool_name,
                        status="failed",
                        error=str(exc),
                    )
                content = value if isinstance(value, str) else json.dumps(value, default=str)
                return ToolResult(
                    call_id=invocation.call_id,
                    tool_name=invocation.tool_name,
                    content=content,
                    data={"return": value} if not isinstance(value, str) else {},
                )

            self.register(spec, handler)
            return inner

        if func is not None:
            return decorator(func)
        return decorator

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(self.resolve_name(name))

    def resolve_name(self, name: str) -> str:
        return self._aliases.get(name, name)

    def specs(self, *, tags: tuple[str, ...] = (), include_disabled: bool = False) -> tuple[ToolSpec, ...]:
        specs = []
        requested_tags = {tag.lower() for tag in tags}
        for spec in self._specs.values():
            if not include_disabled and not spec.enabled:
                continue
            if requested_tags:
                own_tags = {tag.lower() for tag in spec.tags}
                if not requested_tags.intersection(own_tags):
                    continue
            specs.append(spec)
        return tuple(specs)

    def search(self, query: str, *, tags: tuple[str, ...] = (), limit: int = 8) -> tuple[ToolSpec, ...]:
        documents = tuple(
            SearchDocument(
                item=spec,
                text=" ".join(
                    (
                        spec.name,
                        spec.description,
                        " ".join(spec.aliases),
                        " ".join(spec.tags),
                    )
                ),
                name=spec.name,
            )
            for spec in self.specs(tags=tags)
        )
        return rank_documents(query, documents, limit=limit)

    def select_inventory(
        self,
        *,
        tags: tuple[str, ...] = (),
        max_tokens: int = 1600,
    ) -> ToolInventorySelection:
        visible: list[ToolSpec] = []
        used = 0
        specs = tuple(sorted(self.specs(tags=tags), key=lambda item: item.name))
        for spec in specs:
            line = _format_tool_inventory_line(spec) + "\n"
            line_tokens = estimate_tokens(line)
            if visible and used + line_tokens > max_tokens:
                break
            if not visible and line_tokens > max_tokens:
                visible.append(spec)
                used += line_tokens
                break
            visible.append(spec)
            used += line_tokens
        return ToolInventorySelection(
            visible_tools=tuple(visible),
            omitted_count=max(0, len(specs) - len(visible)),
        )

    def render_inventory(self, *, tags: tuple[str, ...] = (), max_tokens: int | None = None) -> str:
        lines = []
        if max_tokens is None:
            selection = ToolInventorySelection(visible_tools=self.specs(tags=tags))
        else:
            selection = self.select_inventory(tags=tags, max_tokens=max_tokens)
        for spec in selection.visible_tools:
            lines.append(_format_tool_inventory_line(spec))
        if selection.omitted_count:
            lines.append(
                f"... and {selection.omitted_count} more tools. "
                "Use search_tools to find specific capabilities."
            )
        return "\n".join(lines)

    def manifest(self, *, tags: tuple[str, ...] = ()) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-tool-registry/v1",
            "tools": [tool_manifest_item(spec) for spec in self.specs(tags=tags)],
        }

    async def invoke(self, invocation: ToolInvocation) -> ToolResult:
        resolved_name = self.resolve_name(invocation.tool_name)
        spec = self._specs.get(resolved_name)
        handler = self._handlers.get(resolved_name)
        if spec is None or handler is None:
            return ToolResult(
                call_id=invocation.call_id,
                tool_name=invocation.tool_name,
                status="failed",
                error=f"unknown tool: {invocation.tool_name}",
            )
        if not spec.enabled:
            return ToolResult(
                call_id=invocation.call_id,
                tool_name=resolved_name,
                status="failed",
                error=f"tool disabled: {resolved_name}",
            )
        error = _validate_arguments(spec, invocation.arguments)
        if error:
            return ToolResult(
                call_id=invocation.call_id,
                tool_name=resolved_name,
                status="failed",
                error=error,
            )
        normalized = ToolInvocation(
            tool_name=resolved_name,
            arguments=dict(invocation.arguments),
            call_id=invocation.call_id,
            metadata=dict(invocation.metadata),
        )
        return await handler(normalized)


class ToolCenter(ToolRuntimePort):
    """Aggregate multiple tool runtimes behind one provider-neutral boundary."""

    def __init__(self) -> None:
        self._mounts: dict[str, ToolRuntimeMount] = {}

    def mount(
        self,
        name: str,
        runtime: ToolRuntimePort,
        *,
        tags: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> ToolRuntimeMount:
        mount_name = name.strip()
        if not mount_name:
            raise ValueError("tool runtime mount name is required")
        if mount_name in self._mounts:
            raise ValueError(f"tool runtime mount already exists: {mount_name}")
        mount = ToolRuntimeMount(
            name=mount_name,
            runtime=runtime,
            tags=tags,
            metadata=dict(metadata or {}),
        )
        self._mounts[mount_name] = mount
        self._route_table()
        return mount

    def mounts(self) -> tuple[ToolRuntimeMount, ...]:
        return tuple(sorted(self._mounts.values(), key=lambda item: item.name))

    async def refresh(self) -> None:
        for mount in self.mounts():
            refresh = getattr(mount.runtime, "refresh", None)
            if callable(refresh):
                await refresh()
        self._route_table()

    def specs(self) -> tuple[ToolSpec, ...]:
        specs: list[ToolSpec] = []
        for mount in self.mounts():
            for spec in mount.runtime.specs():
                specs.append(self._with_mount_metadata(spec, mount))
        return tuple(sorted(specs, key=lambda item: item.name))

    def search(
        self,
        query: str,
        *,
        tags: tuple[str, ...] = (),
        limit: int = 8,
    ) -> tuple[ToolSpec, ...]:
        requested_tags = {tag.lower() for tag in tags}
        documents = []
        for spec in self.specs():
            if requested_tags:
                own_tags = {tag.lower() for tag in spec.tags}
                if not requested_tags.intersection(own_tags):
                    continue
            documents.append(
                SearchDocument(
                    item=spec,
                    text=" ".join(
                        (
                            spec.name,
                            spec.description,
                            " ".join(spec.aliases),
                            " ".join(spec.tags),
                        )
                    ),
                    name=spec.name,
                )
            )
        return rank_documents(query, tuple(documents), limit=limit)

    def select_inventory(
        self,
        *,
        tags: tuple[str, ...] = (),
        max_tokens: int = 1600,
    ) -> ToolInventorySelection:
        visible: list[ToolSpec] = []
        used = 0
        requested_tags = {tag.lower() for tag in tags}
        specs = self.specs()
        if requested_tags:
            specs = tuple(
                spec
                for spec in specs
                if requested_tags.intersection({tag.lower() for tag in spec.tags})
            )
        for spec in specs:
            line = _format_tool_inventory_line(spec) + "\n"
            line_tokens = estimate_tokens(line)
            if visible and used + line_tokens > max_tokens:
                break
            if not visible and line_tokens > max_tokens:
                visible.append(spec)
                used += line_tokens
                break
            visible.append(spec)
            used += line_tokens
        return ToolInventorySelection(
            visible_tools=tuple(visible),
            omitted_count=max(0, len(specs) - len(visible)),
        )

    def render_inventory(self, *, tags: tuple[str, ...] = (), max_tokens: int | None = None) -> str:
        if max_tokens is None:
            requested_tags = {tag.lower() for tag in tags}
            specs = self.specs()
            if requested_tags:
                specs = tuple(
                    spec
                    for spec in specs
                    if requested_tags.intersection({tag.lower() for tag in spec.tags})
                )
            selection = ToolInventorySelection(visible_tools=specs)
        else:
            selection = self.select_inventory(tags=tags, max_tokens=max_tokens)
        lines = [_format_tool_inventory_line(spec) for spec in selection.visible_tools]
        if selection.omitted_count:
            lines.append(
                f"... and {selection.omitted_count} more tools. "
                "Use search_tools to find specific capabilities."
            )
        return "\n".join(lines)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-tool-center/v1",
            "mounts": [mount.manifest() for mount in self.mounts()],
            "tools": [tool_manifest_item(spec) for spec in self.specs()],
        }

    async def invoke(self, invocation: ToolInvocation) -> ToolResult:
        route = self._route_table().get(invocation.tool_name)
        if route is None:
            return ToolResult(
                call_id=invocation.call_id,
                tool_name=invocation.tool_name,
                status="failed",
                error=f"unknown tool: {invocation.tool_name}",
            )
        mount, spec = route
        routed = ToolInvocation(
            tool_name=spec.name,
            arguments=dict(invocation.arguments),
            call_id=invocation.call_id,
            metadata={
                **invocation.metadata,
                "original_tool_name": invocation.tool_name,
                "tool_center_mount": mount.name,
            },
        )
        result = await mount.runtime.invoke(routed)
        return ToolResult(
            call_id=result.call_id,
            tool_name=result.tool_name,
            status=result.status,
            content=result.content,
            data=dict(result.data),
            error=result.error,
            metadata={
                **result.metadata,
                "tool_center": {
                    "mount": mount.name,
                    "requested_tool_name": invocation.tool_name,
                    "resolved_tool_name": spec.name,
                },
            },
        )

    def _route_table(self) -> dict[str, tuple[ToolRuntimeMount, ToolSpec]]:
        routes: dict[str, tuple[ToolRuntimeMount, ToolSpec]] = {}
        for mount in self.mounts():
            for spec in mount.runtime.specs():
                names = (spec.name, *spec.aliases)
                for raw_name in names:
                    name = raw_name.strip()
                    if not name:
                        continue
                    existing = routes.get(name)
                    if existing is not None:
                        other_mount, other_spec = existing
                        raise ValueError(
                            "tool name collision: "
                            f"{name} from {mount.name}/{spec.name} conflicts with "
                            f"{other_mount.name}/{other_spec.name}"
                        )
                    routes[name] = (mount, spec)
        return routes

    @staticmethod
    def _with_mount_metadata(spec: ToolSpec, mount: ToolRuntimeMount) -> ToolSpec:
        return ToolSpec(
            name=spec.name,
            description=spec.description,
            parameters_schema=spec.parameters_schema,
            aliases=spec.aliases,
            tags=tuple(dict.fromkeys((*mount.tags, *spec.tags))),
            enabled=spec.enabled,
            metadata={
                **spec.metadata,
                "tool_center": {
                    "mount": mount.name,
                    "mount_tags": list(mount.tags),
                    "mount_metadata": mount.metadata,
                },
            },
        )


def _validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> str:
    schema = spec.parameters_schema or {}
    required = schema.get("required") or []
    for key in required:
        if key not in arguments:
            return f"tool {spec.name} missing required argument: {key}"
    properties = schema.get("properties") or {}
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    for key, prop_schema in properties.items():
        if key not in arguments:
            continue
        expected = prop_schema.get("type")
        py_type = type_map.get(str(expected))
        if py_type and not isinstance(arguments[key], py_type):
            return (
                f"tool {spec.name} argument {key} must be {expected}, "
                f"got {type(arguments[key]).__name__}"
            )
    return ""


def schema_from_callable(func: ToolFunction) -> dict[str, Any]:
    signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        properties[name] = {"type": _json_schema_type(parameter.annotation)}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _json_schema_type(annotation: Any) -> str:
    if annotation in {str, "str"}:
        return "string"
    if annotation in {int, "int"}:
        return "integer"
    if annotation in {float, "float"}:
        return "number"
    if annotation in {bool, "bool"}:
        return "boolean"
    if annotation in {dict, "dict"}:
        return "object"
    if annotation in {list, tuple, "list", "tuple"}:
        return "array"
    return "string"


class ToolReplayPort(Protocol):
    async def get(self, invocation: ToolInvocation) -> ToolResult | None:
        """Return a previous result for an idempotent invocation."""

    async def put(self, invocation: ToolInvocation, result: ToolResult) -> None:
        """Store a completed invocation result."""


class NullToolReplay:
    async def get(self, invocation: ToolInvocation) -> ToolResult | None:
        return None

    async def put(self, invocation: ToolInvocation, result: ToolResult) -> None:
        return None


class InMemoryToolReplay:
    def __init__(self) -> None:
        self.results: dict[str, ToolResult] = {}

    async def get(self, invocation: ToolInvocation) -> ToolResult | None:
        return self.results.get(invocation.replay_key())

    async def put(self, invocation: ToolInvocation, result: ToolResult) -> None:
        self.results[invocation.replay_key()] = result


def _format_tool_inventory_line(spec: ToolSpec) -> str:
    alias = f" aliases={','.join(spec.aliases)}" if spec.aliases else ""
    tag_text = f" tags={','.join(spec.tags)}" if spec.tags else ""
    return f"- {spec.name}{alias}{tag_text}: {spec.description}".rstrip()


def tool_manifest_item(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "aliases": list(spec.aliases),
        "tags": list(spec.tags),
        "enabled": spec.enabled,
        "parameters_schema": spec.parameters_schema,
        "metadata": spec.metadata,
    }

