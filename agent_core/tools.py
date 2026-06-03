"""Tool protocol types for the agent core."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import re
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from agent_core.backends import storage_backend_manifest
from agent_core.prompt import estimate_tokens
from agent_core.schema import SchemaValidationResult, validate_json_schema_subset
from agent_core.search import SearchDocument, rank_documents


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


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

    def manifest(self) -> dict[str, Any]:
        raw_arguments = json.dumps(self.arguments, sort_keys=True, default=str)
        return {
            "schema_version": "agent-core-tool-invocation/v1",
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "replay_key": self.replay_key(),
            "argument_keys": sorted(str(key) for key in self.arguments),
            "arguments_sha256": hashlib.sha256(raw_arguments.encode("utf-8")).hexdigest()
            if self.arguments
            else "",
            "metadata": dict(self.metadata),
        }


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

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-tool-result/v1",
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "ok": self.ok,
            "content_bytes": self.content_bytes,
            "content_sha256": hashlib.sha256(self.content.encode("utf-8")).hexdigest()
            if self.content
            else "",
            "data_keys": sorted(str(key) for key in self.data),
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ToolRetryPolicy:
    max_attempts: int = 1
    retry_failed_statuses: tuple[str, ...] = ("failed",)
    retry_on_exceptions: bool = True
    metadata_retryable_key: str = "retryable"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_attempts", max(1, int(self.max_attempts)))

    def allows_result_retry(self, result: ToolResult, *, attempt: int) -> bool:
        if attempt >= self.max_attempts:
            return False
        if result.status not in self.retry_failed_statuses:
            return False
        return bool(result.metadata.get(self.metadata_retryable_key, False))

    def allows_exception_retry(self, exc: Exception, *, attempt: int) -> bool:
        if attempt >= self.max_attempts or not self.retry_on_exceptions:
            return False
        return bool(getattr(exc, self.metadata_retryable_key, False))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-tool-retry-policy/v1",
            "max_attempts": self.max_attempts,
            "retry_failed_statuses": list(self.retry_failed_statuses),
            "retry_on_exceptions": self.retry_on_exceptions,
            "metadata_retryable_key": self.metadata_retryable_key,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ToolExecutionAttempt:
    attempt: int
    status: str
    retryable: bool = False
    error: str = ""
    result: ToolResult | None = None
    exception_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-tool-execution-attempt/v1",
            "attempt": self.attempt,
            "status": self.status,
            "retryable": self.retryable,
            "error": self.error,
            "exception_type": self.exception_type,
            "result": self.result.manifest() if self.result is not None else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ToolExecutionRecord:
    invocation: ToolInvocation
    attempts: tuple[ToolExecutionAttempt, ...]
    final_result: ToolResult
    policy: ToolRetryPolicy = field(default_factory=ToolRetryPolicy)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def retried(self) -> bool:
        return self.attempt_count > 1

    def summary_manifest(self) -> dict[str, Any]:
        summary = {
            "schema_version": "agent-core-tool-execution-summary/v1",
            "tool_name": self.invocation.tool_name,
            "call_id": self.invocation.call_id,
            "attempt_count": self.attempt_count,
            "retried": self.retried,
            "final_status": self.final_result.status,
            "final_ok": self.final_result.ok,
            "attempt_statuses": [attempt.status for attempt in self.attempts],
            "retryable_attempts": [
                attempt.attempt for attempt in self.attempts if attempt.retryable
            ],
        }
        schema_validation = self.metadata.get("schema_validation")
        if isinstance(schema_validation, dict):
            summary["schema_validation"] = dict(schema_validation)
        return summary

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-tool-execution-record/v1",
            "invocation": self.invocation.manifest(),
            "attempt_count": self.attempt_count,
            "retried": self.retried,
            "attempts": [attempt.manifest() for attempt in self.attempts],
            "final_result": self.final_result.manifest(),
            "policy": self.policy.manifest(),
            "metadata": dict(self.metadata),
        }


class ToolExecutionCenter:
    """Retry and audit wrapper for any tool runtime."""

    def __init__(
        self,
        runtime: ToolRuntimePort,
        *,
        retry_policy: ToolRetryPolicy | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.retry_policy = retry_policy or ToolRetryPolicy()
        self.metadata = dict(metadata or {})
        self.records: list[ToolExecutionRecord] = []

    def specs(self) -> tuple[ToolSpec, ...]:
        return self.runtime.specs()

    async def invoke(self, invocation: ToolInvocation) -> ToolResult:
        schema_validation = _tool_schema_validation_for_runtime(self.runtime, invocation)
        if schema_validation is not None and not schema_validation.ok:
            validation_manifest = schema_validation.manifest()
            final_result = ToolResult(
                call_id=invocation.call_id,
                tool_name=invocation.tool_name,
                status="failed",
                error=schema_validation.error,
                metadata={"schema_validation": validation_manifest},
            )
            record = ToolExecutionRecord(
                invocation=invocation,
                attempts=(
                    ToolExecutionAttempt(
                        attempt=1,
                        status="schema_invalid",
                        error=schema_validation.error,
                        result=final_result,
                        metadata={"schema_validation": validation_manifest},
                    ),
                ),
                final_result=final_result,
                policy=self.retry_policy,
                metadata={**self.metadata, "schema_validation": validation_manifest},
            )
            self.records.append(record)
            return ToolResult(
                call_id=final_result.call_id,
                tool_name=final_result.tool_name,
                status=final_result.status,
                content=final_result.content,
                data=dict(final_result.data),
                error=final_result.error,
                metadata={
                    **final_result.metadata,
                    "tool_execution": record.summary_manifest(),
                },
            )
        attempts: list[ToolExecutionAttempt] = []
        final_result: ToolResult | None = None
        for attempt_number in range(1, self.retry_policy.max_attempts + 1):
            try:
                result = await self.runtime.invoke(invocation)
            except Exception as exc:
                retryable = self.retry_policy.allows_exception_retry(exc, attempt=attempt_number)
                attempts.append(
                    ToolExecutionAttempt(
                        attempt=attempt_number,
                        status="exception",
                        retryable=retryable,
                        error=str(exc),
                        exception_type=type(exc).__name__,
                    )
                )
                final_result = ToolResult(
                    call_id=invocation.call_id,
                    tool_name=invocation.tool_name,
                    status="failed",
                    error=str(exc),
                    metadata={
                        "exception_type": type(exc).__name__,
                        "retryable": retryable,
                    },
                )
                if retryable:
                    continue
                break
            retryable = self.retry_policy.allows_result_retry(result, attempt=attempt_number)
            attempts.append(
                ToolExecutionAttempt(
                    attempt=attempt_number,
                    status=result.status,
                    retryable=retryable,
                    error=result.error,
                    result=result,
                )
            )
            final_result = result
            if retryable:
                continue
            break
        if final_result is None:
            final_result = ToolResult(
                call_id=invocation.call_id,
                tool_name=invocation.tool_name,
                status="failed",
                error="tool execution produced no result",
            )
        record = ToolExecutionRecord(
            invocation=invocation,
            attempts=tuple(attempts),
            final_result=final_result,
            policy=self.retry_policy,
            metadata={
                **self.metadata,
                **(
                    {"schema_validation": schema_validation.manifest()}
                    if schema_validation is not None
                    else {}
                ),
            },
        )
        self.records.append(record)
        return ToolResult(
            call_id=final_result.call_id,
            tool_name=final_result.tool_name,
            status=final_result.status,
            content=final_result.content,
            data=dict(final_result.data),
            error=final_result.error,
            metadata={
                **final_result.metadata,
                "tool_execution": record.summary_manifest(),
            },
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-tool-execution-center/v1",
            "retry_policy": self.retry_policy.manifest(),
            "record_count": len(self.records),
            "records": [record.manifest() for record in self.records],
            "metadata": dict(self.metadata),
        }


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
        validation = validate_tool_arguments(spec, invocation.arguments)
        if not validation.ok:
            return ToolResult(
                call_id=invocation.call_id,
                tool_name=resolved_name,
                status="failed",
                error=validation.error,
                metadata={"schema_validation": validation.manifest()},
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


def validate_tool_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> SchemaValidationResult:
    return validate_json_schema_subset(
        arguments,
        spec.parameters_schema or {},
        schema_name=f"tool:{spec.name}",
        metadata={"tool": tool_manifest_item(spec)},
    )


def _tool_schema_validation_for_runtime(
    runtime: ToolRuntimePort,
    invocation: ToolInvocation,
) -> SchemaValidationResult | None:
    spec = _tool_spec_for_invocation(runtime, invocation.tool_name)
    if spec is None:
        return None
    return validate_tool_arguments(spec, invocation.arguments)


def _tool_spec_for_invocation(runtime: ToolRuntimePort, tool_name: str) -> ToolSpec | None:
    for spec in runtime.specs():
        if tool_name == spec.name or tool_name in spec.aliases:
            return spec
    return None


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


@dataclass(frozen=True)
class ToolReplayRecord:
    replay_key: str
    invocation: ToolInvocation
    result: ToolResult
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-tool-replay-record/v1",
            "replay_key": self.replay_key,
            "created_at": self.created_at,
            "invocation": self.invocation.manifest(),
            "result": self.result.manifest(),
            "metadata": dict(self.metadata),
        }


class ToolReplayStorePort(Protocol):
    async def load(self, replay_key: str) -> ToolReplayRecord | None:
        """Load a replay record by key."""

    async def save(self, record: ToolReplayRecord) -> None:
        """Persist a replay record."""

    async def records(self) -> tuple[ToolReplayRecord, ...]:
        """Return replay records for manifest/export use."""


class NullToolReplay:
    async def get(self, invocation: ToolInvocation) -> ToolResult | None:
        return None

    async def put(self, invocation: ToolInvocation, result: ToolResult) -> None:
        return None


class InMemoryToolReplay:
    def __init__(self, records: tuple[ToolReplayRecord, ...] = ()) -> None:
        self.results: dict[str, ToolReplayRecord] = {
            record.replay_key: record for record in records
        }

    async def get(self, invocation: ToolInvocation) -> ToolResult | None:
        record = self.results.get(invocation.replay_key())
        return record.result if record is not None else None

    async def put(self, invocation: ToolInvocation, result: ToolResult) -> None:
        replay_key = invocation.replay_key()
        self.results[replay_key] = ToolReplayRecord(
            replay_key=replay_key,
            invocation=invocation,
            result=result,
        )

    def records(self) -> tuple[ToolReplayRecord, ...]:
        return tuple(sorted(self.results.values(), key=lambda item: item.created_at))

    def manifest(self) -> dict[str, Any]:
        records = self.records()
        return {
            "schema_version": "agent-core-in-memory-tool-replay/v1",
            "record_count": len(records),
            "records": [record.manifest() for record in records],
        }


class InMemoryToolReplayStore(ToolReplayStorePort):
    def __init__(self, records: tuple[ToolReplayRecord, ...] = ()) -> None:
        self._records: dict[str, ToolReplayRecord] = {
            record.replay_key: record for record in records
        }

    async def load(self, replay_key: str) -> ToolReplayRecord | None:
        return self._records.get(replay_key)

    async def save(self, record: ToolReplayRecord) -> None:
        self._records[record.replay_key] = record

    async def records(self) -> tuple[ToolReplayRecord, ...]:
        return tuple(sorted(self._records.values(), key=lambda item: item.created_at))

    def manifest(self) -> dict[str, Any]:
        records = tuple(sorted(self._records.values(), key=lambda item: item.created_at))
        return {
            "schema_version": "agent-core-in-memory-tool-replay-store/v1",
            "backend": storage_backend_manifest(role="tool_replay", kind="in_memory"),
            "record_count": len(records),
            "records": [record.manifest() for record in records],
        }


class SQLiteToolReplayStore(ToolReplayStorePort):
    """SQLite-backed replay store for durable local SDK runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    async def load(self, replay_key: str) -> ToolReplayRecord | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                """
                SELECT record_json
                FROM tool_replay_records
                WHERE replay_key = ?
                """,
                (replay_key,),
            ).fetchone()
        if row is None:
            return None
        return _tool_replay_record_from_json(str(row[0] or "{}"))

    async def save(self, record: ToolReplayRecord) -> None:
        raw = json.dumps(
            _tool_replay_record_payload(record),
            default=str,
            ensure_ascii=False,
            sort_keys=True,
        )
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO tool_replay_records(replay_key, record_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(replay_key) DO UPDATE SET
                    record_json = excluded.record_json,
                    created_at = excluded.created_at
                """,
                (record.replay_key, raw, record.created_at),
            )
            conn.commit()

    async def records(self) -> tuple[ToolReplayRecord, ...]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """
                SELECT record_json
                FROM tool_replay_records
                ORDER BY created_at ASC, replay_key ASC
                """
            ).fetchall()
        return tuple(_tool_replay_record_from_json(str(row[0] or "{}")) for row in rows)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-sqlite-tool-replay-store/v1",
            "backend": storage_backend_manifest(
                role="tool_replay",
                kind="sqlite",
                location=str(self.path),
                capabilities=("load", "save", "records"),
            ),
            "path": str(self.path),
        }

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_replay_records (
                    replay_key TEXT PRIMARY KEY,
                    record_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()


class MarkdownToolReplayStore(ToolReplayStorePort):
    """Markdown-backed replay store for inspectable local SDK runs."""

    _START = "<!-- tool-replay-record "
    _END = " -->"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def load(self, replay_key: str) -> ToolReplayRecord | None:
        for record in await self.records():
            if record.replay_key == replay_key:
                return record
        return None

    async def save(self, record: ToolReplayRecord) -> None:
        records = {
            item.replay_key: item
            for item in await self.records()
        }
        records[record.replay_key] = record
        self._write(tuple(sorted(records.values(), key=lambda item: (item.created_at, item.replay_key))))

    async def records(self) -> tuple[ToolReplayRecord, ...]:
        if not self.path.exists():
            return ()
        text = self.path.read_text(encoding="utf-8")
        records: list[ToolReplayRecord] = []
        for match in _TOOL_REPLAY_MARKDOWN_RE.finditer(text):
            try:
                records.append(_tool_replay_record_from_json(_decode_tool_replay_payload(match.group(1))))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return tuple(sorted(records, key=lambda item: (item.created_at, item.replay_key)))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-markdown-tool-replay-store/v1",
            "backend": storage_backend_manifest(
                role="tool_replay",
                kind="markdown",
                location=str(self.path),
                capabilities=("load", "save", "records"),
            ),
            "path": str(self.path),
        }

    def _write(self, records: tuple[ToolReplayRecord, ...]) -> None:
        lines = [
            "# Tool Replay Records",
            "",
            "This file is managed by raven_heart. Replay payloads are stored in comments.",
            "",
        ]
        for record in records:
            raw = _encode_tool_replay_payload(record)
            lines.append(f"{self._START}{raw}{self._END}")
            lines.append(f"- replay_key: `{record.replay_key}`")
            lines.append(f"- tool: `{record.invocation.tool_name}`")
            lines.append(f"- status: `{record.result.status}`")
            lines.append("")
        self.path.write_text("\n".join(lines), encoding="utf-8")


class PersistentToolReplay(ToolReplayPort):
    def __init__(self, store: ToolReplayStorePort) -> None:
        self.store = store

    async def get(self, invocation: ToolInvocation) -> ToolResult | None:
        record = await self.store.load(invocation.replay_key())
        return record.result if record is not None else None

    async def put(self, invocation: ToolInvocation, result: ToolResult) -> None:
        replay_key = invocation.replay_key()
        await self.store.save(
            ToolReplayRecord(
                replay_key=replay_key,
                invocation=invocation,
                result=result,
            )
        )

    async def records(self) -> tuple[ToolReplayRecord, ...]:
        return await self.store.records()

    async def manifest(self) -> dict[str, Any]:
        store_manifest = getattr(self.store, "manifest", None)
        records = await self.store.records()
        return {
            "schema_version": "agent-core-persistent-tool-replay/v1",
            "store": store_manifest() if callable(store_manifest) else {},
            "record_count": len(records),
            "records": [record.manifest() for record in records],
        }


_TOOL_REPLAY_MARKDOWN_RE = re.compile(
    r"<!--\s*tool-replay-record\s+([A-Za-z0-9+/=]+)\s*-->",
    re.DOTALL,
)


def _encode_tool_replay_payload(record: ToolReplayRecord) -> str:
    raw = json.dumps(
        _tool_replay_record_payload(record),
        default=str,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _decode_tool_replay_payload(encoded: str) -> str:
    return base64.b64decode(encoded.encode("ascii")).decode("utf-8")


def _tool_replay_record_payload(record: ToolReplayRecord) -> dict[str, Any]:
    return {
        "schema_version": "agent-core-tool-replay-payload/v1",
        "replay_key": record.replay_key,
        "created_at": record.created_at,
        "invocation": {
            "tool_name": record.invocation.tool_name,
            "arguments": dict(record.invocation.arguments),
            "call_id": record.invocation.call_id,
            "metadata": dict(record.invocation.metadata),
        },
        "result": {
            "call_id": record.result.call_id,
            "tool_name": record.result.tool_name,
            "status": record.result.status,
            "content": record.result.content,
            "data": dict(record.result.data),
            "error": record.result.error,
            "metadata": dict(record.result.metadata),
        },
        "metadata": dict(record.metadata),
    }


def _tool_replay_record_from_json(raw: str) -> ToolReplayRecord:
    payload = json.loads(raw or "{}")
    if not isinstance(payload, dict):
        raise ValueError("invalid tool replay payload")
    invocation_payload = payload.get("invocation")
    result_payload = payload.get("result")
    if not isinstance(invocation_payload, dict) or not isinstance(result_payload, dict):
        raise ValueError("invalid tool replay payload")
    invocation = ToolInvocation(
        tool_name=str(invocation_payload.get("tool_name") or ""),
        arguments=dict(invocation_payload.get("arguments") or {}),
        call_id=str(invocation_payload.get("call_id") or ""),
        metadata=dict(invocation_payload.get("metadata") or {}),
    )
    result = ToolResult(
        call_id=str(result_payload.get("call_id") or invocation.call_id),
        tool_name=str(result_payload.get("tool_name") or invocation.tool_name),
        status=str(result_payload.get("status") or "completed"),
        content=str(result_payload.get("content") or ""),
        data=dict(result_payload.get("data") or {}),
        error=str(result_payload.get("error") or ""),
        metadata=dict(result_payload.get("metadata") or {}),
    )
    replay_key = str(payload.get("replay_key") or invocation.replay_key())
    return ToolReplayRecord(
        replay_key=replay_key,
        invocation=invocation,
        result=result,
        created_at=str(payload.get("created_at") or utc_now_iso()),
        metadata=dict(payload.get("metadata") or {}),
    )


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

