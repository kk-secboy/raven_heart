"""MCP access-center abstractions for agent core."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from agent_core.search import SearchDocument, rank_documents
from agent_core.tools import ToolInvocation, ToolResult, ToolRuntimePort, ToolSpec

if TYPE_CHECKING:
    from agent_core.context import ContextMaterial


@dataclass(frozen=True)
class MCPServerSpec:
    name: str
    transport: str = "runtime"
    tags: tuple[str, ...] = ()
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPServerState:
    server_name: str
    status: str = "registered"
    tool_count: int = 0
    resource_count: int = 0
    prompt_count: int = 0
    refresh_count: int = 0
    last_error: str = ""
    last_refreshed_at: str = ""

    @property
    def healthy(self) -> bool:
        return self.status in {"registered", "connected", "refreshed"}

    def manifest(self) -> dict[str, Any]:
        return {
            "server_name": self.server_name,
            "status": self.status,
            "healthy": self.healthy,
            "tool_count": self.tool_count,
            "resource_count": self.resource_count,
            "prompt_count": self.prompt_count,
            "refresh_count": self.refresh_count,
            "last_error": self.last_error,
            "last_refreshed_at": self.last_refreshed_at,
        }


@dataclass(frozen=True)
class MCPRefreshResult:
    server_name: str
    status: str
    tools: tuple["MCPToolSpec", ...] = ()
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "refreshed"

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-mcp-refresh-result/v1",
            "server_name": self.server_name,
            "status": self.status,
            "ok": self.ok,
            "tool_count": len(self.tools),
            "tools": [
                {
                    "name": tool.reference.resolved_public_name(),
                    "server_name": tool.reference.server_name,
                    "tool_name": tool.reference.tool_name,
                    "enabled": tool.enabled,
                    "tags": list(tool.tags),
                }
                for tool in self.tools
            ],
            "error": self.error,
        }


@dataclass(frozen=True)
class MCPInventoryRefreshResult:
    server_name: str
    status: str
    tool_result: MCPRefreshResult
    resources: tuple["MCPResourceSpec", ...] = ()
    prompts: tuple["MCPPromptSpec", ...] = ()
    resource_error: str = ""
    prompt_error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "refreshed"

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-mcp-inventory-refresh-result/v1",
            "server_name": self.server_name,
            "status": self.status,
            "ok": self.ok,
            "tool_result": self.tool_result.manifest(),
            "tool_count": len(self.tool_result.tools),
            "resource_count": len(self.resources),
            "prompt_count": len(self.prompts),
            "resources": [resource.manifest() for resource in self.resources],
            "prompts": [prompt.manifest() for prompt in self.prompts],
            "resource_error": self.resource_error,
            "prompt_error": self.prompt_error,
        }


@dataclass(frozen=True)
class MCPToolReference:
    server_name: str
    tool_name: str
    public_name: str = ""

    def resolved_public_name(self) -> str:
        if self.public_name:
            return self.public_name
        return f"mcp__{self.server_name}__{self.tool_name}"


@dataclass(frozen=True)
class MCPResourceSpec:
    server_name: str
    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = ""
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "server_name": self.server_name,
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mime_type": self.mime_type,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class MCPResourceContent:
    server_name: str
    uri: str
    text: str = ""
    blob: bytes = b""
    mime_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_bytes(self) -> int:
        return len(self.blob) if self.blob else len(self.text.encode("utf-8"))

    def manifest(self) -> dict[str, Any]:
        return {
            "server_name": self.server_name,
            "uri": self.uri,
            "mime_type": self.mime_type,
            "content_bytes": self.content_bytes,
            "has_text": bool(self.text),
            "has_blob": bool(self.blob),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class MCPPromptSpec:
    server_name: str
    name: str
    description: str = ""
    arguments_schema: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def prompt_id(self) -> str:
        return f"{self.server_name}:{self.name}"

    def manifest(self) -> dict[str, Any]:
        return {
            "server_name": self.server_name,
            "name": self.name,
            "prompt_id": self.prompt_id(),
            "description": self.description,
            "arguments_schema": self.arguments_schema,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class MCPPromptContent:
    server_name: str
    name: str
    messages: tuple[dict[str, Any], ...] = ()
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        lines: list[str] = []
        for message in self.messages:
            role = str(message.get("role") or "user")
            content = message.get("content", "")
            if isinstance(content, dict):
                text = str(content.get("text") or content)
            else:
                text = str(content)
            lines.append(f"[{role}]\n{text}")
        return "\n\n".join(lines)

    def manifest(self) -> dict[str, Any]:
        return {
            "server_name": self.server_name,
            "name": self.name,
            "description": self.description,
            "message_count": len(self.messages),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class MCPContextMaterialRequest:
    """Select MCP resources/prompts as candidate context material."""

    query: str = ""
    resource_uris: tuple[str, ...] = ()
    prompt_ids: tuple[str, ...] = ()
    prompt_arguments: dict[str, dict[str, Any]] = field(default_factory=dict)
    include_resources: bool = True
    include_prompts: bool = True
    limit: int = 8
    max_resource_bytes: int = 8192
    max_prompt_bytes: int = 4096
    role: str = "mcp"
    priority: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-mcp-context-material-request/v1",
            "query_bytes": len(self.query.encode("utf-8")),
            "resource_uris": list(self.resource_uris),
            "prompt_ids": list(self.prompt_ids),
            "prompt_argument_ids": sorted(str(key) for key in self.prompt_arguments),
            "include_resources": self.include_resources,
            "include_prompts": self.include_prompts,
            "limit": self.limit,
            "max_resource_bytes": self.max_resource_bytes,
            "max_prompt_bytes": self.max_prompt_bytes,
            "role": self.role,
            "priority": self.priority,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MCPContextMaterialRecord:
    kind: str
    name: str
    server_name: str
    status: str
    content_bytes: int = 0
    sha256: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "server_name": self.server_name,
            "status": self.status,
            "content_bytes": self.content_bytes,
            "sha256": self.sha256,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MCPContextMaterialResult:
    request: MCPContextMaterialRequest
    materials: tuple["ContextMaterial", ...] = ()
    records: tuple[MCPContextMaterialRecord, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-mcp-context-material-result/v1",
            "material_count": len(self.materials),
            "record_count": len(self.records),
            "failed_count": sum(1 for record in self.records if record.status == "failed"),
            "request": self.request.manifest(),
            "materials": [material.manifest() for material in self.materials],
            "records": [record.manifest() for record in self.records],
        }


@dataclass(frozen=True)
class MCPToolSpec:
    reference: MCPToolReference
    description: str = ""
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_tool_spec(self, server: MCPServerSpec) -> ToolSpec:
        return ToolSpec(
            name=self.reference.resolved_public_name(),
            description=self.description,
            parameters_schema=self.parameters_schema,
            tags=tuple(dict.fromkeys((*server.tags, *self.tags, "mcp"))),
            enabled=server.enabled and self.enabled,
            metadata={
                **self.metadata,
                "mcp": {
                    "server_name": self.reference.server_name,
                    "tool_name": self.reference.tool_name,
                    "transport": server.transport,
                },
            },
        )


class MCPConnectorPort(Protocol):
    async def list_tools(self, server: MCPServerSpec) -> tuple[MCPToolSpec, ...]:
        """List tools exposed by one MCP server."""

    async def invoke_tool(
        self,
        server: MCPServerSpec,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Invoke one remote MCP tool."""


class MCPCenter(ToolRuntimePort):
    """Registry and tool-runtime bridge for MCP servers.

    Concrete runtimes own process management, auth, network sessions, and SDK
    details. The core center only tracks capabilities and exposes them through
    the same tool boundary used by ReAct.
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerSpec] = {}
        self._connectors: dict[str, MCPConnectorPort] = {}
        self._tools: dict[str, MCPToolSpec] = {}
        self._resources: dict[str, MCPResourceSpec] = {}
        self._prompts: dict[str, MCPPromptSpec] = {}
        self._states: dict[str, MCPServerState] = {}
        self._last_inventory_refresh: tuple[MCPInventoryRefreshResult, ...] = ()

    def register_server(self, spec: MCPServerSpec) -> None:
        name = spec.name.strip()
        if not name:
            raise ValueError("mcp server name is required")
        if name in self._servers:
            raise ValueError(f"mcp server already registered: {name}")
        self._servers[name] = spec
        self._states[name] = MCPServerState(
            server_name=name,
            status="registered" if spec.enabled else "disabled",
        )

    def register_connector(self, transport: str, connector: MCPConnectorPort) -> None:
        name = transport.strip()
        if not name:
            raise ValueError("mcp transport is required")
        self._connectors[name] = connector

    def get_server(self, name: str) -> MCPServerSpec | None:
        return self._servers.get(name)

    def servers(self, *, include_disabled: bool = False) -> tuple[MCPServerSpec, ...]:
        servers = self._servers.values()
        if not include_disabled:
            servers = tuple(server for server in servers if server.enabled)
        return tuple(sorted(servers, key=lambda item: item.name))

    def state(self, server_name: str) -> MCPServerState | None:
        return self._states.get(server_name)

    def states(self) -> tuple[MCPServerState, ...]:
        return tuple(sorted(self._states.values(), key=lambda item: item.server_name))

    async def refresh(
        self,
        server_name: str | None = None,
        *,
        fail_fast: bool = True,
    ) -> tuple[MCPToolSpec, ...]:
        results = await self.refresh_status(server_name, fail_fast=fail_fast)
        return tuple(tool for result in results for tool in result.tools)

    async def refresh_status(
        self,
        server_name: str | None = None,
        *,
        fail_fast: bool = True,
    ) -> tuple[MCPRefreshResult, ...]:
        selected = (
            (self._servers[server_name],)
            if server_name is not None
            else self.servers(include_disabled=True)
        )
        results: list[MCPRefreshResult] = []
        for server in selected:
            if not server.enabled:
                self._states[server.name] = MCPServerState(server_name=server.name, status="disabled")
                results.append(MCPRefreshResult(server_name=server.name, status="disabled"))
                continue
            connector = self._connectors.get(server.transport)
            if connector is None:
                error = f"no MCP connector registered for transport: {server.transport}"
                self._mark_failed(server.name, error)
                if fail_fast:
                    raise ValueError(error)
                results.append(MCPRefreshResult(server_name=server.name, status="failed", error=error))
                continue
            try:
                await self._connect(server, connector)
                tools = tuple(self._normalize_tool(server, tool) for tool in await connector.list_tools(server))
            except Exception as exc:
                error = str(exc)
                self._remove_server_tools(server.name)
                self._mark_failed(server.name, error)
                if fail_fast:
                    raise
                results.append(MCPRefreshResult(server_name=server.name, status="failed", error=error))
                continue
            self._remove_server_tools(server.name)
            for tool in tools:
                self._tools[tool.reference.resolved_public_name()] = tool
            current = self._states.get(server.name) or MCPServerState(server.name)
            self._states[server.name] = MCPServerState(
                server_name=server.name,
                status="refreshed",
                tool_count=len(tools),
                resource_count=current.resource_count,
                prompt_count=current.prompt_count,
                refresh_count=current.refresh_count + 1,
                last_refreshed_at=_utc_now_iso(),
            )
            results.append(MCPRefreshResult(server_name=server.name, status="refreshed", tools=tools))
        return tuple(results)

    async def refresh_inventory(
        self,
        server_name: str | None = None,
        *,
        fail_fast: bool = True,
    ) -> tuple[MCPInventoryRefreshResult, ...]:
        """Refresh tools, resources, and prompts as one auditable inventory pass."""

        tool_results = await self.refresh_status(server_name, fail_fast=fail_fast)
        results: list[MCPInventoryRefreshResult] = []
        for tool_result in tool_results:
            resources: tuple[MCPResourceSpec, ...] = ()
            prompts: tuple[MCPPromptSpec, ...] = ()
            resource_error = ""
            prompt_error = ""
            if tool_result.status == "disabled":
                results.append(
                    MCPInventoryRefreshResult(
                        server_name=tool_result.server_name,
                        status="disabled",
                        tool_result=tool_result,
                    )
                )
                continue
            if not tool_result.ok:
                results.append(
                    MCPInventoryRefreshResult(
                        server_name=tool_result.server_name,
                        status="failed",
                        tool_result=tool_result,
                    )
                )
                continue
            try:
                resources = await self.refresh_resources(tool_result.server_name)
            except Exception as exc:
                if fail_fast:
                    raise
                resource_error = str(exc)
            try:
                prompts = await self.refresh_prompts(tool_result.server_name)
            except Exception as exc:
                if fail_fast:
                    raise
                prompt_error = str(exc)
            status = "partial" if resource_error or prompt_error else "refreshed"
            if status == "partial":
                self._mark_partial(
                    tool_result.server_name,
                    resource_error=resource_error,
                    prompt_error=prompt_error,
                )
            results.append(
                MCPInventoryRefreshResult(
                    server_name=tool_result.server_name,
                    status=status,
                    tool_result=tool_result,
                    resources=resources,
                    prompts=prompts,
                    resource_error=resource_error,
                    prompt_error=prompt_error,
                )
            )
        self._last_inventory_refresh = tuple(results)
        return self._last_inventory_refresh

    async def close(self, server_name: str | None = None) -> None:
        selected = (
            (self._servers[server_name],)
            if server_name is not None
            else self.servers(include_disabled=True)
        )
        for server in selected:
            connector = self._connectors.get(server.transport)
            if connector is not None:
                close = getattr(connector, "close", None)
                if callable(close):
                    await close(server)
            self._remove_server_tools(server.name)
            self._remove_server_resources(server.name)
            self._remove_server_prompts(server.name)
            current = self._states.get(server.name) or MCPServerState(server.name)
            self._states[server.name] = MCPServerState(
                server_name=server.name,
                status="closed",
                refresh_count=current.refresh_count,
                resource_count=0,
                prompt_count=0,
                last_error=current.last_error,
                last_refreshed_at=current.last_refreshed_at,
            )

    async def refresh_resources(self, server_name: str | None = None) -> tuple[MCPResourceSpec, ...]:
        selected = (
            (self._servers[server_name],)
            if server_name is not None
            else self.servers(include_disabled=True)
        )
        refreshed: list[MCPResourceSpec] = []
        for server in selected:
            if not server.enabled:
                continue
            connector = self._connectors.get(server.transport)
            if connector is None:
                raise ValueError(f"no MCP connector registered for transport: {server.transport}")
            list_resources = getattr(connector, "list_resources", None)
            if not callable(list_resources):
                continue
            await self._connect(server, connector)
            resources = tuple(self._normalize_resource(server, resource) for resource in await list_resources(server))
            self._remove_server_resources(server.name)
            for resource in resources:
                self._resources[resource.uri] = resource
            self._update_asset_counts(server.name)
            refreshed.extend(resources)
        return tuple(refreshed)

    def resources(self) -> tuple[MCPResourceSpec, ...]:
        return tuple(sorted(self._resources.values(), key=lambda item: item.uri))

    def search_resources(self, query: str, *, limit: int = 8) -> tuple[MCPResourceSpec, ...]:
        documents = tuple(
            SearchDocument(
                item=resource,
                text=" ".join(
                    (
                        resource.uri,
                        resource.name,
                        resource.description,
                        resource.mime_type,
                        " ".join(resource.tags),
                    )
                ),
                name=resource.uri,
            )
            for resource in self.resources()
        )
        return rank_documents(query, documents, limit=limit)

    async def read_resource(self, uri: str) -> MCPResourceContent:
        resource = self._resources.get(uri)
        if resource is None:
            raise KeyError(uri)
        server = self._servers.get(resource.server_name)
        if server is None:
            raise KeyError(resource.server_name)
        connector = self._connectors.get(server.transport)
        if connector is None:
            raise ValueError(f"no MCP connector registered for transport: {server.transport}")
        read_resource = getattr(connector, "read_resource", None)
        if not callable(read_resource):
            raise ValueError(f"MCP connector does not support resources: {server.transport}")
        content = await read_resource(server, uri)
        if isinstance(content, MCPResourceContent):
            return content
        raise TypeError("read_resource must return MCPResourceContent")

    async def refresh_prompts(self, server_name: str | None = None) -> tuple[MCPPromptSpec, ...]:
        selected = (
            (self._servers[server_name],)
            if server_name is not None
            else self.servers(include_disabled=True)
        )
        refreshed: list[MCPPromptSpec] = []
        for server in selected:
            if not server.enabled:
                continue
            connector = self._connectors.get(server.transport)
            if connector is None:
                raise ValueError(f"no MCP connector registered for transport: {server.transport}")
            list_prompts = getattr(connector, "list_prompts", None)
            if not callable(list_prompts):
                continue
            await self._connect(server, connector)
            prompts = tuple(self._normalize_prompt(server, prompt) for prompt in await list_prompts(server))
            self._remove_server_prompts(server.name)
            for prompt in prompts:
                self._prompts[prompt.prompt_id()] = prompt
            self._update_asset_counts(server.name)
            refreshed.extend(prompts)
        return tuple(refreshed)

    def prompts(self) -> tuple[MCPPromptSpec, ...]:
        return tuple(sorted(self._prompts.values(), key=lambda item: item.prompt_id()))

    def search_prompts(self, query: str, *, limit: int = 8) -> tuple[MCPPromptSpec, ...]:
        documents = tuple(
            SearchDocument(
                item=prompt,
                text=" ".join(
                    (
                        prompt.name,
                        prompt.description,
                        " ".join(prompt.tags),
                        prompt.server_name,
                    )
                ),
                name=prompt.prompt_id(),
            )
            for prompt in self.prompts()
        )
        return rank_documents(query, documents, limit=limit)

    async def get_prompt(self, prompt_id: str, arguments: dict[str, Any] | None = None) -> MCPPromptContent:
        prompt = self._prompts.get(prompt_id)
        if prompt is None and ":" in prompt_id:
            prompt = self._prompts.get(prompt_id)
        if prompt is None:
            matches = [item for item in self._prompts.values() if item.name == prompt_id]
            if len(matches) == 1:
                prompt = matches[0]
        if prompt is None:
            raise KeyError(prompt_id)
        server = self._servers.get(prompt.server_name)
        if server is None:
            raise KeyError(prompt.server_name)
        connector = self._connectors.get(server.transport)
        if connector is None:
            raise ValueError(f"no MCP connector registered for transport: {server.transport}")
        get_prompt = getattr(connector, "get_prompt", None)
        if not callable(get_prompt):
            raise ValueError(f"MCP connector does not support prompts: {server.transport}")
        content = await get_prompt(server, prompt.name, dict(arguments or {}))
        if isinstance(content, MCPPromptContent):
            return content
        raise TypeError("get_prompt must return MCPPromptContent")

    async def context_materials(
        self,
        request: MCPContextMaterialRequest | None = None,
    ) -> MCPContextMaterialResult:
        """Read MCP resources/prompts into selector-ready context materials."""

        request = request or MCPContextMaterialRequest()
        from agent_core.context import ContextMaterial

        materials: list[ContextMaterial] = []
        records: list[MCPContextMaterialRecord] = []

        if request.include_resources:
            for resource in self._select_resource_material_specs(request):
                try:
                    content = await self.read_resource(resource.uri)
                    text = content.text or (
                        content.blob.decode("utf-8", errors="ignore") if content.blob else ""
                    )
                    text = _trim_text_to_bytes(text, request.max_resource_bytes)
                    material = ContextMaterial(
                        name=f"mcp_resource:{resource.uri}",
                        content=text,
                        role=request.role,
                        priority=request.priority,
                        metadata={
                            "kind": "resource",
                            "server_name": resource.server_name,
                            "uri": resource.uri,
                            "name": resource.name,
                            "mime_type": content.mime_type or resource.mime_type,
                            "tags": list(resource.tags),
                            **dict(request.metadata),
                        },
                    )
                    materials.append(material)
                    records.append(_mcp_material_record("resource", material.name, resource.server_name, text))
                except Exception as exc:
                    records.append(
                        MCPContextMaterialRecord(
                            kind="resource",
                            name=resource.uri,
                            server_name=resource.server_name,
                            status="failed",
                            error=str(exc),
                        )
                    )

        if request.include_prompts:
            for prompt in self._select_prompt_material_specs(request):
                prompt_id = prompt.prompt_id()
                try:
                    content = await self.get_prompt(prompt_id, request.prompt_arguments.get(prompt_id, {}))
                    text = _trim_text_to_bytes(content.render(), request.max_prompt_bytes)
                    material = ContextMaterial(
                        name=f"mcp_prompt:{prompt_id}",
                        content=text,
                        role=request.role,
                        priority=request.priority,
                        metadata={
                            "kind": "prompt",
                            "server_name": prompt.server_name,
                            "prompt_id": prompt_id,
                            "name": prompt.name,
                            "tags": list(prompt.tags),
                            **dict(request.metadata),
                        },
                    )
                    materials.append(material)
                    records.append(_mcp_material_record("prompt", material.name, prompt.server_name, text))
                except Exception as exc:
                    records.append(
                        MCPContextMaterialRecord(
                            kind="prompt",
                            name=prompt_id,
                            server_name=prompt.server_name,
                            status="failed",
                            error=str(exc),
                        )
                    )

        return MCPContextMaterialResult(
            request=request,
            materials=tuple(materials),
            records=tuple(records),
        )

    def _select_resource_material_specs(
        self,
        request: MCPContextMaterialRequest,
    ) -> tuple[MCPResourceSpec, ...]:
        if request.resource_uris:
            selected = []
            for uri in request.resource_uris:
                resource = self._resources.get(uri)
                if resource is not None:
                    selected.append(resource)
            return tuple(selected)
        if request.query:
            return self.search_resources(request.query, limit=request.limit)
        return self.resources()[: max(0, request.limit)]

    def _select_prompt_material_specs(
        self,
        request: MCPContextMaterialRequest,
    ) -> tuple[MCPPromptSpec, ...]:
        if request.prompt_ids:
            selected = []
            for prompt_id in request.prompt_ids:
                prompt = self._prompts.get(prompt_id)
                if prompt is None:
                    matches = [item for item in self._prompts.values() if item.name == prompt_id]
                    if len(matches) == 1:
                        prompt = matches[0]
                if prompt is not None:
                    selected.append(prompt)
            return tuple(selected)
        if request.query:
            return self.search_prompts(request.query, limit=request.limit)
        return self.prompts()[: max(0, request.limit)]

    def specs(self, *, include_disabled: bool = False) -> tuple[ToolSpec, ...]:
        specs: list[ToolSpec] = []
        for tool in self._tools.values():
            server = self._servers.get(tool.reference.server_name)
            if server is None:
                continue
            spec = tool.to_tool_spec(server)
            if include_disabled or spec.enabled:
                specs.append(spec)
        return tuple(sorted(specs, key=lambda item: item.name))

    def search(self, query: str, *, limit: int = 8) -> tuple[ToolSpec, ...]:
        documents = tuple(
            SearchDocument(
                item=spec,
                text=" ".join((spec.name, spec.description, " ".join(spec.tags))),
                name=spec.name,
            )
            for spec in self.specs()
        )
        return rank_documents(query, documents, limit=limit)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-mcp-center/v1",
            "servers": [
                {
                    "name": server.name,
                    "transport": server.transport,
                    "tags": list(server.tags),
                    "enabled": server.enabled,
                    "metadata": server.metadata,
                    "state": (self._states.get(server.name) or MCPServerState(server.name)).manifest(),
                }
                for server in self.servers(include_disabled=True)
            ],
            "tools": [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "tags": list(spec.tags),
                    "enabled": spec.enabled,
                    "metadata": spec.metadata,
                }
                for spec in self.specs()
            ],
            "resources": [resource.manifest() for resource in self.resources()],
            "prompts": [prompt.manifest() for prompt in self.prompts()],
            "last_inventory_refresh": [
                result.manifest() for result in self._last_inventory_refresh
            ],
        }

    async def invoke(self, invocation: ToolInvocation) -> ToolResult:
        tool = self._tools.get(invocation.tool_name)
        if tool is None:
            return ToolResult(
                call_id=invocation.call_id,
                tool_name=invocation.tool_name,
                status="failed",
                error=f"unknown MCP tool: {invocation.tool_name}",
            )
        server = self._servers.get(tool.reference.server_name)
        if server is None or not server.enabled or not tool.enabled:
            return ToolResult(
                call_id=invocation.call_id,
                tool_name=invocation.tool_name,
                status="failed",
                error=f"MCP tool disabled: {invocation.tool_name}",
            )
        connector = self._connectors.get(server.transport)
        if connector is None:
            return ToolResult(
                call_id=invocation.call_id,
                tool_name=invocation.tool_name,
                status="failed",
                error=f"no MCP connector registered for transport: {server.transport}",
            )
        result = await connector.invoke_tool(server, tool.reference.tool_name, invocation.arguments)
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            status=result.status,
            content=result.content,
            data=dict(result.data),
            error=result.error,
            metadata={
                **result.metadata,
                "mcp": {
                    "server_name": server.name,
                    "tool_name": tool.reference.tool_name,
                    "transport": server.transport,
                },
            },
        )

    @staticmethod
    def _normalize_tool(server: MCPServerSpec, tool: MCPToolSpec) -> MCPToolSpec:
        reference = tool.reference
        if not reference.server_name:
            reference = MCPToolReference(
                server_name=server.name,
                tool_name=reference.tool_name,
                public_name=reference.public_name,
            )
        if reference.server_name != server.name:
            raise ValueError(
                f"MCP tool server mismatch: {reference.server_name} != {server.name}"
            )
        if not reference.tool_name.strip():
            raise ValueError("MCP tool name is required")
        return MCPToolSpec(
            reference=reference,
            description=tool.description,
            parameters_schema=tool.parameters_schema,
            tags=tool.tags,
            enabled=tool.enabled,
            metadata=tool.metadata,
        )

    async def _connect(self, server: MCPServerSpec, connector: MCPConnectorPort) -> None:
        connect = getattr(connector, "connect", None)
        if callable(connect):
            await connect(server)
        current = self._states.get(server.name) or MCPServerState(server.name)
        status = current.status if current.status in {"refreshed", "partial"} else "connected"
        self._states[server.name] = MCPServerState(
            server_name=server.name,
            status=status,
            tool_count=current.tool_count,
            resource_count=current.resource_count,
            prompt_count=current.prompt_count,
            refresh_count=current.refresh_count,
            last_refreshed_at=current.last_refreshed_at,
        )

    def _mark_failed(self, server_name: str, error: str) -> None:
        current = self._states.get(server_name) or MCPServerState(server_name)
        self._states[server_name] = MCPServerState(
            server_name=server_name,
            status="failed",
            tool_count=current.tool_count,
            resource_count=current.resource_count,
            prompt_count=current.prompt_count,
            refresh_count=current.refresh_count,
            last_error=error,
            last_refreshed_at=current.last_refreshed_at,
        )

    def _mark_partial(
        self,
        server_name: str,
        *,
        resource_error: str = "",
        prompt_error: str = "",
    ) -> None:
        current = self._states.get(server_name) or MCPServerState(server_name)
        self._states[server_name] = MCPServerState(
            server_name=server_name,
            status="partial",
            tool_count=current.tool_count,
            resource_count=current.resource_count,
            prompt_count=current.prompt_count,
            refresh_count=current.refresh_count,
            last_error="; ".join(error for error in (resource_error, prompt_error) if error),
            last_refreshed_at=current.last_refreshed_at,
        )

    def _update_asset_counts(self, server_name: str) -> None:
        current = self._states.get(server_name) or MCPServerState(server_name)
        self._states[server_name] = MCPServerState(
            server_name=server_name,
            status=current.status,
            tool_count=sum(
                1 for tool in self._tools.values() if tool.reference.server_name == server_name
            ),
            resource_count=sum(
                1 for resource in self._resources.values() if resource.server_name == server_name
            ),
            prompt_count=sum(
                1 for prompt in self._prompts.values() if prompt.server_name == server_name
            ),
            refresh_count=current.refresh_count,
            last_error=current.last_error,
            last_refreshed_at=current.last_refreshed_at,
        )

    def _remove_server_tools(self, server_name: str) -> None:
        for public_name, tool in tuple(self._tools.items()):
            if tool.reference.server_name == server_name:
                self._tools.pop(public_name, None)

    def _remove_server_resources(self, server_name: str) -> None:
        for uri, resource in tuple(self._resources.items()):
            if resource.server_name == server_name:
                self._resources.pop(uri, None)

    def _remove_server_prompts(self, server_name: str) -> None:
        for prompt_id, prompt in tuple(self._prompts.items()):
            if prompt.server_name == server_name:
                self._prompts.pop(prompt_id, None)

    @staticmethod
    def _normalize_resource(server: MCPServerSpec, resource: MCPResourceSpec) -> MCPResourceSpec:
        if not resource.uri.strip():
            raise ValueError("MCP resource uri is required")
        server_name = resource.server_name or server.name
        if server_name != server.name:
            raise ValueError(f"MCP resource server mismatch: {server_name} != {server.name}")
        return MCPResourceSpec(
            server_name=server_name,
            uri=resource.uri,
            name=resource.name,
            description=resource.description,
            mime_type=resource.mime_type,
            tags=resource.tags,
            metadata=resource.metadata,
        )

    @staticmethod
    def _normalize_prompt(server: MCPServerSpec, prompt: MCPPromptSpec) -> MCPPromptSpec:
        if not prompt.name.strip():
            raise ValueError("MCP prompt name is required")
        server_name = prompt.server_name or server.name
        if server_name != server.name:
            raise ValueError(f"MCP prompt server mismatch: {server_name} != {server.name}")
        return MCPPromptSpec(
            server_name=server_name,
            name=prompt.name,
            description=prompt.description,
            arguments_schema=prompt.arguments_schema,
            tags=prompt.tags,
            metadata=prompt.metadata,
        )


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _mcp_material_record(
    kind: str,
    name: str,
    server_name: str,
    content: str,
) -> MCPContextMaterialRecord:
    raw = content.encode("utf-8")
    return MCPContextMaterialRecord(
        kind=kind,
        name=name,
        server_name=server_name,
        status="included",
        content_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest() if raw else "",
    )


def _trim_text_to_bytes(text: str, max_bytes: int) -> str:
    limit = max(0, int(max_bytes))
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    if limit <= 0:
        return ""
    marker = "\n[...mcp material trimmed...]\n"
    marker_bytes = marker.encode("utf-8")
    if limit <= len(marker_bytes):
        return raw[:limit].decode("utf-8", errors="ignore")
    keep = limit - len(marker_bytes)
    head_bytes = max(1, keep // 2)
    tail_bytes = max(0, keep - head_bytes)
    head = raw[:head_bytes].decode("utf-8", errors="ignore")
    tail = raw[-tail_bytes:].decode("utf-8", errors="ignore") if tail_bytes else ""
    return f"{head}{marker}{tail}".strip()

