from __future__ import annotations

from typing import Any

import pytest

from agent_core.actions import ActionRegistry
from agent_core.harness import AgentHarness
from agent_core.mcp import (
    MCPCenter,
    MCPContextMaterialRequest,
    MCPPromptContent,
    MCPPromptSpec,
    MCPResourceContent,
    MCPResourceSpec,
    MCPServerSpec,
    MCPToolReference,
    MCPToolSpec,
)
from agent_core.prompt import PromptIR
from agent_core.react import ReActConfig, ReActExecutor
from agent_core.testing import InMemoryHarness, MockLLMProvider
from agent_core.tools import ToolInvocation, ToolResult


class FakeMCPConnector:
    def __init__(self) -> None:
        self.invocations: list[tuple[str, str, dict[str, Any]]] = []

    async def list_tools(self, server: MCPServerSpec) -> tuple[MCPToolSpec, ...]:
        return (
            MCPToolSpec(
                reference=MCPToolReference(server_name=server.name, tool_name="read_file"),
                description="Read a file through MCP",
                parameters_schema={
                    "type": "object",
                    "required": ["path"],
                    "properties": {"path": {"type": "string"}},
                },
                tags=("filesystem",),
            ),
        )

    async def invoke_tool(
        self,
        server: MCPServerSpec,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        self.invocations.append((server.name, tool_name, dict(arguments)))
        return ToolResult(
            call_id="remote-call",
            tool_name=tool_name,
            content=f"{server.name}:{tool_name}:{arguments['path']}",
        )

    async def list_resources(self, server: MCPServerSpec) -> tuple[MCPResourceSpec, ...]:
        return (
            MCPResourceSpec(
                server_name=server.name,
                uri="file://README.md",
                name="README",
                description="Project readme",
                mime_type="text/markdown",
                tags=("docs",),
            ),
        )

    async def read_resource(self, server: MCPServerSpec, uri: str) -> MCPResourceContent:
        return MCPResourceContent(
            server_name=server.name,
            uri=uri,
            text=f"{server.name}:{uri}",
            mime_type="text/markdown",
        )

    async def list_prompts(self, server: MCPServerSpec) -> tuple[MCPPromptSpec, ...]:
        return (
            MCPPromptSpec(
                server_name=server.name,
                name="summarize",
                description="Summarize a target",
                arguments_schema={
                    "type": "object",
                    "required": ["target"],
                    "properties": {"target": {"type": "string"}},
                },
                tags=("analysis",),
            ),
        )

    async def get_prompt(
        self,
        server: MCPServerSpec,
        name: str,
        arguments: dict[str, Any],
    ) -> MCPPromptContent:
        return MCPPromptContent(
            server_name=server.name,
            name=name,
            messages=(
                {
                    "role": "user",
                    "content": f"summarize {arguments.get('target', '')}",
                },
            ),
            description="Summarize a target",
        )


class LifecycleMCPConnector(FakeMCPConnector):
    def __init__(self) -> None:
        super().__init__()
        self.connected: list[str] = []
        self.closed: list[str] = []

    async def connect(self, server: MCPServerSpec) -> None:
        self.connected.append(server.name)

    async def close(self, server: MCPServerSpec) -> None:
        self.closed.append(server.name)


class FailingMCPConnector(FakeMCPConnector):
    async def list_tools(self, server: MCPServerSpec) -> tuple[MCPToolSpec, ...]:
        raise RuntimeError(f"cannot refresh {server.name}")


class PartialMCPConnector(FakeMCPConnector):
    async def list_resources(self, server: MCPServerSpec) -> tuple[MCPResourceSpec, ...]:
        raise RuntimeError(f"cannot list resources for {server.name}")


def test_mcp_server_spec_defaults_to_runtime_transport_without_deployment_fields() -> None:
    center = MCPCenter()
    center.register_server(MCPServerSpec(name="runtime-owned", tags=("external",)))
    manifest = center.manifest()
    server = manifest["servers"][0]

    assert server["transport"] == "runtime"
    assert "command" not in server
    assert "args" not in server
    assert "url" not in server
    assert "env" not in server


@pytest.mark.asyncio
async def test_mcp_center_refreshes_searches_manifests_and_invokes_tools() -> None:
    center = MCPCenter()
    connector = FakeMCPConnector()
    center.register_server(MCPServerSpec(name="fs", transport="mock", tags=("local",)))
    center.register_connector("mock", connector)

    refreshed = await center.refresh()
    specs = center.specs()
    search = center.search("file local")
    manifest = center.manifest()
    result = await center.invoke(
        ToolInvocation(
            tool_name="mcp__fs__read_file",
            arguments={"path": "README.md"},
        )
    )

    assert [tool.reference.tool_name for tool in refreshed] == ["read_file"]
    assert specs[0].name == "mcp__fs__read_file"
    assert "mcp" in specs[0].tags
    assert search[0].name == "mcp__fs__read_file"
    assert manifest["schema_version"] == "agent-core-mcp-center/v1"
    assert result.ok
    assert result.content == "fs:read_file:README.md"
    assert result.metadata["mcp"]["server_name"] == "fs"
    assert connector.invocations == [("fs", "read_file", {"path": "README.md"})]
    assert manifest["servers"][0]["state"]["status"] == "refreshed"


@pytest.mark.asyncio
async def test_mcp_center_refresh_inventory_records_tools_resources_and_prompts() -> None:
    center = MCPCenter()
    center.register_server(MCPServerSpec(name="fs", transport="mock", tags=("local",)))
    center.register_connector("mock", FakeMCPConnector())

    results = await center.refresh_inventory()
    manifest = center.manifest()
    state = center.state("fs")

    assert len(results) == 1
    assert results[0].ok is True
    assert results[0].manifest()["schema_version"] == "agent-core-mcp-inventory-refresh-result/v1"
    assert results[0].manifest()["tool_result"]["tool_count"] == 1
    assert results[0].manifest()["resource_count"] == 1
    assert results[0].manifest()["prompt_count"] == 1
    assert state is not None
    assert state.status == "refreshed"
    assert state.tool_count == 1
    assert state.resource_count == 1
    assert state.prompt_count == 1
    assert manifest["last_inventory_refresh"][0]["status"] == "refreshed"
    assert manifest["servers"][0]["state"]["resource_count"] == 1
    assert manifest["servers"][0]["state"]["prompt_count"] == 1


@pytest.mark.asyncio
async def test_mcp_center_refresh_inventory_records_partial_asset_failures() -> None:
    center = MCPCenter()
    center.register_server(MCPServerSpec(name="fs", transport="mock"))
    center.register_connector("mock", PartialMCPConnector())

    results = await center.refresh_inventory(fail_fast=False)
    manifest = results[0].manifest()
    state = center.state("fs")

    assert results[0].ok is False
    assert manifest["status"] == "partial"
    assert manifest["tool_count"] == 1
    assert manifest["resource_count"] == 0
    assert manifest["prompt_count"] == 1
    assert manifest["resource_error"] == "cannot list resources for fs"
    assert state is not None
    assert state.status == "partial"
    assert state.tool_count == 1
    assert state.prompt_count == 1
    assert state.last_error == "cannot list resources for fs"


@pytest.mark.asyncio
async def test_mcp_center_tracks_lifecycle_and_closes_servers() -> None:
    center = MCPCenter()
    connector = LifecycleMCPConnector()
    center.register_server(MCPServerSpec(name="fs", transport="mock"))
    center.register_connector("mock", connector)

    results = await center.refresh_status()
    await center.close("fs")

    assert results[0].ok is True
    assert connector.connected == ["fs"]
    assert connector.closed == ["fs"]
    assert center.state("fs").status == "closed"  # type: ignore[union-attr]
    assert center.specs() == ()


@pytest.mark.asyncio
async def test_mcp_center_records_refresh_failures_without_stale_tools() -> None:
    center = MCPCenter()
    connector = FakeMCPConnector()
    center.register_server(MCPServerSpec(name="fs", transport="mock"))
    center.register_connector("mock", connector)
    await center.refresh()
    center.register_connector("mock", FailingMCPConnector())

    results = await center.refresh_status(fail_fast=False)

    assert results[0].ok is False
    assert results[0].error == "cannot refresh fs"
    assert center.state("fs").status == "failed"  # type: ignore[union-attr]
    assert center.specs() == ()


@pytest.mark.asyncio
async def test_mcp_center_refreshes_searches_manifests_and_reads_resources() -> None:
    center = MCPCenter()
    center.register_server(MCPServerSpec(name="fs", transport="mock"))
    center.register_connector("mock", FakeMCPConnector())

    resources = await center.refresh_resources()
    search = center.search_resources("readme docs")
    content = await center.read_resource("file://README.md")
    manifest = center.manifest()

    assert [resource.uri for resource in resources] == ["file://README.md"]
    assert search[0].name == "README"
    assert content.text == "fs:file://README.md"
    assert content.mime_type == "text/markdown"
    assert manifest["resources"][0]["uri"] == "file://README.md"


@pytest.mark.asyncio
async def test_mcp_center_refreshes_searches_manifests_and_gets_prompts() -> None:
    center = MCPCenter()
    center.register_server(MCPServerSpec(name="prompts", transport="mock"))
    center.register_connector("mock", FakeMCPConnector())

    prompts = await center.refresh_prompts()
    search = center.search_prompts("summarize analysis")
    content = await center.get_prompt("prompts:summarize", {"target": "service"})
    manifest = center.manifest()

    assert [prompt.prompt_id() for prompt in prompts] == ["prompts:summarize"]
    assert search[0].name == "summarize"
    assert content.render() == "[user]\nsummarize service"
    assert manifest["prompts"][0]["prompt_id"] == "prompts:summarize"


@pytest.mark.asyncio
async def test_mcp_center_exports_resources_and_prompts_as_context_materials() -> None:
    center = MCPCenter()
    center.register_server(MCPServerSpec(name="fs", transport="mock"))
    center.register_connector("mock", FakeMCPConnector())
    await center.refresh_inventory()

    result = await center.context_materials(
        MCPContextMaterialRequest(
            query="readme summarize",
            prompt_arguments={"fs:summarize": {"target": "service"}},
            max_resource_bytes=12,
            role="mcp",
            priority=3,
            metadata={"request_id": "m1"},
        )
    )
    manifest = result.manifest()

    assert [material.name for material in result.materials] == [
        "mcp_resource:file://README.md",
        "mcp_prompt:fs:summarize",
    ]
    assert result.materials[0].role == "mcp"
    assert result.materials[0].priority == 3
    assert result.materials[0].metadata["kind"] == "resource"
    assert result.materials[0].metadata["request_id"] == "m1"
    assert len(result.materials[0].content.encode("utf-8")) <= 12
    assert result.materials[1].content == "[user]\nsummarize service"
    assert manifest["schema_version"] == "agent-core-mcp-context-material-result/v1"
    assert manifest["material_count"] == 2
    assert manifest["records"][0]["status"] == "included"
    assert manifest["records"][0]["sha256"]
    assert "fs:file://README.md" not in str(manifest)


@pytest.mark.asyncio
async def test_react_executor_can_use_mcp_center_as_tool_runtime() -> None:
    center = MCPCenter()
    connector = FakeMCPConnector()
    center.register_server(MCPServerSpec(name="fs", transport="mock"))
    center.register_connector("mock", connector)
    await center.refresh()
    provider = MockLLMProvider(
        [
            {
                "action": "call_tool",
                "arguments": {
                    "tool_name": "mcp__fs__read_file",
                    "arguments": {"path": "target.txt"},
                },
            },
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    harness: AgentHarness = InMemoryHarness()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=center,
        action_registry=ActionRegistry(),
        harness=harness,
        config=ReActConfig(max_iterations=3),
    )

    result = await executor.run("read via mcp", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert provider.requests[1].messages[-1].name == "mcp__fs__read_file"
    assert provider.requests[1].messages[-1].content == "fs:read_file:target.txt"


@pytest.mark.asyncio
async def test_react_executor_search_tools_uses_mcp_center_search() -> None:
    center = MCPCenter()
    center.register_server(MCPServerSpec(name="fs", transport="mock", tags=("local",)))
    center.register_connector("mock", FakeMCPConnector())
    await center.refresh()
    provider = MockLLMProvider(
        [
            {"action": "search_tools", "arguments": {"query": "filesystem"}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=center,
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        config=ReActConfig(max_iterations=3),
    )

    result = await executor.run("find mcp tool", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert "mcp__fs__read_file" in provider.requests[1].messages[-1].content

