from __future__ import annotations

from typing import Any

import pytest

from agent_core.actions import ActionRegistry
from agent_core.mcp import MCPCenter, MCPServerSpec, MCPToolReference, MCPToolSpec
from agent_core.prompt import PromptIR
from agent_core.react import ReActConfig, ReActExecutor
from agent_core.testing import InMemoryHarness, MockLLMProvider
from agent_core.tools import ToolCenter, ToolInvocation, ToolRegistry, ToolResult, ToolSpec


class FakeMCPConnector:
    async def list_tools(self, server: MCPServerSpec) -> tuple[MCPToolSpec, ...]:
        return (
            MCPToolSpec(
                reference=MCPToolReference(server_name=server.name, tool_name="read_file"),
                description="Read a file through MCP",
                tags=("filesystem",),
            ),
        )

    async def invoke_tool(
        self,
        server: MCPServerSpec,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        return ToolResult(
            call_id="remote",
            tool_name=tool_name,
            content=f"{server.name}:{tool_name}:{arguments.get('path', '')}",
        )


async def _local_handler(invocation: ToolInvocation) -> ToolResult:
    return ToolResult(
        call_id=invocation.call_id,
        tool_name=invocation.tool_name,
        content=f"local:{invocation.arguments.get('text', '')}",
    )


async def _build_tool_center() -> ToolCenter:
    local = ToolRegistry()
    local.register(
        ToolSpec(
            name="echo",
            description="Echo local text",
            aliases=("say",),
            tags=("local",),
        ),
        _local_handler,
    )
    mcp = MCPCenter()
    mcp.register_server(MCPServerSpec(name="fs", transport="mock", tags=("remote",)))
    mcp.register_connector("mock", FakeMCPConnector())
    await mcp.refresh()

    center = ToolCenter()
    center.mount("local", local, tags=("builtin",))
    center.mount("mcp", mcp, tags=("external",))
    return center


@pytest.mark.asyncio
async def test_tool_center_mounts_searches_manifests_and_invokes_runtimes() -> None:
    center = await _build_tool_center()

    specs = center.specs()
    search = center.search("filesystem remote")
    route_plan = center.route_plan("say")
    local_result = await center.invoke(ToolInvocation(tool_name="say", arguments={"text": "hi"}))
    mcp_result = await center.invoke(
        ToolInvocation(tool_name="mcp__fs__read_file", arguments={"path": "README.md"})
    )
    manifest = center.manifest()

    assert [spec.name for spec in specs] == ["echo", "mcp__fs__read_file"]
    assert "builtin" in specs[0].tags
    assert search[0].name == "mcp__fs__read_file"
    assert route_plan.ready
    assert route_plan.selected_mount == "local"
    assert route_plan.selected_tool_name == "echo"
    assert route_plan.candidates[0].matched_name == "say"
    assert local_result.content == "local:hi"
    assert local_result.tool_name == "echo"
    assert local_result.metadata["tool_center"]["requested_tool_name"] == "say"
    assert mcp_result.content == "fs:read_file:README.md"
    assert mcp_result.metadata["tool_center"]["mount"] == "mcp"
    assert manifest["schema_version"] == "agent-core-tool-center/v1"
    assert [mount["name"] for mount in manifest["mounts"]] == ["local", "mcp"]
    assert manifest["call_count"] == 2
    assert manifest["calls"][0]["route_plan"]["selected_mount"] == "local"
    assert manifest["calls"][0]["route_plan"]["selected_tool_name"] == "echo"
    assert manifest["calls"][0]["result"]["content_sha256"]
    assert "local:hi" not in str(manifest)
    assert manifest["calls"][1]["route_plan"]["selected_mount"] == "mcp"


@pytest.mark.asyncio
async def test_tool_center_route_plan_and_call_records_unknown_or_disabled_tools() -> None:
    registry = ToolRegistry()
    registry.register(ToolSpec(name="disabled", enabled=False), _local_handler)
    center = ToolCenter()
    center.mount("local", registry)

    missing_plan = center.route_plan("missing")
    disabled_plan = center.route_plan("disabled")
    missing = await center.invoke(ToolInvocation(tool_name="missing"))
    disabled = await center.invoke(ToolInvocation(tool_name="disabled"))
    manifest = center.manifest()

    assert not missing_plan.ready
    assert missing_plan.candidates == ()
    assert not disabled_plan.ready
    assert disabled_plan.candidates[0].reason == "disabled"
    assert missing.status == "failed"
    assert missing.error == "unknown tool: missing"
    assert disabled.status == "failed"
    assert disabled.error == "tool disabled: disabled"
    assert manifest["call_count"] == 2
    assert [call["error"] for call in manifest["calls"]] == [
        "unknown tool: missing",
        "tool disabled: disabled",
    ]
    assert manifest["calls"][1]["route_plan"]["candidates"][0]["enabled"] is False


@pytest.mark.asyncio
async def test_tool_center_rejects_name_and_alias_collisions() -> None:
    first = ToolRegistry()
    second = ToolRegistry()
    first.register(ToolSpec(name="scan", aliases=("probe",)), _local_handler)
    second.register(ToolSpec(name="probe"), _local_handler)
    center = ToolCenter()
    center.mount("first", first)

    with pytest.raises(ValueError, match="tool name collision"):
        center.mount("second", second)


@pytest.mark.asyncio
async def test_tool_center_refreshes_mounted_mcp_center() -> None:
    mcp = MCPCenter()
    mcp.register_server(MCPServerSpec(name="fs", transport="mock"))
    mcp.register_connector("mock", FakeMCPConnector())
    center = ToolCenter()
    center.mount("mcp", mcp)

    await center.refresh()

    assert [spec.name for spec in center.specs()] == ["mcp__fs__read_file"]
    assert mcp.state("fs").status == "refreshed"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_react_executor_uses_tool_center_for_local_and_mcp_tools() -> None:
    center = await _build_tool_center()
    provider = MockLLMProvider(
        [
            {"action": "search_tools", "arguments": {"query": "filesystem"}},
            {"action": "say", "arguments": {"text": "hello"}},
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
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=center,
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        config=ReActConfig(max_iterations=5),
    )

    result = await executor.run("mixed tools", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert "mcp__fs__read_file" in provider.requests[1].messages[-1].content
    assert provider.requests[2].messages[-1].name == "echo"
    assert provider.requests[2].messages[-1].content == "local:hello"
    assert provider.requests[3].messages[-1].name == "mcp__fs__read_file"
    assert provider.requests[3].messages[-1].content == "fs:read_file:target.txt"

