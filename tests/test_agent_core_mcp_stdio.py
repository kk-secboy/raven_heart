from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent_core.mcp import MCPCenter, MCPServerSpec
from agent_core.mcp_stdio import MCPStdioConnectorConfig, MCPStdioJSONRPCConnector
from agent_core.tools import ToolInvocation


def _write_fake_mcp_server(path: Path) -> None:
    path.write_text(
        r'''
from __future__ import annotations

import json
import sys


def send(payload):
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    request_id = request.get("id")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": request_id, "result": {"protocolVersion": "test"}})
    elif method == "notifications/initialized":
        continue
    elif method == "tools/list":
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo text",
                            "inputSchema": {
                                "type": "object",
                                "required": ["text"],
                                "properties": {"text": {"type": "string"}},
                            },
                        }
                    ]
                },
            }
        )
    elif method == "tools/call":
        params = request.get("params") or {}
        arguments = params.get("arguments") or {}
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": "echo:" + str(arguments.get("text", ""))}],
                    "isError": False,
                },
            }
        )
    elif method == "resources/list":
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "resources": [
                        {
                            "uri": "file://notes.txt",
                            "name": "notes",
                            "description": "Test notes",
                            "mimeType": "text/plain",
                        }
                    ]
                },
            }
        )
    elif method == "resources/read":
        params = request.get("params") or {}
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "contents": [
                        {
                            "uri": params.get("uri"),
                            "mimeType": "text/plain",
                            "text": "resource:" + str(params.get("uri")),
                        }
                    ]
                },
            }
        )
    elif method == "prompts/list":
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "prompts": [
                        {
                            "name": "summarize",
                            "description": "Summarize target",
                            "arguments": [
                                {
                                    "name": "target",
                                    "description": "Target to summarize",
                                    "required": True,
                                }
                            ],
                        }
                    ]
                },
            }
        )
    elif method == "prompts/get":
        params = request.get("params") or {}
        arguments = params.get("arguments") or {}
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "description": "Summarize target",
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": "summarize:" + str(arguments.get("target", "")),
                            },
                        }
                    ],
                },
            }
        )
    else:
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "unknown method"},
            }
        )
'''.strip(),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_stdio_connector_lists_invokes_and_closes_fake_mcp_server(tmp_path) -> None:
    server_path = tmp_path / "fake_mcp_server.py"
    _write_fake_mcp_server(server_path)
    connector = MCPStdioJSONRPCConnector(
        MCPStdioConnectorConfig(request_timeout_seconds=3, startup_timeout_seconds=3)
    )
    server = MCPServerSpec(
        name="fake",
        transport="stdio",
        command=sys.executable,
        args=(str(server_path),),
    )

    await connector.connect(server)
    tools = await connector.list_tools(server)
    prompts = await connector.list_prompts(server)
    prompt = await connector.get_prompt(server, "summarize", {"target": "svc"})
    resources = await connector.list_resources(server)
    content = await connector.read_resource(server, "file://notes.txt")
    result = await connector.invoke_tool(server, "echo", {"text": "hello"})
    await connector.close(server)

    assert [tool.reference.tool_name for tool in tools] == ["echo"]
    assert tools[0].parameters_schema["required"] == ["text"]
    assert [item.name for item in prompts] == ["summarize"]
    assert prompts[0].arguments_schema["required"] == ["target"]
    assert prompt.render() == "[user]\nsummarize:svc"
    assert [resource.uri for resource in resources] == ["file://notes.txt"]
    assert content.text == "resource:file://notes.txt"
    assert result.ok
    assert result.content == "echo:hello"


@pytest.mark.asyncio
async def test_mcp_center_uses_stdio_connector_as_runtime(tmp_path) -> None:
    server_path = tmp_path / "fake_mcp_server.py"
    _write_fake_mcp_server(server_path)
    center = MCPCenter()
    center.register_connector(
        "stdio",
        MCPStdioJSONRPCConnector(
            MCPStdioConnectorConfig(request_timeout_seconds=3, startup_timeout_seconds=3)
        ),
    )
    center.register_server(
        MCPServerSpec(
            name="fake",
            transport="stdio",
            command=sys.executable,
            args=(str(server_path),),
        )
    )

    refreshed = await center.refresh()
    prompts = await center.refresh_prompts()
    prompt = await center.get_prompt("fake:summarize", {"target": "svc"})
    resources = await center.refresh_resources()
    content = await center.read_resource("file://notes.txt")
    result = await center.invoke(
        ToolInvocation(tool_name="mcp__fake__echo", arguments={"text": "from-center"})
    )
    await center.close()

    assert [tool.reference.resolved_public_name() for tool in refreshed] == ["mcp__fake__echo"]
    assert [item.prompt_id() for item in prompts] == ["fake:summarize"]
    assert prompt.render() == "[user]\nsummarize:svc"
    assert [resource.uri for resource in resources] == ["file://notes.txt"]
    assert content.text == "resource:file://notes.txt"
    assert result.ok
    assert result.content == "echo:from-center"
    assert center.state("fake").status == "closed"  # type: ignore[union-attr]

