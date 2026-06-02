"""Minimal stdio JSON-RPC connector for MCP-like servers."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any

from agent_core.mcp import (
    MCPPromptContent,
    MCPPromptSpec,
    MCPResourceContent,
    MCPResourceSpec,
    MCPServerSpec,
    MCPToolReference,
    MCPToolSpec,
)
from agent_core.tools import ToolResult


@dataclass(frozen=True)
class MCPStdioConnectorConfig:
    protocol_version: str = "2025-06-18"
    client_name: str = "raven-agent-core"
    client_version: str = "0.1.0"
    request_timeout_seconds: float = 10.0
    startup_timeout_seconds: float = 10.0
    extra_env: dict[str, str] = field(default_factory=dict)


class MCPStdioJSONRPCConnector:
    """MCP connector over newline-delimited JSON-RPC stdio.

    This adapter is intentionally SDK-free. It gives the core a concrete,
    testable stdio connector while keeping process/session ownership at the
    connector boundary.
    """

    def __init__(self, config: MCPStdioConnectorConfig | None = None) -> None:
        self.config = config or MCPStdioConnectorConfig()
        self._sessions: dict[str, _StdioSession] = {}

    async def connect(self, server: MCPServerSpec) -> None:
        session = await self._session(server)
        if session.initialized:
            return
        await session.request(
            "initialize",
            {
                "protocolVersion": self.config.protocol_version,
                "capabilities": {},
                "clientInfo": {
                    "name": self.config.client_name,
                    "version": self.config.client_version,
                },
            },
            timeout=self.config.startup_timeout_seconds,
        )
        await session.notify("notifications/initialized", {})
        session.initialized = True

    async def list_tools(self, server: MCPServerSpec) -> tuple[MCPToolSpec, ...]:
        session = await self._session(server)
        result = await session.request("tools/list", {}, timeout=self.config.request_timeout_seconds)
        tools = result.get("tools", ()) if isinstance(result, dict) else ()
        specs: list[MCPToolSpec] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name") or "").strip()
            if not name:
                continue
            schema = tool.get("inputSchema") or tool.get("input_schema") or {}
            specs.append(
                MCPToolSpec(
                    reference=MCPToolReference(server_name=server.name, tool_name=name),
                    description=str(tool.get("description") or ""),
                    parameters_schema=schema if isinstance(schema, dict) else {},
                    tags=("stdio",),
                    metadata={"stdio": {"command": server.command}},
                )
            )
        return tuple(specs)

    async def list_prompts(self, server: MCPServerSpec) -> tuple[MCPPromptSpec, ...]:
        session = await self._session(server)
        result = await session.request("prompts/list", {}, timeout=self.config.request_timeout_seconds)
        prompts = result.get("prompts", ()) if isinstance(result, dict) else ()
        specs: list[MCPPromptSpec] = []
        for prompt in prompts:
            if not isinstance(prompt, dict):
                continue
            name = str(prompt.get("name") or "").strip()
            if not name:
                continue
            arguments = prompt.get("arguments") or ()
            specs.append(
                MCPPromptSpec(
                    server_name=server.name,
                    name=name,
                    description=str(prompt.get("description") or ""),
                    arguments_schema=_prompt_arguments_schema(arguments),
                    tags=("stdio",),
                    metadata={"stdio": {"command": server.command}},
                )
            )
        return tuple(specs)

    async def get_prompt(
        self,
        server: MCPServerSpec,
        name: str,
        arguments: dict[str, Any],
    ) -> MCPPromptContent:
        session = await self._session(server)
        result = await session.request(
            "prompts/get",
            {"name": name, "arguments": dict(arguments)},
            timeout=self.config.request_timeout_seconds,
        )
        return _prompt_content_from_result(server.name, name, result)

    async def list_resources(self, server: MCPServerSpec) -> tuple[MCPResourceSpec, ...]:
        session = await self._session(server)
        result = await session.request("resources/list", {}, timeout=self.config.request_timeout_seconds)
        resources = result.get("resources", ()) if isinstance(result, dict) else ()
        specs: list[MCPResourceSpec] = []
        for resource in resources:
            if not isinstance(resource, dict):
                continue
            uri = str(resource.get("uri") or "").strip()
            if not uri:
                continue
            specs.append(
                MCPResourceSpec(
                    server_name=server.name,
                    uri=uri,
                    name=str(resource.get("name") or ""),
                    description=str(resource.get("description") or ""),
                    mime_type=str(resource.get("mimeType") or resource.get("mime_type") or ""),
                    tags=("stdio",),
                    metadata={"stdio": {"command": server.command}},
                )
            )
        return tuple(specs)

    async def read_resource(self, server: MCPServerSpec, uri: str) -> MCPResourceContent:
        session = await self._session(server)
        result = await session.request(
            "resources/read",
            {"uri": uri},
            timeout=self.config.request_timeout_seconds,
        )
        return _resource_content_from_result(server.name, uri, result)

    async def invoke_tool(
        self,
        server: MCPServerSpec,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        session = await self._session(server)
        result = await session.request(
            "tools/call",
            {"name": tool_name, "arguments": dict(arguments)},
            timeout=self.config.request_timeout_seconds,
        )
        content = _render_tool_content(result)
        status = "failed" if isinstance(result, dict) and result.get("isError") else "completed"
        return ToolResult(
            call_id="",
            tool_name=tool_name,
            status=status,
            content=content if status == "completed" else "",
            error=content if status != "completed" else "",
            data=result if isinstance(result, dict) else {"result": result},
            metadata={"stdio": {"server_name": server.name}},
        )

    async def close(self, server: MCPServerSpec) -> None:
        session = self._sessions.pop(server.name, None)
        if session is not None:
            await session.close()

    async def close_all(self) -> None:
        for server_name in tuple(self._sessions):
            session = self._sessions.pop(server_name)
            await session.close()

    async def _session(self, server: MCPServerSpec) -> "_StdioSession":
        existing = self._sessions.get(server.name)
        if existing is not None and not existing.closed:
            return existing
        if not server.command:
            raise ValueError(f"MCP stdio server command is required: {server.name}")
        env = {
            **os.environ,
            **self.config.extra_env,
            **server.env,
            "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE", "1"),
        }
        process = await asyncio.create_subprocess_exec(
            server.command,
            *server.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        session = _StdioSession(server_name=server.name, process=process)
        self._sessions[server.name] = session
        return session


@dataclass
class _StdioSession:
    server_name: str
    process: asyncio.subprocess.Process
    next_id: int = 1
    initialized: bool = False
    closed: bool = False

    async def request(self, method: str, params: dict[str, Any], *, timeout: float) -> Any:
        request_id = self.next_id
        self.next_id += 1
        await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            message = await asyncio.wait_for(self._read(), timeout=timeout)
            if message.get("id") != request_id:
                continue
            if "error" in message:
                error = message["error"]
                if isinstance(error, dict):
                    raise RuntimeError(str(error.get("message") or error))
                raise RuntimeError(str(error))
            return message.get("result", {})

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()

    async def _write(self, message: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise RuntimeError(f"MCP stdio stdin is closed: {self.server_name}")
        raw = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
        self.process.stdin.write(raw)
        await self.process.stdin.drain()

    async def _read(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError(f"MCP stdio stdout is closed: {self.server_name}")
        raw = await self.process.stdout.readline()
        if not raw:
            stderr = ""
            if self.process.stderr is not None:
                try:
                    stderr = (await asyncio.wait_for(self.process.stderr.read(), timeout=0.1)).decode(
                        "utf-8",
                        errors="replace",
                    )
                except asyncio.TimeoutError:
                    stderr = ""
            raise RuntimeError(f"MCP stdio server exited: {self.server_name} {stderr}".strip())
        message = json.loads(raw.decode("utf-8"))
        if not isinstance(message, dict):
            raise RuntimeError("MCP stdio message must be a JSON object")
        return message


def _render_tool_content(result: Any) -> str:
    if not isinstance(result, dict):
        return str(result)
    content = result.get("content")
    if isinstance(content, list):
        rendered: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    rendered.append(str(item["text"]))
                else:
                    rendered.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                rendered.append(str(item))
        return "\n".join(rendered)
    if content is not None:
        return str(content)
    structured = result.get("structuredContent")
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False, sort_keys=True)
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def _prompt_arguments_schema(arguments: Any) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    if isinstance(arguments, list):
        for item in arguments:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            properties[name] = {
                "type": "string",
                "description": str(item.get("description") or ""),
            }
            if item.get("required"):
                required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _prompt_content_from_result(server_name: str, name: str, result: Any) -> MCPPromptContent:
    if not isinstance(result, dict):
        return MCPPromptContent(
            server_name=server_name,
            name=name,
            messages=({"role": "user", "content": str(result)},),
        )
    messages = result.get("messages")
    normalized: list[dict[str, Any]] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                normalized.append({"role": "user", "content": str(message)})
                continue
            normalized.append(
                {
                    "role": str(message.get("role") or "user"),
                    "content": message.get("content", ""),
                }
            )
    return MCPPromptContent(
        server_name=server_name,
        name=name,
        messages=tuple(normalized),
        description=str(result.get("description") or ""),
        metadata={"stdio": True},
    )


def _resource_content_from_result(server_name: str, uri: str, result: Any) -> MCPResourceContent:
    if not isinstance(result, dict):
        return MCPResourceContent(server_name=server_name, uri=uri, text=str(result))
    contents = result.get("contents")
    if isinstance(contents, list) and contents:
        texts: list[str] = []
        blobs: list[bytes] = []
        mime_type = ""
        for item in contents:
            if not isinstance(item, dict):
                texts.append(str(item))
                continue
            mime_type = mime_type or str(item.get("mimeType") or item.get("mime_type") or "")
            if "text" in item:
                texts.append(str(item["text"]))
            elif "blob" in item:
                try:
                    blobs.append(base64.b64decode(str(item["blob"])))
                except ValueError:
                    texts.append(str(item["blob"]))
        return MCPResourceContent(
            server_name=server_name,
            uri=uri,
            text="\n".join(texts),
            blob=b"".join(blobs),
            mime_type=mime_type,
            metadata={"stdio": True},
        )
    text = result.get("text")
    if text is not None:
        return MCPResourceContent(
            server_name=server_name,
            uri=uri,
            text=str(text),
            mime_type=str(result.get("mimeType") or result.get("mime_type") or ""),
            metadata={"stdio": True},
        )
    return MCPResourceContent(
        server_name=server_name,
        uri=uri,
        text=json.dumps(result, ensure_ascii=False, sort_keys=True),
        metadata={"stdio": True},
    )

