"""Runtime adapters that let RavenStorm SDK objects run through agent_core."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any

from agent_core.config import AgentProfile
from agent_core.memory import MemoryCenter, MemoryHit, MemoryPort, MemoryQuery, MemoryWrite
from agent_core.providers import (
    LLMMessage,
    LLMProviderCenter,
    LLMProviderPort,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    UsageInfo,
)
from agent_core.runner import AgentSession
from agent_core.tools import ToolCenter, ToolInvocation, ToolResult, ToolRuntimePort, ToolSpec
from app.openai_agents_runtime.tool_manifest import ToolManifest, manifest_for_tool


@dataclass(frozen=True)
class SDKToolRuntimeAdapterConfig:
    fail_unknown_tools: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeLLMProviderAdapterConfig:
    provider_name: str = "runtime"
    model: str = ""
    temperature: float | None = None
    max_output_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeMemoryAdapterConfig:
    source: str = "runtime_memory"
    project_id: str = ""
    run_id: str = ""
    task_id: str = ""
    search_method: str = "search"
    write_method: str = "write"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeAgentCoreSessionConfig:
    provider_name: str = "runtime"
    tool_mount_name: str = "sdk_tools"
    memory_store_name: str = "runtime_memory"
    default_model: str = ""
    max_retries: int = 0
    max_cost_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeTaskTreeSessionConfig:
    project_id: str = ""
    plan_md_path: str = ""
    provider_name: str = "runtime"
    tool_mount_name: str = "tasktree_tools"
    memory_store_name: str = "runtime_memory"
    default_model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeAdapterReadiness:
    status: str
    reason: str = ""
    tool_count: int = 0
    tool_names: tuple[str, ...] = ()
    manifest: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ravenstorm-agent-core-runtime-readiness/v1",
            "status": self.status,
            "reason": self.reason,
            "tool_count": self.tool_count,
            "tool_names": list(self.tool_names),
            "manifest": dict(self.manifest),
            "metadata": dict(self.metadata),
        }


class RuntimeMemoryAdapter(MemoryPort):
    """Expose runtime memory backends, including WorkingMemory/Graphiti, as MemoryPort."""

    def __init__(
        self,
        backend: Any,
        *,
        config: RuntimeMemoryAdapterConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or RuntimeMemoryAdapterConfig()
        self.search_calls: list[dict[str, Any]] = []
        self.write_calls: list[dict[str, Any]] = []

    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        payload = {
            "query": query.query,
            "limit": query.limit,
            "project_id": str(query.filters.get("project_id") or self.config.project_id),
            "run_id": str(query.filters.get("run_id") or self.config.run_id),
            "task_id": str(query.filters.get("task_id") or self.config.task_id),
            "filters": {
                key: value
                for key, value in query.filters.items()
                if key not in {"project_id", "run_id", "task_id"}
            },
        }
        self.search_calls.append(payload)
        method = getattr(self.backend, self.config.search_method)
        result = await _call_memory_search(method, payload)
        hits = _memory_hits_from_runtime_result(result, source=self.config.source)
        return tuple(hits[: max(0, query.limit)])

    async def write(self, item: MemoryWrite) -> None:
        payload = {
            "content": item.content,
            "summary": item.content,
            "source": item.source or self.config.source,
            "project_id": str(item.metadata.get("project_id") or self.config.project_id),
            "run_id": str(item.metadata.get("run_id") or self.config.run_id),
            "task_id": str(item.metadata.get("task_id") or self.config.task_id),
            "kind": str(item.metadata.get("kind") or "agent_core_memory"),
            "metadata": {
                **self.config.metadata,
                **{
                    key: value
                    for key, value in item.metadata.items()
                    if key not in {"project_id", "run_id", "task_id", "kind"}
                },
            },
        }
        self.write_calls.append(payload)
        method = getattr(self.backend, self.config.write_method)
        await _call_memory_write(method, payload)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "ravenstorm-agent-core-runtime-memory-adapter/v1",
            "source": self.config.source,
            "project_id": self.config.project_id,
            "run_id": self.config.run_id,
            "task_id": self.config.task_id,
            "search_method": self.config.search_method,
            "write_method": self.config.write_method,
            "metadata": dict(self.config.metadata),
        }


class RuntimeLLMProviderAdapter(LLMProviderPort):
    """Expose existing Raven/runtime LLM callables as agent_core providers."""

    def __init__(
        self,
        completion: Any,
        *,
        config: RuntimeLLMProviderAdapterConfig | None = None,
    ) -> None:
        self.completion = completion
        self.config = config or RuntimeLLMProviderAdapterConfig()
        self.requests: list[dict[str, Any]] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        payload = _llm_payload(request, self.config)
        self.requests.append(payload)
        result = await _call_completion(self.completion, payload, request)
        response = _coerce_llm_response(result)
        return LLMResponse(
            content=response.content,
            action=response.action,
            usage=response.usage,
            finish_reason=response.finish_reason,
            metadata={
                **response.metadata,
                "provider": self.config.provider_name,
                "model": payload["model"],
                **self.config.metadata,
            },
        )

    async def stream(self, request: LLMRequest):
        response = await self.complete(request)
        if response.content:
            yield LLMStreamEvent(type="delta", delta=response.content, metadata=response.metadata)
        if response.action is not None:
            yield LLMStreamEvent(type="action", action=response.action, metadata=response.metadata)
        yield LLMStreamEvent(type="usage", usage=response.usage, metadata=response.metadata)
        yield LLMStreamEvent(type="message_end", metadata=response.metadata)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "ravenstorm-agent-core-runtime-llm-adapter/v1",
            "provider_name": self.config.provider_name,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
            "metadata": dict(self.config.metadata),
        }


class SDKToolRuntimeAdapter(ToolRuntimePort):
    """Expose RavenStorm SDK function tools as agent_core tools."""

    def __init__(
        self,
        tools: tuple[Any, ...] | list[Any],
        *,
        config: SDKToolRuntimeAdapterConfig | None = None,
    ) -> None:
        self.config = config or SDKToolRuntimeAdapterConfig()
        self._tools: dict[str, Any] = {}
        for tool in tools:
            name = _tool_name(tool)
            if not name:
                raise ValueError("SDK tool name is required")
            if name in self._tools:
                raise ValueError(f"duplicate SDK tool: {name}")
            self._tools[name] = tool

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(
            sorted(
                (_tool_spec(tool) for tool in self._tools.values()),
                key=lambda item: item.name,
            )
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "ravenstorm-agent-core-sdk-tool-adapter/v1",
            "tools": [
                {
                    **_tool_spec(tool).metadata,
                    "name": _tool_name(tool),
                    "description": str(getattr(tool, "description", "") or ""),
                }
                for tool in sorted(self._tools.values(), key=_tool_name)
            ],
            "metadata": dict(self.config.metadata),
        }

    async def invoke(self, invocation: ToolInvocation) -> ToolResult:
        tool = self._tools.get(invocation.tool_name)
        if tool is None:
            status = "failed" if self.config.fail_unknown_tools else "skipped"
            return ToolResult(
                call_id=invocation.call_id,
                tool_name=invocation.tool_name,
                status=status,
                error=f"unknown SDK tool: {invocation.tool_name}",
                metadata={"adapter": "sdk_tool_runtime"},
            )
        try:
            content = await _invoke_tool(tool, dict(invocation.arguments))
        except Exception as exc:
            return ToolResult(
                call_id=invocation.call_id,
                tool_name=invocation.tool_name,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                metadata={"adapter": "sdk_tool_runtime", "manifest": _manifest_dict(tool)},
            )
        status = "failed" if str(content).startswith("[tool_error]") else "completed"
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            status=status,
            content=str(content),
            error=str(content) if status == "failed" else "",
            metadata={"adapter": "sdk_tool_runtime", "manifest": _manifest_dict(tool)},
        )


def build_agent_core_session_from_sdk_tools(
    *,
    profile: AgentProfile,
    provider: Any,
    sdk_tools: tuple[Any, ...] | list[Any],
    metadata: dict[str, Any] | None = None,
) -> AgentSession:
    """Build an AgentSession from existing SDK tools and an agent_core provider."""

    return AgentSession(
        profile=profile,
        provider=provider,
        tools=SDKToolRuntimeAdapter(sdk_tools, config=SDKToolRuntimeAdapterConfig(metadata=metadata or {})),
        metadata={"adapter": "openai_agents_runtime", **dict(metadata or {})},
    )


def build_agent_core_runtime_session(
    *,
    profile: AgentProfile,
    completion: Any,
    sdk_tools: tuple[Any, ...] | list[Any],
    memory_backend: Any | None = None,
    config: RuntimeAgentCoreSessionConfig | None = None,
    provider_config: RuntimeLLMProviderAdapterConfig | None = None,
    tool_config: SDKToolRuntimeAdapterConfig | None = None,
    memory_config: RuntimeMemoryAdapterConfig | None = None,
) -> AgentSession:
    """Build a center-backed AgentSession from existing Raven runtime components."""

    cfg = config or RuntimeAgentCoreSessionConfig()
    metadata = {"adapter": "openai_agents_runtime", **dict(cfg.metadata)}

    llm = RuntimeLLMProviderAdapter(
        completion,
        config=provider_config
        or RuntimeLLMProviderAdapterConfig(
            provider_name=cfg.provider_name,
            model=cfg.default_model or profile.model,
            metadata=metadata,
        ),
    )
    provider_center = LLMProviderCenter(
        default_provider=cfg.provider_name,
        default_model=cfg.default_model or profile.model,
        max_retries=cfg.max_retries,
        max_cost_usd=cfg.max_cost_usd,
    )
    provider_center.register(
        cfg.provider_name,
        llm,
        default_model=cfg.default_model or profile.model,
        metadata={"adapter": "runtime_llm_provider"},
    )

    tool_center = ToolCenter()
    tool_center.mount(
        cfg.tool_mount_name,
        SDKToolRuntimeAdapter(
            sdk_tools,
            config=tool_config or SDKToolRuntimeAdapterConfig(metadata=metadata),
        ),
        tags=("sdk", "runtime"),
        metadata={"adapter": "sdk_tool_runtime"},
    )

    session = AgentSession(
        profile=profile,
        provider=provider_center,
        tools=tool_center,
        metadata=metadata,
    )
    if memory_backend is not None:
        memory_center = MemoryCenter(default_store=cfg.memory_store_name)
        memory_center.register(
            cfg.memory_store_name,
            RuntimeMemoryAdapter(
                memory_backend,
                config=memory_config
                or RuntimeMemoryAdapterConfig(
                    source=cfg.memory_store_name,
                    metadata=metadata,
                ),
            ),
            tags=("runtime",),
            metadata={"adapter": "runtime_memory"},
        )
        session.memory = memory_center
    return session


def build_agent_core_tasktree_runtime_session(
    *,
    profile: AgentProfile,
    completion: Any,
    memory_backend: Any | None = None,
    config: RuntimeTaskTreeSessionConfig | None = None,
    tool_builder: Any | None = None,
) -> AgentSession:
    """Build an AgentSession using Raven's TaskTree SDK tool surface.

    The real TaskTree tool factory is imported lazily because it depends on the
    OpenAI Agents SDK and pydantic. Tests and lightweight runtimes can inject a
    compatible tool_builder without importing those dependencies.
    """

    cfg = config or RuntimeTaskTreeSessionConfig()
    builder = tool_builder or _load_tasktree_tool_builder()
    tools_result = builder(plan_md_path=cfg.plan_md_path, project_id=cfg.project_id)
    sdk_tools = tools_result[0] if isinstance(tools_result, tuple) else tools_result
    return build_agent_core_runtime_session(
        profile=profile,
        completion=completion,
        sdk_tools=list(sdk_tools or []),
        memory_backend=memory_backend,
        config=RuntimeAgentCoreSessionConfig(
            provider_name=cfg.provider_name,
            tool_mount_name=cfg.tool_mount_name,
            memory_store_name=cfg.memory_store_name,
            default_model=cfg.default_model or profile.model,
            metadata={
                "adapter": "openai_agents_runtime",
                "tasktree": True,
                "project_id": cfg.project_id,
                **dict(cfg.metadata),
            },
        ),
        memory_config=(
            RuntimeMemoryAdapterConfig(
                source=cfg.memory_store_name,
                project_id=cfg.project_id,
                metadata={"tasktree": True},
            )
            if memory_backend is not None
            else None
        ),
    )


def inspect_agent_core_tasktree_runtime(
    *,
    config: RuntimeTaskTreeSessionConfig | None = None,
    tool_builder: Any | None = None,
) -> RuntimeAdapterReadiness:
    """Probe Raven's TaskTree SDK tool surface without constructing a run session."""

    cfg = config or RuntimeTaskTreeSessionConfig()
    metadata = {
        "adapter": "openai_agents_runtime",
        "tasktree": True,
        "project_id": cfg.project_id,
        **dict(cfg.metadata),
    }
    try:
        builder = tool_builder or _load_tasktree_tool_builder()
        tools_result = builder(plan_md_path=cfg.plan_md_path, project_id=cfg.project_id)
        sdk_tools = tools_result[0] if isinstance(tools_result, tuple) else tools_result
        adapter = SDKToolRuntimeAdapter(
            list(sdk_tools or []),
            config=SDKToolRuntimeAdapterConfig(metadata=metadata),
        )
        specs = adapter.specs()
        return RuntimeAdapterReadiness(
            status="available",
            reason="tasktree tool surface is available",
            tool_count=len(specs),
            tool_names=tuple(spec.name for spec in specs),
            manifest=adapter.manifest(),
            metadata=metadata,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        return RuntimeAdapterReadiness(
            status="unavailable",
            reason=f"{type(exc).__name__}: {exc}",
            metadata={**metadata, "error_type": type(exc).__name__},
        )
    except Exception as exc:
        return RuntimeAdapterReadiness(
            status="unavailable",
            reason=f"{type(exc).__name__}: {exc}",
            metadata={**metadata, "error_type": type(exc).__name__},
        )


def _tool_spec(tool: Any) -> ToolSpec:
    manifest = manifest_for_tool(tool)
    schema = (
        getattr(tool, "params_json_schema", None)
        or getattr(tool, "input_json_schema", None)
        or getattr(tool, "params_schema", None)
        or {}
    )
    return ToolSpec(
        name=_tool_name(tool),
        description=str(getattr(tool, "description", "") or ""),
        parameters_schema=dict(schema) if isinstance(schema, dict) else {},
        tags=(manifest.capability,),
        enabled=_runtime_metadata(tool) is not None or getattr(tool, "agent_core_coroutine", None) is not None,
        metadata={
            "adapter": "sdk_tool_runtime",
            "manifest": manifest.to_dict(),
            "runtime_metadata_available": _runtime_metadata(tool) is not None,
        },
    )


def _llm_payload(request: LLMRequest, config: RuntimeLLMProviderAdapterConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model or config.model,
        "messages": [_message_payload(message) for message in request.messages],
        "metadata": {**config.metadata, **request.metadata},
    }
    temperature = request.temperature if request.temperature is not None else config.temperature
    if temperature is not None:
        payload["temperature"] = temperature
    max_tokens = request.max_output_tokens or config.max_output_tokens
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def _message_payload(message: LLMMessage) -> dict[str, Any]:
    payload = {
        "role": message.role,
        "content": message.content,
    }
    if message.name:
        payload["name"] = message.name
    if message.metadata:
        payload["metadata"] = dict(message.metadata)
    return payload


async def _call_completion(completion: Any, payload: dict[str, Any], request: LLMRequest) -> Any:
    if callable(completion):
        return await _maybe_await(completion(payload))
    complete = getattr(completion, "complete", None)
    if callable(complete):
        try:
            return await _maybe_await(complete(request))
        except TypeError:
            return await _maybe_await(complete(payload))
    create = _chat_completion_create(completion)
    if callable(create):
        kwargs = {key: value for key, value in payload.items() if key != "metadata"}
        return await _maybe_await(create(**kwargs))
    raise TypeError("completion must be callable, expose complete(), or expose chat.completions.create()")


async def _call_memory_search(method: Any, payload: dict[str, Any]) -> Any:
    try:
        return await _maybe_await(
            method(
                payload["query"],
                project_id=payload["project_id"],
                run_id=payload["run_id"],
                task_id=payload["task_id"],
                limit=payload["limit"],
            )
        )
    except TypeError:
        try:
            return await _maybe_await(method(payload))
        except TypeError:
            return await _maybe_await(method(query=payload["query"], limit=payload["limit"]))


async def _call_memory_write(method: Any, payload: dict[str, Any]) -> Any:
    try:
        return await _maybe_await(method(payload))
    except TypeError:
        return await _maybe_await(
            method(
                content=payload["content"],
                project_id=payload["project_id"],
                run_id=payload["run_id"],
                task_id=payload["task_id"],
                metadata=payload["metadata"],
            )
        )


def _memory_hits_from_runtime_result(value: Any, *, source: str) -> list[MemoryHit]:
    if isinstance(value, tuple):
        return _memory_hits_from_sequence(list(value), source=source)
    if isinstance(value, list):
        return _memory_hits_from_sequence(value, source=source)
    if isinstance(value, dict):
        results = value.get("results") or value.get("facts") or value.get("episodes") or value.get("items")
        if isinstance(results, list):
            return _memory_hits_from_sequence(results, source=source, parent=value)
        content = value.get("content") or value.get("summary") or value.get("text")
        if content:
            return [_memory_hit_from_mapping(value, source=source)]
        return []
    if value:
        return [MemoryHit(content=str(value), source=source)]
    return []


def _memory_hits_from_sequence(
    rows: list[Any],
    *,
    source: str,
    parent: dict[str, Any] | None = None,
) -> list[MemoryHit]:
    hits: list[MemoryHit] = []
    for index, row in enumerate(rows):
        if isinstance(row, MemoryHit):
            hits.append(row)
        elif isinstance(row, dict):
            hits.append(_memory_hit_from_mapping(row, source=source, parent=parent, rank=index))
        else:
            content = str(getattr(row, "content", "") or getattr(row, "summary", "") or getattr(row, "text", "") or row)
            hits.append(
                MemoryHit(
                    content=content,
                    score=float(getattr(row, "score", 0.0) or 0.0),
                    source=str(getattr(row, "source", "") or source),
                    metadata={"rank": index},
                )
            )
    return hits


def _memory_hit_from_mapping(
    row: dict[str, Any],
    *,
    source: str,
    parent: dict[str, Any] | None = None,
    rank: int = 0,
) -> MemoryHit:
    metadata = dict(row.get("metadata") or {})
    for key in (
        "memory_id",
        "run_id",
        "task_id",
        "kind",
        "content_hash",
        "quality_score",
        "triage_reason",
        "recall_backend",
        "created_at",
        "uuid",
        "name",
    ):
        if key in row and row.get(key) not in (None, ""):
            metadata[key] = row.get(key)
    if parent is not None:
        metadata.setdefault("runtime_status", parent.get("status"))
        metadata.setdefault("runtime_policy", parent.get("policy"))
    metadata.setdefault("rank", rank)
    return MemoryHit(
        content=str(row.get("content") or row.get("summary") or row.get("text") or row.get("fact") or ""),
        score=float(row.get("score") or row.get("quality_score") or 0.0),
        source=str(row.get("source") or row.get("kind") or source),
        metadata=metadata,
    )


def _chat_completion_create(completion: Any) -> Any:
    chat = getattr(completion, "chat", None)
    completions = getattr(chat, "completions", None)
    return getattr(completions, "create", None)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _coerce_llm_response(value: Any) -> LLMResponse:
    if isinstance(value, LLMResponse):
        return value
    if isinstance(value, str):
        return LLMResponse(content=value)
    if isinstance(value, dict):
        return _response_from_dict(value)
    content = _content_from_openai_like(value)
    usage = _usage_from_any(getattr(value, "usage", None))
    finish_reason = _finish_reason_from_any(value)
    return LLMResponse(content=content, usage=usage, finish_reason=finish_reason)


def _response_from_dict(value: dict[str, Any]) -> LLMResponse:
    if isinstance(value.get("action"), dict):
        action = dict(value["action"])
    elif isinstance(value.get("action"), str):
        action = {
            "action": str(value["action"]),
            "arguments": dict(value.get("arguments") or {}),
        }
    else:
        action = None
    content = str(value.get("content") or value.get("text") or "")
    if not content and isinstance(value.get("choices"), list) and value["choices"]:
        content = _content_from_choice(value["choices"][0])
    return LLMResponse(
        content=content,
        action=action,
        usage=_usage_from_any(value.get("usage")),
        finish_reason=str(value.get("finish_reason") or ""),
        metadata=dict(value.get("metadata") or {}),
    )


def _content_from_openai_like(value: Any) -> str:
    choices = getattr(value, "choices", None)
    if not choices:
        return str(getattr(value, "content", "") or getattr(value, "text", "") or "")
    return _content_from_choice(choices[0])


def _content_from_choice(choice: Any) -> str:
    if isinstance(choice, dict):
        message = choice.get("message") or {}
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return str(choice.get("text") or "")
    message = getattr(choice, "message", None)
    if message is not None:
        return str(getattr(message, "content", "") or "")
    return str(getattr(choice, "text", "") or "")


def _finish_reason_from_any(value: Any) -> str:
    reason = getattr(value, "finish_reason", "")
    if reason:
        return str(reason)
    choices = getattr(value, "choices", None)
    if choices:
        choice = choices[0]
        if isinstance(choice, dict):
            return str(choice.get("finish_reason") or "")
        return str(getattr(choice, "finish_reason", "") or "")
    return ""


def _usage_from_any(value: Any) -> UsageInfo:
    if isinstance(value, UsageInfo):
        return value
    if value is None:
        return UsageInfo()
    if isinstance(value, dict):
        return UsageInfo(
            input_tokens=int(value.get("input_tokens") or value.get("prompt_tokens") or 0),
            output_tokens=int(value.get("output_tokens") or value.get("completion_tokens") or 0),
            total_tokens=int(value.get("total_tokens") or 0),
            cost_usd=float(value.get("cost_usd") or 0.0),
            metadata={key: item for key, item in value.items() if key not in {"input_tokens", "prompt_tokens", "output_tokens", "completion_tokens", "total_tokens", "cost_usd"}},
        )
    return UsageInfo(
        input_tokens=int(getattr(value, "input_tokens", 0) or getattr(value, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(value, "output_tokens", 0) or getattr(value, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(value, "total_tokens", 0) or 0),
        cost_usd=float(getattr(value, "cost_usd", 0.0) or 0.0),
    )


def _manifest_dict(tool: Any) -> dict[str, Any]:
    manifest = manifest_for_tool(tool)
    if isinstance(manifest, ToolManifest):
        return manifest.to_dict()
    return {}


def _tool_name(tool: Any) -> str:
    return str(
        getattr(tool, "name", None)
        or getattr(tool, "tool_name", None)
        or getattr(tool, "__name__", None)
        or tool.__class__.__name__
    ).strip()


async def _invoke_tool(tool: Any, arguments: dict[str, Any]) -> str:
    invoke_sdk_tool = _invoke_sdk_tool()
    if invoke_sdk_tool is not None and _runtime_metadata(tool) is not None:
        return str(await invoke_sdk_tool(tool, **arguments))

    coroutine = getattr(tool, "agent_core_coroutine", None)
    if coroutine is None:
        metadata = _runtime_metadata(tool)
        coroutine = getattr(metadata, "coroutine", None) if metadata is not None else None
    if coroutine is None:
        raise RuntimeError(f"Tool {_tool_name(tool)!r} is not invokable by agent_core adapter")
    value = coroutine(**arguments)
    if inspect.isawaitable(value):
        value = await value
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _invoke_sdk_tool():
    try:
        from app.openai_agents_runtime.tooling import invoke_sdk_tool
    except ModuleNotFoundError:
        return None
    return invoke_sdk_tool


def _load_tasktree_tool_builder():
    from app.tools.task_tracker import create_task_tracker_tools

    return create_task_tracker_tools


def _runtime_metadata(tool: Any) -> Any | None:
    try:
        from app.openai_agents_runtime.tooling import tool_runtime_metadata
    except ModuleNotFoundError:
        return getattr(tool, "agent_core_runtime_metadata", None)
    return tool_runtime_metadata(tool) or getattr(tool, "agent_core_runtime_metadata", None)

