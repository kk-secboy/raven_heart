"""Provider conformance checks for external LLMProviderPort implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from agent_core.providers import (
    LLMMessage,
    LLMProviderError,
    LLMProviderPort,
    LLMRequest,
    LLMResponse,
    LLMResponseFormat,
    LLMStreamEvent,
    LLMToolCall,
    LLMToolChoice,
    LLMToolContract,
    UsageInfo,
)


@dataclass(frozen=True)
class AgentCoreProviderConformanceSpec:
    """Provider-neutral conformance checks to run against one provider."""

    provider_name: str = "provider"
    model: str = "conformance-mini"
    require_text: bool = True
    require_streaming: bool = True
    require_json_mode: bool = True
    require_tool_calls: bool = True
    max_output_tokens: int = 64
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-provider-conformance-spec/v1",
            "provider_name": self.provider_name,
            "model": self.model,
            "require_text": self.require_text,
            "require_streaming": self.require_streaming,
            "require_json_mode": self.require_json_mode,
            "require_tool_calls": self.require_tool_calls,
            "max_output_tokens": self.max_output_tokens,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreProviderConformanceIssue:
    """One provider conformance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-provider-conformance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreProviderConformanceReport:
    """Prompt-safe result of provider conformance checks."""

    status: str
    spec: dict[str, Any] = field(default_factory=dict)
    text: dict[str, Any] = field(default_factory=dict)
    streaming: dict[str, Any] = field(default_factory=dict)
    json_mode: dict[str, Any] = field(default_factory=dict)
    tool_calls: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreProviderConformanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-provider-conformance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "spec": dict(self.spec),
            "text": dict(self.text),
            "streaming": dict(self.streaming),
            "json_mode": dict(self.json_mode),
            "tool_calls": dict(self.tool_calls),
            "metadata": dict(self.metadata),
        }


class ProviderConformanceDeterministicProvider:
    """Deterministic provider used when no external provider is supplied."""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self.stream_requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if request.tools:
            return LLMResponse(
                tool_calls=(
                    LLMToolCall(
                        tool_name=request.tools[0].name,
                        arguments={"query": "conformance"},
                        call_id="conformance-tool-call",
                    ),
                ),
                finish_reason="tool_calls",
                usage=UsageInfo(total_tokens=9),
            )
        if request.response_format is not None:
            return LLMResponse(
                content='{"status":"ok","source":"provider-conformance"}',
                finish_reason="stop",
                usage=UsageInfo(total_tokens=8),
            )
        return LLMResponse(
            content="raven-heart-provider-ok",
            finish_reason="stop",
            usage=UsageInfo(total_tokens=5),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self.stream_requests.append(request)
        yield LLMStreamEvent(type="delta", delta="raven-heart")
        yield LLMStreamEvent(type="usage", usage=UsageInfo(total_tokens=6))
        yield LLMStreamEvent(type="message_end")


@dataclass(frozen=True)
class AgentCoreProviderConformanceHarness:
    """Run provider-neutral conformance checks against one provider implementation."""

    provider: LLMProviderPort | None = None
    spec: AgentCoreProviderConformanceSpec = field(
        default_factory=AgentCoreProviderConformanceSpec
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreProviderConformanceReport:
        provider = self.provider or ProviderConformanceDeterministicProvider()
        text = await _check_text(provider, self.spec) if self.spec.require_text else _skipped("text")
        streaming = (
            await _check_streaming(provider, self.spec)
            if self.spec.require_streaming
            else _skipped("streaming")
        )
        json_mode = (
            await _check_json_mode(provider, self.spec)
            if self.spec.require_json_mode
            else _skipped("json_mode")
        )
        tool_calls = (
            await _check_tool_calls(provider, self.spec)
            if self.spec.require_tool_calls
            else _skipped("tool_calls")
        )
        issues = _provider_conformance_issues(
            spec=self.spec,
            text=text,
            streaming=streaming,
            json_mode=json_mode,
            tool_calls=tool_calls,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreProviderConformanceReport(
            status=status,
            spec=self.spec.manifest(),
            text=text,
            streaming=streaming,
            json_mode=json_mode,
            tool_calls=tool_calls,
            issues=issues,
            metadata={"scenario": "provider_conformance", **dict(self.metadata)},
        )


async def run_agent_core_provider_conformance(
    *,
    provider: LLMProviderPort | None = None,
    spec: AgentCoreProviderConformanceSpec | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreProviderConformanceReport:
    """Run provider conformance checks.

    Passing ``provider=None`` uses a deterministic no-network provider so the
    package-level gate remains hermetic. Runtime/provider adapter packages can
    pass a real provider implementation and the same spec when credentials are
    available.
    """

    return await AgentCoreProviderConformanceHarness(
        provider=provider,
        spec=spec or AgentCoreProviderConformanceSpec(),
        metadata=dict(metadata or {}),
    ).run()


async def _check_text(
    provider: LLMProviderPort,
    spec: AgentCoreProviderConformanceSpec,
) -> dict[str, Any]:
    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content="Return a short conformance token."),
            LLMMessage(role="user", content="Return raven-heart-provider-ok."),
        ],
        model=spec.model,
        max_output_tokens=spec.max_output_tokens,
        metadata={"conformance_check": "text", **dict(spec.metadata)},
    )
    try:
        response = await provider.complete(request)
    except Exception as exc:
        return _failed("text", exc)
    return {
        "schema_version": "agent-core-provider-conformance-check/v1",
        "name": "text",
        "status": "completed",
        "content_bytes": len(response.content.encode("utf-8")),
        "finish_reason": response.finish_reason,
        "tool_call_count": len(response.tool_calls),
        "usage": response.usage.manifest(),
        "response_manifest": response.manifest(),
        "request_manifest": request.manifest(),
    }


async def _check_streaming(
    provider: LLMProviderPort,
    spec: AgentCoreProviderConformanceSpec,
) -> dict[str, Any]:
    request = LLMRequest(
        messages=[LLMMessage(role="user", content="Stream a short conformance token.")],
        model=spec.model,
        max_output_tokens=spec.max_output_tokens,
        metadata={"conformance_check": "streaming", **dict(spec.metadata)},
    )
    events: list[LLMStreamEvent] = []
    try:
        async for event in provider.stream(request):
            events.append(event)
    except Exception as exc:
        return _failed("streaming", exc)
    return {
        "schema_version": "agent-core-provider-conformance-check/v1",
        "name": "streaming",
        "status": "completed",
        "event_count": len(events),
        "event_types": [event.type for event in events],
        "delta_bytes": sum(len(event.delta.encode("utf-8")) for event in events),
        "usage_total_tokens": sum(
            event.usage.total_tokens for event in events if event.usage is not None
        ),
        "events": [event.manifest() for event in events],
        "request_manifest": request.manifest(),
    }


async def _check_json_mode(
    provider: LLMProviderPort,
    spec: AgentCoreProviderConformanceSpec,
) -> dict[str, Any]:
    request = LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content='Return JSON only: {"status":"ok","source":"provider-conformance"}.',
            )
        ],
        model=spec.model,
        max_output_tokens=spec.max_output_tokens,
        response_format=LLMResponseFormat(kind="json"),
        metadata={
            "conformance_check": "json_mode",
            "requires_json_mode": True,
            **dict(spec.metadata),
        },
    )
    try:
        response = await provider.complete(request)
    except Exception as exc:
        return _failed("json_mode", exc)
    parsed_ok = False
    parsed_keys: list[str] = []
    try:
        parsed = json.loads(response.content)
        parsed_ok = isinstance(parsed, dict)
        parsed_keys = sorted(str(key) for key in parsed) if isinstance(parsed, dict) else []
    except json.JSONDecodeError:
        parsed_ok = False
    return {
        "schema_version": "agent-core-provider-conformance-check/v1",
        "name": "json_mode",
        "status": "completed",
        "content_bytes": len(response.content.encode("utf-8")),
        "parsed_ok": parsed_ok,
        "parsed_keys": parsed_keys,
        "finish_reason": response.finish_reason,
        "usage": response.usage.manifest(),
        "response_manifest": response.manifest(),
        "request_manifest": request.manifest(),
    }


async def _check_tool_calls(
    provider: LLMProviderPort,
    spec: AgentCoreProviderConformanceSpec,
) -> dict[str, Any]:
    tool = LLMToolContract(
        name="lookup",
        description="Lookup one conformance item.",
        parameters_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        strict=True,
    )
    request = LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content="Call the lookup tool with query set to conformance.",
            )
        ],
        model=spec.model,
        max_output_tokens=spec.max_output_tokens,
        tools=(tool,),
        tool_choice=LLMToolChoice(mode="tool", tool_name="lookup"),
        metadata={
            "conformance_check": "tool_calls",
            "requires_tool_calls": True,
            **dict(spec.metadata),
        },
    )
    try:
        response = await provider.complete(request)
    except Exception as exc:
        return _failed("tool_calls", exc)
    return {
        "schema_version": "agent-core-provider-conformance-check/v1",
        "name": "tool_calls",
        "status": "completed",
        "tool_call_count": len(response.tool_calls),
        "tool_call_names": [tool_call.tool_name for tool_call in response.tool_calls],
        "finish_reason": response.finish_reason,
        "usage": response.usage.manifest(),
        "response_manifest": response.manifest(),
        "request_manifest": request.manifest(),
    }


def _skipped(name: str) -> dict[str, Any]:
    return {
        "schema_version": "agent-core-provider-conformance-check/v1",
        "name": name,
        "status": "skipped",
    }


def _failed(name: str, exc: Exception) -> dict[str, Any]:
    retryable = exc.retry_hint.retryable if isinstance(exc, LLMProviderError) else False
    return {
        "schema_version": "agent-core-provider-conformance-check/v1",
        "name": name,
        "status": "failed",
        "error": str(exc),
        "retryable": retryable,
        "error_type": exc.__class__.__name__,
    }


def _provider_conformance_issues(
    *,
    spec: AgentCoreProviderConformanceSpec,
    text: dict[str, Any],
    streaming: dict[str, Any],
    json_mode: dict[str, Any],
    tool_calls: dict[str, Any],
) -> tuple[AgentCoreProviderConformanceIssue, ...]:
    issues: list[AgentCoreProviderConformanceIssue] = []
    if spec.require_text:
        _require_completed(issues, "text", text)
        if int(text.get("content_bytes") or 0) <= 0:
            issues.append(
                AgentCoreProviderConformanceIssue(
                    source="text",
                    code="text_response_empty",
                    message="Provider text response was empty.",
                )
            )
    if spec.require_streaming:
        _require_completed(issues, "streaming", streaming)
        event_types = set(streaming.get("event_types") or ())
        if not ({"delta", "message_end"} & event_types):
            issues.append(
                AgentCoreProviderConformanceIssue(
                    source="streaming",
                    code="streaming_events_missing",
                    message="Provider stream did not emit delta or message_end events.",
                    metadata={"event_types": sorted(event_types)},
                )
            )
    if spec.require_json_mode:
        _require_completed(issues, "json_mode", json_mode)
        if json_mode.get("parsed_ok") is not True:
            issues.append(
                AgentCoreProviderConformanceIssue(
                    source="json_mode",
                    code="json_response_invalid",
                    message="Provider JSON-mode response was not valid JSON.",
                )
            )
    if spec.require_tool_calls:
        _require_completed(issues, "tool_calls", tool_calls)
        if "lookup" not in set(tool_calls.get("tool_call_names") or ()):
            issues.append(
                AgentCoreProviderConformanceIssue(
                    source="tool_calls",
                    code="tool_call_missing",
                    message="Provider did not return the required lookup tool call.",
                    metadata={"tool_call_names": list(tool_calls.get("tool_call_names") or ())},
                )
            )
    return tuple(issues)


def _require_completed(
    issues: list[AgentCoreProviderConformanceIssue],
    name: str,
    result: dict[str, Any],
) -> None:
    if result.get("status") == "completed":
        return
    issues.append(
        AgentCoreProviderConformanceIssue(
            source=name,
            code=f"{name}_check_failed",
            message=f"Provider {name} conformance check failed.",
            metadata={"result": result},
        )
    )
