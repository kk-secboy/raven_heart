"""SDK-level LLM provider compatibility acceptance checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from agent_core.providers import (
    LLMContentPart,
    LLMMessage,
    LLMModelCapabilities,
    LLMProviderCenter,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMResponseFormat,
    LLMStreamEvent,
    LLMToolChoice,
    LLMToolContract,
    OpenAICompatibleLLMProviderCodec,
    RetryHint,
    TransportLLMProvider,
    UsageInfo,
)


@dataclass(frozen=True)
class AgentCoreProviderAcceptanceIssue:
    """One blocking issue from the provider compatibility acceptance scenario."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-provider-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreProviderAcceptanceReport:
    """Prompt-safe provider compatibility acceptance report."""

    status: str
    route_matrix: dict[str, Any] = field(default_factory=dict)
    codec_matrix: dict[str, Any] = field(default_factory=dict)
    transport_matrix: dict[str, Any] = field(default_factory=dict)
    streaming_matrix: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreProviderAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-provider-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "route_matrix": dict(self.route_matrix),
            "codec_matrix": dict(self.codec_matrix),
            "transport_matrix": dict(self.transport_matrix),
            "streaming_matrix": dict(self.streaming_matrix),
            "metadata": dict(self.metadata),
        }


class ProviderAcceptanceLLM:
    """Deterministic provider for compatibility checks."""

    def __init__(self, content: str = "ok") -> None:
        self.content = content
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(content=self.content, usage=UsageInfo(total_tokens=3))


class ProviderAcceptanceStreamingLLM:
    """Deterministic streaming provider for compatibility checks."""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("streaming acceptance should not call complete")

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self.requests.append(request)
        yield LLMStreamEvent(type="delta", delta="stream")
        yield LLMStreamEvent(type="usage", usage=UsageInfo(total_tokens=5))
        yield LLMStreamEvent(type="message_end")


class ProviderAcceptanceTransport:
    """Dict transport used by transport/provider-center acceptance checks."""

    def __init__(
        self,
        *,
        response: dict[str, Any] | None = None,
        events: tuple[dict[str, Any], ...] = (),
    ) -> None:
        self.response = dict(response or {})
        self.events = tuple(dict(item) for item in events)
        self.complete_payloads: list[dict[str, Any]] = []
        self.stream_payloads: list[dict[str, Any]] = []

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.complete_payloads.append(payload)
        return dict(self.response)

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        self.stream_payloads.append(payload)
        for event in self.events:
            yield dict(event)

    def manifest(self) -> dict[str, Any]:
        return {"schema_version": "agent-core-provider-acceptance-transport/v1"}


@dataclass(frozen=True)
class AgentCoreProviderAcceptanceHarness:
    """Run deterministic provider compatibility matrix checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreProviderAcceptanceReport:
        route_matrix = await _provider_route_matrix()
        codec_matrix = _provider_codec_matrix()
        transport_matrix = await _provider_transport_matrix()
        streaming_matrix = await _provider_streaming_matrix()
        issues = _provider_acceptance_issues(
            route_matrix=route_matrix,
            codec_matrix=codec_matrix,
            transport_matrix=transport_matrix,
            streaming_matrix=streaming_matrix,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreProviderAcceptanceReport(
            status=status,
            route_matrix=route_matrix,
            codec_matrix=codec_matrix,
            transport_matrix=transport_matrix,
            streaming_matrix=streaming_matrix,
            issues=issues,
            metadata={"scenario": "agent_core_provider_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_provider_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreProviderAcceptanceReport:
    """Run the default provider compatibility acceptance checks."""

    return await AgentCoreProviderAcceptanceHarness(metadata=dict(metadata or {})).run()


async def _provider_route_matrix() -> dict[str, Any]:
    text = ProviderAcceptanceLLM("text-ok")
    vision = ProviderAcceptanceLLM("vision-ok")
    structured = ProviderAcceptanceLLM("structured-ok")
    tools = ProviderAcceptanceLLM("tools-ok")
    center = LLMProviderCenter(default_provider="text")
    center.register(
        "text",
        text,
        default_model="text-mini",
        priority=30,
        default_capabilities=LLMModelCapabilities(
            context_window_tokens=1024,
            max_output_tokens=128,
            modalities=("text",),
        ),
    )
    center.register(
        "vision",
        vision,
        default_model="vision-mini",
        priority=20,
        default_capabilities=LLMModelCapabilities(
            context_window_tokens=2048,
            max_output_tokens=256,
            modalities=("text", "image"),
        ),
    )
    center.register(
        "structured",
        structured,
        default_model="structured-mini",
        priority=10,
        default_capabilities=LLMModelCapabilities(
            context_window_tokens=2048,
            max_output_tokens=64,
            supports_json_mode=True,
            supports_structured_output=True,
        ),
    )
    center.register(
        "tools",
        tools,
        default_model="tools-mini",
        priority=5,
        default_capabilities=LLMModelCapabilities(
            context_window_tokens=2048,
            max_output_tokens=128,
            supports_tool_calls=True,
        ),
    )
    requests = {
        "text": LLMRequest(messages=[LLMMessage(role="user", content="hello")]),
        "vision": LLMRequest(
            messages=[
                LLMMessage(
                    role="user",
                    content="inspect",
                    content_parts=(
                        LLMContentPart(kind="image", uri="file://finding.png"),
                    ),
                )
            ]
        ),
        "structured": LLMRequest(
            messages=[LLMMessage(role="user", content="json")],
            max_output_tokens=256,
            response_format=LLMResponseFormat(
                kind="json_schema",
                name="finding",
                schema={"type": "object"},
            ),
        ),
        "tools": LLMRequest(
            messages=[LLMMessage(role="user", content="lookup")],
            tools=(
                LLMToolContract(
                    name="lookup",
                    parameters_schema={"type": "object"},
                ),
            ),
            tool_choice=LLMToolChoice(mode="auto"),
        ),
    }
    plans = {name: center.route_plan(request).manifest() for name, request in requests.items()}
    responses = {
        name: (await center.complete(request)).manifest()
        for name, request in requests.items()
    }
    manifest = center.manifest()
    return {
        "schema_version": "agent-core-provider-route-acceptance/v1",
        "ready": all(plan["ready"] for plan in plans.values()),
        "selected": {
            name: (plan.get("selected_route") or {}).get("provider_name")
            for name, plan in plans.items()
        },
        "shape_adjusted": {
            name: bool(
                ((plan.get("selected_route") or {}).get("metadata") or {})
                .get("request_shape_plan", {})
                .get("adjusted")
            )
            for name, plan in plans.items()
        },
        "candidate_reasons": {
            name: {
                candidate["provider_name"]: candidate["reason"]
                for candidate in plan.get("candidates", ())
            }
            for name, plan in plans.items()
        },
        "responses": responses,
        "provider_call_count": int(manifest.get("call_count") or 0),
        "provider_names": [item["name"] for item in manifest.get("providers", ())],
    }


def _provider_codec_matrix() -> dict[str, Any]:
    codec = OpenAICompatibleLLMProviderCodec()
    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content="rules"),
            LLMMessage(
                role="user",
                content="inspect",
                content_parts=(LLMContentPart(kind="image", uri="file://finding.png"),),
            ),
        ],
        model="gpt-compatible",
        max_output_tokens=128,
        tools=(LLMToolContract(name="lookup", parameters_schema={"type": "object"}),),
        tool_choice=LLMToolChoice(mode="tool", tool_name="lookup"),
        response_format=LLMResponseFormat(
            kind="json_schema",
            name="finding",
            schema={"type": "object"},
        ),
    )
    encoded = codec.encode_request(request)
    decoded = codec.decode_response(
        {
            "id": "chatcmpl-acceptance",
            "model": "gpt-compatible",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "need lookup",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": "{\"target\":\"demo\"}",
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        }
    )
    stream_event = codec.decode_stream_event(
        {"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]}
    )
    return {
        "schema_version": "agent-core-openai-compatible-codec-acceptance/v1",
        "codec": codec.manifest(),
        "encoded_has_tools": bool(encoded.get("tools")),
        "encoded_has_response_format": bool(encoded.get("response_format")),
        "encoded_has_multimodal_content": isinstance(encoded["messages"][1]["content"], list),
        "decoded_tool_call_names": [call.tool_name for call in decoded.tool_calls],
        "decoded_finish_reason": decoded.finish_reason,
        "decoded_usage_total_tokens": decoded.usage.total_tokens,
        "stream_event_type": stream_event.type,
        "stream_delta_bytes": len(stream_event.delta.encode("utf-8")),
    }


async def _provider_transport_matrix() -> dict[str, Any]:
    primary_transport = ProviderAcceptanceTransport(
        response={"error": {"message": "temporary", "retryable": True}}
    )
    fallback_transport = ProviderAcceptanceTransport(response={"content": "fallback-ok"})
    center = LLMProviderCenter(default_provider="primary", max_retries=0)
    center.register(
        "primary",
        TransportLLMProvider(primary_transport, name="primary"),
        priority=10,
    )
    center.register(
        "fallback",
        TransportLLMProvider(fallback_transport, name="fallback"),
        default_model="fallback-mini",
        priority=1,
    )
    response = await center.complete(LLMRequest(messages=[LLMMessage(role="user", content="task")]))
    manifest = center.manifest()
    return {
        "schema_version": "agent-core-provider-transport-acceptance/v1",
        "status": "completed" if response.content == "fallback-ok" else "failed",
        "fallback_used": bool(fallback_transport.complete_payloads),
        "primary_payload_count": len(primary_transport.complete_payloads),
        "fallback_payload_count": len(fallback_transport.complete_payloads),
        "call_statuses": [call["status"] for call in manifest.get("calls", ())],
        "retryable_failed_call_count": sum(
            1 for call in manifest.get("calls", ()) if call["status"] == "failed" and call["retryable"]
        ),
        "selected_models": [call["model"] for call in manifest.get("calls", ())],
    }


async def _provider_streaming_matrix() -> dict[str, Any]:
    batch = ProviderAcceptanceLLM("batch")
    stream = ProviderAcceptanceStreamingLLM()
    center = LLMProviderCenter(default_provider="batch")
    center.register(
        "batch",
        batch,
        default_model="batch-mini",
        priority=10,
        default_capabilities=LLMModelCapabilities(supports_streaming=False),
    )
    center.register(
        "stream",
        stream,
        default_model="stream-mini",
        priority=1,
        default_capabilities=LLMModelCapabilities(supports_streaming=True),
    )
    events = [
        event
        async for event in center.stream(LLMRequest(messages=[LLMMessage(role="user", content="stream")]))
    ]
    manifest = center.manifest()
    return {
        "schema_version": "agent-core-provider-streaming-acceptance/v1",
        "event_types": [event.type for event in events],
        "delta_bytes": sum(len(event.delta.encode("utf-8")) for event in events),
        "selected_provider": manifest["calls"][0]["provider_name"] if manifest.get("calls") else "",
        "streamed": bool(manifest.get("calls") and manifest["calls"][0]["streamed"]),
        "batch_request_count": len(batch.requests),
        "stream_request_count": len(stream.requests),
    }


def _provider_acceptance_issues(
    *,
    route_matrix: dict[str, Any],
    codec_matrix: dict[str, Any],
    transport_matrix: dict[str, Any],
    streaming_matrix: dict[str, Any],
) -> tuple[AgentCoreProviderAcceptanceIssue, ...]:
    issues: list[AgentCoreProviderAcceptanceIssue] = []
    expected_selected = {
        "text": "text",
        "vision": "vision",
        "structured": "structured",
        "tools": "tools",
    }
    selected = route_matrix.get("selected") if isinstance(route_matrix.get("selected"), dict) else {}
    for scenario, provider in expected_selected.items():
        if selected.get(scenario) != provider:
            issues.append(
                AgentCoreProviderAcceptanceIssue(
                    source="route_matrix",
                    code="provider_route_unexpected",
                    message=f"Provider route mismatch for {scenario}.",
                    metadata={"expected": provider, "actual": selected.get(scenario)},
                )
            )
    shape_adjusted = route_matrix.get("shape_adjusted") if isinstance(route_matrix.get("shape_adjusted"), dict) else {}
    if shape_adjusted.get("structured") is not True:
        issues.append(
            AgentCoreProviderAcceptanceIssue(
                source="route_matrix",
                code="request_shape_not_adjusted",
                message="Structured provider request shape was not adjusted to provider limits.",
            )
        )
    if codec_matrix.get("encoded_has_tools") is not True or codec_matrix.get("encoded_has_response_format") is not True:
        issues.append(
            AgentCoreProviderAcceptanceIssue(
                source="codec_matrix",
                code="openai_codec_contract_missing",
                message="OpenAI-compatible codec did not preserve tools and response format.",
            )
        )
    if "lookup" not in set(codec_matrix.get("decoded_tool_call_names") or ()):
        issues.append(
            AgentCoreProviderAcceptanceIssue(
                source="codec_matrix",
                code="openai_codec_tool_decode_missing",
                message="OpenAI-compatible codec did not decode tool calls.",
            )
        )
    if transport_matrix.get("fallback_used") is not True:
        issues.append(
            AgentCoreProviderAcceptanceIssue(
                source="transport_matrix",
                code="transport_fallback_missing",
                message="Transport provider fallback was not used.",
            )
        )
    if int(transport_matrix.get("retryable_failed_call_count") or 0) < 1:
        issues.append(
            AgentCoreProviderAcceptanceIssue(
                source="transport_matrix",
                code="transport_retryable_failure_missing",
                message="Transport provider did not record a retryable failure.",
            )
        )
    if streaming_matrix.get("selected_provider") != "stream" or streaming_matrix.get("streamed") is not True:
        issues.append(
            AgentCoreProviderAcceptanceIssue(
                source="streaming_matrix",
                code="streaming_route_missing",
                message="Streaming request did not route to the streaming-capable provider.",
            )
        )
    if streaming_matrix.get("event_types") != ["delta", "usage", "message_end"]:
        issues.append(
            AgentCoreProviderAcceptanceIssue(
                source="streaming_matrix",
                code="streaming_events_unexpected",
                message="Streaming provider did not emit the expected event contract.",
            )
        )
    return tuple(issues)
