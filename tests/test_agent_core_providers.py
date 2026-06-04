from __future__ import annotations

import pytest

from agent_core.actions import ActionRegistry
from agent_core.config import RuntimeBudget
from agent_core.harness import InMemoryAgentJournal
from agent_core.prompt import PromptIR
from agent_core.providers import (
    DefaultLLMProviderCodec,
    LLMCallRecord,
    LLMContentPart,
    LLMBudgetExceededError,
    LLMMessage,
    LLMModelCapabilities,
    LLMProviderCenter,
    LLMProviderError,
    LLMProviderNotFoundError,
    LLMRequest,
    LLMResponse,
    LLMResponseFormat,
    LLMRetryPolicy,
    LLMStreamAccumulator,
    LLMStreamEvent,
    LLMToolCall,
    LLMToolChoice,
    LLMToolContract,
    LLMUsageLimits,
    OpenAICompatibleLLMProviderCodec,
    RetryHint,
    TransportLLMProvider,
    UsageInfo,
)
from agent_core.react import ReActConfig, ReActExecutor
from agent_core.testing import MockLLMProvider, MockToolRuntime
from agent_core.tools import ToolSpec


class _FailingProvider:
    def __init__(self, *, retryable: bool, after_seconds: float | None = None) -> None:
        self.retryable = retryable
        self.after_seconds = after_seconds
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        raise LLMProviderError(
            "provider failed",
            retry_hint=RetryHint(
                retryable=self.retryable,
                after_seconds=self.after_seconds,
                reason="test",
            ),
        )


class _StreamingProvider:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("streaming test should not call complete")

    async def stream(self, request: LLMRequest):
        self.requests.append(request)
        yield LLMStreamEvent(type="delta", delta="hello")
        yield LLMStreamEvent(type="usage", usage=UsageInfo(total_tokens=3))
        yield LLMStreamEvent(type="message_end")


class _StreamingErrorProvider:
    def __init__(self, *, retryable: bool) -> None:
        self.retryable = retryable
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("streaming error test should not call complete")

    async def stream(self, request: LLMRequest):
        self.requests.append(request)
        yield LLMStreamEvent(type="delta", delta="partial")
        yield LLMStreamEvent(
            type="error",
            error="stream failed",
            metadata={"retryable": self.retryable, "reason": "test"},
        )


class _StreamingCostProvider:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("streaming cost test should not call complete")

    async def stream(self, request: LLMRequest):
        self.requests.append(request)
        yield LLMStreamEvent(type="delta", delta="expensive")
        yield LLMStreamEvent(type="usage", usage=UsageInfo(total_tokens=10, cost_usd=0.02))
        yield LLMStreamEvent(type="message_end")


class _DictTransport:
    def __init__(
        self,
        *,
        response: dict | None = None,
        events: tuple[dict, ...] = (),
        error: Exception | None = None,
    ) -> None:
        self.response = response or {}
        self.events = events
        self.error = error
        self.complete_payloads: list[dict] = []
        self.stream_payloads: list[dict] = []

    async def complete(self, payload: dict) -> dict:
        self.complete_payloads.append(payload)
        if self.error is not None:
            raise self.error
        return dict(self.response)

    async def stream(self, payload: dict):
        self.stream_payloads.append(payload)
        if self.error is not None:
            raise self.error
        for event in self.events:
            yield dict(event)

    def manifest(self) -> dict:
        return {"schema_version": "test-dict-transport/v1"}


@pytest.mark.asyncio
async def test_provider_center_routes_by_default_provider_and_model() -> None:
    fast = MockLLMProvider([{"action": "finish", "arguments": {"output": "fast"}}])
    strong = MockLLMProvider([{"action": "finish", "arguments": {"output": "strong"}}])
    center = LLMProviderCenter(default_provider="fast")
    center.register(
        "fast",
        fast,
        models=("fast-mini",),
        default_model="fast-mini",
        priority=1,
        tags=("cheap",),
    )
    center.register(
        "strong",
        strong,
        models=("strong-pro",),
        default_model="strong-pro",
        priority=10,
        tags=("reasoning",),
    )

    default_response = await center.complete(LLMRequest(messages=[]))
    strong_response = await center.complete(LLMRequest(messages=[], model="strong-pro"))

    assert default_response.action == {"action": "finish", "arguments": {"output": "fast"}}
    assert fast.requests[0].model == "fast-mini"
    assert fast.requests[0].metadata["provider"] == "fast"
    assert strong_response.action == {"action": "finish", "arguments": {"output": "strong"}}
    assert strong.requests[0].model == "strong-pro"
    assert strong.requests[0].metadata["provider"] == "strong"


def test_provider_center_builds_preflight_route_plan_without_calling_provider() -> None:
    small = MockLLMProvider(["small"])
    vision = MockLLMProvider(["vision"])
    center = LLMProviderCenter(default_provider="small", fallback_enabled=True)
    center.register(
        "small",
        small,
        default_model="small-mini",
        priority=10,
        default_capabilities=LLMModelCapabilities(modalities=("text",)),
    )
    center.register(
        "vision",
        vision,
        default_model="vision-pro",
        priority=1,
        default_capabilities=LLMModelCapabilities(
            supports_streaming=True,
            modalities=("text", "image"),
        ),
    )

    plan = center.route_plan(
        LLMRequest(
            messages=[
                LLMMessage(
                    role="user",
                    content="inspect",
                    content_parts=(LLMContentPart(kind="image", uri="file://finding.png"),),
                )
            ],
        ),
        streamed=True,
    )
    manifest = plan.manifest()

    assert plan.ready
    assert plan.selected_route is not None
    assert plan.selected_route.provider_name == "vision"
    assert [candidate.provider_name for candidate in plan.candidates] == ["small", "vision"]
    assert {candidate.provider_name: candidate.reason for candidate in plan.candidates} == {
        "small": "unsupported_capabilities",
        "vision": "selected",
    }
    assert not small.requests
    assert not vision.requests
    assert manifest["schema_version"] == "agent-core-llm-provider-route-plan/v1"
    assert manifest["candidates"][0]["metadata"]["model_capabilities"]["modalities"] == ["text"]


def test_provider_center_route_plan_explains_missing_explicit_provider() -> None:
    center = LLMProviderCenter(default_provider="mock")
    center.register("mock", MockLLMProvider([]), default_model="mock-mini")

    plan = center.route_plan(LLMRequest(messages=[], metadata={"provider": "missing"}))

    assert not plan.ready
    assert plan.selected_route is None
    assert plan.candidates[0].provider_name == "missing"
    assert plan.candidates[0].reason == "provider_not_registered"
    assert plan.candidates[1].provider_name == "mock"
    assert plan.candidates[1].reason == "provider_not_requested"


@pytest.mark.asyncio
async def test_provider_center_honors_explicit_provider_metadata() -> None:
    fast = MockLLMProvider(["fast"])
    strong = MockLLMProvider(["strong"])
    center = LLMProviderCenter(default_provider="fast")
    center.register("fast", fast, default_model="fast-mini")
    center.register("strong", strong, default_model="strong-pro")

    response = await center.complete(LLMRequest(messages=[], metadata={"provider": "strong"}))

    assert response.content == "strong"
    assert not fast.requests
    assert strong.requests[0].model == "strong-pro"


def test_provider_center_search_manifest_and_missing_provider() -> None:
    center = LLMProviderCenter(default_provider="fast", default_model="fallback")
    center.register("fast", MockLLMProvider([]), models=("fast-mini",), tags=("cheap",), priority=1)
    center.register("strong", MockLLMProvider([]), models=("strong-pro",), tags=("reasoning",), priority=10)

    assert [spec.name for spec in center.specs()] == ["strong", "fast"]
    assert [spec.name for spec in center.search(tag="cheap")] == ["fast"]
    assert [spec.name for spec in center.search(model="strong-pro")] == ["strong"]
    assert center.manifest()["providers"][0]["name"] == "strong"
    assert center.manifest()["retry_policy"]["schema_version"] == "agent-core-llm-retry-policy/v1"
    assert center.manifest()["usage_limits"]["schema_version"] == "agent-core-llm-usage-limits/v1"
    assert LLMRequest(messages=[LLMMessage(role="user", content="hello")]).manifest()[
        "messages"
    ][0]["content_bytes"] == 5

    with pytest.raises(LLMProviderNotFoundError):
        center.select(LLMRequest(messages=[], metadata={"provider": "missing"}))


@pytest.mark.asyncio
async def test_provider_center_shapes_request_to_declared_output_limit() -> None:
    provider = MockLLMProvider(["ok"])
    center = LLMProviderCenter(default_provider="limited")
    center.register(
        "limited",
        provider,
        default_model="limited-mini",
        default_capabilities=LLMModelCapabilities(
            context_window_tokens=4096,
            max_output_tokens=64,
        ),
    )

    plan = center.route_plan(LLMRequest(messages=[], max_output_tokens=256))
    response = await center.complete(LLMRequest(messages=[], max_output_tokens=256))
    shape_plan = center.calls[0].metadata["request_shape_plan"]

    assert response.content == "ok"
    assert provider.requests[0].max_output_tokens == 64
    assert provider.requests[0].metadata["request_shape_plan"]["adjusted"] is True
    assert provider.requests[0].metadata["request_shape_plan"]["final_max_output_tokens"] == 64
    assert shape_plan["schema_version"] == "agent-core-llm-request-shape-plan/v1"
    assert shape_plan["original_max_output_tokens"] == 256
    assert shape_plan["final_max_output_tokens"] == 64
    assert shape_plan["provider_max_output_tokens"] == 64
    assert shape_plan["decisions"] == ["max_output_tokens_capped_to_provider_limit"]
    assert center.calls[0].metadata["original_request"]["max_output_tokens"] == 256
    assert center.calls[0].metadata["request"]["max_output_tokens"] == 64
    assert plan.ready
    assert plan.selected_route is not None
    assert plan.selected_route.metadata["request_shape_plan"]["adjusted"] is True
    assert plan.candidates[0].metadata["request_shape_plan"]["final_max_output_tokens"] == 64


def test_llm_message_supports_provider_neutral_content_parts() -> None:
    message = LLMMessage(
        role="user",
        content="inspect this",
        content_parts=(
            LLMContentPart.text_part("caption"),
            LLMContentPart(
                kind="image",
                uri="file://evidence.png",
                mime_type="image/png",
                metadata={"source": "artifact"},
            ),
        ),
    )
    request = LLMRequest(messages=[message])
    manifest = request.manifest()
    encoded = DefaultLLMProviderCodec().encode_request(request)

    assert manifest["messages"][0]["content_bytes"] == len("inspect this")
    assert manifest["messages"][0]["content_part_count"] == 2
    assert manifest["messages"][0]["content_parts"][0]["text_bytes"] == len("caption")
    assert manifest["messages"][0]["content_parts"][1]["has_uri"] is True
    assert "file://evidence.png" not in str(manifest)
    assert encoded["messages"][0]["content"] == "inspect this"
    assert encoded["messages"][0]["content_parts"][1]["uri"] == "file://evidence.png"

    with pytest.raises(ValueError):
        LLMContentPart(kind="image")


def test_llm_request_supports_provider_neutral_tool_and_response_contracts() -> None:
    request = LLMRequest(
        messages=[LLMMessage(role="user", content="look up target")],
        tools=(
            LLMToolContract(
                name="lookup",
                description="Look up one record",
                parameters_schema={
                    "type": "object",
                    "required": ["id"],
                    "properties": {"id": {"type": "string"}},
                },
                strict=True,
            ),
        ),
        tool_choice=LLMToolChoice(mode="tool", tool_name="lookup"),
        response_format=LLMResponseFormat(
            kind="json_schema",
            name="lookup_result",
            schema={
                "type": "object",
                "required": ["summary"],
                "properties": {"summary": {"type": "string"}},
            },
            strict=True,
        ),
    )
    manifest = request.manifest()
    encoded = DefaultLLMProviderCodec().encode_request(request)

    assert manifest["tool_count"] == 1
    assert manifest["tools"][0]["name"] == "lookup"
    assert manifest["tool_choice"]["mode"] == "tool"
    assert manifest["response_format"]["kind"] == "json_schema"
    assert encoded["tools"][0]["parameters_schema"]["required"] == ["id"]
    assert encoded["tool_choice"]["tool_name"] == "lookup"
    assert encoded["response_format"]["name"] == "lookup_result"

    with pytest.raises(ValueError):
        LLMToolChoice(mode="tool")
    with pytest.raises(ValueError):
        LLMRequest(messages=[], tool_choice=LLMToolChoice(mode="required"))


def test_openai_compatible_codec_encodes_chat_completion_payloads() -> None:
    codec = OpenAICompatibleLLMProviderCodec()
    payload = codec.encode_request(
        LLMRequest(
            messages=[
                LLMMessage(role="system", content="rules"),
                LLMMessage(
                    role="user",
                    content="inspect",
                    content_parts=(
                        LLMContentPart(
                            kind="image",
                            uri="file://finding.png",
                            mime_type="image/png",
                            metadata={"detail": "high"},
                        ),
                    ),
                ),
            ],
            model="gpt-compatible",
            temperature=0.2,
            max_output_tokens=128,
            tools=(
                LLMToolContract(
                    name="lookup",
                    description="Lookup target",
                    parameters_schema={"type": "object"},
                    strict=True,
                ),
            ),
            tool_choice=LLMToolChoice(mode="tool", tool_name="lookup"),
            response_format=LLMResponseFormat(
                kind="json_schema",
                name="finding",
                schema={"type": "object"},
                strict=True,
            ),
            metadata={"request_id": "r1"},
        )
    )

    assert payload["model"] == "gpt-compatible"
    assert payload["max_tokens"] == 128
    assert payload["messages"][0] == {"role": "system", "content": "rules"}
    assert payload["messages"][1]["content"][0] == {"type": "text", "text": "inspect"}
    assert payload["messages"][1]["content"][1]["image_url"]["url"] == "file://finding.png"
    assert payload["messages"][1]["content"][1]["image_url"]["detail"] == "high"
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["function"]["name"] == "lookup"
    assert payload["tools"][0]["function"]["strict"] is True
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["name"] == "finding"
    assert codec.manifest()["schema_version"] == "agent-core-openai-compatible-llm-provider-codec/v1"


def test_openai_compatible_codec_decodes_chat_completion_payloads() -> None:
    codec = OpenAICompatibleLLMProviderCodec()
    response = codec.decode_response(
        {
            "id": "chatcmpl-1",
            "model": "gpt-compatible",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
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
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    delta = codec.decode_stream_event(
        {"choices": [{"delta": {"content": "hel"}, "finish_reason": None}]}
    )
    tool_event = codec.decode_stream_event(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {
                                    "name": "scan",
                                    "arguments": "{\"target\":\"demo\"}",
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    )
    usage = codec.decode_stream_event(
        {"usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
    )
    end = codec.decode_stream_event(
        {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    )

    assert response.content == "need lookup"
    assert response.tool_calls[0].tool_name == "lookup"
    assert response.tool_calls[0].arguments == {"target": "demo"}
    assert response.tool_calls[0].call_id == "call-1"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5
    assert response.finish_reason == "tool_calls"
    assert response.metadata["id"] == "chatcmpl-1"
    assert delta.type == "delta"
    assert delta.delta == "hel"
    assert tool_event.type == "tool_call"
    assert tool_event.tool_call is not None
    assert tool_event.tool_call.tool_name == "scan"
    assert usage.type == "usage"
    assert usage.usage is not None
    assert usage.usage.total_tokens == 5
    assert end.type == "message_end"
    assert end.metadata["finish_reason"] == "stop"


def test_llm_response_supports_provider_native_tool_calls() -> None:
    tool_call = LLMToolCall(
        tool_name="lookup",
        arguments={"target": "demo"},
        call_id="call-1",
        metadata={"provider": "mock"},
    )
    response = LLMResponse(content="need lookup", tool_calls=(tool_call,))
    manifest = response.manifest()
    decoded = DefaultLLMProviderCodec().decode_response(
        {
            "content": "need lookup",
            "tool_calls": [
                {
                    "tool_name": "lookup",
                    "arguments": {"target": "demo"},
                    "call_id": "call-1",
                    "metadata": {"provider": "mock"},
                }
            ],
        }
    )

    assert manifest["tool_call_count"] == 1
    assert manifest["tool_calls"][0]["tool_name"] == "lookup"
    assert manifest["tool_calls"][0]["argument_keys"] == ["target"]
    assert "demo" not in str(manifest)
    assert decoded.tool_calls[0].tool_name == "lookup"
    assert decoded.tool_calls[0].arguments == {"target": "demo"}
    assert decoded.tool_calls[0].call_id == "call-1"


@pytest.mark.asyncio
async def test_provider_center_records_response_tool_call_manifest() -> None:
    provider = MockLLMProvider(
        [
            LLMResponse(
                tool_calls=(
                    LLMToolCall(
                        tool_name="lookup",
                        arguments={"target": "demo"},
                        call_id="call-1",
                    ),
                )
            )
        ]
    )
    center = LLMProviderCenter(default_provider="mock")
    center.register("mock", provider)

    await center.complete(LLMRequest(messages=[]))

    response_manifest = center.calls[0].metadata["response"]
    assert response_manifest["tool_call_count"] == 1
    assert response_manifest["tool_calls"][0]["tool_name"] == "lookup"
    assert response_manifest["tool_calls"][0]["argument_keys"] == ["target"]
    assert "demo" not in str(response_manifest)


def test_llm_tool_contract_can_be_derived_from_tool_spec_like_objects() -> None:
    contract = LLMToolContract.from_tool_spec(
        ToolSpec(
            name="scan",
            description="Run scan",
            parameters_schema={
                "type": "object",
                "properties": {"target": {"type": "string"}},
            },
            metadata={"risk": "read_only"},
        ),
        strict=True,
        metadata={"surface": "native_tool_call"},
    )

    assert contract.name == "scan"
    assert contract.description == "Run scan"
    assert contract.parameters_schema["properties"]["target"]["type"] == "string"
    assert contract.strict is True
    assert contract.metadata["source"] == "tool_spec"
    assert contract.metadata["risk"] == "read_only"
    assert contract.metadata["surface"] == "native_tool_call"


@pytest.mark.asyncio
async def test_provider_center_routes_by_declared_model_capabilities() -> None:
    small = MockLLMProvider(["small"])
    strong = MockLLMProvider(["strong"])
    center = LLMProviderCenter(default_provider="small")
    center.register(
        "small",
        small,
        default_model="small-mini",
        priority=10,
        default_capabilities=LLMModelCapabilities(
            context_window_tokens=512,
            max_output_tokens=64,
            supports_structured_output=False,
        ),
    )
    center.register(
        "strong",
        strong,
        default_model="strong-pro",
        priority=1,
        default_capabilities=LLMModelCapabilities(
            context_window_tokens=8192,
            max_output_tokens=1024,
            supports_structured_output=True,
            supports_json_mode=True,
        ),
    )

    response = await center.complete(
        LLMRequest(
            messages=[],
            max_output_tokens=256,
            metadata={
                "requires_structured_output": True,
                "required_capabilities": ("json_mode",),
                "estimated_total_tokens": 1000,
            },
        )
    )
    route = center.select(
        LLMRequest(
            messages=[],
            metadata={"requires_structured_output": True},
        )
    )
    manifest = center.manifest()

    assert response.content == "strong"
    assert not small.requests
    assert strong.requests[0].model == "strong-pro"
    assert strong.requests[0].metadata["model_capabilities"]["supports_structured_output"] is True
    assert center.calls[0].metadata["request"]["metadata"]["model_capabilities"][
        "supports_structured_output"
    ] is True
    assert center.calls[0].metadata["route_plan"]["selected_route"]["provider_name"] == "strong"
    assert {
        candidate["provider_name"]: candidate["reason"]
        for candidate in center.calls[0].metadata["route_plan"]["candidates"]
    } == {
        "small": "unsupported_capabilities",
        "strong": "selected",
    }
    assert center.calls[0].metadata["original_request"]["metadata"]["requires_structured_output"] is True
    assert route.provider_name == "strong"
    assert route.metadata["model_capabilities"]["supports_json_mode"] is True
    assert manifest["providers"][0]["default_capabilities"]["context_window_tokens"] == 512


@pytest.mark.asyncio
async def test_provider_center_routes_by_native_tool_and_response_format_contracts() -> None:
    small = MockLLMProvider(["small"])
    capable = MockLLMProvider(["capable"])
    center = LLMProviderCenter(default_provider="small")
    center.register(
        "small",
        small,
        default_model="small-mini",
        priority=10,
        default_capabilities=LLMModelCapabilities(
            supports_tool_calls=False,
            supports_structured_output=False,
            supports_json_mode=False,
        ),
    )
    center.register(
        "capable",
        capable,
        default_model="capable-pro",
        priority=1,
        default_capabilities=LLMModelCapabilities(
            supports_tool_calls=True,
            supports_structured_output=True,
            supports_json_mode=True,
        ),
    )

    response = await center.complete(
        LLMRequest(
            messages=[],
            tools=(LLMToolContract(name="lookup"),),
            response_format=LLMResponseFormat(
                kind="json_schema",
                name="answer",
                schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            ),
        )
    )

    assert response.content == "capable"
    assert not small.requests
    assert capable.requests[0].model == "capable-pro"
    assert capable.requests[0].tools[0].name == "lookup"
    assert capable.requests[0].response_format is not None
    assert center.calls[0].metadata["original_request"]["tool_count"] == 1
    assert (
        center.calls[0].metadata["original_request"]["response_format"]["kind"]
        == "json_schema"
    )


@pytest.mark.asyncio
async def test_provider_center_routes_by_message_modalities() -> None:
    text_only = MockLLMProvider(["text"])
    vision = MockLLMProvider(["vision"])
    center = LLMProviderCenter(default_provider="text")
    center.register(
        "text",
        text_only,
        default_model="text-mini",
        priority=10,
        default_capabilities=LLMModelCapabilities(modalities=("text",)),
    )
    center.register(
        "vision",
        vision,
        default_model="vision-pro",
        priority=1,
        default_capabilities=LLMModelCapabilities(modalities=("text", "image")),
    )

    response = await center.complete(
        LLMRequest(
            messages=[
                LLMMessage(
                    role="user",
                    content="analyze image",
                    content_parts=(
                        LLMContentPart(
                            kind="image",
                            uri="file://finding.png",
                            mime_type="image/png",
                        ),
                    ),
                )
            ],
        )
    )

    assert response.content == "vision"
    assert not text_only.requests
    assert vision.requests[0].model == "vision-pro"
    assert vision.requests[0].metadata["model_capabilities"]["modalities"] == [
        "text",
        "image",
    ]


def test_model_capabilities_understand_response_format_kinds() -> None:
    json_only = LLMModelCapabilities(supports_json_mode=True)
    structured = LLMModelCapabilities(supports_structured_output=True)

    assert json_only.supports_request(
        LLMRequest(messages=[], response_format=LLMResponseFormat(kind="json"))
    )
    assert not json_only.supports_request(
        LLMRequest(
            messages=[],
            response_format=LLMResponseFormat(
                kind="json_schema",
                schema={"type": "object"},
            ),
        )
    )
    assert structured.supports_request(
        LLMRequest(
            messages=[],
            response_format=LLMResponseFormat(
                kind="json_schema",
                schema={"type": "object"},
            ),
        )
    )


@pytest.mark.asyncio
async def test_provider_center_rejects_explicit_provider_when_capabilities_do_not_match() -> None:
    center = LLMProviderCenter(default_provider="small")
    center.register(
        "small",
        MockLLMProvider(["small"]),
        default_model="small-mini",
        default_capabilities=LLMModelCapabilities(
            context_window_tokens=128,
            max_output_tokens=32,
            supports_tool_calls=False,
        ),
    )

    with pytest.raises(LLMProviderNotFoundError):
        await center.complete(
            LLMRequest(
                messages=[],
                max_output_tokens=64,
                metadata={
                    "provider": "small",
                    "requires_tool_calls": True,
                    "estimated_total_tokens": 256,
                },
            )
        )


def test_llm_stream_accumulator_builds_response_and_prompt_safe_manifest() -> None:
    accumulator = LLMStreamAccumulator()

    accumulator.add(LLMStreamEvent(type="message_start"))
    accumulator.add(LLMStreamEvent(type="delta", delta="hel"))
    accumulator.add(
        LLMStreamEvent(
            type="action",
            action={"action": "finish", "arguments": {"output": "hello"}},
        )
    )
    accumulator.add(
        LLMStreamEvent(
            type="tool_call",
            tool_call=LLMToolCall(
                tool_name="lookup",
                arguments={"target": "demo"},
                call_id="call-1",
            ),
        )
    )
    accumulator.add(LLMStreamEvent(type="delta", delta="lo"))
    accumulator.add(LLMStreamEvent(type="usage", usage=UsageInfo(total_tokens=5, cost_usd=0.01)))
    accumulator.add(LLMStreamEvent(type="message_end"))

    response = accumulator.response(metadata={"provider": "mock"})
    manifest = accumulator.manifest()

    assert response.content == "hello"
    assert response.action == {"action": "finish", "arguments": {"output": "hello"}}
    assert response.tool_calls[0].tool_name == "lookup"
    assert response.usage.total_tokens == 5
    assert response.finish_reason == "stop"
    assert response.metadata["streamed"] is True
    assert response.metadata["provider"] == "mock"
    assert response.metadata["stream"]["event_count"] == 7
    assert response.metadata["stream"]["tool_call_count"] == 1
    assert manifest["schema_version"] == "agent-core-llm-stream-accumulator/v1"
    assert manifest["summary"]["content_bytes"] == 5
    assert "hello" not in str(manifest)
    assert "demo" not in str(manifest)


@pytest.mark.asyncio
async def test_transport_llm_provider_encodes_request_and_decodes_response() -> None:
    transport = _DictTransport(
        response={
            "content": "ok",
            "action": {"action": "finish", "arguments": {"output": "ok"}},
            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5, "cost_usd": 0.01},
            "finish_reason": "stop",
            "metadata": {"raw_id": "resp-1"},
        }
    )
    provider = TransportLLMProvider(transport, name="dict")

    response = await provider.complete(
        LLMRequest(
            messages=[LLMMessage(role="user", content="task", name="u")],
            model="dict-mini",
            temperature=0.1,
            metadata={"request_id": "r1"},
        )
    )
    manifest = provider.manifest()

    assert transport.complete_payloads[0]["schema_version"] == "agent-core-llm-transport-request/v1"
    assert transport.complete_payloads[0]["messages"][0]["content"] == "task"
    assert transport.complete_payloads[0]["metadata"]["request_id"] == "r1"
    assert response.content == "ok"
    assert response.action == {"action": "finish", "arguments": {"output": "ok"}}
    assert response.usage.total_tokens == 5
    assert response.metadata["transport_provider"] == "dict"
    assert manifest["codec"]["schema_version"] == "agent-core-default-llm-provider-codec/v1"
    assert manifest["transport"]["schema_version"] == "test-dict-transport/v1"


@pytest.mark.asyncio
async def test_transport_llm_provider_streams_and_maps_payload_errors() -> None:
    transport = _DictTransport(
        events=(
            {"type": "delta", "delta": "hello"},
            {"type": "usage", "usage": {"total_tokens": 4}},
            {"type": "message_end"},
        )
    )
    provider = TransportLLMProvider(transport, name="dict")

    events = [event async for event in provider.stream(LLMRequest(messages=[]))]

    assert [event.type for event in events] == ["delta", "usage", "message_end"]
    assert events[0].metadata["transport_provider"] == "dict"
    assert events[1].usage is not None
    assert events[1].usage.total_tokens == 4

    failing = TransportLLMProvider(
        _DictTransport(response={"error": {"message": "rate limited", "retryable": True, "after_seconds": 1}})
    )
    with pytest.raises(LLMProviderError) as exc_info:
        await failing.complete(LLMRequest(messages=[]))

    assert exc_info.value.retry_hint.retryable is True
    assert exc_info.value.retry_hint.after_seconds == 1.0


@pytest.mark.asyncio
async def test_transport_llm_provider_works_with_provider_center_retries() -> None:
    primary = TransportLLMProvider(
        _DictTransport(response={"error": {"message": "temporary", "retryable": True}}),
        name="primary",
    )
    fallback_transport = _DictTransport(response={"content": "fallback"})
    fallback = TransportLLMProvider(fallback_transport, name="fallback")
    center = LLMProviderCenter(default_provider="primary", max_retries=1)
    center.register("primary", primary, priority=10)
    center.register("fallback", fallback, default_model="fallback-mini", priority=1)

    response = await center.complete(LLMRequest(messages=[]))

    assert response.content == "fallback"
    assert [call.status for call in center.calls] == ["failed", "failed", "completed"]
    assert center.calls[0].retryable is True
    assert fallback_transport.complete_payloads[0]["model"] == "fallback-mini"
    assert DefaultLLMProviderCodec().manifest()["schema_version"] == "agent-core-default-llm-provider-codec/v1"


@pytest.mark.asyncio
async def test_provider_center_retries_retryable_provider_errors() -> None:
    failing = _FailingProvider(retryable=True)
    fallback = MockLLMProvider([{"action": "finish", "arguments": {"output": "fallback"}}])
    center = LLMProviderCenter(default_provider="primary", max_retries=1)
    center.register("primary", failing, priority=10)
    center.register("fallback", fallback, priority=1)

    response = await center.complete(LLMRequest(messages=[]))

    assert response.action == {"action": "finish", "arguments": {"output": "fallback"}}
    assert len(failing.requests) == 2
    assert len(center.failures) == 2
    assert [call.status for call in center.calls] == ["failed", "failed", "completed"]
    assert center.manifest()["call_count"] == 3
    assert center.manifest()["calls"][0]["retryable"] is True
    assert fallback.requests[0].metadata["provider"] == "fallback"


@pytest.mark.asyncio
async def test_provider_center_honors_retry_after_policy() -> None:
    failing = _FailingProvider(retryable=True, after_seconds=30.0)
    fallback = MockLLMProvider(["ok"])
    center = LLMProviderCenter(
        default_provider="primary",
        retry_policy=LLMRetryPolicy(max_retries=2, max_retry_after_seconds=5.0),
    )
    center.register("primary", failing, priority=10)
    center.register("fallback", fallback, priority=1)

    response = await center.complete(LLMRequest(messages=[]))

    assert response.content == "ok"
    assert len(failing.requests) == 1
    assert center.calls[0].retryable is False
    assert center.manifest()["retry_policy"]["max_retry_after_seconds"] == 5.0


@pytest.mark.asyncio
async def test_provider_center_falls_back_after_non_retryable_failure() -> None:
    failing = _FailingProvider(retryable=False)
    fallback = MockLLMProvider(["ok"])
    center = LLMProviderCenter(default_provider="primary", max_retries=2)
    center.register("primary", failing, priority=10)
    center.register("fallback", fallback, priority=1)

    response = await center.complete(LLMRequest(messages=[]))

    assert response.content == "ok"
    assert len(failing.requests) == 1
    assert fallback.requests


@pytest.mark.asyncio
async def test_provider_center_does_not_fallback_for_explicit_provider() -> None:
    failing = _FailingProvider(retryable=False)
    fallback = MockLLMProvider(["ok"])
    center = LLMProviderCenter(default_provider="fallback")
    center.register("primary", failing, priority=10)
    center.register("fallback", fallback, priority=1)

    with pytest.raises(LLMProviderError):
        await center.complete(LLMRequest(messages=[], metadata={"provider": "primary"}))

    assert not fallback.requests


@pytest.mark.asyncio
async def test_provider_center_enforces_estimated_and_actual_cost_budget() -> None:
    provider = MockLLMProvider(
        [
            LLMResponse(content="too-expensive", usage=UsageInfo(cost_usd=0.02)),
            LLMResponse(content="ok", usage=UsageInfo(cost_usd=0.01)),
        ]
    )
    center = LLMProviderCenter(default_provider="local", max_cost_usd=0.01)
    center.register("local", provider)

    with pytest.raises(LLMBudgetExceededError):
        await center.complete(LLMRequest(messages=[], metadata={"estimated_cost_usd": 0.02}))

    with pytest.raises(LLMBudgetExceededError):
        await center.complete(LLMRequest(messages=[]))

    assert center.usage.cost_usd == 0.0
    response = await center.complete(LLMRequest(messages=[], metadata={"max_cost_usd": 0.02}))
    assert response.content == "ok"
    assert center.usage.cost_usd == 0.01


@pytest.mark.asyncio
async def test_provider_center_enforces_call_attempt_and_token_limits() -> None:
    provider = MockLLMProvider(
        [
            LLMResponse(content="first", usage=UsageInfo(input_tokens=2, output_tokens=3, total_tokens=5)),
            LLMResponse(content="second", usage=UsageInfo(input_tokens=2, output_tokens=3, total_tokens=5)),
        ]
    )
    center = LLMProviderCenter(
        default_provider="local",
        usage_limits=LLMUsageLimits(max_call_attempts=1, max_total_tokens=8),
    )
    center.register("local", provider)

    first = await center.complete(LLMRequest(messages=[]))
    with pytest.raises(LLMBudgetExceededError):
        await center.complete(LLMRequest(messages=[]))

    assert first.content == "first"
    assert center.usage.total_tokens == 5
    assert center.manifest()["usage_limits"]["max_call_attempts"] == 1

    token_limited = LLMProviderCenter(
        default_provider="local",
        usage_limits=LLMUsageLimits(max_input_tokens=4, max_output_tokens=4, max_total_tokens=6),
    )
    token_limited.register(
        "local",
        MockLLMProvider([LLMResponse(content="too many", usage=UsageInfo(total_tokens=7))]),
    )
    with pytest.raises(LLMBudgetExceededError):
        await token_limited.complete(LLMRequest(messages=[], metadata={"estimated_input_tokens": 5}))
    with pytest.raises(LLMBudgetExceededError):
        await token_limited.complete(LLMRequest(messages=[]))

    assert token_limited.usage.total_tokens == 0


@pytest.mark.asyncio
async def test_provider_center_records_streaming_call_manifests() -> None:
    provider = _StreamingProvider()
    center = LLMProviderCenter(default_provider="local")
    center.register("local", provider, default_model="local-mini")

    events = [
        event
        async for event in center.stream(
            LLMRequest(messages=[LLMMessage(role="user", content="task")])
        )
    ]
    manifest = center.manifest()

    assert [event.type for event in events] == ["delta", "usage", "message_end"]
    assert center.calls[0].streamed is True
    assert center.calls[0].status == "completed"
    assert center.calls[0].metadata["stream_summary"]["schema_version"] == "agent-core-llm-stream-summary/v1"
    assert center.calls[0].metadata["stream_summary"]["event_types"] == ["delta", "usage", "message_end"]
    assert center.calls[0].metadata["stream_summary"]["delta_bytes"] == 5
    assert center.calls[0].metadata["request"]["metadata"]["provider"] == "local"
    assert center.calls[0].metadata["original_request"]["message_count"] == 1
    assert manifest["calls"][0]["streamed"] is True
    assert manifest["calls"][0]["metadata"]["stream_summary"]["content_bytes"] == 5
    assert manifest["calls"][0]["usage"]["total_tokens"] == 3
    assert LLMCallRecord(
        provider_name="local",
        model="local-mini",
        attempt=1,
        status="completed",
    ).manifest()["provider_name"] == "local"


@pytest.mark.asyncio
async def test_provider_center_routes_streaming_to_stream_capable_provider() -> None:
    non_stream = _StreamingProvider()
    stream_capable = _StreamingProvider()
    center = LLMProviderCenter(default_provider="batch")
    center.register(
        "batch",
        non_stream,
        default_model="batch-mini",
        priority=10,
        default_capabilities=LLMModelCapabilities(supports_streaming=False),
    )
    center.register(
        "stream",
        stream_capable,
        default_model="stream-mini",
        priority=1,
        default_capabilities=LLMModelCapabilities(supports_streaming=True),
    )

    events = [event async for event in center.stream(LLMRequest(messages=[]))]

    assert [event.type for event in events] == ["delta", "usage", "message_end"]
    assert not non_stream.requests
    assert stream_capable.requests[0].model == "stream-mini"
    assert stream_capable.requests[0].metadata["provider"] == "stream"
    assert center.calls[0].provider_name == "stream"


@pytest.mark.asyncio
async def test_provider_center_retries_stream_error_events_before_yielding_output() -> None:
    failing = _StreamingErrorProvider(retryable=True)
    fallback = _StreamingProvider()
    center = LLMProviderCenter(default_provider="primary", max_retries=1)
    center.register("primary", failing, priority=10)
    center.register("fallback", fallback, default_model="fallback-mini", priority=1)

    events = [event async for event in center.stream(LLMRequest(messages=[]))]

    assert [event.delta for event in events if event.delta] == ["hello"]
    assert len(failing.requests) == 2
    assert len(fallback.requests) == 1
    assert [call.status for call in center.calls] == ["failed", "failed", "completed"]
    assert center.calls[0].streamed is True
    assert center.calls[0].retryable is True
    assert center.failures[0]["streamed"] is True
    assert fallback.requests[0].metadata["provider"] == "fallback"


@pytest.mark.asyncio
async def test_provider_center_blocks_over_budget_stream_before_emitting_events() -> None:
    provider = _StreamingCostProvider()
    center = LLMProviderCenter(default_provider="local", max_cost_usd=0.01)
    center.register("local", provider)

    with pytest.raises(LLMBudgetExceededError):
        _ = [event async for event in center.stream(LLMRequest(messages=[]))]

    assert center.usage.cost_usd == 0.0
    assert center.calls == []


@pytest.mark.asyncio
async def test_react_executor_can_use_provider_center_as_llm_port() -> None:
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}])
    center = LLMProviderCenter(default_provider="local")
    center.register("local", provider, default_model="local-mini")

    executor = ReActExecutor(
        provider=center,
        tool_runtime=MockToolRuntime(),
        action_registry=ActionRegistry(),
        harness=InMemoryAgentJournal(),
        config=ReActConfig(max_iterations=2, budget=RuntimeBudget(max_cost_usd=0.25)),
    )

    result = await executor.run("task", PromptIR.from_parts(high_static="rules"))

    assert result.status == "completed"
    assert result.output == "done"
    assert provider.requests[0].model == "local-mini"
    assert provider.requests[0].metadata["provider"] == "local"
    assert provider.requests[0].metadata["max_cost_usd"] == 0.25


@pytest.mark.asyncio
async def test_react_executor_can_execute_provider_native_tool_calls() -> None:
    provider = MockLLMProvider(
        [
            LLMResponse(
                content="checking",
                tool_calls=(
                    LLMToolCall(
                        tool_name="lookup",
                        arguments={"target": "demo"},
                        call_id="call-1",
                    ),
                ),
            ),
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    tools = MockToolRuntime({"lookup": "lookup result"})
    harness = InMemoryAgentJournal()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=tools,
        action_registry=ActionRegistry(),
        harness=harness,
        config=ReActConfig(max_iterations=3, native_tool_calls=True),
    )

    result = await executor.run("task", PromptIR.from_parts(high_static="rules"))

    assert result.status == "completed"
    assert result.output == "done"
    assert provider.requests[0].tools[0].name == "lookup"
    assert provider.requests[0].tool_choice is not None
    assert provider.requests[0].tool_choice.mode == "auto"
    assert provider.requests[1].messages[-1].role == "tool"
    assert provider.requests[1].messages[-1].content == "lookup result"
    assert provider.requests[1].messages[-1].metadata["provider_tool_call_id"] == "call-1"
    assert tools.invocations[0].call_id == "call-1"
    assert tools.invocations[0].arguments == {"target": "demo"}

