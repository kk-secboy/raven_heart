from __future__ import annotations

import pytest

from agent_core.actions import ActionRegistry
from agent_core.config import RuntimeBudget
from agent_core.harness import InMemoryAgentJournal
from agent_core.prompt import PromptIR
from agent_core.providers import (
    DefaultLLMProviderCodec,
    LLMCallRecord,
    LLMBudgetExceededError,
    LLMMessage,
    LLMProviderCenter,
    LLMProviderError,
    LLMProviderNotFoundError,
    LLMRequest,
    LLMResponse,
    LLMRetryPolicy,
    LLMStreamEvent,
    LLMUsageLimits,
    RetryHint,
    TransportLLMProvider,
    UsageInfo,
)
from agent_core.react import ReActConfig, ReActExecutor
from agent_core.testing import MockLLMProvider, MockToolRuntime


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
    assert manifest["calls"][0]["streamed"] is True
    assert manifest["calls"][0]["usage"]["total_tokens"] == 3
    assert LLMCallRecord(
        provider_name="local",
        model="local-mini",
        attempt=1,
        status="completed",
    ).manifest()["provider_name"] == "local"


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

