"""SDK-level provider resilience acceptance checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.providers import (
    LLMProviderCenter,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMRetryPolicy,
    LLMStreamEvent,
    RetryHint,
    UsageInfo,
)


@dataclass(frozen=True)
class AgentCoreProviderResilienceAcceptanceIssue:
    """One blocking provider resilience acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-provider-resilience-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreProviderResilienceAcceptanceReport:
    """Prompt-safe report for provider retry and fallback behavior."""

    status: str
    retry_completion: dict[str, Any] = field(default_factory=dict)
    retry_after: dict[str, Any] = field(default_factory=dict)
    non_retry_fallback: dict[str, Any] = field(default_factory=dict)
    explicit_provider: dict[str, Any] = field(default_factory=dict)
    stream_fallback: dict[str, Any] = field(default_factory=dict)
    trace_evals: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreProviderResilienceAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-provider-resilience-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "retry_completion": dict(self.retry_completion),
            "retry_after": dict(self.retry_after),
            "non_retry_fallback": dict(self.non_retry_fallback),
            "explicit_provider": dict(self.explicit_provider),
            "stream_fallback": dict(self.stream_fallback),
            "trace_evals": dict(self.trace_evals),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreProviderResilienceAcceptanceHarness:
    """Run deterministic provider retry and fallback checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreProviderResilienceAcceptanceReport:
        retry_completion = await _retry_completion_acceptance()
        retry_after = await _retry_after_acceptance()
        non_retry_fallback = await _non_retry_fallback_acceptance()
        explicit_provider = await _explicit_provider_acceptance()
        stream_fallback = await _stream_fallback_acceptance()
        trace_evals = {
            "retry_completion": _provider_trace_eval(retry_completion).manifest(),
            "retry_after": _provider_trace_eval(retry_after).manifest(),
            "non_retry_fallback": _provider_trace_eval(non_retry_fallback).manifest(),
            "explicit_provider": _provider_trace_eval(explicit_provider).manifest(),
            "stream_fallback": DefaultTraceEvaluator()
            .evaluate(
                _trace_manifest(stream_fallback, status="completed"),
                TraceEvalSpec(
                    name="provider-resilience-stream-fallback",
                    require_journal_ok=False,
                    require_provider_streaming=True,
                    required_provider_stream_event_types=("delta", "usage", "message_end"),
                    max_provider_stream_errors=2,
                    require_provider_error_classification=True,
                    required_provider_error_kinds=("provider_failed",),
                ),
            )
            .manifest(),
        }
        issues = _provider_resilience_issues(
            retry_completion=retry_completion,
            retry_after=retry_after,
            non_retry_fallback=non_retry_fallback,
            explicit_provider=explicit_provider,
            stream_fallback=stream_fallback,
            trace_evals=trace_evals,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreProviderResilienceAcceptanceReport(
            status=status,
            retry_completion=_prompt_safe(retry_completion),
            retry_after=_prompt_safe(retry_after),
            non_retry_fallback=_prompt_safe(non_retry_fallback),
            explicit_provider=_prompt_safe(explicit_provider),
            stream_fallback=_prompt_safe(stream_fallback),
            trace_evals=trace_evals,
            issues=issues,
            metadata={
                "scenario": "agent_core_provider_resilience_acceptance",
                **dict(self.metadata),
            },
        )


async def run_agent_core_provider_resilience_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreProviderResilienceAcceptanceReport:
    """Run the default provider retry and fallback acceptance checks."""

    return await AgentCoreProviderResilienceAcceptanceHarness(
        metadata=dict(metadata or {})
    ).run()


class _FailingProvider:
    def __init__(
        self,
        *,
        retryable: bool,
        after_seconds: float | None = None,
        label: str = "temporary provider outage",
    ) -> None:
        self.retryable = retryable
        self.after_seconds = after_seconds
        self.label = label
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        raise LLMProviderError(
            self.label,
            retry_hint=RetryHint(
                retryable=self.retryable,
                after_seconds=self.after_seconds,
                reason=self.label,
            ),
        )


class _FallbackProvider:
    def __init__(self, content: str = "fallback-ok") -> None:
        self.content = content
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(content=self.content, usage=UsageInfo(total_tokens=2))


class _StreamingErrorProvider:
    def __init__(self, *, retryable: bool = True) -> None:
        self.retryable = retryable
        self.requests: list[LLMRequest] = []

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self.requests.append(request)
        yield LLMStreamEvent(
            type="error",
            error="stream failed",
            metadata={"retryable": self.retryable, "reason": "temporary stream outage"},
        )


class _StreamingFallbackProvider:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self.requests.append(request)
        yield LLMStreamEvent(type="delta", delta="hello")
        yield LLMStreamEvent(type="usage", usage=UsageInfo(total_tokens=3))
        yield LLMStreamEvent(type="message_end")


async def _retry_completion_acceptance() -> dict[str, Any]:
    primary = _FailingProvider(retryable=True)
    fallback = _FallbackProvider("completion-fallback")
    center = LLMProviderCenter(default_provider="primary", max_retries=1)
    center.register("primary", primary, default_model="primary-mini", priority=10)
    center.register("fallback", fallback, default_model="fallback-mini", priority=1)
    response = await center.complete(LLMRequest(messages=[]))
    return _scenario_manifest(
        "agent-core-provider-resilience-retry-completion/v1",
        center=center,
        status="completed" if response.content == "completion-fallback" else "failed",
        primary_request_count=len(primary.requests),
        fallback_request_count=len(fallback.requests),
        output=response.content,
    )


async def _retry_after_acceptance() -> dict[str, Any]:
    primary = _FailingProvider(retryable=True, after_seconds=30.0)
    fallback = _FallbackProvider("retry-after-fallback")
    center = LLMProviderCenter(
        default_provider="primary",
        retry_policy=LLMRetryPolicy(max_retries=2, max_retry_after_seconds=5.0),
    )
    center.register("primary", primary, default_model="primary-mini", priority=10)
    center.register("fallback", fallback, default_model="fallback-mini", priority=1)
    response = await center.complete(LLMRequest(messages=[]))
    manifest = _scenario_manifest(
        "agent-core-provider-resilience-retry-after/v1",
        center=center,
        status="completed" if response.content == "retry-after-fallback" else "failed",
        primary_request_count=len(primary.requests),
        fallback_request_count=len(fallback.requests),
        output=response.content,
    )
    manifest["max_retry_after_seconds"] = center.retry_policy.max_retry_after_seconds
    return manifest


async def _non_retry_fallback_acceptance() -> dict[str, Any]:
    primary = _FailingProvider(retryable=False, label="permanent provider outage")
    fallback = _FallbackProvider("non-retry-fallback")
    center = LLMProviderCenter(default_provider="primary", max_retries=2)
    center.register("primary", primary, default_model="primary-mini", priority=10)
    center.register("fallback", fallback, default_model="fallback-mini", priority=1)
    response = await center.complete(LLMRequest(messages=[]))
    return _scenario_manifest(
        "agent-core-provider-resilience-non-retry-fallback/v1",
        center=center,
        status="completed" if response.content == "non-retry-fallback" else "failed",
        primary_request_count=len(primary.requests),
        fallback_request_count=len(fallback.requests),
        output=response.content,
    )


async def _explicit_provider_acceptance() -> dict[str, Any]:
    primary = _FailingProvider(retryable=False, label="explicit provider outage")
    fallback = _FallbackProvider("should-not-run")
    center = LLMProviderCenter(default_provider="fallback", max_retries=2)
    center.register("primary", primary, default_model="primary-mini", priority=10)
    center.register("fallback", fallback, default_model="fallback-mini", priority=1)
    error = ""
    try:
        await center.complete(LLMRequest(messages=[], metadata={"provider": "primary"}))
    except LLMProviderError as exc:
        error = str(exc)
    return _scenario_manifest(
        "agent-core-provider-resilience-explicit-provider/v1",
        center=center,
        status="blocked_as_expected" if error and not fallback.requests else "failed",
        primary_request_count=len(primary.requests),
        fallback_request_count=len(fallback.requests),
        output="",
        error=error,
    )


async def _stream_fallback_acceptance() -> dict[str, Any]:
    primary = _StreamingErrorProvider(retryable=True)
    fallback = _StreamingFallbackProvider()
    center = LLMProviderCenter(default_provider="primary", max_retries=1)
    center.register("primary", primary, default_model="primary-mini", priority=10)
    center.register("fallback", fallback, default_model="fallback-mini", priority=1)
    events = [event async for event in center.stream(LLMRequest(messages=[]))]
    manifest = _scenario_manifest(
        "agent-core-provider-resilience-stream-fallback/v1",
        center=center,
        status="completed" if any(event.delta == "hello" for event in events) else "failed",
        primary_request_count=len(primary.requests),
        fallback_request_count=len(fallback.requests),
        output="".join(event.delta for event in events),
    )
    manifest["event_types"] = [event.type for event in events]
    manifest["delta_bytes"] = sum(len(event.delta.encode("utf-8")) for event in events)
    manifest["streamed_call_count"] = sum(1 for call in center.manifest()["calls"] if call["streamed"])
    return manifest


def _scenario_manifest(
    schema_version: str,
    *,
    center: LLMProviderCenter,
    status: str,
    primary_request_count: int,
    fallback_request_count: int,
    output: str,
    error: str = "",
) -> dict[str, Any]:
    manifest = center.manifest()
    failed_calls = [call for call in manifest["calls"] if call["status"] == "failed"]
    completed_calls = [call for call in manifest["calls"] if call["status"] == "completed"]
    return {
        "schema_version": schema_version,
        "status": status,
        "output": output,
        "error": error,
        "primary_request_count": primary_request_count,
        "fallback_request_count": fallback_request_count,
        "call_statuses": [call["status"] for call in manifest["calls"]],
        "failed_call_count": len(failed_calls),
        "completed_call_count": len(completed_calls),
        "retryable_failed_call_count": sum(1 for call in failed_calls if call["retryable"]),
        "non_retryable_failed_call_count": sum(
            1 for call in failed_calls if not call["retryable"]
        ),
        "error_kinds": [
            call["error_classification"]["kind"]
            for call in failed_calls
            if call.get("error_classification")
        ],
        "fallback_used": fallback_request_count > 0,
        "provider_manifest": manifest,
    }


def _provider_trace_eval(report: dict[str, Any]) -> Any:
    return DefaultTraceEvaluator().evaluate(
        _trace_manifest(report, status="completed"),
        TraceEvalSpec(
            name="provider-resilience-errors",
            require_journal_ok=False,
            require_provider_error_classification=True,
            required_provider_error_kinds=("provider_failed",),
        ),
    )


def _trace_manifest(report: dict[str, Any], *, status: str) -> dict[str, Any]:
    return {
        "schema_version": "agent-core-provider-resilience-trace/v1",
        "run": {"status": status},
        "provider": dict(report.get("provider_manifest") or {}),
    }


def _provider_resilience_issues(
    *,
    retry_completion: dict[str, Any],
    retry_after: dict[str, Any],
    non_retry_fallback: dict[str, Any],
    explicit_provider: dict[str, Any],
    stream_fallback: dict[str, Any],
    trace_evals: dict[str, Any],
) -> tuple[AgentCoreProviderResilienceAcceptanceIssue, ...]:
    issues: list[AgentCoreProviderResilienceAcceptanceIssue] = []
    _check_retry_completion(issues, retry_completion)
    _check_retry_after(issues, retry_after)
    _check_non_retry_fallback(issues, non_retry_fallback)
    _check_explicit_provider(issues, explicit_provider)
    _check_stream_fallback(issues, stream_fallback)
    for name, trace_eval in trace_evals.items():
        if isinstance(trace_eval, dict) and trace_eval.get("ok") is True:
            continue
        issues.append(
            AgentCoreProviderResilienceAcceptanceIssue(
                source=f"{name}_trace_eval",
                code="provider_resilience_trace_eval_failed",
                message=f"{name} provider resilience trace eval failed.",
                metadata={"trace_eval": dict(trace_eval) if isinstance(trace_eval, dict) else {}},
            )
        )
    return tuple(issues)


def _check_retry_completion(
    issues: list[AgentCoreProviderResilienceAcceptanceIssue],
    report: dict[str, Any],
) -> None:
    if report.get("status") != "completed" or report.get("fallback_used") is not True:
        issues.append(
            AgentCoreProviderResilienceAcceptanceIssue(
                source="retry_completion",
                code="retry_completion_fallback_missing",
                message="Retryable completion failure did not use fallback.",
                metadata=_prompt_safe(report),
            )
        )
    if report.get("primary_request_count") != 2 or report.get("fallback_request_count") != 1:
        issues.append(
            AgentCoreProviderResilienceAcceptanceIssue(
                source="retry_completion",
                code="retry_completion_attempts_unexpected",
                message="Retryable completion failure did not retry once before fallback.",
                metadata=_prompt_safe(report),
            )
        )


def _check_retry_after(
    issues: list[AgentCoreProviderResilienceAcceptanceIssue],
    report: dict[str, Any],
) -> None:
    if report.get("status") != "completed" or report.get("fallback_used") is not True:
        issues.append(
            AgentCoreProviderResilienceAcceptanceIssue(
                source="retry_after",
                code="retry_after_fallback_missing",
                message="Retry-after limited provider failure did not use fallback.",
                metadata=_prompt_safe(report),
            )
        )
    if report.get("primary_request_count") != 1 or report.get("retryable_failed_call_count") != 0:
        issues.append(
            AgentCoreProviderResilienceAcceptanceIssue(
                source="retry_after",
                code="retry_after_policy_not_honored",
                message="Retry-after policy did not suppress retryable retry attempts.",
                metadata=_prompt_safe(report),
            )
        )


def _check_non_retry_fallback(
    issues: list[AgentCoreProviderResilienceAcceptanceIssue],
    report: dict[str, Any],
) -> None:
    if report.get("status") != "completed" or report.get("fallback_used") is not True:
        issues.append(
            AgentCoreProviderResilienceAcceptanceIssue(
                source="non_retry_fallback",
                code="non_retry_fallback_missing",
                message="Non-retryable provider failure did not move to fallback.",
                metadata=_prompt_safe(report),
            )
        )
    if report.get("primary_request_count") != 1:
        issues.append(
            AgentCoreProviderResilienceAcceptanceIssue(
                source="non_retry_fallback",
                code="non_retry_attempts_unexpected",
                message="Non-retryable provider failure retried unexpectedly.",
                metadata=_prompt_safe(report),
            )
        )


def _check_explicit_provider(
    issues: list[AgentCoreProviderResilienceAcceptanceIssue],
    report: dict[str, Any],
) -> None:
    if report.get("status") != "blocked_as_expected":
        issues.append(
            AgentCoreProviderResilienceAcceptanceIssue(
                source="explicit_provider",
                code="explicit_provider_not_blocked",
                message="Explicit provider request did not fail closed.",
                metadata=_prompt_safe(report),
            )
        )
    if report.get("fallback_request_count") != 0:
        issues.append(
            AgentCoreProviderResilienceAcceptanceIssue(
                source="explicit_provider",
                code="explicit_provider_fallback_used",
                message="Explicit provider request used fallback unexpectedly.",
                metadata=_prompt_safe(report),
            )
        )


def _check_stream_fallback(
    issues: list[AgentCoreProviderResilienceAcceptanceIssue],
    report: dict[str, Any],
) -> None:
    if report.get("status") != "completed" or report.get("output") != "hello":
        issues.append(
            AgentCoreProviderResilienceAcceptanceIssue(
                source="stream_fallback",
                code="stream_fallback_not_completed",
                message="Retryable stream failure did not complete through fallback.",
                metadata=_prompt_safe(report),
            )
        )
    if report.get("primary_request_count") != 2 or report.get("fallback_request_count") != 1:
        issues.append(
            AgentCoreProviderResilienceAcceptanceIssue(
                source="stream_fallback",
                code="stream_fallback_attempts_unexpected",
                message="Retryable stream failure did not retry once before fallback.",
                metadata=_prompt_safe(report),
            )
        )
    if report.get("event_types") != ["delta", "usage", "message_end"]:
        issues.append(
            AgentCoreProviderResilienceAcceptanceIssue(
                source="stream_fallback",
                code="stream_fallback_events_unexpected",
                message="Stream fallback did not preserve expected provider events.",
                metadata=_prompt_safe(report),
            )
        )


def _prompt_safe(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "provider_manifest"}
