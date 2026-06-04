"""SDK-level provider budget and usage acceptance checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.providers import (
    LLMBudgetExceededError,
    LLMMessage,
    LLMProviderCenter,
    LLMRequest,
    LLMResponse,
    LLMUsageLimits,
    UsageInfo,
)


@dataclass(frozen=True)
class AgentCoreBudgetAcceptanceIssue:
    """One budget acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-budget-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreBudgetAcceptanceReport:
    """Prompt-safe report for provider budget and usage checks."""

    status: str
    estimated_cost_block: dict[str, Any] = field(default_factory=dict)
    actual_cost_block: dict[str, Any] = field(default_factory=dict)
    override_success: dict[str, Any] = field(default_factory=dict)
    call_attempt_block: dict[str, Any] = field(default_factory=dict)
    token_block: dict[str, Any] = field(default_factory=dict)
    trace_eval: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreBudgetAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-budget-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "estimated_cost_block": dict(self.estimated_cost_block),
            "actual_cost_block": dict(self.actual_cost_block),
            "override_success": dict(self.override_success),
            "call_attempt_block": dict(self.call_attempt_block),
            "token_block": dict(self.token_block),
            "trace_eval": dict(self.trace_eval),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreBudgetAcceptanceHarness:
    """Run deterministic SDK budget checks without concrete provider clients."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreBudgetAcceptanceReport:
        estimated_cost_block = await _estimated_cost_block()
        actual_cost_block, override_success = await _actual_cost_and_override()
        call_attempt_block = await _call_attempt_block()
        token_block = await _token_block()
        trace_eval = DefaultTraceEvaluator().evaluate(
            _provider_usage_trace(override_success),
            TraceEvalSpec(
                name="budget-acceptance-provider-usage",
                expected_status="completed",
                max_provider_calls=1,
                max_cost_usd=0.02,
                required_provider_names=("budget",),
            ),
        ).manifest()
        issues = _budget_acceptance_issues(
            estimated_cost_block=estimated_cost_block,
            actual_cost_block=actual_cost_block,
            override_success=override_success,
            call_attempt_block=call_attempt_block,
            token_block=token_block,
            trace_eval=trace_eval,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreBudgetAcceptanceReport(
            status=status,
            estimated_cost_block=estimated_cost_block,
            actual_cost_block=actual_cost_block,
            override_success=override_success,
            call_attempt_block=call_attempt_block,
            token_block=token_block,
            trace_eval=trace_eval,
            issues=issues,
            metadata={"scenario": "agent_core_budget_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_budget_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreBudgetAcceptanceReport:
    """Run deterministic provider budget and usage acceptance checks."""

    return await AgentCoreBudgetAcceptanceHarness(metadata=dict(metadata or {})).run()


class _BudgetAcceptanceProvider:
    def __init__(self, responses: tuple[LLMResponse, ...]) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.responses:
            return LLMResponse(content="ok")
        return self.responses.pop(0)


async def _estimated_cost_block() -> dict[str, Any]:
    provider = _BudgetAcceptanceProvider((LLMResponse(content="should-not-call"),))
    center = _budget_center(LLMUsageLimits(max_cost_usd=0.01))
    center.register("budget", provider)
    error = await _budget_error(
        center.complete(
            LLMRequest(
                messages=[LLMMessage(role="user", content="estimate")],
                metadata={"estimated_cost_usd": 0.02},
            )
        )
    )
    return {
        "schema_version": "agent-core-budget-check/v1",
        "check": "estimated_cost",
        "blocked": bool(error),
        "error": error,
        "provider_request_count": len(provider.requests),
        "center": center.manifest(),
    }


async def _actual_cost_and_override() -> tuple[dict[str, Any], dict[str, Any]]:
    provider = _BudgetAcceptanceProvider(
        (
            LLMResponse(content="too-expensive", usage=UsageInfo(cost_usd=0.02)),
            LLMResponse(
                content="ok",
                usage=UsageInfo(input_tokens=1, output_tokens=2, total_tokens=3, cost_usd=0.01),
            ),
        )
    )
    center = _budget_center(LLMUsageLimits(max_cost_usd=0.01))
    center.register("budget", provider)
    error = await _budget_error(
        center.complete(LLMRequest(messages=[LLMMessage(role="user", content="actual")]))
    )
    actual = {
        "schema_version": "agent-core-budget-check/v1",
        "check": "actual_cost",
        "blocked": bool(error),
        "error": error,
        "provider_request_count": len(provider.requests),
        "center_after_block": center.manifest(),
    }
    response = await center.complete(
        LLMRequest(
            messages=[LLMMessage(role="user", content="override")],
            metadata={"max_cost_usd": 0.02},
        )
    )
    override = {
        "schema_version": "agent-core-budget-check/v1",
        "check": "request_override",
        "completed": response.content == "ok",
        "response": response.manifest(),
        "provider_request_count": len(provider.requests),
        "center": center.manifest(),
    }
    return actual, override


async def _call_attempt_block() -> dict[str, Any]:
    provider = _BudgetAcceptanceProvider(
        (
            LLMResponse(content="first", usage=UsageInfo(total_tokens=1)),
            LLMResponse(content="second", usage=UsageInfo(total_tokens=1)),
        )
    )
    center = _budget_center(LLMUsageLimits(max_call_attempts=1))
    center.register("budget", provider)
    first = await center.complete(LLMRequest(messages=[LLMMessage(role="user", content="one")]))
    error = await _budget_error(
        center.complete(LLMRequest(messages=[LLMMessage(role="user", content="two")]))
    )
    return {
        "schema_version": "agent-core-budget-check/v1",
        "check": "call_attempt",
        "first_completed": first.content == "first",
        "blocked": bool(error),
        "error": error,
        "provider_request_count": len(provider.requests),
        "center": center.manifest(),
    }


async def _token_block() -> dict[str, Any]:
    estimated_provider = _BudgetAcceptanceProvider((LLMResponse(content="should-not-call"),))
    estimated_center = _budget_center(LLMUsageLimits(max_input_tokens=4))
    estimated_center.register("budget", estimated_provider)
    estimated_error = await _budget_error(
        estimated_center.complete(
            LLMRequest(
                messages=[LLMMessage(role="user", content="estimated token")],
                metadata={"estimated_input_tokens": 5},
            )
        )
    )

    actual_provider = _BudgetAcceptanceProvider(
        (LLMResponse(content="too-many", usage=UsageInfo(total_tokens=7)),)
    )
    actual_center = _budget_center(LLMUsageLimits(max_total_tokens=6))
    actual_center.register("budget", actual_provider)
    actual_error = await _budget_error(
        actual_center.complete(LLMRequest(messages=[LLMMessage(role="user", content="actual token")]))
    )
    return {
        "schema_version": "agent-core-budget-check/v1",
        "check": "token",
        "estimated_blocked": bool(estimated_error),
        "estimated_error": estimated_error,
        "estimated_provider_request_count": len(estimated_provider.requests),
        "actual_blocked": bool(actual_error),
        "actual_error": actual_error,
        "actual_provider_request_count": len(actual_provider.requests),
        "estimated_center": estimated_center.manifest(),
        "actual_center": actual_center.manifest(),
    }


def _budget_center(limits: LLMUsageLimits) -> LLMProviderCenter:
    return LLMProviderCenter(
        default_provider="budget",
        default_model="budget-mini",
        usage_limits=limits,
    )


async def _budget_error(awaitable: Any) -> str:
    try:
        await awaitable
    except LLMBudgetExceededError as exc:
        return str(exc)
    return ""


def _provider_usage_trace(override_success: dict[str, Any]) -> dict[str, Any]:
    center = dict(override_success.get("center") or {})
    return {
        "schema_version": "agent-core-run-trace-bundle/v1",
        "run": {
            "run_id": "budget-acceptance",
            "status": "completed",
            "iterations": 1,
            "output_bytes": 2,
        },
        "summary": {
            "status": "completed",
            "provider_call_count": 1,
        },
        "provider": center,
    }


def _budget_acceptance_issues(
    *,
    estimated_cost_block: dict[str, Any],
    actual_cost_block: dict[str, Any],
    override_success: dict[str, Any],
    call_attempt_block: dict[str, Any],
    token_block: dict[str, Any],
    trace_eval: dict[str, Any],
) -> tuple[AgentCoreBudgetAcceptanceIssue, ...]:
    checks = (
        (
            "estimated_cost_block",
            estimated_cost_block.get("blocked") is True
            and estimated_cost_block.get("provider_request_count") == 0,
            "Estimated cost did not block before provider execution.",
        ),
        (
            "actual_cost_block",
            actual_cost_block.get("blocked") is True
            and actual_cost_block.get("provider_request_count") == 1
            and _usage_cost(actual_cost_block.get("center_after_block")) == 0.0,
            "Actual cost did not block without recording over-budget usage.",
        ),
        (
            "request_override",
            override_success.get("completed") is True
            and _usage_cost(override_success.get("center")) == 0.01,
            "Request-level budget override did not allow the lower-cost call.",
        ),
        (
            "call_attempt_block",
            call_attempt_block.get("blocked") is True
            and call_attempt_block.get("provider_request_count") == 1,
            "Call-attempt budget did not block before the second provider call.",
        ),
        (
            "token_block",
            token_block.get("estimated_blocked") is True
            and token_block.get("estimated_provider_request_count") == 0
            and token_block.get("actual_blocked") is True
            and token_block.get("actual_provider_request_count") == 1
            and _usage_total_tokens(token_block.get("actual_center")) == 0,
            "Token budgets did not block estimated and actual over-budget calls.",
        ),
        (
            "trace_eval",
            trace_eval.get("ok") is True,
            "Budget acceptance provider usage trace did not pass trace eval.",
        ),
    )
    issues: list[AgentCoreBudgetAcceptanceIssue] = []
    for source, ok, message in checks:
        if ok:
            continue
        issues.append(
            AgentCoreBudgetAcceptanceIssue(
                source=source,
                code=f"{source}_failed",
                message=message,
                metadata={
                    "estimated_cost_block": estimated_cost_block,
                    "actual_cost_block": actual_cost_block,
                    "override_success": override_success,
                    "call_attempt_block": call_attempt_block,
                    "token_block": token_block,
                    "trace_eval": trace_eval,
                },
            )
        )
    return tuple(issues)


def _usage_cost(center_manifest: Any) -> float:
    if not isinstance(center_manifest, dict):
        return 0.0
    usage = center_manifest.get("usage")
    if not isinstance(usage, dict):
        return 0.0
    return float(usage.get("cost_usd") or 0.0)


def _usage_total_tokens(center_manifest: Any) -> int:
    if not isinstance(center_manifest, dict):
        return 0
    usage = center_manifest.get("usage")
    if not isinstance(usage, dict):
        return 0
    return int(usage.get("total_tokens") or 0)
