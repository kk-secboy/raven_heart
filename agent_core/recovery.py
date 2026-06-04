"""SDK-level recovery acceptance checks for provider and tool failures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.providers import (
    LLMProviderCenter,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    RetryHint,
)
from agent_core.tools import (
    ToolExecutionCenter,
    ToolInvocation,
    ToolResult,
    ToolRetryPolicy,
    ToolSpec,
)


@dataclass(frozen=True)
class AgentCoreRecoveryIssue:
    """One blocking recovery acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-recovery-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreRecoveryReport:
    """Prompt-safe provider/tool recovery acceptance report."""

    status: str
    provider_recovery: dict[str, Any] = field(default_factory=dict)
    tool_recovery: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreRecoveryIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-recovery-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "provider_recovery": dict(self.provider_recovery),
            "tool_recovery": dict(self.tool_recovery),
            "metadata": dict(self.metadata),
        }


class RecoveryFailingProvider:
    """Provider that fails deterministically with a retryable provider error."""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        raise LLMProviderError(
            "temporary provider outage",
            retry_hint=RetryHint(retryable=True, reason="temporary"),
        )


class RecoveryFallbackProvider:
    """Provider that succeeds after the primary provider fails."""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(content="fallback-ok")


class RecoveryFlakyToolRuntime:
    """Tool runtime that fails once with a retryable result, then succeeds."""

    def __init__(self) -> None:
        self.invocations: list[ToolInvocation] = []

    def specs(self) -> tuple[ToolSpec, ...]:
        return (ToolSpec(name="flaky", description="Flaky recovery tool"),)

    async def invoke(self, invocation: ToolInvocation) -> ToolResult:
        self.invocations.append(invocation)
        if len(self.invocations) == 1:
            return ToolResult(
                call_id=invocation.call_id,
                tool_name=invocation.tool_name,
                status="failed",
                error="temporary tool outage",
                metadata={"retryable": True},
            )
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            content="tool-ok",
        )


@dataclass(frozen=True)
class AgentCoreRecoveryHarness:
    """Run deterministic provider fallback and tool retry checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreRecoveryReport:
        provider_recovery = await _provider_recovery_manifest()
        tool_recovery = await _tool_recovery_manifest()
        issues = _recovery_issues(provider_recovery, tool_recovery)
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreRecoveryReport(
            status=status,
            provider_recovery=provider_recovery,
            tool_recovery=tool_recovery,
            issues=issues,
            metadata={"scenario": "agent_core_recovery", **dict(self.metadata)},
        )


async def run_agent_core_recovery_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreRecoveryReport:
    """Run the default provider/tool recovery acceptance checks."""

    return await AgentCoreRecoveryHarness(metadata=dict(metadata or {})).run()


async def _provider_recovery_manifest() -> dict[str, Any]:
    primary = RecoveryFailingProvider()
    fallback = RecoveryFallbackProvider()
    center = LLMProviderCenter(default_provider="primary", max_retries=1)
    center.register("primary", primary, default_model="primary-mini", priority=10)
    center.register("fallback", fallback, default_model="fallback-mini", priority=1)
    response = await center.complete(LLMRequest(messages=[]))
    manifest = center.manifest()
    failed_calls = [call for call in manifest["calls"] if call["status"] == "failed"]
    completed_calls = [call for call in manifest["calls"] if call["status"] == "completed"]
    error_kinds = [
        call["error_classification"]["kind"]
        for call in failed_calls
        if call.get("error_classification")
    ]
    return {
        "schema_version": "agent-core-provider-recovery/v1",
        "status": "completed" if response.content == "fallback-ok" else "failed",
        "output": response.content,
        "primary_request_count": len(primary.requests),
        "fallback_request_count": len(fallback.requests),
        "call_statuses": [call["status"] for call in manifest["calls"]],
        "failed_call_count": len(failed_calls),
        "completed_call_count": len(completed_calls),
        "retryable_failed_call_count": sum(1 for call in failed_calls if call["retryable"]),
        "error_kinds": error_kinds,
        "fallback_used": bool(fallback.requests),
        "provider_manifest": manifest,
    }


async def _tool_recovery_manifest() -> dict[str, Any]:
    runtime = RecoveryFlakyToolRuntime()
    center = ToolExecutionCenter(runtime, retry_policy=ToolRetryPolicy(max_attempts=2))
    result = await center.invoke(ToolInvocation(tool_name="flaky", call_id="recovery-tool-call"))
    manifest = center.manifest()
    record = manifest["records"][0] if manifest["records"] else {}
    summary = result.metadata.get("tool_execution") if isinstance(result.metadata, dict) else {}
    return {
        "schema_version": "agent-core-tool-recovery/v1",
        "status": result.status,
        "ok": result.ok,
        "output": result.content,
        "invocation_count": len(runtime.invocations),
        "attempt_count": int(summary.get("attempt_count") or record.get("attempt_count") or 0),
        "retried": bool(summary.get("retried") or record.get("retried")),
        "retryable_attempts": list(summary.get("retryable_attempts") or ()),
        "attempt_statuses": list(summary.get("attempt_statuses") or ()),
        "error_kinds": [
            attempt["error_classification"]["kind"]
            for attempt in record.get("attempts", ())
            if attempt.get("error_classification")
        ],
        "tool_manifest": manifest,
    }


def _recovery_issues(
    provider_recovery: dict[str, Any],
    tool_recovery: dict[str, Any],
) -> tuple[AgentCoreRecoveryIssue, ...]:
    issues: list[AgentCoreRecoveryIssue] = []
    if provider_recovery.get("status") != "completed":
        issues.append(
            AgentCoreRecoveryIssue(
                source="provider",
                code="provider_recovery_not_completed",
                message="Provider recovery scenario did not complete.",
            )
        )
    if provider_recovery.get("fallback_used") is not True:
        issues.append(
            AgentCoreRecoveryIssue(
                source="provider",
                code="provider_fallback_missing",
                message="Provider recovery scenario did not use fallback.",
            )
        )
    if int(provider_recovery.get("retryable_failed_call_count") or 0) < 1:
        issues.append(
            AgentCoreRecoveryIssue(
                source="provider",
                code="provider_retryable_failure_missing",
                message="Provider recovery scenario did not record retryable failures.",
            )
        )
    if "provider_failed" not in set(provider_recovery.get("error_kinds") or ()):
        issues.append(
            AgentCoreRecoveryIssue(
                source="provider",
                code="provider_error_classification_missing",
                message="Provider recovery scenario did not classify provider failures.",
            )
        )
    if tool_recovery.get("ok") is not True:
        issues.append(
            AgentCoreRecoveryIssue(
                source="tool",
                code="tool_recovery_not_completed",
                message="Tool recovery scenario did not complete.",
            )
        )
    if tool_recovery.get("retried") is not True:
        issues.append(
            AgentCoreRecoveryIssue(
                source="tool",
                code="tool_retry_missing",
                message="Tool recovery scenario did not retry.",
            )
        )
    if int(tool_recovery.get("attempt_count") or 0) < 2:
        issues.append(
            AgentCoreRecoveryIssue(
                source="tool",
                code="tool_attempt_count_too_low",
                message="Tool recovery scenario did not record at least two attempts.",
            )
        )
    if "tool_failed" not in set(tool_recovery.get("error_kinds") or ()):
        issues.append(
            AgentCoreRecoveryIssue(
                source="tool",
                code="tool_error_classification_missing",
                message="Tool recovery scenario did not classify tool failures.",
            )
        )
    return tuple(issues)
