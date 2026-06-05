"""SDK-level interrupt, deadline, and timeout recovery acceptance checks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from agent_core.config import AgentProfile
from agent_core.events import ListEventSink
from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.harness import CancelToken
from agent_core.providers import LLMRequest, LLMResponse
from agent_core.runner import AgentRunRequest, AgentRunner, AgentSession, AgentSessionManager
from agent_core.testing import MockLLMProvider
from agent_core.tools import ToolInvocation, ToolResult, ToolSpec


@dataclass(frozen=True)
class AgentCoreInterruptAcceptanceIssue:
    """One blocking interrupt/deadline acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-interrupt-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreInterruptAcceptanceReport:
    """Prompt-safe report for interrupt and timeout recovery checks."""

    status: str
    pre_cancel: dict[str, Any] = field(default_factory=dict)
    provider_timeout: dict[str, Any] = field(default_factory=dict)
    tool_timeout: dict[str, Any] = field(default_factory=dict)
    manager_cancel_reset: dict[str, Any] = field(default_factory=dict)
    trace_evals: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreInterruptAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-interrupt-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "pre_cancel": dict(self.pre_cancel),
            "provider_timeout": dict(self.provider_timeout),
            "tool_timeout": dict(self.tool_timeout),
            "manager_cancel_reset": dict(self.manager_cancel_reset),
            "trace_evals": dict(self.trace_evals),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreInterruptAcceptanceHarness:
    """Run deterministic pure-SDK interrupt and deadline checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreInterruptAcceptanceReport:
        pre_cancel = await _pre_cancel_acceptance()
        provider_timeout = await _provider_timeout_acceptance()
        tool_timeout = await _tool_timeout_acceptance()
        manager_cancel_reset = await _manager_cancel_reset_acceptance()
        trace_evals = {
            "schema_version": "agent-core-interrupt-trace-evals/v1",
            "pre_cancel": _trace_eval(
                pre_cancel["trace"],
                TraceEvalSpec(
                    name="interrupt-pre-cancel",
                    expected_status="cancelled",
                    require_journal_ok=True,
                    required_event_types=("run_started", "run_cancelled"),
                ),
            ),
            "provider_timeout": _trace_eval(
                provider_timeout["trace"],
                TraceEvalSpec(
                    name="interrupt-provider-timeout",
                    expected_status="timeout",
                    require_journal_ok=True,
                    required_event_types=("run_started", "run_timeout", "checkpoint"),
                    require_failure_summary=True,
                    required_failure_sources=("run",),
                    required_failure_kinds=("timeout",),
                    max_failure_count=2,
                ),
            ),
            "tool_timeout": _trace_eval(
                tool_timeout["trace"],
                TraceEvalSpec(
                    name="interrupt-tool-timeout",
                    expected_status="timeout",
                    require_journal_ok=True,
                    required_event_types=(
                        "run_started",
                        "model_response",
                        "tool_started",
                        "run_timeout",
                        "checkpoint",
                    ),
                    require_failure_summary=True,
                    required_failure_sources=("run",),
                    required_failure_kinds=("timeout",),
                    max_failure_count=2,
                ),
            ),
            "manager_cancel_reset": _trace_eval(
                manager_cancel_reset["second_trace"],
                TraceEvalSpec(
                    name="interrupt-manager-reset-success",
                    expected_status="completed",
                    require_journal_ok=True,
                    required_event_types=("run_started", "run_finished", "checkpoint"),
                ),
            ),
        }
        issues = _interrupt_acceptance_issues(
            pre_cancel=pre_cancel,
            provider_timeout=provider_timeout,
            tool_timeout=tool_timeout,
            manager_cancel_reset=manager_cancel_reset,
            trace_evals=trace_evals,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreInterruptAcceptanceReport(
            status=status,
            pre_cancel=_without_trace(pre_cancel),
            provider_timeout=_without_trace(provider_timeout),
            tool_timeout=_without_trace(tool_timeout),
            manager_cancel_reset=_without_trace(manager_cancel_reset),
            trace_evals=trace_evals,
            issues=issues,
            metadata={"scenario": "agent_core_interrupt_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_interrupt_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreInterruptAcceptanceReport:
    """Run the default interrupt, deadline, and recovery acceptance checks."""

    return await AgentCoreInterruptAcceptanceHarness(metadata=dict(metadata or {})).run()


async def _pre_cancel_acceptance() -> dict[str, Any]:
    provider = _BlockingProvider()
    events = ListEventSink()
    cancel_token = CancelToken()
    cancel_token.cancel("preflight stop", metadata={"source": "acceptance"})
    session = AgentSession(
        profile=AgentProfile(name="interrupt-pre-cancel"),
        provider=provider,
        tools=_BlockingToolRuntime(),
        event_sink=events,
        cancel_token=cancel_token,
    )
    outcome = await AgentRunner(session).run(AgentRunRequest(task="must not call provider"))
    return {
        "schema_version": "agent-core-interrupt-pre-cancel-acceptance/v1",
        "status": outcome.result.status,
        "interrupt": dict(outcome.result.metadata.get("interrupt") or {}),
        "provider_request_count": len(provider.requests),
        "event_types": _event_types(outcome.trace_manifest),
        "trace": outcome.trace_manifest,
    }


async def _provider_timeout_acceptance() -> dict[str, Any]:
    provider = _BlockingProvider()
    events = ListEventSink()
    session = AgentSession(
        profile=AgentProfile(name="interrupt-provider-timeout"),
        provider=provider,
        tools=_BlockingToolRuntime(),
        event_sink=events,
    )
    outcome = await AgentRunner(session).run(
        AgentRunRequest(task="provider timeout", timeout_seconds=0.01)
    )
    return {
        "schema_version": "agent-core-interrupt-provider-timeout-acceptance/v1",
        "status": outcome.result.status,
        "interrupt": dict(outcome.result.metadata.get("interrupt") or {}),
        "provider_request_count": len(provider.requests),
        "checkpoint_phases": _checkpoint_phases(outcome.trace_manifest),
        "event_types": _event_types(outcome.trace_manifest),
        "trace": outcome.trace_manifest,
    }


async def _tool_timeout_acceptance() -> dict[str, Any]:
    tool_runtime = _BlockingToolRuntime(tool_name="slow_lookup")
    provider = MockLLMProvider(
        [
            {
                "action": "call_tool",
                "arguments": {
                    "tool_name": "slow_lookup",
                    "arguments": {"query": "deadline"},
                },
            }
        ]
    )
    events = ListEventSink()
    session = AgentSession(
        profile=AgentProfile(name="interrupt-tool-timeout"),
        provider=provider,
        tools=tool_runtime,
        event_sink=events,
    )
    outcome = await AgentRunner(session).run(
        AgentRunRequest(task="tool timeout", timeout_seconds=0.01)
    )
    return {
        "schema_version": "agent-core-interrupt-tool-timeout-acceptance/v1",
        "status": outcome.result.status,
        "interrupt": dict(outcome.result.metadata.get("interrupt") or {}),
        "provider_request_count": len(provider.requests),
        "tool_invocation_count": len(tool_runtime.invocations),
        "checkpoint_phases": _checkpoint_phases(outcome.trace_manifest),
        "event_types": _event_types(outcome.trace_manifest),
        "trace": outcome.trace_manifest,
    }


async def _manager_cancel_reset_acceptance() -> dict[str, Any]:
    provider = _StepBlockingProvider()
    manager = AgentSessionManager()
    manager.register(
        AgentSession(
            profile=AgentProfile(name="interrupt-manager-reset"),
            provider=provider,
            tools=_BlockingToolRuntime(),
            event_sink=ListEventSink(),
        )
    )
    first_key = manager.start("interrupt-manager-reset", "cancel active run")
    await provider.wait_for_request_count(1)
    cancel_result = manager.cancel(first_key, "operator stop")
    cancelling = manager.run_state(first_key).manifest()
    provider.release_request(0)
    first_outcome = await manager.wait(first_key)

    second_key = manager.start("interrupt-manager-reset", "run after cancel")
    await provider.wait_for_request_count(2)
    provider.release_request(1)
    second_outcome = await manager.wait(second_key)
    session = manager.session("interrupt-manager-reset")
    return {
        "schema_version": "agent-core-interrupt-manager-cancel-reset-acceptance/v1",
        "first_key": first_key,
        "second_key": second_key,
        "cancel_result": cancel_result,
        "cancelling_status": cancelling.get("status"),
        "first_status": first_outcome.result.status,
        "second_status": second_outcome.result.status,
        "first_interrupt": dict(first_outcome.result.metadata.get("interrupt") or {}),
        "provider_request_count": len(provider.requests),
        "cancel_token_after_second": session.cancel_token.manifest(),
        "first_event_types": _event_types(first_outcome.trace_manifest),
        "second_event_types": _event_types(second_outcome.trace_manifest),
        "first_run": manager.run_state(first_key).manifest(),
        "second_run": manager.run_state(second_key).manifest(),
        "second_trace": second_outcome.trace_manifest,
    }


class _BlockingProvider:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        await self.release.wait()
        return LLMResponse(action={"action": "finish", "arguments": {"output": "released"}})


class _StepBlockingProvider:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self._releases: list[asyncio.Event] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        release = asyncio.Event()
        self.requests.append(request)
        self._releases.append(release)
        await release.wait()
        return LLMResponse(
            action={
                "action": "finish",
                "arguments": {"output": f"released-{len(self.requests)}"},
            }
        )

    async def wait_for_request_count(self, count: int, *, timeout_seconds: float = 1.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while len(self.requests) < count:
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"provider request count did not reach {count}")
            await asyncio.sleep(0)

    def release_request(self, index: int) -> None:
        self._releases[index].set()


class _BlockingToolRuntime:
    def __init__(self, *, tool_name: str = "blocked_tool") -> None:
        self.tool_name = tool_name
        self.release = asyncio.Event()
        self.invocations: list[ToolInvocation] = []

    def specs(self) -> tuple[ToolSpec, ...]:
        return (ToolSpec(name=self.tool_name),)

    async def invoke(self, invocation: ToolInvocation) -> ToolResult:
        self.invocations.append(invocation)
        await self.release.wait()
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            content="released",
        )


def _trace_eval(trace: dict[str, Any], spec: TraceEvalSpec) -> dict[str, Any]:
    return DefaultTraceEvaluator().evaluate(trace, spec).manifest()


def _event_types(trace: dict[str, Any]) -> list[str]:
    event_log = trace.get("event_log") if isinstance(trace.get("event_log"), dict) else {}
    return [
        str(event.get("type") or "")
        for event in event_log.get("events") or ()
        if isinstance(event, dict)
    ]


def _checkpoint_phases(trace: dict[str, Any]) -> list[str]:
    journal = trace.get("journal_replay") if isinstance(trace.get("journal_replay"), dict) else {}
    phases: list[str] = []
    for event in journal.get("events") or ():
        if not isinstance(event, dict) or event.get("event_type") != "checkpoint":
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
        phase = str(state.get("phase") or "")
        if phase:
            phases.append(phase)
    return phases


def _interrupt_acceptance_issues(
    *,
    pre_cancel: dict[str, Any],
    provider_timeout: dict[str, Any],
    tool_timeout: dict[str, Any],
    manager_cancel_reset: dict[str, Any],
    trace_evals: dict[str, Any],
) -> tuple[AgentCoreInterruptAcceptanceIssue, ...]:
    issues: list[AgentCoreInterruptAcceptanceIssue] = []
    _check_pre_cancel(issues, pre_cancel)
    _check_provider_timeout(issues, provider_timeout)
    _check_tool_timeout(issues, tool_timeout)
    _check_manager_cancel_reset(issues, manager_cancel_reset)
    for name in ("pre_cancel", "provider_timeout", "tool_timeout", "manager_cancel_reset"):
        report = trace_evals.get(name) if isinstance(trace_evals.get(name), dict) else {}
        if report.get("ok") is not True:
            issues.append(
                AgentCoreInterruptAcceptanceIssue(
                    source="trace_eval",
                    code=f"{name}_trace_eval_failed",
                    message=f"Trace evaluator did not accept interrupt contract: {name}",
                    metadata={"trace_eval": dict(report)},
                )
            )
    return tuple(issues)


def _check_pre_cancel(
    issues: list[AgentCoreInterruptAcceptanceIssue],
    scenario: dict[str, Any],
) -> None:
    interrupt = scenario.get("interrupt") if isinstance(scenario.get("interrupt"), dict) else {}
    if scenario.get("status") != "cancelled" or interrupt.get("kind") != "cancelled":
        issues.append(
            AgentCoreInterruptAcceptanceIssue(
                source="pre_cancel",
                code="pre_cancel_not_cancelled",
                message="Pre-cancelled run did not finish as cancelled.",
                metadata=_without_trace(scenario),
            )
        )
    if int(scenario.get("provider_request_count") or 0) != 0:
        issues.append(
            AgentCoreInterruptAcceptanceIssue(
                source="pre_cancel",
                code="pre_cancel_called_provider",
                message="Pre-cancelled run consumed provider capacity.",
                metadata=_without_trace(scenario),
            )
        )
    if "run_cancelled" not in set(scenario.get("event_types") or ()):
        issues.append(
            AgentCoreInterruptAcceptanceIssue(
                source="pre_cancel",
                code="pre_cancel_event_missing",
                message="Pre-cancelled run did not emit run_cancelled.",
                metadata=_without_trace(scenario),
            )
        )


def _check_provider_timeout(
    issues: list[AgentCoreInterruptAcceptanceIssue],
    scenario: dict[str, Any],
) -> None:
    interrupt = scenario.get("interrupt") if isinstance(scenario.get("interrupt"), dict) else {}
    if scenario.get("status") != "timeout" or interrupt.get("kind") != "timeout":
        issues.append(
            AgentCoreInterruptAcceptanceIssue(
                source="provider_timeout",
                code="provider_timeout_not_terminal",
                message="Provider deadline did not finish as timeout.",
                metadata=_without_trace(scenario),
            )
        )
    if int(scenario.get("provider_request_count") or 0) != 1:
        issues.append(
            AgentCoreInterruptAcceptanceIssue(
                source="provider_timeout",
                code="provider_timeout_request_count_wrong",
                message="Provider timeout should have exactly one provider request.",
                metadata=_without_trace(scenario),
            )
        )
    if "provider" not in set(scenario.get("checkpoint_phases") or ()):
        issues.append(
            AgentCoreInterruptAcceptanceIssue(
                source="provider_timeout",
                code="provider_timeout_checkpoint_phase_missing",
                message="Provider timeout checkpoint did not record provider phase.",
                metadata=_without_trace(scenario),
            )
        )


def _check_tool_timeout(
    issues: list[AgentCoreInterruptAcceptanceIssue],
    scenario: dict[str, Any],
) -> None:
    interrupt = scenario.get("interrupt") if isinstance(scenario.get("interrupt"), dict) else {}
    if scenario.get("status") != "timeout" or interrupt.get("kind") != "timeout":
        issues.append(
            AgentCoreInterruptAcceptanceIssue(
                source="tool_timeout",
                code="tool_timeout_not_terminal",
                message="Tool deadline did not finish as timeout.",
                metadata=_without_trace(scenario),
            )
        )
    if int(scenario.get("tool_invocation_count") or 0) != 1:
        issues.append(
            AgentCoreInterruptAcceptanceIssue(
                source="tool_timeout",
                code="tool_timeout_invocation_count_wrong",
                message="Tool timeout should have exactly one tool invocation.",
                metadata=_without_trace(scenario),
            )
        )
    if "tool" not in set(scenario.get("checkpoint_phases") or ()):
        issues.append(
            AgentCoreInterruptAcceptanceIssue(
                source="tool_timeout",
                code="tool_timeout_checkpoint_phase_missing",
                message="Tool timeout checkpoint did not record tool phase.",
                metadata=_without_trace(scenario),
            )
        )


def _check_manager_cancel_reset(
    issues: list[AgentCoreInterruptAcceptanceIssue],
    scenario: dict[str, Any],
) -> None:
    token = (
        scenario.get("cancel_token_after_second")
        if isinstance(scenario.get("cancel_token_after_second"), dict)
        else {}
    )
    first_interrupt = (
        scenario.get("first_interrupt")
        if isinstance(scenario.get("first_interrupt"), dict)
        else {}
    )
    if scenario.get("cancel_result") is not True or scenario.get("cancelling_status") != "cancelling":
        issues.append(
            AgentCoreInterruptAcceptanceIssue(
                source="manager_cancel_reset",
                code="manager_cancel_not_accepted",
                message="Manager did not accept active cancellation and mark run as cancelling.",
                metadata=_without_trace(scenario),
            )
        )
    if scenario.get("first_status") != "cancelled" or first_interrupt.get("kind") != "cancelled":
        issues.append(
            AgentCoreInterruptAcceptanceIssue(
                source="manager_cancel_reset",
                code="manager_cancel_not_terminal",
                message="Active manager cancellation did not finish as cancelled.",
                metadata=_without_trace(scenario),
            )
        )
    if scenario.get("second_status") != "completed":
        issues.append(
            AgentCoreInterruptAcceptanceIssue(
                source="manager_cancel_reset",
                code="cancel_token_poisoned_next_run",
                message="Same session did not complete a new run after active cancellation.",
                metadata=_without_trace(scenario),
            )
        )
    if token.get("cancelled") is not False:
        issues.append(
            AgentCoreInterruptAcceptanceIssue(
                source="manager_cancel_reset",
                code="cancel_token_not_reset",
                message="Session cancel token remained cancelled after a successful follow-up run.",
                metadata=_without_trace(scenario),
            )
        )
    if int(scenario.get("provider_request_count") or 0) != 2:
        issues.append(
            AgentCoreInterruptAcceptanceIssue(
                source="manager_cancel_reset",
                code="manager_reset_request_count_wrong",
                message="Manager cancel/reset scenario should issue two provider requests.",
                metadata=_without_trace(scenario),
            )
        )


def _without_trace(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in scenario.items()
        if key not in {"trace", "second_trace"}
    }
