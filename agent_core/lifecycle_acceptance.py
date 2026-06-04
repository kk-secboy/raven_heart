"""SDK-level scheduling, cancellation, and timeout acceptance checks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from agent_core.config import AgentProfile
from agent_core.events import ListEventSink
from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.providers import LLMRequest, LLMResponse
from agent_core.runner import (
    AgentManagerConcurrencyPolicy,
    AgentRunRequest,
    AgentSession,
    AgentSessionManager,
)
from agent_core.testing import MockLLMProvider, MockToolRuntime


@dataclass(frozen=True)
class AgentCoreLifecycleAcceptanceIssue:
    """One blocking lifecycle/scheduling acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-lifecycle-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreLifecycleAcceptanceReport:
    """Prompt-safe scheduling, cancellation, and timeout acceptance report."""

    status: str
    queue: dict[str, Any] = field(default_factory=dict)
    queued_cancel: dict[str, Any] = field(default_factory=dict)
    active_cancel: dict[str, Any] = field(default_factory=dict)
    timeout: dict[str, Any] = field(default_factory=dict)
    trace_eval: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreLifecycleAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-lifecycle-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "queue": dict(self.queue),
            "queued_cancel": dict(self.queued_cancel),
            "active_cancel": dict(self.active_cancel),
            "timeout": dict(self.timeout),
            "trace_eval": dict(self.trace_eval),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreLifecycleAcceptanceHarness:
    """Run deterministic SDK scheduling and interrupt checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreLifecycleAcceptanceReport:
        queue = await _queue_acceptance()
        queued_cancel = await _queued_cancel_acceptance()
        active_cancel = await _active_cancel_acceptance()
        timeout = await _timeout_acceptance()
        trace_eval = DefaultTraceEvaluator().evaluate(
            timeout["trace"],
            TraceEvalSpec(
                name="lifecycle-timeout-acceptance",
                expected_status="timeout",
                require_journal_ok=True,
                required_event_types=("run_started", "run_finished", "checkpoint"),
                require_failure_summary=True,
                required_failure_sources=("run",),
                required_failure_kinds=("timeout",),
                max_failure_count=1,
            ),
        ).manifest()
        issues = _lifecycle_acceptance_issues(
            queue=queue,
            queued_cancel=queued_cancel,
            active_cancel=active_cancel,
            timeout=timeout,
            trace_eval=trace_eval,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreLifecycleAcceptanceReport(
            status=status,
            queue=queue,
            queued_cancel=queued_cancel,
            active_cancel=active_cancel,
            timeout=_prompt_safe_timeout(timeout),
            trace_eval=trace_eval,
            issues=issues,
            metadata={"scenario": "agent_core_lifecycle_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_lifecycle_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreLifecycleAcceptanceReport:
    """Run the default scheduling, cancellation, and timeout acceptance checks."""

    return await AgentCoreLifecycleAcceptanceHarness(metadata=dict(metadata or {})).run()


async def _queue_acceptance() -> dict[str, Any]:
    first_provider = _BlockingLifecycleProvider()
    second_provider = MockLLMProvider(
        [{"action": "finish", "arguments": {"output": "queued done"}}]
    )
    session = AgentSession(
        profile=AgentProfile(name="lifecycle-queue"),
        provider=first_provider,
        tools=MockToolRuntime(),
    )
    manager = AgentSessionManager(
        concurrency_policy=AgentManagerConcurrencyPolicy(reject_when_full=False)
    )
    manager.register(session)
    first_key = manager.start("lifecycle-queue", "first task")
    await asyncio.sleep(0)
    session.provider = second_provider
    second_key = manager.start("lifecycle-queue", "second task")
    await asyncio.sleep(0.02)
    before = manager.schedule_snapshot().manifest()
    capacity_before = manager.capacity_status("lifecycle-queue").manifest()
    first_provider.release.set()
    first = await manager.wait(first_key)
    second = await manager.wait(second_key)
    after = manager.schedule_snapshot().manifest()
    return {
        "schema_version": "agent-core-lifecycle-queue-acceptance/v1",
        "first_key": first_key,
        "second_key": second_key,
        "first_status": first.result.status,
        "second_status": second.result.status,
        "second_output_bytes": len(second.result.output.encode("utf-8")),
        "before": before,
        "after": after,
        "capacity_before": capacity_before,
        "second_run": manager.run_state(second_key).manifest(),
        "second_provider_request_count": len(second_provider.requests),
    }


async def _queued_cancel_acceptance() -> dict[str, Any]:
    first_provider = _BlockingLifecycleProvider()
    second_provider = MockLLMProvider(
        [{"action": "finish", "arguments": {"output": "should not run"}}]
    )
    session = AgentSession(
        profile=AgentProfile(name="lifecycle-queued-cancel"),
        provider=first_provider,
        tools=MockToolRuntime(),
    )
    manager = AgentSessionManager(
        concurrency_policy=AgentManagerConcurrencyPolicy(reject_when_full=False)
    )
    manager.register(session)
    first_key = manager.start("lifecycle-queued-cancel", "first task")
    await asyncio.sleep(0)
    session.provider = second_provider
    second_key = manager.start("lifecycle-queued-cancel", "second task")
    await asyncio.sleep(0)
    cancel_result = manager.cancel(second_key, "drop queued")
    cancelled = await manager.wait(second_key)
    first_provider.release.set()
    first = await manager.wait(first_key)
    return {
        "schema_version": "agent-core-lifecycle-queued-cancel-acceptance/v1",
        "first_key": first_key,
        "second_key": second_key,
        "cancel_result": cancel_result,
        "first_status": first.result.status,
        "cancelled_status": cancelled.result.status,
        "queued_interrupt": dict(cancelled.result.metadata.get("interrupt") or {}),
        "second_run": manager.run_state(second_key).manifest(),
        "snapshot": manager.schedule_snapshot().manifest(),
        "second_provider_request_count": len(second_provider.requests),
    }


async def _active_cancel_acceptance() -> dict[str, Any]:
    provider = _BlockingLifecycleProvider()
    manager = AgentSessionManager()
    manager.register(
        AgentSession(
            profile=AgentProfile(name="lifecycle-active-cancel"),
            provider=provider,
            tools=MockToolRuntime(),
        )
    )
    run_key = manager.start("lifecycle-active-cancel", "active task")
    await asyncio.sleep(0)
    cancel_result = manager.cancel(run_key, "operator stop")
    cancelling = manager.run_state(run_key).manifest()
    provider.release.set()
    outcome = await manager.wait(run_key)
    return {
        "schema_version": "agent-core-lifecycle-active-cancel-acceptance/v1",
        "run_key": run_key,
        "cancel_result": cancel_result,
        "cancelling": cancelling,
        "status": outcome.result.status,
        "run": manager.run_state(run_key).manifest(),
        "interrupt": dict(outcome.result.metadata.get("interrupt") or {}),
    }


async def _timeout_acceptance() -> dict[str, Any]:
    provider = _BlockingLifecycleProvider()
    events = ListEventSink()
    manager = AgentSessionManager()
    manager.register(
        AgentSession(
            profile=AgentProfile(name="lifecycle-timeout"),
            provider=provider,
            tools=MockToolRuntime(),
            event_sink=events,
        )
    )
    outcome = await manager.run(
        "lifecycle-timeout",
        AgentRunRequest(task="timeout task", timeout_seconds=0.01),
    )
    run = manager.runs()[0]
    return {
        "schema_version": "agent-core-lifecycle-timeout-acceptance/v1",
        "run_key": run.run_key,
        "status": outcome.result.status,
        "run_status": run.status,
        "interrupt": dict(outcome.result.metadata.get("interrupt") or {}),
        "event_log": events.manifest(),
        "trace": outcome.trace_manifest,
    }


class _BlockingLifecycleProvider:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        await self.release.wait()
        return LLMResponse(action={"action": "finish", "arguments": {"output": "done"}})


def _lifecycle_acceptance_issues(
    *,
    queue: dict[str, Any],
    queued_cancel: dict[str, Any],
    active_cancel: dict[str, Any],
    timeout: dict[str, Any],
    trace_eval: dict[str, Any],
) -> tuple[AgentCoreLifecycleAcceptanceIssue, ...]:
    issues: list[AgentCoreLifecycleAcceptanceIssue] = []
    _check_queue(issues, queue)
    _check_queued_cancel(issues, queued_cancel)
    _check_active_cancel(issues, active_cancel)
    _check_timeout(issues, timeout, trace_eval)
    return tuple(issues)


def _check_queue(
    issues: list[AgentCoreLifecycleAcceptanceIssue],
    queue: dict[str, Any],
) -> None:
    before = queue.get("before") if isinstance(queue.get("before"), dict) else {}
    after = queue.get("after") if isinstance(queue.get("after"), dict) else {}
    queued = before.get("queued_by_session") if isinstance(before.get("queued_by_session"), dict) else {}
    if int(before.get("queued_run_count") or 0) != 1 or not queued.get("lifecycle-queue"):
        issues.append(
            AgentCoreLifecycleAcceptanceIssue(
                source="queue",
                code="queued_run_missing",
                message="Capacity queue did not expose one queued run before release.",
                metadata={"before": dict(before)},
            )
        )
    if queue.get("first_status") != "completed" or queue.get("second_status") != "completed":
        issues.append(
            AgentCoreLifecycleAcceptanceIssue(
                source="queue",
                code="queued_run_not_completed",
                message="Queued run did not complete after active capacity was released.",
                metadata={"queue": dict(queue)},
            )
        )
    if int(after.get("queued_run_count") or 0) != 0:
        issues.append(
            AgentCoreLifecycleAcceptanceIssue(
                source="queue",
                code="queue_not_drained",
                message="Queue was not drained after runs completed.",
                metadata={"after": dict(after)},
            )
        )


def _check_queued_cancel(
    issues: list[AgentCoreLifecycleAcceptanceIssue],
    queued_cancel: dict[str, Any],
) -> None:
    interrupt = queued_cancel.get("queued_interrupt") if isinstance(queued_cancel.get("queued_interrupt"), dict) else {}
    snapshot = queued_cancel.get("snapshot") if isinstance(queued_cancel.get("snapshot"), dict) else {}
    if queued_cancel.get("cancel_result") is not True or queued_cancel.get("cancelled_status") != "cancelled":
        issues.append(
            AgentCoreLifecycleAcceptanceIssue(
                source="queued_cancel",
                code="queued_cancel_failed",
                message="Unclaimed queued run was not cancelled as terminal manager state.",
                metadata={"queued_cancel": dict(queued_cancel)},
            )
        )
    if interrupt.get("queued") is not True:
        issues.append(
            AgentCoreLifecycleAcceptanceIssue(
                source="queued_cancel",
                code="queued_interrupt_missing",
                message="Queued cancellation did not preserve queued interrupt metadata.",
                metadata={"interrupt": dict(interrupt)},
            )
        )
    if int(queued_cancel.get("second_provider_request_count") or 0) != 0:
        issues.append(
            AgentCoreLifecycleAcceptanceIssue(
                source="queued_cancel",
                code="cancelled_queued_run_executed",
                message="Cancelled queued run consumed provider capacity.",
                metadata={"queued_cancel": dict(queued_cancel)},
            )
        )
    if int(snapshot.get("queued_run_count") or 0) != 0:
        issues.append(
            AgentCoreLifecycleAcceptanceIssue(
                source="queued_cancel",
                code="cancelled_run_left_in_queue",
                message="Cancelled queued run remained in pending queue snapshot.",
                metadata={"snapshot": dict(snapshot)},
            )
        )


def _check_active_cancel(
    issues: list[AgentCoreLifecycleAcceptanceIssue],
    active_cancel: dict[str, Any],
) -> None:
    cancelling = active_cancel.get("cancelling") if isinstance(active_cancel.get("cancelling"), dict) else {}
    if active_cancel.get("cancel_result") is not True:
        issues.append(
            AgentCoreLifecycleAcceptanceIssue(
                source="active_cancel",
                code="active_cancel_not_requested",
                message="Active run cancellation request was not accepted.",
                metadata={"active_cancel": dict(active_cancel)},
            )
        )
    if cancelling.get("status") != "cancelling":
        issues.append(
            AgentCoreLifecycleAcceptanceIssue(
                source="active_cancel",
                code="active_run_not_marked_cancelling",
                message="Active manager run was not marked cancelling before provider release.",
                metadata={"cancelling": dict(cancelling)},
            )
        )
    if active_cancel.get("status") != "cancelled":
        issues.append(
            AgentCoreLifecycleAcceptanceIssue(
                source="active_cancel",
                code="active_run_not_cancelled",
                message="Active cancellation did not finish as cancelled.",
                metadata={"active_cancel": dict(active_cancel)},
            )
        )


def _check_timeout(
    issues: list[AgentCoreLifecycleAcceptanceIssue],
    timeout: dict[str, Any],
    trace_eval: dict[str, Any],
) -> None:
    interrupt = timeout.get("interrupt") if isinstance(timeout.get("interrupt"), dict) else {}
    event_log = timeout.get("event_log") if isinstance(timeout.get("event_log"), dict) else {}
    event_types = tuple(
        str(event.get("type") or "")
        for event in event_log.get("events") or ()
        if isinstance(event, dict)
    )
    if timeout.get("status") != "timeout" or timeout.get("run_status") != "timeout":
        issues.append(
            AgentCoreLifecycleAcceptanceIssue(
                source="timeout",
                code="timeout_not_terminal",
                message="Timeout run did not finish as terminal timeout state.",
                metadata={"timeout": _prompt_safe_timeout(timeout)},
            )
        )
    if interrupt.get("kind") != "timeout":
        issues.append(
            AgentCoreLifecycleAcceptanceIssue(
                source="timeout",
                code="timeout_interrupt_missing",
                message="Timeout result did not include timeout interrupt metadata.",
                metadata={"interrupt": dict(interrupt)},
            )
        )
    if "run_started" not in event_types or "run_timeout" not in event_types:
        issues.append(
            AgentCoreLifecycleAcceptanceIssue(
                source="timeout",
                code="timeout_event_log_missing",
                message="Timeout event sink did not record run_started and run_timeout.",
                metadata={"event_types": list(event_types)},
            )
        )
    if trace_eval.get("ok") is not True:
        issues.append(
            AgentCoreLifecycleAcceptanceIssue(
                source="trace_eval",
                code="timeout_trace_eval_failed",
                message="Trace evaluator did not accept the timeout lifecycle contract.",
                metadata={"trace_eval": dict(trace_eval)},
            )
        )


def _prompt_safe_timeout(timeout: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in timeout.items() if key != "trace"}
