"""SDK-level streaming and event-stream acceptance checks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from agent_core.config import AgentProfile
from agent_core.events import EventStreamCursor, ListEventSink
from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.events import EventStreamBatch, EventStreamTail
from agent_core.providers import LLMProviderCenter, LLMRequest, LLMResponse
from agent_core.runner import AgentRunner, AgentRunRequest, AgentSession, AgentSessionManager
from agent_core.testing import MockLLMProvider, MockToolRuntime


@dataclass(frozen=True)
class AgentCoreEventAcceptanceIssue:
    """One blocking streaming/event acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-event-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreEventAcceptanceReport:
    """Prompt-safe streaming and event-stream acceptance report."""

    status: str
    streaming_run: dict[str, Any] = field(default_factory=dict)
    manager_stream: dict[str, Any] = field(default_factory=dict)
    event_log: dict[str, Any] = field(default_factory=dict)
    trace_eval: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreEventAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-event-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "streaming_run": dict(self.streaming_run),
            "manager_stream": dict(self.manager_stream),
            "event_log": dict(self.event_log),
            "trace_eval": dict(self.trace_eval),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreEventAcceptanceHarness:
    """Run deterministic SDK event-stream checks without runtime adapters."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreEventAcceptanceReport:
        streaming = await _streaming_run_acceptance()
        manager = await _manager_event_stream_acceptance()
        trace_eval = DefaultTraceEvaluator().evaluate(
            streaming["trace"],
            TraceEvalSpec(
                name="event-acceptance",
                expected_status="completed",
                require_journal_ok=True,
                require_provider_streaming=True,
                required_provider_stream_event_types=("action", "message_end"),
                max_provider_stream_errors=0,
                require_event_log=True,
                required_event_log_types=("run_started", "model_stream", "run_finished"),
                require_terminal_event=True,
                require_event_sequence_monotonic=True,
                max_duplicate_event_sequences=0,
            ),
        ).manifest()
        issues = _event_acceptance_issues(
            streaming_run=streaming["streaming_run"],
            manager_stream=manager,
            event_log=streaming["event_log"],
            trace_eval=trace_eval,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreEventAcceptanceReport(
            status=status,
            streaming_run=streaming["streaming_run"],
            manager_stream=manager,
            event_log=streaming["event_log"],
            trace_eval=trace_eval,
            issues=issues,
            metadata={"scenario": "agent_core_event_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_event_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreEventAcceptanceReport:
    """Run the default streaming and event-stream acceptance checks."""

    return await AgentCoreEventAcceptanceHarness(metadata=dict(metadata or {})).run()


async def _streaming_run_acceptance() -> dict[str, Any]:
    provider = MockLLMProvider(
        [{"action": "finish", "arguments": {"output": "streamed event output"}}]
    )
    center = LLMProviderCenter(default_provider="mock")
    center.register("mock", provider, default_model="event-mini")
    events = ListEventSink()
    session = AgentSession(
        profile=AgentProfile(name="event-streaming", model="event-mini"),
        provider=center,
        tools=MockToolRuntime(),
        event_sink=events,
    )
    outcome = await AgentRunner(session).run(
        AgentRunRequest(task="stream event task", stream=True)
    )
    event_log = events.manifest()
    first_batch = EventStreamCursor(run_id=outcome.result.run_id, limit=3)
    batch = EventStreamBatch.from_log(events, first_batch).manifest()
    tail = EventStreamTail.from_log(events, first_batch, max_batches=10).manifest()
    return {
        "streaming_run": {
            "schema_version": "agent-core-event-streaming-run/v1",
            "run_id": outcome.result.run_id,
            "status": outcome.result.status,
            "output_bytes": len(outcome.result.output.encode("utf-8")),
            "stream_requested": True,
            "model_stream_event_count": _event_count(event_log, "model_stream"),
            "batch": batch,
            "tail": tail,
            "trace_summary": dict(outcome.trace_manifest.get("summary") or {}),
        },
        "event_log": event_log,
        "trace": outcome.trace_manifest,
    }


async def _manager_event_stream_acceptance() -> dict[str, Any]:
    provider = _BlockingEventProvider()
    events = ListEventSink()
    manager = AgentSessionManager()
    manager.register(
        AgentSession(
            profile=AgentProfile(name="evented-manager"),
            provider=provider,
            tools=MockToolRuntime(),
            event_sink=events,
        )
    )
    run_key = manager.start("evented-manager", "background event task")
    for _ in range(100):
        if manager.run_state(run_key).result_run_id:
            break
        await asyncio.sleep(0.01)
    before = manager.event_batch(
        run_key,
        EventStreamCursor(
            run_key=run_key,
            session_name="evented-manager",
            limit=3,
        ),
    )
    before_tail = manager.event_tail(
        run_key,
        EventStreamCursor(
            run_key=run_key,
            session_name="evented-manager",
            limit=2,
        ),
        max_batches=2,
    )
    provider.release.set()
    outcome = await manager.wait(run_key)
    after = manager.event_tail(run_key, before.next_cursor(), max_batches=10)
    return {
        "schema_version": "agent-core-manager-event-stream-acceptance/v1",
        "run_key": run_key,
        "run_id": outcome.result.run_id,
        "status": outcome.result.status,
        "before": before.manifest(),
        "before_tail": before_tail.manifest(),
        "after": after.manifest(),
        "event_log": events.manifest(),
        "manager": manager.manifest(),
    }


class _BlockingEventProvider:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        await self.release.wait()
        return LLMResponse(action={"action": "finish", "arguments": {"output": "done"}})


def _event_acceptance_issues(
    *,
    streaming_run: dict[str, Any],
    manager_stream: dict[str, Any],
    event_log: dict[str, Any],
    trace_eval: dict[str, Any],
) -> tuple[AgentCoreEventAcceptanceIssue, ...]:
    issues: list[AgentCoreEventAcceptanceIssue] = []
    if streaming_run.get("status") != "completed":
        issues.append(
            AgentCoreEventAcceptanceIssue(
                source="streaming_run",
                code="streaming_run_not_completed",
                message="Streaming run did not complete.",
                metadata={"streaming_run": dict(streaming_run)},
            )
        )
    if int(streaming_run.get("model_stream_event_count") or 0) < 1:
        issues.append(
            AgentCoreEventAcceptanceIssue(
                source="streaming_run",
                code="model_stream_event_missing",
                message="Streaming run did not emit a model_stream event.",
                metadata={"event_log": dict(event_log)},
            )
        )
    if trace_eval.get("ok") is not True:
        issues.append(
            AgentCoreEventAcceptanceIssue(
                source="trace_eval",
                code="event_trace_eval_failed",
                message="Trace evaluator did not accept the event-stream contract.",
                metadata={"trace_eval": dict(trace_eval)},
            )
        )
    before = manager_stream.get("before") if isinstance(manager_stream.get("before"), dict) else {}
    after = manager_stream.get("after") if isinstance(manager_stream.get("after"), dict) else {}
    if int(before.get("event_count") or 0) < 1:
        issues.append(
            AgentCoreEventAcceptanceIssue(
                source="manager_stream",
                code="manager_before_events_missing",
                message="Manager event batch did not expose events before completion.",
                metadata={"before": dict(before)},
            )
        )
    if before.get("terminal") is True:
        issues.append(
            AgentCoreEventAcceptanceIssue(
                source="manager_stream",
                code="manager_before_batch_terminal",
                message="Manager pre-completion event batch was terminal too early.",
                metadata={"before": dict(before)},
            )
        )
    if after.get("terminal") is not True:
        issues.append(
            AgentCoreEventAcceptanceIssue(
                source="manager_stream",
                code="manager_after_tail_not_terminal",
                message="Manager event tail did not reach a terminal event after completion.",
                metadata={"after": dict(after)},
            )
        )
    if manager_stream.get("status") != "completed":
        issues.append(
            AgentCoreEventAcceptanceIssue(
                source="manager_stream",
                code="manager_run_not_completed",
                message="Managed background run did not complete.",
                metadata={"manager_stream": dict(manager_stream)},
            )
        )
    return tuple(issues)


def _event_count(event_log: dict[str, Any], event_type: str) -> int:
    return sum(
        1
        for event in event_log.get("events") or ()
        if isinstance(event, dict) and event.get("type") == event_type
    )
