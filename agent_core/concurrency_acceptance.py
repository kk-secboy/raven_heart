"""SDK-level concurrent session isolation acceptance checks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from agent_core.config import AgentProfile
from agent_core.events import ListEventSink
from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.providers import LLMProviderCenter, LLMRequest, LLMResponse
from agent_core.runner import (
    AgentManagerConcurrencyPolicy,
    AgentRunRequest,
    AgentSession,
    AgentSessionManager,
)
from agent_core.testing import MockToolRuntime
from agent_core.trace import InMemoryRunTraceStore


@dataclass(frozen=True)
class AgentCoreConcurrencyAcceptanceIssue:
    """One blocking concurrent session isolation issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-concurrency-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreConcurrencyAcceptanceReport:
    """Prompt-safe report for concurrent SDK session isolation."""

    status: str
    runs: dict[str, Any] = field(default_factory=dict)
    active_snapshot: dict[str, Any] = field(default_factory=dict)
    manager: dict[str, Any] = field(default_factory=dict)
    providers: dict[str, Any] = field(default_factory=dict)
    tools: dict[str, Any] = field(default_factory=dict)
    event_streams: dict[str, Any] = field(default_factory=dict)
    traces: dict[str, Any] = field(default_factory=dict)
    trace_evals: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreConcurrencyAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-concurrency-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "runs": dict(self.runs),
            "active_snapshot": dict(self.active_snapshot),
            "manager": dict(self.manager),
            "providers": dict(self.providers),
            "tools": dict(self.tools),
            "event_streams": dict(self.event_streams),
            "traces": dict(self.traces),
            "trace_evals": dict(self.trace_evals),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreConcurrencyAcceptanceHarness:
    """Run deterministic concurrent session isolation checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreConcurrencyAcceptanceReport:
        barrier = _ConcurrentProviderBarrier(expected=2)
        alpha_provider = _ConcurrentProvider("alpha", barrier)
        beta_provider = _ConcurrentProvider("beta", barrier)
        alpha_tools = MockToolRuntime({"lookup": "alpha-tool-result"})
        beta_tools = MockToolRuntime({"lookup": "beta-tool-result"})
        alpha_events = ListEventSink()
        beta_events = ListEventSink()
        trace_store = InMemoryRunTraceStore()
        manager = AgentSessionManager(
            concurrency_policy=AgentManagerConcurrencyPolicy(
                max_active_runs=2,
                max_active_runs_per_session=1,
            )
        )
        manager.register(
            _session(
                name="alpha",
                provider=alpha_provider,
                tools=alpha_tools,
                events=alpha_events,
                trace_store=trace_store,
            )
        )
        manager.register(
            _session(
                name="beta",
                provider=beta_provider,
                tools=beta_tools,
                events=beta_events,
                trace_store=trace_store,
            )
        )
        alpha_key = manager.start(
            "alpha",
            AgentRunRequest(task="alpha task", metadata={"tenant": "alpha"}),
        )
        beta_key = manager.start(
            "beta",
            AgentRunRequest(task="beta task", metadata={"tenant": "beta"}),
        )
        await asyncio.wait_for(barrier.all_arrived.wait(), timeout=2)
        active_snapshot = manager.schedule_snapshot().manifest()
        barrier.release.set()
        alpha_outcome, beta_outcome = await asyncio.gather(
            manager.wait(alpha_key),
            manager.wait(beta_key),
        )
        traces = {
            "schema_version": "agent-core-concurrency-trace-store-acceptance/v1",
            "store": trace_store.manifest(),
            "record_count": len(await trace_store.records()),
            "run_ids": [alpha_outcome.result.run_id, beta_outcome.result.run_id],
        }
        trace_evals = {
            "alpha": DefaultTraceEvaluator()
            .evaluate(alpha_outcome.trace_manifest, _trace_spec("alpha-provider"))
            .manifest(),
            "beta": DefaultTraceEvaluator()
            .evaluate(beta_outcome.trace_manifest, _trace_spec("beta-provider"))
            .manifest(),
        }
        runs = {
            "schema_version": "agent-core-concurrency-runs-acceptance/v1",
            "run_keys": [alpha_key, beta_key],
            "run_ids": [alpha_outcome.result.run_id, beta_outcome.result.run_id],
            "statuses": {
                "alpha": alpha_outcome.result.status,
                "beta": beta_outcome.result.status,
            },
            "outputs": {
                "alpha": alpha_outcome.result.output,
                "beta": beta_outcome.result.output,
            },
            "manager_statuses": {
                "alpha": manager.run_state(alpha_key).status,
                "beta": manager.run_state(beta_key).status,
            },
        }
        providers = {
            "schema_version": "agent-core-concurrency-provider-acceptance/v1",
            "alpha_request_count": len(alpha_provider.requests),
            "beta_request_count": len(beta_provider.requests),
            "alpha_metadata": [dict(request.metadata) for request in alpha_provider.requests],
            "beta_metadata": [dict(request.metadata) for request in beta_provider.requests],
        }
        tools = {
            "schema_version": "agent-core-concurrency-tool-acceptance/v1",
            "alpha_invocation_count": len(alpha_tools.invocations),
            "beta_invocation_count": len(beta_tools.invocations),
            "alpha_tool_names": [item.tool_name for item in alpha_tools.invocations],
            "beta_tool_names": [item.tool_name for item in beta_tools.invocations],
            "alpha_arguments": [dict(item.arguments) for item in alpha_tools.invocations],
            "beta_arguments": [dict(item.arguments) for item in beta_tools.invocations],
        }
        event_streams = {
            "schema_version": "agent-core-concurrency-event-stream-acceptance/v1",
            "alpha": _event_summary(manager.event_tail(alpha_key).manifest()),
            "beta": _event_summary(manager.event_tail(beta_key).manifest()),
        }
        manager_manifest = manager.manifest()
        issues = _concurrency_issues(
            runs=runs,
            active_snapshot=active_snapshot,
            manager=manager_manifest,
            providers=providers,
            tools=tools,
            event_streams=event_streams,
            traces=traces,
            trace_evals=trace_evals,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreConcurrencyAcceptanceReport(
            status=status,
            runs=runs,
            active_snapshot=_active_snapshot_summary(active_snapshot),
            manager=_manager_summary(manager_manifest),
            providers=providers,
            tools=tools,
            event_streams=event_streams,
            traces=traces,
            trace_evals=trace_evals,
            issues=issues,
            metadata={
                "scenario": "agent_core_concurrency_acceptance",
                **dict(self.metadata),
            },
        )


async def run_agent_core_concurrency_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreConcurrencyAcceptanceReport:
    """Run the default concurrent session isolation acceptance checks."""

    return await AgentCoreConcurrencyAcceptanceHarness(metadata=dict(metadata or {})).run()


class _ConcurrentProviderBarrier:
    def __init__(self, *, expected: int) -> None:
        self.expected = expected
        self.arrived = 0
        self.lock = asyncio.Lock()
        self.all_arrived = asyncio.Event()
        self.release = asyncio.Event()

    async def arrive(self) -> None:
        async with self.lock:
            self.arrived += 1
            if self.arrived >= self.expected:
                self.all_arrived.set()


class _ConcurrentProvider:
    def __init__(self, label: str, barrier: _ConcurrentProviderBarrier) -> None:
        self.label = label
        self.barrier = barrier
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            await self.barrier.arrive()
            await self.barrier.release.wait()
            return LLMResponse(
                action={
                    "action": "call_tool",
                    "arguments": {
                        "tool_name": "lookup",
                        "arguments": {"owner": self.label},
                    },
                }
            )
        return LLMResponse(
            action={
                "action": "finish",
                "arguments": {"output": f"{self.label}:done"},
            }
        )


def _session(
    *,
    name: str,
    provider: _ConcurrentProvider,
    tools: MockToolRuntime,
    events: ListEventSink,
    trace_store: InMemoryRunTraceStore,
) -> AgentSession:
    center = LLMProviderCenter(default_provider=f"{name}-provider")
    center.register(f"{name}-provider", provider, default_model=f"{name}-mini")
    return AgentSession(
        profile=AgentProfile(name=name, model=f"{name}-mini"),
        provider=center,
        tools=tools,
        event_sink=events,
        trace_store=trace_store,
    )


def _trace_spec(provider_name: str) -> TraceEvalSpec:
    return TraceEvalSpec(
        name=f"concurrency-{provider_name}",
        expected_status="completed",
        max_provider_calls=2,
        required_provider_names=(provider_name,),
        required_tool_names=("lookup",),
        required_tool_execution_names=("lookup",),
        required_tool_execution_ok_names=("lookup",),
        max_failure_count=0,
    )


def _event_summary(tail: dict[str, Any]) -> dict[str, Any]:
    events = [
        event
        for batch in tail.get("batches") or ()
        if isinstance(batch, dict)
        for event in batch.get("events") or ()
        if isinstance(event, dict)
    ]
    return {
        "schema_version": "agent-core-concurrency-event-summary/v1",
        "terminal": bool(tail.get("terminal")),
        "event_count": len(events),
        "event_types": [str(event.get("type") or "") for event in events],
        "run_keys": sorted(
            {
                str((event.get("payload") or {}).get("run_key") or "")
                for event in events
                if isinstance(event.get("payload"), dict)
            }
        ),
        "session_names": sorted(
            {
                str((event.get("payload") or {}).get("session_name") or "")
                for event in events
                if isinstance(event.get("payload"), dict)
            }
        ),
    }


def _active_snapshot_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "agent-core-concurrency-active-snapshot-summary/v1",
        "active_by_session": dict(snapshot.get("active_by_session") or {}),
        "active_run_count": int(snapshot.get("active_run_count") or 0),
        "queued_by_session": dict(snapshot.get("queued_by_session") or {}),
    }


def _manager_summary(manager: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "agent-core-concurrency-manager-summary/v1",
        "session_names": sorted(str(name) for name in (manager.get("sessions") or {})),
        "run_count": len(manager.get("runs") or ()),
        "active_run_count": int(manager.get("active_run_count") or 0),
        "queued_run_count": int(manager.get("queued_run_count") or 0),
        "run_statuses": [
            str(run.get("status") or "")
            for run in manager.get("runs") or ()
            if isinstance(run, dict)
        ],
    }


def _concurrency_issues(
    *,
    runs: dict[str, Any],
    active_snapshot: dict[str, Any],
    manager: dict[str, Any],
    providers: dict[str, Any],
    tools: dict[str, Any],
    event_streams: dict[str, Any],
    traces: dict[str, Any],
    trace_evals: dict[str, Any],
) -> tuple[AgentCoreConcurrencyAcceptanceIssue, ...]:
    issues: list[AgentCoreConcurrencyAcceptanceIssue] = []
    if runs.get("statuses") != {"alpha": "completed", "beta": "completed"}:
        issues.append(
            AgentCoreConcurrencyAcceptanceIssue(
                source="runs",
                code="concurrent_runs_not_completed",
                message="Concurrent runs did not both complete.",
                metadata=dict(runs),
            )
        )
    if len(set(runs.get("run_keys") or ())) != 2 or len(set(runs.get("run_ids") or ())) != 2:
        issues.append(
            AgentCoreConcurrencyAcceptanceIssue(
                source="runs",
                code="concurrent_run_identity_collision",
                message="Concurrent runs did not produce distinct run keys and run IDs.",
                metadata=dict(runs),
            )
        )
    if int(active_snapshot.get("active_run_count") or 0) != 2:
        issues.append(
            AgentCoreConcurrencyAcceptanceIssue(
                source="active_snapshot",
                code="concurrent_active_overlap_missing",
                message="Manager did not observe two active runs before release.",
                metadata=_active_snapshot_summary(active_snapshot),
            )
        )
    if int(manager.get("active_run_count") or 0) != 0 or int(manager.get("queued_run_count") or 0) != 0:
        issues.append(
            AgentCoreConcurrencyAcceptanceIssue(
                source="manager",
                code="concurrent_manager_not_drained",
                message="Manager did not drain active and queued runs after completion.",
                metadata=_manager_summary(manager),
            )
        )
    if providers.get("alpha_request_count") != 2 or providers.get("beta_request_count") != 2:
        issues.append(
            AgentCoreConcurrencyAcceptanceIssue(
                source="providers",
                code="concurrent_provider_requests_unexpected",
                message="Concurrent providers did not receive isolated two-turn requests.",
                metadata=dict(providers),
            )
        )
    if tools.get("alpha_arguments") != [{"owner": "alpha"}] or tools.get("beta_arguments") != [
        {"owner": "beta"}
    ]:
        issues.append(
            AgentCoreConcurrencyAcceptanceIssue(
                source="tools",
                code="concurrent_tool_arguments_crossed",
                message="Concurrent tool invocations did not preserve per-session arguments.",
                metadata=dict(tools),
            )
        )
    for name in ("alpha", "beta"):
        summary = event_streams.get(name) if isinstance(event_streams.get(name), dict) else {}
        if summary.get("terminal") is not True or name not in set(summary.get("session_names") or ()):
            issues.append(
                AgentCoreConcurrencyAcceptanceIssue(
                    source="event_streams",
                    code=f"concurrent_{name}_event_stream_not_isolated",
                    message=f"{name} event stream did not stay isolated and terminal.",
                    metadata=dict(summary),
                )
            )
    if int(traces.get("record_count") or 0) != 2:
        issues.append(
            AgentCoreConcurrencyAcceptanceIssue(
                source="traces",
                code="concurrent_trace_store_count_unexpected",
                message="Shared trace store did not persist exactly two run traces.",
                metadata=dict(traces),
            )
        )
    for name, trace_eval in trace_evals.items():
        if isinstance(trace_eval, dict) and trace_eval.get("ok") is True:
            continue
        issues.append(
            AgentCoreConcurrencyAcceptanceIssue(
                source=f"{name}_trace_eval",
                code="concurrent_trace_eval_failed",
                message=f"{name} trace eval failed.",
                metadata={"trace_eval": dict(trace_eval) if isinstance(trace_eval, dict) else {}},
            )
        )
    return tuple(issues)
