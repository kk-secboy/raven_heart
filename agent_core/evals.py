"""Trace replay and evaluation primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from agent_core.trace import RunTraceStorePort


EvalSeverity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class TraceReplayStep:
    sequence: int
    source: str
    event_type: str
    run_id: str = ""
    turn_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "source": self.source,
            "event_type": self.event_type,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class TraceReplayResult:
    run_id: str
    steps: tuple[TraceReplayStep, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def event_types(self) -> tuple[str, ...]:
        return tuple(step.event_type for step in self.steps)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-replay/v1",
            "run_id": self.run_id,
            "step_count": len(self.steps),
            "steps": [step.manifest() for step in self.steps],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceEvalIssue:
    severity: EvalSeverity
    code: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceEvalSpec:
    name: str = "trace-eval"
    expected_status: str = ""
    max_iterations: int | None = None
    max_provider_calls: int | None = None
    max_cost_usd: float | None = None
    require_journal_ok: bool = True
    required_event_types: tuple[str, ...] = ()
    required_tool_names: tuple[str, ...] = ()
    forbidden_event_types: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-eval-spec/v1",
            "name": self.name,
            "expected_status": self.expected_status,
            "max_iterations": self.max_iterations,
            "max_provider_calls": self.max_provider_calls,
            "max_cost_usd": self.max_cost_usd,
            "require_journal_ok": self.require_journal_ok,
            "required_event_types": list(self.required_event_types),
            "required_tool_names": list(self.required_tool_names),
            "forbidden_event_types": list(self.forbidden_event_types),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceEvalReport:
    run_id: str
    spec_name: str
    ok: bool
    issues: tuple[TraceEvalIssue, ...] = ()
    replay: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-eval-report/v1",
            "run_id": self.run_id,
            "spec_name": self.spec_name,
            "ok": self.ok,
            "issue_count": len(self.issues),
            "issues": [issue.manifest() for issue in self.issues],
            "replay": dict(self.replay),
            "summary": dict(self.summary),
            "metadata": dict(self.metadata),
        }


class TraceEvaluatorPort(Protocol):
    def evaluate(self, trace: dict[str, Any], spec: TraceEvalSpec | None = None) -> TraceEvalReport:
        """Evaluate one run trace manifest."""


class TraceReplayHarness:
    """Build a deterministic replay timeline from a run trace manifest."""

    def replay(self, trace: dict[str, Any]) -> TraceReplayResult:
        run_id = _trace_run_id(trace)
        steps: list[TraceReplayStep] = []
        for item in _journal_events(trace):
            steps.append(
                TraceReplayStep(
                    sequence=len(steps) + 1,
                    source="journal",
                    event_type=str(item.get("event_type") or ""),
                    run_id=str(item.get("run_id") or run_id),
                    turn_id=str(item.get("turn_id") or ""),
                    payload=_payload(item),
                )
            )
        for item in _event_log_events(trace):
            steps.append(
                TraceReplayStep(
                    sequence=len(steps) + 1,
                    source="event_log",
                    event_type=str(item.get("type") or ""),
                    run_id=str(item.get("run_id") or run_id),
                    turn_id=str(item.get("turn_id") or ""),
                    payload=_payload(item),
                )
            )
        return TraceReplayResult(
            run_id=run_id,
            steps=tuple(steps),
            metadata={"source_schema_version": str(trace.get("schema_version") or "")},
        )


class DefaultTraceEvaluator(TraceEvaluatorPort):
    def __init__(self, replay_harness: TraceReplayHarness | None = None) -> None:
        self.replay_harness = replay_harness or TraceReplayHarness()

    def evaluate(self, trace: dict[str, Any], spec: TraceEvalSpec | None = None) -> TraceEvalReport:
        spec = spec or TraceEvalSpec()
        replay = self.replay_harness.replay(trace)
        issues: list[TraceEvalIssue] = []
        run = trace.get("run") if isinstance(trace.get("run"), dict) else {}
        summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
        provider = trace.get("provider") if isinstance(trace.get("provider"), dict) else {}

        status = str(run.get("status") or "")
        if spec.expected_status and status != spec.expected_status:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "status_mismatch",
                    f"expected status {spec.expected_status}, got {status}",
                )
            )
        iterations = int(run.get("iterations") or 0)
        if spec.max_iterations is not None and iterations > spec.max_iterations:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "iterations_exceeded",
                    f"iterations {iterations} exceeded limit {spec.max_iterations}",
                )
            )
        provider_calls = int(summary.get("provider_call_count") or 0)
        if spec.max_provider_calls is not None and provider_calls > spec.max_provider_calls:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "provider_calls_exceeded",
                    f"provider calls {provider_calls} exceeded limit {spec.max_provider_calls}",
                )
            )
        cost = _provider_cost(provider)
        if spec.max_cost_usd is not None and cost > spec.max_cost_usd:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "cost_exceeded",
                    f"cost {cost} exceeded limit {spec.max_cost_usd}",
                )
            )
        if spec.require_journal_ok and summary.get("journal_ok") is False:
            issues.append(TraceEvalIssue("error", "journal_not_ok", "journal replay reported issues"))

        event_types = set(replay.event_types())
        for event_type in spec.required_event_types:
            if event_type not in event_types:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_event_type",
                        f"required event type missing: {event_type}",
                    )
                )
        for event_type in spec.forbidden_event_types:
            if event_type in event_types:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_event_type",
                        f"forbidden event type present: {event_type}",
                    )
                )
        tool_names = _tool_names(trace)
        for tool_name in spec.required_tool_names:
            if tool_name not in tool_names:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_tool_name",
                        f"required tool missing: {tool_name}",
                    )
                )

        return TraceEvalReport(
            run_id=replay.run_id,
            spec_name=spec.name,
            ok=not any(issue.severity == "error" for issue in issues),
            issues=tuple(issues),
            replay=replay.manifest(),
            summary={
                "status": status,
                "iterations": iterations,
                "provider_call_count": provider_calls,
                "cost_usd": cost,
                "event_types": sorted(event_types),
                "tool_names": sorted(tool_names),
            },
            metadata={"spec": spec.manifest()},
        )

    def manifest(self) -> dict[str, Any]:
        return {"schema_version": "agent-core-default-trace-evaluator/v1"}


class TraceEvalHarness:
    """Evaluate stored run traces without depending on a runtime."""

    def __init__(
        self,
        *,
        trace_store: RunTraceStorePort,
        evaluator: TraceEvaluatorPort | None = None,
    ) -> None:
        self.trace_store = trace_store
        self.evaluator = evaluator or DefaultTraceEvaluator()

    async def evaluate_run(
        self,
        run_id: str,
        spec: TraceEvalSpec | None = None,
    ) -> TraceEvalReport:
        trace = await self.trace_store.load(run_id)
        if trace is None:
            raise KeyError(f"unknown run trace: {run_id}")
        return self.evaluator.evaluate(trace, spec)

    async def evaluate_all(self, spec: TraceEvalSpec | None = None) -> tuple[TraceEvalReport, ...]:
        reports = []
        for trace in await self.trace_store.records():
            reports.append(self.evaluator.evaluate(trace, spec))
        return tuple(reports)

    def manifest(self) -> dict[str, Any]:
        evaluator_manifest = getattr(self.evaluator, "manifest", None)
        trace_store_manifest = getattr(self.trace_store, "manifest", None)
        return {
            "schema_version": "agent-core-trace-eval-harness/v1",
            "evaluator": evaluator_manifest() if callable(evaluator_manifest) else {},
            "trace_store": trace_store_manifest() if callable(trace_store_manifest) else {},
        }


def _trace_run_id(trace: dict[str, Any]) -> str:
    run = trace.get("run")
    if not isinstance(run, dict):
        return ""
    return str(run.get("run_id") or "")


def _journal_events(trace: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    replay = trace.get("journal_replay")
    if not isinstance(replay, dict):
        return ()
    events = replay.get("events")
    if not isinstance(events, list):
        return ()
    return tuple(dict(item) for item in events if isinstance(item, dict))


def _event_log_events(trace: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    log = trace.get("event_log")
    if not isinstance(log, dict):
        return ()
    events = log.get("events")
    if not isinstance(events, list):
        return ()
    return tuple(dict(item) for item in events if isinstance(item, dict))


def _payload(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload")
    return dict(payload) if isinstance(payload, dict) else dict(item)


def _provider_cost(provider: dict[str, Any]) -> float:
    usage = provider.get("usage")
    if isinstance(usage, dict):
        return float(usage.get("cost_usd") or 0.0)
    calls = provider.get("calls")
    if not isinstance(calls, list):
        return 0.0
    total = 0.0
    for call in calls:
        if not isinstance(call, dict):
            continue
        call_usage = call.get("usage")
        if isinstance(call_usage, dict):
            total += float(call_usage.get("cost_usd") or 0.0)
    return total


def _tool_names(trace: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    replay = trace.get("tool_replay")
    if isinstance(replay, dict):
        records = replay.get("records")
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict):
                    continue
                result = record.get("result")
                if isinstance(result, dict) and result.get("tool_name"):
                    names.add(str(result.get("tool_name")))
                invocation = record.get("invocation")
                if isinstance(invocation, dict) and invocation.get("tool_name"):
                    names.add(str(invocation.get("tool_name")))
    for event in _event_log_events(trace):
        payload = event.get("payload")
        if isinstance(payload, dict) and payload.get("tool_name"):
            names.add(str(payload.get("tool_name")))
    for event in _journal_events(trace):
        payload = event.get("payload")
        if isinstance(payload, dict) and payload.get("tool_name"):
            names.add(str(payload.get("tool_name")))
    return names
