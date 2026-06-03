"""Trace replay and evaluation primitives."""

from __future__ import annotations

import json
from collections import Counter
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
    require_resume: bool = False
    require_resume_plan: bool = False
    require_resume_plan_ready: bool = False
    expected_resume_checkpoint_id: str = ""
    require_tool_execution: bool = False
    required_event_types: tuple[str, ...] = ()
    required_tool_names: tuple[str, ...] = ()
    required_tool_execution_names: tuple[str, ...] = ()
    required_tool_execution_ok_names: tuple[str, ...] = ()
    required_tool_retry_names: tuple[str, ...] = ()
    forbidden_tool_retry_names: tuple[str, ...] = ()
    min_tool_attempts: dict[str, int] = field(default_factory=dict)
    require_storage_backends: bool = False
    required_storage_backend_roles: tuple[str, ...] = ()
    required_storage_backend_kinds: tuple[str, ...] = ()
    forbidden_storage_backend_kinds: tuple[str, ...] = ()
    forbid_external_storage_backends: bool = False
    max_external_storage_backends: int | None = None
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
            "require_resume": self.require_resume,
            "require_resume_plan": self.require_resume_plan,
            "require_resume_plan_ready": self.require_resume_plan_ready,
            "expected_resume_checkpoint_id": self.expected_resume_checkpoint_id,
            "require_tool_execution": self.require_tool_execution,
            "required_event_types": list(self.required_event_types),
            "required_tool_names": list(self.required_tool_names),
            "required_tool_execution_names": list(self.required_tool_execution_names),
            "required_tool_execution_ok_names": list(self.required_tool_execution_ok_names),
            "required_tool_retry_names": list(self.required_tool_retry_names),
            "forbidden_tool_retry_names": list(self.forbidden_tool_retry_names),
            "min_tool_attempts": dict(self.min_tool_attempts),
            "require_storage_backends": self.require_storage_backends,
            "required_storage_backend_roles": list(self.required_storage_backend_roles),
            "required_storage_backend_kinds": list(self.required_storage_backend_kinds),
            "forbidden_storage_backend_kinds": list(self.forbidden_storage_backend_kinds),
            "forbid_external_storage_backends": self.forbid_external_storage_backends,
            "max_external_storage_backends": self.max_external_storage_backends,
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


@dataclass(frozen=True)
class TraceReplayDiffSpec:
    name: str = "trace-replay-diff"
    compare_run_fields: tuple[str, ...] = ("status", "iterations")
    compare_summary_keys: tuple[str, ...] = (
        "provider_call_count",
        "tool_replay_record_count",
        "policy_decision_record_count",
        "approval_record_count",
        "event_log_count",
    )
    compare_event_order: bool = True
    compare_sources: bool = True
    compare_payloads: bool = False
    ignore_event_types: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-replay-diff-spec/v1",
            "name": self.name,
            "compare_run_fields": list(self.compare_run_fields),
            "compare_summary_keys": list(self.compare_summary_keys),
            "compare_event_order": self.compare_event_order,
            "compare_sources": self.compare_sources,
            "compare_payloads": self.compare_payloads,
            "ignore_event_types": list(self.ignore_event_types),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceReplayDiffIssue:
    severity: EvalSeverity
    code: str
    message: str
    index: int | None = None
    expected: dict[str, Any] = field(default_factory=dict)
    actual: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "index": self.index,
            "expected": dict(self.expected),
            "actual": dict(self.actual),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceReplayDiffReport:
    baseline_run_id: str
    actual_run_id: str
    spec_name: str
    ok: bool
    issues: tuple[TraceReplayDiffIssue, ...] = ()
    baseline_replay: dict[str, Any] = field(default_factory=dict)
    actual_replay: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-replay-diff-report/v1",
            "baseline_run_id": self.baseline_run_id,
            "actual_run_id": self.actual_run_id,
            "spec_name": self.spec_name,
            "ok": self.ok,
            "issue_count": len(self.issues),
            "issues": [issue.manifest() for issue in self.issues],
            "baseline_replay": dict(self.baseline_replay),
            "actual_replay": dict(self.actual_replay),
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
        resume_plan = _trace_dict(trace, "resume_plan")
        if resume_plan:
            steps.append(
                TraceReplayStep(
                    sequence=len(steps) + 1,
                    source="resume_plan",
                    event_type="resume_plan_selected",
                    run_id=str(resume_plan.get("run_id") or run_id),
                    payload=dict(resume_plan),
                )
            )
        resume = _trace_dict(trace, "resume")
        if resume:
            steps.append(
                TraceReplayStep(
                    sequence=len(steps) + 1,
                    source="resume",
                    event_type="resume_checkpoint_loaded",
                    run_id=str(resume.get("run_id") or run_id),
                    turn_id=str(resume.get("turn_id") or ""),
                    payload=dict(resume),
                )
            )
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
        resume = _trace_dict(trace, "resume")
        resume_plan = _trace_dict(trace, "resume_plan")
        storage_backends = _storage_backends(trace)
        storage_backend_roles = _storage_backend_values(storage_backends, "role")
        storage_backend_kinds = _storage_backend_values(storage_backends, "kind")
        external_storage_backends = tuple(
            backend for backend in storage_backends if backend.get("core_builtin") is False
        )

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
        if spec.require_resume and not resume:
            issues.append(TraceEvalIssue("error", "resume_missing", "resume manifest is required"))
        if spec.require_resume_plan and not resume_plan:
            issues.append(TraceEvalIssue("error", "resume_plan_missing", "resume plan manifest is required"))
        if spec.require_resume_plan_ready and not bool(resume_plan.get("ready")):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "resume_plan_not_ready",
                    "resume plan is required to be ready",
                    metadata={"status": resume_plan.get("status")},
                )
            )
        if spec.expected_resume_checkpoint_id:
            resume_checkpoint_id = str(resume.get("checkpoint_id") or "")
            resume_plan_checkpoint_id = str(resume_plan.get("checkpoint_id") or "")
            if spec.expected_resume_checkpoint_id not in {
                resume_checkpoint_id,
                resume_plan_checkpoint_id,
            }:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "resume_checkpoint_mismatch",
                        "resume checkpoint id did not match expected value",
                        metadata={
                            "expected_checkpoint_id": spec.expected_resume_checkpoint_id,
                            "resume_checkpoint_id": resume_checkpoint_id,
                            "resume_plan_checkpoint_id": resume_plan_checkpoint_id,
                        },
                    )
                )

        if spec.require_storage_backends and not storage_backends:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "storage_backends_missing",
                    "storage backend trace is required",
                )
            )
        for role in spec.required_storage_backend_roles:
            if role not in storage_backend_roles:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_storage_backend_role",
                        f"required storage backend role missing: {role}",
                    )
                )
        for kind in spec.required_storage_backend_kinds:
            if kind not in storage_backend_kinds:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_storage_backend_kind",
                        f"required storage backend kind missing: {kind}",
                    )
                )
        for kind in spec.forbidden_storage_backend_kinds:
            if kind in storage_backend_kinds:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_storage_backend_kind",
                        f"forbidden storage backend kind present: {kind}",
                    )
                )
        if spec.forbid_external_storage_backends and external_storage_backends:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "external_storage_backend_forbidden",
                    "external storage backends are forbidden",
                    metadata={"external_backend_count": len(external_storage_backends)},
                )
            )
        if (
            spec.max_external_storage_backends is not None
            and len(external_storage_backends) > spec.max_external_storage_backends
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "external_storage_backend_limit_exceeded",
                    "external storage backend count exceeded limit",
                    metadata={
                        "actual": len(external_storage_backends),
                        "limit": spec.max_external_storage_backends,
                    },
                )
            )

        tool_executions = _tool_execution_summaries(trace)
        if spec.require_tool_execution and not tool_executions:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "tool_execution_missing",
                    "tool execution summary is required",
                )
            )

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
        tool_execution_by_name = _tool_execution_by_name(tool_executions)
        for tool_name in spec.required_tool_execution_names:
            if tool_name not in tool_execution_by_name:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_tool_execution",
                        f"required tool execution missing: {tool_name}",
                    )
                )
        for tool_name in spec.required_tool_execution_ok_names:
            executions = tool_execution_by_name.get(tool_name, ())
            if not any(bool(execution.get("final_ok")) for execution in executions):
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "tool_execution_not_ok",
                        f"required tool execution did not finish ok: {tool_name}",
                    )
                )
        for tool_name in spec.required_tool_retry_names:
            executions = tool_execution_by_name.get(tool_name, ())
            if not any(bool(execution.get("retried")) for execution in executions):
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_tool_retry",
                        f"required tool retry missing: {tool_name}",
                    )
                )
        for tool_name in spec.forbidden_tool_retry_names:
            executions = tool_execution_by_name.get(tool_name, ())
            if any(bool(execution.get("retried")) for execution in executions):
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_tool_retry",
                        f"forbidden tool retry present: {tool_name}",
                    )
                )
        for tool_name, min_attempts in spec.min_tool_attempts.items():
            executions = tool_execution_by_name.get(str(tool_name), ())
            actual = max((int(execution.get("attempt_count") or 0) for execution in executions), default=0)
            if actual < int(min_attempts):
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "tool_attempts_below_minimum",
                        f"tool attempts below minimum: {tool_name}",
                        metadata={
                            "tool_name": str(tool_name),
                            "expected_min_attempts": int(min_attempts),
                            "actual_attempts": actual,
                        },
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
                "tool_execution_count": len(tool_executions),
                "retried_tool_names": sorted(
                    {
                        str(execution.get("tool_name") or "")
                        for execution in tool_executions
                        if execution.get("retried") and execution.get("tool_name")
                    }
                ),
                "max_tool_attempt_count": max(
                    (int(execution.get("attempt_count") or 0) for execution in tool_executions),
                    default=0,
                ),
                "has_resume": bool(resume),
                "has_resume_plan": bool(resume_plan),
                "resume_plan_ready": bool(resume_plan.get("ready")) if resume_plan else False,
                "resume_checkpoint_id": str(resume.get("checkpoint_id") or ""),
                "resume_plan_checkpoint_id": str(resume_plan.get("checkpoint_id") or ""),
                "storage_backend_count": len(storage_backends),
                "storage_backend_roles": sorted(storage_backend_roles),
                "storage_backend_kinds": sorted(storage_backend_kinds),
                "external_storage_backend_count": len(external_storage_backends),
            },
            metadata={"spec": spec.manifest()},
        )

    def manifest(self) -> dict[str, Any]:
        return {"schema_version": "agent-core-default-trace-evaluator/v1"}


class TraceReplayComparator:
    """Compare two run trace manifests through their deterministic replay shape."""

    def __init__(self, replay_harness: TraceReplayHarness | None = None) -> None:
        self.replay_harness = replay_harness or TraceReplayHarness()

    def compare(
        self,
        baseline: dict[str, Any],
        actual: dict[str, Any],
        spec: TraceReplayDiffSpec | None = None,
    ) -> TraceReplayDiffReport:
        spec = spec or TraceReplayDiffSpec()
        baseline_replay = self.replay_harness.replay(baseline)
        actual_replay = self.replay_harness.replay(actual)
        issues: list[TraceReplayDiffIssue] = []

        self._compare_run_fields(baseline, actual, spec, issues)
        self._compare_summary(baseline, actual, spec, issues)
        self._compare_steps(baseline_replay, actual_replay, spec, issues)

        baseline_steps = _filtered_steps(baseline_replay.steps, spec.ignore_event_types)
        actual_steps = _filtered_steps(actual_replay.steps, spec.ignore_event_types)
        return TraceReplayDiffReport(
            baseline_run_id=baseline_replay.run_id,
            actual_run_id=actual_replay.run_id,
            spec_name=spec.name,
            ok=not any(issue.severity == "error" for issue in issues),
            issues=tuple(issues),
            baseline_replay=baseline_replay.manifest(),
            actual_replay=actual_replay.manifest(),
            summary={
                "baseline_step_count": len(baseline_steps),
                "actual_step_count": len(actual_steps),
                "baseline_event_types": [step.event_type for step in baseline_steps],
                "actual_event_types": [step.event_type for step in actual_steps],
            },
            metadata={"spec": spec.manifest()},
        )

    def manifest(self) -> dict[str, Any]:
        return {"schema_version": "agent-core-trace-replay-comparator/v1"}

    def _compare_run_fields(
        self,
        baseline: dict[str, Any],
        actual: dict[str, Any],
        spec: TraceReplayDiffSpec,
        issues: list[TraceReplayDiffIssue],
    ) -> None:
        expected_run = baseline.get("run") if isinstance(baseline.get("run"), dict) else {}
        actual_run = actual.get("run") if isinstance(actual.get("run"), dict) else {}
        for field_name in spec.compare_run_fields:
            expected_value = expected_run.get(field_name)
            actual_value = actual_run.get(field_name)
            if expected_value == actual_value:
                continue
            issues.append(
                TraceReplayDiffIssue(
                    "error",
                    "run_field_mismatch",
                    f"run field differs: {field_name}",
                    expected={field_name: expected_value},
                    actual={field_name: actual_value},
                )
            )

    def _compare_summary(
        self,
        baseline: dict[str, Any],
        actual: dict[str, Any],
        spec: TraceReplayDiffSpec,
        issues: list[TraceReplayDiffIssue],
    ) -> None:
        expected_summary = baseline.get("summary") if isinstance(baseline.get("summary"), dict) else {}
        actual_summary = actual.get("summary") if isinstance(actual.get("summary"), dict) else {}
        for key in spec.compare_summary_keys:
            expected_value = expected_summary.get(key)
            actual_value = actual_summary.get(key)
            if expected_value == actual_value:
                continue
            issues.append(
                TraceReplayDiffIssue(
                    "error",
                    "summary_mismatch",
                    f"summary counter differs: {key}",
                    expected={key: expected_value},
                    actual={key: actual_value},
                )
            )

    def _compare_steps(
        self,
        baseline_replay: TraceReplayResult,
        actual_replay: TraceReplayResult,
        spec: TraceReplayDiffSpec,
        issues: list[TraceReplayDiffIssue],
    ) -> None:
        expected_steps = _filtered_steps(baseline_replay.steps, spec.ignore_event_types)
        actual_steps = _filtered_steps(actual_replay.steps, spec.ignore_event_types)
        if spec.compare_event_order:
            self._compare_ordered_steps(expected_steps, actual_steps, spec, issues)
            return
        expected_counts = Counter(_step_signature(step, spec) for step in expected_steps)
        actual_counts = Counter(_step_signature(step, spec) for step in actual_steps)
        for signature, count in sorted((expected_counts - actual_counts).items()):
            issues.append(
                TraceReplayDiffIssue(
                    "error",
                    "missing_step_count",
                    f"replay step count is missing: {signature}",
                    expected={"signature": list(signature), "count": count},
                    actual={"count": actual_counts.get(signature, 0)},
                )
            )
        for signature, count in sorted((actual_counts - expected_counts).items()):
            issues.append(
                TraceReplayDiffIssue(
                    "error",
                    "unexpected_step_count",
                    f"replay step count is unexpected: {signature}",
                    expected={"count": expected_counts.get(signature, 0)},
                    actual={"signature": list(signature), "count": count},
                )
            )

    def _compare_ordered_steps(
        self,
        expected_steps: tuple[TraceReplayStep, ...],
        actual_steps: tuple[TraceReplayStep, ...],
        spec: TraceReplayDiffSpec,
        issues: list[TraceReplayDiffIssue],
    ) -> None:
        max_len = max(len(expected_steps), len(actual_steps))
        for index in range(max_len):
            expected = expected_steps[index] if index < len(expected_steps) else None
            actual = actual_steps[index] if index < len(actual_steps) else None
            if expected is None and actual is not None:
                issues.append(
                    TraceReplayDiffIssue(
                        "error",
                        "unexpected_step",
                        "actual replay has an extra step",
                        index=index,
                        actual=_step_comparison_manifest(actual, spec),
                    )
                )
                continue
            if actual is None and expected is not None:
                issues.append(
                    TraceReplayDiffIssue(
                        "error",
                        "missing_step",
                        "actual replay is missing a step",
                        index=index,
                        expected=_step_comparison_manifest(expected, spec),
                    )
                )
                continue
            if expected is None or actual is None:
                continue
            expected_manifest = _step_comparison_manifest(expected, spec)
            actual_manifest = _step_comparison_manifest(actual, spec)
            if expected_manifest == actual_manifest:
                continue
            issues.append(
                TraceReplayDiffIssue(
                    "error",
                    "step_mismatch",
                    "replay step differs",
                    index=index,
                    expected=expected_manifest,
                    actual=actual_manifest,
                )
            )


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


def _trace_dict(trace: dict[str, Any], key: str) -> dict[str, Any]:
    value = trace.get(key)
    return dict(value) if isinstance(value, dict) else {}


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


def _filtered_steps(
    steps: tuple[TraceReplayStep, ...],
    ignore_event_types: tuple[str, ...],
) -> tuple[TraceReplayStep, ...]:
    ignored = set(ignore_event_types)
    return tuple(step for step in steps if step.event_type not in ignored)


def _step_comparison_manifest(step: TraceReplayStep, spec: TraceReplayDiffSpec) -> dict[str, Any]:
    manifest: dict[str, Any] = {"event_type": step.event_type}
    if spec.compare_sources:
        manifest["source"] = step.source
    if spec.compare_payloads:
        manifest["payload"] = _stable_payload(step.payload)
    return manifest


def _step_signature(step: TraceReplayStep, spec: TraceReplayDiffSpec) -> tuple[str, ...]:
    parts = [step.event_type]
    if spec.compare_sources:
        parts.append(step.source)
    if spec.compare_payloads:
        parts.append(json.dumps(_stable_payload(step.payload), ensure_ascii=False, sort_keys=True))
    return tuple(parts)


def _stable_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _json_safe_dict(payload)


def _json_safe_dict(value: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
        safe[str(key)] = _json_safe(item)
    return safe


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return _json_safe_dict(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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


def _storage_backends(trace: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    manifest = trace.get("storage_backends")
    if not isinstance(manifest, dict):
        return ()
    backends = manifest.get("backends")
    if not isinstance(backends, (list, tuple)):
        return ()
    return tuple(dict(item) for item in backends if isinstance(item, dict))


def _storage_backend_values(backends: tuple[dict[str, Any], ...], key: str) -> set[str]:
    return {str(item.get(key) or "") for item in backends if item.get(key)}


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


def _tool_execution_summaries(trace: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(summary: dict[str, Any], *, fallback_tool_name: str = "", fallback_call_id: str = "") -> None:
        normalized = _normalize_tool_execution_summary(
            summary,
            fallback_tool_name=fallback_tool_name,
            fallback_call_id=fallback_call_id,
        )
        key = _tool_execution_key(normalized)
        if key in seen:
            return
        seen.add(key)
        summaries.append(normalized)

    for event in _event_log_events(trace):
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        execution = payload.get("execution")
        if isinstance(execution, dict):
            add(
                execution,
                fallback_tool_name=str(payload.get("tool_name") or ""),
                fallback_call_id=str(payload.get("call_id") or ""),
            )

    for event in _journal_events(trace):
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        metadata = payload.get("metadata")
        execution = metadata.get("tool_execution") if isinstance(metadata, dict) else None
        if isinstance(execution, dict):
            add(
                execution,
                fallback_tool_name=str(payload.get("tool_name") or ""),
                fallback_call_id=str(payload.get("call_id") or ""),
            )

    replay = trace.get("tool_replay")
    if isinstance(replay, dict):
        records = replay.get("records")
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict):
                    continue
                result = record.get("result")
                invocation = record.get("invocation")
                result_metadata = result.get("metadata") if isinstance(result, dict) else None
                execution = (
                    result_metadata.get("tool_execution")
                    if isinstance(result_metadata, dict)
                    else None
                )
                if isinstance(execution, dict):
                    fallback_tool_name = ""
                    fallback_call_id = ""
                    if isinstance(result, dict):
                        fallback_tool_name = str(result.get("tool_name") or "")
                        fallback_call_id = str(result.get("call_id") or "")
                    if isinstance(invocation, dict):
                        fallback_tool_name = fallback_tool_name or str(invocation.get("tool_name") or "")
                        fallback_call_id = fallback_call_id or str(invocation.get("call_id") or "")
                    add(
                        execution,
                        fallback_tool_name=fallback_tool_name,
                        fallback_call_id=fallback_call_id,
                    )
    return tuple(summaries)


def _normalize_tool_execution_summary(
    summary: dict[str, Any],
    *,
    fallback_tool_name: str = "",
    fallback_call_id: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": str(summary.get("schema_version") or "agent-core-tool-execution-summary/v1"),
        "tool_name": str(summary.get("tool_name") or fallback_tool_name),
        "call_id": str(summary.get("call_id") or fallback_call_id),
        "attempt_count": _safe_int(summary.get("attempt_count")),
        "retried": bool(summary.get("retried")),
        "final_status": str(summary.get("final_status") or ""),
        "final_ok": bool(summary.get("final_ok")),
        "attempt_statuses": [
            str(item) for item in summary.get("attempt_statuses") or ()
        ],
        "retryable_attempts": [
            _safe_int(item) for item in summary.get("retryable_attempts") or ()
        ],
    }


def _tool_execution_key(summary: dict[str, Any]) -> str:
    return "|".join(
        (
            str(summary.get("tool_name") or ""),
            str(summary.get("call_id") or ""),
            str(summary.get("attempt_count") or 0),
            str(summary.get("final_status") or ""),
        )
    )


def _tool_execution_by_name(
    summaries: tuple[dict[str, Any], ...],
) -> dict[str, tuple[dict[str, Any], ...]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        name = str(summary.get("tool_name") or "")
        if not name:
            continue
        grouped.setdefault(name, []).append(summary)
    return {key: tuple(value) for key, value in grouped.items()}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
