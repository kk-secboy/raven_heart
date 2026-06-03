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
    max_embedding_calls: int | None = None
    max_cost_usd: float | None = None
    required_provider_names: tuple[str, ...] = ()
    required_provider_models: tuple[str, ...] = ()
    require_provider_model_capabilities: bool = False
    required_provider_model_capabilities: tuple[str, ...] = ()
    forbidden_provider_model_capabilities: tuple[str, ...] = ()
    require_provider_streaming: bool = False
    required_provider_stream_event_types: tuple[str, ...] = ()
    forbidden_provider_stream_event_types: tuple[str, ...] = ()
    max_provider_stream_errors: int | None = None
    require_provider_tool_calls: bool = False
    required_provider_tool_call_names: tuple[str, ...] = ()
    max_provider_tool_calls: int | None = None
    require_provider_route_plan: bool = False
    required_provider_route_candidate_names: tuple[str, ...] = ()
    required_provider_route_selected_names: tuple[str, ...] = ()
    forbidden_provider_route_reasons: tuple[str, ...] = ()
    required_embedding_provider_names: tuple[str, ...] = ()
    required_embedding_models: tuple[str, ...] = ()
    required_embedding_dimensions: tuple[int, ...] = ()
    require_embedding_calls: bool = False
    require_journal_ok: bool = True
    require_resume: bool = False
    require_resume_plan: bool = False
    require_resume_plan_ready: bool = False
    expected_resume_checkpoint_id: str = ""
    require_tool_execution: bool = False
    required_event_types: tuple[str, ...] = ()
    require_event_log: bool = False
    required_event_log_types: tuple[str, ...] = ()
    forbidden_event_log_types: tuple[str, ...] = ()
    require_terminal_event: bool = False
    terminal_event_types: tuple[str, ...] = ("run_finished", "run_cancelled", "run_timeout")
    require_event_sequence_monotonic: bool = False
    max_duplicate_event_sequences: int | None = None
    required_tool_names: tuple[str, ...] = ()
    required_tool_execution_names: tuple[str, ...] = ()
    required_tool_execution_ok_names: tuple[str, ...] = ()
    required_tool_retry_names: tuple[str, ...] = ()
    forbidden_tool_retry_names: tuple[str, ...] = ()
    require_tool_schema_validation: bool = False
    required_tool_schema_validation_names: tuple[str, ...] = ()
    required_tool_schema_valid_names: tuple[str, ...] = ()
    forbidden_tool_schema_invalid_names: tuple[str, ...] = ()
    max_tool_schema_invalid: int | None = None
    min_tool_attempts: dict[str, int] = field(default_factory=dict)
    require_storage_backends: bool = False
    required_storage_backend_roles: tuple[str, ...] = ()
    required_storage_backend_kinds: tuple[str, ...] = ()
    forbidden_storage_backend_kinds: tuple[str, ...] = ()
    forbid_external_storage_backends: bool = False
    max_external_storage_backends: int | None = None
    require_context_injections: bool = False
    required_context_injection_names: tuple[str, ...] = ()
    required_context_injection_sources: tuple[str, ...] = ()
    required_context_injection_targets: tuple[str, ...] = ()
    required_context_injection_statuses: tuple[str, ...] = ()
    forbidden_context_injection_statuses: tuple[str, ...] = ()
    required_included_context_injection_sources: tuple[str, ...] = ()
    required_trimmed_context_injection_sources: tuple[str, ...] = ()
    required_excluded_context_injection_sources: tuple[str, ...] = ()
    forbidden_context_injection_sources: tuple[str, ...] = ()
    forbid_trimmed_context_injections: bool = False
    max_trimmed_context_injections: int | None = None
    max_excluded_context_injections: int | None = None
    require_memory_governance: bool = False
    required_memory_governance_decisions: tuple[str, ...] = ()
    forbidden_memory_governance_decisions: tuple[str, ...] = ()
    max_denied_memory_writes: int | None = None
    max_rewritten_memory_writes: int | None = None
    max_high_risk_memory_writes: int | None = None
    require_prompt_bucket_budget: bool = False
    required_prompt_bucket_budget_roles: tuple[str, ...] = ()
    required_prompt_bucket_budget_statuses: tuple[str, ...] = ()
    forbidden_prompt_bucket_budget_statuses: tuple[str, ...] = ()
    forbid_over_budget_prompt_buckets: bool = False
    max_prompt_bucket_budget_trimmed: int | None = None
    max_prompt_bucket_budget_over_budget: int | None = None
    require_prompt_trim: bool = False
    forbid_prompt_trim: bool = False
    required_prompt_trim_roles: tuple[str, ...] = ()
    forbidden_prompt_trim_roles: tuple[str, ...] = ()
    max_prompt_trim_final_bytes: int | None = None
    max_prompt_trim_original_bytes: int | None = None
    forbidden_event_types: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-eval-spec/v1",
            "name": self.name,
            "expected_status": self.expected_status,
            "max_iterations": self.max_iterations,
            "max_provider_calls": self.max_provider_calls,
            "max_embedding_calls": self.max_embedding_calls,
            "max_cost_usd": self.max_cost_usd,
            "required_provider_names": list(self.required_provider_names),
            "required_provider_models": list(self.required_provider_models),
            "require_provider_model_capabilities": self.require_provider_model_capabilities,
            "required_provider_model_capabilities": list(
                self.required_provider_model_capabilities
            ),
            "forbidden_provider_model_capabilities": list(
                self.forbidden_provider_model_capabilities
            ),
            "require_provider_streaming": self.require_provider_streaming,
            "required_provider_stream_event_types": list(
                self.required_provider_stream_event_types
            ),
            "forbidden_provider_stream_event_types": list(
                self.forbidden_provider_stream_event_types
            ),
            "max_provider_stream_errors": self.max_provider_stream_errors,
            "require_provider_tool_calls": self.require_provider_tool_calls,
            "required_provider_tool_call_names": list(self.required_provider_tool_call_names),
            "max_provider_tool_calls": self.max_provider_tool_calls,
            "require_provider_route_plan": self.require_provider_route_plan,
            "required_provider_route_candidate_names": list(
                self.required_provider_route_candidate_names
            ),
            "required_provider_route_selected_names": list(
                self.required_provider_route_selected_names
            ),
            "forbidden_provider_route_reasons": list(self.forbidden_provider_route_reasons),
            "required_embedding_provider_names": list(self.required_embedding_provider_names),
            "required_embedding_models": list(self.required_embedding_models),
            "required_embedding_dimensions": list(self.required_embedding_dimensions),
            "require_embedding_calls": self.require_embedding_calls,
            "require_journal_ok": self.require_journal_ok,
            "require_resume": self.require_resume,
            "require_resume_plan": self.require_resume_plan,
            "require_resume_plan_ready": self.require_resume_plan_ready,
            "expected_resume_checkpoint_id": self.expected_resume_checkpoint_id,
            "require_tool_execution": self.require_tool_execution,
            "required_event_types": list(self.required_event_types),
            "require_event_log": self.require_event_log,
            "required_event_log_types": list(self.required_event_log_types),
            "forbidden_event_log_types": list(self.forbidden_event_log_types),
            "require_terminal_event": self.require_terminal_event,
            "terminal_event_types": list(self.terminal_event_types),
            "require_event_sequence_monotonic": self.require_event_sequence_monotonic,
            "max_duplicate_event_sequences": self.max_duplicate_event_sequences,
            "required_tool_names": list(self.required_tool_names),
            "required_tool_execution_names": list(self.required_tool_execution_names),
            "required_tool_execution_ok_names": list(self.required_tool_execution_ok_names),
            "required_tool_retry_names": list(self.required_tool_retry_names),
            "forbidden_tool_retry_names": list(self.forbidden_tool_retry_names),
            "require_tool_schema_validation": self.require_tool_schema_validation,
            "required_tool_schema_validation_names": list(
                self.required_tool_schema_validation_names
            ),
            "required_tool_schema_valid_names": list(self.required_tool_schema_valid_names),
            "forbidden_tool_schema_invalid_names": list(
                self.forbidden_tool_schema_invalid_names
            ),
            "max_tool_schema_invalid": self.max_tool_schema_invalid,
            "min_tool_attempts": dict(self.min_tool_attempts),
            "require_storage_backends": self.require_storage_backends,
            "required_storage_backend_roles": list(self.required_storage_backend_roles),
            "required_storage_backend_kinds": list(self.required_storage_backend_kinds),
            "forbidden_storage_backend_kinds": list(self.forbidden_storage_backend_kinds),
            "forbid_external_storage_backends": self.forbid_external_storage_backends,
            "max_external_storage_backends": self.max_external_storage_backends,
            "require_context_injections": self.require_context_injections,
            "required_context_injection_names": list(self.required_context_injection_names),
            "required_context_injection_sources": list(self.required_context_injection_sources),
            "required_context_injection_targets": list(self.required_context_injection_targets),
            "required_context_injection_statuses": list(self.required_context_injection_statuses),
            "forbidden_context_injection_statuses": list(self.forbidden_context_injection_statuses),
            "required_included_context_injection_sources": list(
                self.required_included_context_injection_sources
            ),
            "required_trimmed_context_injection_sources": list(
                self.required_trimmed_context_injection_sources
            ),
            "required_excluded_context_injection_sources": list(
                self.required_excluded_context_injection_sources
            ),
            "forbidden_context_injection_sources": list(self.forbidden_context_injection_sources),
            "forbid_trimmed_context_injections": self.forbid_trimmed_context_injections,
            "max_trimmed_context_injections": self.max_trimmed_context_injections,
            "max_excluded_context_injections": self.max_excluded_context_injections,
            "require_memory_governance": self.require_memory_governance,
            "required_memory_governance_decisions": list(self.required_memory_governance_decisions),
            "forbidden_memory_governance_decisions": list(self.forbidden_memory_governance_decisions),
            "max_denied_memory_writes": self.max_denied_memory_writes,
            "max_rewritten_memory_writes": self.max_rewritten_memory_writes,
            "max_high_risk_memory_writes": self.max_high_risk_memory_writes,
            "require_prompt_bucket_budget": self.require_prompt_bucket_budget,
            "required_prompt_bucket_budget_roles": list(self.required_prompt_bucket_budget_roles),
            "required_prompt_bucket_budget_statuses": list(
                self.required_prompt_bucket_budget_statuses
            ),
            "forbidden_prompt_bucket_budget_statuses": list(
                self.forbidden_prompt_bucket_budget_statuses
            ),
            "forbid_over_budget_prompt_buckets": self.forbid_over_budget_prompt_buckets,
            "max_prompt_bucket_budget_trimmed": self.max_prompt_bucket_budget_trimmed,
            "max_prompt_bucket_budget_over_budget": self.max_prompt_bucket_budget_over_budget,
            "require_prompt_trim": self.require_prompt_trim,
            "forbid_prompt_trim": self.forbid_prompt_trim,
            "required_prompt_trim_roles": list(self.required_prompt_trim_roles),
            "forbidden_prompt_trim_roles": list(self.forbidden_prompt_trim_roles),
            "max_prompt_trim_final_bytes": self.max_prompt_trim_final_bytes,
            "max_prompt_trim_original_bytes": self.max_prompt_trim_original_bytes,
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
        "embedding_call_count",
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
        provider = trace.get("provider") if isinstance(trace.get("provider"), dict) else {}
        for item in _provider_call_records(provider):
            steps.append(
                TraceReplayStep(
                    sequence=len(steps) + 1,
                    source="provider",
                    event_type=_provider_replay_event_type(item),
                    run_id=run_id,
                    payload=_provider_call_replay_payload(item),
                )
            )
        embedding = trace.get("embedding") if isinstance(trace.get("embedding"), dict) else {}
        for item in _embedding_call_records(embedding):
            steps.append(
                TraceReplayStep(
                    sequence=len(steps) + 1,
                    source="embedding",
                    event_type=_embedding_replay_event_type(item),
                    run_id=run_id,
                    payload=_embedding_call_replay_payload(item),
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
        embedding = trace.get("embedding") if isinstance(trace.get("embedding"), dict) else {}
        provider_call_records = _provider_call_records(provider)
        embedding_call_records = _embedding_call_records(embedding)
        provider_names = _provider_call_values(provider_call_records, "provider_name")
        provider_models = _provider_call_values(provider_call_records, "model")
        embedding_provider_names = _embedding_call_values(embedding_call_records, "provider_name")
        embedding_models = _embedding_call_values(embedding_call_records, "model")
        embedding_dimensions = _embedding_call_dimensions(embedding_call_records)
        provider_model_capability_manifests = _provider_model_capability_manifests(
            provider_call_records
        )
        provider_model_capabilities = _provider_model_capability_names(
            provider_model_capability_manifests
        )
        provider_stream_summaries = _provider_stream_summaries(provider_call_records)
        provider_stream_event_types = _provider_stream_event_types(provider_stream_summaries)
        provider_stream_error_count = _provider_stream_error_count(
            provider_call_records,
            provider_stream_summaries,
        )
        provider_tool_calls = _provider_tool_calls(provider_call_records, provider_stream_summaries)
        provider_tool_call_names = _provider_tool_call_names(provider_tool_calls)
        provider_route_plans = _provider_route_plans(provider_call_records)
        provider_route_candidates = _provider_route_candidates(provider_route_plans)
        provider_route_candidate_names = _provider_route_candidate_names(provider_route_candidates)
        provider_route_selected_names = _provider_route_selected_names(provider_route_candidates)
        provider_route_reasons = _provider_route_reasons(provider_route_candidates)
        resume = _trace_dict(trace, "resume")
        resume_plan = _trace_dict(trace, "resume_plan")
        storage_backends = _storage_backends(trace)
        storage_backend_roles = _storage_backend_values(storage_backends, "role")
        storage_backend_kinds = _storage_backend_values(storage_backends, "kind")
        external_storage_backends = tuple(
            backend for backend in storage_backends if backend.get("core_builtin") is False
        )
        context_injections = _context_injections(trace)
        context_injection_names = _context_injection_values(context_injections, "name")
        context_injection_sources = _context_injection_values(context_injections, "source")
        context_injection_targets = _context_injection_values(context_injections, "target")
        context_injection_statuses = _context_injection_values(context_injections, "status")
        included_context_injections = tuple(
            injection for injection in context_injections if injection.get("included") is True
        )
        trimmed_context_injections = tuple(
            injection for injection in context_injections if injection.get("trimmed") is True
        )
        excluded_context_injections = tuple(
            injection for injection in context_injections if injection.get("included") is False
        )
        included_context_injection_sources = _context_injection_values(
            included_context_injections,
            "source",
        )
        trimmed_context_injection_sources = _context_injection_values(
            trimmed_context_injections,
            "source",
        )
        excluded_context_injection_sources = _context_injection_values(
            excluded_context_injections,
            "source",
        )
        memory_governance_decisions = _memory_governance_decisions(trace)
        memory_governance_statuses = _memory_governance_values(
            memory_governance_decisions,
            "decision",
        )
        denied_memory_writes = tuple(
            decision for decision in memory_governance_decisions if decision.get("decision") == "deny"
        )
        rewritten_memory_writes = tuple(
            decision for decision in memory_governance_decisions if decision.get("decision") == "rewrite"
        )
        high_risk_memory_writes = tuple(
            decision for decision in memory_governance_decisions if decision.get("risk_level") == "high"
        )
        prompt_bucket_budget = _prompt_bucket_budget(trace)
        prompt_bucket_budget_decisions = _prompt_bucket_budget_decisions(prompt_bucket_budget)
        prompt_bucket_budget_roles = _prompt_bucket_budget_values(
            prompt_bucket_budget_decisions,
            "role",
        )
        prompt_bucket_budget_statuses = _prompt_bucket_budget_values(
            prompt_bucket_budget_decisions,
            "status",
        )
        trimmed_prompt_buckets = tuple(
            decision
            for decision in prompt_bucket_budget_decisions
            if decision.get("status") == "trimmed"
        )
        over_budget_prompt_buckets = tuple(
            decision
            for decision in prompt_bucket_budget_decisions
            if _prompt_bucket_budget_decision_over_budget(decision)
        )
        prompt_trim = _prompt_trim(trace)
        prompt_trim_roles = _prompt_trim_roles(prompt_trim)
        prompt_trim_final_bytes = _prompt_trim_int(prompt_trim, "final_bytes")
        prompt_trim_original_bytes = _prompt_trim_int(prompt_trim, "original_bytes")
        event_log_events = _event_log_events(trace)
        event_log_types = _event_log_types(event_log_events)
        terminal_event_types = set(spec.terminal_event_types)
        terminal_events = tuple(
            event for event in event_log_events if str(event.get("type") or "") in terminal_event_types
        )
        event_log_sequences = _event_log_sequences(event_log_events)
        duplicate_event_sequence_count = _duplicate_event_sequence_count(event_log_sequences)
        event_sequence_monotonic = _event_sequence_monotonic(event_log_sequences)

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
        embedding_calls = int(summary.get("embedding_call_count") or len(embedding_call_records))
        if spec.require_embedding_calls and not embedding_calls:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "embedding_calls_missing",
                    "embedding calls are required",
                )
            )
        if spec.max_embedding_calls is not None and embedding_calls > spec.max_embedding_calls:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "embedding_calls_exceeded",
                    f"embedding calls {embedding_calls} exceeded limit {spec.max_embedding_calls}",
                )
            )
        for provider_name in spec.required_embedding_provider_names:
            if provider_name not in embedding_provider_names:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_embedding_provider_name",
                        f"required embedding provider missing: {provider_name}",
                    )
                )
        for model in spec.required_embedding_models:
            if model not in embedding_models:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_embedding_model",
                        f"required embedding model missing: {model}",
                    )
                )
        for dimensions in spec.required_embedding_dimensions:
            if int(dimensions) not in embedding_dimensions:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_embedding_dimensions",
                        f"required embedding dimensions missing: {dimensions}",
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
        for provider_name in spec.required_provider_names:
            if provider_name not in provider_names:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_provider_name",
                        f"required provider missing: {provider_name}",
                    )
                )
        for model in spec.required_provider_models:
            if model not in provider_models:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_provider_model",
                        f"required provider model missing: {model}",
                    )
                )
        if spec.require_provider_model_capabilities and not provider_model_capability_manifests:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "provider_model_capabilities_missing",
                    "provider model capability trace is required",
                )
            )
        for capability in spec.required_provider_model_capabilities:
            if capability not in provider_model_capabilities:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_provider_model_capability",
                        f"required provider model capability missing: {capability}",
                    )
                )
        for capability in spec.forbidden_provider_model_capabilities:
            if capability in provider_model_capabilities:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_provider_model_capability",
                        f"forbidden provider model capability present: {capability}",
                    )
                )
        if spec.require_provider_streaming and not _provider_stream_call_count(provider_call_records):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "provider_streaming_missing",
                    "provider streaming call is required",
                )
            )
        for event_type in spec.required_provider_stream_event_types:
            if event_type not in provider_stream_event_types:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_provider_stream_event_type",
                        f"required provider stream event type missing: {event_type}",
                    )
                )
        for event_type in spec.forbidden_provider_stream_event_types:
            if event_type in provider_stream_event_types:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_provider_stream_event_type",
                        f"forbidden provider stream event type present: {event_type}",
                    )
                )
        if (
            spec.max_provider_stream_errors is not None
            and provider_stream_error_count > spec.max_provider_stream_errors
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "provider_stream_error_limit_exceeded",
                    "provider stream error count exceeded limit",
                    metadata={
                        "actual": provider_stream_error_count,
                        "limit": spec.max_provider_stream_errors,
                    },
                )
            )
        if spec.require_provider_tool_calls and not provider_tool_calls:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "provider_tool_calls_missing",
                    "provider-native tool calls are required",
                )
            )
        for tool_name in spec.required_provider_tool_call_names:
            if tool_name not in provider_tool_call_names:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_provider_tool_call",
                        f"required provider-native tool call missing: {tool_name}",
                    )
                )
        if (
            spec.max_provider_tool_calls is not None
            and len(provider_tool_calls) > spec.max_provider_tool_calls
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "provider_tool_call_limit_exceeded",
                    "provider-native tool call count exceeded limit",
                    metadata={
                        "actual": len(provider_tool_calls),
                        "limit": spec.max_provider_tool_calls,
                    },
                )
            )
        if spec.require_provider_route_plan and not provider_route_plans:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "provider_route_plan_missing",
                    "provider route plan trace is required",
                )
            )
        for provider_name in spec.required_provider_route_candidate_names:
            if provider_name not in provider_route_candidate_names:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_provider_route_candidate",
                        f"required provider route candidate missing: {provider_name}",
                    )
                )
        for provider_name in spec.required_provider_route_selected_names:
            if provider_name not in provider_route_selected_names:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_provider_route_selected",
                        f"required selected provider route missing: {provider_name}",
                    )
                )
        for reason in spec.forbidden_provider_route_reasons:
            if reason in provider_route_reasons:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_provider_route_reason",
                        f"forbidden provider route reason present: {reason}",
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

        if spec.require_context_injections and not context_injections:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "context_injections_missing",
                    "context injection trace is required",
                )
            )
        for name in spec.required_context_injection_names:
            if name not in context_injection_names:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_context_injection_name",
                        f"required context injection name missing: {name}",
                    )
                )
        for source in spec.required_context_injection_sources:
            if source not in context_injection_sources:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_context_injection_source",
                        f"required context injection source missing: {source}",
                    )
                )
        for target in spec.required_context_injection_targets:
            if target not in context_injection_targets:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_context_injection_target",
                        f"required context injection target missing: {target}",
                    )
                )
        for status_value in spec.required_context_injection_statuses:
            if status_value not in context_injection_statuses:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_context_injection_status",
                        f"required context injection status missing: {status_value}",
                    )
                )
        for status_value in spec.forbidden_context_injection_statuses:
            if status_value in context_injection_statuses:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_context_injection_status",
                        f"forbidden context injection status present: {status_value}",
                    )
                )
        for source in spec.required_included_context_injection_sources:
            if source not in included_context_injection_sources:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_included_context_injection_source",
                        f"required included context injection source missing: {source}",
                    )
                )
        for source in spec.required_trimmed_context_injection_sources:
            if source not in trimmed_context_injection_sources:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_trimmed_context_injection_source",
                        f"required trimmed context injection source missing: {source}",
                    )
                )
        for source in spec.required_excluded_context_injection_sources:
            if source not in excluded_context_injection_sources:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_excluded_context_injection_source",
                        f"required excluded context injection source missing: {source}",
                    )
                )
        for source in spec.forbidden_context_injection_sources:
            if source in context_injection_sources:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_context_injection_source",
                        f"forbidden context injection source present: {source}",
                    )
                )
        if spec.forbid_trimmed_context_injections and trimmed_context_injections:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "context_injection_trimmed_forbidden",
                    "trimmed context injections are forbidden",
                    metadata={"trimmed_count": len(trimmed_context_injections)},
                )
            )
        if (
            spec.max_trimmed_context_injections is not None
            and len(trimmed_context_injections) > spec.max_trimmed_context_injections
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "context_injection_trimmed_limit_exceeded",
                    "trimmed context injection count exceeded limit",
                    metadata={
                        "actual": len(trimmed_context_injections),
                        "limit": spec.max_trimmed_context_injections,
                    },
                )
            )
        if (
            spec.max_excluded_context_injections is not None
            and len(excluded_context_injections) > spec.max_excluded_context_injections
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "context_injection_excluded_limit_exceeded",
                    "excluded context injection count exceeded limit",
                    metadata={
                        "actual": len(excluded_context_injections),
                        "limit": spec.max_excluded_context_injections,
                    },
                )
            )

        if spec.require_memory_governance and not memory_governance_decisions:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "memory_governance_missing",
                    "memory governance trace is required",
                )
            )
        for decision in spec.required_memory_governance_decisions:
            if decision not in memory_governance_statuses:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_memory_governance_decision",
                        f"required memory governance decision missing: {decision}",
                    )
                )
        for decision in spec.forbidden_memory_governance_decisions:
            if decision in memory_governance_statuses:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_memory_governance_decision",
                        f"forbidden memory governance decision present: {decision}",
                    )
                )
        if (
            spec.max_denied_memory_writes is not None
            and len(denied_memory_writes) > spec.max_denied_memory_writes
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "denied_memory_write_limit_exceeded",
                    "denied memory write count exceeded limit",
                    metadata={"actual": len(denied_memory_writes), "limit": spec.max_denied_memory_writes},
                )
            )
        if (
            spec.max_rewritten_memory_writes is not None
            and len(rewritten_memory_writes) > spec.max_rewritten_memory_writes
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "rewritten_memory_write_limit_exceeded",
                    "rewritten memory write count exceeded limit",
                    metadata={
                        "actual": len(rewritten_memory_writes),
                        "limit": spec.max_rewritten_memory_writes,
                    },
                )
            )
        if (
            spec.max_high_risk_memory_writes is not None
            and len(high_risk_memory_writes) > spec.max_high_risk_memory_writes
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "high_risk_memory_write_limit_exceeded",
                    "high-risk memory write count exceeded limit",
                    metadata={
                        "actual": len(high_risk_memory_writes),
                        "limit": spec.max_high_risk_memory_writes,
                    },
                )
            )

        if spec.require_prompt_bucket_budget and not prompt_bucket_budget:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_bucket_budget_missing",
                    "prompt bucket budget trace is required",
                )
            )
        for role in spec.required_prompt_bucket_budget_roles:
            if role not in prompt_bucket_budget_roles:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_prompt_bucket_budget_role",
                        f"required prompt bucket budget role missing: {role}",
                    )
                )
        for status in spec.required_prompt_bucket_budget_statuses:
            if status not in prompt_bucket_budget_statuses:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_prompt_bucket_budget_status",
                        f"required prompt bucket budget status missing: {status}",
                    )
                )
        for status in spec.forbidden_prompt_bucket_budget_statuses:
            if status in prompt_bucket_budget_statuses:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_prompt_bucket_budget_status",
                        f"forbidden prompt bucket budget status present: {status}",
                    )
                )
        if spec.forbid_over_budget_prompt_buckets and over_budget_prompt_buckets:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_bucket_budget_over_budget_forbidden",
                    "over-budget prompt buckets are forbidden",
                    metadata={"over_budget_count": len(over_budget_prompt_buckets)},
                )
            )
        if (
            spec.max_prompt_bucket_budget_trimmed is not None
            and len(trimmed_prompt_buckets) > spec.max_prompt_bucket_budget_trimmed
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_bucket_budget_trimmed_limit_exceeded",
                    "trimmed prompt bucket count exceeded limit",
                    metadata={
                        "actual": len(trimmed_prompt_buckets),
                        "limit": spec.max_prompt_bucket_budget_trimmed,
                    },
                )
            )
        if (
            spec.max_prompt_bucket_budget_over_budget is not None
            and len(over_budget_prompt_buckets) > spec.max_prompt_bucket_budget_over_budget
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_bucket_budget_over_budget_limit_exceeded",
                    "over-budget prompt bucket count exceeded limit",
                    metadata={
                        "actual": len(over_budget_prompt_buckets),
                        "limit": spec.max_prompt_bucket_budget_over_budget,
                    },
                )
            )
        if spec.require_prompt_trim and not prompt_trim:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_trim_missing",
                    "prompt trim trace is required",
                )
            )
        if spec.forbid_prompt_trim and prompt_trim:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_trim_forbidden",
                    "prompt trim trace is forbidden",
                )
            )
        for role in spec.required_prompt_trim_roles:
            if role not in prompt_trim_roles:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_prompt_trim_role",
                        f"required prompt trim role missing: {role}",
                    )
                )
        for role in spec.forbidden_prompt_trim_roles:
            if role in prompt_trim_roles:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_prompt_trim_role",
                        f"forbidden prompt trim role present: {role}",
                    )
                )
        if (
            spec.max_prompt_trim_final_bytes is not None
            and prompt_trim_final_bytes > spec.max_prompt_trim_final_bytes
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_trim_final_bytes_exceeded",
                    "prompt trim final bytes exceeded limit",
                    metadata={
                        "actual": prompt_trim_final_bytes,
                        "limit": spec.max_prompt_trim_final_bytes,
                    },
                )
            )
        if (
            spec.max_prompt_trim_original_bytes is not None
            and prompt_trim_original_bytes > spec.max_prompt_trim_original_bytes
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_trim_original_bytes_exceeded",
                    "prompt trim original bytes exceeded limit",
                    metadata={
                        "actual": prompt_trim_original_bytes,
                        "limit": spec.max_prompt_trim_original_bytes,
                    },
                )
            )

        tool_executions = _tool_execution_summaries(trace)
        tool_schema_validations = _tool_schema_validations(tool_executions)
        invalid_tool_schema_validations = tuple(
            validation for validation in tool_schema_validations if validation.get("ok") is False
        )
        tool_schema_validated_names = _tool_schema_validation_values(
            tool_schema_validations,
            "tool_name",
        )
        tool_schema_valid_names = _tool_schema_validation_values(
            tuple(validation for validation in tool_schema_validations if validation.get("ok") is True),
            "tool_name",
        )
        tool_schema_invalid_names = _tool_schema_validation_values(
            invalid_tool_schema_validations,
            "tool_name",
        )
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
        if spec.require_event_log and not event_log_events:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "event_log_missing",
                    "event log trace is required",
                )
            )
        for event_type in spec.required_event_log_types:
            if event_type not in event_log_types:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_event_log_type",
                        f"required event log type missing: {event_type}",
                    )
                )
        for event_type in spec.forbidden_event_log_types:
            if event_type in event_log_types:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_event_log_type",
                        f"forbidden event log type present: {event_type}",
                    )
                )
        if spec.require_terminal_event and not terminal_events:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "terminal_event_missing",
                    "terminal event log entry is required",
                    metadata={"terminal_event_types": list(spec.terminal_event_types)},
                )
            )
        if spec.require_event_sequence_monotonic and not event_sequence_monotonic:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "event_sequence_not_monotonic",
                    "event log sequence values are not strictly increasing",
                    metadata={"sequences": list(event_log_sequences)},
                )
            )
        if (
            spec.max_duplicate_event_sequences is not None
            and duplicate_event_sequence_count > spec.max_duplicate_event_sequences
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "duplicate_event_sequence_limit_exceeded",
                    "duplicate event log sequence count exceeded limit",
                    metadata={
                        "actual": duplicate_event_sequence_count,
                        "limit": spec.max_duplicate_event_sequences,
                    },
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
        if spec.require_tool_schema_validation and not tool_schema_validations:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "tool_schema_validation_missing",
                    "tool schema validation trace is required",
                )
            )
        for tool_name in spec.required_tool_schema_validation_names:
            if tool_name not in tool_schema_validated_names:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_tool_schema_validation",
                        f"required tool schema validation missing: {tool_name}",
                    )
                )
        for tool_name in spec.required_tool_schema_valid_names:
            if tool_name not in tool_schema_valid_names:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "tool_schema_not_valid",
                        f"required tool schema validation did not pass: {tool_name}",
                    )
                )
        for tool_name in spec.forbidden_tool_schema_invalid_names:
            if tool_name in tool_schema_invalid_names:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_tool_schema_invalid",
                        f"forbidden tool schema invalid result present: {tool_name}",
                    )
                )
        if (
            spec.max_tool_schema_invalid is not None
            and len(invalid_tool_schema_validations) > spec.max_tool_schema_invalid
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "tool_schema_invalid_limit_exceeded",
                    "tool schema invalid count exceeded limit",
                    metadata={
                        "actual": len(invalid_tool_schema_validations),
                        "limit": spec.max_tool_schema_invalid,
                    },
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
                "embedding_call_count": embedding_calls,
                "embedding_provider_names": sorted(embedding_provider_names),
                "embedding_models": sorted(embedding_models),
                "embedding_dimensions": sorted(embedding_dimensions),
                "provider_names": sorted(provider_names),
                "provider_models": sorted(provider_models),
                "provider_model_capability_count": len(provider_model_capability_manifests),
                "provider_model_capabilities": sorted(provider_model_capabilities),
                "provider_stream_call_count": _provider_stream_call_count(provider_call_records),
                "provider_stream_summary_count": len(provider_stream_summaries),
                "provider_stream_event_types": sorted(provider_stream_event_types),
                "provider_stream_error_count": provider_stream_error_count,
                "provider_tool_call_count": len(provider_tool_calls),
                "provider_tool_call_names": sorted(provider_tool_call_names),
                "provider_route_plan_count": len(provider_route_plans),
                "provider_route_candidate_names": sorted(provider_route_candidate_names),
                "provider_route_selected_names": sorted(provider_route_selected_names),
                "provider_route_reasons": sorted(provider_route_reasons),
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
                "tool_schema_validation_count": len(tool_schema_validations),
                "tool_schema_invalid_count": len(invalid_tool_schema_validations),
                "tool_schema_validated_names": sorted(tool_schema_validated_names),
                "tool_schema_valid_names": sorted(tool_schema_valid_names),
                "tool_schema_invalid_names": sorted(tool_schema_invalid_names),
                "has_resume": bool(resume),
                "has_resume_plan": bool(resume_plan),
                "resume_plan_ready": bool(resume_plan.get("ready")) if resume_plan else False,
                "resume_checkpoint_id": str(resume.get("checkpoint_id") or ""),
                "resume_plan_checkpoint_id": str(resume_plan.get("checkpoint_id") or ""),
                "storage_backend_count": len(storage_backends),
                "storage_backend_roles": sorted(storage_backend_roles),
                "storage_backend_kinds": sorted(storage_backend_kinds),
                "external_storage_backend_count": len(external_storage_backends),
                "context_injection_count": len(context_injections),
                "context_injection_names": sorted(context_injection_names),
                "context_injection_sources": sorted(context_injection_sources),
                "context_injection_targets": sorted(context_injection_targets),
                "context_injection_statuses": sorted(context_injection_statuses),
                "included_context_injection_sources": sorted(
                    included_context_injection_sources
                ),
                "trimmed_context_injection_sources": sorted(trimmed_context_injection_sources),
                "excluded_context_injection_sources": sorted(excluded_context_injection_sources),
                "trimmed_context_injection_count": len(trimmed_context_injections),
                "excluded_context_injection_count": len(excluded_context_injections),
                "memory_governance_decision_count": len(memory_governance_decisions),
                "memory_governance_decisions": sorted(memory_governance_statuses),
                "denied_memory_write_count": len(denied_memory_writes),
                "rewritten_memory_write_count": len(rewritten_memory_writes),
                "high_risk_memory_write_count": len(high_risk_memory_writes),
                "has_prompt_bucket_budget": bool(prompt_bucket_budget),
                "prompt_bucket_budget_role_count": len(prompt_bucket_budget_roles),
                "prompt_bucket_budget_roles": sorted(prompt_bucket_budget_roles),
                "prompt_bucket_budget_statuses": sorted(prompt_bucket_budget_statuses),
                "prompt_bucket_budget_trimmed_count": len(trimmed_prompt_buckets),
                "prompt_bucket_budget_over_budget_count": len(over_budget_prompt_buckets),
                "has_prompt_trim": bool(prompt_trim),
                "prompt_trim_roles": sorted(prompt_trim_roles),
                "prompt_trim_original_bytes": prompt_trim_original_bytes,
                "prompt_trim_final_bytes": prompt_trim_final_bytes,
                "event_log_types": sorted(event_log_types),
                "event_log_sequence_monotonic": event_sequence_monotonic,
                "duplicate_event_sequence_count": duplicate_event_sequence_count,
                "terminal_event_types": sorted(
                    {str(event.get("type") or "") for event in terminal_events}
                ),
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


def _event_log_types(events: tuple[dict[str, Any], ...]) -> set[str]:
    return {str(event.get("type") or "") for event in events if event.get("type")}


def _event_log_sequences(events: tuple[dict[str, Any], ...]) -> tuple[int, ...]:
    return tuple(
        _safe_int(event.get("sequence"))
        for event in events
        if event.get("sequence") is not None
    )


def _duplicate_event_sequence_count(sequences: tuple[int, ...]) -> int:
    counts = Counter(sequences)
    return sum(count - 1 for count in counts.values() if count > 1)


def _event_sequence_monotonic(sequences: tuple[int, ...]) -> bool:
    if not sequences:
        return True
    return all(current > previous for previous, current in zip(sequences, sequences[1:]))


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


def _provider_call_records(provider: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    calls = provider.get("calls")
    if not isinstance(calls, (list, tuple)):
        return ()
    return tuple(dict(item) for item in calls if isinstance(item, dict))


def _provider_replay_event_type(call: dict[str, Any]) -> str:
    status = str(call.get("status") or "completed")
    if call.get("streamed") is True:
        return "provider_stream_failed" if status == "failed" else "provider_stream_completed"
    return "provider_call_failed" if status == "failed" else "provider_call_completed"


def _provider_call_replay_payload(call: dict[str, Any]) -> dict[str, Any]:
    metadata = call.get("metadata") if isinstance(call.get("metadata"), dict) else {}
    stream_summary = metadata.get("stream_summary") if isinstance(metadata, dict) else None
    request = metadata.get("request") if isinstance(metadata, dict) else None
    response = metadata.get("response") if isinstance(metadata, dict) else None
    request_metadata = request.get("metadata") if isinstance(request, dict) else {}
    payload = {
        "provider_name": str(call.get("provider_name") or ""),
        "model": str(call.get("model") or ""),
        "attempt": _safe_int(call.get("attempt") or 0),
        "status": str(call.get("status") or "completed"),
        "streamed": bool(call.get("streamed")),
        "retryable": bool(call.get("retryable")),
        "usage": dict(call.get("usage")) if isinstance(call.get("usage"), dict) else {},
        "requested_model": str(request.get("model") or "") if isinstance(request, dict) else "",
        "model_capabilities": dict(request_metadata.get("model_capabilities"))
        if isinstance(request_metadata, dict)
        and isinstance(request_metadata.get("model_capabilities"), dict)
        else {},
    }
    if isinstance(stream_summary, dict):
        payload["stream_summary"] = dict(stream_summary)
    if isinstance(response, dict):
        payload["response"] = dict(response)
    if call.get("error"):
        payload["error"] = str(call.get("error") or "")
    return payload


def _embedding_call_records(embedding: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    calls = embedding.get("calls")
    if not isinstance(calls, (list, tuple)):
        return ()
    return tuple(dict(item) for item in calls if isinstance(item, dict))


def _embedding_replay_event_type(call: dict[str, Any]) -> str:
    status = str(call.get("status") or "completed")
    return "embedding_call_failed" if status == "failed" else "embedding_call_completed"


def _embedding_call_replay_payload(call: dict[str, Any]) -> dict[str, Any]:
    metadata = call.get("metadata") if isinstance(call.get("metadata"), dict) else {}
    request = metadata.get("request") if isinstance(metadata.get("request"), dict) else {}
    return {
        "provider_name": str(call.get("provider_name") or ""),
        "model": str(call.get("model") or ""),
        "status": str(call.get("status") or "completed"),
        "input_count": _safe_int(call.get("input_count") or 0),
        "dimensions": _safe_int(call.get("dimensions") or 0),
        "requested_model": str(request.get("model") or ""),
        "request_dimensions": _safe_int(request.get("dimensions") or 0),
        "request_input_count": _safe_int(request.get("input_count") or 0),
    }


def _embedding_call_values(calls: tuple[dict[str, Any], ...], key: str) -> set[str]:
    return {str(item.get(key) or "") for item in calls if item.get(key)}


def _embedding_call_dimensions(calls: tuple[dict[str, Any], ...]) -> set[int]:
    return {_safe_int(item.get("dimensions") or 0) for item in calls if item.get("dimensions")}


def _provider_call_values(calls: tuple[dict[str, Any], ...], key: str) -> set[str]:
    return {str(item.get(key) or "") for item in calls if item.get(key)}


def _provider_model_capability_manifests(
    calls: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    manifests: list[dict[str, Any]] = []
    for call in calls:
        metadata = call.get("metadata")
        if not isinstance(metadata, dict):
            continue
        capability = metadata.get("model_capabilities")
        if isinstance(capability, dict):
            manifests.append(dict(capability))
            continue
        request = metadata.get("request")
        if not isinstance(request, dict):
            continue
        request_metadata = request.get("metadata")
        if not isinstance(request_metadata, dict):
            continue
        capability = request_metadata.get("model_capabilities")
        if isinstance(capability, dict):
            manifests.append(dict(capability))
    return tuple(manifests)


def _provider_model_capability_names(manifests: tuple[dict[str, Any], ...]) -> set[str]:
    names: set[str] = set()
    for manifest in manifests:
        raw = manifest.get("capabilities")
        if isinstance(raw, (list, tuple)):
            names.update(str(item) for item in raw if item)
        for field_name, capability_name in (
            ("supports_streaming", "streaming"),
            ("supports_tool_calls", "tool_calls"),
            ("supports_structured_output", "structured_output"),
            ("supports_json_mode", "json_mode"),
        ):
            if manifest.get(field_name) is True:
                names.add(capability_name)
        modalities = manifest.get("modalities")
        if isinstance(modalities, (list, tuple)):
            names.update(f"modality:{item}" for item in modalities if item)
    return names


def _provider_stream_call_count(calls: tuple[dict[str, Any], ...]) -> int:
    return sum(1 for call in calls if call.get("streamed") is True)


def _provider_stream_summaries(calls: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    summaries = []
    for call in calls:
        metadata = call.get("metadata")
        if not isinstance(metadata, dict):
            continue
        summary = metadata.get("stream_summary")
        if isinstance(summary, dict):
            summaries.append(
                {
                    **dict(summary),
                    "provider_name": str(call.get("provider_name") or ""),
                    "model": str(call.get("model") or ""),
                    "status": str(call.get("status") or ""),
                }
            )
    return tuple(summaries)


def _provider_stream_event_types(summaries: tuple[dict[str, Any], ...]) -> set[str]:
    event_types: set[str] = set()
    for summary in summaries:
        raw = summary.get("event_types")
        if isinstance(raw, (list, tuple)):
            event_types.update(str(item) for item in raw if item)
    return event_types


def _provider_stream_error_count(
    calls: tuple[dict[str, Any], ...],
    summaries: tuple[dict[str, Any], ...],
) -> int:
    failed_streamed_calls = sum(
        1
        for call in calls
        if call.get("streamed") is True and str(call.get("status") or "") == "failed"
    )
    summary_errors = sum(
        1
        for summary in summaries
        if summary.get("error") or "error" in set(summary.get("event_types") or ())
    )
    return failed_streamed_calls + summary_errors


def _provider_tool_calls(
    calls: tuple[dict[str, Any], ...],
    summaries: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    tool_calls: list[dict[str, Any]] = []
    for call in calls:
        metadata = call.get("metadata")
        if not isinstance(metadata, dict):
            continue
        response = metadata.get("response")
        if isinstance(response, dict):
            tool_calls.extend(_tool_call_manifests(response))
    for summary in summaries:
        tool_calls.extend(_tool_call_manifests(summary))
    return tuple(tool_calls)


def _tool_call_manifests(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = manifest.get("tool_calls")
    if isinstance(raw, (list, tuple)):
        return tuple(dict(item) for item in raw if isinstance(item, dict))
    if int(manifest.get("tool_call_count") or 0) > 0:
        return tuple({"tool_name": ""} for _ in range(int(manifest.get("tool_call_count") or 0)))
    return ()


def _provider_tool_call_names(tool_calls: tuple[dict[str, Any], ...]) -> set[str]:
    return {str(item.get("tool_name") or "") for item in tool_calls if item.get("tool_name")}


def _provider_route_plans(calls: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    plans: list[dict[str, Any]] = []
    for call in calls:
        metadata = call.get("metadata")
        if not isinstance(metadata, dict):
            continue
        plan = metadata.get("route_plan")
        if isinstance(plan, dict):
            plans.append(dict(plan))
    return tuple(plans)


def _provider_route_candidates(plans: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    candidates: list[dict[str, Any]] = []
    for plan in plans:
        raw = plan.get("candidates")
        if isinstance(raw, (list, tuple)):
            candidates.extend(dict(item) for item in raw if isinstance(item, dict))
    return tuple(candidates)


def _provider_route_candidate_names(candidates: tuple[dict[str, Any], ...]) -> set[str]:
    return {str(item.get("provider_name") or "") for item in candidates if item.get("provider_name")}


def _provider_route_selected_names(candidates: tuple[dict[str, Any], ...]) -> set[str]:
    return {
        str(item.get("provider_name") or "")
        for item in candidates
        if item.get("selected") is True and item.get("provider_name")
    }


def _provider_route_reasons(candidates: tuple[dict[str, Any], ...]) -> set[str]:
    return {str(item.get("reason") or "") for item in candidates if item.get("reason")}


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


def _context_injections(trace: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    manifest = trace.get("context_injections")
    if isinstance(manifest, dict):
        raw = manifest.get("injections")
        if isinstance(raw, (list, tuple)):
            return tuple(dict(item) for item in raw if isinstance(item, dict))
    prompt = trace.get("prompt")
    prompt_metadata = prompt.get("metadata") if isinstance(prompt, dict) else {}
    raw = prompt_metadata.get("context_injections") if isinstance(prompt_metadata, dict) else ()
    if isinstance(raw, (list, tuple)):
        return tuple(dict(item) for item in raw if isinstance(item, dict))
    return ()


def _context_injection_values(injections: tuple[dict[str, Any], ...], key: str) -> set[str]:
    return {str(item.get(key) or "") for item in injections if item.get(key)}


def _memory_governance_decisions(trace: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    manifest = trace.get("memory_governance")
    if isinstance(manifest, dict):
        raw = manifest.get("decisions")
        if isinstance(raw, (list, tuple)):
            return tuple(dict(item) for item in raw if isinstance(item, dict))
    session = trace.get("session")
    memory = session.get("memory") if isinstance(session, dict) else {}
    governance = memory.get("governance") if isinstance(memory, dict) else {}
    raw = governance.get("decisions") if isinstance(governance, dict) else ()
    if isinstance(raw, (list, tuple)):
        return tuple(dict(item) for item in raw if isinstance(item, dict))
    return ()


def _memory_governance_values(decisions: tuple[dict[str, Any], ...], key: str) -> set[str]:
    return {str(item.get(key) or "") for item in decisions if item.get(key)}


def _prompt_bucket_budget(trace: dict[str, Any]) -> dict[str, Any]:
    manifest = trace.get("prompt_bucket_budget")
    if isinstance(manifest, dict) and manifest:
        return dict(manifest)
    prompt = trace.get("prompt")
    prompt_metadata = prompt.get("metadata") if isinstance(prompt, dict) else {}
    budget = prompt_metadata.get("bucket_budget") if isinstance(prompt_metadata, dict) else {}
    return dict(budget) if isinstance(budget, dict) else {}


def _prompt_bucket_budget_decisions(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = manifest.get("decisions") if isinstance(manifest, dict) else ()
    if isinstance(raw, (list, tuple)):
        return tuple(dict(item) for item in raw if isinstance(item, dict))
    return ()


def _prompt_bucket_budget_values(decisions: tuple[dict[str, Any], ...], key: str) -> set[str]:
    return {str(item.get(key) or "") for item in decisions if item.get(key)}


def _prompt_bucket_budget_decision_over_budget(decision: dict[str, Any]) -> bool:
    max_bytes = decision.get("max_bytes")
    if max_bytes is None:
        return False
    try:
        return int(decision.get("final_bytes") or 0) > int(max_bytes)
    except (TypeError, ValueError):
        return False


def _prompt_trim(trace: dict[str, Any]) -> dict[str, Any]:
    prompt = trace.get("prompt")
    prompt_metadata = prompt.get("metadata") if isinstance(prompt, dict) else {}
    trim = prompt_metadata.get("trim") if isinstance(prompt_metadata, dict) else {}
    if isinstance(trim, dict) and trim:
        return dict(trim)
    metadata = trace.get("metadata")
    trim = metadata.get("prompt_trim") if isinstance(metadata, dict) else {}
    return dict(trim) if isinstance(trim, dict) else {}


def _prompt_trim_roles(trim: dict[str, Any]) -> set[str]:
    raw = trim.get("trimmed_roles") if isinstance(trim, dict) else ()
    roles: set[str] = set()
    if not isinstance(raw, (list, tuple)):
        return roles
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role:
            roles.add(str(role))
    return roles


def _prompt_trim_int(trim: dict[str, Any], key: str) -> int:
    try:
        return int(trim.get(key) or 0)
    except (TypeError, ValueError):
        return 0


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
    normalized = {
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
    schema_validation = summary.get("schema_validation")
    if isinstance(schema_validation, dict):
        normalized["schema_validation"] = dict(schema_validation)
    return normalized


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


def _tool_schema_validations(
    summaries: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    validations = []
    for summary in summaries:
        validation = summary.get("schema_validation")
        if not isinstance(validation, dict):
            continue
        validations.append(
            {
                **dict(validation),
                "tool_name": str(summary.get("tool_name") or ""),
                "call_id": str(summary.get("call_id") or ""),
            }
        )
    return tuple(validations)


def _tool_schema_validation_values(
    validations: tuple[dict[str, Any], ...],
    key: str,
) -> set[str]:
    return {str(item.get(key) or "") for item in validations if item.get(key)}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
