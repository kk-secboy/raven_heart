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
    require_provider_request_shape_plan: bool = False
    forbid_provider_request_shape_plan: bool = False
    require_provider_request_shape_adjusted: bool = False
    forbid_provider_request_shape_adjusted: bool = False
    required_provider_request_shape_decisions: tuple[str, ...] = ()
    required_provider_request_shape_provider_names: tuple[str, ...] = ()
    max_provider_request_shape_final_output_tokens: int | None = None
    require_lifecycle_hooks: bool = False
    required_lifecycle_event_types: tuple[str, ...] = ()
    forbidden_lifecycle_event_types: tuple[str, ...] = ()
    required_lifecycle_hook_statuses: tuple[str, ...] = ()
    max_lifecycle_hook_failures: int | None = None
    required_embedding_provider_names: tuple[str, ...] = ()
    required_embedding_models: tuple[str, ...] = ()
    required_embedding_dimensions: tuple[int, ...] = ()
    require_embedding_calls: bool = False
    require_journal_ok: bool = True
    require_resume: bool = False
    require_resume_plan: bool = False
    require_resume_plan_ready: bool = False
    expected_resume_checkpoint_id: str = ""
    require_preflight: bool = False
    require_preflight_passed: bool = False
    require_preflight_blocked: bool = False
    required_preflight_issue_codes: tuple[str, ...] = ()
    forbidden_preflight_issue_codes: tuple[str, ...] = ()
    max_preflight_blocking_issues: int | None = None
    require_handoff: bool = False
    required_handoff_statuses: tuple[str, ...] = ()
    required_handoff_selected_sessions: tuple[str, ...] = ()
    required_handoff_source_sessions: tuple[str, ...] = ()
    max_handoff_denied: int | None = None
    max_handoff_not_found: int | None = None
    require_tool_execution: bool = False
    required_event_types: tuple[str, ...] = ()
    require_event_log: bool = False
    required_event_log_types: tuple[str, ...] = ()
    forbidden_event_log_types: tuple[str, ...] = ()
    require_terminal_event: bool = False
    terminal_event_types: tuple[str, ...] = ("run_finished", "run_cancelled", "run_timeout")
    require_event_sequence_monotonic: bool = False
    max_duplicate_event_sequences: int | None = None
    require_approvals: bool = False
    required_approval_statuses: tuple[str, ...] = ()
    forbidden_approval_statuses: tuple[str, ...] = ()
    required_approval_subjects: tuple[str, ...] = ()
    required_approval_subject_kinds: tuple[str, ...] = ()
    max_pending_approvals: int | None = None
    max_rejected_approvals: int | None = None
    require_approved_approval_subjects: tuple[str, ...] = ()
    require_artifacts: bool = False
    required_artifact_kinds: tuple[str, ...] = ()
    required_artifact_tool_names: tuple[str, ...] = ()
    required_artifact_content_types: tuple[str, ...] = ()
    max_artifact_count: int | None = None
    max_artifact_total_bytes: int | None = None
    max_artifact_size_bytes: int | None = None
    require_structured_output: bool = False
    require_structured_output_ok: bool = False
    required_structured_output_schema_names: tuple[str, ...] = ()
    forbidden_structured_output_errors: tuple[str, ...] = ()
    max_structured_output_repairs: int | None = None
    max_structured_output_failures: int | None = None
    require_planner_trace: bool = False
    required_plan_ids: tuple[str, ...] = ()
    required_plan_step_ids: tuple[str, ...] = ()
    required_plan_step_statuses: tuple[str, ...] = ()
    required_plan_execution_statuses: tuple[str, ...] = ()
    max_failed_plan_steps: int | None = None
    max_blocked_plan_reports: int | None = None
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
    require_tool_center: bool = False
    required_tool_center_selected_mounts: tuple[str, ...] = ()
    required_tool_center_selected_tools: tuple[str, ...] = ()
    required_tool_center_requested_tools: tuple[str, ...] = ()
    require_tool_center_ready_routes: bool = False
    max_tool_center_failed_calls: int | None = None
    require_agent_tools: bool = False
    required_agent_tool_names: tuple[str, ...] = ()
    required_agent_tool_sessions: tuple[str, ...] = ()
    required_agent_tool_statuses: tuple[str, ...] = ()
    max_agent_tool_failures: int | None = None
    require_mcp_center: bool = False
    required_mcp_server_names: tuple[str, ...] = ()
    required_mcp_refreshed_servers: tuple[str, ...] = ()
    forbidden_mcp_server_statuses: tuple[str, ...] = ()
    max_mcp_failed_servers: int | None = None
    max_mcp_partial_inventory_refreshes: int | None = None
    require_skill_center: bool = False
    required_loaded_skills: tuple[str, ...] = ()
    required_skill_resource_view_ids: tuple[str, ...] = ()
    required_skill_resource_view_skills: tuple[str, ...] = ()
    max_skill_resource_views: int | None = None
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
    require_context_material_selection: bool = False
    required_selected_context_material_names: tuple[str, ...] = ()
    required_context_material_statuses: tuple[str, ...] = ()
    forbidden_context_material_statuses: tuple[str, ...] = ()
    required_context_material_targets: tuple[str, ...] = ()
    max_dropped_context_materials: int | None = None
    max_selected_context_material_bytes: int | None = None
    require_memory_governance: bool = False
    required_memory_governance_decisions: tuple[str, ...] = ()
    forbidden_memory_governance_decisions: tuple[str, ...] = ()
    max_denied_memory_writes: int | None = None
    max_rewritten_memory_writes: int | None = None
    max_high_risk_memory_writes: int | None = None
    require_prompt_budget: bool = False
    forbid_prompt_budget: bool = False
    require_prompt_budget_provider_limited: bool = False
    forbid_prompt_budget_provider_limited: bool = False
    required_prompt_budget_sources: tuple[str, ...] = ()
    required_prompt_budget_provider_names: tuple[str, ...] = ()
    max_prompt_budget_target_bytes: int | None = None
    max_prompt_budget_provider_input_bytes: int | None = None
    require_prompt_bucket_budget: bool = False
    required_prompt_bucket_budget_roles: tuple[str, ...] = ()
    required_prompt_bucket_budget_statuses: tuple[str, ...] = ()
    forbidden_prompt_bucket_budget_statuses: tuple[str, ...] = ()
    forbid_over_budget_prompt_buckets: bool = False
    max_prompt_bucket_budget_trimmed: int | None = None
    max_prompt_bucket_budget_over_budget: int | None = None
    require_prompt_semantic_trim: bool = False
    forbid_prompt_semantic_trim: bool = False
    required_prompt_semantic_trim_roles: tuple[str, ...] = ()
    required_prompt_semantic_trim_statuses: tuple[str, ...] = ()
    forbidden_prompt_semantic_trim_statuses: tuple[str, ...] = ()
    max_prompt_semantic_trimmed: int | None = None
    max_prompt_semantic_dropped_units: int | None = None
    max_prompt_semantic_trim_final_bytes: int | None = None
    max_prompt_semantic_trim_original_bytes: int | None = None
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
            "require_provider_request_shape_plan": self.require_provider_request_shape_plan,
            "forbid_provider_request_shape_plan": self.forbid_provider_request_shape_plan,
            "require_provider_request_shape_adjusted": (
                self.require_provider_request_shape_adjusted
            ),
            "forbid_provider_request_shape_adjusted": (
                self.forbid_provider_request_shape_adjusted
            ),
            "required_provider_request_shape_decisions": list(
                self.required_provider_request_shape_decisions
            ),
            "required_provider_request_shape_provider_names": list(
                self.required_provider_request_shape_provider_names
            ),
            "max_provider_request_shape_final_output_tokens": (
                self.max_provider_request_shape_final_output_tokens
            ),
            "require_lifecycle_hooks": self.require_lifecycle_hooks,
            "required_lifecycle_event_types": list(self.required_lifecycle_event_types),
            "forbidden_lifecycle_event_types": list(self.forbidden_lifecycle_event_types),
            "required_lifecycle_hook_statuses": list(self.required_lifecycle_hook_statuses),
            "max_lifecycle_hook_failures": self.max_lifecycle_hook_failures,
            "required_embedding_provider_names": list(self.required_embedding_provider_names),
            "required_embedding_models": list(self.required_embedding_models),
            "required_embedding_dimensions": list(self.required_embedding_dimensions),
            "require_embedding_calls": self.require_embedding_calls,
            "require_journal_ok": self.require_journal_ok,
            "require_resume": self.require_resume,
            "require_resume_plan": self.require_resume_plan,
            "require_resume_plan_ready": self.require_resume_plan_ready,
            "expected_resume_checkpoint_id": self.expected_resume_checkpoint_id,
            "require_preflight": self.require_preflight,
            "require_preflight_passed": self.require_preflight_passed,
            "require_preflight_blocked": self.require_preflight_blocked,
            "required_preflight_issue_codes": list(self.required_preflight_issue_codes),
            "forbidden_preflight_issue_codes": list(self.forbidden_preflight_issue_codes),
            "max_preflight_blocking_issues": self.max_preflight_blocking_issues,
            "require_handoff": self.require_handoff,
            "required_handoff_statuses": list(self.required_handoff_statuses),
            "required_handoff_selected_sessions": list(
                self.required_handoff_selected_sessions
            ),
            "required_handoff_source_sessions": list(self.required_handoff_source_sessions),
            "max_handoff_denied": self.max_handoff_denied,
            "max_handoff_not_found": self.max_handoff_not_found,
            "require_tool_execution": self.require_tool_execution,
            "required_event_types": list(self.required_event_types),
            "require_event_log": self.require_event_log,
            "required_event_log_types": list(self.required_event_log_types),
            "forbidden_event_log_types": list(self.forbidden_event_log_types),
            "require_terminal_event": self.require_terminal_event,
            "terminal_event_types": list(self.terminal_event_types),
            "require_event_sequence_monotonic": self.require_event_sequence_monotonic,
            "max_duplicate_event_sequences": self.max_duplicate_event_sequences,
            "require_approvals": self.require_approvals,
            "required_approval_statuses": list(self.required_approval_statuses),
            "forbidden_approval_statuses": list(self.forbidden_approval_statuses),
            "required_approval_subjects": list(self.required_approval_subjects),
            "required_approval_subject_kinds": list(self.required_approval_subject_kinds),
            "max_pending_approvals": self.max_pending_approvals,
            "max_rejected_approvals": self.max_rejected_approvals,
            "require_approved_approval_subjects": list(
                self.require_approved_approval_subjects
            ),
            "require_artifacts": self.require_artifacts,
            "required_artifact_kinds": list(self.required_artifact_kinds),
            "required_artifact_tool_names": list(self.required_artifact_tool_names),
            "required_artifact_content_types": list(self.required_artifact_content_types),
            "max_artifact_count": self.max_artifact_count,
            "max_artifact_total_bytes": self.max_artifact_total_bytes,
            "max_artifact_size_bytes": self.max_artifact_size_bytes,
            "require_structured_output": self.require_structured_output,
            "require_structured_output_ok": self.require_structured_output_ok,
            "required_structured_output_schema_names": list(
                self.required_structured_output_schema_names
            ),
            "forbidden_structured_output_errors": list(self.forbidden_structured_output_errors),
            "max_structured_output_repairs": self.max_structured_output_repairs,
            "max_structured_output_failures": self.max_structured_output_failures,
            "require_planner_trace": self.require_planner_trace,
            "required_plan_ids": list(self.required_plan_ids),
            "required_plan_step_ids": list(self.required_plan_step_ids),
            "required_plan_step_statuses": list(self.required_plan_step_statuses),
            "required_plan_execution_statuses": list(self.required_plan_execution_statuses),
            "max_failed_plan_steps": self.max_failed_plan_steps,
            "max_blocked_plan_reports": self.max_blocked_plan_reports,
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
            "require_tool_center": self.require_tool_center,
            "required_tool_center_selected_mounts": list(
                self.required_tool_center_selected_mounts
            ),
            "required_tool_center_selected_tools": list(
                self.required_tool_center_selected_tools
            ),
            "required_tool_center_requested_tools": list(
                self.required_tool_center_requested_tools
            ),
            "require_tool_center_ready_routes": self.require_tool_center_ready_routes,
            "max_tool_center_failed_calls": self.max_tool_center_failed_calls,
            "require_agent_tools": self.require_agent_tools,
            "required_agent_tool_names": list(self.required_agent_tool_names),
            "required_agent_tool_sessions": list(self.required_agent_tool_sessions),
            "required_agent_tool_statuses": list(self.required_agent_tool_statuses),
            "max_agent_tool_failures": self.max_agent_tool_failures,
            "require_mcp_center": self.require_mcp_center,
            "required_mcp_server_names": list(self.required_mcp_server_names),
            "required_mcp_refreshed_servers": list(self.required_mcp_refreshed_servers),
            "forbidden_mcp_server_statuses": list(self.forbidden_mcp_server_statuses),
            "max_mcp_failed_servers": self.max_mcp_failed_servers,
            "max_mcp_partial_inventory_refreshes": self.max_mcp_partial_inventory_refreshes,
            "require_skill_center": self.require_skill_center,
            "required_loaded_skills": list(self.required_loaded_skills),
            "required_skill_resource_view_ids": list(self.required_skill_resource_view_ids),
            "required_skill_resource_view_skills": list(
                self.required_skill_resource_view_skills
            ),
            "max_skill_resource_views": self.max_skill_resource_views,
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
            "require_context_material_selection": self.require_context_material_selection,
            "required_selected_context_material_names": list(
                self.required_selected_context_material_names
            ),
            "required_context_material_statuses": list(self.required_context_material_statuses),
            "forbidden_context_material_statuses": list(
                self.forbidden_context_material_statuses
            ),
            "required_context_material_targets": list(self.required_context_material_targets),
            "max_dropped_context_materials": self.max_dropped_context_materials,
            "max_selected_context_material_bytes": self.max_selected_context_material_bytes,
            "require_memory_governance": self.require_memory_governance,
            "required_memory_governance_decisions": list(self.required_memory_governance_decisions),
            "forbidden_memory_governance_decisions": list(self.forbidden_memory_governance_decisions),
            "max_denied_memory_writes": self.max_denied_memory_writes,
            "max_rewritten_memory_writes": self.max_rewritten_memory_writes,
            "max_high_risk_memory_writes": self.max_high_risk_memory_writes,
            "require_prompt_budget": self.require_prompt_budget,
            "forbid_prompt_budget": self.forbid_prompt_budget,
            "require_prompt_budget_provider_limited": self.require_prompt_budget_provider_limited,
            "forbid_prompt_budget_provider_limited": self.forbid_prompt_budget_provider_limited,
            "required_prompt_budget_sources": list(self.required_prompt_budget_sources),
            "required_prompt_budget_provider_names": list(
                self.required_prompt_budget_provider_names
            ),
            "max_prompt_budget_target_bytes": self.max_prompt_budget_target_bytes,
            "max_prompt_budget_provider_input_bytes": self.max_prompt_budget_provider_input_bytes,
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
            "require_prompt_semantic_trim": self.require_prompt_semantic_trim,
            "forbid_prompt_semantic_trim": self.forbid_prompt_semantic_trim,
            "required_prompt_semantic_trim_roles": list(
                self.required_prompt_semantic_trim_roles
            ),
            "required_prompt_semantic_trim_statuses": list(
                self.required_prompt_semantic_trim_statuses
            ),
            "forbidden_prompt_semantic_trim_statuses": list(
                self.forbidden_prompt_semantic_trim_statuses
            ),
            "max_prompt_semantic_trimmed": self.max_prompt_semantic_trimmed,
            "max_prompt_semantic_dropped_units": self.max_prompt_semantic_dropped_units,
            "max_prompt_semantic_trim_final_bytes": self.max_prompt_semantic_trim_final_bytes,
            "max_prompt_semantic_trim_original_bytes": self.max_prompt_semantic_trim_original_bytes,
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
        preflight = _preflight_trace(trace)
        if preflight:
            steps.append(
                TraceReplayStep(
                    sequence=len(steps) + 1,
                    source="preflight",
                    event_type=f"preflight_{preflight.get('status') or 'unknown'}",
                    run_id=run_id,
                    payload=dict(preflight),
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
        for item in _planner_replay_steps(trace):
            steps.append(
                TraceReplayStep(
                    sequence=len(steps) + 1,
                    source="planner_trace",
                    event_type=str(item.get("event_type") or ""),
                    run_id=run_id,
                    payload=dict(item.get("payload") or {}),
                )
            )
        for item in _handoff_replay_steps(trace):
            steps.append(
                TraceReplayStep(
                    sequence=len(steps) + 1,
                    source="handoff_trace",
                    event_type=str(item.get("event_type") or ""),
                    run_id=run_id,
                    payload=dict(item.get("payload") or {}),
                )
            )
        for item in _agent_tool_replay_steps(trace):
            steps.append(
                TraceReplayStep(
                    sequence=len(steps) + 1,
                    source="agent_tool_trace",
                    event_type=str(item.get("event_type") or ""),
                    run_id=run_id,
                    payload=dict(item.get("payload") or {}),
                )
            )
        for item in _prompt_shaping_replay_steps(trace):
            steps.append(
                TraceReplayStep(
                    sequence=len(steps) + 1,
                    source=str(item.get("source") or "prompt"),
                    event_type=str(item.get("event_type") or ""),
                    run_id=run_id,
                    payload=dict(item.get("payload") or {}),
                )
            )
        for item in _capability_center_replay_steps(trace):
            steps.append(
                TraceReplayStep(
                    sequence=len(steps) + 1,
                    source=str(item.get("source") or "capability_center"),
                    event_type=str(item.get("event_type") or ""),
                    run_id=run_id,
                    payload=dict(item.get("payload") or {}),
                )
            )
        for item in _approval_replay_steps(trace):
            steps.append(
                TraceReplayStep(
                    sequence=len(steps) + 1,
                    source="approval_trace",
                    event_type=str(item.get("event_type") or ""),
                    run_id=str(item.get("run_id") or run_id),
                    turn_id=str(item.get("turn_id") or ""),
                    payload=dict(item.get("payload") or {}),
                )
            )
        for item in _artifact_replay_steps(trace):
            steps.append(
                TraceReplayStep(
                    sequence=len(steps) + 1,
                    source="artifact_trace",
                    event_type="artifact_stored",
                    run_id=run_id,
                    payload=dict(item),
                )
            )
        for item in _structured_output_replay_steps(trace):
            steps.append(
                TraceReplayStep(
                    sequence=len(steps) + 1,
                    source="structured_output",
                    event_type=str(item.get("event_type") or ""),
                    run_id=str(item.get("run_id") or run_id),
                    turn_id=str(item.get("turn_id") or ""),
                    payload=dict(item.get("payload") or {}),
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
        for item in _lifecycle_hook_records(_lifecycle_hooks(trace)):
            steps.append(
                TraceReplayStep(
                    sequence=len(steps) + 1,
                    source="lifecycle_hooks",
                    event_type=_lifecycle_hook_replay_event_type(item),
                    run_id=run_id,
                    payload=_lifecycle_hook_replay_payload(item),
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
        provider_request_shape_plans = _provider_request_shape_plans(provider_call_records)
        provider_request_shape_provider_names = _provider_request_shape_values(
            provider_request_shape_plans,
            "provider_name",
        )
        provider_request_shape_decisions = _provider_request_shape_decisions(
            provider_request_shape_plans
        )
        adjusted_provider_request_shapes = tuple(
            plan for plan in provider_request_shape_plans if plan.get("adjusted") is True
        )
        provider_request_shape_final_output_tokens = tuple(
            _safe_int(plan.get("final_max_output_tokens"))
            for plan in provider_request_shape_plans
            if plan.get("final_max_output_tokens") is not None
        )
        provider_request_shape_max_final_output_tokens = max(
            provider_request_shape_final_output_tokens,
            default=0,
        )
        lifecycle_hooks = _lifecycle_hooks(trace)
        lifecycle_hook_records = _lifecycle_hook_records(lifecycle_hooks)
        lifecycle_event_types = _lifecycle_event_types(lifecycle_hook_records)
        lifecycle_hook_statuses = _lifecycle_hook_statuses(lifecycle_hook_records)
        lifecycle_hook_failure_count = _lifecycle_hook_failure_count(lifecycle_hook_records)
        resume = _trace_dict(trace, "resume")
        resume_plan = _trace_dict(trace, "resume_plan")
        preflight = _preflight_trace(trace)
        preflight_issue_codes = set(_manifest_values(preflight, "codes"))
        preflight_blocking_codes = set(_manifest_values(preflight, "blocking_codes"))
        preflight_status = str(preflight.get("status") or "")
        preflight_blocking_count = _safe_int(preflight.get("blocking_count"))
        handoff_trace = _handoff_trace(trace)
        handoff_records = _handoff_records(handoff_trace)
        handoff_statuses = _handoff_values(handoff_records, "status")
        handoff_selected_sessions = _handoff_values(handoff_records, "selected_session")
        handoff_source_sessions = _handoff_values(handoff_records, "source_session")
        denied_handoffs = tuple(
            record for record in handoff_records if record.get("status") == "denied"
        )
        not_found_handoffs = tuple(
            record for record in handoff_records if record.get("status") == "not_found"
        )
        tool_center = _tool_center_trace(trace)
        tool_center_calls = _tool_center_calls(tool_center)
        tool_center_route_plans = _tool_center_route_plans(tool_center_calls)
        tool_center_selected_mounts = _tool_center_route_values(
            tool_center_route_plans,
            "selected_mount",
        )
        tool_center_selected_tools = _tool_center_route_values(
            tool_center_route_plans,
            "selected_tool_name",
        )
        tool_center_requested_tools = _tool_center_call_values(
            tool_center_calls,
            "requested_tool_name",
        )
        failed_tool_center_calls = tuple(
            call for call in tool_center_calls if str(call.get("status") or "") != "completed"
        )
        not_ready_tool_center_routes = tuple(
            plan for plan in tool_center_route_plans if plan.get("ready") is not True
        )
        agent_tool_trace = _agent_tool_trace(trace)
        agent_tool_records = _agent_tool_records(agent_tool_trace)
        agent_tool_names = _agent_tool_values(agent_tool_records, "tool_name")
        agent_tool_sessions = _agent_tool_values(agent_tool_records, "session_name")
        agent_tool_statuses = _agent_tool_values(agent_tool_records, "status")
        failed_agent_tools = tuple(
            record
            for record in agent_tool_records
            if str(record.get("status") or "") != "completed"
        )
        mcp_center = _mcp_center_trace(trace)
        mcp_servers = _mcp_server_records(mcp_center)
        mcp_server_names = _mcp_server_values(mcp_servers, "name")
        mcp_server_statuses = _mcp_server_statuses(mcp_servers, mcp_center)
        mcp_refreshed_servers = set(_manifest_values(mcp_center, "refreshed_servers"))
        failed_mcp_servers = set(_manifest_values(mcp_center, "failed_servers"))
        partial_mcp_servers = set(_manifest_values(mcp_center, "partial_servers"))
        skill_center = _skill_center_trace(trace)
        loaded_skill_names = set(_manifest_values(skill_center, "loaded_skill_names"))
        skill_resource_view_ids = set(_manifest_values(skill_center, "resource_view_ids"))
        skill_resource_view_skills = set(
            _manifest_values(skill_center, "resource_view_skill_names")
        )
        skill_resource_views = _skill_resource_views(skill_center)
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
        context_material_selection = _context_material_selection_trace(trace)
        context_material_records = _context_material_selection_records(
            context_material_selection
        )
        selected_context_materials = tuple(
            item for item in context_material_records if item.get("selected") is True
        )
        dropped_context_materials = tuple(
            item for item in context_material_records if item.get("selected") is False
        )
        selected_context_material_names = _context_material_values(
            selected_context_materials,
            "name",
        )
        context_material_statuses = _context_material_values(
            context_material_records,
            "status",
        )
        context_material_targets = _context_material_values(
            context_material_records,
            "target",
        )
        selected_context_material_bytes = sum(
            _safe_int(item.get("bytes")) for item in selected_context_materials
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
        prompt_budget = _prompt_budget(trace)
        prompt_budget_source = str(prompt_budget.get("source") or "")
        prompt_budget_provider_name = str(prompt_budget.get("provider_name") or "")
        prompt_budget_provider_limited = bool(prompt_budget.get("provider_limited"))
        prompt_budget_target_bytes = _prompt_budget_int(prompt_budget, "target_prompt_bytes")
        prompt_budget_provider_input_bytes = _prompt_budget_int(
            prompt_budget,
            "provider_input_budget_bytes",
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
        prompt_semantic_trim = _prompt_semantic_trim(trace)
        prompt_semantic_trim_decisions = _prompt_semantic_trim_decisions(prompt_semantic_trim)
        prompt_semantic_trim_roles = _prompt_semantic_trim_values(
            prompt_semantic_trim_decisions,
            "role",
        )
        prompt_semantic_trim_statuses = _prompt_semantic_trim_values(
            prompt_semantic_trim_decisions,
            "status",
        )
        trimmed_prompt_semantic_buckets = tuple(
            decision
            for decision in prompt_semantic_trim_decisions
            if decision.get("status") == "trimmed"
        )
        prompt_semantic_dropped_units = sum(
            _safe_int(decision.get("dropped_units"))
            for decision in prompt_semantic_trim_decisions
        )
        prompt_semantic_trim_final_bytes = _prompt_trim_int(prompt_semantic_trim, "final_bytes")
        prompt_semantic_trim_original_bytes = _prompt_trim_int(prompt_semantic_trim, "original_bytes")
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
        approval_trace = _approval_trace(trace)
        approval_records = _approval_records(approval_trace)
        approval_statuses = _approval_values(approval_records, "status")
        approval_subjects = _approval_values(approval_records, "subject")
        approval_subject_kinds = _approval_values(approval_records, "subject_kind")
        pending_approvals = tuple(
            record for record in approval_records if record.get("status") == "pending"
        )
        rejected_approvals = tuple(
            record for record in approval_records if record.get("status") == "rejected"
        )
        approved_approval_subjects = {
            str(record.get("subject") or "")
            for record in approval_records
            if record.get("status") == "approved" and record.get("subject")
        }
        artifact_trace = _artifact_trace(trace)
        artifact_records = _artifact_records(artifact_trace)
        artifact_kinds = _artifact_values(artifact_records, "kind")
        artifact_tool_names = _artifact_values(artifact_records, "tool_name")
        artifact_content_types = _artifact_values(artifact_records, "content_type")
        artifact_total_bytes = sum(_safe_int(record.get("size_bytes")) for record in artifact_records)
        artifact_max_bytes = max(
            (_safe_int(record.get("size_bytes")) for record in artifact_records),
            default=0,
        )
        structured_output_trace = _structured_output_trace(trace)
        structured_output_records = _structured_output_records(structured_output_trace)
        structured_output_schema_names = _structured_output_values(
            structured_output_records,
            "schema_name",
        )
        structured_output_errors = _structured_output_values(
            structured_output_records,
            "error",
        )
        structured_output_repairs = tuple(
            record
            for record in structured_output_records
            if record.get("status") == "structured_output_error"
        )
        structured_output_failures = tuple(
            record for record in structured_output_records if record.get("ok") is False
        )
        structured_output_ok = any(record.get("ok") is True for record in structured_output_records)
        planner_trace = _planner_trace(trace)
        planner_plans = _planner_plans(planner_trace)
        planner_reports = _planner_reports(planner_trace)
        planner_steps = _planner_steps(planner_plans)
        planner_execution_steps = _planner_execution_steps(planner_reports)
        planner_plan_ids = _planner_values(planner_plans, "plan_id")
        planner_step_ids = _planner_values(planner_steps, "step_id")
        planner_step_statuses = _planner_values(planner_steps, "status")
        planner_execution_statuses = _planner_values(planner_reports, "status")
        failed_plan_steps = tuple(
            item
            for item in (*planner_steps, *planner_execution_steps)
            if str(item.get("status") or "") == "failed"
        )
        blocked_plan_reports = tuple(
            item for item in planner_reports if str(item.get("status") or "") == "blocked"
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
        if spec.require_provider_request_shape_plan and not provider_request_shape_plans:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "provider_request_shape_plan_missing",
                    "provider request shape plan trace is required",
                )
            )
        if spec.forbid_provider_request_shape_plan and provider_request_shape_plans:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "provider_request_shape_plan_forbidden",
                    "provider request shape plan trace is forbidden",
                )
            )
        if (
            spec.require_provider_request_shape_adjusted
            and not adjusted_provider_request_shapes
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "provider_request_shape_adjusted_missing",
                    "adjusted provider request shape is required",
                )
            )
        if spec.forbid_provider_request_shape_adjusted and adjusted_provider_request_shapes:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "provider_request_shape_adjusted_forbidden",
                    "adjusted provider request shape is forbidden",
                )
            )
        for decision in spec.required_provider_request_shape_decisions:
            if decision not in provider_request_shape_decisions:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_provider_request_shape_decision",
                        f"required provider request shape decision missing: {decision}",
                    )
                )
        for provider_name in spec.required_provider_request_shape_provider_names:
            if provider_name not in provider_request_shape_provider_names:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_provider_request_shape_provider",
                        f"required provider request shape provider missing: {provider_name}",
                    )
                )
        if (
            spec.max_provider_request_shape_final_output_tokens is not None
            and provider_request_shape_max_final_output_tokens
            > spec.max_provider_request_shape_final_output_tokens
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "provider_request_shape_final_output_tokens_exceeded",
                    "provider request shape final output tokens exceeded limit",
                    metadata={
                        "actual": provider_request_shape_max_final_output_tokens,
                        "limit": spec.max_provider_request_shape_final_output_tokens,
                    },
                )
            )
        if spec.require_lifecycle_hooks and not lifecycle_hook_records:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "lifecycle_hooks_missing",
                    "lifecycle hook trace is required",
                )
            )
        for event_type in spec.required_lifecycle_event_types:
            if event_type not in lifecycle_event_types:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_lifecycle_event_type",
                        f"required lifecycle event type missing: {event_type}",
                    )
                )
        for event_type in spec.forbidden_lifecycle_event_types:
            if event_type in lifecycle_event_types:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_lifecycle_event_type",
                        f"forbidden lifecycle event type present: {event_type}",
                    )
                )
        for status_value in spec.required_lifecycle_hook_statuses:
            if status_value not in lifecycle_hook_statuses:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_lifecycle_hook_status",
                        f"required lifecycle hook status missing: {status_value}",
                    )
                )
        if (
            spec.max_lifecycle_hook_failures is not None
            and lifecycle_hook_failure_count > spec.max_lifecycle_hook_failures
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "lifecycle_hook_failure_limit_exceeded",
                    "lifecycle hook failure count exceeded limit",
                    metadata={
                        "actual": lifecycle_hook_failure_count,
                        "limit": spec.max_lifecycle_hook_failures,
                    },
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
        if spec.require_preflight and not preflight:
            issues.append(
                TraceEvalIssue("error", "preflight_missing", "preflight trace is required")
            )
        if spec.require_preflight_passed and preflight_status != "passed":
            issues.append(
                TraceEvalIssue(
                    "error",
                    "preflight_not_passed",
                    "preflight is required to pass",
                    metadata={"status": preflight_status},
                )
            )
        if spec.require_preflight_blocked and preflight_status != "blocked":
            issues.append(
                TraceEvalIssue(
                    "error",
                    "preflight_not_blocked",
                    "preflight is required to block the run",
                    metadata={"status": preflight_status},
                )
            )
        for code in spec.required_preflight_issue_codes:
            if code not in preflight_issue_codes:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_preflight_issue_code",
                        f"required preflight issue code missing: {code}",
                    )
                )
        for code in spec.forbidden_preflight_issue_codes:
            if code in preflight_issue_codes:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_preflight_issue_code",
                        f"forbidden preflight issue code present: {code}",
                    )
                )
        if (
            spec.max_preflight_blocking_issues is not None
            and preflight_blocking_count > spec.max_preflight_blocking_issues
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "preflight_blocking_issue_limit_exceeded",
                    "preflight blocking issue count exceeded limit",
                    metadata={
                        "actual": preflight_blocking_count,
                        "limit": spec.max_preflight_blocking_issues,
                    },
                )
            )
        if spec.require_handoff and not handoff_trace:
            issues.append(TraceEvalIssue("error", "handoff_missing", "handoff trace is required"))
        for status_value in spec.required_handoff_statuses:
            if status_value not in handoff_statuses:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_handoff_status",
                        f"required handoff status missing: {status_value}",
                    )
                )
        for session_name in spec.required_handoff_selected_sessions:
            if session_name not in handoff_selected_sessions:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_handoff_selected_session",
                        f"required handoff selected session missing: {session_name}",
                    )
                )
        for session_name in spec.required_handoff_source_sessions:
            if session_name not in handoff_source_sessions:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_handoff_source_session",
                        f"required handoff source session missing: {session_name}",
                    )
                )
        if spec.max_handoff_denied is not None and len(denied_handoffs) > spec.max_handoff_denied:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "handoff_denied_limit_exceeded",
                    "denied handoff count exceeded limit",
                    metadata={"actual": len(denied_handoffs), "limit": spec.max_handoff_denied},
                )
            )
        if (
            spec.max_handoff_not_found is not None
            and len(not_found_handoffs) > spec.max_handoff_not_found
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "handoff_not_found_limit_exceeded",
                    "not-found handoff count exceeded limit",
                    metadata={
                        "actual": len(not_found_handoffs),
                        "limit": spec.max_handoff_not_found,
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

        if spec.require_context_material_selection and not context_material_selection:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "context_material_selection_missing",
                    "context material selection trace is required",
                )
            )
        for name in spec.required_selected_context_material_names:
            if name not in selected_context_material_names:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_selected_context_material_name",
                        f"required selected context material missing: {name}",
                    )
                )
        for status_value in spec.required_context_material_statuses:
            if status_value not in context_material_statuses:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_context_material_status",
                        f"required context material status missing: {status_value}",
                    )
                )
        for status_value in spec.forbidden_context_material_statuses:
            if status_value in context_material_statuses:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_context_material_status",
                        f"forbidden context material status present: {status_value}",
                    )
                )
        for target in spec.required_context_material_targets:
            if target not in context_material_targets:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_context_material_target",
                        f"required context material target missing: {target}",
                    )
                )
        if (
            spec.max_dropped_context_materials is not None
            and len(dropped_context_materials) > spec.max_dropped_context_materials
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "context_material_dropped_limit_exceeded",
                    "dropped context material count exceeded limit",
                    metadata={
                        "actual": len(dropped_context_materials),
                        "limit": spec.max_dropped_context_materials,
                    },
                )
            )
        if (
            spec.max_selected_context_material_bytes is not None
            and selected_context_material_bytes > spec.max_selected_context_material_bytes
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "context_material_selected_bytes_exceeded",
                    "selected context material bytes exceeded limit",
                    metadata={
                        "actual": selected_context_material_bytes,
                        "limit": spec.max_selected_context_material_bytes,
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

        if spec.require_prompt_budget and not prompt_budget:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_budget_missing",
                    "prompt budget trace is required",
                )
            )
        if spec.forbid_prompt_budget and prompt_budget:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_budget_forbidden",
                    "prompt budget trace is forbidden",
                )
            )
        if spec.require_prompt_budget_provider_limited and not prompt_budget_provider_limited:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_budget_provider_limit_missing",
                    "provider-limited prompt budget is required",
                )
            )
        if spec.forbid_prompt_budget_provider_limited and prompt_budget_provider_limited:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_budget_provider_limit_forbidden",
                    "provider-limited prompt budget is forbidden",
                )
            )
        for source in spec.required_prompt_budget_sources:
            if source != prompt_budget_source:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_prompt_budget_source",
                        f"required prompt budget source missing: {source}",
                        metadata={"actual": prompt_budget_source},
                    )
                )
        for provider_name in spec.required_prompt_budget_provider_names:
            if provider_name != prompt_budget_provider_name:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_prompt_budget_provider",
                        f"required prompt budget provider missing: {provider_name}",
                        metadata={"actual": prompt_budget_provider_name},
                    )
                )
        if (
            spec.max_prompt_budget_target_bytes is not None
            and prompt_budget_target_bytes > spec.max_prompt_budget_target_bytes
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_budget_target_bytes_exceeded",
                    "prompt budget target bytes exceeded limit",
                    metadata={
                        "actual": prompt_budget_target_bytes,
                        "limit": spec.max_prompt_budget_target_bytes,
                    },
                )
            )
        if (
            spec.max_prompt_budget_provider_input_bytes is not None
            and prompt_budget_provider_input_bytes > spec.max_prompt_budget_provider_input_bytes
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_budget_provider_input_bytes_exceeded",
                    "prompt budget provider input bytes exceeded limit",
                    metadata={
                        "actual": prompt_budget_provider_input_bytes,
                        "limit": spec.max_prompt_budget_provider_input_bytes,
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
        if spec.require_prompt_semantic_trim and not prompt_semantic_trim:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_semantic_trim_missing",
                    "prompt semantic trim trace is required",
                )
            )
        if spec.forbid_prompt_semantic_trim and prompt_semantic_trim:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_semantic_trim_forbidden",
                    "prompt semantic trim trace is forbidden",
                )
            )
        for role in spec.required_prompt_semantic_trim_roles:
            if role not in prompt_semantic_trim_roles:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_prompt_semantic_trim_role",
                        f"required prompt semantic trim role missing: {role}",
                    )
                )
        for status in spec.required_prompt_semantic_trim_statuses:
            if status not in prompt_semantic_trim_statuses:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_prompt_semantic_trim_status",
                        f"required prompt semantic trim status missing: {status}",
                    )
                )
        for status in spec.forbidden_prompt_semantic_trim_statuses:
            if status in prompt_semantic_trim_statuses:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_prompt_semantic_trim_status",
                        f"forbidden prompt semantic trim status present: {status}",
                    )
                )
        if (
            spec.max_prompt_semantic_trimmed is not None
            and len(trimmed_prompt_semantic_buckets) > spec.max_prompt_semantic_trimmed
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_semantic_trimmed_limit_exceeded",
                    "semantic-trimmed prompt bucket count exceeded limit",
                    metadata={
                        "actual": len(trimmed_prompt_semantic_buckets),
                        "limit": spec.max_prompt_semantic_trimmed,
                    },
                )
            )
        if (
            spec.max_prompt_semantic_dropped_units is not None
            and prompt_semantic_dropped_units > spec.max_prompt_semantic_dropped_units
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_semantic_dropped_units_exceeded",
                    "semantic prompt dropped unit count exceeded limit",
                    metadata={
                        "actual": prompt_semantic_dropped_units,
                        "limit": spec.max_prompt_semantic_dropped_units,
                    },
                )
            )
        if (
            spec.max_prompt_semantic_trim_final_bytes is not None
            and prompt_semantic_trim_final_bytes > spec.max_prompt_semantic_trim_final_bytes
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_semantic_trim_final_bytes_exceeded",
                    "prompt semantic trim final bytes exceeded limit",
                    metadata={
                        "actual": prompt_semantic_trim_final_bytes,
                        "limit": spec.max_prompt_semantic_trim_final_bytes,
                    },
                )
            )
        if (
            spec.max_prompt_semantic_trim_original_bytes is not None
            and prompt_semantic_trim_original_bytes > spec.max_prompt_semantic_trim_original_bytes
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "prompt_semantic_trim_original_bytes_exceeded",
                    "prompt semantic trim original bytes exceeded limit",
                    metadata={
                        "actual": prompt_semantic_trim_original_bytes,
                        "limit": spec.max_prompt_semantic_trim_original_bytes,
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
        if spec.require_approvals and not approval_trace:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "approval_trace_missing",
                    "approval trace is required",
                )
            )
        for approval_status in spec.required_approval_statuses:
            if approval_status not in approval_statuses:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_approval_status",
                        f"required approval status missing: {approval_status}",
                    )
                )
        for approval_status in spec.forbidden_approval_statuses:
            if approval_status in approval_statuses:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_approval_status",
                        f"forbidden approval status present: {approval_status}",
                    )
                )
        for subject in spec.required_approval_subjects:
            if subject not in approval_subjects:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_approval_subject",
                        f"required approval subject missing: {subject}",
                    )
                )
        for subject_kind in spec.required_approval_subject_kinds:
            if subject_kind not in approval_subject_kinds:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_approval_subject_kind",
                        f"required approval subject kind missing: {subject_kind}",
                    )
                )
        if (
            spec.max_pending_approvals is not None
            and len(pending_approvals) > spec.max_pending_approvals
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "pending_approval_limit_exceeded",
                    "pending approval count exceeded limit",
                    metadata={
                        "actual": len(pending_approvals),
                        "limit": spec.max_pending_approvals,
                    },
                )
            )
        if (
            spec.max_rejected_approvals is not None
            and len(rejected_approvals) > spec.max_rejected_approvals
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "rejected_approval_limit_exceeded",
                    "rejected approval count exceeded limit",
                    metadata={
                        "actual": len(rejected_approvals),
                        "limit": spec.max_rejected_approvals,
                    },
                )
            )
        for subject in spec.require_approved_approval_subjects:
            if subject not in approved_approval_subjects:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "approval_subject_not_approved",
                        f"required approval subject was not approved: {subject}",
                    )
                )
        if spec.require_artifacts and not artifact_trace:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "artifact_trace_missing",
                    "artifact trace is required",
                )
            )
        for kind in spec.required_artifact_kinds:
            if kind not in artifact_kinds:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_artifact_kind",
                        f"required artifact kind missing: {kind}",
                    )
                )
        for tool_name in spec.required_artifact_tool_names:
            if tool_name not in artifact_tool_names:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_artifact_tool_name",
                        f"required artifact tool name missing: {tool_name}",
                    )
                )
        for content_type in spec.required_artifact_content_types:
            if content_type not in artifact_content_types:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_artifact_content_type",
                        f"required artifact content type missing: {content_type}",
                    )
                )
        if spec.max_artifact_count is not None and len(artifact_records) > spec.max_artifact_count:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "artifact_count_limit_exceeded",
                    "artifact count exceeded limit",
                    metadata={
                        "actual": len(artifact_records),
                        "limit": spec.max_artifact_count,
                    },
                )
            )
        if (
            spec.max_artifact_total_bytes is not None
            and artifact_total_bytes > spec.max_artifact_total_bytes
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "artifact_total_bytes_limit_exceeded",
                    "artifact total bytes exceeded limit",
                    metadata={
                        "actual": artifact_total_bytes,
                        "limit": spec.max_artifact_total_bytes,
                    },
                )
            )
        if (
            spec.max_artifact_size_bytes is not None
            and artifact_max_bytes > spec.max_artifact_size_bytes
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "artifact_size_limit_exceeded",
                    "artifact size exceeded limit",
                    metadata={
                        "actual": artifact_max_bytes,
                        "limit": spec.max_artifact_size_bytes,
                    },
                )
            )
        if spec.require_structured_output and not structured_output_trace:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "structured_output_trace_missing",
                    "structured output trace is required",
                )
            )
        if spec.require_structured_output_ok and not structured_output_ok:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "structured_output_ok_missing",
                    "structured output must have at least one successful validation",
                )
            )
        for schema_name in spec.required_structured_output_schema_names:
            if schema_name not in structured_output_schema_names:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_structured_output_schema_name",
                        f"required structured output schema missing: {schema_name}",
                    )
                )
        for error in spec.forbidden_structured_output_errors:
            if error in structured_output_errors:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_structured_output_error",
                        f"forbidden structured output error present: {error}",
                    )
                )
        if (
            spec.max_structured_output_repairs is not None
            and len(structured_output_repairs) > spec.max_structured_output_repairs
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "structured_output_repair_limit_exceeded",
                    "structured output repair count exceeded limit",
                    metadata={
                        "actual": len(structured_output_repairs),
                        "limit": spec.max_structured_output_repairs,
                    },
                )
            )
        if (
            spec.max_structured_output_failures is not None
            and len(structured_output_failures) > spec.max_structured_output_failures
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "structured_output_failure_limit_exceeded",
                    "structured output failure count exceeded limit",
                    metadata={
                        "actual": len(structured_output_failures),
                        "limit": spec.max_structured_output_failures,
                    },
                )
            )
        if spec.require_planner_trace and not planner_trace:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "planner_trace_missing",
                    "planner trace is required",
                )
            )
        for plan_id in spec.required_plan_ids:
            if plan_id not in planner_plan_ids:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_plan_id",
                        f"required plan id missing: {plan_id}",
                    )
                )
        for step_id in spec.required_plan_step_ids:
            if step_id not in planner_step_ids:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_plan_step_id",
                        f"required plan step id missing: {step_id}",
                    )
                )
        for status_value in spec.required_plan_step_statuses:
            if status_value not in planner_step_statuses:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_plan_step_status",
                        f"required plan step status missing: {status_value}",
                    )
                )
        for status_value in spec.required_plan_execution_statuses:
            if status_value not in planner_execution_statuses:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_plan_execution_status",
                        f"required plan execution status missing: {status_value}",
                    )
                )
        if (
            spec.max_failed_plan_steps is not None
            and len(failed_plan_steps) > spec.max_failed_plan_steps
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "failed_plan_step_limit_exceeded",
                    "failed plan step count exceeded limit",
                    metadata={
                        "actual": len(failed_plan_steps),
                        "limit": spec.max_failed_plan_steps,
                    },
                )
            )
        if (
            spec.max_blocked_plan_reports is not None
            and len(blocked_plan_reports) > spec.max_blocked_plan_reports
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "blocked_plan_report_limit_exceeded",
                    "blocked plan report count exceeded limit",
                    metadata={
                        "actual": len(blocked_plan_reports),
                        "limit": spec.max_blocked_plan_reports,
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

        if spec.require_tool_center and not tool_center:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "tool_center_missing",
                    "tool center trace is required",
                )
            )
        for mount in spec.required_tool_center_selected_mounts:
            if mount not in tool_center_selected_mounts:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_tool_center_selected_mount",
                        f"required tool center selected mount missing: {mount}",
                    )
                )
        for tool_name in spec.required_tool_center_selected_tools:
            if tool_name not in tool_center_selected_tools:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_tool_center_selected_tool",
                        f"required tool center selected tool missing: {tool_name}",
                    )
                )
        for tool_name in spec.required_tool_center_requested_tools:
            if tool_name not in tool_center_requested_tools:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_tool_center_requested_tool",
                        f"required tool center requested tool missing: {tool_name}",
                    )
                )
        if spec.require_tool_center_ready_routes and not_ready_tool_center_routes:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "tool_center_route_not_ready",
                    "tool center route plan must be ready for every recorded call",
                    metadata={"not_ready_count": len(not_ready_tool_center_routes)},
                )
            )
        if (
            spec.max_tool_center_failed_calls is not None
            and len(failed_tool_center_calls) > spec.max_tool_center_failed_calls
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "tool_center_failed_call_limit_exceeded",
                    "tool center failed call count exceeded limit",
                    metadata={
                        "actual": len(failed_tool_center_calls),
                        "limit": spec.max_tool_center_failed_calls,
                    },
                )
            )
        if spec.require_agent_tools and not agent_tool_trace:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "agent_tools_missing",
                    "agent-tool trace is required",
                )
            )
        for tool_name in spec.required_agent_tool_names:
            if tool_name not in agent_tool_names:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_agent_tool_name",
                        f"required agent-tool name missing: {tool_name}",
                    )
                )
        for session_name in spec.required_agent_tool_sessions:
            if session_name not in agent_tool_sessions:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_agent_tool_session",
                        f"required agent-tool session missing: {session_name}",
                    )
                )
        for status_value in spec.required_agent_tool_statuses:
            if status_value not in agent_tool_statuses:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_agent_tool_status",
                        f"required agent-tool status missing: {status_value}",
                    )
                )
        if (
            spec.max_agent_tool_failures is not None
            and len(failed_agent_tools) > spec.max_agent_tool_failures
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "agent_tool_failure_limit_exceeded",
                    "agent-tool failed call count exceeded limit",
                    metadata={
                        "actual": len(failed_agent_tools),
                        "limit": spec.max_agent_tool_failures,
                    },
                )
            )
        if spec.require_mcp_center and not mcp_center:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "mcp_center_missing",
                    "MCP center trace is required",
                )
            )
        for server_name in spec.required_mcp_server_names:
            if server_name not in mcp_server_names:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_mcp_server",
                        f"required MCP server missing: {server_name}",
                    )
                )
        for server_name in spec.required_mcp_refreshed_servers:
            if server_name not in mcp_refreshed_servers:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "mcp_server_not_refreshed",
                        f"required MCP server was not refreshed: {server_name}",
                    )
                )
        for status in spec.forbidden_mcp_server_statuses:
            matching = sorted(
                name for name, item_status in mcp_server_statuses.items() if item_status == status
            )
            if matching:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "forbidden_mcp_server_status",
                        f"forbidden MCP server status present: {status}",
                        metadata={"status": status, "servers": matching},
                    )
                )
        if (
            spec.max_mcp_failed_servers is not None
            and len(failed_mcp_servers) > spec.max_mcp_failed_servers
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "mcp_failed_server_limit_exceeded",
                    "MCP failed server count exceeded limit",
                    metadata={
                        "actual": len(failed_mcp_servers),
                        "limit": spec.max_mcp_failed_servers,
                    },
                )
            )
        if (
            spec.max_mcp_partial_inventory_refreshes is not None
            and len(partial_mcp_servers) > spec.max_mcp_partial_inventory_refreshes
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "mcp_partial_inventory_refresh_limit_exceeded",
                    "MCP partial inventory refresh count exceeded limit",
                    metadata={
                        "actual": len(partial_mcp_servers),
                        "limit": spec.max_mcp_partial_inventory_refreshes,
                    },
                )
            )
        if spec.require_skill_center and not skill_center:
            issues.append(
                TraceEvalIssue(
                    "error",
                    "skill_center_missing",
                    "skill center trace is required",
                )
            )
        for skill_name in spec.required_loaded_skills:
            if skill_name not in loaded_skill_names:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_loaded_skill",
                        f"required loaded skill missing: {skill_name}",
                    )
                )
        for view_id in spec.required_skill_resource_view_ids:
            if view_id not in skill_resource_view_ids:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_skill_resource_view",
                        f"required skill resource view missing: {view_id}",
                    )
                )
        for skill_name in spec.required_skill_resource_view_skills:
            if skill_name not in skill_resource_view_skills:
                issues.append(
                    TraceEvalIssue(
                        "error",
                        "missing_skill_resource_view_skill",
                        f"required skill resource view skill missing: {skill_name}",
                    )
                )
        if (
            spec.max_skill_resource_views is not None
            and len(skill_resource_views) > spec.max_skill_resource_views
        ):
            issues.append(
                TraceEvalIssue(
                    "error",
                    "skill_resource_view_limit_exceeded",
                    "skill resource view count exceeded limit",
                    metadata={
                        "actual": len(skill_resource_views),
                        "limit": spec.max_skill_resource_views,
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
                "provider_request_shape_plan_count": len(provider_request_shape_plans),
                "provider_request_shape_adjusted_count": len(
                    adjusted_provider_request_shapes
                ),
                "provider_request_shape_provider_names": sorted(
                    provider_request_shape_provider_names
                ),
                "provider_request_shape_decisions": sorted(
                    provider_request_shape_decisions
                ),
                "provider_request_shape_max_final_output_tokens": (
                    provider_request_shape_max_final_output_tokens
                ),
                "lifecycle_hook_record_count": len(lifecycle_hook_records),
                "lifecycle_event_types": sorted(lifecycle_event_types),
                "lifecycle_hook_statuses": sorted(lifecycle_hook_statuses),
                "lifecycle_hook_failure_count": lifecycle_hook_failure_count,
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
                "has_tool_center": bool(tool_center),
                "tool_center_call_count": len(tool_center_calls),
                "tool_center_failed_count": len(failed_tool_center_calls),
                "tool_center_route_plan_count": len(tool_center_route_plans),
                "tool_center_not_ready_route_count": len(not_ready_tool_center_routes),
                "tool_center_selected_mounts": sorted(tool_center_selected_mounts),
                "tool_center_selected_tools": sorted(tool_center_selected_tools),
                "tool_center_requested_tools": sorted(tool_center_requested_tools),
                "has_agent_tool_trace": bool(agent_tool_trace),
                "agent_tool_record_count": len(agent_tool_records),
                "agent_tool_failed_count": len(failed_agent_tools),
                "agent_tool_names": sorted(agent_tool_names),
                "agent_tool_sessions": sorted(agent_tool_sessions),
                "agent_tool_statuses": sorted(agent_tool_statuses),
                "has_mcp_center": bool(mcp_center),
                "mcp_server_count": len(mcp_servers),
                "mcp_server_names": sorted(mcp_server_names),
                "mcp_server_statuses": dict(sorted(mcp_server_statuses.items())),
                "mcp_refreshed_servers": sorted(mcp_refreshed_servers),
                "mcp_failed_servers": sorted(failed_mcp_servers),
                "mcp_partial_servers": sorted(partial_mcp_servers),
                "has_skill_center": bool(skill_center),
                "loaded_skill_count": len(loaded_skill_names),
                "loaded_skill_names": sorted(loaded_skill_names),
                "skill_resource_view_count": len(skill_resource_views),
                "skill_resource_view_ids": sorted(skill_resource_view_ids),
                "skill_resource_view_skills": sorted(skill_resource_view_skills),
                "has_resume": bool(resume),
                "has_resume_plan": bool(resume_plan),
                "resume_plan_ready": bool(resume_plan.get("ready")) if resume_plan else False,
                "resume_checkpoint_id": str(resume.get("checkpoint_id") or ""),
                "resume_plan_checkpoint_id": str(resume_plan.get("checkpoint_id") or ""),
                "has_preflight": bool(preflight),
                "preflight_status": preflight_status,
                "preflight_issue_codes": sorted(preflight_issue_codes),
                "preflight_blocking_codes": sorted(preflight_blocking_codes),
                "preflight_blocking_count": preflight_blocking_count,
                "has_handoff": bool(handoff_trace),
                "handoff_record_count": len(handoff_records),
                "handoff_statuses": sorted(handoff_statuses),
                "handoff_selected_sessions": sorted(handoff_selected_sessions),
                "handoff_source_sessions": sorted(handoff_source_sessions),
                "handoff_denied_count": len(denied_handoffs),
                "handoff_not_found_count": len(not_found_handoffs),
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
                "has_context_material_selection": bool(context_material_selection),
                "context_material_selection_count": len(context_material_records),
                "selected_context_material_count": len(selected_context_materials),
                "dropped_context_material_count": len(dropped_context_materials),
                "selected_context_material_names": sorted(selected_context_material_names),
                "context_material_statuses": sorted(context_material_statuses),
                "context_material_targets": sorted(context_material_targets),
                "selected_context_material_bytes": selected_context_material_bytes,
                "memory_governance_decision_count": len(memory_governance_decisions),
                "memory_governance_decisions": sorted(memory_governance_statuses),
                "denied_memory_write_count": len(denied_memory_writes),
                "rewritten_memory_write_count": len(rewritten_memory_writes),
                "high_risk_memory_write_count": len(high_risk_memory_writes),
                "has_prompt_budget": bool(prompt_budget),
                "prompt_budget_source": prompt_budget_source,
                "prompt_budget_provider_name": prompt_budget_provider_name,
                "prompt_budget_provider_limited": prompt_budget_provider_limited,
                "prompt_budget_target_bytes": prompt_budget_target_bytes,
                "prompt_budget_provider_input_bytes": prompt_budget_provider_input_bytes,
                "has_prompt_bucket_budget": bool(prompt_bucket_budget),
                "prompt_bucket_budget_role_count": len(prompt_bucket_budget_roles),
                "prompt_bucket_budget_roles": sorted(prompt_bucket_budget_roles),
                "prompt_bucket_budget_statuses": sorted(prompt_bucket_budget_statuses),
                "prompt_bucket_budget_trimmed_count": len(trimmed_prompt_buckets),
                "prompt_bucket_budget_over_budget_count": len(over_budget_prompt_buckets),
                "has_prompt_semantic_trim": bool(prompt_semantic_trim),
                "prompt_semantic_trim_roles": sorted(prompt_semantic_trim_roles),
                "prompt_semantic_trim_statuses": sorted(prompt_semantic_trim_statuses),
                "prompt_semantic_trimmed_count": len(trimmed_prompt_semantic_buckets),
                "prompt_semantic_dropped_units": prompt_semantic_dropped_units,
                "prompt_semantic_trim_original_bytes": prompt_semantic_trim_original_bytes,
                "prompt_semantic_trim_final_bytes": prompt_semantic_trim_final_bytes,
                "has_prompt_trim": bool(prompt_trim),
                "prompt_trim_roles": sorted(prompt_trim_roles),
                "prompt_trim_original_bytes": prompt_trim_original_bytes,
                "prompt_trim_final_bytes": prompt_trim_final_bytes,
                "event_log_types": sorted(event_log_types),
                "event_log_sequence_monotonic": event_sequence_monotonic,
                "duplicate_event_sequence_count": duplicate_event_sequence_count,
                "has_approval_trace": bool(approval_trace),
                "approval_record_count": len(approval_records),
                "approval_statuses": sorted(approval_statuses),
                "approval_subjects": sorted(approval_subjects),
                "approval_subject_kinds": sorted(approval_subject_kinds),
                "pending_approval_count": len(pending_approvals),
                "rejected_approval_count": len(rejected_approvals),
                "approved_approval_subjects": sorted(approved_approval_subjects),
                "has_artifact_trace": bool(artifact_trace),
                "artifact_count": len(artifact_records),
                "artifact_kinds": sorted(artifact_kinds),
                "artifact_tool_names": sorted(artifact_tool_names),
                "artifact_content_types": sorted(artifact_content_types),
                "artifact_total_bytes": artifact_total_bytes,
                "artifact_max_bytes": artifact_max_bytes,
                "has_structured_output_trace": bool(structured_output_trace),
                "structured_output_record_count": len(structured_output_records),
                "structured_output_ok": structured_output_ok,
                "structured_output_schema_names": sorted(structured_output_schema_names),
                "structured_output_errors": sorted(structured_output_errors),
                "structured_output_repair_count": len(structured_output_repairs),
                "structured_output_failure_count": len(structured_output_failures),
                "has_planner_trace": bool(planner_trace),
                "planner_plan_count": len(planner_plans),
                "planner_step_count": len(planner_steps),
                "planner_execution_report_count": len(planner_reports),
                "planner_execution_step_count": len(planner_execution_steps),
                "planner_failed_step_count": len(failed_plan_steps),
                "planner_blocked_report_count": len(blocked_plan_reports),
                "planner_plan_ids": sorted(planner_plan_ids),
                "planner_step_ids": sorted(planner_step_ids),
                "planner_step_statuses": sorted(planner_step_statuses),
                "planner_execution_statuses": sorted(planner_execution_statuses),
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


def _prompt_shaping_replay_steps(trace: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    steps: list[dict[str, Any]] = []
    context_selection = _context_material_selection_trace(trace)
    if context_selection:
        steps.append(
            {
                "source": "context_material_selection",
                "event_type": "context_material_selection_applied",
                "payload": _context_material_selection_replay_payload(context_selection),
            }
        )
    prompt_budget = _prompt_budget(trace)
    if prompt_budget:
        steps.append(
            {
                "source": "prompt_budget",
                "event_type": "prompt_budget_applied",
                "payload": _prompt_budget_replay_payload(prompt_budget),
            }
        )
    bucket_budget = _prompt_bucket_budget(trace)
    if bucket_budget:
        steps.append(
            {
                "source": "prompt_bucket_budget",
                "event_type": "prompt_bucket_budget_applied",
                "payload": _prompt_bucket_budget_replay_payload(bucket_budget),
            }
        )
    semantic_trim = _prompt_semantic_trim(trace)
    if semantic_trim:
        steps.append(
            {
                "source": "prompt_semantic_trim",
                "event_type": "prompt_semantic_trim_applied",
                "payload": _prompt_semantic_trim_replay_payload(semantic_trim),
            }
        )
    trim = _prompt_trim(trace)
    if trim:
        steps.append(
            {
                "source": "prompt_trim",
                "event_type": "prompt_trim_applied",
                "payload": _prompt_trim_replay_payload(trim),
            }
        )
    return tuple(steps)


def _planner_replay_steps(trace: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    planner = _planner_trace(trace)
    if not planner:
        return ()
    steps: list[dict[str, Any]] = []
    for plan in _planner_plans(planner):
        steps.append(
            {
                "event_type": "plan_created",
                "payload": {
                    "plan_id": str(plan.get("plan_id") or ""),
                    "terminal": bool(plan.get("terminal")),
                    "ready_steps": list(plan.get("ready_steps") or ()),
                    "status_counts": dict(plan.get("status_counts") or {}),
                },
            }
        )
    for report in _planner_reports(planner):
        status = str(report.get("status") or "unknown")
        plan = report.get("plan") if isinstance(report.get("plan"), dict) else {}
        steps.append(
            {
                "event_type": f"plan_execution_{status}",
                "payload": {
                    "status": status,
                    "plan_id": str(plan.get("plan_id") or ""),
                    "step_count": len(_dict_items(report.get("steps"))),
                    "metadata": dict(report.get("metadata") or {})
                    if isinstance(report.get("metadata"), dict)
                    else {},
                },
            }
        )
    return tuple(steps)


def _handoff_replay_steps(trace: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    steps: list[dict[str, Any]] = []
    for record in _handoff_records(_handoff_trace(trace)):
        status = str(record.get("status") or "unknown")
        steps.append(
            {
                "event_type": f"handoff_{status}",
                "payload": {
                    "status": status,
                    "selected_session": str(record.get("selected_session") or ""),
                    "source_session": str(record.get("source_session") or ""),
                    "target_session": str(record.get("target_session") or ""),
                    "candidate_count": _safe_int(record.get("candidate_count")),
                    "reason": str(record.get("reason") or ""),
                },
            }
        )
    return tuple(steps)


def _agent_tool_replay_steps(trace: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    steps: list[dict[str, Any]] = []
    for record in _agent_tool_records(_agent_tool_trace(trace)):
        status = str(record.get("status") or "unknown")
        steps.append(
            {
                "event_type": f"agent_tool_{status}",
                "payload": {
                    "status": status,
                    "tool_name": str(record.get("tool_name") or ""),
                    "session_name": str(record.get("session_name") or ""),
                    "run_id": str(record.get("run_id") or ""),
                    "iterations": _safe_int(record.get("iterations")),
                    "task_bytes": _safe_int(record.get("task_bytes")),
                    "output_bytes": _safe_int(record.get("output_bytes")),
                    "error": str(record.get("error") or ""),
                },
            }
        )
    return tuple(steps)


def _capability_center_replay_steps(trace: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    steps: list[dict[str, Any]] = []
    mcp_center = _mcp_center_trace(trace)
    for refresh in _dict_items(mcp_center.get("last_inventory_refresh")):
        status = str(refresh.get("status") or "unknown")
        steps.append(
            {
                "source": "mcp_center",
                "event_type": f"mcp_inventory_{status}",
                "payload": _mcp_inventory_replay_payload(refresh),
            }
        )
    if mcp_center and not steps:
        for server in _mcp_server_records(mcp_center):
            state = server.get("state") if isinstance(server.get("state"), dict) else {}
            status = str(state.get("status") or server.get("status") or "registered")
            steps.append(
                {
                    "source": "mcp_center",
                    "event_type": f"mcp_server_{status}",
                    "payload": _mcp_server_replay_payload(server),
                }
            )
    skill_center = _skill_center_trace(trace)
    for skill_name in _manifest_values(skill_center, "loaded_skill_names"):
        steps.append(
            {
                "source": "skill_center",
                "event_type": "skill_loaded",
                "payload": {"skill_name": skill_name},
            }
        )
    for view in _skill_resource_views(skill_center):
        steps.append(
            {
                "source": "skill_center",
                "event_type": "skill_resource_view_loaded",
                "payload": _skill_resource_view_replay_payload(view),
            }
        )
    return tuple(steps)


def _approval_replay_steps(trace: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    steps: list[dict[str, Any]] = []
    for record in _approval_records(_approval_trace(trace)):
        status = str(record.get("status") or "pending")
        steps.append(
            {
                "event_type": f"approval_{status}",
                "run_id": str(record.get("run_id") or ""),
                "turn_id": str(record.get("turn_id") or ""),
                "payload": {
                    "approval_id": str(record.get("approval_id") or ""),
                    "subject": str(record.get("subject") or ""),
                    "subject_kind": str(record.get("subject_kind") or ""),
                    "status": status,
                    "decision_status": str(record.get("decision_status") or ""),
                    "decision_actor": str(record.get("decision_actor") or ""),
                },
            }
        )
    return tuple(steps)


def _artifact_replay_steps(trace: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "artifact_id": str(record.get("artifact_id") or ""),
            "uri": str(record.get("uri") or ""),
            "content_type": str(record.get("content_type") or ""),
            "size_bytes": _safe_int(record.get("size_bytes")),
            "sha256": str(record.get("sha256") or ""),
            "kind": str(record.get("kind") or ""),
            "tool_name": str(record.get("tool_name") or ""),
            "call_id": str(record.get("call_id") or ""),
        }
        for record in _artifact_records(_artifact_trace(trace))
    )


def _structured_output_replay_steps(trace: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    steps: list[dict[str, Any]] = []
    for record in _structured_output_records(_structured_output_trace(trace)):
        status = str(record.get("status") or "structured_output")
        if record.get("ok") is True:
            event_type = "structured_output_ok"
        elif status == "structured_output_error":
            event_type = "structured_output_repair_requested"
        else:
            event_type = "structured_output_failed"
        steps.append(
            {
                "event_type": event_type,
                "run_id": str(record.get("run_id") or ""),
                "turn_id": str(record.get("turn_id") or ""),
                "payload": {
                    "status": status,
                    "ok": bool(record.get("ok")),
                    "schema_name": str(record.get("schema_name") or ""),
                    "raw_output_bytes": _safe_int(record.get("raw_output_bytes")),
                    "error": str(record.get("error") or ""),
                    "repair_attempt": _safe_int(record.get("repair_attempt")),
                    "iteration": _safe_int(record.get("iteration")),
                },
            }
        )
    return tuple(steps)


def _mcp_inventory_replay_payload(refresh: dict[str, Any]) -> dict[str, Any]:
    return {
        "server_name": str(refresh.get("server_name") or ""),
        "status": str(refresh.get("status") or ""),
        "ok": bool(refresh.get("ok")),
        "tool_count": _safe_int(refresh.get("tool_count")),
        "resource_count": _safe_int(refresh.get("resource_count")),
        "prompt_count": _safe_int(refresh.get("prompt_count")),
        "has_resource_error": bool(refresh.get("resource_error")),
        "has_prompt_error": bool(refresh.get("prompt_error")),
    }


def _mcp_server_replay_payload(server: dict[str, Any]) -> dict[str, Any]:
    state = server.get("state") if isinstance(server.get("state"), dict) else {}
    return {
        "server_name": str(server.get("name") or server.get("server_name") or ""),
        "transport": str(server.get("transport") or ""),
        "enabled": bool(server.get("enabled", True)),
        "status": str(state.get("status") or server.get("status") or ""),
        "tool_count": _safe_int(state.get("tool_count")),
        "resource_count": _safe_int(state.get("resource_count")),
        "prompt_count": _safe_int(state.get("prompt_count")),
    }


def _skill_resource_view_replay_payload(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "view_id": str(view.get("view_id") or ""),
        "skill_name": str(view.get("skill_name") or ""),
        "file_path": str(view.get("file_path") or ""),
        "offset": _safe_int(view.get("offset")),
        "total_lines": _safe_int(view.get("total_lines")),
        "max_bytes": _safe_int(view.get("max_bytes")),
    }


def _prompt_bucket_budget_replay_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    decisions = _prompt_bucket_budget_decisions(manifest)
    return {
        "schema_version": str(manifest.get("schema_version") or ""),
        "trimmed_count": _safe_int(manifest.get("trimmed_count")),
        "protected_count": _safe_int(manifest.get("protected_count")),
        "over_budget_count": _safe_int(manifest.get("over_budget_count")),
        "roles": sorted(_prompt_bucket_budget_values(decisions, "role")),
        "statuses": sorted(_prompt_bucket_budget_values(decisions, "status")),
        "decision_count": len(decisions),
    }


def _prompt_budget_replay_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": str(manifest.get("schema_version") or ""),
        "source": str(manifest.get("source") or ""),
        "provider_limited": bool(manifest.get("provider_limited")),
        "provider_name": str(manifest.get("provider_name") or ""),
        "model": str(manifest.get("model") or ""),
        "profile_max_prompt_bytes": _safe_int(manifest.get("profile_max_prompt_bytes")),
        "target_prompt_bytes": _safe_int(manifest.get("target_prompt_bytes")),
        "context_window_tokens": _safe_int(manifest.get("context_window_tokens")),
        "reserved_output_tokens": _safe_int(manifest.get("reserved_output_tokens")),
        "provider_input_budget_bytes": _safe_int(
            manifest.get("provider_input_budget_bytes")
        ),
    }


def _prompt_semantic_trim_replay_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    decisions = _prompt_semantic_trim_decisions(manifest)
    return {
        "schema_version": str(manifest.get("schema_version") or ""),
        "target_bytes": _safe_int(manifest.get("target_bytes")),
        "original_bytes": _safe_int(manifest.get("original_bytes")),
        "final_bytes": _safe_int(manifest.get("final_bytes")),
        "converged": bool(manifest.get("converged", True)),
        "trimmed_count": _safe_int(manifest.get("trimmed_count")),
        "roles": sorted(_prompt_semantic_trim_values(decisions, "role")),
        "statuses": sorted(_prompt_semantic_trim_values(decisions, "status")),
        "dropped_units": sum(_safe_int(decision.get("dropped_units")) for decision in decisions),
        "decision_count": len(decisions),
    }


def _context_material_selection_replay_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    records = _context_material_selection_records(manifest)
    selected = tuple(item for item in records if item.get("selected") is True)
    dropped = tuple(item for item in records if item.get("selected") is False)
    return {
        "schema_version": str(manifest.get("schema_version") or ""),
        "selection_count": len(records),
        "selected_count": len(selected),
        "dropped_count": len(dropped),
        "selected_bytes": sum(_safe_int(item.get("bytes")) for item in selected),
        "selected_names": sorted(_context_material_values(selected, "name")),
        "statuses": sorted(_context_material_values(records, "status")),
        "targets": sorted(_context_material_values(records, "target")),
    }


def _prompt_trim_replay_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": str(manifest.get("schema_version") or ""),
        "target_bytes": _safe_int(manifest.get("target_bytes")),
        "original_bytes": _safe_int(manifest.get("original_bytes")),
        "final_bytes": _safe_int(manifest.get("final_bytes")),
        "converged": bool(manifest.get("converged", True)),
        "roles": sorted(_prompt_trim_roles(manifest)),
        "trimmed_role_count": len(_prompt_trim_roles(manifest)),
    }


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
    request_shape_plan = _provider_request_shape_plan(call)
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
    if request_shape_plan:
        payload["request_shape_plan"] = request_shape_plan
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


def _provider_request_shape_plan(call: dict[str, Any]) -> dict[str, Any]:
    metadata = call.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    direct = metadata.get("request_shape_plan")
    if isinstance(direct, dict):
        return dict(direct)
    request = metadata.get("request")
    if isinstance(request, dict):
        request_metadata = request.get("metadata")
        if isinstance(request_metadata, dict):
            plan = request_metadata.get("request_shape_plan")
            if isinstance(plan, dict):
                return dict(plan)
    route_plan = metadata.get("route_plan")
    if isinstance(route_plan, dict):
        selected_route = route_plan.get("selected_route")
        if isinstance(selected_route, dict):
            route_metadata = selected_route.get("metadata")
            if isinstance(route_metadata, dict):
                plan = route_metadata.get("request_shape_plan")
                if isinstance(plan, dict):
                    return dict(plan)
    return {}


def _provider_request_shape_plans(
    calls: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    return tuple(plan for call in calls if (plan := _provider_request_shape_plan(call)))


def _provider_request_shape_values(
    plans: tuple[dict[str, Any], ...],
    key: str,
) -> set[str]:
    return {str(item.get(key) or "") for item in plans if item.get(key)}


def _provider_request_shape_decisions(plans: tuple[dict[str, Any], ...]) -> set[str]:
    decisions: set[str] = set()
    for plan in plans:
        raw = plan.get("decisions")
        if isinstance(raw, (list, tuple)):
            decisions.update(str(item) for item in raw if item)
    return decisions


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


def _lifecycle_hooks(trace: dict[str, Any]) -> dict[str, Any]:
    session = trace.get("session")
    if not isinstance(session, dict):
        return {}
    lifecycle = session.get("lifecycle_hooks")
    return dict(lifecycle) if isinstance(lifecycle, dict) else {}


def _lifecycle_hook_records(lifecycle_hooks: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    records = lifecycle_hooks.get("records")
    if not isinstance(records, (list, tuple)):
        return ()
    return tuple(dict(item) for item in records if isinstance(item, dict))


def _lifecycle_event_types(records: tuple[dict[str, Any], ...]) -> set[str]:
    return {str(item.get("event_type") or "") for item in records if item.get("event_type")}


def _lifecycle_hook_statuses(records: tuple[dict[str, Any], ...]) -> set[str]:
    return {str(item.get("status") or "") for item in records if item.get("status")}


def _lifecycle_hook_failure_count(records: tuple[dict[str, Any], ...]) -> int:
    return sum(1 for item in records if str(item.get("status") or "") == "failed")


def _lifecycle_hook_replay_event_type(record: dict[str, Any]) -> str:
    event_type = str(record.get("event_type") or "unknown")
    return f"lifecycle_hook_{event_type}"


def _lifecycle_hook_replay_payload(record: dict[str, Any]) -> dict[str, Any]:
    event = record.get("event")
    return {
        "hook_name": str(record.get("hook_name") or ""),
        "event_type": str(record.get("event_type") or ""),
        "status": str(record.get("status") or ""),
        "error": str(record.get("error") or ""),
        "event": dict(event) if isinstance(event, dict) else {},
    }


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


def _context_material_selection_trace(trace: dict[str, Any]) -> dict[str, Any]:
    manifest = trace.get("context_material_selection")
    if isinstance(manifest, dict) and manifest:
        return dict(manifest)
    prompt = trace.get("prompt")
    prompt_metadata = prompt.get("metadata") if isinstance(prompt, dict) else {}
    manifest = (
        prompt_metadata.get("context_material_selection")
        if isinstance(prompt_metadata, dict)
        else {}
    )
    if isinstance(manifest, dict) and manifest:
        return dict(manifest)
    records: list[dict[str, Any]] = []
    for injection in _context_injections(trace):
        metadata = injection.get("metadata") if isinstance(injection.get("metadata"), dict) else {}
        selection = metadata.get("context_material_selection")
        if not isinstance(selection, dict):
            continue
        records.append(
            {
                "name": str(injection.get("name") or ""),
                "role": str(injection.get("source") or ""),
                "target": str(selection.get("target") or injection.get("target") or ""),
                "status": "selected",
                "selected": True,
                "score": _safe_float(selection.get("score")),
                "rank": _safe_int(selection.get("rank")),
                "reason": str(selection.get("reason") or ""),
                "priority": _safe_int(injection.get("priority")),
                "bytes": _safe_int(injection.get("bytes") or injection.get("final_bytes")),
                "sha256": str(injection.get("sha256") or ""),
                "metadata": dict(metadata),
            }
        )
    if not records:
        return {}
    return _context_material_selection_trace_from_records(tuple(records))


def _context_material_selection_trace_from_records(
    records: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    selected = tuple(item for item in records if item.get("selected") is True)
    dropped = tuple(item for item in records if item.get("selected") is False)
    return {
        "schema_version": "agent-core-context-material-selection-trace/v1",
        "selection_count": len(records),
        "selected_count": len(selected),
        "dropped_count": len(dropped),
        "selected_bytes": sum(_safe_int(item.get("bytes")) for item in selected),
        "statuses": _count_values(records, "status"),
        "targets": _count_values(selected, "target"),
        "names": _count_values(records, "name"),
        "roles": _count_values(records, "role"),
        "selections": list(records),
    }


def _context_material_selection_records(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in _dict_items(manifest.get("selections")))


def _context_material_values(records: tuple[dict[str, Any], ...], key: str) -> set[str]:
    return {str(item.get(key) or "") for item in records if item.get(key)}


def _planner_trace(trace: dict[str, Any]) -> dict[str, Any]:
    manifest = trace.get("planner_trace")
    if isinstance(manifest, dict) and manifest:
        return dict(manifest)
    session = trace.get("session") if isinstance(trace.get("session"), dict) else {}
    for key in ("planner_trace", "planner", "plan_executor"):
        value = session.get(key) if isinstance(session, dict) else None
        if isinstance(value, dict) and value:
            return _planner_trace_from_manifest(value)
    return {}


def _planner_trace_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    schema = str(manifest.get("schema_version") or "")
    if schema == "agent-core-planner-trace/v1":
        return dict(manifest)
    plans, reports = _planner_materials_from_manifest(manifest)
    steps = _planner_steps(plans)
    execution_steps = _planner_execution_steps(reports)
    failed_steps = tuple(
        item
        for item in (*steps, *execution_steps)
        if str(item.get("status") or "") == "failed"
    )
    blocked_reports = tuple(
        item for item in reports if str(item.get("status") or "") == "blocked"
    )
    return {
        "schema_version": "agent-core-planner-trace/v1",
        "plan_count": len(plans),
        "step_count": len(steps),
        "execution_report_count": len(reports),
        "execution_step_count": len(execution_steps),
        "failed_step_count": len(failed_steps),
        "blocked_report_count": len(blocked_reports),
        "plans": list(plans),
        "reports": list(reports),
    }


def _planner_materials_from_manifest(
    manifest: dict[str, Any],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    schema = str(manifest.get("schema_version") or "")
    if schema == "agent-core-plan/v1":
        return ((dict(manifest),), ())
    if schema == "agent-core-plan-execution-report/v1":
        plan = manifest.get("plan") if isinstance(manifest.get("plan"), dict) else {}
        return ((dict(plan),) if plan else (), (dict(manifest),))
    if schema == "agent-core-plan-executor/v1":
        reports = tuple(dict(item) for item in _dict_items(manifest.get("reports")))
        plans = tuple(
            dict(item.get("plan"))
            for item in reports
            if isinstance(item.get("plan"), dict)
        )
        return plans, reports
    reports = tuple(dict(item) for item in _dict_items(manifest.get("reports")))
    plans = tuple(dict(item) for item in _dict_items(manifest.get("plans")))
    if reports:
        report_plans = tuple(
            dict(item.get("plan"))
            for item in reports
            if isinstance(item.get("plan"), dict)
        )
        return (*plans, *report_plans), reports
    if plans:
        return plans, ()
    plan = manifest.get("plan") if isinstance(manifest.get("plan"), dict) else {}
    return ((dict(plan),) if plan else (), ())


def _planner_plans(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in _dict_items(manifest.get("plans")))


def _planner_reports(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in _dict_items(manifest.get("reports")))


def _planner_steps(plans: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for plan in plans:
        plan_id = str(plan.get("plan_id") or "")
        for step in _dict_items(plan.get("steps")):
            records.append({"plan_id": plan_id, **dict(step)})
    return tuple(records)


def _planner_execution_steps(
    reports: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for report in reports:
        status = str(report.get("status") or "")
        for step in _dict_items(report.get("steps")):
            records.append({"report_status": status, **dict(step)})
    return tuple(records)


def _planner_values(records: tuple[dict[str, Any], ...], key: str) -> set[str]:
    return {str(item.get(key) or "") for item in records if item.get(key)}


def _handoff_trace(trace: dict[str, Any]) -> dict[str, Any]:
    manifest = trace.get("handoff_trace")
    if isinstance(manifest, dict) and manifest:
        return dict(manifest)
    prompt = trace.get("prompt")
    metadata = prompt.get("metadata") if isinstance(prompt, dict) else {}
    raw = metadata.get("handoff") if isinstance(metadata, dict) else {}
    records = tuple(_handoff_record(item) for item in _dict_items(raw))
    records = tuple(record for record in records if record.get("status"))
    if not records:
        return {}
    return _handoff_trace_from_records(records)


def _handoff_trace_from_records(records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    selected = tuple(item for item in records if item.get("status") == "selected")
    denied = tuple(item for item in records if item.get("status") == "denied")
    not_found = tuple(item for item in records if item.get("status") == "not_found")
    return {
        "schema_version": "agent-core-handoff-trace/v1",
        "record_count": len(records),
        "selected_count": len(selected),
        "denied_count": len(denied),
        "not_found_count": len(not_found),
        "statuses": _count_values(records, "status"),
        "selected_sessions": _count_values(records, "selected_session"),
        "source_sessions": _count_values(records, "source_session"),
        "target_sessions": _count_values(records, "target_session"),
        "records": list(records),
    }


def _handoff_record(record: dict[str, Any]) -> dict[str, Any]:
    request = record.get("request") if isinstance(record.get("request"), dict) else {}
    candidates = tuple(_dict_items(record.get("candidates")))
    selected_session = str(record.get("selected_session") or "")
    status = str(record.get("status") or "")
    if not status and selected_session:
        status = "selected"
    if not status:
        return {}
    return {
        "status": status,
        "selected_session": selected_session,
        "reason": str(record.get("reason") or ""),
        "source_session": str(request.get("source_session") or ""),
        "target_session": str(request.get("target_session") or ""),
        "task_bytes": len(str(request.get("task") or "").encode("utf-8")),
        "required_tags": list(request.get("required_tags") or ()),
        "required_tools": list(request.get("required_tools") or ()),
        "required_skills": list(request.get("required_skills") or ()),
        "candidate_count": len(candidates),
        "candidate_sessions": [
            str(candidate.get("session_name") or "")
            for candidate in candidates
            if candidate.get("session_name")
        ],
        "metadata": dict(record.get("metadata") if isinstance(record.get("metadata"), dict) else {}),
    }


def _handoff_records(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in _dict_items(manifest.get("records")))


def _handoff_values(records: tuple[dict[str, Any], ...], key: str) -> set[str]:
    return {str(item.get(key) or "") for item in records if item.get(key)}


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


def _prompt_budget(trace: dict[str, Any]) -> dict[str, Any]:
    manifest = trace.get("prompt_budget")
    if isinstance(manifest, dict) and manifest:
        return dict(manifest)
    prompt = trace.get("prompt")
    prompt_metadata = prompt.get("metadata") if isinstance(prompt, dict) else {}
    budget = prompt_metadata.get("prompt_budget") if isinstance(prompt_metadata, dict) else {}
    if isinstance(budget, dict) and budget:
        return dict(budget)
    metadata = trace.get("metadata")
    budget = metadata.get("prompt_budget") if isinstance(metadata, dict) else {}
    return dict(budget) if isinstance(budget, dict) else {}


def _prompt_budget_int(manifest: dict[str, Any], key: str) -> int:
    try:
        return int(manifest.get(key) or 0)
    except (TypeError, ValueError):
        return 0


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


def _prompt_semantic_trim(trace: dict[str, Any]) -> dict[str, Any]:
    manifest = trace.get("prompt_semantic_trim")
    if isinstance(manifest, dict) and manifest:
        return dict(manifest)
    prompt = trace.get("prompt")
    prompt_metadata = prompt.get("metadata") if isinstance(prompt, dict) else {}
    semantic = prompt_metadata.get("semantic_trim") if isinstance(prompt_metadata, dict) else {}
    if isinstance(semantic, dict) and semantic:
        return dict(semantic)
    metadata = trace.get("metadata")
    semantic = metadata.get("prompt_semantic_trim") if isinstance(metadata, dict) else {}
    return dict(semantic) if isinstance(semantic, dict) else {}


def _prompt_semantic_trim_decisions(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = manifest.get("decisions") if isinstance(manifest, dict) else ()
    if isinstance(raw, (list, tuple)):
        return tuple(dict(item) for item in raw if isinstance(item, dict))
    return ()


def _prompt_semantic_trim_values(decisions: tuple[dict[str, Any], ...], key: str) -> set[str]:
    return {str(item.get(key) or "") for item in decisions if item.get(key)}


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


def _tool_center_trace(trace: dict[str, Any]) -> dict[str, Any]:
    manifest = trace.get("tool_center")
    if isinstance(manifest, dict) and manifest:
        return dict(manifest)
    session = trace.get("session")
    tools = session.get("tools") if isinstance(session, dict) else {}
    if isinstance(tools, dict) and tools.get("schema_version") == "agent-core-tool-center/v1":
        return {
            "schema_version": "agent-core-tool-center-trace/v1",
            "call_count": int(tools.get("call_count") or 0),
            "calls": list(tools.get("calls") or ()),
        }
    return {}


def _tool_center_calls(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = manifest.get("calls") if isinstance(manifest, dict) else ()
    if isinstance(raw, (list, tuple)):
        return tuple(dict(item) for item in raw if isinstance(item, dict))
    return ()


def _tool_center_route_plans(calls: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(call.get("route_plan"))
        for call in calls
        if isinstance(call.get("route_plan"), dict)
    )


def _tool_center_call_values(calls: tuple[dict[str, Any], ...], key: str) -> set[str]:
    return {str(item.get(key) or "") for item in calls if item.get(key)}


def _tool_center_route_values(route_plans: tuple[dict[str, Any], ...], key: str) -> set[str]:
    return {str(item.get(key) or "") for item in route_plans if item.get(key)}


def _preflight_trace(trace: dict[str, Any]) -> dict[str, Any]:
    manifest = trace.get("preflight")
    return dict(manifest) if isinstance(manifest, dict) and manifest else {}


def _agent_tool_trace(trace: dict[str, Any]) -> dict[str, Any]:
    manifest = trace.get("agent_tool_trace")
    if isinstance(manifest, dict) and manifest:
        return dict(manifest)
    records: list[dict[str, Any]] = []
    for call in _tool_center_calls(_tool_center_trace(trace)):
        result = call.get("result") if isinstance(call.get("result"), dict) else {}
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        record = _agent_tool_record(metadata)
        if record:
            records.append(record)
    if not records:
        return {}
    return _agent_tool_trace_from_records(tuple(records))


def _agent_tool_trace_from_records(records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    failed = tuple(record for record in records if str(record.get("status") or "") != "completed")
    completed = tuple(record for record in records if str(record.get("status") or "") == "completed")
    return {
        "schema_version": "agent-core-agent-tool-trace/v1",
        "record_count": len(records),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "tools": dict(Counter(str(item.get("tool_name") or "") for item in records if item.get("tool_name"))),
        "sessions": dict(
            Counter(str(item.get("session_name") or "") for item in records if item.get("session_name"))
        ),
        "statuses": dict(Counter(str(item.get("status") or "") for item in records if item.get("status"))),
        "records": [dict(record) for record in records],
    }


def _agent_tool_records(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in _dict_items(manifest.get("records")))


def _agent_tool_values(records: tuple[dict[str, Any], ...], key: str) -> set[str]:
    return {str(record.get(key) or "") for record in records if record.get(key)}


def _agent_tool_record(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("schema_version") != "agent-core-agent-tool-call/v1":
        return {}
    tool = record.get("tool") if isinstance(record.get("tool"), dict) else {}
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    invocation = record.get("invocation") if isinstance(record.get("invocation"), dict) else {}
    status = str(result.get("status") or "")
    error = str(record.get("error") or "")
    if not status:
        status = "failed" if error else "unknown"
    return {
        "tool_name": str(tool.get("tool_name") or invocation.get("tool_name") or ""),
        "session_name": str(tool.get("session_name") or ""),
        "status": status,
        "run_id": str(result.get("run_id") or ""),
        "trace_run_id": str(result.get("trace_run_id") or ""),
        "iterations": _safe_int(result.get("iterations")),
        "task_bytes": _safe_int(record.get("task_bytes")),
        "output_bytes": _safe_int(result.get("output_bytes")),
        "error": error,
    }


def _mcp_center_trace(trace: dict[str, Any]) -> dict[str, Any]:
    manifest = trace.get("mcp_center")
    if isinstance(manifest, dict) and manifest:
        return dict(manifest)
    session = trace.get("session")
    if not isinstance(session, dict):
        return {}
    mcp = session.get("mcp") if isinstance(session.get("mcp"), dict) else {}
    if mcp.get("schema_version") == "agent-core-mcp-center/v1":
        return _mcp_center_trace_from_manifest(mcp)
    capabilities = session.get("capabilities") if isinstance(session.get("capabilities"), dict) else {}
    mcp = capabilities.get("mcp") if isinstance(capabilities.get("mcp"), dict) else {}
    if mcp.get("schema_version") == "agent-core-mcp-center/v1":
        return _mcp_center_trace_from_manifest(mcp)
    return {}


def _mcp_center_trace_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    servers = tuple(dict(item) for item in _dict_items(manifest.get("servers")))
    refreshes = tuple(dict(item) for item in _dict_items(manifest.get("last_inventory_refresh")))
    statuses: dict[str, str] = {}
    refreshed: list[str] = []
    failed: list[str] = []
    partial: list[str] = []
    for server in servers:
        name = str(server.get("name") or server.get("server_name") or "")
        if not name:
            continue
        state = server.get("state") if isinstance(server.get("state"), dict) else {}
        status = str(state.get("status") or server.get("status") or "")
        statuses[name] = status
        if status == "refreshed":
            refreshed.append(name)
        elif status == "failed":
            failed.append(name)
        elif status == "partial":
            partial.append(name)
    for refresh in refreshes:
        name = str(refresh.get("server_name") or "")
        status = str(refresh.get("status") or "")
        if name and status == "failed" and name not in failed:
            failed.append(name)
        if name and status == "partial" and name not in partial:
            partial.append(name)
    return {
        "schema_version": "agent-core-mcp-center-trace/v1",
        "server_count": len(servers),
        "server_names": sorted(statuses),
        "server_statuses": dict(sorted(statuses.items())),
        "refreshed_servers": sorted(refreshed),
        "failed_servers": sorted(failed),
        "partial_servers": sorted(partial),
        "servers": [dict(item) for item in servers],
        "last_inventory_refresh": [dict(item) for item in refreshes],
    }


def _mcp_server_records(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in _dict_items(manifest.get("servers")))


def _mcp_server_values(servers: tuple[dict[str, Any], ...], key: str) -> set[str]:
    values: set[str] = set()
    for server in servers:
        value = str(server.get(key) or "")
        if value:
            values.add(value)
    return values


def _mcp_server_statuses(
    servers: tuple[dict[str, Any], ...],
    manifest: dict[str, Any],
) -> dict[str, str]:
    statuses = manifest.get("server_statuses")
    if isinstance(statuses, dict):
        return {
            str(name): str(status)
            for name, status in statuses.items()
            if str(name) and str(status)
        }
    result: dict[str, str] = {}
    for server in servers:
        name = str(server.get("name") or server.get("server_name") or "")
        state = server.get("state") if isinstance(server.get("state"), dict) else {}
        status = str(state.get("status") or server.get("status") or "")
        if name and status:
            result[name] = status
    return result


def _skill_center_trace(trace: dict[str, Any]) -> dict[str, Any]:
    manifest = trace.get("skill_center")
    if isinstance(manifest, dict) and manifest:
        return dict(manifest)
    session = trace.get("session")
    if not isinstance(session, dict):
        return {}
    skills = session.get("skills") if isinstance(session.get("skills"), dict) else {}
    if skills.get("schema_version") == "agent-core-skills-context/v1":
        return _skill_center_trace_from_manifest(skills)
    capabilities = session.get("capabilities") if isinstance(session.get("capabilities"), dict) else {}
    skills = capabilities.get("skills") if isinstance(capabilities.get("skills"), dict) else {}
    if skills.get("schema_version") == "agent-core-skills-context/v1":
        return _skill_center_trace_from_manifest(skills)
    return {}


def _skill_center_trace_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    loaded = tuple(dict(item) for item in _dict_items(manifest.get("loaded_skills")))
    views = tuple(dict(item) for item in _dict_items(manifest.get("views")))
    loaded_names = tuple(str(item.get("name") or "") for item in loaded if item.get("name"))
    view_ids = tuple(str(item.get("view_id") or "") for item in views if item.get("view_id"))
    view_skills = tuple(str(item.get("skill_name") or "") for item in views if item.get("skill_name"))
    return {
        "schema_version": "agent-core-skill-center-trace/v1",
        "available_skill_count": int(manifest.get("available_skills_count") or 0),
        "loaded_skill_count": len(loaded),
        "resource_view_count": len(views),
        "loaded_skill_names": sorted(loaded_names),
        "resource_view_ids": sorted(view_ids),
        "resource_view_skill_names": sorted(set(view_skills)),
        "loaded_skills": [dict(item) for item in loaded],
        "views": [dict(item) for item in views],
    }


def _skill_resource_views(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in _dict_items(manifest.get("views")))


def _manifest_values(manifest: dict[str, Any], key: str) -> tuple[str, ...]:
    raw = manifest.get(key) if isinstance(manifest, dict) else ()
    if isinstance(raw, dict):
        return tuple(str(item) for item in raw if str(item))
    if isinstance(raw, (list, tuple, set)):
        return tuple(str(item) for item in raw if str(item))
    return ()


def _approval_trace(trace: dict[str, Any]) -> dict[str, Any]:
    manifest = trace.get("approval_trace")
    if isinstance(manifest, dict) and manifest:
        return dict(manifest)
    approvals = trace.get("approvals") if isinstance(trace.get("approvals"), dict) else {}
    records = tuple(_approval_trace_record(item) for item in _dict_items(approvals.get("records")))
    records = tuple(record for record in records if record.get("approval_id"))
    if not records:
        return {}
    statuses = _count_values(records, "status")
    subjects = _count_values(records, "subject")
    subject_kinds = _count_values(records, "subject_kind")
    return {
        "schema_version": "agent-core-approval-trace/v1",
        "record_count": len(records),
        "pending_count": sum(1 for record in records if record.get("status") == "pending"),
        "approved_count": sum(1 for record in records if record.get("status") == "approved"),
        "rejected_count": sum(1 for record in records if record.get("status") == "rejected"),
        "cancelled_count": sum(1 for record in records if record.get("status") == "cancelled"),
        "statuses": statuses,
        "subjects": subjects,
        "subject_kinds": subject_kinds,
        "records": list(records),
    }


def _approval_trace_record(record: dict[str, Any]) -> dict[str, Any]:
    request = record.get("request") if isinstance(record.get("request"), dict) else {}
    decision = record.get("decision") if isinstance(record.get("decision"), dict) else {}
    subject = str(request.get("subject") or record.get("subject") or "")
    status = str(record.get("status") or decision.get("status") or "")
    return {
        "approval_id": str(record.get("approval_id") or ""),
        "run_id": str(record.get("run_id") or ""),
        "turn_id": str(record.get("turn_id") or ""),
        "status": status,
        "subject": subject,
        "subject_kind": str(record.get("subject_kind") or (subject.split(":", 1)[0] if ":" in subject else "")),
        "reason": str(request.get("reason") or record.get("reason") or ""),
        "created_at": str(record.get("created_at") or ""),
        "decision_status": str(decision.get("status") or record.get("decision_status") or ""),
        "decision_actor": str(decision.get("actor") or record.get("decision_actor") or ""),
        "decision_reason": str(decision.get("reason") or record.get("decision_reason") or ""),
        "decided_at": str(decision.get("decided_at") or record.get("decided_at") or ""),
        "metadata": dict(record.get("metadata") or {}) if isinstance(record.get("metadata"), dict) else {},
    }


def _approval_records(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in _dict_items(manifest.get("records")))


def _approval_values(records: tuple[dict[str, Any], ...], key: str) -> set[str]:
    return {str(item.get(key) or "") for item in records if item.get(key)}


def _artifact_trace(trace: dict[str, Any]) -> dict[str, Any]:
    manifest = trace.get("artifact_trace")
    if isinstance(manifest, dict) and manifest:
        return dict(manifest)
    session = trace.get("session") if isinstance(trace.get("session"), dict) else {}
    store = session.get("artifact_store") if isinstance(session.get("artifact_store"), dict) else {}
    records = tuple(_artifact_trace_record(item) for item in _dict_items(store.get("artifacts")))
    records = tuple(record for record in records if record.get("artifact_id"))
    if not records:
        return {}
    total_bytes = sum(_safe_int(record.get("size_bytes")) for record in records)
    return {
        "schema_version": "agent-core-artifact-trace/v1",
        "artifact_count": len(records),
        "total_bytes": total_bytes,
        "max_artifact_bytes": max((_safe_int(record.get("size_bytes")) for record in records), default=0),
        "content_types": _count_values(records, "content_type"),
        "kinds": _count_values(records, "kind"),
        "tool_names": _count_values(records, "tool_name"),
        "artifacts": list(records),
    }


def _artifact_trace_record(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return {
        "artifact_id": str(record.get("artifact_id") or ""),
        "uri": str(record.get("uri") or ""),
        "content_type": str(record.get("content_type") or ""),
        "size_bytes": _safe_int(record.get("size_bytes")),
        "sha256": str(record.get("sha256") or ""),
        "kind": str(metadata.get("kind") or record.get("kind") or ""),
        "tool_name": str(metadata.get("tool_name") or record.get("tool_name") or ""),
        "call_id": str(metadata.get("call_id") or record.get("call_id") or ""),
        "status": str(metadata.get("status") or record.get("status") or ""),
        "metadata": dict(metadata),
    }


def _artifact_records(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in _dict_items(manifest.get("artifacts")))


def _artifact_values(records: tuple[dict[str, Any], ...], key: str) -> set[str]:
    return {str(item.get(key) or "") for item in records if item.get(key)}


def _structured_output_trace(trace: dict[str, Any]) -> dict[str, Any]:
    manifest = trace.get("structured_output_trace")
    if isinstance(manifest, dict) and manifest:
        return dict(manifest)
    journal = trace.get("journal_replay") if isinstance(trace.get("journal_replay"), dict) else {}
    records: list[dict[str, Any]] = []
    for event in _dict_items(journal.get("events")):
        if str(event.get("event_type") or "") != "checkpoint":
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
        record = _structured_output_record_from_state(
            state,
            run_id=str(event.get("run_id") or payload.get("run_id") or ""),
            turn_id=str(event.get("turn_id") or payload.get("turn_id") or ""),
            sequence=_safe_int(payload.get("sequence")),
        )
        if record:
            records.append(record)
    if not records:
        return {}
    return _structured_output_trace_from_records(tuple(records))


def _structured_output_trace_from_records(records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    ok_records = tuple(item for item in records if item.get("ok") is True)
    failed_records = tuple(item for item in records if item.get("ok") is False)
    repair_records = tuple(item for item in records if item.get("status") == "structured_output_error")
    return {
        "schema_version": "agent-core-structured-output-trace/v1",
        "record_count": len(records),
        "ok_count": len(ok_records),
        "failed_count": len(failed_records),
        "repair_count": len(repair_records),
        "statuses": _count_values(records, "status"),
        "schema_names": _count_values(records, "schema_name"),
        "errors": _count_values(records, "error"),
        "records": list(records),
    }


def _structured_output_record_from_state(
    state: dict[str, Any],
    *,
    run_id: str = "",
    turn_id: str = "",
    sequence: int = 0,
) -> dict[str, Any]:
    status = str(state.get("status") or "")
    result = state.get("structured_output") if isinstance(state.get("structured_output"), dict) else {}
    if status not in {"structured_output_error", "structured_output_failed"} and not result:
        return {}
    ok = result.get("ok")
    if ok is None and status in {"structured_output_error", "structured_output_failed"}:
        ok = False
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    schema_validation = (
        metadata.get("schema_validation")
        if isinstance(metadata.get("schema_validation"), dict)
        else {}
    )
    return {
        "run_id": run_id,
        "turn_id": turn_id,
        "sequence": sequence,
        "status": status or ("structured_output_ok" if ok is True else "structured_output_failed"),
        "ok": bool(ok) if ok is not None else False,
        "schema_name": str(metadata.get("schema_name") or schema_validation.get("schema_name") or ""),
        "raw_output_bytes": _safe_int(result.get("raw_output_bytes")),
        "error": str(result.get("error") or state.get("error") or ""),
        "repair_attempt": _safe_int(state.get("repair_attempt")),
        "iteration": _safe_int(state.get("iteration")),
        "schema_validation": dict(schema_validation),
    }


def _structured_output_records(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in _dict_items(manifest.get("records")))


def _structured_output_values(records: tuple[dict[str, Any], ...], key: str) -> set[str]:
    return {str(item.get(key) or "") for item in records if item.get(key)}


def _count_values(records: tuple[dict[str, Any], ...], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _dict_items(value: Any) -> tuple[dict[str, Any], ...]:
    if isinstance(value, dict):
        return (dict(value),)
    if isinstance(value, (list, tuple)):
        return tuple(dict(item) for item in value if isinstance(item, dict))
    return ()


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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
