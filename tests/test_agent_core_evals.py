from __future__ import annotations

import pytest

from agent_core.config import AgentProfile, RuntimeBudget
from agent_core.events import ListEventSink
from agent_core.evals import (
    DefaultTraceEvaluator,
    TraceEvalHarness,
    TraceEvalSpec,
    TraceReplayComparator,
    TraceReplayDiffSpec,
    TraceReplayHarness,
)
from agent_core.backends import (
    StorageBackendCatalog,
    StorageBackendRequirement,
    storage_backend_manifest,
)
from agent_core.runner import AgentRunner, AgentSession
from agent_core.providers import LLMModelCapabilities, LLMProviderCenter
from agent_core.harness import InMemoryAgentJournal
from agent_core.testing import MockLLMProvider, MockToolRuntime
from agent_core.tools import (
    InMemoryToolReplay,
    ToolExecutionCenter,
    ToolInvocation,
    ToolRegistry,
    ToolResult,
    ToolRetryPolicy,
    ToolSpec,
)
from agent_core.trace import InMemoryRunTraceStore


def _trace_manifest() -> dict[str, object]:
    return {
        "schema_version": "agent-core-run-trace-bundle/v1",
        "run": {"run_id": "run-1", "status": "completed", "iterations": 2, "output_bytes": 4},
        "summary": {
            "journal_ok": True,
            "journal_event_count": 2,
            "provider_call_count": 2,
            "tool_replay_record_count": 1,
            "approval_record_count": 3,
            "event_log_count": 2,
        },
        "journal_replay": {
            "ok": True,
            "events": [
                {
                    "sequence": 1,
                    "event_type": "run_started",
                    "run_id": "run-1",
                    "payload": {"run_id": "run-1"},
                },
                {
                    "sequence": 2,
                    "event_type": "run_finished",
                    "run_id": "run-1",
                    "payload": {"run_id": "run-1", "status": "completed"},
                },
            ],
        },
        "event_log": {
            "event_count": 2,
            "events": [
                {
                    "sequence": 1,
                    "type": "tool_started",
                    "run_id": "run-1",
                    "payload": {"tool_name": "lookup"},
                },
                {
                    "sequence": 2,
                    "type": "tool_finished",
                    "run_id": "run-1",
                    "payload": {"tool_name": "lookup"},
                },
            ],
        },
        "provider": {
            "call_count": 2,
            "calls": [
                {
                    "provider_name": "mock",
                    "model": "mock-mini",
                    "usage": {"cost_usd": 0.01},
                    "metadata": {
                        "request_shape_plan": {
                            "schema_version": "agent-core-llm-request-shape-plan/v1",
                            "requested_provider": "",
                            "requested_model": "",
                            "provider_name": "mock",
                            "model": "mock-mini",
                            "streamed": False,
                            "original_max_output_tokens": 256,
                            "final_max_output_tokens": 64,
                            "provider_max_output_tokens": 64,
                            "context_window_tokens": 4096,
                            "adjusted": True,
                            "decisions": ["max_output_tokens_capped_to_provider_limit"],
                            "metadata": {"provider_priority": 10},
                        },
                        "request": {
                            "max_output_tokens": 64,
                            "metadata": {
                                "model_capabilities": {
                                    "schema_version": "agent-core-llm-model-capabilities/v1",
                                    "supports_streaming": True,
                                    "supports_tool_calls": True,
                                    "supports_structured_output": True,
                                    "supports_json_mode": True,
                                    "modalities": ["text"],
                                    "capabilities": [
                                        "streaming",
                                        "tool_calls",
                                        "structured_output",
                                        "json_mode",
                                        "modality:text",
                                    ],
                                }
                            }
                        }
                    },
                },
                {
                    "provider_name": "mock",
                    "model": "mock-mini",
                    "usage": {"cost_usd": 0.02},
                    "metadata": {
                        "request": {
                            "metadata": {
                                "model_capabilities": {
                                    "schema_version": "agent-core-llm-model-capabilities/v1",
                                    "supports_streaming": True,
                                    "supports_tool_calls": True,
                                    "supports_structured_output": True,
                                    "supports_json_mode": True,
                                    "modalities": ["text"],
                                    "capabilities": [
                                        "streaming",
                                        "tool_calls",
                                        "structured_output",
                                        "json_mode",
                                        "modality:text",
                                    ],
                                }
                            }
                        }
                    },
                },
            ],
        },
        "prompt": {
            "metadata": {
                "handoff": {
                    "schema_version": "agent-core-handoff-decision/v1",
                    "status": "selected",
                    "selected_session": "code-reviewer",
                    "reason": "",
                    "request": {
                        "schema_version": "agent-core-handoff-request/v1",
                        "task": "review patch",
                        "source_session": "planner",
                        "target_session": "",
                        "required_tags": ["review"],
                        "required_tools": ["diff"],
                        "required_skills": [],
                        "metadata": {"ticket": "T-1"},
                    },
                    "candidates": [
                        {
                            "schema_version": "agent-core-handoff-spec/v1",
                            "session_name": "code-reviewer",
                            "description": "Review code",
                            "tags": ["code", "review"],
                            "tools": ["diff"],
                            "skills": ["review"],
                            "priority": 5,
                            "enabled": True,
                        }
                    ],
                    "metadata": {},
                },
                "context_injections": [
                    {
                        "name": "memory_recall",
                        "source": "memory",
                        "target": "semi_dynamic_1",
                        "status": "trimmed",
                        "included": True,
                        "trimmed": True,
                    },
                    {
                        "name": "operator_hint",
                        "source": "runtime",
                        "target": "dynamic",
                        "status": "included",
                        "included": True,
                        "trimmed": False,
                    },
                    {
                        "name": "denied_static",
                        "source": "runtime",
                        "target": "high_static",
                        "status": "target_denied",
                        "included": False,
                        "trimmed": False,
                    },
                ],
                "context_material_selection": {
                    "schema_version": "agent-core-context-material-selection-result/v1",
                    "selection_count": 3,
                    "selected_count": 1,
                    "dropped_count": 2,
                    "selected_bytes": 64,
                    "statuses": {
                        "count_exceeded": 1,
                        "score_below_threshold": 1,
                        "selected": 1,
                    },
                    "targets": {"timeline_open": 1},
                    "selections": [
                        {
                            "name": "auth_trace",
                            "role": "timeline",
                            "target": "timeline_open",
                            "status": "selected",
                            "selected": True,
                            "score": 314.5,
                            "rank": 1,
                            "reason": "semantic_priority_selected",
                            "priority": 3,
                            "bytes": 64,
                            "sha256": "authsha",
                        },
                        {
                            "name": "schema_hint",
                            "role": "schema",
                            "target": "semi_dynamic_2",
                            "status": "count_exceeded",
                            "selected": False,
                            "score": 211.2,
                            "rank": 2,
                            "reason": "max_materials_exceeded",
                            "priority": 2,
                            "bytes": 42,
                            "sha256": "schemasha",
                        },
                        {
                            "name": "old_note",
                            "role": "memory",
                            "target": "semi_dynamic_1",
                            "status": "score_below_threshold",
                            "selected": False,
                            "score": 0.5,
                            "rank": 0,
                            "reason": "score_below_threshold",
                            "priority": 1,
                            "bytes": 38,
                            "sha256": "oldsha",
                        },
                    ],
                },
                "trim": {
                    "schema_version": "agent-core-prompt-trim/v1",
                    "target_bytes": 900,
                    "original_bytes": 1200,
                    "final_bytes": 840,
                    "trimmed_roles": [
                        {
                            "role": "timeline_open",
                            "original_bytes": 500,
                            "trimmed_bytes": 140,
                            "reason": "trim volatile timeline context first",
                        }
                    ],
                },
                "semantic_trim": {
                    "schema_version": "agent-core-prompt-semantic-trim-result/v1",
                    "target_bytes": 900,
                    "original_bytes": 1180,
                    "final_bytes": 820,
                    "trimmed_count": 1,
                    "decisions": [
                        {
                            "role": "timeline_open",
                            "status": "trimmed",
                            "original_bytes": 520,
                            "final_bytes": 180,
                            "selected_units": 2,
                            "dropped_units": 3,
                        },
                        {
                            "role": "dynamic",
                            "status": "protected",
                            "original_bytes": 220,
                            "final_bytes": 220,
                            "protected": True,
                            "dropped_units": 0,
                        },
                    ],
                },
            }
        },
        "tool_replay": {
            "record_count": 1,
            "records": [
                {
                    "invocation": {"tool_name": "lookup"},
                    "result": {"tool_name": "lookup"},
                }
            ],
        },
        "tool_center": {
            "schema_version": "agent-core-tool-center-trace/v1",
            "call_count": 2,
            "failed_count": 1,
            "route_plan_count": 2,
            "ready_route_plan_count": 1,
            "selected_mounts": {"local": 1},
            "selected_tools": {"lookup": 1},
            "requested_tools": {"lookup": 1, "missing": 1},
            "calls": [
                {
                    "requested_tool_name": "lookup",
                    "status": "completed",
                    "route_plan": {
                        "schema_version": "agent-core-tool-route-plan/v1",
                        "ready": True,
                        "selected_mount": "local",
                        "selected_tool_name": "lookup",
                    },
                },
                {
                    "requested_tool_name": "missing",
                    "status": "failed",
                    "error": "unknown tool: missing",
                    "route_plan": {
                        "schema_version": "agent-core-tool-route-plan/v1",
                        "ready": False,
                        "selected_mount": "",
                        "selected_tool_name": "",
                    },
                },
            ],
        },
        "mcp_center": {
            "schema_version": "agent-core-mcp-center-trace/v1",
            "server_count": 2,
            "tool_count": 1,
            "resource_count": 1,
            "prompt_count": 1,
            "inventory_refresh_count": 2,
            "failed_server_count": 1,
            "disabled_server_count": 0,
            "partial_inventory_refresh_count": 1,
            "server_names": ["broken", "fs"],
            "server_statuses": {"broken": "failed", "fs": "refreshed"},
            "refreshed_servers": ["fs"],
            "failed_servers": ["broken"],
            "disabled_servers": [],
            "partial_servers": ["fs"],
            "transports": {"mock": 2},
            "servers": [
                {
                    "name": "fs",
                    "transport": "mock",
                    "enabled": True,
                    "state": {"status": "refreshed", "tool_count": 1},
                },
                {
                    "name": "broken",
                    "transport": "mock",
                    "enabled": True,
                    "state": {"status": "failed", "last_error": "boom"},
                },
            ],
            "last_inventory_refresh": [
                {
                    "server_name": "fs",
                    "status": "partial",
                    "ok": False,
                    "tool_count": 1,
                    "resource_count": 1,
                    "prompt_count": 0,
                    "prompt_error": "prompt listing failed",
                },
                {
                    "server_name": "broken",
                    "status": "failed",
                    "ok": False,
                    "tool_count": 0,
                    "resource_count": 0,
                    "prompt_count": 0,
                },
            ],
        },
        "skill_center": {
            "schema_version": "agent-core-skill-center-trace/v1",
            "available_skill_count": 3,
            "loaded_skill_count": 1,
            "resource_view_count": 1,
            "loaded_skill_names": ["review"],
            "resource_view_ids": ["review:rules.md:abcd"],
            "resource_view_skill_names": ["review"],
            "loaded_skills": [{"name": "review", "description": "Review code"}],
            "views": [
                {
                    "view_id": "review:rules.md:abcd",
                    "skill_name": "review",
                    "file_path": "rules.md",
                    "offset": 1,
                    "total_lines": 4,
                    "max_bytes": 1024,
                }
            ],
        },
        "approval_trace": {
            "schema_version": "agent-core-approval-trace/v1",
            "record_count": 3,
            "pending_count": 1,
            "approved_count": 1,
            "rejected_count": 1,
            "cancelled_count": 0,
            "statuses": {"approved": 1, "pending": 1, "rejected": 1},
            "subjects": {"tool:deploy": 1, "action:finish": 1, "tool:delete": 1},
            "subject_kinds": {"tool": 2, "action": 1},
            "records": [
                {
                    "approval_id": "approval-1",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "status": "approved",
                    "subject": "tool:deploy",
                    "subject_kind": "tool",
                    "reason": "deploy gate",
                    "decision_status": "approved",
                    "decision_actor": "operator",
                    "decision_reason": "ok",
                },
                {
                    "approval_id": "approval-2",
                    "run_id": "run-1",
                    "turn_id": "turn-2",
                    "status": "pending",
                    "subject": "action:finish",
                    "subject_kind": "action",
                    "reason": "final gate",
                    "decision_status": "",
                    "decision_actor": "",
                    "decision_reason": "",
                },
                {
                    "approval_id": "approval-3",
                    "run_id": "run-1",
                    "turn_id": "turn-3",
                    "status": "rejected",
                    "subject": "tool:delete",
                    "subject_kind": "tool",
                    "reason": "dangerous",
                    "decision_status": "rejected",
                    "decision_actor": "operator",
                    "decision_reason": "no",
                },
            ],
        },
        "artifact_trace": {
            "schema_version": "agent-core-artifact-trace/v1",
            "artifact_count": 1,
            "total_bytes": 4096,
            "max_artifact_bytes": 4096,
            "content_types": {"text/plain; charset=utf-8": 1},
            "kinds": {"tool_result": 1},
            "tool_names": {"dump": 1},
            "artifacts": [
                {
                    "artifact_id": "artifact-1",
                    "uri": "artifact://artifact-1",
                    "content_type": "text/plain; charset=utf-8",
                    "size_bytes": 4096,
                    "sha256": "abc123",
                    "kind": "tool_result",
                    "tool_name": "dump",
                    "call_id": "call-1",
                    "status": "completed",
                }
            ],
        },
        "structured_output_trace": {
            "schema_version": "agent-core-structured-output-trace/v1",
            "record_count": 2,
            "ok_count": 1,
            "failed_count": 1,
            "repair_count": 1,
            "statuses": {"structured_output_error": 1, "finished": 1},
            "schema_names": {"risk_summary": 1},
            "errors": {"$.risk is required": 1},
            "records": [
                {
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "sequence": 1,
                    "status": "structured_output_error",
                    "ok": False,
                    "schema_name": "",
                    "raw_output_bytes": 0,
                    "error": "$.risk is required",
                    "repair_attempt": 1,
                    "iteration": 0,
                    "schema_validation": {},
                },
                {
                    "run_id": "run-1",
                    "turn_id": "turn-2",
                    "sequence": 2,
                    "status": "finished",
                    "ok": True,
                    "schema_name": "risk_summary",
                    "raw_output_bytes": 28,
                    "error": "",
                    "repair_attempt": 0,
                    "iteration": 1,
                    "schema_validation": {"schema_name": "risk_summary", "ok": True},
                },
            ],
        },
        "planner_trace": {
            "schema_version": "agent-core-planner-trace/v1",
            "plan_count": 1,
            "step_count": 2,
            "execution_report_count": 1,
            "execution_step_count": 2,
            "failed_step_count": 0,
            "blocked_report_count": 0,
            "plans": [
                {
                    "schema_version": "agent-core-plan/v1",
                    "plan_id": "plan-1",
                    "goal": "ship sdk",
                    "terminal": True,
                    "ready_steps": [],
                    "status_counts": {"completed": 2},
                    "steps": [
                        {"step_id": "audit", "goal": "Audit core", "status": "completed"},
                        {"step_id": "ship", "goal": "Ship sdk", "status": "completed"},
                    ],
                }
            ],
            "reports": [
                {
                    "schema_version": "agent-core-plan-execution-report/v1",
                    "status": "completed",
                    "steps": [
                        {
                            "plan_id": "plan-1",
                            "step_id": "audit",
                            "session_name": "worker",
                            "status": "completed",
                        },
                        {
                            "plan_id": "plan-1",
                            "step_id": "ship",
                            "session_name": "worker",
                            "status": "completed",
                        },
                    ],
                }
            ],
        },
        "storage_backends": {
            "schema_version": "agent-core-storage-backend-trace/v1",
            "backend_count": 3,
            "core_builtin_count": 2,
            "external_backend_count": 1,
            "roles": {"memory": 1, "run_trace": 1, "event_log": 1},
            "kinds": {"postgres": 1, "sqlite": 1, "markdown": 1},
            "backends": [
                storage_backend_manifest(
                    role="memory",
                    kind="postgres",
                    name="tenant-memory",
                    core_builtin=False,
                ),
                storage_backend_manifest(role="run_trace", kind="sqlite"),
                storage_backend_manifest(role="event_log", kind="markdown"),
            ],
        },
        "memory_governance": {
            "schema_version": "agent-core-memory-governance-trace/v1",
            "decision_count": 2,
            "allowed_count": 1,
            "denied_count": 1,
            "rewritten_count": 1,
            "decisions": [
                {
                    "decision": "rewrite",
                    "allowed": True,
                    "risk_level": "low",
                    "store": "local",
                    "reason": "memory_write_policy_rewrite",
                },
                {
                    "decision": "deny",
                    "allowed": False,
                    "risk_level": "high",
                    "store": "local",
                    "reason": "memory_target_or_secret_leak",
                },
            ],
        },
        "prompt_budget": {
            "schema_version": "agent-core-prompt-budget-plan/v1",
            "profile_max_prompt_bytes": 4096,
            "target_prompt_bytes": 900,
            "provider_limited": True,
            "provider_name": "mock",
            "model": "mock-mini",
            "context_window_tokens": 300,
            "reserved_output_tokens": 75,
            "provider_input_budget_bytes": 900,
            "source": "provider_context_window",
        },
        "prompt_bucket_budget": {
            "schema_version": "agent-core-prompt-bucket-budget-result/v1",
            "trimmed_count": 1,
            "protected_count": 1,
            "over_budget_count": 1,
            "decisions": [
                {
                    "role": "semi_dynamic_1",
                    "status": "trimmed",
                    "original_bytes": 500,
                    "final_bytes": 140,
                    "max_bytes": 160,
                },
                {
                    "role": "dynamic",
                    "status": "protected",
                    "original_bytes": 220,
                    "final_bytes": 220,
                    "max_bytes": 80,
                    "protected": True,
                },
            ],
        },
        "prompt_semantic_trim": {
            "schema_version": "agent-core-prompt-semantic-trim-result/v1",
            "target_bytes": 900,
            "original_bytes": 1180,
            "final_bytes": 820,
            "trimmed_count": 1,
            "decisions": [
                {
                    "role": "timeline_open",
                    "status": "trimmed",
                    "original_bytes": 520,
                    "final_bytes": 180,
                    "selected_units": 2,
                    "dropped_units": 3,
                },
                {
                    "role": "dynamic",
                    "status": "protected",
                    "original_bytes": 220,
                    "final_bytes": 220,
                    "protected": True,
                    "dropped_units": 0,
                },
            ],
        },
    }


def test_trace_replay_harness_combines_journal_and_event_log() -> None:
    replay = TraceReplayHarness().replay(_trace_manifest())
    manifest = replay.manifest()

    assert manifest["schema_version"] == "agent-core-trace-replay/v1"
    assert replay.run_id == "run-1"
    assert replay.event_types() == (
        "run_started",
        "run_finished",
        "tool_started",
        "tool_finished",
        "plan_created",
        "plan_execution_completed",
        "handoff_selected",
        "context_material_selection_applied",
        "prompt_budget_applied",
        "prompt_bucket_budget_applied",
        "prompt_semantic_trim_applied",
        "prompt_trim_applied",
        "mcp_inventory_partial",
        "mcp_inventory_failed",
        "skill_loaded",
        "skill_resource_view_loaded",
        "approval_approved",
        "approval_pending",
        "approval_rejected",
        "artifact_stored",
        "structured_output_repair_requested",
        "structured_output_ok",
        "provider_call_completed",
        "provider_call_completed",
    )
    assert manifest["steps"][2]["source"] == "event_log"
    assert manifest["steps"][4]["source"] == "planner_trace"
    assert manifest["steps"][4]["payload"]["plan_id"] == "plan-1"
    assert manifest["steps"][5]["source"] == "planner_trace"
    assert manifest["steps"][5]["payload"]["status"] == "completed"
    assert manifest["steps"][6]["source"] == "handoff_trace"
    assert manifest["steps"][6]["payload"]["selected_session"] == "code-reviewer"
    assert manifest["steps"][7]["source"] == "context_material_selection"
    assert manifest["steps"][7]["payload"]["selected_names"] == ["auth_trace"]
    assert manifest["steps"][8]["source"] == "prompt_budget"
    assert manifest["steps"][8]["payload"]["provider_limited"] is True
    assert manifest["steps"][9]["source"] == "prompt_bucket_budget"
    assert manifest["steps"][10]["source"] == "prompt_semantic_trim"
    assert manifest["steps"][10]["payload"]["roles"] == ["dynamic", "timeline_open"]
    assert manifest["steps"][10]["payload"]["dropped_units"] == 3
    assert manifest["steps"][12]["source"] == "mcp_center"
    assert manifest["steps"][12]["payload"]["server_name"] == "fs"
    assert manifest["steps"][14]["source"] == "skill_center"
    assert manifest["steps"][15]["payload"]["view_id"] == "review:rules.md:abcd"
    assert manifest["steps"][16]["source"] == "approval_trace"
    assert manifest["steps"][16]["payload"]["subject"] == "tool:deploy"
    assert manifest["steps"][18]["source"] == "approval_trace"
    assert manifest["steps"][19]["source"] == "artifact_trace"
    assert manifest["steps"][19]["payload"]["artifact_id"] == "artifact-1"
    assert manifest["steps"][20]["source"] == "structured_output"
    assert manifest["steps"][20]["payload"]["error"] == "$.risk is required"
    assert manifest["steps"][21]["source"] == "structured_output"
    assert manifest["steps"][21]["payload"]["schema_name"] == "risk_summary"
    assert manifest["steps"][22]["source"] == "provider"
    assert manifest["steps"][22]["payload"]["provider_name"] == "mock"
    assert manifest["steps"][22]["payload"]["request_shape_plan"]["adjusted"] is True
    assert (
        manifest["steps"][22]["payload"]["request_shape_plan"]["final_max_output_tokens"]
        == 64
    )


def test_trace_replay_harness_includes_lifecycle_hook_steps() -> None:
    trace = _trace_manifest()
    trace["session"] = {
        "lifecycle_hooks": {
            "records": [
                {
                    "hook_name": "audit",
                    "event_type": "run_starting",
                    "status": "completed",
                    "event": {
                        "schema_version": "agent-core-lifecycle-event/v1",
                        "type": "run_starting",
                        "task_bytes": 12,
                        "task_sha256": "hash",
                    },
                },
                {
                    "hook_name": "audit",
                    "event_type": "run_completed",
                    "status": "failed",
                    "error": "audit failed",
                    "event": {
                        "schema_version": "agent-core-lifecycle-event/v1",
                        "type": "run_completed",
                        "task_bytes": 12,
                        "task_sha256": "hash",
                    },
                },
            ],
        }
    }

    replay = TraceReplayHarness().replay(trace)
    lifecycle_steps = tuple(step for step in replay.steps if step.source == "lifecycle_hooks")

    assert [step.event_type for step in lifecycle_steps] == [
        "lifecycle_hook_run_starting",
        "lifecycle_hook_run_completed",
    ]
    assert lifecycle_steps[0].payload["hook_name"] == "audit"
    assert lifecycle_steps[0].payload["status"] == "completed"
    assert lifecycle_steps[1].payload["status"] == "failed"
    assert lifecycle_steps[1].payload["error"] == "audit failed"
    assert lifecycle_steps[1].payload["event"]["task_sha256"] == "hash"


def test_default_trace_evaluator_accepts_expected_trace() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            name="happy-path",
            expected_status="completed",
            max_iterations=3,
            max_provider_calls=3,
            max_cost_usd=0.05,
            required_provider_names=("mock",),
            required_provider_models=("mock-mini",),
            require_provider_model_capabilities=True,
            required_provider_model_capabilities=("structured_output", "json_mode"),
            required_event_types=("run_finished", "tool_finished"),
            required_tool_names=("lookup",),
        ),
    )

    assert report.ok
    assert report.summary["cost_usd"] == 0.03
    assert report.summary["provider_names"] == ["mock"]
    assert report.summary["provider_models"] == ["mock-mini"]
    assert report.summary["provider_model_capabilities"] == [
        "json_mode",
        "modality:text",
        "streaming",
        "structured_output",
        "tool_calls",
    ]
    assert report.summary["tool_names"] == ["lookup"]
    assert report.manifest()["schema_version"] == "agent-core-trace-eval-report/v1"


def test_default_trace_evaluator_reports_contract_failures() -> None:
    trace = _trace_manifest()
    trace["summary"] = {**trace["summary"], "journal_ok": False}

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            expected_status="failed",
            max_iterations=1,
            max_provider_calls=1,
            max_cost_usd=0.01,
            required_event_types=("approval_requested",),
            required_tool_names=("scan",),
            forbidden_event_types=("tool_finished",),
        ),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "status_mismatch",
        "iterations_exceeded",
        "provider_calls_exceeded",
        "cost_exceeded",
        "journal_not_ok",
        "missing_event_type",
        "missing_tool_name",
        "forbidden_event_type",
    } <= codes


def test_trace_eval_validates_event_log_stream_contracts() -> None:
    trace = _trace_manifest()
    trace["summary"] = {**trace["summary"], "event_log_count": 4}
    trace["event_log"] = {
        "event_count": 4,
        "events": [
            {"sequence": 1, "type": "run_started", "run_id": "run-1"},
            {"sequence": 2, "type": "model_stream", "run_id": "run-1"},
            {"sequence": 3, "type": "tool_finished", "run_id": "run-1"},
            {"sequence": 4, "type": "run_finished", "run_id": "run-1"},
        ],
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_event_log=True,
            required_event_log_types=("run_started", "run_finished"),
            require_terminal_event=True,
            require_event_sequence_monotonic=True,
            max_duplicate_event_sequences=0,
        ),
    )

    assert report.ok
    assert report.summary["event_log_types"] == [
        "model_stream",
        "run_finished",
        "run_started",
        "tool_finished",
    ]
    assert report.summary["event_log_sequence_monotonic"] is True
    assert report.summary["duplicate_event_sequence_count"] == 0
    assert report.summary["terminal_event_types"] == ["run_finished"]
    assert report.metadata["spec"]["require_event_log"] is True


def test_trace_eval_reports_event_log_stream_contract_failures() -> None:
    trace = _trace_manifest()
    trace["event_log"] = {
        "event_count": 3,
        "events": [
            {"sequence": 2, "type": "tool_finished", "run_id": "run-1"},
            {"sequence": 2, "type": "model_stream", "run_id": "run-1"},
            {"sequence": 1, "type": "run_started", "run_id": "run-1"},
        ],
    }
    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            required_event_log_types=("approval_requested",),
            forbidden_event_log_types=("tool_finished",),
            require_terminal_event=True,
            terminal_event_types=("run_timeout",),
            require_event_sequence_monotonic=True,
            max_duplicate_event_sequences=0,
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        {key: value for key, value in _trace_manifest().items() if key != "event_log"},
        TraceEvalSpec(require_event_log=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "missing_event_log_type",
        "forbidden_event_log_type",
        "terminal_event_missing",
        "event_sequence_not_monotonic",
        "duplicate_event_sequence_limit_exceeded",
    } <= codes
    assert missing.issues[0].code == "event_log_missing"


def test_trace_eval_reports_provider_capability_contract_failures() -> None:
    trace = _trace_manifest()
    trace["provider"] = {"call_count": 1, "calls": [{"provider_name": "cheap", "model": "tiny"}]}

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            required_provider_names=("mock",),
            required_provider_models=("mock-mini",),
            require_provider_model_capabilities=True,
            required_provider_model_capabilities=("structured_output",),
            forbidden_provider_model_capabilities=("streaming",),
        ),
    )
    forbidden = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(forbidden_provider_model_capabilities=("json_mode",)),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "missing_provider_name",
        "missing_provider_model",
        "provider_model_capabilities_missing",
        "missing_provider_model_capability",
    } <= codes
    assert not forbidden.ok
    assert {issue.code for issue in forbidden.issues} == {
        "forbidden_provider_model_capability"
    }


def test_trace_eval_validates_provider_stream_contracts() -> None:
    trace = {
        **_trace_manifest(),
        "provider": {
            "call_count": 1,
            "calls": [
                {
                    "provider_name": "mock",
                    "model": "mock-mini",
                    "status": "completed",
                    "streamed": True,
                    "usage": {"cost_usd": 0.01},
                    "metadata": {
                        "stream_summary": {
                            "schema_version": "agent-core-llm-stream-summary/v1",
                            "event_count": 4,
                            "event_types": ["message_start", "delta", "usage", "message_end"],
                            "delta_bytes": 5,
                            "content_bytes": 5,
                            "has_action": False,
                            "has_usage": True,
                            "finish_reason": "stop",
                            "error": "",
                        }
                    },
                }
            ],
        },
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_provider_streaming=True,
            required_provider_stream_event_types=("delta", "usage", "message_end"),
            max_provider_stream_errors=0,
        ),
    )

    assert report.ok
    assert report.summary["provider_stream_call_count"] == 1
    assert report.summary["provider_stream_summary_count"] == 1
    assert report.summary["provider_stream_event_types"] == [
        "delta",
        "message_end",
        "message_start",
        "usage",
    ]
    assert report.summary["provider_stream_error_count"] == 0


def test_trace_eval_validates_provider_native_tool_call_contracts() -> None:
    trace = {
        **_trace_manifest(),
        "provider": {
            "call_count": 2,
            "calls": [
                {
                    "provider_name": "mock",
                    "model": "mock-mini",
                    "status": "completed",
                    "streamed": False,
                    "metadata": {
                        "response": {
                            "schema_version": "agent-core-llm-response/v1",
                            "tool_call_count": 1,
                            "tool_calls": [
                                {
                                    "schema_version": "agent-core-llm-tool-call/v1",
                                    "tool_name": "lookup",
                                    "call_id": "call-1",
                                    "argument_keys": ["target"],
                                    "arguments_sha256": "hash",
                                    "metadata": {},
                                }
                            ],
                        }
                    },
                },
                {
                    "provider_name": "mock",
                    "model": "mock-mini",
                    "status": "completed",
                    "streamed": True,
                    "metadata": {
                        "stream_summary": {
                            "schema_version": "agent-core-llm-stream-summary/v1",
                            "event_types": ["tool_call", "message_end"],
                            "tool_call_count": 1,
                            "tool_calls": [
                                {
                                    "schema_version": "agent-core-llm-tool-call/v1",
                                    "tool_name": "scan",
                                    "call_id": "call-2",
                                    "argument_keys": ["url"],
                                    "arguments_sha256": "hash",
                                    "metadata": {},
                                }
                            ],
                        }
                    },
                },
            ],
        },
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_provider_tool_calls=True,
            required_provider_tool_call_names=("lookup", "scan"),
            max_provider_tool_calls=2,
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            required_provider_tool_call_names=("missing",),
            max_provider_tool_calls=1,
        ),
    )

    assert report.ok
    assert report.summary["provider_tool_call_count"] == 2
    assert report.summary["provider_tool_call_names"] == ["lookup", "scan"]
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {
        "missing_provider_tool_call",
        "provider_tool_call_limit_exceeded",
    }


def test_trace_eval_validates_provider_native_tool_result_contracts() -> None:
    trace = {
        **_trace_manifest(),
        "journal_replay": {
            "ok": True,
            "events": [
                {
                    "sequence": 1,
                    "event_type": "tool_call",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "payload": {
                        "run_id": "run-1",
                        "turn_id": "turn-1",
                        "call_id": "call-1",
                        "tool_name": "lookup",
                        "status": "completed",
                        "error": "",
                        "metadata": {
                            "provider_tool_call": {
                                "schema_version": "agent-core-llm-tool-call/v1",
                                "tool_name": "lookup",
                                "call_id": "call-1",
                                "argument_keys": ["target"],
                                "arguments_sha256": "hash",
                                "metadata": {},
                            },
                            "tool_execution": {
                                "schema_version": "agent-core-tool-execution-summary/v1",
                                "tool_name": "lookup",
                                "call_id": "call-1",
                                "attempt_count": 1,
                                "retried": False,
                                "final_status": "completed",
                                "final_ok": True,
                                "attempt_statuses": ["completed"],
                                "retryable_attempts": [],
                            },
                        },
                    },
                }
            ],
        },
    }

    replay = TraceReplayHarness().replay(trace)
    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_provider_tool_results=True,
            require_provider_tool_result_execution=True,
            required_provider_tool_result_names=("lookup",),
            required_provider_tool_result_statuses=("completed",),
            max_provider_tool_result_failures=0,
        ),
    )

    assert "provider_tool_result_recorded" in replay.event_types()
    result_step = next(
        step for step in replay.steps if step.event_type == "provider_tool_result_recorded"
    )
    assert result_step.source == "provider_tool_result"
    assert result_step.payload["tool_name"] == "lookup"
    assert result_step.payload["provider_tool_call"]["call_id"] == "call-1"
    assert report.ok
    assert report.summary["provider_tool_result_count"] == 1
    assert report.summary["provider_tool_result_names"] == ["lookup"]
    assert report.summary["provider_tool_result_statuses"] == ["completed"]
    assert report.summary["provider_tool_result_failure_count"] == 0
    assert report.summary["provider_tool_result_execution_count"] == 1


def test_trace_eval_reports_provider_native_tool_result_contract_failures() -> None:
    missing = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(require_provider_tool_results=True),
    )
    failed_trace = {
        **_trace_manifest(),
        "journal_replay": {
            "ok": True,
            "events": [
                {
                    "event_type": "tool_call",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "payload": {
                        "call_id": "call-1",
                        "tool_name": "lookup",
                        "status": "failed",
                        "error": "boom",
                        "metadata": {
                            "provider_tool_call": {
                                "tool_name": "lookup",
                                "call_id": "call-1",
                                "argument_keys": ["target"],
                                "arguments_sha256": "hash",
                            }
                        },
                    },
                }
            ],
        },
    }
    failed = DefaultTraceEvaluator().evaluate(
        failed_trace,
        TraceEvalSpec(
            require_provider_tool_result_execution=True,
            required_provider_tool_result_names=("scan",),
            required_provider_tool_result_statuses=("completed",),
            max_provider_tool_result_failures=0,
        ),
    )

    assert {issue.code for issue in missing.issues} == {"provider_tool_results_missing"}
    assert {
        "provider_tool_result_execution_missing",
        "missing_provider_tool_result",
        "missing_provider_tool_result_status",
        "provider_tool_result_failure_limit_exceeded",
    } <= {issue.code for issue in failed.issues}


def test_trace_eval_validates_provider_route_plan_contracts() -> None:
    trace = _trace_manifest()
    trace["provider"]["calls"][0]["metadata"]["route_plan"] = {
        "schema_version": "agent-core-llm-provider-route-plan/v1",
        "requested_provider": "",
        "requested_model": "",
        "streamed": False,
        "fallback_enabled": True,
        "ready": True,
        "selected_route": {
            "schema_version": "agent-core-llm-provider-route/v1",
            "provider_name": "strong",
            "model": "strong-pro",
            "metadata": {},
        },
        "candidates": [
            {
                "schema_version": "agent-core-llm-provider-route-candidate/v1",
                "provider_name": "small",
                "model": "small-mini",
                "priority": 10,
                "selected": False,
                "fallback_candidate": False,
                "supported": False,
                "reason": "unsupported_capabilities",
                "metadata": {},
            },
            {
                "schema_version": "agent-core-llm-provider-route-candidate/v1",
                "provider_name": "strong",
                "model": "strong-pro",
                "priority": 1,
                "selected": True,
                "fallback_candidate": False,
                "supported": True,
                "reason": "selected",
                "metadata": {},
            },
        ],
        "metadata": {},
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_provider_route_plan=True,
            required_provider_route_candidate_names=("small", "strong"),
            required_provider_route_selected_names=("strong",),
            forbidden_provider_route_reasons=("provider_not_registered",),
        ),
    )

    assert report.ok
    assert report.summary["provider_route_plan_count"] == 1
    assert report.summary["provider_route_candidate_names"] == ["small", "strong"]
    assert report.summary["provider_route_selected_names"] == ["strong"]
    assert report.summary["provider_route_reasons"] == [
        "selected",
        "unsupported_capabilities",
    ]


def test_trace_eval_reports_provider_route_plan_contract_failures() -> None:
    missing = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(require_provider_route_plan=True),
    )
    trace = _trace_manifest()
    trace["provider"]["calls"][0]["metadata"]["route_plan"] = {
        "schema_version": "agent-core-llm-provider-route-plan/v1",
        "ready": False,
        "candidates": [
            {
                "provider_name": "small",
                "selected": False,
                "reason": "unsupported_capabilities",
            }
        ],
    }
    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            required_provider_route_candidate_names=("strong",),
            required_provider_route_selected_names=("strong",),
            forbidden_provider_route_reasons=("unsupported_capabilities",),
        ),
    )

    assert missing.issues[0].code == "provider_route_plan_missing"
    assert {
        "missing_provider_route_candidate",
        "missing_provider_route_selected",
        "forbidden_provider_route_reason",
    } <= {issue.code for issue in report.issues}


def test_trace_eval_validates_provider_error_classification_contracts() -> None:
    trace = _trace_manifest()
    trace["provider"]["calls"][0] = {
        **trace["provider"]["calls"][0],
        "status": "failed",
        "error": "provider failed",
        "retryable": True,
        "error_classification": {
            "schema_version": "agent-core-error-classification/v1",
            "stage": "provider",
            "kind": "provider_failed",
            "status": "failed",
            "retryable": True,
            "message": "provider failed",
        },
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_provider_error_classification=True,
            required_provider_error_kinds=("provider_failed",),
            forbidden_provider_error_kinds=("unknown",),
        ),
    )
    failed = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            required_provider_error_kinds=("timeout",),
            forbidden_provider_error_kinds=("provider_failed",),
        ),
    )

    assert report.ok
    assert report.summary["provider_error_classification_count"] == 1
    assert report.summary["provider_error_kinds"] == ["provider_failed"]
    assert {
        "missing_provider_error_kind",
        "forbidden_provider_error_kind",
    } <= {issue.code for issue in failed.issues}


def test_trace_eval_validates_provider_route_preflight_contracts() -> None:
    trace = {
        **_trace_manifest(),
        "provider_route_preflight": {
            "schema_version": "agent-core-llm-provider-route-plan/v1",
            "ready": True,
            "streamed": True,
            "selected_route": {"provider_name": "strong", "model": "strong-pro"},
            "candidates": [
                {
                    "provider_name": "small",
                    "selected": False,
                    "reason": "unsupported_capabilities",
                },
                {
                    "provider_name": "strong",
                    "selected": True,
                    "reason": "selected",
                },
            ],
        },
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_provider_route_preflight=True,
            require_provider_route_preflight_ready=True,
            required_provider_route_preflight_candidate_names=("small", "strong"),
            forbidden_provider_route_preflight_reasons=("provider_not_registered",),
        ),
    )

    assert report.ok
    assert report.summary["has_provider_route_preflight"] is True
    assert report.summary["provider_route_preflight_ready"] is True
    assert report.summary["provider_route_preflight_candidate_names"] == [
        "small",
        "strong",
    ]
    assert report.summary["provider_route_preflight_reasons"] == [
        "selected",
        "unsupported_capabilities",
    ]


def test_trace_eval_reports_provider_route_preflight_contract_failures() -> None:
    missing = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(require_provider_route_preflight=True),
    )
    trace = {
        **_trace_manifest(),
        "provider_route_preflight": {
            "schema_version": "agent-core-llm-provider-route-plan/v1",
            "ready": False,
            "candidates": [
                {
                    "provider_name": "small",
                    "selected": False,
                    "reason": "unsupported_capabilities",
                }
            ],
        },
    }
    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_provider_route_preflight_ready=True,
            required_provider_route_preflight_candidate_names=("strong",),
            forbidden_provider_route_preflight_reasons=("unsupported_capabilities",),
        ),
    )

    assert missing.issues[0].code == "provider_route_preflight_missing"
    assert {
        "provider_route_preflight_not_ready",
        "missing_provider_route_preflight_candidate",
        "forbidden_provider_route_preflight_reason",
    } <= {issue.code for issue in report.issues}


def test_trace_eval_validates_failure_summary_contracts() -> None:
    trace = {
        **_trace_manifest(),
        "failure_summary": {
            "schema_version": "agent-core-failure-summary/v1",
            "failure_count": 2,
            "retryable_count": 1,
            "sources": {"provider": 1, "tool_center": 1},
            "kinds": {"provider_failed": 1, "tool_failed": 1},
            "statuses": {"failed": 2},
            "records": [
                {
                    "schema_version": "agent-core-failure-record/v1",
                    "source": "provider",
                    "kind": "provider_failed",
                    "status": "failed",
                    "retryable": True,
                },
                {
                    "schema_version": "agent-core-failure-record/v1",
                    "source": "tool_center",
                    "kind": "tool_failed",
                    "status": "failed",
                    "retryable": False,
                },
            ],
        },
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_failure_summary=True,
            required_failure_sources=("provider", "tool_center"),
            required_failure_kinds=("provider_failed", "tool_failed"),
            forbidden_failure_kinds=("unknown",),
            max_failure_count=2,
        ),
    )
    failed = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            required_failure_sources=("preflight",),
            required_failure_kinds=("timeout",),
            forbidden_failure_kinds=("tool_failed",),
            max_failure_count=1,
        ),
    )

    assert report.ok
    assert report.summary["has_failure_summary"] is True
    assert report.summary["failure_count"] == 2
    assert report.summary["failure_sources"] == ["provider", "tool_center"]
    assert report.summary["failure_kinds"] == ["provider_failed", "tool_failed"]
    assert {
        "missing_failure_source",
        "missing_failure_kind",
        "forbidden_failure_kind",
        "failure_count_limit_exceeded",
    } <= {issue.code for issue in failed.issues}


def test_trace_eval_validates_provider_request_shape_contracts() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_provider_request_shape_plan=True,
            require_provider_request_shape_adjusted=True,
            required_provider_request_shape_provider_names=("mock",),
            required_provider_request_shape_decisions=(
                "max_output_tokens_capped_to_provider_limit",
            ),
            max_provider_request_shape_final_output_tokens=64,
        ),
    )

    assert report.ok
    assert report.summary["provider_request_shape_plan_count"] == 1
    assert report.summary["provider_request_shape_adjusted_count"] == 1
    assert report.summary["provider_request_shape_provider_names"] == ["mock"]
    assert report.summary["provider_request_shape_decisions"] == [
        "max_output_tokens_capped_to_provider_limit"
    ]
    assert report.summary["provider_request_shape_max_final_output_tokens"] == 64
    assert report.metadata["spec"]["require_provider_request_shape_plan"] is True


def test_trace_eval_reports_provider_request_shape_contract_failures() -> None:
    trace = _trace_manifest()
    trace["provider"]["calls"][0]["metadata"].pop("request_shape_plan")
    trace["provider"]["calls"][0]["metadata"]["request"]["metadata"].pop(
        "request_shape_plan",
        None,
    )
    missing = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(require_provider_request_shape_plan=True),
    )
    forbidden = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            forbid_provider_request_shape_plan=True,
            forbid_provider_request_shape_adjusted=True,
            required_provider_request_shape_provider_names=("other",),
            required_provider_request_shape_decisions=("other_decision",),
            max_provider_request_shape_final_output_tokens=32,
        ),
    )

    assert {issue.code for issue in missing.issues} == {
        "provider_request_shape_plan_missing"
    }
    assert {
        "provider_request_shape_plan_forbidden",
        "provider_request_shape_adjusted_forbidden",
        "missing_provider_request_shape_provider",
        "missing_provider_request_shape_decision",
        "provider_request_shape_final_output_tokens_exceeded",
    } <= {issue.code for issue in forbidden.issues}


def test_trace_eval_validates_lifecycle_hook_contracts() -> None:
    trace = _trace_manifest()
    trace["session"] = {
        "lifecycle_hooks": {
            "schema_version": "agent-core-lifecycle-hook-center/v1",
            "record_count": 2,
            "records": [
                {
                    "schema_version": "agent-core-lifecycle-hook-record/v1",
                    "hook_name": "audit",
                    "event_type": "run_starting",
                    "status": "completed",
                    "event": {
                        "schema_version": "agent-core-lifecycle-event/v1",
                        "type": "run_starting",
                        "task_bytes": 12,
                        "task_sha256": "hash",
                    },
                },
                {
                    "schema_version": "agent-core-lifecycle-hook-record/v1",
                    "hook_name": "audit",
                    "event_type": "run_completed",
                    "status": "completed",
                    "event": {
                        "schema_version": "agent-core-lifecycle-event/v1",
                        "type": "run_completed",
                        "task_bytes": 12,
                        "task_sha256": "hash",
                    },
                },
            ],
        }
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_lifecycle_hooks=True,
            required_lifecycle_event_types=("run_starting", "run_completed"),
            required_lifecycle_hook_statuses=("completed",),
            max_lifecycle_hook_failures=0,
        ),
    )

    assert report.ok
    assert report.summary["lifecycle_hook_record_count"] == 2
    assert report.summary["lifecycle_event_types"] == ["run_completed", "run_starting"]
    assert report.summary["lifecycle_hook_statuses"] == ["completed"]
    assert report.summary["lifecycle_hook_failure_count"] == 0
    assert report.metadata["spec"]["require_lifecycle_hooks"] is True


def test_trace_eval_reports_lifecycle_hook_contract_failures() -> None:
    missing = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(require_lifecycle_hooks=True),
    )
    trace = _trace_manifest()
    trace["session"] = {
        "lifecycle_hooks": {
            "records": [
                {
                    "event_type": "run_failed",
                    "status": "failed",
                    "error": "hook failed",
                }
            ],
        }
    }
    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            required_lifecycle_event_types=("run_completed",),
            forbidden_lifecycle_event_types=("run_failed",),
            required_lifecycle_hook_statuses=("completed",),
            max_lifecycle_hook_failures=0,
        ),
    )
    codes = {issue.code for issue in report.issues}

    assert missing.issues[0].code == "lifecycle_hooks_missing"
    assert {
        "missing_lifecycle_event_type",
        "forbidden_lifecycle_event_type",
        "missing_lifecycle_hook_status",
        "lifecycle_hook_failure_limit_exceeded",
    } <= codes


def test_trace_replay_harness_includes_provider_stream_steps() -> None:
    trace = {
        **_trace_manifest(),
        "provider": {
            "call_count": 2,
            "calls": [
                {
                    "provider_name": "mock",
                    "model": "mock-mini",
                    "attempt": 1,
                    "status": "failed",
                    "streamed": True,
                    "retryable": True,
                    "error": "stream failed",
                },
                {
                    "provider_name": "mock",
                    "model": "mock-mini",
                    "attempt": 2,
                    "status": "completed",
                    "streamed": True,
                    "metadata": {
                        "stream_summary": {
                            "schema_version": "agent-core-llm-stream-summary/v1",
                            "event_types": ["delta", "message_end"],
                            "content_bytes": 6,
                        }
                    },
                },
            ],
        },
    }

    replay = TraceReplayHarness().replay(trace)
    provider_steps = [step for step in replay.steps if step.source == "provider"]

    assert [step.event_type for step in provider_steps] == [
        "provider_stream_failed",
        "provider_stream_completed",
    ]
    assert provider_steps[0].payload["retryable"] is True
    assert provider_steps[0].payload["error"] == "stream failed"
    assert provider_steps[1].payload["stream_summary"]["event_types"] == ["delta", "message_end"]


def test_trace_eval_validates_embedding_provider_contracts() -> None:
    trace = {
        **_trace_manifest(),
        "summary": {**_trace_manifest()["summary"], "embedding_call_count": 1},
        "embedding": {
            "schema_version": "agent-core-embedding-provider-center/v1",
            "call_count": 1,
            "calls": [
                {
                    "provider_name": "local",
                    "model": "embed-small",
                    "input_count": 2,
                    "dimensions": 32,
                    "status": "completed",
                    "metadata": {
                        "request": {
                            "schema_version": "agent-core-embedding-request/v1",
                            "model": "embed-small",
                            "dimensions": 32,
                            "input_count": 2,
                            "metadata": {"run_id": "run-1", "turn_id": "turn-1"},
                        }
                    },
                }
            ],
        },
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_embedding_calls=True,
            max_embedding_calls=1,
            required_embedding_provider_names=("local",),
            required_embedding_models=("embed-small",),
            required_embedding_dimensions=(32,),
            required_event_types=("embedding_call_completed",),
        ),
    )
    replay = TraceReplayHarness().replay(trace)
    embedding_steps = [step for step in replay.steps if step.source == "embedding"]

    assert report.ok
    assert report.summary["embedding_call_count"] == 1
    assert report.summary["embedding_provider_names"] == ["local"]
    assert report.summary["embedding_models"] == ["embed-small"]
    assert report.summary["embedding_dimensions"] == [32]
    assert embedding_steps[0].event_type == "embedding_call_completed"
    assert embedding_steps[0].payload["request_input_count"] == 2


def test_trace_eval_reports_embedding_contract_failures() -> None:
    trace = {
        **_trace_manifest(),
        "summary": {**_trace_manifest()["summary"], "embedding_call_count": 1},
        "embedding": {
            "call_count": 1,
            "calls": [
                {
                    "provider_name": "local",
                    "model": "embed-small",
                    "input_count": 1,
                    "dimensions": 16,
                    "status": "failed",
                }
            ],
        },
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            max_embedding_calls=0,
            required_embedding_provider_names=("remote",),
            required_embedding_models=("embed-large",),
            required_embedding_dimensions=(32,),
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(require_embedding_calls=True),
    )

    assert not report.ok
    assert {
        "embedding_calls_exceeded",
        "missing_embedding_provider_name",
        "missing_embedding_model",
        "missing_embedding_dimensions",
    } <= {issue.code for issue in report.issues}
    assert {issue.code for issue in missing.issues} == {"embedding_calls_missing"}


def test_trace_eval_reports_provider_stream_contract_failures() -> None:
    trace = {
        **_trace_manifest(),
        "provider": {
            "call_count": 2,
            "calls": [
                {
                    "provider_name": "mock",
                    "model": "mock-mini",
                    "status": "failed",
                    "streamed": True,
                    "metadata": {},
                },
                {
                    "provider_name": "mock",
                    "model": "mock-mini",
                    "status": "completed",
                    "streamed": True,
                    "metadata": {
                        "stream_summary": {
                            "schema_version": "agent-core-llm-stream-summary/v1",
                            "event_types": ["delta", "error"],
                            "error": "stream failed",
                        }
                    },
                },
            ],
        },
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            required_provider_stream_event_types=("usage",),
            forbidden_provider_stream_event_types=("error",),
            max_provider_stream_errors=0,
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(require_provider_streaming=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "missing_provider_stream_event_type",
        "forbidden_provider_stream_event_type",
        "provider_stream_error_limit_exceeded",
    } <= codes
    assert report.summary["provider_stream_error_count"] == 2
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {"provider_streaming_missing"}


def test_trace_replay_and_eval_understand_resume_manifests() -> None:
    trace = {
        **_trace_manifest(),
        "resume": {
            "schema_version": "agent-core-resume/v1",
            "run_id": "source-run",
            "turn_id": "turn-1",
            "checkpoint_id": "checkpoint-1",
            "sequence": 2,
            "state": {"step": "halfway"},
        },
        "resume_plan": {
            "schema_version": "agent-core-resume-plan-summary/v1",
            "status": "ready",
            "ready": True,
            "selected_by": "run_id",
            "candidate_count": 1,
            "run_id": "source-run",
            "checkpoint_id": "checkpoint-1",
            "checkpoint_sequence": 2,
            "terminal": False,
            "issue_count": 0,
            "issue_codes": [],
        },
    }

    replay = TraceReplayHarness().replay(trace)
    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            expected_status="completed",
            require_resume=True,
            require_resume_plan=True,
            require_resume_plan_ready=True,
            expected_resume_checkpoint_id="checkpoint-1",
            required_event_types=("resume_plan_selected", "resume_checkpoint_loaded"),
        ),
    )

    assert replay.event_types()[:2] == ("resume_plan_selected", "resume_checkpoint_loaded")
    assert report.ok
    assert report.summary["has_resume"] is True
    assert report.summary["resume_plan_ready"] is True
    assert report.summary["resume_checkpoint_id"] == "checkpoint-1"


def test_trace_eval_reports_resume_contract_failures() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_resume=True,
            require_resume_plan=True,
            require_resume_plan_ready=True,
            expected_resume_checkpoint_id="missing-checkpoint",
        ),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "resume_missing",
        "resume_plan_missing",
        "resume_plan_not_ready",
        "resume_checkpoint_mismatch",
    } <= codes


def test_trace_replay_and_eval_understand_preflight_reports() -> None:
    trace = {
        **_trace_manifest(),
        "preflight": {
            "schema_version": "agent-core-run-preflight-report/v1",
            "status": "passed",
            "ok": True,
            "issue_count": 1,
            "blocking_count": 0,
            "codes": ["empty_task"],
            "blocking_codes": [],
        },
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_preflight=True,
            require_preflight_passed=True,
            required_preflight_issue_codes=("empty_task",),
            forbidden_preflight_issue_codes=("missing_tool",),
            max_preflight_blocking_issues=0,
            required_event_types=("preflight_passed",),
        ),
    )
    replay = TraceReplayHarness().replay(trace)
    preflight_steps = [step for step in replay.steps if step.source == "preflight"]

    assert report.ok
    assert report.summary["has_preflight"] is True
    assert report.summary["preflight_status"] == "passed"
    assert report.summary["preflight_issue_codes"] == ["empty_task"]
    assert preflight_steps[0].event_type == "preflight_passed"


def test_trace_eval_reports_preflight_contract_failures() -> None:
    trace = {
        **_trace_manifest(),
        "preflight": {
            "schema_version": "agent-core-run-preflight-report/v1",
            "status": "blocked",
            "ok": False,
            "issue_count": 1,
            "blocking_count": 1,
            "codes": ["missing_tool"],
            "blocking_codes": ["missing_tool"],
        },
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_preflight_passed=True,
            required_preflight_issue_codes=("memory_required",),
            forbidden_preflight_issue_codes=("missing_tool",),
            max_preflight_blocking_issues=0,
        ),
    )
    blocked = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(require_preflight=True, require_preflight_blocked=True),
    )
    missing = DefaultTraceEvaluator().evaluate(
        {key: value for key, value in _trace_manifest().items() if key != "preflight"},
        TraceEvalSpec(require_preflight=True),
    )
    codes = {issue.code for issue in report.issues}

    assert blocked.ok
    assert not report.ok
    assert {
        "preflight_not_passed",
        "missing_preflight_issue_code",
        "forbidden_preflight_issue_code",
        "preflight_blocking_issue_limit_exceeded",
    } <= codes
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {"preflight_missing"}


def test_trace_eval_validates_tool_execution_retry_contracts() -> None:
    trace = {
        **_trace_manifest(),
        "event_log": {
            "event_count": 2,
            "events": [
                {"type": "tool_started", "run_id": "run-1", "payload": {"tool_name": "lookup"}},
                {
                    "type": "tool_finished",
                    "run_id": "run-1",
                    "payload": {
                        "tool_name": "lookup",
                        "call_id": "call-1",
                        "execution": {
                            "schema_version": "agent-core-tool-execution-summary/v1",
                            "tool_name": "lookup",
                            "call_id": "call-1",
                            "attempt_count": 2,
                            "retried": True,
                            "final_status": "completed",
                            "final_ok": True,
                            "attempt_statuses": ["failed", "completed"],
                            "retryable_attempts": [1],
                            "schema_validation": {
                                "schema_version": "agent-core-schema-validation-result/v1",
                                "ok": True,
                                "schema_name": "tool:lookup",
                                "issue_count": 0,
                                "issues": [],
                            },
                        },
                    },
                },
            ],
        },
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_tool_execution=True,
            required_tool_execution_names=("lookup",),
            required_tool_execution_ok_names=("lookup",),
            required_tool_retry_names=("lookup",),
            require_tool_schema_validation=True,
            required_tool_schema_validation_names=("lookup",),
            required_tool_schema_valid_names=("lookup",),
            min_tool_attempts={"lookup": 2},
        ),
    )
    forbidden = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(forbidden_tool_retry_names=("lookup",)),
    )

    assert report.ok
    assert report.summary["tool_execution_count"] == 1
    assert report.summary["retried_tool_names"] == ["lookup"]
    assert report.summary["max_tool_attempt_count"] == 2
    assert report.summary["tool_schema_validation_count"] == 1
    assert report.summary["tool_schema_valid_names"] == ["lookup"]
    assert not forbidden.ok
    assert {issue.code for issue in forbidden.issues} == {"forbidden_tool_retry"}


def test_trace_eval_reports_tool_execution_contract_failures() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_tool_execution=True,
            required_tool_execution_names=("lookup",),
            required_tool_execution_ok_names=("lookup",),
            required_tool_retry_names=("lookup",),
            min_tool_attempts={"lookup": 2},
        ),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "tool_execution_missing",
        "missing_tool_execution",
        "tool_execution_not_ok",
        "missing_tool_retry",
        "tool_attempts_below_minimum",
    } <= codes


def test_trace_eval_validates_tool_error_classification_contracts() -> None:
    trace = {
        **_trace_manifest(),
        "event_log": {
            "event_count": 1,
            "events": [
                {
                    "type": "tool_finished",
                    "run_id": "run-1",
                    "payload": {
                        "tool_name": "lookup",
                        "call_id": "call-1",
                        "execution": {
                            "schema_version": "agent-core-tool-execution-summary/v1",
                            "tool_name": "lookup",
                            "call_id": "call-1",
                            "attempt_count": 1,
                            "retried": False,
                            "final_status": "failed",
                            "final_ok": False,
                            "attempt_statuses": ["schema_invalid"],
                            "retryable_attempts": [],
                            "error_classification": {
                                "schema_version": "agent-core-error-classification/v1",
                                "stage": "tool",
                                "kind": "schema_invalid",
                                "status": "schema_invalid",
                                "retryable": False,
                                "message": "$.query is required",
                            },
                        },
                    },
                }
            ],
        },
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_tool_execution=True,
            require_tool_error_classification=True,
            required_tool_error_kinds=("schema_invalid",),
            forbidden_tool_error_kinds=("unknown",),
        ),
    )
    failed = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            required_tool_error_kinds=("exception",),
            forbidden_tool_error_kinds=("schema_invalid",),
        ),
    )

    assert report.ok
    assert report.summary["tool_error_classification_count"] == 1
    assert report.summary["tool_error_kinds"] == ["schema_invalid"]
    assert {
        "missing_tool_error_kind",
        "forbidden_tool_error_kind",
    } <= {issue.code for issue in failed.issues}


def test_trace_eval_reports_tool_schema_validation_contract_failures() -> None:
    trace = {
        **_trace_manifest(),
        "event_log": {
            "event_count": 1,
            "events": [
                {
                    "type": "tool_finished",
                    "run_id": "run-1",
                    "payload": {
                        "tool_name": "lookup",
                        "call_id": "call-1",
                        "execution": {
                            "schema_version": "agent-core-tool-execution-summary/v1",
                            "tool_name": "lookup",
                            "call_id": "call-1",
                            "attempt_count": 1,
                            "retried": False,
                            "final_status": "failed",
                            "final_ok": False,
                            "attempt_statuses": ["schema_invalid"],
                            "retryable_attempts": [],
                            "schema_validation": {
                                "schema_version": "agent-core-schema-validation-result/v1",
                                "ok": False,
                                "schema_name": "tool:lookup",
                                "issue_count": 1,
                                "issues": [{"code": "required_missing", "path": "$.query"}],
                            },
                        },
                    },
                }
            ],
        },
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            required_tool_schema_validation_names=("scan",),
            required_tool_schema_valid_names=("lookup",),
            forbidden_tool_schema_invalid_names=("lookup",),
            max_tool_schema_invalid=0,
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(require_tool_schema_validation=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "missing_tool_schema_validation",
        "tool_schema_not_valid",
        "forbidden_tool_schema_invalid",
        "tool_schema_invalid_limit_exceeded",
    } <= codes
    assert report.summary["tool_schema_invalid_names"] == ["lookup"]
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {"tool_schema_validation_missing"}


def test_trace_eval_validates_tool_center_contracts() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_tool_center=True,
            required_tool_center_selected_mounts=("local",),
            required_tool_center_selected_tools=("lookup",),
            required_tool_center_requested_tools=("lookup", "missing"),
            max_tool_center_failed_calls=1,
        ),
    )

    assert report.ok
    assert report.summary["has_tool_center"] is True
    assert report.summary["tool_center_call_count"] == 2
    assert report.summary["tool_center_failed_count"] == 1
    assert report.summary["tool_center_route_plan_count"] == 2
    assert report.summary["tool_center_not_ready_route_count"] == 1
    assert report.summary["tool_center_selected_mounts"] == ["local"]
    assert report.summary["tool_center_selected_tools"] == ["lookup"]
    assert report.summary["tool_center_requested_tools"] == ["lookup", "missing"]


def test_trace_eval_reports_tool_center_contract_failures() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            required_tool_center_selected_mounts=("mcp",),
            required_tool_center_selected_tools=("scan",),
            required_tool_center_requested_tools=("scan",),
            require_tool_center_ready_routes=True,
            max_tool_center_failed_calls=0,
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        {key: value for key, value in _trace_manifest().items() if key != "tool_center"},
        TraceEvalSpec(require_tool_center=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "missing_tool_center_selected_mount",
        "missing_tool_center_selected_tool",
        "missing_tool_center_requested_tool",
        "tool_center_route_not_ready",
        "tool_center_failed_call_limit_exceeded",
    } <= codes
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {"tool_center_missing"}


def test_trace_replay_and_eval_understand_agent_tool_trace() -> None:
    trace = {
        **_trace_manifest(),
        "agent_tool_trace": {
            "schema_version": "agent-core-agent-tool-trace/v1",
            "record_count": 1,
            "completed_count": 1,
            "failed_count": 0,
            "tools": {"agent_code_reviewer": 1},
            "sessions": {"code-reviewer": 1},
            "statuses": {"completed": 1},
            "records": [
                {
                    "tool_name": "agent_code_reviewer",
                    "session_name": "code-reviewer",
                    "status": "completed",
                    "run_id": "child-run",
                    "trace_run_id": "child-run",
                    "iterations": 2,
                    "task_bytes": 24,
                    "output_bytes": 64,
                    "error": "",
                }
            ],
        },
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_agent_tools=True,
            required_agent_tool_names=("agent_code_reviewer",),
            required_agent_tool_sessions=("code-reviewer",),
            required_agent_tool_statuses=("completed",),
            max_agent_tool_failures=0,
            required_event_types=("agent_tool_completed",),
        ),
    )
    replay = TraceReplayHarness().replay(trace)
    agent_tool_steps = [step for step in replay.steps if step.source == "agent_tool_trace"]

    assert report.ok
    assert report.summary["has_agent_tool_trace"] is True
    assert report.summary["agent_tool_record_count"] == 1
    assert report.summary["agent_tool_failed_count"] == 0
    assert report.summary["agent_tool_names"] == ["agent_code_reviewer"]
    assert report.summary["agent_tool_sessions"] == ["code-reviewer"]
    assert agent_tool_steps[0].event_type == "agent_tool_completed"
    assert agent_tool_steps[0].payload["session_name"] == "code-reviewer"


def test_trace_eval_reports_agent_tool_contract_failures() -> None:
    trace = {
        **_trace_manifest(),
        "agent_tool_trace": {
            "schema_version": "agent-core-agent-tool-trace/v1",
            "record_count": 1,
            "completed_count": 0,
            "failed_count": 1,
            "tools": {"agent_ops": 1},
            "sessions": {"ops": 1},
            "statuses": {"failed": 1},
            "records": [
                {
                    "tool_name": "agent_ops",
                    "session_name": "ops",
                    "status": "failed",
                    "run_id": "child-run",
                    "iterations": 1,
                    "task_bytes": 24,
                    "output_bytes": 0,
                    "error": "child failed",
                }
            ],
        },
    }

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            required_agent_tool_names=("agent_code_reviewer",),
            required_agent_tool_sessions=("code-reviewer",),
            required_agent_tool_statuses=("completed",),
            max_agent_tool_failures=0,
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        {key: value for key, value in _trace_manifest().items() if key != "agent_tool_trace"},
        TraceEvalSpec(require_agent_tools=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "missing_agent_tool_name",
        "missing_agent_tool_session",
        "missing_agent_tool_status",
        "agent_tool_failure_limit_exceeded",
    } <= codes
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {"agent_tools_missing"}


def test_trace_eval_validates_approval_contracts() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_approvals=True,
            required_approval_statuses=("approved", "pending", "rejected"),
            required_approval_subjects=("tool:deploy", "action:finish"),
            required_approval_subject_kinds=("tool", "action"),
            max_pending_approvals=1,
            max_rejected_approvals=1,
            require_approved_approval_subjects=("tool:deploy",),
        ),
    )

    assert report.ok
    assert report.summary["has_approval_trace"] is True
    assert report.summary["approval_record_count"] == 3
    assert report.summary["approval_statuses"] == ["approved", "pending", "rejected"]
    assert report.summary["approval_subject_kinds"] == ["action", "tool"]
    assert report.summary["pending_approval_count"] == 1
    assert report.summary["rejected_approval_count"] == 1
    assert report.summary["approved_approval_subjects"] == ["tool:deploy"]


def test_trace_eval_reports_approval_contract_failures() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            required_approval_statuses=("cancelled",),
            forbidden_approval_statuses=("rejected",),
            required_approval_subjects=("tool:scan",),
            required_approval_subject_kinds=("mcp",),
            max_pending_approvals=0,
            max_rejected_approvals=0,
            require_approved_approval_subjects=("action:finish",),
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        {key: value for key, value in _trace_manifest().items() if key != "approval_trace"},
        TraceEvalSpec(require_approvals=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "missing_approval_status",
        "forbidden_approval_status",
        "missing_approval_subject",
        "missing_approval_subject_kind",
        "pending_approval_limit_exceeded",
        "rejected_approval_limit_exceeded",
        "approval_subject_not_approved",
    } <= codes
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {"approval_trace_missing"}


def test_trace_eval_validates_artifact_contracts() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_artifacts=True,
            required_artifact_kinds=("tool_result",),
            required_artifact_tool_names=("dump",),
            required_artifact_content_types=("text/plain; charset=utf-8",),
            max_artifact_count=1,
            max_artifact_total_bytes=4096,
            max_artifact_size_bytes=4096,
        ),
    )

    assert report.ok
    assert report.summary["has_artifact_trace"] is True
    assert report.summary["artifact_count"] == 1
    assert report.summary["artifact_kinds"] == ["tool_result"]
    assert report.summary["artifact_tool_names"] == ["dump"]
    assert report.summary["artifact_content_types"] == ["text/plain; charset=utf-8"]
    assert report.summary["artifact_total_bytes"] == 4096
    assert report.summary["artifact_max_bytes"] == 4096


def test_trace_eval_reports_artifact_contract_failures() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            required_artifact_kinds=("memory_snapshot",),
            required_artifact_tool_names=("scan",),
            required_artifact_content_types=("application/json",),
            max_artifact_count=0,
            max_artifact_total_bytes=1024,
            max_artifact_size_bytes=1024,
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        {key: value for key, value in _trace_manifest().items() if key != "artifact_trace"},
        TraceEvalSpec(require_artifacts=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "missing_artifact_kind",
        "missing_artifact_tool_name",
        "missing_artifact_content_type",
        "artifact_count_limit_exceeded",
        "artifact_total_bytes_limit_exceeded",
        "artifact_size_limit_exceeded",
    } <= codes
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {"artifact_trace_missing"}


def test_trace_eval_validates_structured_output_contracts() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_structured_output=True,
            require_structured_output_ok=True,
            required_structured_output_schema_names=("risk_summary",),
            max_structured_output_repairs=1,
            max_structured_output_failures=1,
        ),
    )

    assert report.ok
    assert report.summary["has_structured_output_trace"] is True
    assert report.summary["structured_output_record_count"] == 2
    assert report.summary["structured_output_ok"] is True
    assert report.summary["structured_output_schema_names"] == ["risk_summary"]
    assert report.summary["structured_output_errors"] == ["$.risk is required"]
    assert report.summary["structured_output_repair_count"] == 1
    assert report.summary["structured_output_failure_count"] == 1
    assert report.metadata["spec"]["require_structured_output"] is True


def test_trace_eval_reports_structured_output_contract_failures() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            required_structured_output_schema_names=("asset_finding",),
            forbidden_structured_output_errors=("$.risk is required",),
            max_structured_output_repairs=0,
            max_structured_output_failures=0,
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        {key: value for key, value in _trace_manifest().items() if key != "structured_output_trace"},
        TraceEvalSpec(require_structured_output=True, require_structured_output_ok=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "missing_structured_output_schema_name",
        "forbidden_structured_output_error",
        "structured_output_repair_limit_exceeded",
        "structured_output_failure_limit_exceeded",
    } <= codes
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {
        "structured_output_trace_missing",
        "structured_output_ok_missing",
    }


def test_trace_eval_validates_planner_contracts() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_planner_trace=True,
            required_plan_ids=("plan-1",),
            required_plan_step_ids=("audit", "ship"),
            required_plan_step_statuses=("completed",),
            required_plan_execution_statuses=("completed",),
            max_failed_plan_steps=0,
            max_blocked_plan_reports=0,
        ),
    )

    assert report.ok
    assert report.summary["has_planner_trace"] is True
    assert report.summary["planner_plan_count"] == 1
    assert report.summary["planner_step_count"] == 2
    assert report.summary["planner_execution_report_count"] == 1
    assert report.summary["planner_execution_step_count"] == 2
    assert report.summary["planner_plan_ids"] == ["plan-1"]
    assert report.summary["planner_step_ids"] == ["audit", "ship"]
    assert report.summary["planner_step_statuses"] == ["completed"]
    assert report.summary["planner_execution_statuses"] == ["completed"]
    assert report.metadata["spec"]["require_planner_trace"] is True


def test_trace_eval_reports_planner_contract_failures() -> None:
    trace = _trace_manifest()
    planner = dict(trace["planner_trace"])
    planner["plans"] = [
        {
            "schema_version": "agent-core-plan/v1",
            "plan_id": "bad-plan",
            "goal": "blocked",
            "terminal": False,
            "ready_steps": [],
            "status_counts": {"failed": 1},
            "steps": [{"step_id": "bad-step", "goal": "Fail", "status": "failed"}],
        }
    ]
    planner["reports"] = [
        {
            "schema_version": "agent-core-plan-execution-report/v1",
            "status": "blocked",
            "steps": [
                {
                    "plan_id": "bad-plan",
                    "step_id": "bad-step",
                    "session_name": "worker",
                    "status": "failed",
                }
            ],
        }
    ]
    trace["planner_trace"] = planner

    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            required_plan_ids=("plan-1",),
            required_plan_step_ids=("audit",),
            required_plan_step_statuses=("completed",),
            required_plan_execution_statuses=("completed",),
            max_failed_plan_steps=0,
            max_blocked_plan_reports=0,
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        {key: value for key, value in _trace_manifest().items() if key != "planner_trace"},
        TraceEvalSpec(require_planner_trace=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "missing_plan_id",
        "missing_plan_step_id",
        "missing_plan_step_status",
        "missing_plan_execution_status",
        "failed_plan_step_limit_exceeded",
        "blocked_plan_report_limit_exceeded",
    } <= codes
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {"planner_trace_missing"}


def test_trace_eval_validates_mcp_and_skill_center_contracts() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_mcp_center=True,
            required_mcp_server_names=("fs", "broken"),
            required_mcp_refreshed_servers=("fs",),
            max_mcp_failed_servers=1,
            max_mcp_partial_inventory_refreshes=1,
            require_skill_center=True,
            required_loaded_skills=("review",),
            required_skill_resource_view_ids=("review:rules.md:abcd",),
            required_skill_resource_view_skills=("review",),
            max_skill_resource_views=1,
        ),
    )

    assert report.ok
    assert report.summary["has_mcp_center"] is True
    assert report.summary["mcp_server_count"] == 2
    assert report.summary["mcp_server_names"] == ["broken", "fs"]
    assert report.summary["mcp_refreshed_servers"] == ["fs"]
    assert report.summary["mcp_failed_servers"] == ["broken"]
    assert report.summary["mcp_partial_servers"] == ["fs"]
    assert report.summary["has_skill_center"] is True
    assert report.summary["loaded_skill_names"] == ["review"]
    assert report.summary["skill_resource_view_ids"] == ["review:rules.md:abcd"]


def test_trace_eval_reports_mcp_and_skill_center_contract_failures() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            required_mcp_server_names=("db",),
            required_mcp_refreshed_servers=("broken",),
            forbidden_mcp_server_statuses=("failed",),
            max_mcp_failed_servers=0,
            max_mcp_partial_inventory_refreshes=0,
            required_loaded_skills=("triage",),
            required_skill_resource_view_ids=("triage:index.md:1234",),
            required_skill_resource_view_skills=("triage",),
            max_skill_resource_views=0,
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        {
            key: value
            for key, value in _trace_manifest().items()
            if key not in {"mcp_center", "skill_center"}
        },
        TraceEvalSpec(require_mcp_center=True, require_skill_center=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "missing_mcp_server",
        "mcp_server_not_refreshed",
        "forbidden_mcp_server_status",
        "mcp_failed_server_limit_exceeded",
        "mcp_partial_inventory_refresh_limit_exceeded",
        "missing_loaded_skill",
        "missing_skill_resource_view",
        "missing_skill_resource_view_skill",
        "skill_resource_view_limit_exceeded",
    } <= codes
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {
        "mcp_center_missing",
        "skill_center_missing",
    }


def test_trace_eval_validates_storage_backend_contracts() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_storage_backends=True,
            required_storage_backend_roles=("memory", "run_trace"),
            required_storage_backend_kinds=("postgres", "sqlite"),
            max_external_storage_backends=1,
        ),
    )

    assert report.ok
    assert report.summary["storage_backend_count"] == 3
    assert report.summary["external_storage_backend_count"] == 1
    assert report.summary["storage_backend_roles"] == ["event_log", "memory", "run_trace"]
    assert report.summary["storage_backend_kinds"] == ["markdown", "postgres", "sqlite"]
    assert report.metadata["spec"]["require_storage_backends"] is True


def test_trace_eval_reports_storage_backend_contract_failures() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            required_storage_backend_roles=("approval",),
            required_storage_backend_kinds=("graph",),
            forbidden_storage_backend_kinds=("postgres",),
            forbid_external_storage_backends=True,
            max_external_storage_backends=0,
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        {key: value for key, value in _trace_manifest().items() if key != "storage_backends"},
        TraceEvalSpec(require_storage_backends=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "missing_storage_backend_role",
        "missing_storage_backend_kind",
        "forbidden_storage_backend_kind",
        "external_storage_backend_forbidden",
        "external_storage_backend_limit_exceeded",
    } <= codes
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {"storage_backends_missing"}


def test_trace_eval_validates_storage_backend_preflight_contracts() -> None:
    catalog = StorageBackendCatalog(
        (
            storage_backend_manifest(
                role="memory",
                kind="postgres",
                name="tenant-memory",
                namespace="tenant-a",
                core_builtin=False,
                capabilities=("semantic", "vector"),
            ),
            storage_backend_manifest(
                role="context_material",
                kind="vector",
                name="tenant-context",
                namespace="tenant-a",
                core_builtin=False,
                capabilities=("semantic",),
            ),
        )
    )
    preflight = catalog.preflight(
        (
            StorageBackendRequirement(
                role="memory",
                allowed_kinds=("postgres", "vector"),
                required_capabilities=("semantic", "vector"),
                namespace="tenant-a",
            ),
            StorageBackendRequirement(
                role="context_material",
                allowed_kinds=("vector", "graph"),
                required_capabilities=("semantic",),
                namespace="tenant-a",
            ),
        )
    ).manifest()
    trace = {**_trace_manifest(), "storage_backend_preflight": preflight}

    replay = TraceReplayHarness().replay(trace)
    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_storage_backend_preflight=True,
            require_storage_backend_preflight_ready=True,
            required_storage_backend_preflight_roles=("memory", "context_material"),
            forbidden_storage_backend_preflight_reasons=("missing_capabilities",),
            max_storage_backend_preflight_blocking=0,
        ),
    )

    assert report.ok
    assert "storage_backend_preflight_ready" in replay.event_types()
    assert report.summary["has_storage_backend_preflight"] is True
    assert report.summary["storage_backend_preflight_ready"] is True
    assert report.summary["storage_backend_preflight_roles"] == [
        "context_material",
        "memory",
    ]
    assert report.summary["storage_backend_preflight_blocking_count"] == 0
    assert report.metadata["spec"]["require_storage_backend_preflight"] is True


def test_trace_eval_reports_storage_backend_preflight_failures() -> None:
    catalog = StorageBackendCatalog(
        (
            storage_backend_manifest(
                role="memory",
                kind="sqlite",
                name="local-memory",
                capabilities=("keyword",),
            ),
            storage_backend_manifest(
                role="run_trace",
                kind="sqlite",
                name="local-traces",
            ),
        )
    )
    preflight = catalog.preflight(
        (
            StorageBackendRequirement(
                role="memory",
                allowed_kinds=("postgres", "vector"),
                required_capabilities=("semantic",),
            ),
            StorageBackendRequirement(
                role="run_trace",
                allowed_kinds=("postgres",),
                require_durable=True,
            ),
        )
    ).manifest()
    trace = {**_trace_manifest(), "storage_backend_preflight": preflight}
    missing = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(require_storage_backend_preflight=True),
    )
    report = DefaultTraceEvaluator().evaluate(
        trace,
        TraceEvalSpec(
            require_storage_backend_preflight_ready=True,
            required_storage_backend_preflight_roles=("approval",),
            forbidden_storage_backend_preflight_reasons=("missing_capabilities",),
            max_storage_backend_preflight_blocking=0,
        ),
    )

    assert {issue.code for issue in missing.issues} == {
        "storage_backend_preflight_missing"
    }
    assert {
        "storage_backend_preflight_not_ready",
        "missing_storage_backend_preflight_role",
        "forbidden_storage_backend_preflight_reason",
        "storage_backend_preflight_blocking_limit_exceeded",
    } <= {issue.code for issue in report.issues}


def test_trace_eval_validates_handoff_contracts() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_handoff=True,
            required_handoff_statuses=("selected",),
            required_handoff_selected_sessions=("code-reviewer",),
            required_handoff_source_sessions=("planner",),
            max_handoff_denied=0,
            max_handoff_not_found=0,
        ),
    )

    assert report.ok
    assert report.summary["has_handoff"] is True
    assert report.summary["handoff_record_count"] == 1
    assert report.summary["handoff_statuses"] == ["selected"]
    assert report.summary["handoff_selected_sessions"] == ["code-reviewer"]
    assert report.summary["handoff_source_sessions"] == ["planner"]
    assert report.metadata["spec"]["require_handoff"] is True


def test_trace_eval_reports_handoff_contract_failures() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            required_handoff_statuses=("denied",),
            required_handoff_selected_sessions=("ops",),
            required_handoff_source_sessions=("scheduler",),
            max_handoff_denied=-1,
            max_handoff_not_found=-1,
        ),
    )
    missing_trace = _trace_manifest()
    missing_prompt = dict(missing_trace["prompt"])
    missing_metadata = dict(missing_prompt.get("metadata") or {})
    missing_metadata.pop("handoff", None)
    missing_prompt["metadata"] = missing_metadata
    missing_trace["prompt"] = missing_prompt
    missing = DefaultTraceEvaluator().evaluate(
        missing_trace,
        TraceEvalSpec(require_handoff=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "missing_handoff_status",
        "missing_handoff_selected_session",
        "missing_handoff_source_session",
        "handoff_denied_limit_exceeded",
        "handoff_not_found_limit_exceeded",
    } <= codes
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {"handoff_missing"}


def test_trace_eval_validates_context_injection_contracts() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_context_injections=True,
            required_context_injection_names=("memory_recall", "operator_hint"),
            required_context_injection_sources=("memory", "runtime"),
            required_context_injection_targets=("semi_dynamic_1", "dynamic"),
            required_context_injection_statuses=("trimmed", "included", "target_denied"),
            required_included_context_injection_sources=("memory", "runtime"),
            required_trimmed_context_injection_sources=("memory",),
            required_excluded_context_injection_sources=("runtime",),
            max_trimmed_context_injections=1,
            max_excluded_context_injections=1,
        ),
    )

    assert report.ok
    assert report.summary["context_injection_count"] == 3
    assert report.summary["context_injection_names"] == [
        "denied_static",
        "memory_recall",
        "operator_hint",
    ]
    assert report.summary["trimmed_context_injection_count"] == 1
    assert report.summary["excluded_context_injection_count"] == 1
    assert report.summary["context_injection_sources"] == ["memory", "runtime"]
    assert report.summary["context_injection_statuses"] == [
        "included",
        "target_denied",
        "trimmed",
    ]
    assert report.summary["included_context_injection_sources"] == ["memory", "runtime"]
    assert report.summary["trimmed_context_injection_sources"] == ["memory"]
    assert report.summary["excluded_context_injection_sources"] == ["runtime"]
    assert report.metadata["spec"]["require_context_injections"] is True


def test_trace_eval_reports_context_injection_contract_failures() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            required_context_injection_names=("missing_context",),
            required_context_injection_sources=("approval",),
            required_context_injection_targets=("timeline_open",),
            required_context_injection_statuses=("budget_exceeded",),
            forbidden_context_injection_statuses=("target_denied",),
            required_included_context_injection_sources=("approval",),
            required_trimmed_context_injection_sources=("runtime",),
            required_excluded_context_injection_sources=("memory",),
            forbidden_context_injection_sources=("runtime",),
            forbid_trimmed_context_injections=True,
            max_trimmed_context_injections=0,
            max_excluded_context_injections=0,
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        {key: value for key, value in _trace_manifest().items() if key != "prompt"},
        TraceEvalSpec(require_context_injections=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "missing_context_injection_name",
        "missing_context_injection_source",
        "missing_context_injection_target",
        "missing_context_injection_status",
        "forbidden_context_injection_status",
        "missing_included_context_injection_source",
        "missing_trimmed_context_injection_source",
        "missing_excluded_context_injection_source",
        "forbidden_context_injection_source",
        "context_injection_trimmed_forbidden",
        "context_injection_trimmed_limit_exceeded",
        "context_injection_excluded_limit_exceeded",
    } <= codes
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {"context_injections_missing"}


def test_trace_eval_validates_context_material_selection_contracts() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_context_material_selection=True,
            required_selected_context_material_names=("auth_trace",),
            required_context_material_statuses=("selected", "count_exceeded"),
            required_context_material_targets=("timeline_open", "semi_dynamic_2"),
            max_dropped_context_materials=2,
            max_selected_context_material_bytes=64,
        ),
    )

    assert report.ok
    assert report.summary["has_context_material_selection"] is True
    assert report.summary["context_material_selection_count"] == 3
    assert report.summary["selected_context_material_count"] == 1
    assert report.summary["dropped_context_material_count"] == 2
    assert report.summary["selected_context_material_names"] == ["auth_trace"]
    assert report.summary["context_material_statuses"] == [
        "count_exceeded",
        "score_below_threshold",
        "selected",
    ]
    assert report.summary["context_material_targets"] == [
        "semi_dynamic_1",
        "semi_dynamic_2",
        "timeline_open",
    ]
    assert report.summary["selected_context_material_bytes"] == 64
    assert report.metadata["spec"]["require_context_material_selection"] is True


def test_trace_eval_reports_context_material_selection_contract_failures() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            required_selected_context_material_names=("schema_hint",),
            required_context_material_statuses=("budget_exceeded",),
            forbidden_context_material_statuses=("score_below_threshold",),
            required_context_material_targets=("dynamic",),
            max_dropped_context_materials=1,
            max_selected_context_material_bytes=32,
        ),
    )
    missing_trace = _trace_manifest()
    missing_prompt = dict(missing_trace["prompt"])
    missing_metadata = dict(missing_prompt.get("metadata") or {})
    missing_metadata.pop("context_material_selection", None)
    missing_prompt["metadata"] = missing_metadata
    missing_trace["prompt"] = missing_prompt
    missing = DefaultTraceEvaluator().evaluate(
        missing_trace,
        TraceEvalSpec(require_context_material_selection=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "missing_selected_context_material_name",
        "missing_context_material_status",
        "forbidden_context_material_status",
        "missing_context_material_target",
        "context_material_dropped_limit_exceeded",
        "context_material_selected_bytes_exceeded",
    } <= codes
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {"context_material_selection_missing"}


def test_trace_eval_validates_prompt_trim_contracts() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_prompt_trim=True,
            required_prompt_trim_roles=("timeline_open",),
            max_prompt_trim_original_bytes=1200,
            max_prompt_trim_final_bytes=900,
        ),
    )

    assert report.ok
    assert report.summary["has_prompt_trim"] is True
    assert report.summary["prompt_trim_roles"] == ["timeline_open"]
    assert report.summary["prompt_trim_original_bytes"] == 1200
    assert report.summary["prompt_trim_final_bytes"] == 840
    assert report.metadata["spec"]["require_prompt_trim"] is True


def test_trace_eval_reports_prompt_trim_contract_failures() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            forbid_prompt_trim=True,
            required_prompt_trim_roles=("semi_dynamic_1",),
            forbidden_prompt_trim_roles=("timeline_open",),
            max_prompt_trim_original_bytes=1000,
            max_prompt_trim_final_bytes=800,
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        {key: value for key, value in _trace_manifest().items() if key != "prompt"},
        TraceEvalSpec(require_prompt_trim=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "prompt_trim_forbidden",
        "missing_prompt_trim_role",
        "forbidden_prompt_trim_role",
        "prompt_trim_original_bytes_exceeded",
        "prompt_trim_final_bytes_exceeded",
    } <= codes
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {"prompt_trim_missing"}


def test_trace_eval_validates_memory_governance_contracts() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_memory_governance=True,
            required_memory_governance_decisions=("rewrite", "deny"),
            max_denied_memory_writes=1,
            max_rewritten_memory_writes=1,
            max_high_risk_memory_writes=1,
        ),
    )

    assert report.ok
    assert report.summary["memory_governance_decision_count"] == 2
    assert report.summary["memory_governance_decisions"] == ["deny", "rewrite"]
    assert report.summary["denied_memory_write_count"] == 1
    assert report.summary["rewritten_memory_write_count"] == 1
    assert report.summary["high_risk_memory_write_count"] == 1
    assert report.metadata["spec"]["require_memory_governance"] is True


def test_trace_eval_reports_memory_governance_contract_failures() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            required_memory_governance_decisions=("allow",),
            forbidden_memory_governance_decisions=("deny",),
            max_denied_memory_writes=0,
            max_rewritten_memory_writes=0,
            max_high_risk_memory_writes=0,
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        {key: value for key, value in _trace_manifest().items() if key != "memory_governance"},
        TraceEvalSpec(require_memory_governance=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "missing_memory_governance_decision",
        "forbidden_memory_governance_decision",
        "denied_memory_write_limit_exceeded",
        "rewritten_memory_write_limit_exceeded",
        "high_risk_memory_write_limit_exceeded",
    } <= codes
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {"memory_governance_missing"}


def test_trace_eval_validates_prompt_budget_contracts() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_prompt_budget=True,
            require_prompt_budget_provider_limited=True,
            required_prompt_budget_sources=("provider_context_window",),
            required_prompt_budget_provider_names=("mock",),
            max_prompt_budget_target_bytes=900,
            max_prompt_budget_provider_input_bytes=900,
        ),
    )

    assert report.ok
    assert report.summary["has_prompt_budget"] is True
    assert report.summary["prompt_budget_source"] == "provider_context_window"
    assert report.summary["prompt_budget_provider_name"] == "mock"
    assert report.summary["prompt_budget_provider_limited"] is True
    assert report.summary["prompt_budget_target_bytes"] == 900
    assert report.summary["prompt_budget_provider_input_bytes"] == 900
    assert report.metadata["spec"]["require_prompt_budget"] is True


def test_trace_eval_reports_prompt_budget_contract_failures() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            forbid_prompt_budget=True,
            forbid_prompt_budget_provider_limited=True,
            required_prompt_budget_sources=("profile_budget",),
            required_prompt_budget_provider_names=("other",),
            max_prompt_budget_target_bytes=800,
            max_prompt_budget_provider_input_bytes=800,
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        {key: value for key, value in _trace_manifest().items() if key != "prompt_budget"},
        TraceEvalSpec(require_prompt_budget=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "prompt_budget_forbidden",
        "prompt_budget_provider_limit_forbidden",
        "missing_prompt_budget_source",
        "missing_prompt_budget_provider",
        "prompt_budget_target_bytes_exceeded",
        "prompt_budget_provider_input_bytes_exceeded",
    } <= codes
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {"prompt_budget_missing"}


def test_trace_eval_validates_prompt_bucket_budget_contracts() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_prompt_bucket_budget=True,
            required_prompt_bucket_budget_roles=("semi_dynamic_1", "dynamic"),
            required_prompt_bucket_budget_statuses=("trimmed", "protected"),
            max_prompt_bucket_budget_trimmed=1,
            max_prompt_bucket_budget_over_budget=1,
        ),
    )

    assert report.ok
    assert report.summary["has_prompt_bucket_budget"] is True
    assert report.summary["prompt_bucket_budget_roles"] == ["dynamic", "semi_dynamic_1"]
    assert report.summary["prompt_bucket_budget_statuses"] == ["protected", "trimmed"]
    assert report.summary["prompt_bucket_budget_trimmed_count"] == 1
    assert report.summary["prompt_bucket_budget_over_budget_count"] == 1
    assert report.metadata["spec"]["require_prompt_bucket_budget"] is True


def test_trace_eval_reports_prompt_bucket_budget_contract_failures() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            required_prompt_bucket_budget_roles=("timeline_open",),
            required_prompt_bucket_budget_statuses=("within_budget",),
            forbidden_prompt_bucket_budget_statuses=("protected",),
            forbid_over_budget_prompt_buckets=True,
            max_prompt_bucket_budget_trimmed=0,
            max_prompt_bucket_budget_over_budget=0,
        ),
    )
    missing = DefaultTraceEvaluator().evaluate(
        {key: value for key, value in _trace_manifest().items() if key != "prompt_bucket_budget"},
        TraceEvalSpec(require_prompt_bucket_budget=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "missing_prompt_bucket_budget_role",
        "missing_prompt_bucket_budget_status",
        "forbidden_prompt_bucket_budget_status",
        "prompt_bucket_budget_over_budget_forbidden",
        "prompt_bucket_budget_trimmed_limit_exceeded",
        "prompt_bucket_budget_over_budget_limit_exceeded",
    } <= codes
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {"prompt_bucket_budget_missing"}


def test_trace_eval_validates_prompt_semantic_trim_contracts() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            require_prompt_semantic_trim=True,
            required_prompt_semantic_trim_roles=("timeline_open", "dynamic"),
            required_prompt_semantic_trim_statuses=("trimmed", "protected"),
            max_prompt_semantic_trimmed=1,
            max_prompt_semantic_dropped_units=3,
            max_prompt_semantic_trim_original_bytes=1200,
            max_prompt_semantic_trim_final_bytes=900,
        ),
    )

    assert report.ok
    assert report.summary["has_prompt_semantic_trim"] is True
    assert report.summary["prompt_semantic_trim_roles"] == ["dynamic", "timeline_open"]
    assert report.summary["prompt_semantic_trim_statuses"] == ["protected", "trimmed"]
    assert report.summary["prompt_semantic_trimmed_count"] == 1
    assert report.summary["prompt_semantic_dropped_units"] == 3
    assert report.summary["prompt_semantic_trim_original_bytes"] == 1180
    assert report.summary["prompt_semantic_trim_final_bytes"] == 820
    assert report.metadata["spec"]["require_prompt_semantic_trim"] is True


def test_trace_eval_reports_prompt_semantic_trim_contract_failures() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            forbid_prompt_semantic_trim=True,
            required_prompt_semantic_trim_roles=("semi_dynamic_1",),
            required_prompt_semantic_trim_statuses=("unchanged",),
            forbidden_prompt_semantic_trim_statuses=("protected",),
            max_prompt_semantic_trimmed=0,
            max_prompt_semantic_dropped_units=2,
            max_prompt_semantic_trim_original_bytes=1000,
            max_prompt_semantic_trim_final_bytes=800,
        ),
    )
    missing_trace = _trace_manifest()
    missing_trace.pop("prompt_semantic_trim", None)
    missing_prompt = dict(missing_trace["prompt"])
    missing_metadata = dict(missing_prompt.get("metadata") or {})
    missing_metadata.pop("semantic_trim", None)
    missing_prompt["metadata"] = missing_metadata
    missing_trace["prompt"] = missing_prompt
    missing = DefaultTraceEvaluator().evaluate(
        missing_trace,
        TraceEvalSpec(require_prompt_semantic_trim=True),
    )
    codes = {issue.code for issue in report.issues}

    assert not report.ok
    assert {
        "prompt_semantic_trim_forbidden",
        "missing_prompt_semantic_trim_role",
        "missing_prompt_semantic_trim_status",
        "forbidden_prompt_semantic_trim_status",
        "prompt_semantic_trimmed_limit_exceeded",
        "prompt_semantic_dropped_units_exceeded",
        "prompt_semantic_trim_original_bytes_exceeded",
        "prompt_semantic_trim_final_bytes_exceeded",
    } <= codes
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {"prompt_semantic_trim_missing"}


def test_trace_replay_comparator_accepts_matching_trace() -> None:
    trace = _trace_manifest()
    report = TraceReplayComparator().compare(trace, trace)

    assert report.ok
    assert report.summary["baseline_step_count"] == 24
    assert report.manifest()["schema_version"] == "agent-core-trace-replay-diff-report/v1"
    assert report.metadata["spec"]["schema_version"] == "agent-core-trace-replay-diff-spec/v1"


def test_trace_replay_comparator_reports_ordered_differences() -> None:
    baseline = _trace_manifest()
    actual = _trace_manifest()
    actual["run"] = {**actual["run"], "status": "failed"}
    actual["summary"] = {**actual["summary"], "provider_call_count": 3}
    actual["event_log"] = {
        **actual["event_log"],
        "events": [
            {"type": "tool_finished", "run_id": "run-2", "payload": {"tool_name": "lookup"}},
            {"type": "tool_started", "run_id": "run-2", "payload": {"tool_name": "lookup"}},
        ],
    }

    report = TraceReplayComparator().compare(baseline, actual)
    codes = [issue.code for issue in report.issues]

    assert not report.ok
    assert "run_field_mismatch" in codes
    assert "summary_mismatch" in codes
    assert "step_mismatch" in codes
    assert report.issues[-1].index == 3
    assert report.summary["actual_event_types"] == [
        "run_started",
        "run_finished",
        "tool_finished",
        "tool_started",
        "plan_created",
        "plan_execution_completed",
        "handoff_selected",
        "context_material_selection_applied",
        "prompt_budget_applied",
        "prompt_bucket_budget_applied",
        "prompt_semantic_trim_applied",
        "prompt_trim_applied",
        "mcp_inventory_partial",
        "mcp_inventory_failed",
        "skill_loaded",
        "skill_resource_view_loaded",
        "approval_approved",
        "approval_pending",
        "approval_rejected",
        "artifact_stored",
        "structured_output_repair_requested",
        "structured_output_ok",
        "provider_call_completed",
        "provider_call_completed",
    ]


def test_trace_replay_comparator_can_ignore_order_and_payloads() -> None:
    baseline = _trace_manifest()
    actual = _trace_manifest()
    actual["event_log"] = {
        **actual["event_log"],
        "events": [
            {"type": "tool_finished", "run_id": "run-2", "payload": {"tool_name": "other"}},
            {"type": "tool_started", "run_id": "run-2", "payload": {"tool_name": "other"}},
        ],
    }

    unordered = TraceReplayComparator().compare(
        baseline,
        actual,
        TraceReplayDiffSpec(compare_event_order=False),
    )
    with_payloads = TraceReplayComparator().compare(
        baseline,
        actual,
        TraceReplayDiffSpec(compare_event_order=False, compare_payloads=True),
    )

    assert unordered.ok
    assert not with_payloads.ok
    assert {issue.code for issue in with_payloads.issues} == {
        "missing_step_count",
        "unexpected_step_count",
    }


@pytest.mark.asyncio
async def test_trace_eval_harness_evaluates_stored_agent_runner_trace() -> None:
    trace_store = InMemoryRunTraceStore()
    provider = MockLLMProvider(
        [
            {
                "action": "call_tool",
                "arguments": {"tool_name": "lookup", "arguments": {"query": "target"}},
            },
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    center = LLMProviderCenter(default_provider="mock")
    center.register(
        "mock",
        provider,
        default_model="mock-mini",
        default_capabilities=LLMModelCapabilities(
            supports_tool_calls=True,
            supports_structured_output=True,
            supports_json_mode=True,
        ),
    )
    session = AgentSession(
        profile=AgentProfile(name="eval", model="mock-mini", budget=RuntimeBudget(max_iterations=4)),
        provider=center,
        tools=MockToolRuntime({"lookup": "found"}),
        tool_replay=InMemoryToolReplay(),
        trace_store=trace_store,
        event_sink=ListEventSink(),
    )

    outcome = await AgentRunner(session).run("inspect")
    harness = TraceEvalHarness(trace_store=trace_store)
    report = await harness.evaluate_run(
        outcome.result.run_id,
        TraceEvalSpec(
            expected_status="completed",
            max_iterations=3,
            required_provider_names=("mock",),
            required_provider_models=("mock-mini",),
            require_provider_model_capabilities=True,
            required_provider_model_capabilities=("structured_output", "json_mode"),
            required_event_types=("run_finished", "tool_finished"),
            required_tool_names=("lookup",),
        ),
    )

    assert report.ok
    assert report.run_id == outcome.result.run_id
    assert report.summary["provider_call_count"] == 2
    assert report.summary["provider_names"] == ["mock"]
    assert "structured_output" in report.summary["provider_model_capabilities"]
    assert report.replay["step_count"] >= 4
    assert harness.manifest()["schema_version"] == "agent-core-trace-eval-harness/v1"


@pytest.mark.asyncio
async def test_trace_eval_harness_validates_stored_resume_runner_trace() -> None:
    trace_store = InMemoryRunTraceStore()
    journal = InMemoryAgentJournal()
    original_run = await journal.start_run("original")
    original_turn = await journal.start_turn(original_run, 0)
    checkpoint = await journal.checkpoint(original_turn, {"step": "resume"})
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "resumed"}}])
    session = AgentSession(
        profile=AgentProfile(name="resume-eval"),
        provider=provider,
        tools=MockToolRuntime(),
        harness=journal,
        trace_store=trace_store,
        event_sink=ListEventSink(),
    )

    outcome = await AgentRunner(session).resume("continue")
    harness = TraceEvalHarness(trace_store=trace_store)
    report = await harness.evaluate_run(
        outcome.result.run_id,
        TraceEvalSpec(
            expected_status="completed",
            require_resume=True,
            require_resume_plan=True,
            require_resume_plan_ready=True,
            expected_resume_checkpoint_id=checkpoint.checkpoint_id,
            required_event_types=("resume_plan_selected", "resume_checkpoint_loaded", "run_finished"),
        ),
    )

    assert report.ok
    assert outcome.trace_manifest["summary"]["has_resume_plan"] is True
    assert outcome.trace_manifest["summary"]["resume_plan_ready"] is True
    assert report.summary["resume_plan_checkpoint_id"] == checkpoint.checkpoint_id


@pytest.mark.asyncio
async def test_trace_eval_harness_validates_stored_tool_retry_trace() -> None:
    attempts = 0
    registry = ToolRegistry()

    async def lookup(invocation: ToolInvocation) -> ToolResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ToolResult(
                call_id=invocation.call_id,
                tool_name=invocation.tool_name,
                status="failed",
                error="temporary",
                metadata={"retryable": True},
            )
        return ToolResult(call_id=invocation.call_id, tool_name=invocation.tool_name, content="found")

    registry.register(ToolSpec(name="lookup"), lookup)
    trace_store = InMemoryRunTraceStore()
    session = AgentSession(
        profile=AgentProfile(name="tool-retry-eval", budget=RuntimeBudget(max_iterations=3)),
        provider=MockLLMProvider(
            [
                {"action": "call_tool", "arguments": {"tool_name": "lookup", "arguments": {}}},
                {"action": "finish", "arguments": {"output": "done"}},
            ]
        ),
        tools=ToolExecutionCenter(registry, retry_policy=ToolRetryPolicy(max_attempts=2)),
        trace_store=trace_store,
        event_sink=ListEventSink(),
    )

    outcome = await AgentRunner(session).run("retry lookup")
    report = await TraceEvalHarness(trace_store=trace_store).evaluate_run(
        outcome.result.run_id,
        TraceEvalSpec(
            expected_status="completed",
            require_tool_execution=True,
            required_tool_execution_names=("lookup",),
            required_tool_execution_ok_names=("lookup",),
            required_tool_retry_names=("lookup",),
            min_tool_attempts={"lookup": 2},
        ),
    )

    assert report.ok
    assert attempts == 2
    assert report.summary["retried_tool_names"] == ["lookup"]
