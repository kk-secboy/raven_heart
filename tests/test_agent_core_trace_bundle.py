from __future__ import annotations

import pytest

from agent_core.config import AgentProfile
from agent_core.events import ListEventSink
from agent_core.harness import InMemoryAgentJournal, InMemoryJournalStore, PersistentAgentJournal
from agent_core.memory import InMemoryMemoryStore
from agent_core.policy import InMemoryPolicyDecisionStore
from agent_core.providers import LLMProviderCenter
from agent_core.runner import AgentRunner, AgentSession
from agent_core.testing import MockLLMProvider
from agent_core.tools import (
    InMemoryToolReplayStore,
    PersistentToolReplay,
    ToolCenter,
    ToolInvocation,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from agent_core.trace import (
    AgentRunTraceBundle,
    AgentToolTrace,
    ApprovalTrace,
    ContextInjectionTrace,
    ContextMaterialSelectionTrace,
    HandoffTrace,
    InMemoryRunTraceStore,
    MarkdownRunTraceStore,
    MCPCenterTrace,
    MemoryGovernanceTrace,
    PlannerTrace,
    RunTraceQuery,
    SQLiteRunTraceStore,
    SkillCenterTrace,
    StorageBackendTrace,
    StructuredOutputTrace,
    TraceCorrelationIndex,
)
from agent_core.artifacts import InMemoryArtifactStore
from agent_core.backends import storage_backend_manifest


async def _lookup(invocation: ToolInvocation) -> ToolResult:
    return ToolResult(call_id=invocation.call_id, tool_name=invocation.tool_name, content="found")


def test_agent_run_trace_bundle_summarizes_core_manifests() -> None:
    bundle = AgentRunTraceBundle(
        run_id="run-1",
        status="completed",
        iterations=2,
        output_bytes=4,
        journal_replay={
            "ok": True,
            "event_count": 7,
            "events": [
                {
                    "event_type": "checkpoint",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "payload": {
                        "sequence": 1,
                        "state": {
                            "status": "structured_output_error",
                            "error": "$.risk is required",
                            "repair_attempt": 1,
                            "iteration": 0,
                        },
                    },
                },
                {
                    "event_type": "checkpoint",
                    "run_id": "run-1",
                    "turn_id": "turn-2",
                    "payload": {
                        "sequence": 2,
                        "state": {
                            "status": "finished",
                            "iteration": 1,
                            "structured_output": {
                                "ok": True,
                                "raw_output_bytes": 28,
                                "metadata": {
                                    "schema_name": "risk_summary",
                                    "schema_validation": {
                                        "schema_name": "risk_summary",
                                        "ok": True,
                                    },
                                },
                            },
                        },
                    },
                },
            ],
        },
        provider={"call_count": 2},
        tool_replay={"record_count": 1},
        policy_decisions={"record_count": 2},
        approvals={
            "record_count": 3,
            "records": [
                {
                    "approval_id": "approval-1",
                    "status": "approved",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "request": {"subject": "tool:deploy", "reason": "deploy gate"},
                    "decision": {"status": "approved", "actor": "operator", "reason": "ok"},
                },
                {
                    "approval_id": "approval-2",
                    "status": "pending",
                    "run_id": "run-1",
                    "turn_id": "turn-2",
                    "request": {"subject": "action:finish", "reason": "final gate"},
                    "decision": None,
                },
                {
                    "approval_id": "approval-3",
                    "status": "rejected",
                    "run_id": "run-1",
                    "turn_id": "turn-3",
                    "request": {"subject": "tool:delete", "reason": "dangerous"},
                    "decision": {"status": "rejected", "actor": "operator", "reason": "no"},
                },
            ],
        },
        event_log={"event_count": 9},
        resume={"checkpoint_id": "c1"},
        resume_plan={"ready": True, "checkpoint_id": "c1"},
        timeline_reduction={"compressed_bytes": 10},
        capability_discovery={"match_count": 4},
        memory_search={"hit_count": 2},
        planner_trace={
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
                        {"step_id": "audit", "goal": "Audit", "status": "completed"},
                        {"step_id": "ship", "goal": "Ship", "status": "completed"},
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
        session={
            "capabilities": {
                "mcp": {
                    "schema_version": "agent-core-mcp-center/v1",
                    "servers": [
                        {
                            "name": "fs",
                            "transport": "stdio",
                            "enabled": True,
                            "state": {
                                "status": "refreshed",
                                "tool_count": 1,
                                "resource_count": 1,
                                "prompt_count": 1,
                            },
                        }
                    ],
                    "tools": [{"name": "mcp__fs__read_file"}],
                    "resources": [{"server_name": "fs", "uri": "file://README.md"}],
                    "prompts": [{"server_name": "fs", "name": "summarize"}],
                    "last_inventory_refresh": [
                        {
                            "server_name": "fs",
                            "status": "refreshed",
                            "ok": True,
                            "tool_count": 1,
                            "resource_count": 1,
                            "prompt_count": 1,
                        }
                    ],
                },
                "skills": {
                    "schema_version": "agent-core-skills-context/v1",
                    "available_skills_count": 2,
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
                },
                "artifact_store": {
                    "schema_version": "agent-core-artifact-store/v1",
                    "artifacts": [
                        {
                            "artifact_id": "artifact-1",
                            "uri": "artifact://artifact-1",
                            "content_type": "text/plain; charset=utf-8",
                            "size_bytes": 4096,
                            "sha256": "abc123",
                            "metadata": {
                                "kind": "tool_result",
                                "tool_name": "dump",
                                "call_id": "call-1",
                                "status": "completed",
                            },
                        }
                    ],
                },
                "memory": {
                "governance": {
                    "decisions": [
                        {
                            "decision": "rewrite",
                            "allowed": True,
                            "risk_level": "low",
                            "store": "local",
                        },
                        {
                            "decision": "deny",
                            "allowed": False,
                            "risk_level": "high",
                            "store": "local",
                        },
                    ]
                }
            }
        },
        prompt={
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
                        "metadata": {},
                    },
                    "candidates": [
                        {
                            "schema_version": "agent-core-handoff-spec/v1",
                            "session_name": "code-reviewer",
                            "tags": ["code", "review"],
                            "tools": ["diff"],
                            "skills": ["review"],
                            "priority": 5,
                            "enabled": True,
                        }
                    ],
                },
                "trim": {"target_bytes": 900},
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
                ],
                "context_material_selection": {
                    "schema_version": "agent-core-context-material-selection-result/v1",
                    "selection_count": 2,
                    "selected_count": 1,
                    "dropped_count": 1,
                    "selected_bytes": 64,
                    "selections": [
                        {
                            "name": "memory_recall",
                            "role": "memory",
                            "target": "semi_dynamic_1",
                            "status": "selected",
                            "selected": True,
                            "score": 101.0,
                            "rank": 1,
                            "reason": "semantic_priority_selected",
                            "priority": 4,
                            "bytes": 64,
                            "sha256": "memsha",
                        },
                        {
                            "name": "old_note",
                            "role": "memory",
                            "target": "semi_dynamic_1",
                            "status": "score_below_threshold",
                            "selected": False,
                            "score": 0.2,
                            "rank": 0,
                            "reason": "score_below_threshold",
                            "priority": 1,
                            "bytes": 20,
                            "sha256": "oldsha",
                        },
                    ],
                },
                "bucket_budget": {
                    "schema_version": "agent-core-prompt-bucket-budget-result/v1",
                    "trimmed_count": 1,
                    "over_budget_count": 0,
                    "decisions": [
                        {
                            "role": "semi_dynamic_1",
                            "status": "trimmed",
                            "original_bytes": 500,
                            "final_bytes": 140,
                        }
                    ],
                },
            }
        },
    )

    manifest = bundle.manifest()

    assert manifest["schema_version"] == "agent-core-run-trace-bundle/v1"
    assert manifest["run"]["run_id"] == "run-1"
    assert manifest["summary"]["journal_ok"] is True
    assert manifest["summary"]["journal_event_count"] == 7
    assert manifest["summary"]["provider_call_count"] == 2
    assert manifest["summary"]["tool_replay_record_count"] == 1
    assert manifest["summary"]["mcp_server_count"] == 1
    assert manifest["summary"]["mcp_failed_server_count"] == 0
    assert manifest["summary"]["mcp_partial_inventory_refresh_count"] == 0
    assert manifest["summary"]["skill_loaded_count"] == 1
    assert manifest["summary"]["skill_resource_view_count"] == 1
    assert manifest["mcp_center"]["refreshed_servers"] == ["fs"]
    assert manifest["skill_center"]["loaded_skill_names"] == ["review"]
    assert manifest["summary"]["policy_decision_record_count"] == 2
    assert manifest["summary"]["approval_record_count"] == 3
    assert manifest["summary"]["approval_pending_count"] == 1
    assert manifest["summary"]["approval_approved_count"] == 1
    assert manifest["summary"]["approval_rejected_count"] == 1
    assert manifest["summary"]["artifact_count"] == 1
    assert manifest["summary"]["artifact_total_bytes"] == 4096
    assert manifest["summary"]["artifact_max_bytes"] == 4096
    assert manifest["artifact_trace"]["kinds"] == {"tool_result": 1}
    assert manifest["artifact_trace"]["tool_names"] == {"dump": 1}
    assert manifest["summary"]["structured_output_record_count"] == 2
    assert manifest["summary"]["structured_output_repair_count"] == 1
    assert manifest["summary"]["structured_output_failed_count"] == 1
    assert manifest["structured_output_trace"]["schema_names"] == {"risk_summary": 1}
    assert manifest["summary"]["planner_plan_count"] == 1
    assert manifest["summary"]["planner_step_count"] == 2
    assert manifest["summary"]["planner_execution_report_count"] == 1
    assert manifest["summary"]["planner_execution_step_count"] == 2
    assert manifest["summary"]["planner_failed_step_count"] == 0
    assert manifest["planner_trace"]["plans"][0]["plan_id"] == "plan-1"
    assert manifest["approval_trace"]["statuses"] == {
        "approved": 1,
        "pending": 1,
        "rejected": 1,
    }
    assert manifest["approval_trace"]["subject_kinds"] == {"action": 1, "tool": 2}
    assert manifest["summary"]["event_log_count"] == 9
    assert manifest["summary"]["correlation_entry_count"] == 5
    assert manifest["summary"]["has_resume"] is True
    assert manifest["summary"]["has_resume_plan"] is True
    assert manifest["summary"]["resume_plan_ready"] is True
    assert manifest["summary"]["has_timeline_reduction"] is True
    assert manifest["summary"]["capability_discovery_match_count"] == 4
    assert manifest["summary"]["memory_search_hit_count"] == 2
    assert manifest["summary"]["context_injection_count"] == 2
    assert manifest["summary"]["context_injection_trimmed_count"] == 1
    assert manifest["context_injections"]["sources"] == {"memory": 1, "runtime": 1}
    assert manifest["summary"]["context_material_selection_count"] == 2
    assert manifest["summary"]["context_material_selected_count"] == 1
    assert manifest["summary"]["context_material_dropped_count"] == 1
    assert manifest["context_material_selection"]["statuses"] == {
        "score_below_threshold": 1,
        "selected": 1,
    }
    assert manifest["summary"]["handoff_record_count"] == 1
    assert manifest["summary"]["handoff_selected_count"] == 1
    assert manifest["handoff_trace"]["selected_sessions"] == {"code-reviewer": 1}
    assert manifest["summary"]["memory_governance_decision_count"] == 2
    assert manifest["summary"]["memory_governance_denied_count"] == 1
    assert manifest["summary"]["memory_governance_rewritten_count"] == 1
    assert manifest["memory_governance"]["decisions_by_status"] == {"deny": 1, "rewrite": 1}
    assert manifest["summary"]["has_prompt_bucket_budget"] is True
    assert manifest["summary"]["prompt_bucket_budget_trimmed_count"] == 1
    assert manifest["prompt_bucket_budget"]["trimmed_count"] == 1
    assert manifest["summary"]["has_prompt_budget"] is True
    assert manifest["summary"]["prompt_budget_provider_limited"] is True
    assert manifest["prompt_budget"]["target_prompt_bytes"] == 900
    assert manifest["summary"]["has_prompt_trim"] is True
    assert manifest["capability_discovery"]["match_count"] == 4
    assert manifest["memory_search"]["hit_count"] == 2
    assert manifest["resume_plan"]["checkpoint_id"] == "c1"


def test_structured_output_trace_summarizes_journal_checkpoints() -> None:
    trace = StructuredOutputTrace.from_journal(
        {
            "events": [
                {
                    "event_type": "checkpoint",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "payload": {
                        "sequence": 1,
                        "state": {
                            "status": "structured_output_failed",
                            "error": "invalid json",
                            "iteration": 0,
                        },
                    },
                },
                {
                    "event_type": "checkpoint",
                    "run_id": "run-1",
                    "turn_id": "turn-2",
                    "payload": {
                        "sequence": 2,
                        "state": {
                            "status": "finished",
                            "structured_output": {
                                "ok": True,
                                "raw_output_bytes": 28,
                                "metadata": {"schema_name": "risk_summary"},
                            },
                        },
                    },
                },
            ]
        }
    ).manifest()

    assert trace["schema_version"] == "agent-core-structured-output-trace/v1"
    assert trace["record_count"] == 2
    assert trace["ok_count"] == 1
    assert trace["failed_count"] == 1
    assert trace["statuses"] == {"finished": 1, "structured_output_failed": 1}
    assert trace["errors"] == {"invalid json": 1}


def test_planner_trace_summarizes_executor_manifest() -> None:
    trace = PlannerTrace.from_manifest(
        {
            "schema_version": "agent-core-plan-executor/v1",
            "default_session": "worker",
            "max_steps": 8,
            "reports": [
                {
                    "schema_version": "agent-core-plan-execution-report/v1",
                    "status": "blocked",
                    "plan": {
                        "schema_version": "agent-core-plan/v1",
                        "plan_id": "plan-1",
                        "goal": "ship",
                        "terminal": False,
                        "ready_steps": ["audit"],
                        "status_counts": {"failed": 1, "pending": 1},
                        "steps": [
                            {"step_id": "audit", "goal": "Audit", "status": "pending"},
                            {"step_id": "build", "goal": "Build", "status": "failed"},
                        ],
                    },
                    "steps": [
                        {
                            "plan_id": "plan-1",
                            "step_id": "build",
                            "session_name": "worker",
                            "status": "failed",
                        }
                    ],
                }
            ],
        }
    ).manifest()

    assert trace["schema_version"] == "agent-core-planner-trace/v1"
    assert trace["plan_count"] == 1
    assert trace["ready_step_count"] == 1
    assert trace["step_count"] == 2
    assert trace["execution_report_count"] == 1
    assert trace["execution_step_count"] == 1
    assert trace["failed_step_count"] == 2
    assert trace["blocked_report_count"] == 1
    assert trace["step_statuses"] == {"failed": 1, "pending": 1}
    assert trace["execution_statuses"] == {"blocked": 1}


def test_trace_correlation_index_cross_references_trace_materials() -> None:
    index = TraceCorrelationIndex.from_trace_components(
        run_id="run-1",
        provider={
            "calls": [
                {
                    "provider_name": "mock",
                    "model": "mock-mini",
                    "attempt": 0,
                    "status": "completed",
                    "metadata": {
                        "request": {"metadata": {"run_id": "run-1", "turn_id": "turn-1"}}
                    },
                }
            ]
        },
        tool_replay={
            "records": [
                {
                    "replay_key": "lookup:{}",
                    "invocation": {"call_id": "call-1", "tool_name": "lookup"},
                    "result": {"call_id": "call-1", "tool_name": "lookup", "status": "completed"},
                }
            ]
        },
        policy_decisions={
            "records": [
                {
                    "decision_id": "decision-1",
                    "sequence": 1,
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "subject": "tool:lookup",
                    "decision": {"status": "allow"},
                    "metadata": {"call_id": "call-1"},
                }
            ]
        },
        approvals={
            "records": [
                {
                    "approval_id": "approval-1",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "status": "approved",
                    "request": {"subject": "tool:lookup", "reason": "ok"},
                    "decision": {"status": "approved", "actor": "operator"},
                }
            ]
        },
        event_log={
            "events": [
                {
                    "type": "tool_finished",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "sequence": 3,
                    "payload": {"call_id": "call-1", "status": "completed"},
                },
                {
                    "type": "approval_requested",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "sequence": 4,
                    "payload": {"approval_id": "approval-1", "request": {"subject": "tool:lookup"}},
                },
            ]
        },
    )
    manifest = index.manifest()

    assert manifest["schema_version"] == "agent-core-trace-correlation-index/v1"
    assert manifest["entry_count"] == 6
    assert "run-1:turn-1" in manifest["groups"]["turns"]
    assert any("tool_replay" in key for key in manifest["groups"]["calls"]["call-1"])
    assert manifest["groups"]["approvals"]["approval-1"]
    assert manifest["groups"]["policy_decisions"]["decision-1"]


def test_trace_bundle_summarizes_and_correlates_embedding_calls() -> None:
    bundle = AgentRunTraceBundle(
        run_id="run-1",
        status="completed",
        embedding={
            "schema_version": "agent-core-embedding-provider-center/v1",
            "call_count": 1,
            "calls": [
                {
                    "provider_name": "local",
                    "model": "embed-small",
                    "input_count": 3,
                    "dimensions": 64,
                    "status": "completed",
                    "metadata": {
                        "request": {
                            "schema_version": "agent-core-embedding-request/v1",
                            "model": "embed-small",
                            "dimensions": 64,
                            "input_count": 3,
                            "metadata": {"run_id": "run-1", "turn_id": "turn-1"},
                        }
                    },
                }
            ],
        },
    )

    manifest = bundle.manifest()

    assert manifest["summary"]["embedding_call_count"] == 1
    assert manifest["embedding"]["calls"][0]["provider_name"] == "local"
    assert manifest["correlation"]["entry_count"] == 1
    entry = manifest["correlation"]["entries"][0]
    assert entry["source"] == "embedding"
    assert entry["kind"] == "embedding_call"
    assert entry["turn_id"] == "turn-1"
    assert entry["metadata"]["dimensions"] == 64


def test_storage_backend_trace_collects_and_deduplicates_component_backends() -> None:
    memory_backend = storage_backend_manifest(
        role="memory",
        kind="postgres",
        name="tenant-memory",
        core_builtin=False,
    )
    duplicate_memory_backend = dict(memory_backend)
    trace = StorageBackendTrace.from_trace_components(
        session={
            "memory": {"stores": [{"backend": memory_backend}]},
            "trace_store": {"backend": storage_backend_manifest(role="run_trace", kind="sqlite")},
        },
        memory_search={"plan": {"selected_stores": [{"backend": duplicate_memory_backend}]}},
        event_log={"backend": storage_backend_manifest(role="event_log", kind="markdown")},
    ).manifest()

    assert trace["schema_version"] == "agent-core-storage-backend-trace/v1"
    assert trace["backend_count"] == 3
    assert trace["external_backend_count"] == 1
    assert trace["roles"] == {"event_log": 1, "memory": 1, "run_trace": 1}
    assert trace["kinds"] == {"markdown": 1, "postgres": 1, "sqlite": 1}
    memory = next(item for item in trace["backends"] if item["role"] == "memory")
    assert memory["core_builtin"] is False
    assert len(memory["sources"]) == 2


def test_run_trace_bundle_summarizes_preflight_report() -> None:
    manifest = AgentRunTraceBundle(
        run_id="run-preflight",
        status="denied",
        preflight={
            "schema_version": "agent-core-run-preflight-report/v1",
            "status": "blocked",
            "ok": False,
            "issue_count": 2,
            "blocking_count": 1,
            "codes": ["empty_task", "missing_tool"],
            "blocking_codes": ["missing_tool"],
            "request": {
                "provider_route_plan": {
                    "schema_version": "agent-core-llm-provider-route-plan/v1",
                    "ready": False,
                    "streamed": True,
                    "selected_route": None,
                    "candidates": [
                        {
                            "provider_name": "mock",
                            "reason": "unsupported_capabilities",
                            "selected": False,
                            "supported": False,
                        }
                    ],
                },
                "storage_backend_preflight": {
                    "schema_version": "agent-core-storage-backend-preflight/v1",
                    "ready": False,
                    "status": "blocked",
                    "blocking_count": 1,
                    "missing_roles": ["memory"],
                    "blocking_reasons": ["missing_capabilities"],
                }
            },
        },
    ).manifest()

    assert manifest["summary"]["has_preflight"] is True
    assert manifest["summary"]["preflight_blocked"] is True
    assert manifest["summary"]["preflight_issue_count"] == 2
    assert manifest["summary"]["preflight_blocking_count"] == 1
    assert manifest["summary"]["has_provider_route_preflight"] is True
    assert manifest["summary"]["provider_route_preflight_ready"] is False
    assert manifest["summary"]["provider_route_preflight_candidate_count"] == 1
    assert manifest["summary"]["has_storage_backend_preflight"] is True
    assert manifest["summary"]["storage_backend_preflight_ready"] is False
    assert manifest["summary"]["storage_backend_preflight_blocking_count"] == 1
    assert manifest["preflight"]["blocking_codes"] == ["missing_tool"]
    assert manifest["provider_route_preflight"]["streamed"] is True
    assert manifest["storage_backend_preflight"]["missing_roles"] == ["memory"]


def test_context_injection_trace_summarizes_prompt_injection_decisions() -> None:
    trace = ContextInjectionTrace.from_prompt(
        {
            "metadata": {
                "context_injections": [
                    {
                        "name": "memory",
                        "source": "memory",
                        "target": "semi_dynamic_1",
                        "status": "trimmed",
                        "included": True,
                        "trimmed": True,
                    },
                    {
                        "name": "denied",
                        "source": "runtime",
                        "target": "high_static",
                        "status": "target_denied",
                        "included": False,
                        "trimmed": False,
                    },
                ]
            }
        }
    ).manifest()

    assert trace["schema_version"] == "agent-core-context-injection-trace/v1"
    assert trace["injection_count"] == 2
    assert trace["included_count"] == 1
    assert trace["excluded_count"] == 1
    assert trace["trimmed_count"] == 1
    assert trace["sources"] == {"memory": 1, "runtime": 1}
    assert trace["targets"] == {"high_static": 1, "semi_dynamic_1": 1}


def test_context_material_selection_trace_summarizes_prompt_selection_decisions() -> None:
    trace = ContextMaterialSelectionTrace.from_prompt(
        {
            "metadata": {
                "context_material_selection": {
                    "schema_version": "agent-core-context-material-selection-result/v1",
                    "selections": [
                        {
                            "name": "auth_trace",
                            "role": "timeline",
                            "target": "timeline_open",
                            "status": "selected",
                            "selected": True,
                            "bytes": 64,
                        },
                        {
                            "name": "old_note",
                            "role": "memory",
                            "target": "semi_dynamic_1",
                            "status": "score_below_threshold",
                            "selected": False,
                            "bytes": 20,
                        },
                    ],
                }
            }
        }
    ).manifest()

    assert trace["schema_version"] == "agent-core-context-material-selection-trace/v1"
    assert trace["selection_count"] == 2
    assert trace["selected_count"] == 1
    assert trace["dropped_count"] == 1
    assert trace["selected_bytes"] == 64
    assert trace["statuses"] == {"score_below_threshold": 1, "selected": 1}
    assert trace["targets"] == {"timeline_open": 1}
    assert trace["names"] == {"auth_trace": 1, "old_note": 1}


def test_context_material_selection_trace_can_fallback_to_selected_injections() -> None:
    trace = ContextMaterialSelectionTrace.from_prompt(
        {
            "metadata": {
                "context_injections": [
                    {
                        "name": "auth_trace",
                        "source": "trace",
                        "target": "timeline_open",
                        "priority": 3,
                        "bytes": 64,
                        "metadata": {
                            "context_material_selection": {
                                "score": 314.5,
                                "rank": 1,
                                "reason": "semantic_priority_selected",
                                "target": "timeline_open",
                            }
                        },
                    }
                ]
            }
        }
    ).manifest()

    assert trace["selection_count"] == 1
    assert trace["selected_count"] == 1
    assert trace["dropped_count"] == 0
    assert trace["targets"] == {"timeline_open": 1}
    assert trace["selections"][0]["score"] == 314.5


def test_handoff_trace_summarizes_prompt_handoff_decisions() -> None:
    trace = HandoffTrace.from_prompt(
        {
            "metadata": {
                "handoff": [
                    {
                        "status": "selected",
                        "selected_session": "code-reviewer",
                        "request": {
                            "task": "review patch",
                            "source_session": "planner",
                            "required_tags": ["review"],
                            "required_tools": ["diff"],
                        },
                        "candidates": [{"session_name": "code-reviewer"}],
                    },
                    {
                        "status": "denied",
                        "selected_session": "",
                        "reason": "target session is disabled",
                        "request": {"task": "ops task", "source_session": "planner"},
                    },
                ]
            }
        }
    ).manifest()

    assert trace["schema_version"] == "agent-core-handoff-trace/v1"
    assert trace["record_count"] == 2
    assert trace["selected_count"] == 1
    assert trace["denied_count"] == 1
    assert trace["statuses"] == {"denied": 1, "selected": 1}
    assert trace["selected_sessions"] == {"code-reviewer": 1}
    assert trace["source_sessions"] == {"planner": 2}
    assert trace["records"][0]["candidate_sessions"] == ["code-reviewer"]


def test_agent_tool_trace_summarizes_tool_center_agent_calls() -> None:
    trace = AgentToolTrace.from_tool_center(
        {
            "calls": [
                {
                    "result": {
                        "metadata": {
                            "schema_version": "agent-core-agent-tool-call/v1",
                            "tool": {
                                "tool_name": "agent_code_reviewer",
                                "session_name": "code-reviewer",
                                "enabled": True,
                                "tags": ["code", "agent_tool"],
                            },
                            "invocation": {"tool_name": "agent_code_reviewer"},
                            "task_bytes": 12,
                            "result": {
                                "run_id": "child-run",
                                "status": "completed",
                                "iterations": 2,
                                "output_bytes": 48,
                                "trace_run_id": "child-run",
                            },
                        }
                    }
                },
                {
                    "result": {
                        "metadata": {
                            "schema_version": "agent-core-agent-tool-call/v1",
                            "tool": {
                                "tool_name": "agent_ops",
                                "session_name": "ops",
                            },
                            "invocation": {"tool_name": "agent_ops"},
                            "task_bytes": 10,
                            "result": {"status": "failed"},
                            "error": "child failed",
                        }
                    }
                },
            ]
        }
    ).manifest()

    assert trace["schema_version"] == "agent-core-agent-tool-trace/v1"
    assert trace["record_count"] == 2
    assert trace["completed_count"] == 1
    assert trace["failed_count"] == 1
    assert trace["tools"] == {"agent_code_reviewer": 1, "agent_ops": 1}
    assert trace["sessions"] == {"code-reviewer": 1, "ops": 1}
    assert trace["statuses"] == {"completed": 1, "failed": 1}
    assert trace["records"][0]["run_id"] == "child-run"


def test_run_trace_bundle_includes_agent_tool_trace_from_tool_center() -> None:
    bundle = AgentRunTraceBundle(
        run_id="parent-run",
        status="completed",
        session={
            "tools": {
                "schema_version": "agent-core-tool-center/v1",
                "calls": [
                    {
                        "requested_tool_name": "agent_code_reviewer",
                        "status": "completed",
                        "route_plan": {
                            "schema_version": "agent-core-tool-route-plan/v1",
                            "ready": True,
                            "selected_mount": "agents",
                            "selected_tool_name": "agent_code_reviewer",
                        },
                        "result": {
                            "metadata": {
                                "schema_version": "agent-core-agent-tool-call/v1",
                                "tool": {
                                    "tool_name": "agent_code_reviewer",
                                    "session_name": "code-reviewer",
                                },
                                "invocation": {"tool_name": "agent_code_reviewer"},
                                "task_bytes": 18,
                                "result": {
                                    "run_id": "child-run",
                                    "status": "completed",
                                    "iterations": 1,
                                    "output_bytes": 32,
                                    "trace_run_id": "child-run",
                                },
                            }
                        },
                    }
                ],
            }
        },
    ).manifest()

    assert bundle["summary"]["tool_center_call_count"] == 1
    assert bundle["summary"]["agent_tool_record_count"] == 1
    assert bundle["summary"]["agent_tool_completed_count"] == 1
    assert bundle["summary"]["agent_tool_failed_count"] == 0
    assert bundle["agent_tool_trace"]["sessions"] == {"code-reviewer": 1}
    assert bundle["agent_tool_trace"]["records"][0]["tool_name"] == "agent_code_reviewer"


def test_memory_governance_trace_summarizes_session_memory_decisions() -> None:
    trace = MemoryGovernanceTrace.from_session(
        {
            "memory": {
                "governance": {
                    "decisions": [
                        {"decision": "rewrite", "allowed": True, "risk_level": "low", "store": "local"},
                        {"decision": "deny", "allowed": False, "risk_level": "high", "store": "local"},
                    ]
                }
            }
        }
    ).manifest()

    assert trace["schema_version"] == "agent-core-memory-governance-trace/v1"
    assert trace["decision_count"] == 2
    assert trace["allowed_count"] == 1
    assert trace["denied_count"] == 1
    assert trace["rewritten_count"] == 1
    assert trace["decisions_by_status"] == {"deny": 1, "rewrite": 1}
    assert trace["risk_levels"] == {"high": 1, "low": 1}


@pytest.mark.asyncio
async def test_run_trace_stores_persist_bundle_manifests(tmp_path) -> None:
    bundle = AgentRunTraceBundle(
        run_id="run-1",
        status="completed",
        iterations=1,
        provider={"call_count": 1},
    ).manifest()
    memory = InMemoryRunTraceStore()
    sqlite = SQLiteRunTraceStore(tmp_path / "traces.sqlite")
    markdown = MarkdownRunTraceStore(tmp_path / "traces.md")

    for store in (memory, sqlite, markdown):
        await store.save(bundle)
        loaded = await store.load("run-1")
        records = await store.records()

        assert loaded is not None
        assert loaded["run"]["run_id"] == "run-1"
        assert records[0]["summary"]["provider_call_count"] == 1

    assert memory.manifest()["record_count"] == 1
    assert sqlite.manifest()["schema_version"] == "agent-core-sqlite-run-trace-store/v1"
    assert markdown.manifest()["schema_version"] == "agent-core-markdown-run-trace-store/v1"
    assert "agent-core-run-trace-bundle" not in (tmp_path / "traces.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_run_trace_stores_query_by_status_metadata_and_limit(tmp_path) -> None:
    records = (
        AgentRunTraceBundle(
            run_id="run-a",
            status="completed",
            metadata={"tenant": "alpha"},
        ).manifest(),
        AgentRunTraceBundle(
            run_id="run-b",
            status="failed",
            metadata={"tenant": "alpha"},
        ).manifest(),
        AgentRunTraceBundle(
            run_id="run-c",
            status="completed",
            metadata={"tenant": "beta"},
        ).manifest(),
    )
    stores = (
        InMemoryRunTraceStore(),
        SQLiteRunTraceStore(tmp_path / "query-traces.sqlite"),
        MarkdownRunTraceStore(tmp_path / "query-traces.md"),
    )

    for store in stores:
        for record in records:
            await store.save(record)

        completed_alpha = await store.query(
            RunTraceQuery(statuses=("completed",), metadata={"tenant": "alpha"})
        )
        latest_two = await store.query(RunTraceQuery(limit=2, reverse=True))
        selected = await store.query(RunTraceQuery(run_ids=("run-b", "missing")))

        assert [item["run"]["run_id"] for item in completed_alpha] == ["run-a"]
        assert [item["run"]["run_id"] for item in latest_two] == ["run-c", "run-b"]
        assert [item["run"]["run_id"] for item in selected] == ["run-b"]


@pytest.mark.asyncio
async def test_agent_runner_exports_run_trace_bundle() -> None:
    provider = MockLLMProvider(
        [
            {"action": "lookup", "arguments": {"query": "target"}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    center = LLMProviderCenter(default_provider="mock")
    center.register("mock", provider, default_model="mock-mini")
    tools = ToolRegistry()
    tools.register(ToolSpec(name="lookup"), _lookup)
    tool_center = ToolCenter()
    tool_center.mount("local", tools)
    event_sink = ListEventSink()
    session = AgentSession(
        profile=AgentProfile(name="traceable", model="mock-mini"),
        provider=center,
        tools=tool_center,
        harness=PersistentAgentJournal(InMemoryJournalStore()),
        memory=InMemoryMemoryStore(),
        event_sink=event_sink,
        tool_replay=PersistentToolReplay(InMemoryToolReplayStore()),
        trace_store=InMemoryRunTraceStore(),
        artifact_store=InMemoryArtifactStore(),
        policy_decision_store=InMemoryPolicyDecisionStore(),
    )

    outcome = await AgentRunner(session).run("inspect")
    trace = outcome.trace_manifest

    assert outcome.result.status == "completed"
    assert trace["schema_version"] == "agent-core-run-trace-bundle/v1"
    assert trace["run"]["run_id"] == outcome.result.run_id
    assert trace["run"]["status"] == "completed"
    assert trace["summary"]["journal_ok"] is True
    assert trace["summary"]["journal_event_count"] > 0
    assert trace["summary"]["provider_call_count"] == 2
    assert trace["summary"]["tool_replay_record_count"] == 1
    assert trace["summary"]["tool_center_call_count"] == 1
    assert trace["summary"]["tool_center_failed_count"] == 0
    assert trace["summary"]["tool_center_route_plan_count"] == 1
    assert trace["summary"]["policy_decision_record_count"] == 3
    assert trace["summary"]["event_log_count"] == event_sink.manifest()["event_count"]
    assert trace["summary"]["correlation_entry_count"] > 0
    assert trace["summary"]["has_resume_plan"] is False
    assert trace["summary"]["resume_plan_ready"] is False
    assert trace["journal_replay"]["ok"] is True
    assert trace["provider"]["calls"][0]["provider_name"] == "mock"
    assert trace["provider"]["calls"][0]["metadata"]["request"]["metadata"]["run_id"] == outcome.result.run_id
    assert trace["tool_replay"]["records"][0]["result"]["tool_name"] == "lookup"
    assert trace["tool_center"]["calls"][0]["route_plan"]["selected_mount"] == "local"
    assert trace["tool_center"]["calls"][0]["route_plan"]["selected_tool_name"] == "lookup"
    assert trace["tool_center"]["selected_mounts"] == {"local": 1}
    assert trace["policy_decisions"]["records"][1]["subject"] == "tool:lookup"
    call_id = trace["tool_replay"]["records"][0]["result"]["call_id"]
    assert trace["correlation"]["groups"]["calls"][call_id]
    assert trace["prompt"]["metadata"]["profile"] == "traceable"
    assert trace["summary"]["capability_discovery_match_count"] == trace["capability_discovery"]["match_count"]
    assert trace["summary"]["memory_search_hit_count"] == 0
    assert trace["summary"]["context_injection_count"] == 0
    assert trace["summary"]["storage_backend_count"] >= 7
    assert trace["storage_backends"]["roles"]["memory"] == 1
    assert trace["storage_backends"]["roles"]["journal"] == 1
    assert trace["storage_backends"]["roles"]["tool_replay"] == 1
    assert trace["storage_backends"]["roles"]["run_trace"] == 1
    assert trace["storage_backends"]["roles"]["event_log"] == 1
    assert trace["storage_backends"]["roles"]["artifact"] == 1
    assert trace["storage_backends"]["roles"]["policy_decision"] == 1
    assert trace["capability_discovery"]["schema_version"] == "agent-core-capability-discovery/v1"
    assert trace["memory_search"]["enabled"] is True


@pytest.mark.asyncio
async def test_agent_runner_saves_trace_bundle_when_store_is_configured() -> None:
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}])
    trace_store = InMemoryRunTraceStore()
    session = AgentSession(
        profile=AgentProfile(name="trace-store"),
        provider=provider,
        tools=ToolRegistry(),
        harness=InMemoryAgentJournal(),
        trace_store=trace_store,
    )

    outcome = await AgentRunner(session).run("finish")
    saved = await trace_store.load(outcome.result.run_id)

    assert saved is not None
    assert saved["run"]["status"] == "completed"
    assert saved["prompt"]["metadata"]["profile"] == "trace-store"
    assert outcome.session_manifest["trace_store"]["record_count"] == 0
