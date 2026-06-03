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
from agent_core.backends import storage_backend_manifest
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
            "approval_record_count": 0,
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
        "provider_call_completed",
        "provider_call_completed",
    )
    assert manifest["steps"][2]["source"] == "event_log"
    assert manifest["steps"][4]["source"] == "provider"
    assert manifest["steps"][4]["payload"]["provider_name"] == "mock"


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


def test_trace_replay_comparator_accepts_matching_trace() -> None:
    trace = _trace_manifest()
    report = TraceReplayComparator().compare(trace, trace)

    assert report.ok
    assert report.summary["baseline_step_count"] == 6
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
