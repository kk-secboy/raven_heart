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
from agent_core.runner import AgentRunner, AgentSession
from agent_core.providers import LLMProviderCenter
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
                {"type": "tool_started", "run_id": "run-1", "payload": {"tool_name": "lookup"}},
                {"type": "tool_finished", "run_id": "run-1", "payload": {"tool_name": "lookup"}},
            ],
        },
        "provider": {
            "call_count": 2,
            "calls": [{"usage": {"cost_usd": 0.01}}, {"usage": {"cost_usd": 0.02}}],
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
    )
    assert manifest["steps"][2]["source"] == "event_log"


def test_default_trace_evaluator_accepts_expected_trace() -> None:
    report = DefaultTraceEvaluator().evaluate(
        _trace_manifest(),
        TraceEvalSpec(
            name="happy-path",
            expected_status="completed",
            max_iterations=3,
            max_provider_calls=3,
            max_cost_usd=0.05,
            required_event_types=("run_finished", "tool_finished"),
            required_tool_names=("lookup",),
        ),
    )

    assert report.ok
    assert report.summary["cost_usd"] == 0.03
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


def test_trace_replay_comparator_accepts_matching_trace() -> None:
    trace = _trace_manifest()
    report = TraceReplayComparator().compare(trace, trace)

    assert report.ok
    assert report.summary["baseline_step_count"] == 4
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
    center.register("mock", provider, default_model="mock-mini")
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
            required_event_types=("run_finished", "tool_finished"),
            required_tool_names=("lookup",),
        ),
    )

    assert report.ok
    assert report.run_id == outcome.result.run_id
    assert report.summary["provider_call_count"] == 2
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
