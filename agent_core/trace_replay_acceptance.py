"""SDK-level trace replay compatibility acceptance checks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from agent_core.config import AgentProfile
from agent_core.events import ListEventSink
from agent_core.evals import (
    DefaultTraceEvaluator,
    TraceEvalHarness,
    TraceEvalSpec,
    TraceReplayComparator,
    TraceReplayDiffSpec,
    TraceReplayHarness,
)
from agent_core.providers import LLMProviderCenter
from agent_core.runner import AgentRunRequest, AgentRunner, AgentSession
from agent_core.testing import MockLLMProvider, MockToolRuntime
from agent_core.trace import InMemoryRunTraceStore


@dataclass(frozen=True)
class AgentCoreTraceReplayAcceptanceIssue:
    """One blocking trace replay compatibility issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-replay-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreTraceReplayAcceptanceReport:
    """Prompt-safe report for trace replay and replay-diff compatibility."""

    status: str
    current_replay: dict[str, Any] = field(default_factory=dict)
    stored_eval: dict[str, Any] = field(default_factory=dict)
    compatibility_compare: dict[str, Any] = field(default_factory=dict)
    drift_detection: dict[str, Any] = field(default_factory=dict)
    legacy_replay: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreTraceReplayAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-replay-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "current_replay": dict(self.current_replay),
            "stored_eval": dict(self.stored_eval),
            "compatibility_compare": dict(self.compatibility_compare),
            "drift_detection": dict(self.drift_detection),
            "legacy_replay": dict(self.legacy_replay),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreTraceReplayAcceptanceHarness:
    """Run deterministic trace replay compatibility checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreTraceReplayAcceptanceReport:
        trace_store = InMemoryRunTraceStore()
        provider = MockLLMProvider(
            [
                {
                    "action": "call_tool",
                    "arguments": {
                        "tool_name": "lookup",
                        "arguments": {"query": "replay"},
                    },
                },
                {"action": "finish", "arguments": {"output": "replay-ok"}},
            ]
        )
        center = LLMProviderCenter(default_provider="replay")
        center.register("replay", provider, default_model="replay-mini")
        session = AgentSession(
            profile=AgentProfile(name="trace-replay-acceptance", model="replay-mini"),
            provider=center,
            tools=MockToolRuntime({"lookup": "replay-tool-result"}),
            event_sink=ListEventSink(),
            trace_store=trace_store,
        )
        outcome = await AgentRunner(session).run(
            AgentRunRequest(task="build replay trace", metadata={"acceptance": "trace_replay"})
        )
        trace = outcome.trace_manifest
        replay = TraceReplayHarness().replay(trace).manifest()
        stored_eval = (
            await TraceEvalHarness(trace_store=trace_store).evaluate_run(
                outcome.result.run_id,
                TraceEvalSpec(
                    name="trace-replay-stored-eval",
                    expected_status="completed",
                    max_provider_calls=2,
                    required_tool_names=("lookup",),
                    required_event_types=("run_started", "run_finished", "tool_finished"),
                ),
            )
        ).manifest()
        compatibility_compare = TraceReplayComparator().compare(
            trace,
            _normalized_trace_copy(trace),
            _compatibility_diff_spec(),
        ).manifest()
        drift_detection = TraceReplayComparator().compare(
            trace,
            _trace_with_missing_provider_step(trace),
            _compatibility_diff_spec(),
        ).manifest()
        legacy_trace = _legacy_minimal_trace()
        legacy_replay_result = TraceReplayHarness().replay(legacy_trace)
        legacy_eval = DefaultTraceEvaluator().evaluate(
            legacy_trace,
            TraceEvalSpec(
                name="trace-replay-legacy-eval",
                expected_status="completed",
                require_journal_ok=True,
                required_event_types=("run_started", "run_finished"),
            ),
        ).manifest()
        legacy_replay = {
            "schema_version": "agent-core-trace-replay-legacy-acceptance/v1",
            "replay": legacy_replay_result.manifest(),
            "eval": legacy_eval,
        }
        current_replay = {
            "schema_version": "agent-core-trace-replay-current-acceptance/v1",
            "run_id": outcome.result.run_id,
            "run_status": outcome.result.status,
            "step_count": int(replay.get("step_count") or 0),
            "event_types": list(replay.get("steps") and [step["event_type"] for step in replay["steps"]] or ()),
            "trace_store": trace_store.manifest(),
        }
        issues = _trace_replay_issues(
            current_replay=current_replay,
            stored_eval=stored_eval,
            compatibility_compare=compatibility_compare,
            drift_detection=drift_detection,
            legacy_replay=legacy_replay,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreTraceReplayAcceptanceReport(
            status=status,
            current_replay=current_replay,
            stored_eval=stored_eval,
            compatibility_compare=compatibility_compare,
            drift_detection=_diff_summary(drift_detection),
            legacy_replay=_legacy_summary(legacy_replay),
            issues=issues,
            metadata={"scenario": "agent_core_trace_replay_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_trace_replay_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreTraceReplayAcceptanceReport:
    """Run the default trace replay compatibility acceptance checks."""

    return await AgentCoreTraceReplayAcceptanceHarness(metadata=dict(metadata or {})).run()


def _compatibility_diff_spec() -> TraceReplayDiffSpec:
    return TraceReplayDiffSpec(
        name="trace-replay-compatibility",
        compare_event_order=True,
        compare_sources=True,
        compare_payloads=False,
    )


def _normalized_trace_copy(trace: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(trace)
    normalized["schema_version"] = "agent-core-run-trace-bundle/v1-compatible-copy"
    normalized.setdefault("metadata", {})["compatibility_copy"] = True
    return normalized


def _trace_with_missing_provider_step(trace: dict[str, Any]) -> dict[str, Any]:
    changed = deepcopy(trace)
    provider = changed.get("provider") if isinstance(changed.get("provider"), dict) else {}
    calls = list(provider.get("calls") or ())
    if calls:
        provider["calls"] = calls[:-1]
        provider["call_count"] = len(provider["calls"])
    return changed


def _legacy_minimal_trace() -> dict[str, Any]:
    return {
        "schema_version": "agent-core-run-trace-bundle/v0",
        "run": {
            "run_id": "legacy-run",
            "status": "completed",
            "iterations": 1,
        },
        "summary": {
            "provider_call_count": 0,
            "embedding_call_count": 0,
            "tool_replay_record_count": 0,
            "policy_decision_record_count": 0,
            "approval_record_count": 0,
            "event_log_count": 0,
        },
        "journal_replay": {
            "schema_version": "agent-core-journal-replay/v0",
            "ok": True,
            "event_count": 2,
            "events": [
                {
                    "sequence": 1,
                    "event_type": "run_started",
                    "run_id": "legacy-run",
                    "payload": {"run_id": "legacy-run"},
                },
                {
                    "sequence": 2,
                    "event_type": "run_finished",
                    "run_id": "legacy-run",
                    "payload": {"run_id": "legacy-run", "status": "completed"},
                },
            ],
        },
    }


def _trace_replay_issues(
    *,
    current_replay: dict[str, Any],
    stored_eval: dict[str, Any],
    compatibility_compare: dict[str, Any],
    drift_detection: dict[str, Any],
    legacy_replay: dict[str, Any],
) -> tuple[AgentCoreTraceReplayAcceptanceIssue, ...]:
    issues: list[AgentCoreTraceReplayAcceptanceIssue] = []
    if int(current_replay.get("step_count") or 0) < 6:
        issues.append(
            AgentCoreTraceReplayAcceptanceIssue(
                source="current_replay",
                code="current_replay_too_short",
                message="Current trace replay did not include enough SDK execution steps.",
                metadata=dict(current_replay),
            )
        )
    event_types = set(current_replay.get("event_types") or ())
    for event_type in ("run_started", "tool_finished", "run_finished", "provider_call_completed"):
        if event_type not in event_types:
            issues.append(
                AgentCoreTraceReplayAcceptanceIssue(
                    source="current_replay",
                    code="current_replay_event_missing",
                    message=f"Current trace replay is missing event type: {event_type}",
                    metadata={"event_type": event_type, "event_types": sorted(event_types)},
                )
            )
    if stored_eval.get("ok") is not True:
        issues.append(
            AgentCoreTraceReplayAcceptanceIssue(
                source="stored_eval",
                code="stored_trace_eval_failed",
                message="Stored trace eval did not pass.",
                metadata={"stored_eval": dict(stored_eval)},
            )
        )
    if compatibility_compare.get("ok") is not True:
        issues.append(
            AgentCoreTraceReplayAcceptanceIssue(
                source="compatibility_compare",
                code="compatible_trace_compare_failed",
                message="Replay comparator rejected a normalized compatible trace copy.",
                metadata={"compatibility_compare": dict(compatibility_compare)},
            )
        )
    if drift_detection.get("ok") is not False or int(drift_detection.get("issue_count") or 0) < 1:
        issues.append(
            AgentCoreTraceReplayAcceptanceIssue(
                source="drift_detection",
                code="trace_drift_not_detected",
                message="Replay comparator did not detect a missing provider replay step.",
                metadata={"drift_detection": dict(drift_detection)},
            )
        )
    legacy_eval = legacy_replay.get("eval") if isinstance(legacy_replay.get("eval"), dict) else {}
    legacy_manifest = (
        (legacy_replay.get("replay") or {}) if isinstance(legacy_replay.get("replay"), dict) else {}
    )
    if legacy_eval.get("ok") is not True or int(legacy_manifest.get("step_count") or 0) != 2:
        issues.append(
            AgentCoreTraceReplayAcceptanceIssue(
                source="legacy_replay",
                code="legacy_trace_replay_failed",
                message="Legacy minimal trace replay/eval did not pass.",
                metadata={"legacy_replay": dict(legacy_replay)},
            )
        )
    return tuple(issues)


def _diff_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "agent-core-trace-replay-drift-summary/v1",
        "ok": bool(report.get("ok")),
        "issue_count": int(report.get("issue_count") or 0),
        "issue_codes": [
            str(issue.get("code") or "")
            for issue in report.get("issues") or ()
            if isinstance(issue, dict)
        ],
        "summary": dict(report.get("summary") or {}),
    }


def _legacy_summary(report: dict[str, Any]) -> dict[str, Any]:
    replay = report.get("replay") if isinstance(report.get("replay"), dict) else {}
    eval_report = report.get("eval") if isinstance(report.get("eval"), dict) else {}
    return {
        "schema_version": "agent-core-trace-replay-legacy-summary/v1",
        "step_count": int(replay.get("step_count") or 0),
        "event_types": [
            str(step.get("event_type") or "")
            for step in replay.get("steps") or ()
            if isinstance(step, dict)
        ],
        "eval_ok": bool(eval_report.get("ok")),
        "source_schema_version": str((replay.get("metadata") or {}).get("source_schema_version") or ""),
    }
