"""Acceptance gate for unified context window audit reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.context_window import (
    ContextWindowBuilder,
    ContextWindowPolicy,
)


@dataclass(frozen=True)
class AgentCoreContextWindowAcceptanceIssue:
    """One context window acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-context-window-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreContextWindowAcceptanceReport:
    """Prompt-safe context window acceptance report."""

    status: str
    ready_report: dict[str, Any] = field(default_factory=dict)
    blocked_report: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreContextWindowAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-context-window-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "ready_report": dict(self.ready_report),
            "blocked_report": dict(self.blocked_report),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreContextWindowAcceptanceHarness:
    """Run deterministic context window audit checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreContextWindowAcceptanceReport:
        trace = _context_window_trace()
        ready_report = ContextWindowBuilder(
            policy=ContextWindowPolicy(
                max_prompt_bytes=1600,
                max_excluded_contexts=2,
                max_trimmed_contexts=2,
                required_sources=("memory", "runtime", "trace"),
            )
        ).from_trace(trace, metadata={"scenario": "context_window_ready"}).manifest()
        blocked_report = ContextWindowBuilder(
            policy=ContextWindowPolicy(
                max_prompt_bytes=500,
                max_excluded_contexts=0,
                max_trimmed_contexts=0,
                required_sources=("memory", "runtime", "trace", "mcp"),
            )
        ).from_trace(trace, metadata={"scenario": "context_window_blocked"}).manifest()
        issues = _context_window_acceptance_issues(
            ready_report=ready_report,
            blocked_report=blocked_report,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreContextWindowAcceptanceReport(
            status=status,
            ready_report=ready_report,
            blocked_report=blocked_report,
            issues=issues,
            metadata={"scenario": "agent_core_context_window_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_context_window_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreContextWindowAcceptanceReport:
    """Run deterministic context window acceptance checks."""

    return await AgentCoreContextWindowAcceptanceHarness(
        metadata=dict(metadata or {})
    ).run()


def _context_window_trace() -> dict[str, Any]:
    return {
        "schema_version": "agent-core-run-trace-bundle/v1",
        "run": {"run_id": "context-window-run", "status": "completed"},
        "prompt": {
            "schema_version": "agent-core-prompt-ir/v1",
            "prompt_bytes": 1200,
            "prompt_sha256": "abc123",
            "buckets": [
                {"role": "semi_dynamic_1", "bytes": 180},
                {"role": "semi_dynamic_2", "bytes": 120},
                {"role": "timeline_open", "bytes": 300},
                {"role": "dynamic", "bytes": 80},
            ],
            "metadata": {
                "bucket_budget": {
                    "schema_version": "agent-core-prompt-bucket-budget-result/v1",
                    "trimmed_count": 1,
                    "decisions": [
                        {"role": "semi_dynamic_1", "status": "trimmed"},
                    ],
                },
                "semantic_trim": {
                    "schema_version": "agent-core-prompt-semantic-trim-result/v1",
                    "trimmed_count": 1,
                    "decisions": [
                        {"role": "timeline_open", "status": "trimmed"},
                    ],
                },
            },
        },
        "context_material_selection": {
            "schema_version": "agent-core-context-material-selection-trace/v1",
            "selection_count": 4,
            "selected_count": 3,
            "dropped_count": 1,
            "selections": [
                {
                    "name": "memory-hit",
                    "role": "memory",
                    "target": "semi_dynamic_1",
                    "status": "selected",
                    "selected": True,
                    "score": 3.2,
                    "rank": 1,
                    "bytes": 260,
                    "sha256": "memhash",
                    "metadata": {"source": "memory"},
                },
                {
                    "name": "runtime-hint",
                    "role": "runtime",
                    "target": "semi_dynamic_2",
                    "status": "selected",
                    "selected": True,
                    "score": 2.7,
                    "rank": 2,
                    "bytes": 120,
                    "sha256": "runtimehash",
                    "metadata": {"source": "runtime"},
                },
                {
                    "name": "trace-evidence",
                    "role": "timeline",
                    "target": "timeline_open",
                    "status": "selected",
                    "selected": True,
                    "score": 2.2,
                    "rank": 3,
                    "bytes": 360,
                    "sha256": "tracehash",
                    "metadata": {"source": "trace"},
                },
                {
                    "name": "static-denied",
                    "role": "system",
                    "target": "high_static",
                    "status": "target_denied",
                    "selected": False,
                    "score": 4.0,
                    "rank": 0,
                    "bytes": 100,
                    "sha256": "deniedhash",
                    "metadata": {"source": "runtime"},
                },
            ],
        },
        "context_injections": {
            "schema_version": "agent-core-context-injection-trace/v1",
            "injection_count": 3,
            "included_count": 3,
            "excluded_count": 0,
            "trimmed_count": 2,
            "injections": [
                {
                    "name": "memory-hit",
                    "source": "memory",
                    "target": "semi_dynamic_1",
                    "status": "trimmed",
                    "included": True,
                    "trimmed": True,
                    "original_bytes": 260,
                    "final_bytes": 120,
                },
                {
                    "name": "runtime-hint",
                    "source": "runtime",
                    "target": "semi_dynamic_2",
                    "status": "included",
                    "included": True,
                    "trimmed": False,
                    "original_bytes": 120,
                    "final_bytes": 120,
                },
                {
                    "name": "trace-evidence",
                    "source": "trace",
                    "target": "timeline_open",
                    "status": "trimmed",
                    "included": True,
                    "trimmed": True,
                    "original_bytes": 360,
                    "final_bytes": 160,
                },
            ],
        },
    }


def _context_window_acceptance_issues(
    *,
    ready_report: dict[str, Any],
    blocked_report: dict[str, Any],
) -> tuple[AgentCoreContextWindowAcceptanceIssue, ...]:
    issues: list[AgentCoreContextWindowAcceptanceIssue] = []
    if ready_report.get("ready") is not True:
        issues.extend(_wrap_report_issues("ready_report", ready_report.get("issues") or ()))
    if int(ready_report.get("selected_count") or 0) != 3:
        issues.append(
            AgentCoreContextWindowAcceptanceIssue(
                source="ready_report",
                code="selected_count_unexpected",
                message="Context window ready report did not preserve selected count.",
                metadata={"selected_count": ready_report.get("selected_count")},
            )
        )
    if int(ready_report.get("trimmed_count") or 0) != 2:
        issues.append(
            AgentCoreContextWindowAcceptanceIssue(
                source="ready_report",
                code="trimmed_count_unexpected",
                message="Context window ready report did not preserve trimmed count.",
                metadata={"trimmed_count": ready_report.get("trimmed_count")},
            )
        )
    if blocked_report.get("status") != "blocked":
        issues.append(
            AgentCoreContextWindowAcceptanceIssue(
                source="blocked_report",
                code="blocked_report_not_blocked",
                message="Strict context window policy did not block the report.",
                metadata={"status": blocked_report.get("status")},
            )
        )
    expected_codes = {
        "prompt_bytes_exceeded",
        "excluded_context_count_exceeded",
        "trimmed_context_count_exceeded",
        "required_context_source_missing",
    }
    actual_codes = {
        str(issue.get("code") or "")
        for issue in blocked_report.get("issues") or ()
        if isinstance(issue, dict)
    }
    if not expected_codes <= actual_codes:
        issues.append(
            AgentCoreContextWindowAcceptanceIssue(
                source="blocked_report",
                code="blocked_issue_codes_missing",
                message="Strict context window policy did not report all expected issue codes.",
                metadata={
                    "expected_codes": sorted(expected_codes),
                    "actual_codes": sorted(actual_codes),
                },
            )
        )
    return tuple(issues)


def _wrap_report_issues(
    source: str,
    report_issues: Any,
) -> tuple[AgentCoreContextWindowAcceptanceIssue, ...]:
    return tuple(
        AgentCoreContextWindowAcceptanceIssue(
            source=source,
            code=str(issue.get("code") or "context_window_issue"),
            message=str(issue.get("message") or ""),
            severity=str(issue.get("severity") or "error"),
            metadata=dict(issue.get("metadata") or {}),
        )
        for issue in report_issues
        if isinstance(issue, dict)
    )
