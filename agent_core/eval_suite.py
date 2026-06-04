"""Trace evaluation suite contracts for regression gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec, TraceEvaluatorPort
from agent_core.trace import InMemoryRunTraceStore, RunTraceStorePort


@dataclass(frozen=True)
class TraceEvalCase:
    """One run/spec pair in a provider-neutral eval suite."""

    name: str
    run_id: str
    spec: TraceEvalSpec = field(default_factory=TraceEvalSpec)
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("eval case name is required")
        if not self.run_id:
            raise ValueError("eval case run_id is required")
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags if str(tag)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-eval-case/v1",
            "name": self.name,
            "run_id": self.run_id,
            "spec": self.spec.manifest(),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceEvalSuite:
    """A named set of trace eval cases."""

    name: str = "trace-eval-suite"
    cases: tuple[TraceEvalCase, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "metadata", dict(self.metadata))
        names = [case.name for case in self.cases]
        if len(names) != len(set(names)):
            raise ValueError("eval case names must be unique")

    def select(self, *, tags: tuple[str, ...] = ()) -> "TraceEvalSuite":
        requested = {str(tag) for tag in tags if str(tag)}
        if not requested:
            return self
        return TraceEvalSuite(
            name=self.name,
            cases=tuple(case for case in self.cases if requested <= set(case.tags)),
            metadata={**self.metadata, "selected_tags": sorted(requested)},
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-eval-suite/v1",
            "name": self.name,
            "case_count": len(self.cases),
            "cases": [case.manifest() for case in self.cases],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceEvalSuiteIssue:
    """One suite-level issue, including missing traces."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-eval-suite-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceEvalCaseReport:
    """Prompt-safe report for one suite case."""

    case: dict[str, Any]
    status: str
    eval_report: dict[str, Any] = field(default_factory=dict)
    issues: tuple[TraceEvalSuiteIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-eval-case-report/v1",
            "case": dict(self.case),
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "eval_report": dict(self.eval_report),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceEvalSuiteReport:
    """Prompt-safe aggregate report for one eval suite run."""

    suite: dict[str, Any]
    status: str
    case_reports: tuple[TraceEvalCaseReport, ...] = ()
    issues: tuple[TraceEvalSuiteIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for report in self.case_reports:
            counts[report.status] = counts.get(report.status, 0) + 1
        return counts

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-eval-suite-report/v1",
            "suite": dict(self.suite),
            "status": self.status,
            "ready": self.ready,
            "case_count": len(self.case_reports),
            "status_counts": self.status_counts(),
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "case_reports": [report.manifest() for report in self.case_reports],
            "metadata": dict(self.metadata),
        }


class TraceEvalSuiteRunner:
    """Run a suite of trace eval cases against a trace store."""

    def __init__(
        self,
        *,
        trace_store: RunTraceStorePort,
        evaluator: TraceEvaluatorPort | None = None,
    ) -> None:
        self.trace_store = trace_store
        self.evaluator = evaluator or DefaultTraceEvaluator()

    async def evaluate(
        self,
        suite: TraceEvalSuite,
        *,
        tags: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> TraceEvalSuiteReport:
        selected = suite.select(tags=tags)
        case_reports = []
        suite_issues: list[TraceEvalSuiteIssue] = []
        for case in selected.cases:
            report = await self._evaluate_case(case)
            case_reports.append(report)
            suite_issues.extend(report.issues)
        if not selected.cases:
            suite_issues.append(
                TraceEvalSuiteIssue(
                    source=selected.name,
                    code="suite_has_no_cases",
                    message="Trace eval suite did not select any cases.",
                    metadata={"tags": list(tags)},
                )
            )
        status = "blocked" if any(issue.severity == "error" for issue in suite_issues) else "ready"
        return TraceEvalSuiteReport(
            suite=selected.manifest(),
            status=status,
            case_reports=tuple(case_reports),
            issues=tuple(suite_issues),
            metadata={"source": type(self).__name__, **dict(metadata or {})},
        )

    async def _evaluate_case(self, case: TraceEvalCase) -> TraceEvalCaseReport:
        trace = await self.trace_store.load(case.run_id)
        if trace is None:
            issue = TraceEvalSuiteIssue(
                source=case.name,
                code="trace_missing",
                message=f"Trace not found for case: {case.name}",
                metadata={"run_id": case.run_id},
            )
            return TraceEvalCaseReport(
                case=case.manifest(),
                status="missing",
                issues=(issue,),
            )
        eval_report = self.evaluator.evaluate(trace, case.spec).manifest()
        issues = tuple(_suite_issues_from_eval(case, eval_report))
        status = "ready" if not issues else "blocked"
        return TraceEvalCaseReport(
            case=case.manifest(),
            status=status,
            eval_report=eval_report,
            issues=issues,
        )

    def manifest(self) -> dict[str, Any]:
        evaluator_manifest = getattr(self.evaluator, "manifest", None)
        trace_store_manifest = getattr(self.trace_store, "manifest", None)
        return {
            "schema_version": "agent-core-trace-eval-suite-runner/v1",
            "evaluator": evaluator_manifest() if callable(evaluator_manifest) else {},
            "trace_store": trace_store_manifest() if callable(trace_store_manifest) else {},
        }


@dataclass(frozen=True)
class AgentCoreEvalSuiteAcceptanceReport:
    """Prompt-safe acceptance report for the generic eval suite runner."""

    status: str
    suite_report: dict[str, Any] = field(default_factory=dict)
    missing_case_report: dict[str, Any] = field(default_factory=dict)
    issues: tuple[TraceEvalSuiteIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-eval-suite-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "suite_report": dict(self.suite_report),
            "missing_case_report": dict(self.missing_case_report),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreEvalSuiteAcceptanceHarness:
    """Acceptance gate for the generic trace eval suite runner."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreEvalSuiteAcceptanceReport:
        store = InMemoryRunTraceStore()
        await store.save(_suite_trace("suite-code", provider="code-provider", tool="diff"))
        await store.save(_suite_trace("suite-security", provider="security-provider", tool="scan"))
        runner = TraceEvalSuiteRunner(trace_store=store)
        suite = TraceEvalSuite(
            name="agent-core-generic-regression-suite",
            cases=(
                TraceEvalCase(
                    name="code-regression",
                    run_id="suite-code",
                    spec=TraceEvalSpec(
                        name="code-regression",
                        expected_status="completed",
                        max_provider_calls=1,
                        required_provider_names=("code-provider",),
                        required_tool_names=("diff",),
                        require_journal_ok=True,
                    ),
                    tags=("code", "regression"),
                ),
                TraceEvalCase(
                    name="security-regression",
                    run_id="suite-security",
                    spec=TraceEvalSpec(
                        name="security-regression",
                        expected_status="completed",
                        max_provider_calls=1,
                        required_provider_names=("security-provider",),
                        required_tool_names=("scan",),
                        require_journal_ok=True,
                    ),
                    tags=("security", "regression"),
                ),
            ),
        )
        suite_report = (await runner.evaluate(suite, tags=("regression",))).manifest()
        missing_report = (
            await runner.evaluate(
                TraceEvalSuite(
                    name="missing-trace-suite",
                    cases=(
                        TraceEvalCase(
                            name="missing-trace",
                            run_id="missing-run",
                            spec=TraceEvalSpec(name="missing-trace"),
                        ),
                    ),
                )
            )
        ).manifest()
        issues = _eval_suite_acceptance_issues(suite_report, missing_report)
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreEvalSuiteAcceptanceReport(
            status=status,
            suite_report=suite_report,
            missing_case_report=missing_report,
            issues=issues,
            metadata={"scenario": "agent_core_eval_suite_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_eval_suite_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreEvalSuiteAcceptanceReport:
    """Run the default generic eval suite acceptance gate."""

    return await AgentCoreEvalSuiteAcceptanceHarness(metadata=dict(metadata or {})).run()


def _suite_issues_from_eval(
    case: TraceEvalCase,
    eval_report: dict[str, Any],
) -> tuple[TraceEvalSuiteIssue, ...]:
    issues = []
    if eval_report.get("ok") is True:
        return ()
    for issue in eval_report.get("issues") or ():
        if not isinstance(issue, dict):
            continue
        issues.append(
            TraceEvalSuiteIssue(
                source=case.name,
                code=str(issue.get("code") or "trace_eval_issue"),
                message=str(issue.get("message") or "Trace eval issue."),
                severity=str(issue.get("severity") or "error"),
                metadata={
                    "run_id": case.run_id,
                    "spec_name": case.spec.name,
                    "trace_eval_issue": dict(issue),
                },
            )
        )
    if not issues:
        issues.append(
            TraceEvalSuiteIssue(
                source=case.name,
                code="trace_eval_failed",
                message="Trace eval failed without issue details.",
                metadata={"run_id": case.run_id, "spec_name": case.spec.name},
            )
        )
    return tuple(issues)


def _eval_suite_acceptance_issues(
    suite_report: dict[str, Any],
    missing_report: dict[str, Any],
) -> tuple[TraceEvalSuiteIssue, ...]:
    issues: list[TraceEvalSuiteIssue] = []
    if suite_report.get("ready") is not True:
        issues.append(
            TraceEvalSuiteIssue(
                source="suite_report",
                code="suite_not_ready",
                message="Generic eval suite did not pass.",
                metadata={"suite_report": suite_report},
            )
        )
    if int(suite_report.get("case_count") or 0) != 2:
        issues.append(
            TraceEvalSuiteIssue(
                source="suite_report",
                code="suite_case_count_unexpected",
                message="Generic eval suite did not evaluate two cases.",
                metadata={"case_count": suite_report.get("case_count")},
            )
        )
    counts = suite_report.get("status_counts") if isinstance(suite_report.get("status_counts"), dict) else {}
    if int(counts.get("ready") or 0) != 2:
        issues.append(
            TraceEvalSuiteIssue(
                source="suite_report",
                code="suite_ready_count_unexpected",
                message="Generic eval suite did not report two ready cases.",
                metadata={"status_counts": counts},
            )
        )
    missing_codes = {
        str(issue.get("code") or "")
        for issue in missing_report.get("issues") or ()
        if isinstance(issue, dict)
    }
    if missing_report.get("status") != "blocked" or "trace_missing" not in missing_codes:
        issues.append(
            TraceEvalSuiteIssue(
                source="missing_case_report",
                code="missing_trace_not_reported",
                message="Eval suite did not report missing trace as a blocked case.",
                metadata={"missing_case_report": missing_report},
            )
        )
    return tuple(issues)


def _suite_trace(run_id: str, *, provider: str, tool: str) -> dict[str, Any]:
    return {
        "schema_version": "agent-core-run-trace-bundle/v1",
        "run": {
            "run_id": run_id,
            "status": "completed",
            "iterations": 1,
            "output_bytes": 16,
        },
        "summary": {
            "status": "completed",
            "journal_ok": True,
            "journal_event_count": 2,
            "provider_call_count": 1,
            "tool_replay_record_count": 1,
        },
        "journal_replay": {
            "ok": True,
            "events": [
                {
                    "sequence": 1,
                    "event_type": "run_started",
                    "run_id": run_id,
                    "payload": {"run_id": run_id},
                },
                {
                    "sequence": 2,
                    "event_type": "run_finished",
                    "run_id": run_id,
                    "payload": {"run_id": run_id, "status": "completed"},
                },
            ],
        },
        "provider": {
            "call_count": 1,
            "calls": [
                {
                    "provider_name": provider,
                    "model": f"{provider}-mini",
                    "status": "completed",
                    "usage": {"total_tokens": 8, "cost_usd": 0.0},
                    "metadata": {},
                }
            ],
        },
        "tool_replay": {
            "record_count": 1,
            "records": [
                {
                    "invocation": {
                        "tool_name": tool,
                        "arguments": {},
                    },
                    "result": {
                        "tool_name": tool,
                        "status": "completed",
                        "call_id": f"{run_id}-{tool}",
                        "metadata": {},
                    },
                    "metadata": {},
                }
            ],
        },
    }
