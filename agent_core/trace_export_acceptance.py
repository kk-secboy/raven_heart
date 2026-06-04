"""Acceptance gate for prompt-safe trace export bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.trace import AgentRunTraceBundle
from agent_core.trace_export import TraceExportBuilder, TraceExportPolicy


@dataclass(frozen=True)
class AgentCoreTraceExportAcceptanceIssue:
    """One trace export acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-export-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreTraceExportAcceptanceReport:
    """Prompt-safe trace export acceptance report."""

    status: str
    export_bundle: dict[str, Any] = field(default_factory=dict)
    leak_scan: dict[str, Any] = field(default_factory=dict)
    trace_eval: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreTraceExportAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-export-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "export_bundle": dict(self.export_bundle),
            "leak_scan": dict(self.leak_scan),
            "trace_eval": dict(self.trace_eval),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreTraceExportAcceptanceHarness:
    """Run deterministic trace export safety checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreTraceExportAcceptanceReport:
        secrets = (
            "sk-trace-export-secret-1234567890",
            "Bearer trace-export-token-123456",
            "token=tool-export-secret-123456",
            "password=trace-export-password-123456",
            "token=trace-export-inline-secret-123456",
        )
        trace = AgentRunTraceBundle(
            run_id="trace-export-run",
            status="completed",
            iterations=1,
            output_bytes=2,
            prompt={
                "schema_version": "agent-core-prompt-ir/v1",
                "metadata": {
                    "api_key": secrets[0],
                    "public_note": "inline token=trace-export-inline-secret-123456",
                    "context_injections": [
                        {
                            "name": "memory",
                            "source": "memory",
                            "target": "user",
                            "status": "included",
                            "included": True,
                        }
                    ],
                },
            },
            journal_replay={
                "ok": True,
                "event_count": 2,
                "events": [
                    {
                        "sequence": 1,
                        "event_type": "run_started",
                        "run_id": "trace-export-run",
                        "payload": {"run_id": "trace-export-run"},
                    },
                    {
                        "sequence": 2,
                        "event_type": "run_finished",
                        "run_id": "trace-export-run",
                        "payload": {"run_id": "trace-export-run", "status": "completed"},
                    },
                ],
            },
            provider={
                "call_count": 1,
                "calls": [
                    {
                        "provider_name": "mock",
                        "model": "mock-mini",
                        "status": "completed",
                        "metadata": {
                            "request": {
                                "headers": {
                                    "Authorization": secrets[1],
                                    "X-API-Key": secrets[0],
                                },
                                "metadata": {"run_id": "trace-export-run"},
                            }
                        },
                    }
                ],
            },
            tool_replay={
                "record_count": 1,
                "records": [
                    {
                        "invocation": {
                            "tool_name": "lookup",
                            "arguments": {
                                "query": "safe",
                                "secret_note": secrets[2],
                            },
                        },
                        "result": {"status": "completed", "tool_name": "lookup"},
                    }
                ],
            },
            session={
                "name": "demo",
                "metadata": {"operator_password": secrets[3]},
            },
        ).manifest()
        exported = TraceExportBuilder(policy=TraceExportPolicy()).export(
            {"run_trace": trace},
            metadata={"scenario": "trace_export_acceptance"},
        ).manifest()
        exported_trace = exported["records"][0]["payload"]
        trace_eval = DefaultTraceEvaluator().evaluate(
            exported_trace,
            TraceEvalSpec(
                expected_status="completed",
                max_provider_calls=1,
                require_journal_ok=True,
                required_event_types=("run_started", "run_finished"),
            ),
        ).manifest()
        leak_scan = _scan_export_for_leaks(exported, secrets=secrets)
        issues = _trace_export_acceptance_issues(exported, leak_scan, trace_eval)
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreTraceExportAcceptanceReport(
            status=status,
            export_bundle=exported,
            leak_scan=leak_scan,
            trace_eval=trace_eval,
            issues=tuple(issues),
            metadata={"scenario": "agent_core_trace_export_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_trace_export_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreTraceExportAcceptanceReport:
    """Run deterministic trace export acceptance checks."""

    return await AgentCoreTraceExportAcceptanceHarness(metadata=dict(metadata or {})).run()


def _scan_export_for_leaks(
    export_bundle: dict[str, Any],
    *,
    secrets: tuple[str, ...],
) -> dict[str, Any]:
    rendered = str(export_bundle)
    leaked = tuple(secret for secret in secrets if secret in rendered)
    reasons = sorted(
        {
            str(decision.get("reason") or "")
            for decision in export_bundle.get("redaction", {}).get("decisions", ())
            if isinstance(decision, dict)
        }
    )
    return {
        "schema_version": "agent-core-trace-export-leak-scan/v1",
        "leaked_count": len(leaked),
        "leaked_secrets": list(leaked),
        "decision_count": int(export_bundle.get("redacted_count") or 0),
        "decision_reasons": reasons,
    }


def _trace_export_acceptance_issues(
    export_bundle: dict[str, Any],
    leak_scan: dict[str, Any],
    trace_eval: dict[str, Any],
) -> tuple[AgentCoreTraceExportAcceptanceIssue, ...]:
    issues: list[AgentCoreTraceExportAcceptanceIssue] = []
    if export_bundle.get("prompt_safe") is not True:
        issues.append(
            AgentCoreTraceExportAcceptanceIssue(
                source="export_bundle",
                code="export_not_prompt_safe",
                message="Trace export bundle is not marked prompt-safe.",
            )
        )
    if int(export_bundle.get("record_count") or 0) != 1:
        issues.append(
            AgentCoreTraceExportAcceptanceIssue(
                source="export_bundle",
                code="record_count_unexpected",
                message="Trace export bundle did not contain the expected record count.",
                metadata={"record_count": export_bundle.get("record_count")},
            )
        )
    if int(export_bundle.get("redacted_count") or 0) < 4:
        issues.append(
            AgentCoreTraceExportAcceptanceIssue(
                source="export_bundle",
                code="redaction_count_unexpected",
                message="Trace export did not redact expected sensitive trace fields.",
                metadata={"redacted_count": export_bundle.get("redacted_count")},
            )
        )
    if int(leak_scan.get("leaked_count") or 0) != 0:
        issues.append(
            AgentCoreTraceExportAcceptanceIssue(
                source="leak_scan",
                code="secret_leak_detected",
                message="Trace export bundle still contains secret material.",
                metadata={"leak_scan": leak_scan},
            )
        )
    reasons = set(str(item) for item in leak_scan.get("decision_reasons") or ())
    if not {"sensitive_key", "secret_value"} <= reasons:
        issues.append(
            AgentCoreTraceExportAcceptanceIssue(
                source="redaction",
                code="redaction_reason_missing",
                message="Trace export did not report both key-based and value-based redaction.",
                metadata={"decision_reasons": sorted(reasons)},
            )
        )
    if trace_eval.get("ok") is not True:
        issues.append(
            AgentCoreTraceExportAcceptanceIssue(
                source="trace_eval",
                code="exported_trace_eval_failed",
                message="Exported redacted trace no longer satisfies basic trace eval.",
                metadata={"trace_eval": trace_eval},
            )
        )
    return tuple(issues)
