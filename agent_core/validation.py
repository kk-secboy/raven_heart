"""Aggregate SDK validation suite for host-runtime migration gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.acceptance import run_agent_core_acceptance
from agent_core.manifest import (
    AgentCoreSDKManifest,
    agent_core_sdk_manifest,
    evaluate_agent_core_api_stability,
    evaluate_agent_core_readiness,
)
from agent_core.recovery import run_agent_core_recovery_acceptance
from agent_core.resume_acceptance import run_agent_core_resume_acceptance


@dataclass(frozen=True)
class AgentCoreValidationIssue:
    """One blocking issue from the aggregate SDK validation suite."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-validation-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreValidationReport:
    """Prompt-safe aggregate SDK validation report."""

    status: str
    readiness: dict[str, Any] = field(default_factory=dict)
    api_stability: dict[str, Any] = field(default_factory=dict)
    acceptance: dict[str, Any] = field(default_factory=dict)
    recovery: dict[str, Any] = field(default_factory=dict)
    resume: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreValidationIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-validation-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "readiness": dict(self.readiness),
            "api_stability": dict(self.api_stability),
            "acceptance": dict(self.acceptance),
            "recovery": dict(self.recovery),
            "resume": dict(self.resume),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreValidationSuite:
    """Run all pure-SDK gates needed before host-runtime adapter migration."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(
        self,
        *,
        sdk_manifest: AgentCoreSDKManifest | dict[str, Any] | None = None,
    ) -> AgentCoreValidationReport:
        manifest = sdk_manifest or agent_core_sdk_manifest()
        readiness = evaluate_agent_core_readiness(manifest).manifest()
        api_stability = evaluate_agent_core_api_stability(manifest).manifest()
        acceptance = (
            await run_agent_core_acceptance(
                sdk_manifest=manifest,
                metadata={"validation_gate": "acceptance", **dict(self.metadata)},
            )
        ).manifest()
        recovery = (
            await run_agent_core_recovery_acceptance(
                metadata={"validation_gate": "recovery", **dict(self.metadata)}
            )
        ).manifest()
        resume = (
            await run_agent_core_resume_acceptance(
                metadata={"validation_gate": "resume", **dict(self.metadata)}
            )
        ).manifest()
        issues = _validation_issues(
            readiness=readiness,
            api_stability=api_stability,
            acceptance=acceptance,
            recovery=recovery,
            resume=resume,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreValidationReport(
            status=status,
            readiness=readiness,
            api_stability=api_stability,
            acceptance=acceptance,
            recovery=recovery,
            resume=resume,
            issues=issues,
            metadata={"scenario": "agent_core_validation", **dict(self.metadata)},
        )


async def run_agent_core_validation(
    *,
    sdk_manifest: AgentCoreSDKManifest | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreValidationReport:
    """Run the default aggregate SDK validation suite."""

    return await AgentCoreValidationSuite(metadata=dict(metadata or {})).run(
        sdk_manifest=sdk_manifest
    )


def _validation_issues(
    *,
    readiness: dict[str, Any],
    api_stability: dict[str, Any],
    acceptance: dict[str, Any],
    recovery: dict[str, Any],
    resume: dict[str, Any],
) -> tuple[AgentCoreValidationIssue, ...]:
    issues: list[AgentCoreValidationIssue] = []
    _extend_report_issues(issues, source="readiness", report=readiness)
    _extend_api_stability_issues(issues, api_stability)
    _extend_report_issues(issues, source="acceptance", report=acceptance)
    _extend_report_issues(issues, source="recovery", report=recovery)
    _extend_report_issues(issues, source="resume", report=resume)
    return tuple(issues)


def _extend_report_issues(
    issues: list[AgentCoreValidationIssue],
    *,
    source: str,
    report: dict[str, Any],
) -> None:
    if report.get("ready") is True:
        return
    for issue in report.get("issues") or ():
        if not isinstance(issue, dict):
            continue
        issues.append(
            AgentCoreValidationIssue(
                source=source,
                code=str(issue.get("code") or f"{source}_issue"),
                message=str(issue.get("message") or ""),
                severity=str(issue.get("severity") or "error"),
                metadata=dict(issue.get("metadata") or {}),
            )
        )
    if not any(issue.source == source for issue in issues):
        issues.append(
            AgentCoreValidationIssue(
                source=source,
                code=f"{source}_not_ready",
                message=f"{source} gate is not ready.",
                metadata={"status": report.get("status")},
            )
        )


def _extend_api_stability_issues(
    issues: list[AgentCoreValidationIssue],
    report: dict[str, Any],
) -> None:
    if report.get("ready") is True:
        return
    missing = tuple(str(name) for name in report.get("missing_stable_api") or ())
    for name in missing:
        issues.append(
            AgentCoreValidationIssue(
                source="api_stability",
                code="stable_api_missing",
                message=f"Stable API is missing: {name}",
                metadata={"api_name": name},
            )
        )
    if not missing:
        issues.append(
            AgentCoreValidationIssue(
                source="api_stability",
                code="api_stability_not_ready",
                message="API stability gate is not ready.",
                metadata={"status": report.get("status")},
            )
        )
