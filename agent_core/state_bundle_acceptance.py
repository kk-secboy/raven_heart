"""Acceptance gate for portable SDK state bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.state_bundle import (
    AgentStateBundleBuilder,
    AgentStateBundlePolicy,
)


@dataclass(frozen=True)
class AgentCoreStateBundleAcceptanceIssue:
    """One state bundle acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-state-bundle-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreStateBundleAcceptanceReport:
    """Prompt-safe state bundle acceptance report."""

    status: str
    ready_bundle: dict[str, Any] = field(default_factory=dict)
    blocked_bundle: dict[str, Any] = field(default_factory=dict)
    leak_scan: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreStateBundleAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-state-bundle-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "ready_bundle": dict(self.ready_bundle),
            "blocked_bundle": dict(self.blocked_bundle),
            "leak_scan": dict(self.leak_scan),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreStateBundleAcceptanceHarness:
    """Run deterministic state bundle portability checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreStateBundleAcceptanceReport:
        secrets = (
            "sk-state-bundle-secret-1234567890",
            "Bearer state-bundle-token-123456",
            "password=state-bundle-password-123456",
        )
        components = _state_components(secrets)
        policy = AgentStateBundlePolicy(
            required_roles=(
                "journal",
                "run_state",
                "memory",
                "event_log",
                "run_trace",
            )
        )
        ready_bundle = AgentStateBundleBuilder(policy=policy).build(
            components,
            metadata={"scenario": "state_bundle_acceptance_ready"},
        ).manifest()
        blocked_bundle = AgentStateBundleBuilder(
            policy=AgentStateBundlePolicy(required_roles=("journal", "artifact"))
        ).build(
            {"journal": components["journal"]},
            metadata={"scenario": "state_bundle_acceptance_blocked"},
        ).manifest()
        leak_scan = _scan_for_leaks(ready_bundle, secrets=secrets)
        issues = _state_bundle_acceptance_issues(
            ready_bundle=ready_bundle,
            blocked_bundle=blocked_bundle,
            leak_scan=leak_scan,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreStateBundleAcceptanceReport(
            status=status,
            ready_bundle=ready_bundle,
            blocked_bundle=blocked_bundle,
            leak_scan=leak_scan,
            issues=issues,
            metadata={"scenario": "agent_core_state_bundle_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_state_bundle_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreStateBundleAcceptanceReport:
    """Run deterministic state bundle acceptance checks."""

    return await AgentCoreStateBundleAcceptanceHarness(
        metadata=dict(metadata or {})
    ).run()


def _state_components(secrets: tuple[str, str, str]) -> dict[str, dict[str, Any]]:
    return {
        "journal": {
            "schema_version": "agent-core-journal/v1",
            "backend": {"role": "journal", "kind": "sqlite"},
            "runs": [{"run_id": "run-1", "status": "completed"}],
            "checkpoints": [{"run_id": "run-1", "checkpoint_id": "cp-1"}],
            "metadata": {"api_key": secrets[0]},
        },
        "run_state": {
            "schema_version": "agent-core-run-store/v1",
            "backend": {"role": "run_state", "kind": "markdown"},
            "runs": [{"run_id": "run-1", "session_name": "demo", "status": "completed"}],
        },
        "memory": {
            "schema_version": "agent-core-memory-store/v1",
            "backend": {"role": "memory", "kind": "external", "external": True},
            "record_count": 1,
            "records": [{"source": "seed", "content": "public memory"}],
            "metadata": {"authorization": secrets[1]},
        },
        "event_log": {
            "schema_version": "agent-core-event-log/v1",
            "backend": {"role": "event_log", "kind": "sqlite"},
            "events": [
                {"sequence": 1, "type": "run_started"},
                {"sequence": 2, "type": "run_finished"},
            ],
        },
        "trace": {
            "schema_version": "agent-core-run-trace-bundle/v1",
            "backend": {"role": "run_trace", "kind": "markdown"},
            "run": {"run_id": "run-1", "status": "completed"},
            "provider": {"metadata": {"note": secrets[2]}},
        },
    }


def _scan_for_leaks(
    bundle: dict[str, Any],
    *,
    secrets: tuple[str, ...],
) -> dict[str, Any]:
    rendered = str(bundle)
    leaked = tuple(secret for secret in secrets if secret in rendered)
    return {
        "schema_version": "agent-core-state-bundle-leak-scan/v1",
        "leaked_count": len(leaked),
        "leaked_secrets": list(leaked),
        "decision_count": int(bundle.get("redacted_count") or 0),
        "decision_reasons": sorted(
            {
                str(decision.get("reason") or "")
                for decision in bundle.get("redaction", {}).get("decisions", ())
                if isinstance(decision, dict)
            }
        ),
    }


def _state_bundle_acceptance_issues(
    *,
    ready_bundle: dict[str, Any],
    blocked_bundle: dict[str, Any],
    leak_scan: dict[str, Any],
) -> tuple[AgentCoreStateBundleAcceptanceIssue, ...]:
    issues: list[AgentCoreStateBundleAcceptanceIssue] = []
    if ready_bundle.get("ready") is not True:
        issues.append(
            AgentCoreStateBundleAcceptanceIssue(
                source="ready_bundle",
                code="ready_bundle_not_ready",
                message="State bundle ready scenario did not pass.",
                metadata={"issues": ready_bundle.get("issues")},
            )
        )
    if int(ready_bundle.get("component_count") or 0) != 5:
        issues.append(
            AgentCoreStateBundleAcceptanceIssue(
                source="ready_bundle",
                code="component_count_unexpected",
                message="State bundle did not include expected component count.",
                metadata={"component_count": ready_bundle.get("component_count")},
            )
        )
    restore_roles = [
        str(step.get("role") or "")
        for step in ready_bundle.get("restore_plan", {}).get("steps", ())
        if isinstance(step, dict)
    ]
    expected_prefix = ["journal", "run_state", "memory"]
    if restore_roles[:3] != expected_prefix:
        issues.append(
            AgentCoreStateBundleAcceptanceIssue(
                source="ready_bundle",
                code="restore_order_unexpected",
                message="State bundle restore order did not prioritize core state roles.",
                metadata={"restore_roles": restore_roles, "expected_prefix": expected_prefix},
            )
        )
    if int(leak_scan.get("leaked_count") or 0) != 0:
        issues.append(
            AgentCoreStateBundleAcceptanceIssue(
                source="leak_scan",
                code="secret_leak_detected",
                message="State bundle still contains secret material.",
                metadata={"leak_scan": leak_scan},
            )
        )
    if int(ready_bundle.get("redacted_count") or 0) < 3:
        issues.append(
            AgentCoreStateBundleAcceptanceIssue(
                source="redaction",
                code="redaction_count_unexpected",
                message="State bundle did not redact all expected sensitive fields.",
                metadata={"redacted_count": ready_bundle.get("redacted_count")},
            )
        )
    if blocked_bundle.get("status") != "blocked":
        issues.append(
            AgentCoreStateBundleAcceptanceIssue(
                source="blocked_bundle",
                code="blocked_bundle_not_blocked",
                message="Missing required state role did not block the bundle.",
                metadata={"status": blocked_bundle.get("status")},
            )
        )
    blocked_codes = {
        str(issue.get("code") or "")
        for issue in blocked_bundle.get("issues") or ()
        if isinstance(issue, dict)
    }
    if "required_state_role_missing" not in blocked_codes:
        issues.append(
            AgentCoreStateBundleAcceptanceIssue(
                source="blocked_bundle",
                code="required_role_issue_missing",
                message="Blocked bundle did not report the missing required role.",
                metadata={"blocked_codes": sorted(blocked_codes)},
            )
        )
    return tuple(issues)
