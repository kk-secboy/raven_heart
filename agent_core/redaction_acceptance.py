"""Acceptance gate for prompt-safe redaction contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.redaction import RedactionPolicy, redact_payload


@dataclass(frozen=True)
class AgentCoreRedactionAcceptanceIssue:
    """One redaction acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-redaction-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreRedactionAcceptanceReport:
    """Prompt-safe redaction acceptance report."""

    status: str
    redaction: dict[str, Any] = field(default_factory=dict)
    leak_scan: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreRedactionAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-redaction-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "redaction": dict(self.redaction),
            "leak_scan": dict(self.leak_scan),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreRedactionAcceptanceHarness:
    """Run deterministic SDK prompt-safe redaction checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreRedactionAcceptanceReport:
        secrets = {
            "api_key": "sk-test-secret-value-1234567890",
            "authorization": "Bearer raven-secret-token-123456",
            "password": "correct-horse-battery-staple",
            "tool_secret": "token=tool-secret-value-123456",
        }
        payload = {
            "request": {
                "url": "https://example.test/api",
                "headers": {
                    "Authorization": secrets["authorization"],
                    "X-API-Key": secrets["api_key"],
                },
                "metadata": {
                    "tenant": "demo",
                    "password": secrets["password"],
                },
            },
            "tool": {
                "name": "lookup",
                "arguments": {
                    "query": "public query",
                    "secret_note": secrets["tool_secret"],
                },
            },
            "notes": [
                "safe observation",
                "inline token=inline-secret-value-123456",
            ],
        }
        result = redact_payload(
            payload,
            policy=RedactionPolicy(max_string_bytes=256),
            metadata={"scenario": "redaction_acceptance"},
        ).manifest()
        leak_scan = _scan_for_leaks(result, secrets=tuple(secrets.values()))
        issues = _redaction_acceptance_issues(result, leak_scan)
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreRedactionAcceptanceReport(
            status=status,
            redaction=result,
            leak_scan=leak_scan,
            issues=issues,
            metadata={"scenario": "agent_core_redaction_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_redaction_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreRedactionAcceptanceReport:
    """Run deterministic prompt-safe redaction acceptance checks."""

    return await AgentCoreRedactionAcceptanceHarness(metadata=dict(metadata or {})).run()


def _scan_for_leaks(payload: dict[str, Any], *, secrets: tuple[str, ...]) -> dict[str, Any]:
    rendered = str(payload)
    leaked = tuple(secret for secret in secrets if secret in rendered)
    inline_leak = "inline-secret-value-123456" in rendered
    return {
        "schema_version": "agent-core-redaction-leak-scan/v1",
        "leaked_count": len(leaked) + int(inline_leak),
        "leaked_secret_count": len(leaked),
        "inline_secret_leaked": inline_leak,
        "decision_count": int(payload.get("redacted_count") or 0),
        "decision_reasons": sorted(
            {
                str(decision.get("reason") or "")
                for decision in payload.get("decisions", ())
                if isinstance(decision, dict)
            }
        ),
    }


def _redaction_acceptance_issues(
    redaction: dict[str, Any],
    leak_scan: dict[str, Any],
) -> tuple[AgentCoreRedactionAcceptanceIssue, ...]:
    issues: list[AgentCoreRedactionAcceptanceIssue] = []
    if redaction.get("redacted") is not True or int(redaction.get("redacted_count") or 0) < 5:
        issues.append(
            AgentCoreRedactionAcceptanceIssue(
                source="redaction",
                code="redaction_count_unexpected",
                message="Redaction policy did not redact the expected sensitive payload fields.",
                metadata={"redaction": redaction},
            )
        )
    if int(leak_scan.get("leaked_count") or 0) != 0:
        issues.append(
            AgentCoreRedactionAcceptanceIssue(
                source="leak_scan",
                code="secret_leak_detected",
                message="Prompt-safe redaction result still contains secret material.",
                metadata={"leak_scan": leak_scan},
            )
        )
    reasons = set(str(item) for item in leak_scan.get("decision_reasons") or ())
    if not {"sensitive_key", "secret_value"} <= reasons:
        issues.append(
            AgentCoreRedactionAcceptanceIssue(
                source="redaction",
                code="redaction_reason_missing",
                message="Redaction did not report both key-based and value-based decisions.",
                metadata={"decision_reasons": sorted(reasons)},
            )
        )
    return tuple(issues)
