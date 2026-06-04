"""Error types for the provider-neutral agent core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


FailureKind = Literal[
    "unknown",
    "schema_invalid",
    "exception",
    "tool_failed",
    "provider_failed",
    "policy_denied",
    "timeout",
    "cancelled",
    "budget_exceeded",
]


@dataclass(frozen=True)
class ErrorClassification:
    """Prompt-safe failure classification shared by SDK audit records."""

    stage: str = ""
    kind: FailureKind = "unknown"
    status: str = ""
    retryable: bool = False
    exception_type: str = ""
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-error-classification/v1",
            "stage": self.stage,
            "kind": self.kind,
            "status": self.status,
            "retryable": self.retryable,
            "exception_type": self.exception_type,
            "message": self.message,
            "metadata": dict(self.metadata),
        }


def classify_error(
    *,
    stage: str = "",
    status: str = "",
    message: str = "",
    retryable: bool = False,
    exception_type: str = "",
    metadata: dict[str, Any] | None = None,
) -> ErrorClassification:
    data = dict(metadata or {})
    explicit_kind = str(data.get("failure_kind") or data.get("kind") or "")
    kind = _failure_kind_from_text(explicit_kind)
    if kind == "unknown":
        kind = _infer_failure_kind(
            stage=stage,
            status=status,
            exception_type=exception_type,
            message=message,
        )
    return ErrorClassification(
        stage=stage,
        kind=kind,
        status=status,
        retryable=bool(retryable),
        exception_type=exception_type,
        message=message,
        metadata=data,
    )


def _failure_kind_from_text(value: str) -> FailureKind:
    allowed = {
        "schema_invalid",
        "exception",
        "tool_failed",
        "provider_failed",
        "policy_denied",
        "timeout",
        "cancelled",
        "budget_exceeded",
    }
    return value if value in allowed else "unknown"  # type: ignore[return-value]


def _infer_failure_kind(
    *,
    stage: str,
    status: str,
    exception_type: str,
    message: str,
) -> FailureKind:
    lowered = f"{stage} {status} {exception_type} {message}".lower()
    if "schema" in lowered or status == "schema_invalid":
        return "schema_invalid"
    if "timeout" in lowered:
        return "timeout"
    if "cancel" in lowered:
        return "cancelled"
    if "budget" in lowered or "limit" in lowered:
        return "budget_exceeded"
    if "denied" in lowered or "policy" in lowered or "approval" in lowered:
        return "policy_denied"
    if exception_type:
        return "exception"
    if "provider" in lowered or stage == "provider":
        return "provider_failed"
    if "tool" in lowered or stage == "tool":
        return "tool_failed"
    return "unknown"


class AgentCoreError(Exception):
    """Base class for all agent core failures."""


class ProviderError(AgentCoreError):
    """LLM provider returned an error or unusable response."""


class ActionError(AgentCoreError):
    """Model output could not be parsed or validated as an action."""


class ToolError(AgentCoreError):
    """Tool execution failed at the core protocol boundary."""


class HarnessError(AgentCoreError):
    """Run lifecycle, checkpoint, or resume state failed."""


class ResumeError(HarnessError):
    """A checkpoint could not be resumed safely."""


class PolicyError(AgentCoreError):
    """A policy decision denied or interrupted an agent operation."""

