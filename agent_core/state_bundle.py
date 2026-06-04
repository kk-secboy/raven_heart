"""Portable SDK state bundles for migration, archive, and recovery preflight."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from agent_core.redaction import RedactionPolicy, redact_payload


_DEFAULT_RESTORE_ORDER: tuple[str, ...] = (
    "journal",
    "run_state",
    "planner_state",
    "approval",
    "policy_decision",
    "memory",
    "context_material",
    "tool_replay",
    "event_log",
    "run_trace",
    "artifact",
)


@dataclass(frozen=True)
class AgentStateBundlePolicy:
    """Policy for building a prompt-safe portable SDK state bundle."""

    required_roles: tuple[str, ...] = ()
    redact: bool = True
    redaction_policy: RedactionPolicy = field(default_factory=RedactionPolicy)
    restore_order: tuple[str, ...] = _DEFAULT_RESTORE_ORDER
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-state-bundle-policy/v1",
            "required_roles": list(self.required_roles),
            "redact": self.redact,
            "redaction_policy": self.redaction_policy.manifest(),
            "restore_order": list(self.restore_order),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentStateBundleIssue:
    """One state bundle policy issue."""

    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-state-bundle-issue/v1",
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentStateBundleComponent:
    """One SDK state component in a portable state bundle."""

    name: str
    role: str
    kind: str
    payload: dict[str, Any]
    payload_schema_version: str = ""
    original_bytes: int = 0
    exported_bytes: int = 0
    original_sha256: str = ""
    exported_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-state-bundle-component/v1",
            "name": self.name,
            "role": self.role,
            "kind": self.kind,
            "payload_schema_version": self.payload_schema_version,
            "original_bytes": self.original_bytes,
            "exported_bytes": self.exported_bytes,
            "original_sha256": self.original_sha256,
            "exported_sha256": self.exported_sha256,
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentStateBundleRestorePlan:
    """Role-ordered restore preflight plan for a state bundle."""

    steps: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-state-bundle-restore-plan/v1",
            "step_count": self.step_count,
            "steps": [dict(step) for step in self.steps],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentStateBundle:
    """Prompt-safe portable bundle of generic SDK state manifests."""

    status: str
    components: tuple[AgentStateBundleComponent, ...] = ()
    policy: AgentStateBundlePolicy = field(default_factory=AgentStateBundlePolicy)
    restore_plan: AgentStateBundleRestorePlan = field(default_factory=AgentStateBundleRestorePlan)
    redaction: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentStateBundleIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        roles = _count_components(self.components, "role")
        kinds = _count_components(self.components, "kind")
        return {
            "schema_version": "agent-core-state-bundle/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "component_count": len(self.components),
            "roles": roles,
            "kinds": kinds,
            "original_bytes": sum(component.original_bytes for component in self.components),
            "exported_bytes": sum(component.exported_bytes for component in self.components),
            "redacted_count": int(self.redaction.get("redacted_count") or 0),
            "components": [component.manifest() for component in self.components],
            "restore_plan": self.restore_plan.manifest(),
            "policy": self.policy.manifest(),
            "redaction": dict(self.redaction),
            "issues": [issue.manifest() for issue in self.issues],
            "metadata": dict(self.metadata),
        }


class AgentStateBundleBuilder:
    """Build prompt-safe portable state bundles from SDK component manifests."""

    def __init__(self, *, policy: AgentStateBundlePolicy | None = None) -> None:
        self.policy = policy or AgentStateBundlePolicy()

    def build(
        self,
        components: dict[str, dict[str, Any]] | tuple[dict[str, Any], ...] | list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AgentStateBundle:
        raw_components = _normalize_components(components)
        payloads = {name: dict(payload) for name, payload in raw_components}
        measurements = {name: _measure_payload(payload) for name, payload in raw_components}
        redaction = (
            redact_payload(payloads, policy=self.policy.redaction_policy)
            if self.policy.redact
            else redact_payload(
                payloads,
                policy=RedactionPolicy(
                    sensitive_key_patterns=(),
                    redact_secret_values=False,
                    max_string_bytes=2**31 - 1,
                ),
                metadata={"redaction_disabled": True},
            )
        )
        exported_payloads = redaction.payload if isinstance(redaction.payload, dict) else {}
        bundle_components: list[AgentStateBundleComponent] = []
        for name, original_payload in raw_components:
            exported_payload = exported_payloads.get(name)
            if not isinstance(exported_payload, dict):
                exported_payload = {}
            original_bytes, original_sha = measurements[name]
            exported_bytes, exported_sha = _measure_payload(exported_payload)
            role, kind = _component_role_kind(original_payload)
            bundle_components.append(
                AgentStateBundleComponent(
                    name=name,
                    role=role,
                    kind=kind,
                    payload=exported_payload,
                    payload_schema_version=str(original_payload.get("schema_version") or ""),
                    original_bytes=original_bytes,
                    exported_bytes=exported_bytes,
                    original_sha256=original_sha,
                    exported_sha256=exported_sha,
                )
            )
        ordered = tuple(sorted(bundle_components, key=self._component_sort_key))
        restore_plan = AgentStateBundleRestorePlan(
            steps=_restore_steps(ordered, self.policy),
            metadata={"policy_restore_order": list(self.policy.restore_order)},
        )
        issues = _bundle_issues(ordered, self.policy)
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentStateBundle(
            status=status,
            components=ordered,
            policy=self.policy,
            restore_plan=restore_plan,
            redaction=_redaction_audit(redaction.manifest()),
            issues=issues,
            metadata={"scenario": "state_bundle", **dict(metadata or {})},
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-state-bundle-builder/v1",
            "policy": self.policy.manifest(),
        }

    def _component_sort_key(self, component: AgentStateBundleComponent) -> tuple[int, str, str]:
        try:
            order = self.policy.restore_order.index(component.role)
        except ValueError:
            order = len(self.policy.restore_order)
        return (order, component.role, component.name)


def build_agent_state_bundle(
    components: dict[str, dict[str, Any]] | tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    policy: AgentStateBundlePolicy | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentStateBundle:
    """Build a prompt-safe portable SDK state bundle."""

    return AgentStateBundleBuilder(policy=policy).build(components, metadata=metadata)


def _normalize_components(
    components: dict[str, dict[str, Any]] | tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    if isinstance(components, dict):
        return tuple(
            (str(name), dict(payload))
            for name, payload in components.items()
            if isinstance(payload, dict)
        )
    records: list[tuple[str, dict[str, Any]]] = []
    for index, payload in enumerate(components, start=1):
        if not isinstance(payload, dict):
            continue
        role, _kind = _component_role_kind(payload)
        records.append((str(payload.get("name") or role or f"component-{index}"), dict(payload)))
    return tuple(records)


def _component_role_kind(payload: dict[str, Any]) -> tuple[str, str]:
    backend = payload.get("backend") if isinstance(payload.get("backend"), dict) else {}
    role = str(payload.get("role") or backend.get("role") or "")
    kind = str(payload.get("kind") or backend.get("kind") or "")
    if not role:
        schema = str(payload.get("schema_version") or "")
        role = _role_from_schema(schema)
    if not kind:
        kind = "manifest"
    return role, kind


def _role_from_schema(schema_version: str) -> str:
    schema = schema_version.lower()
    if "journal" in schema:
        return "journal"
    if "run-trace" in schema or "trace" in schema:
        return "run_trace"
    if "memory" in schema:
        return "memory"
    if "event" in schema:
        return "event_log"
    if "approval" in schema:
        return "approval"
    if "policy" in schema:
        return "policy_decision"
    if "planner" in schema or "plan" in schema:
        return "planner_state"
    if "tool-replay" in schema or "tool" in schema:
        return "tool_replay"
    if "artifact" in schema:
        return "artifact"
    return "unknown"


def _restore_steps(
    components: tuple[AgentStateBundleComponent, ...],
    policy: AgentStateBundlePolicy,
) -> tuple[dict[str, Any], ...]:
    steps = []
    for sequence, component in enumerate(components, start=1):
        try:
            order = policy.restore_order.index(component.role)
        except ValueError:
            order = len(policy.restore_order)
        steps.append(
            {
                "sequence": sequence,
                "restore_order": order,
                "component": component.name,
                "role": component.role,
                "kind": component.kind,
                "payload_schema_version": component.payload_schema_version,
                "exported_sha256": component.exported_sha256,
            }
        )
    return tuple(steps)


def _bundle_issues(
    components: tuple[AgentStateBundleComponent, ...],
    policy: AgentStateBundlePolicy,
) -> tuple[AgentStateBundleIssue, ...]:
    issues: list[AgentStateBundleIssue] = []
    roles = {component.role for component in components if component.role}
    for role in policy.required_roles:
        if role not in roles:
            issues.append(
                AgentStateBundleIssue(
                    code="required_state_role_missing",
                    message=f"Required state role is missing: {role}",
                    metadata={"role": role, "roles": sorted(roles)},
                )
            )
    if not components:
        issues.append(
            AgentStateBundleIssue(
                code="state_bundle_empty",
                message="State bundle has no components.",
            )
        )
    return tuple(issues)


def _redaction_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "agent-core-state-bundle-redaction-audit/v1",
        "redacted": bool(manifest.get("redacted")),
        "redacted_count": int(manifest.get("redacted_count") or 0),
        "decisions": list(manifest.get("decisions") or ()),
        "metadata": dict(manifest.get("metadata") or {}),
    }


def _measure_payload(payload: dict[str, Any]) -> tuple[int, str]:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return len(raw), hashlib.sha256(raw).hexdigest()


def _count_components(
    components: tuple[AgentStateBundleComponent, ...],
    field: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for component in components:
        value = str(getattr(component, field) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
