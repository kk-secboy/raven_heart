"""Policy decision protocol for the agent core."""

from __future__ import annotations

import base64
import json
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from agent_core.actions import ParsedAction
from agent_core.tools import ToolInvocation


PolicyStatus = Literal["allow", "deny", "approval_required"]
PolicySubjectKind = Literal["action", "tool"]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ApprovalRequest:
    reason: str
    subject: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-approval-request/v1",
            "reason": self.reason,
            "subject": self.subject,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PolicyDecision:
    status: PolicyStatus = "allow"
    reason: str = ""
    approval: ApprovalRequest | None = None

    @property
    def allowed(self) -> bool:
        return self.status == "allow"

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-policy-decision/v1",
            "status": self.status,
            "reason": self.reason,
            "approval": self.approval.manifest() if self.approval else None,
        }


@dataclass(frozen=True)
class PolicyDecisionRecord:
    decision_id: str
    subject_kind: PolicySubjectKind
    subject_name: str
    decision: PolicyDecision
    sequence: int = 0
    run_id: str = ""
    turn_id: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def subject(self) -> str:
        return f"{self.subject_kind}:{self.subject_name}"

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-policy-decision-record/v1",
            "decision_id": self.decision_id,
            "sequence": self.sequence,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "created_at": self.created_at,
            "subject_kind": self.subject_kind,
            "subject_name": self.subject_name,
            "subject": self.subject,
            "decision": self.decision.manifest(),
            "metadata": dict(self.metadata),
        }


class PolicyPort(Protocol):
    async def check_action(self, action: ParsedAction) -> PolicyDecision:
        """Check whether an action is allowed."""

    async def check_tool(self, invocation: ToolInvocation) -> PolicyDecision:
        """Check whether a tool invocation is allowed."""


class PolicyDecisionStorePort(Protocol):
    async def submit(
        self,
        decision: PolicyDecision,
        *,
        subject_kind: PolicySubjectKind,
        subject_name: str,
        run_id: str = "",
        turn_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> PolicyDecisionRecord:
        """Persist one policy gate decision."""

    async def records(self) -> tuple[PolicyDecisionRecord, ...]:
        """Return recorded policy decisions."""


class AllowAllPolicy:
    async def check_action(self, action: ParsedAction) -> PolicyDecision:
        return PolicyDecision()

    async def check_tool(self, invocation: ToolInvocation) -> PolicyDecision:
        return PolicyDecision()


class NullPolicyDecisionStore(PolicyDecisionStorePort):
    async def submit(
        self,
        decision: PolicyDecision,
        *,
        subject_kind: PolicySubjectKind,
        subject_name: str,
        run_id: str = "",
        turn_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> PolicyDecisionRecord:
        return PolicyDecisionRecord(
            decision_id=uuid4().hex,
            subject_kind=subject_kind,
            subject_name=subject_name,
            decision=decision,
            sequence=0,
            run_id=run_id,
            turn_id=turn_id,
            metadata=dict(metadata or {}),
        )

    async def records(self) -> tuple[PolicyDecisionRecord, ...]:
        return ()

    def manifest(self) -> dict[str, Any]:
        return {"schema_version": "agent-core-null-policy-decision-store/v1", "record_count": 0}


class InMemoryPolicyDecisionStore(PolicyDecisionStorePort):
    def __init__(self, records: tuple[PolicyDecisionRecord, ...] = ()) -> None:
        self._records: dict[str, PolicyDecisionRecord] = {
            record.decision_id: record for record in records
        }

    async def submit(
        self,
        decision: PolicyDecision,
        *,
        subject_kind: PolicySubjectKind,
        subject_name: str,
        run_id: str = "",
        turn_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> PolicyDecisionRecord:
        record = PolicyDecisionRecord(
            decision_id=uuid4().hex,
            subject_kind=subject_kind,
            subject_name=subject_name,
            decision=decision,
            sequence=max((record.sequence for record in self._records.values()), default=0) + 1,
            run_id=run_id,
            turn_id=turn_id,
            metadata=dict(metadata or {}),
        )
        self._records[record.decision_id] = record
        return record

    async def records(self) -> tuple[PolicyDecisionRecord, ...]:
        return tuple(sorted(self._records.values(), key=_policy_decision_sort_key))

    def manifest(self) -> dict[str, Any]:
        records = tuple(sorted(self._records.values(), key=_policy_decision_sort_key))
        return {
            "schema_version": "agent-core-in-memory-policy-decision-store/v1",
            "record_count": len(records),
            "records": [record.manifest() for record in records],
        }


class SQLitePolicyDecisionStore(PolicyDecisionStorePort):
    """SQLite-backed policy decision audit store for durable local SDK runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    async def submit(
        self,
        decision: PolicyDecision,
        *,
        subject_kind: PolicySubjectKind,
        subject_name: str,
        run_id: str = "",
        turn_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> PolicyDecisionRecord:
        sequence = self._next_sequence()
        record = PolicyDecisionRecord(
            decision_id=uuid4().hex,
            subject_kind=subject_kind,
            subject_name=subject_name,
            decision=decision,
            sequence=sequence,
            run_id=run_id,
            turn_id=turn_id,
            metadata=dict(metadata or {}),
        )
        raw = json.dumps(_policy_decision_record_payload(record), ensure_ascii=False, sort_keys=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO policy_decision_records(
                    decision_id, sequence, run_id, turn_id, subject_kind, subject_name,
                    status, created_at, record_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.decision_id,
                    record.sequence,
                    record.run_id,
                    record.turn_id,
                    record.subject_kind,
                    record.subject_name,
                    record.decision.status,
                    record.created_at,
                    raw,
                ),
            )
            conn.commit()
        return record

    async def records(self) -> tuple[PolicyDecisionRecord, ...]:
        return self._records()

    def manifest(self) -> dict[str, Any]:
        records = self._records()
        return {
            "schema_version": "agent-core-sqlite-policy-decision-store/v1",
            "path": str(self.path),
            "record_count": len(records),
            "records": [record.manifest() for record in records],
        }

    def _records(self) -> tuple[PolicyDecisionRecord, ...]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """
                SELECT record_json
                FROM policy_decision_records
                ORDER BY sequence ASC, created_at ASC, decision_id ASC
                """
            ).fetchall()
        return tuple(_policy_decision_record_from_json(str(row[0] or "{}")) for row in rows)

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_decision_records (
                    decision_id TEXT PRIMARY KEY,
                    sequence INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    subject_kind TEXT NOT NULL,
                    subject_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    record_json TEXT NOT NULL
                )
                """
            )
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(policy_decision_records)").fetchall()
            }
            if "sequence" not in columns:
                conn.execute(
                    "ALTER TABLE policy_decision_records ADD COLUMN sequence INTEGER NOT NULL DEFAULT 0"
                )
                conn.execute("UPDATE policy_decision_records SET sequence = rowid WHERE sequence = 0")
            conn.commit()

    def _next_sequence(self) -> int:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT COALESCE(MAX(sequence), 0) FROM policy_decision_records").fetchone()
        return int(row[0] or 0) + 1


class MarkdownPolicyDecisionStore(PolicyDecisionStorePort):
    """Markdown-backed policy decision audit store for inspectable local SDK runs."""

    _START = "<!-- policy-decision-record "
    _END = " -->"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def submit(
        self,
        decision: PolicyDecision,
        *,
        subject_kind: PolicySubjectKind,
        subject_name: str,
        run_id: str = "",
        turn_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> PolicyDecisionRecord:
        existing = await self.records()
        record = PolicyDecisionRecord(
            decision_id=uuid4().hex,
            subject_kind=subject_kind,
            subject_name=subject_name,
            decision=decision,
            sequence=max((item.sequence for item in existing), default=0) + 1,
            run_id=run_id,
            turn_id=turn_id,
            metadata=dict(metadata or {}),
        )
        records = {item.decision_id: item for item in existing}
        records[record.decision_id] = record
        self._write(tuple(sorted(records.values(), key=_policy_decision_sort_key)))
        return record

    async def records(self) -> tuple[PolicyDecisionRecord, ...]:
        return self._records()

    def manifest(self) -> dict[str, Any]:
        records = self._records()
        return {
            "schema_version": "agent-core-markdown-policy-decision-store/v1",
            "path": str(self.path),
            "record_count": len(records),
            "records": [record.manifest() for record in records],
        }

    def _records(self) -> tuple[PolicyDecisionRecord, ...]:
        if not self.path.exists():
            return ()
        text = self.path.read_text(encoding="utf-8")
        records: list[PolicyDecisionRecord] = []
        for match in _POLICY_DECISION_MARKDOWN_RE.finditer(text):
            try:
                records.append(_policy_decision_record_from_json(_decode_policy_decision_payload(match.group(1))))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return tuple(sorted(records, key=_policy_decision_sort_key))

    def _write(self, records: tuple[PolicyDecisionRecord, ...]) -> None:
        lines = [
            "# Policy Decision Records",
            "",
            "This file is managed by raven_heart. Policy payloads are stored in comments.",
            "",
        ]
        for record in records:
            raw = _encode_policy_decision_payload(record)
            lines.append(f"{self._START}{raw}{self._END}")
            lines.append(f"- decision_id: `{record.decision_id}`")
            lines.append(f"- subject: `{record.subject}`")
            lines.append(f"- status: `{record.decision.status}`")
            lines.append("")
        self.path.write_text("\n".join(lines), encoding="utf-8")


@dataclass(frozen=True)
class PolicyRule:
    name: str
    status: PolicyStatus
    reason: str = ""
    action_names: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()
    tool_tags: tuple[str, ...] = ()
    mcp_servers: tuple[str, ...] = ()
    mcp_transports: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def check_action(self, action: ParsedAction) -> PolicyDecision | None:
        if not self.action_names:
            return None
        if not _matches_any(action.name, self.action_names):
            return None
        return self._decision(subject=f"action:{action.name}")

    def check_tool(self, invocation: ToolInvocation) -> PolicyDecision | None:
        if not self._has_tool_predicate:
            return None
        spec = _tool_spec_metadata(invocation)
        names = {
            invocation.tool_name,
            str(spec.get("name") or ""),
            *tuple(str(alias) for alias in spec.get("aliases") or ()),
        }
        if self.tool_names and not any(_matches_any(name, self.tool_names) for name in names):
            return None
        tags = {str(tag).lower() for tag in spec.get("tags") or ()}
        if self.tool_tags and not tags.intersection({tag.lower() for tag in self.tool_tags}):
            return None
        mcp = spec.get("metadata", {}).get("mcp", {}) if isinstance(spec.get("metadata"), dict) else {}
        if self.mcp_servers and not _matches_any(str(mcp.get("server_name") or ""), self.mcp_servers):
            return None
        if self.mcp_transports and not _matches_any(str(mcp.get("transport") or ""), self.mcp_transports):
            return None
        if self.metadata and not _metadata_matches(spec.get("metadata", {}), self.metadata):
            return None
        return self._decision(subject=f"tool:{invocation.tool_name}")

    @property
    def _has_tool_predicate(self) -> bool:
        return bool(
            self.tool_names
            or self.tool_tags
            or self.mcp_servers
            or self.mcp_transports
            or self.metadata
        )

    def _decision(self, *, subject: str) -> PolicyDecision:
        reason = self.reason or f"policy rule matched: {self.name}"
        approval = None
        if self.status == "approval_required":
            approval = ApprovalRequest(
                reason=reason,
                subject=subject,
                metadata={"rule": self.name},
            )
        return PolicyDecision(status=self.status, reason=reason, approval=approval)


class RuleBasedPolicy:
    def __init__(self, rules: Iterable[PolicyRule] = ()) -> None:
        self.rules = tuple(rules)

    async def check_action(self, action: ParsedAction) -> PolicyDecision:
        return self._first_decision(rule.check_action(action) for rule in self.rules)

    async def check_tool(self, invocation: ToolInvocation) -> PolicyDecision:
        return self._first_decision(rule.check_tool(invocation) for rule in self.rules)

    @staticmethod
    def _first_decision(decisions: Iterable[PolicyDecision | None]) -> PolicyDecision:
        approval: PolicyDecision | None = None
        for decision in decisions:
            if decision is None or decision.status == "allow":
                continue
            if decision.status == "deny":
                return decision
            if decision.status == "approval_required" and approval is None:
                approval = decision
        return approval or PolicyDecision()


class CompositePolicy:
    def __init__(self, policies: Iterable[PolicyPort]) -> None:
        self.policies = tuple(policies)

    async def check_action(self, action: ParsedAction) -> PolicyDecision:
        approval: PolicyDecision | None = None
        for policy in self.policies:
            decision = await policy.check_action(action)
            if decision.status == "deny":
                return decision
            if decision.status == "approval_required" and approval is None:
                approval = decision
        return approval or PolicyDecision()

    async def check_tool(self, invocation: ToolInvocation) -> PolicyDecision:
        approval: PolicyDecision | None = None
        for policy in self.policies:
            decision = await policy.check_tool(invocation)
            if decision.status == "deny":
                return decision
            if decision.status == "approval_required" and approval is None:
                approval = decision
        return approval or PolicyDecision()


_POLICY_DECISION_MARKDOWN_RE = re.compile(
    r"<!--\s*policy-decision-record\s+([A-Za-z0-9+/=]+)\s*-->",
    re.DOTALL,
)


def _policy_decision_record_payload(record: PolicyDecisionRecord) -> dict[str, Any]:
    approval = record.decision.approval
    return {
        "schema_version": "agent-core-policy-decision-record-payload/v1",
        "decision_id": record.decision_id,
        "sequence": record.sequence,
        "run_id": record.run_id,
        "turn_id": record.turn_id,
        "created_at": record.created_at,
        "subject_kind": record.subject_kind,
        "subject_name": record.subject_name,
        "decision": {
            "status": record.decision.status,
            "reason": record.decision.reason,
            "approval": {
                "reason": approval.reason,
                "subject": approval.subject,
                "metadata": dict(approval.metadata),
            }
            if approval
            else None,
        },
        "metadata": dict(record.metadata),
    }


def _policy_decision_record_from_json(raw: str) -> PolicyDecisionRecord:
    payload = json.loads(raw or "{}")
    if not isinstance(payload, dict):
        raise ValueError("invalid policy decision payload")
    decision_payload = payload.get("decision")
    if not isinstance(decision_payload, dict):
        raise ValueError("invalid policy decision payload")
    approval_payload = decision_payload.get("approval")
    approval = None
    if isinstance(approval_payload, dict):
        approval = ApprovalRequest(
            reason=str(approval_payload.get("reason") or ""),
            subject=str(approval_payload.get("subject") or ""),
            metadata=dict(approval_payload.get("metadata") or {}),
        )
    return PolicyDecisionRecord(
        decision_id=str(payload.get("decision_id") or uuid4().hex),
        sequence=int(payload.get("sequence") or 0),
        run_id=str(payload.get("run_id") or ""),
        turn_id=str(payload.get("turn_id") or ""),
        created_at=str(payload.get("created_at") or utc_now_iso()),
        subject_kind=_policy_subject_kind(str(payload.get("subject_kind") or "action")),
        subject_name=str(payload.get("subject_name") or ""),
        decision=PolicyDecision(
            status=_policy_status(str(decision_payload.get("status") or "allow")),
            reason=str(decision_payload.get("reason") or ""),
            approval=approval,
        ),
        metadata=dict(payload.get("metadata") or {}),
    )


def _encode_policy_decision_payload(record: PolicyDecisionRecord) -> str:
    raw = json.dumps(
        _policy_decision_record_payload(record),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _decode_policy_decision_payload(encoded: str) -> str:
    return base64.b64decode(encoded.encode("ascii")).decode("utf-8")


def _policy_status(value: str) -> PolicyStatus:
    if value not in {"allow", "deny", "approval_required"}:
        raise ValueError(f"invalid policy status: {value}")
    return value  # type: ignore[return-value]


def _policy_subject_kind(value: str) -> PolicySubjectKind:
    if value not in {"action", "tool"}:
        raise ValueError(f"invalid policy subject kind: {value}")
    return value  # type: ignore[return-value]


def _policy_decision_sort_key(record: PolicyDecisionRecord) -> tuple[int, str, str]:
    return (record.sequence, record.created_at, record.decision_id)


def _matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatchcase(value, pattern) for pattern in patterns)


def _tool_spec_metadata(invocation: ToolInvocation) -> dict[str, Any]:
    spec = invocation.metadata.get("tool_spec")
    return spec if isinstance(spec, dict) else {}


def _metadata_matches(metadata: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(metadata, dict):
        return False
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if isinstance(expected_value, dict):
            if not _metadata_matches(actual, expected_value):
                return False
            continue
        if actual != expected_value:
            return False
    return True

