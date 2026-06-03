"""Human-in-loop approval primitives for agent core."""

from __future__ import annotations

import base64
import json
import re
import sqlite3
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from agent_core.policy import ApprovalRequest


ApprovalStatus = Literal["pending", "approved", "rejected", "cancelled"]
ApprovalDecisionStatus = Literal["approved", "rejected", "cancelled"]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ApprovalDecisionRecord:
    status: ApprovalDecisionStatus
    reason: str = ""
    actor: str = ""
    decided_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "actor": self.actor,
            "decided_at": self.decided_at,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ApprovalRecord:
    approval_id: str
    request: ApprovalRequest
    status: ApprovalStatus = "pending"
    run_id: str = ""
    turn_id: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    decision: ApprovalDecisionRecord | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pending(self) -> bool:
        return self.status == "pending"

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-approval-record/v1",
            "approval_id": self.approval_id,
            "status": self.status,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "created_at": self.created_at,
            "request": self.request.manifest(),
            "decision": self.decision.manifest() if self.decision else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ApprovalGrant:
    """Resolved approval material that can unlock one matching policy gate."""

    approval_id: str
    subject: str
    status: ApprovalDecisionStatus
    reason: str = ""
    actor: str = ""
    decided_at: str = ""
    request_metadata: dict[str, Any] = field(default_factory=dict)
    decision_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.status == "approved"

    @classmethod
    def from_record(cls, record: ApprovalRecord) -> "ApprovalGrant":
        if record.decision is None:
            raise ValueError(f"approval has no decision: {record.approval_id}")
        return cls(
            approval_id=record.approval_id,
            subject=record.request.subject,
            status=record.decision.status,
            reason=record.decision.reason,
            actor=record.decision.actor,
            decided_at=record.decision.decided_at,
            request_metadata=dict(record.request.metadata),
            decision_metadata=dict(record.decision.metadata),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "subject": self.subject,
            "status": self.status,
            "reason": self.reason,
            "actor": self.actor,
            "decided_at": self.decided_at,
            "request_metadata": dict(self.request_metadata),
            "decision_metadata": dict(self.decision_metadata),
        }


@dataclass(frozen=True)
class ApprovalResumeContext:
    """Approved decisions supplied to a resumed run."""

    grants: tuple[ApprovalGrant, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_records(
        cls,
        records: tuple[ApprovalRecord, ...],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "ApprovalResumeContext":
        return cls(
            grants=tuple(ApprovalGrant.from_record(record) for record in records),
            metadata=dict(metadata or {}),
        )

    @property
    def empty(self) -> bool:
        return not self.grants

    def approved_for(self, request: ApprovalRequest | None) -> ApprovalGrant | None:
        if request is None:
            return None
        for grant in self.grants:
            if grant.approved and grant.subject == request.subject:
                return grant
        return None

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-approval-resume/v1",
            "grant_count": len(self.grants),
            "approved_count": sum(1 for grant in self.grants if grant.approved),
            "grants": [grant.manifest() for grant in self.grants],
            "metadata": dict(self.metadata),
        }


class ApprovalStorePort(Protocol):
    async def submit(
        self,
        request: ApprovalRequest,
        *,
        run_id: str = "",
        turn_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalRecord:
        """Submit an approval request and return its record."""

    async def decide(self, approval_id: str, decision: ApprovalDecisionRecord) -> ApprovalRecord:
        """Resolve an approval request."""

    async def get(self, approval_id: str) -> ApprovalRecord | None:
        """Return one approval record, if present."""

    async def pending(self) -> tuple[ApprovalRecord, ...]:
        """Return currently pending approval records."""


class NullApprovalStore(ApprovalStorePort):
    """Ephemeral no-op approval store for runtimes that do not persist approvals."""

    async def submit(
        self,
        request: ApprovalRequest,
        *,
        run_id: str = "",
        turn_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalRecord:
        return ApprovalRecord(
            approval_id=uuid4().hex,
            request=request,
            run_id=run_id,
            turn_id=turn_id,
            metadata=dict(metadata or {}),
        )

    async def decide(self, approval_id: str, decision: ApprovalDecisionRecord) -> ApprovalRecord:
        raise KeyError(f"unknown approval: {approval_id}")

    async def get(self, approval_id: str) -> ApprovalRecord | None:
        return None

    async def pending(self) -> tuple[ApprovalRecord, ...]:
        return ()

    def manifest(self) -> dict[str, Any]:
        return {"schema_version": "agent-core-null-approval-store/v1", "pending_count": 0}


class InMemoryApprovalStore(ApprovalStorePort):
    """In-memory approval queue suitable for tests and lightweight runtimes."""

    def __init__(self, records: tuple[ApprovalRecord, ...] = ()) -> None:
        self._records: dict[str, ApprovalRecord] = {record.approval_id: record for record in records}

    async def submit(
        self,
        request: ApprovalRequest,
        *,
        run_id: str = "",
        turn_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalRecord:
        record = ApprovalRecord(
            approval_id=uuid4().hex,
            request=request,
            run_id=run_id,
            turn_id=turn_id,
            metadata=dict(metadata or {}),
        )
        self._records[record.approval_id] = record
        return record

    async def decide(self, approval_id: str, decision: ApprovalDecisionRecord) -> ApprovalRecord:
        record = await self.get(approval_id)
        if record is None:
            raise KeyError(f"unknown approval: {approval_id}")
        if not record.pending:
            raise ValueError(f"approval already resolved: {approval_id}")
        resolved = replace(record, status=decision.status, decision=decision)
        self._records[approval_id] = resolved
        return resolved

    async def get(self, approval_id: str) -> ApprovalRecord | None:
        return self._records.get(approval_id)

    async def pending(self) -> tuple[ApprovalRecord, ...]:
        return tuple(record for record in self.records() if record.pending)

    def records(self) -> tuple[ApprovalRecord, ...]:
        return tuple(sorted(self._records.values(), key=lambda item: item.created_at))

    def manifest(self) -> dict[str, Any]:
        records = self.records()
        return {
            "schema_version": "agent-core-in-memory-approval-store/v1",
            "record_count": len(records),
            "pending_count": sum(1 for record in records if record.pending),
            "records": [record.manifest() for record in records],
        }


class SQLiteApprovalStore(ApprovalStorePort):
    """SQLite-backed approval queue for durable local SDK runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    async def submit(
        self,
        request: ApprovalRequest,
        *,
        run_id: str = "",
        turn_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalRecord:
        record = ApprovalRecord(
            approval_id=uuid4().hex,
            request=request,
            run_id=run_id,
            turn_id=turn_id,
            metadata=dict(metadata or {}),
        )
        self._save(record)
        return record

    async def decide(self, approval_id: str, decision: ApprovalDecisionRecord) -> ApprovalRecord:
        record = await self.get(approval_id)
        if record is None:
            raise KeyError(f"unknown approval: {approval_id}")
        if not record.pending:
            raise ValueError(f"approval already resolved: {approval_id}")
        resolved = replace(record, status=decision.status, decision=decision)
        self._save(resolved)
        return resolved

    async def get(self, approval_id: str) -> ApprovalRecord | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                """
                SELECT record_json
                FROM approval_records
                WHERE approval_id = ?
                """,
                (approval_id,),
            ).fetchone()
        if row is None:
            return None
        return _approval_record_from_json(str(row[0] or "{}"))

    async def pending(self) -> tuple[ApprovalRecord, ...]:
        return tuple(record for record in self.records() if record.pending)

    def records(self) -> tuple[ApprovalRecord, ...]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """
                SELECT record_json
                FROM approval_records
                ORDER BY created_at ASC, approval_id ASC
                """
            ).fetchall()
        return tuple(_approval_record_from_json(str(row[0] or "{}")) for row in rows)

    def manifest(self) -> dict[str, Any]:
        records = self.records()
        return {
            "schema_version": "agent-core-sqlite-approval-store/v1",
            "path": str(self.path),
            "record_count": len(records),
            "pending_count": sum(1 for record in records if record.pending),
            "records": [record.manifest() for record in records],
        }

    def _save(self, record: ApprovalRecord) -> None:
        raw = json.dumps(_approval_record_payload(record), ensure_ascii=False, sort_keys=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO approval_records(approval_id, record_json, status, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(approval_id) DO UPDATE SET
                    record_json = excluded.record_json,
                    status = excluded.status,
                    created_at = excluded.created_at
                """,
                (record.approval_id, raw, record.status, record.created_at),
            )
            conn.commit()

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_records (
                    approval_id TEXT PRIMARY KEY,
                    record_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()


class MarkdownApprovalStore(ApprovalStorePort):
    """Markdown-backed approval queue for inspectable local SDK runs."""

    _START = "<!-- approval-record "
    _END = " -->"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def submit(
        self,
        request: ApprovalRequest,
        *,
        run_id: str = "",
        turn_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalRecord:
        record = ApprovalRecord(
            approval_id=uuid4().hex,
            request=request,
            run_id=run_id,
            turn_id=turn_id,
            metadata=dict(metadata or {}),
        )
        self._save(record)
        return record

    async def decide(self, approval_id: str, decision: ApprovalDecisionRecord) -> ApprovalRecord:
        record = await self.get(approval_id)
        if record is None:
            raise KeyError(f"unknown approval: {approval_id}")
        if not record.pending:
            raise ValueError(f"approval already resolved: {approval_id}")
        resolved = replace(record, status=decision.status, decision=decision)
        self._save(resolved)
        return resolved

    async def get(self, approval_id: str) -> ApprovalRecord | None:
        for record in self.records():
            if record.approval_id == approval_id:
                return record
        return None

    async def pending(self) -> tuple[ApprovalRecord, ...]:
        return tuple(record for record in self.records() if record.pending)

    def records(self) -> tuple[ApprovalRecord, ...]:
        if not self.path.exists():
            return ()
        text = self.path.read_text(encoding="utf-8")
        records: list[ApprovalRecord] = []
        for match in _APPROVAL_MARKDOWN_RE.finditer(text):
            try:
                records.append(_approval_record_from_json(_decode_approval_payload(match.group(1))))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return tuple(sorted(records, key=lambda item: (item.created_at, item.approval_id)))

    def manifest(self) -> dict[str, Any]:
        records = self.records()
        return {
            "schema_version": "agent-core-markdown-approval-store/v1",
            "path": str(self.path),
            "record_count": len(records),
            "pending_count": sum(1 for record in records if record.pending),
            "records": [record.manifest() for record in records],
        }

    def _save(self, record: ApprovalRecord) -> None:
        records = {item.approval_id: item for item in self.records()}
        records[record.approval_id] = record
        ordered = tuple(sorted(records.values(), key=lambda item: (item.created_at, item.approval_id)))
        self._write(ordered)

    def _write(self, records: tuple[ApprovalRecord, ...]) -> None:
        lines = [
            "# Approval Records",
            "",
            "This file is managed by raven_heart. Approval payloads are stored in comments.",
            "",
        ]
        for record in records:
            raw = _encode_approval_payload(record)
            lines.append(f"{self._START}{raw}{self._END}")
            lines.append(f"- approval_id: `{record.approval_id}`")
            lines.append(f"- subject: `{record.request.subject}`")
            lines.append(f"- status: `{record.status}`")
            if record.decision is not None:
                lines.append(f"- actor: `{record.decision.actor}`")
            lines.append("")
        self.path.write_text("\n".join(lines), encoding="utf-8")


_APPROVAL_MARKDOWN_RE = re.compile(
    r"<!--\s*approval-record\s+([A-Za-z0-9+/=]+)\s*-->",
    re.DOTALL,
)


def _encode_approval_payload(record: ApprovalRecord) -> str:
    raw = json.dumps(
        _approval_record_payload(record),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _decode_approval_payload(encoded: str) -> str:
    return base64.b64decode(encoded.encode("ascii")).decode("utf-8")


def _approval_record_payload(record: ApprovalRecord) -> dict[str, Any]:
    return {
        "schema_version": "agent-core-approval-record-payload/v1",
        "approval_id": record.approval_id,
        "status": record.status,
        "run_id": record.run_id,
        "turn_id": record.turn_id,
        "created_at": record.created_at,
        "request": {
            "reason": record.request.reason,
            "subject": record.request.subject,
            "metadata": dict(record.request.metadata),
        },
        "decision": {
            "status": record.decision.status,
            "reason": record.decision.reason,
            "actor": record.decision.actor,
            "decided_at": record.decision.decided_at,
            "metadata": dict(record.decision.metadata),
        }
        if record.decision
        else None,
        "metadata": dict(record.metadata),
    }


def _approval_record_from_json(raw: str) -> ApprovalRecord:
    payload = json.loads(raw or "{}")
    if not isinstance(payload, dict):
        raise ValueError("invalid approval payload")
    request_payload = payload.get("request")
    if not isinstance(request_payload, dict):
        raise ValueError("invalid approval payload")
    decision_payload = payload.get("decision")
    decision = None
    if isinstance(decision_payload, dict):
        decision = ApprovalDecisionRecord(
            status=_decision_status(str(decision_payload.get("status") or "approved")),
            reason=str(decision_payload.get("reason") or ""),
            actor=str(decision_payload.get("actor") or ""),
            decided_at=str(decision_payload.get("decided_at") or utc_now_iso()),
            metadata=dict(decision_payload.get("metadata") or {}),
        )
    return ApprovalRecord(
        approval_id=str(payload.get("approval_id") or uuid4().hex),
        request=ApprovalRequest(
            reason=str(request_payload.get("reason") or ""),
            subject=str(request_payload.get("subject") or ""),
            metadata=dict(request_payload.get("metadata") or {}),
        ),
        status=_approval_status(str(payload.get("status") or "pending")),
        run_id=str(payload.get("run_id") or ""),
        turn_id=str(payload.get("turn_id") or ""),
        created_at=str(payload.get("created_at") or utc_now_iso()),
        decision=decision,
        metadata=dict(payload.get("metadata") or {}),
    )


def _approval_status(value: str) -> ApprovalStatus:
    if value not in {"pending", "approved", "rejected", "cancelled"}:
        raise ValueError(f"invalid approval status: {value}")
    return value  # type: ignore[return-value]


def _decision_status(value: str) -> ApprovalDecisionStatus:
    if value not in {"approved", "rejected", "cancelled"}:
        raise ValueError(f"invalid approval decision status: {value}")
    return value  # type: ignore[return-value]
