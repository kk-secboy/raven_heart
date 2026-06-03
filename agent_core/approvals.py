"""Human-in-loop approval primitives for agent core."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
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
