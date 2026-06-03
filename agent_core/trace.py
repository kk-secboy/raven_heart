"""Trace records for prompt, tool, and cost observations."""

from __future__ import annotations

import base64
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agent_core.harness import AgentJournalSnapshot, TERMINAL_RUN_STATUSES


@dataclass(frozen=True)
class PromptTrace:
    run_id: str
    turn_id: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ToolTrace:
    run_id: str
    turn_id: str
    call_id: str
    tool_name: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CostTrace:
    run_id: str
    turn_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class CapabilityTrace:
    run_id: str = ""
    turn_id: str = ""
    actions: dict[str, Any] = field(default_factory=dict)
    tools: dict[str, Any] = field(default_factory=dict)
    skills: dict[str, Any] = field(default_factory=dict)
    mcp: dict[str, Any] = field(default_factory=dict)
    catalog: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-capability-trace/v1",
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "actions": self.actions,
            "tools": self.tools,
            "skills": self.skills,
            "mcp": self.mcp,
            "catalog": self.catalog,
        }


@dataclass(frozen=True)
class AgentRunTraceBundle:
    """One run's provider-neutral trace materials."""

    run_id: str
    status: str
    iterations: int = 0
    output_bytes: int = 0
    session: dict[str, Any] = field(default_factory=dict)
    prompt: dict[str, Any] = field(default_factory=dict)
    journal_replay: dict[str, Any] = field(default_factory=dict)
    provider: dict[str, Any] = field(default_factory=dict)
    tool_replay: dict[str, Any] = field(default_factory=dict)
    policy_decisions: dict[str, Any] = field(default_factory=dict)
    approvals: dict[str, Any] = field(default_factory=dict)
    event_log: dict[str, Any] = field(default_factory=dict)
    resume: dict[str, Any] = field(default_factory=dict)
    timeline_reduction: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        journal_ok = self.journal_replay.get("ok")
        return {
            "schema_version": "agent-core-run-trace-bundle/v1",
            "run": {
                "run_id": self.run_id,
                "status": self.status,
                "iterations": self.iterations,
                "output_bytes": self.output_bytes,
            },
            "summary": {
                "journal_ok": journal_ok,
                "journal_event_count": int(self.journal_replay.get("event_count") or 0),
                "provider_call_count": int(self.provider.get("call_count") or 0),
                "tool_replay_record_count": int(self.tool_replay.get("record_count") or 0),
                "policy_decision_record_count": int(self.policy_decisions.get("record_count") or 0),
                "approval_record_count": int(self.approvals.get("record_count") or 0),
                "event_log_count": int(self.event_log.get("event_count") or 0),
                "has_resume": bool(self.resume),
                "has_timeline_reduction": bool(self.timeline_reduction),
            },
            "session": dict(self.session),
            "prompt": dict(self.prompt),
            "journal_replay": dict(self.journal_replay),
            "provider": dict(self.provider),
            "tool_replay": dict(self.tool_replay),
            "policy_decisions": dict(self.policy_decisions),
            "approvals": dict(self.approvals),
            "event_log": dict(self.event_log),
            "resume": dict(self.resume),
            "timeline_reduction": dict(self.timeline_reduction),
            "metadata": dict(self.metadata),
        }


class RunTraceStorePort(Protocol):
    async def save(self, manifest: dict[str, Any]) -> None:
        """Persist one run trace bundle manifest."""

    async def load(self, run_id: str) -> dict[str, Any] | None:
        """Load one run trace bundle by run id."""

    async def records(self) -> tuple[dict[str, Any], ...]:
        """Return persisted run trace manifests."""


class NullRunTraceStore(RunTraceStorePort):
    async def save(self, manifest: dict[str, Any]) -> None:
        return None

    async def load(self, run_id: str) -> dict[str, Any] | None:
        return None

    async def records(self) -> tuple[dict[str, Any], ...]:
        return ()

    def manifest(self) -> dict[str, Any]:
        return {"schema_version": "agent-core-null-run-trace-store/v1", "record_count": 0}


class InMemoryRunTraceStore(RunTraceStorePort):
    def __init__(self, records: tuple[dict[str, Any], ...] = ()) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        for record in records:
            run_id = _run_id_from_trace_manifest(record)
            if run_id:
                self._records[run_id] = dict(record)

    async def save(self, manifest: dict[str, Any]) -> None:
        run_id = _run_id_from_trace_manifest(manifest)
        if not run_id:
            raise ValueError("trace manifest is missing run.run_id")
        self._records[run_id] = dict(manifest)

    async def load(self, run_id: str) -> dict[str, Any] | None:
        record = self._records.get(run_id)
        return dict(record) if record is not None else None

    async def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in sorted(self._records.values(), key=_trace_sort_key))

    def manifest(self) -> dict[str, Any]:
        records = tuple(sorted(self._records.values(), key=_trace_sort_key))
        return {
            "schema_version": "agent-core-in-memory-run-trace-store/v1",
            "record_count": len(records),
            "records": [_trace_record_summary(record) for record in records],
        }


class SQLiteRunTraceStore(RunTraceStorePort):
    """SQLite-backed run trace store for durable local SDK runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    async def save(self, manifest: dict[str, Any]) -> None:
        run_id = _run_id_from_trace_manifest(manifest)
        if not run_id:
            raise ValueError("trace manifest is missing run.run_id")
        raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO run_trace_records(run_id, status, manifest_json)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status = excluded.status,
                    manifest_json = excluded.manifest_json
                """,
                (run_id, _trace_status(manifest), raw),
            )
            conn.commit()

    async def load(self, run_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                """
                SELECT manifest_json
                FROM run_trace_records
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        value = json.loads(str(row[0] or "{}"))
        return dict(value) if isinstance(value, dict) else {}

    async def records(self) -> tuple[dict[str, Any], ...]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """
                SELECT manifest_json
                FROM run_trace_records
                ORDER BY rowid ASC
                """
            ).fetchall()
        records = []
        for row in rows:
            value = json.loads(str(row[0] or "{}"))
            if isinstance(value, dict):
                records.append(dict(value))
        return tuple(records)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-sqlite-run-trace-store/v1",
            "path": str(self.path),
        }

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_trace_records (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    manifest_json TEXT NOT NULL
                )
                """
            )
            conn.commit()


class MarkdownRunTraceStore(RunTraceStorePort):
    """Markdown-backed run trace store for inspectable local SDK runs."""

    _START = "<!-- run-trace-record "
    _END = " -->"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def save(self, manifest: dict[str, Any]) -> None:
        run_id = _run_id_from_trace_manifest(manifest)
        if not run_id:
            raise ValueError("trace manifest is missing run.run_id")
        records = {_run_id_from_trace_manifest(item): dict(item) for item in await self.records()}
        records[run_id] = dict(manifest)
        self._write(tuple(sorted(records.values(), key=_trace_sort_key)))

    async def load(self, run_id: str) -> dict[str, Any] | None:
        for record in await self.records():
            if _run_id_from_trace_manifest(record) == run_id:
                return dict(record)
        return None

    async def records(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        text = self.path.read_text(encoding="utf-8")
        records: list[dict[str, Any]] = []
        for match in _RUN_TRACE_MARKDOWN_RE.finditer(text):
            try:
                value = json.loads(_decode_trace_payload(match.group(1)))
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(value, dict):
                records.append(dict(value))
        return tuple(sorted(records, key=_trace_sort_key))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-markdown-run-trace-store/v1",
            "path": str(self.path),
        }

    def _write(self, records: tuple[dict[str, Any], ...]) -> None:
        lines = [
            "# Run Trace Records",
            "",
            "This file is managed by raven_heart. Trace payloads are stored in comments.",
            "",
        ]
        for record in records:
            raw = _encode_trace_payload(record)
            summary = _trace_record_summary(record)
            lines.append(f"{self._START}{raw}{self._END}")
            lines.append(f"- run_id: `{summary['run_id']}`")
            lines.append(f"- status: `{summary['status']}`")
            lines.append(f"- provider_calls: `{summary['provider_call_count']}`")
            lines.append("")
        self.path.write_text("\n".join(lines), encoding="utf-8")


@dataclass(frozen=True)
class ReplayIssue:
    severity: str
    code: str
    message: str
    run_id: str = ""
    turn_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentReplayEvent:
    sequence: int
    event_type: str
    run_id: str = ""
    turn_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_type": self.event_type,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class AgentJournalReplay:
    events: tuple[AgentReplayEvent, ...]
    issues: tuple[ReplayIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any], *, run_id: str = "") -> "AgentJournalReplay":
        return cls.from_snapshot(_snapshot_from_manifest(manifest), run_id=run_id)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: AgentJournalSnapshot,
        *,
        run_id: str = "",
    ) -> "AgentJournalReplay":
        events: list[AgentReplayEvent] = []

        def include(item: dict[str, Any]) -> bool:
            return not run_id or str(item.get("run_id") or "") == run_id

        def add(event_type: str, payload: dict[str, Any], *, turn_id: str = "") -> None:
            item_run_id = str(payload.get("run_id") or "")
            events.append(
                AgentReplayEvent(
                    sequence=len(events) + 1,
                    event_type=event_type,
                    run_id=item_run_id,
                    turn_id=turn_id or str(payload.get("turn_id") or ""),
                    payload=dict(payload),
                )
            )

        for item in snapshot.runs:
            if include(item):
                add("run_started", dict(item))
        for item in snapshot.turns:
            if include(item):
                add("turn_started", dict(item))
        for item in snapshot.prompts:
            if include(item):
                add("prompt_recorded", dict(item))
        for item in snapshot.model_events:
            if include(item):
                add("model_event", dict(item))
        for item in snapshot.tool_calls:
            if include(item):
                add("tool_call", dict(item))
        for item in snapshot.checkpoints:
            if include(item):
                add("checkpoint", dict(item))
        for item in snapshot.errors:
            if include(item):
                add("error", dict(item))
        for item in snapshot.finished:
            if include(item):
                add("run_finished", dict(item))

        issues = _audit_snapshot(snapshot, run_id=run_id)
        return cls(
            events=tuple(events),
            issues=issues,
            metadata={
                "source_schema_version": snapshot.schema_version,
                "run_id": run_id,
            },
        )

    def event_types(self) -> tuple[str, ...]:
        return tuple(event.event_type for event in self.events)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-journal-replay/v1",
            "ok": self.ok,
            "event_count": len(self.events),
            "events": [event.manifest() for event in self.events],
            "issues": [issue.manifest() for issue in self.issues],
            "metadata": dict(self.metadata),
        }


def _audit_snapshot(snapshot: AgentJournalSnapshot, *, run_id: str = "") -> tuple[ReplayIssue, ...]:
    issues: list[ReplayIssue] = []
    runs = {str(item.get("run_id") or ""): dict(item) for item in snapshot.runs}
    turns = {
        (str(item.get("run_id") or ""), str(item.get("turn_id") or "")): dict(item)
        for item in snapshot.turns
    }
    finished_run_ids = {str(item.get("run_id") or "") for item in snapshot.finished}

    def include(item: dict[str, Any]) -> bool:
        return not run_id or str(item.get("run_id") or "") == run_id

    def issue(
        code: str,
        message: str,
        *,
        severity: str = "error",
        item: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        item = item or {}
        issues.append(
            ReplayIssue(
                severity=severity,
                code=code,
                message=message,
                run_id=str(item.get("run_id") or ""),
                turn_id=str(item.get("turn_id") or ""),
                metadata=dict(metadata or {}),
            )
        )

    for item in snapshot.turns:
        if include(item) and str(item.get("run_id") or "") not in runs:
            issue("unknown_run", "turn references an unknown run", item=dict(item))

    event_groups = (
        ("prompt", snapshot.prompts),
        ("model_event", snapshot.model_events),
        ("tool_call", snapshot.tool_calls),
        ("checkpoint", snapshot.checkpoints),
    )
    for group_name, group in event_groups:
        for item in group:
            if not include(item):
                continue
            item_run_id = str(item.get("run_id") or "")
            item_turn_id = str(item.get("turn_id") or "")
            if item_run_id not in runs:
                issue("unknown_run", f"{group_name} references an unknown run", item=dict(item))
            if item_turn_id and (item_run_id, item_turn_id) not in turns:
                issue("unknown_turn", f"{group_name} references an unknown turn", item=dict(item))

    checkpoint_sequence_by_run: dict[str, int] = {}
    for item in snapshot.checkpoints:
        if not include(item):
            continue
        item_run_id = str(item.get("run_id") or "")
        sequence = int(item.get("sequence") or 0)
        previous = checkpoint_sequence_by_run.get(item_run_id, 0)
        if sequence <= previous:
            issue(
                "non_monotonic_checkpoint_sequence",
                "checkpoint sequence must increase per run",
                item=dict(item),
                metadata={"previous_sequence": previous, "sequence": sequence},
            )
        checkpoint_sequence_by_run[item_run_id] = sequence

    for item in snapshot.finished:
        if include(item) and str(item.get("run_id") or "") not in runs:
            issue("unknown_run", "finished record references an unknown run", item=dict(item))

    for run in snapshot.runs:
        if not include(run):
            continue
        run_id_value = str(run.get("run_id") or "")
        status = str(run.get("status") or "")
        if status in TERMINAL_RUN_STATUSES and run_id_value not in finished_run_ids:
            issue(
                "terminal_run_missing_finished_record",
                "terminal run is missing a finished record",
                item=dict(run),
            )

    return tuple(issues)


def _snapshot_from_manifest(manifest: dict[str, Any]) -> AgentJournalSnapshot:
    return AgentJournalSnapshot(
        schema_version=str(manifest.get("schema_version") or "agent-core-journal/v1"),
        runs=tuple(dict(item) for item in manifest.get("runs", ())),
        turns=tuple(dict(item) for item in manifest.get("turns", ())),
        prompts=tuple(dict(item) for item in manifest.get("prompts", ())),
        model_events=tuple(dict(item) for item in manifest.get("model_events", ())),
        tool_calls=tuple(dict(item) for item in manifest.get("tool_calls", ())),
        checkpoints=tuple(dict(item) for item in manifest.get("checkpoints", ())),
        errors=tuple(dict(item) for item in manifest.get("errors", ())),
        finished=tuple(dict(item) for item in manifest.get("finished", ())),
    )


_RUN_TRACE_MARKDOWN_RE = re.compile(
    r"<!--\s*run-trace-record\s+([A-Za-z0-9+/=]+)\s*-->",
    re.DOTALL,
)


def _run_id_from_trace_manifest(manifest: dict[str, Any]) -> str:
    run = manifest.get("run")
    if not isinstance(run, dict):
        return ""
    return str(run.get("run_id") or "")


def _trace_status(manifest: dict[str, Any]) -> str:
    run = manifest.get("run")
    if not isinstance(run, dict):
        return ""
    return str(run.get("status") or "")


def _trace_sort_key(manifest: dict[str, Any]) -> tuple[str, str]:
    return (_run_id_from_trace_manifest(manifest), _trace_status(manifest))


def _trace_record_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    run = manifest.get("run") if isinstance(manifest.get("run"), dict) else {}
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    return {
        "run_id": str(run.get("run_id") or ""),
        "status": str(run.get("status") or ""),
        "iterations": int(run.get("iterations") or 0),
        "provider_call_count": int(summary.get("provider_call_count") or 0),
        "tool_replay_record_count": int(summary.get("tool_replay_record_count") or 0),
        "policy_decision_record_count": int(summary.get("policy_decision_record_count") or 0),
        "approval_record_count": int(summary.get("approval_record_count") or 0),
        "event_log_count": int(summary.get("event_log_count") or 0),
    }


def _encode_trace_payload(manifest: dict[str, Any]) -> str:
    raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _decode_trace_payload(encoded: str) -> str:
    return base64.b64decode(encoded.encode("ascii")).decode("utf-8")

