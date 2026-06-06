"""Standard event stream for agent core execution."""

from __future__ import annotations

import base64
import json
import re
import sqlite3
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from agent_core.backends import storage_backend_manifest


EventType = Literal[
    "run_started",
    "turn_started",
    "prompt_ready",
    "prompt_profile",
    "model_response",
    "model_stream",
    "action_parsed",
    "approval_requested",
    "approval_resumed",
    "policy_decision",
    "tool_started",
    "tool_finished",
    "timeline_updated",
    "checkpoint_created",
    "loop_warning",
    "error",
    "run_finished",
    "run_cancelled",
    "run_timeout",
]

_EVENT_TYPES = {
    "run_started",
    "turn_started",
    "prompt_ready",
    "prompt_profile",
    "model_response",
    "model_stream",
    "action_parsed",
    "approval_requested",
    "approval_resumed",
    "policy_decision",
    "tool_started",
    "tool_finished",
    "timeline_updated",
    "checkpoint_created",
    "loop_warning",
    "error",
    "run_finished",
    "run_cancelled",
    "run_timeout",
}
_TERMINAL_EVENT_TYPES = {"run_finished", "run_cancelled", "run_timeout"}


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class AgentEvent:
    type: EventType
    run_id: str = ""
    turn_id: str = ""
    sequence: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def manifest(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "sequence": self.sequence,
            "payload": dict(self.payload),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class EventStreamCursor:
    """Provider-neutral cursor for runtime-owned SSE/WebSocket/polling streams."""

    run_id: str = ""
    run_key: str = ""
    session_name: str = ""
    after_sequence: int = 0
    limit: int = 100
    event_types: tuple[EventType, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "after_sequence", max(0, int(self.after_sequence)))
        object.__setattr__(self, "limit", max(1, int(self.limit)))
        object.__setattr__(self, "event_types", tuple(dict.fromkeys(self.event_types)))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-event-stream-cursor/v1",
            "run_id": self.run_id,
            "run_key": self.run_key,
            "session_name": self.session_name,
            "after_sequence": self.after_sequence,
            "limit": self.limit,
            "event_types": list(self.event_types),
        }


@dataclass(frozen=True)
class EventStreamBatch:
    """One prompt-safe page of events ready for a runtime stream adapter."""

    cursor: EventStreamCursor
    events: tuple[AgentEvent, ...] = ()
    next_after_sequence: int = 0
    has_more: bool = False
    terminal: bool = False

    @classmethod
    def from_log(
        cls,
        log: "EventLogPort",
        cursor: EventStreamCursor | None = None,
    ) -> "EventStreamBatch":
        request = cursor or EventStreamCursor()
        records = log.records(run_id=request.run_id or None)
        selected = tuple(
            event
            for event in sorted(records, key=lambda item: item.sequence)
            if event.sequence > request.after_sequence
            and _event_matches_cursor(event, request)
            and (not request.event_types or event.type in request.event_types)
        )
        page = selected[: request.limit]
        return cls(
            cursor=request,
            events=page,
            next_after_sequence=page[-1].sequence if page else request.after_sequence,
            has_more=len(selected) > len(page),
            terminal=any(event.type in _TERMINAL_EVENT_TYPES for event in page),
        )

    def next_cursor(self) -> EventStreamCursor:
        return EventStreamCursor(
            run_id=self.cursor.run_id,
            run_key=self.cursor.run_key,
            session_name=self.cursor.session_name,
            after_sequence=self.next_after_sequence,
            limit=self.cursor.limit,
            event_types=self.cursor.event_types,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-event-stream-batch/v1",
            "cursor": self.cursor.manifest(),
            "next_cursor": self.next_cursor().manifest(),
            "event_count": len(self.events),
            "events": [event.manifest() for event in self.events],
            "next_after_sequence": self.next_after_sequence,
            "has_more": self.has_more,
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class EventStreamTail:
    """A bounded drain of available event pages for polling adapters."""

    start_cursor: EventStreamCursor
    batches: tuple[EventStreamBatch, ...] = ()
    max_batches: int = 10
    stop_at_terminal: bool = True

    @classmethod
    def from_log(
        cls,
        log: "EventLogPort",
        cursor: EventStreamCursor | None = None,
        *,
        max_batches: int = 10,
        stop_at_terminal: bool = True,
    ) -> "EventStreamTail":
        request = cursor or EventStreamCursor()
        limit = max(1, int(max_batches))
        batches: list[EventStreamBatch] = []
        current = request
        for _ in range(limit):
            batch = EventStreamBatch.from_log(log, current)
            batches.append(batch)
            if stop_at_terminal and batch.terminal:
                break
            if not batch.has_more:
                break
            current = batch.next_cursor()
        return cls(
            start_cursor=request,
            batches=tuple(batches),
            max_batches=limit,
            stop_at_terminal=stop_at_terminal,
        )

    @property
    def event_count(self) -> int:
        return sum(len(batch.events) for batch in self.batches)

    @property
    def terminal(self) -> bool:
        return any(batch.terminal for batch in self.batches)

    @property
    def has_more(self) -> bool:
        return bool(self.batches and self.batches[-1].has_more and not self.terminal)

    def next_cursor(self) -> EventStreamCursor:
        if not self.batches:
            return self.start_cursor
        return self.batches[-1].next_cursor()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-event-stream-tail/v1",
            "start_cursor": self.start_cursor.manifest(),
            "next_cursor": self.next_cursor().manifest(),
            "batch_count": len(self.batches),
            "event_count": self.event_count,
            "max_batches": self.max_batches,
            "stop_at_terminal": self.stop_at_terminal,
            "has_more": self.has_more,
            "terminal": self.terminal,
            "batches": [batch.manifest() for batch in self.batches],
        }


class EventSinkPort(Protocol):
    async def emit(self, event: AgentEvent) -> None:
        """Emit an agent event."""


class EventLogPort(EventSinkPort, Protocol):
    def records(self, *, run_id: str | None = None) -> tuple[AgentEvent, ...]:
        """Return stored events, optionally scoped to one run."""

    def manifest(self) -> dict[str, Any]:
        """Return a prompt-safe event log manifest."""


def _event_matches_cursor(event: AgentEvent, cursor: EventStreamCursor) -> bool:
    if cursor.run_id and event.run_id != cursor.run_id:
        return False
    if cursor.run_key and str(event.payload.get("run_key") or "") != cursor.run_key:
        return False
    if cursor.session_name and str(event.payload.get("session_name") or "") != cursor.session_name:
        return False
    return True


class NullEventSink:
    async def emit(self, event: AgentEvent) -> None:
        return None


class ListEventSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        if event.sequence <= 0:
            event = replace(event, sequence=len(self.events) + 1)
        self.events.append(event)

    def records(self, *, run_id: str | None = None) -> tuple[AgentEvent, ...]:
        events = tuple(self.events)
        if run_id is None:
            return events
        return tuple(event for event in events if event.run_id == run_id)

    def manifest(self) -> dict[str, Any]:
        events = self.records()
        return {
            "schema_version": "agent-core-event-log/v1",
            "backend": storage_backend_manifest(role="event_log", kind="in_memory"),
            "event_count": len(events),
            "events": [event.manifest() for event in events],
        }


class SQLiteEventSink:
    """SQLite-backed event log for durable local SDK runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    async def emit(self, event: AgentEvent) -> None:
        if event.sequence <= 0:
            event = replace(event, sequence=self._next_sequence())
        payload = json.dumps(event.payload, ensure_ascii=False, sort_keys=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO agent_events(sequence, type, run_id, turn_id, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.sequence,
                    event.type,
                    event.run_id,
                    event.turn_id,
                    payload,
                    event.created_at,
                ),
            )

    def records(self, *, run_id: str | None = None) -> tuple[AgentEvent, ...]:
        sql = """
            SELECT type, run_id, turn_id, sequence, payload_json, created_at
            FROM agent_events
        """
        params: tuple[Any, ...] = ()
        if run_id is not None:
            sql += " WHERE run_id = ?"
            params = (run_id,)
        sql += " ORDER BY row_id ASC"
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return tuple(_event_from_row(row) for row in rows)

    def manifest(self) -> dict[str, Any]:
        events = self.records()
        return {
            "schema_version": "agent-core-sqlite-event-sink/v1",
            "backend": storage_backend_manifest(
                role="event_log",
                kind="sqlite",
                location=str(self.path),
                capabilities=("emit", "records"),
            ),
            "path": str(self.path),
            "event_count": len(events),
            "events": [event.manifest() for event in events],
        }

    def _next_sequence(self) -> int:
        with sqlite3.connect(self.path) as conn:
            value = conn.execute("SELECT COUNT(*) FROM agent_events").fetchone()[0]
        return int(value) + 1

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_events (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sequence INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_events_run_id
                ON agent_events(run_id)
                """
            )


class MarkdownEventSink:
    """Markdown-backed event log for inspectable local SDK runs."""

    _START = "<!-- agent-core-event "
    _END = " -->"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("# Agent Events\n\n", encoding="utf-8")

    async def emit(self, event: AgentEvent) -> None:
        if event.sequence <= 0:
            event = replace(event, sequence=len(self.records()) + 1)
        events = (*self.records(), event)
        self._write(events)

    def records(self, *, run_id: str | None = None) -> tuple[AgentEvent, ...]:
        text = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        events: list[AgentEvent] = []
        for match in _EVENT_MARKDOWN_RE.finditer(text):
            try:
                events.append(_event_from_payload(_decode_event_payload(match.group("payload"))))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        if run_id is not None:
            events = [event for event in events if event.run_id == run_id]
        return tuple(events)

    def manifest(self) -> dict[str, Any]:
        events = self.records()
        return {
            "schema_version": "agent-core-markdown-event-sink/v1",
            "backend": storage_backend_manifest(
                role="event_log",
                kind="markdown",
                location=str(self.path),
                capabilities=("emit", "records"),
            ),
            "path": str(self.path),
            "event_count": len(events),
            "events": [event.manifest() for event in events],
        }

    def _write(self, events: tuple[AgentEvent, ...]) -> None:
        lines = [
            "# Agent Events",
            "",
            "This file is managed by raven_heart. Event payloads are stored in comments.",
            "",
        ]
        for event in events:
            lines.append(f"{self._START}{_encode_event_payload(event)}{self._END}")
            lines.append(
                f"- sequence: `{event.sequence}` type: `{event.type}` "
                f"run_id: `{event.run_id or '-'}` turn_id: `{event.turn_id or '-'}`"
            )
            lines.append("")
        self.path.write_text("\n".join(lines), encoding="utf-8")


_EVENT_MARKDOWN_RE = re.compile(
    r"<!--\s*agent-core-event\s+(?P<payload>[A-Za-z0-9+/=]+)\s*-->",
    re.DOTALL,
)


def _encode_event_payload(event: AgentEvent) -> str:
    raw = json.dumps(event.manifest(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _decode_event_payload(encoded: str) -> dict[str, Any]:
    raw = base64.b64decode(encoded.encode("ascii")).decode("utf-8")
    payload = json.loads(raw or "{}")
    if not isinstance(payload, dict):
        raise ValueError("invalid event payload")
    return payload


def _event_from_row(row: tuple[Any, ...]) -> AgentEvent:
    payload = json.loads(str(row[4] or "{}"))
    return AgentEvent(
        type=_event_type(str(row[0] or "")),
        run_id=str(row[1] or ""),
        turn_id=str(row[2] or ""),
        sequence=int(row[3] or 0),
        payload=dict(payload) if isinstance(payload, dict) else {},
        created_at=str(row[5] or utc_now_iso()),
    )


def _event_from_payload(payload: dict[str, Any]) -> AgentEvent:
    event_payload = payload.get("payload")
    return AgentEvent(
        type=_event_type(str(payload.get("type") or "")),
        run_id=str(payload.get("run_id") or ""),
        turn_id=str(payload.get("turn_id") or ""),
        sequence=int(payload.get("sequence") or 0),
        payload=dict(event_payload) if isinstance(event_payload, dict) else {},
        created_at=str(payload.get("created_at") or utc_now_iso()),
    )


def _event_type(value: str) -> EventType:
    if value not in _EVENT_TYPES:
        raise ValueError(f"invalid event type: {value}")
    return value  # type: ignore[return-value]

