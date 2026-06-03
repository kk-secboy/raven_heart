"""Run lifecycle, checkpoint, and resume protocol."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from agent_core.errors import HarnessError, ResumeError


RunStatus = Literal[
    "running",
    "completed",
    "cancelled",
    "failed",
    "max_iterations",
    "stalled",
    "denied",
    "approval_required",
]
TurnStatus = Literal["running", "completed", "cancelled", "failed"]
TERMINAL_RUN_STATUSES = frozenset(
    {
        "completed",
        "cancelled",
        "failed",
        "max_iterations",
        "stalled",
        "denied",
        "approval_required",
    }
)
TERMINAL_TURN_STATUSES = frozenset({"completed", "cancelled", "failed"})


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class RunState:
    run_id: str = field(default_factory=lambda: uuid4().hex)
    status: RunStatus = "running"
    task: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnState:
    run_id: str
    turn_id: str = field(default_factory=lambda: uuid4().hex)
    index: int = 0
    status: TurnStatus = "running"
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Checkpoint:
    run_id: str
    checkpoint_id: str = field(default_factory=lambda: uuid4().hex)
    turn_id: str = ""
    sequence: int = 0
    state: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class ResumeToken:
    run_id: str
    checkpoint_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-resume-token/v1",
            "run_id": self.run_id,
            "checkpoint_id": self.checkpoint_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ResumeCandidate:
    """Latest checkpoint material for one resumable run."""

    run_id: str
    status: RunStatus
    task: str
    checkpoint_id: str
    checkpoint_sequence: int
    turn_id: str = ""
    checkpoint_state: dict[str, Any] = field(default_factory=dict)
    run_metadata: dict[str, Any] = field(default_factory=dict)
    token: ResumeToken | None = None
    run_created_at: str = ""
    checkpoint_created_at: str = ""

    @classmethod
    def from_parts(cls, run: RunState, checkpoint: Checkpoint) -> "ResumeCandidate":
        token = ResumeToken(
            run_id=checkpoint.run_id,
            checkpoint_id=checkpoint.checkpoint_id,
            metadata={"turn_id": checkpoint.turn_id, "sequence": checkpoint.sequence},
        )
        return cls(
            run_id=run.run_id,
            status=run.status,
            task=run.task,
            checkpoint_id=checkpoint.checkpoint_id,
            checkpoint_sequence=checkpoint.sequence,
            turn_id=checkpoint.turn_id,
            checkpoint_state=dict(checkpoint.state),
            run_metadata=dict(run.metadata),
            token=token,
            run_created_at=run.created_at,
            checkpoint_created_at=checkpoint.created_at,
        )

    @property
    def terminal(self) -> bool:
        return _is_terminal_run(self.status)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-resume-candidate/v1",
            "run_id": self.run_id,
            "status": self.status,
            "terminal": self.terminal,
            "task": self.task,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_sequence": self.checkpoint_sequence,
            "turn_id": self.turn_id,
            "checkpoint_state": dict(self.checkpoint_state),
            "run_metadata": dict(self.run_metadata),
            "token": self.token.manifest() if self.token is not None else None,
            "run_created_at": self.run_created_at,
            "checkpoint_created_at": self.checkpoint_created_at,
        }


@dataclass(frozen=True)
class ResumeIndex:
    """Manifest-friendly index of latest checkpoints by run."""

    candidates: tuple[ResumeCandidate, ...] = ()
    include_terminal: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: "AgentJournalSnapshot",
        *,
        include_terminal: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> "ResumeIndex":
        runs = {
            str(item.get("run_id") or ""): RunState(
                run_id=str(item.get("run_id") or ""),
                status=_run_status(str(item.get("status") or "running")),
                task=str(item.get("task") or ""),
                created_at=str(item.get("created_at") or utc_now_iso()),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in snapshot.runs
            if str(item.get("run_id") or "")
        }
        latest_by_run: dict[str, Checkpoint] = {}
        for item in snapshot.checkpoints:
            checkpoint = Checkpoint(
                run_id=str(item.get("run_id") or ""),
                checkpoint_id=str(item.get("checkpoint_id") or ""),
                turn_id=str(item.get("turn_id") or ""),
                sequence=int(item.get("sequence") or 0),
                state=dict(item.get("state") or {}),
                created_at=str(item.get("created_at") or utc_now_iso()),
            )
            if not checkpoint.run_id or checkpoint.run_id not in runs:
                continue
            previous = latest_by_run.get(checkpoint.run_id)
            if previous is None or checkpoint.sequence >= previous.sequence:
                latest_by_run[checkpoint.run_id] = checkpoint
        candidates = []
        for run_id, checkpoint in latest_by_run.items():
            run = runs[run_id]
            if not include_terminal and _is_terminal_run(run.status):
                continue
            candidates.append(ResumeCandidate.from_parts(run, checkpoint))
        return cls(
            candidates=tuple(
                sorted(candidates, key=lambda item: (item.checkpoint_created_at, item.run_id))
            ),
            include_terminal=include_terminal,
            metadata=dict(metadata or {}),
        )

    def latest(self) -> ResumeCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[-1]

    def for_run(self, run_id: str) -> ResumeCandidate | None:
        for candidate in self.candidates:
            if candidate.run_id == run_id:
                return candidate
        return None

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-resume-index/v1",
            "candidate_count": len(self.candidates),
            "include_terminal": self.include_terminal,
            "candidates": [candidate.manifest() for candidate in self.candidates],
            "metadata": dict(self.metadata),
        }


@dataclass
class CancelToken:
    cancelled: bool = False
    reason: str = ""

    def cancel(self, reason: str = "") -> None:
        self.cancelled = True
        self.reason = reason


class AgentHarness(Protocol):
    async def start_run(self, task: str, metadata: dict[str, Any] | None = None) -> RunState:
        """Start a run."""

    async def start_turn(self, run: RunState, index: int) -> TurnState:
        """Start one turn."""

    async def record_prompt(self, turn: TurnState, manifest: dict[str, Any]) -> None:
        """Record prompt metadata without requiring raw prompt disclosure."""

    async def record_model_event(self, turn: TurnState, event: dict[str, Any]) -> None:
        """Record a provider/model event."""

    async def record_tool_call(self, turn: TurnState, event: dict[str, Any]) -> None:
        """Record a tool call boundary."""

    async def checkpoint(self, turn: TurnState, state: dict[str, Any]) -> Checkpoint:
        """Create a resumable checkpoint."""

    async def resume(self, token: ResumeToken) -> Checkpoint:
        """Load a checkpoint."""

    async def finish_run(self, run: RunState, status: str, result: dict[str, Any]) -> None:
        """Finish a run."""

    async def record_error(
        self,
        *,
        run: RunState | None = None,
        turn: TurnState | None = None,
        error: BaseException | str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a lifecycle or runtime error boundary."""


@dataclass(frozen=True)
class AgentJournalSnapshot:
    schema_version: str = "agent-core-journal/v1"
    runs: tuple[dict[str, Any], ...] = ()
    turns: tuple[dict[str, Any], ...] = ()
    prompts: tuple[dict[str, Any], ...] = ()
    model_events: tuple[dict[str, Any], ...] = ()
    tool_calls: tuple[dict[str, Any], ...] = ()
    checkpoints: tuple[dict[str, Any], ...] = ()
    errors: tuple[dict[str, Any], ...] = ()
    finished: tuple[dict[str, Any], ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "runs": list(self.runs),
            "turns": list(self.turns),
            "prompts": list(self.prompts),
            "model_events": list(self.model_events),
            "tool_calls": list(self.tool_calls),
            "checkpoints": list(self.checkpoints),
            "errors": list(self.errors),
            "finished": list(self.finished),
        }


class AgentJournalStorePort(Protocol):
    """Persistence boundary for harness journals."""

    def load_snapshot(self) -> AgentJournalSnapshot | None:
        """Load the latest durable journal snapshot."""

    def save_snapshot(self, snapshot: AgentJournalSnapshot) -> None:
        """Persist the latest journal snapshot."""

    def manifest(self) -> dict[str, Any]:
        """Describe the backing store without leaking implementation details."""


class InMemoryAgentJournal(AgentHarness):
    """In-memory run journal suitable for replay tests and adapter prototypes."""

    def __init__(self, snapshot: AgentJournalSnapshot | None = None) -> None:
        self.runs: dict[str, RunState] = {}
        self.turns: list[TurnState] = []
        self.prompts: list[dict[str, Any]] = []
        self.model_events: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.checkpoints: list[Checkpoint] = []
        self.errors: list[dict[str, Any]] = []
        self.finished: list[dict[str, Any]] = []
        if snapshot is not None:
            self._load_snapshot(snapshot)

    async def start_run(self, task: str, metadata: dict[str, Any] | None = None) -> RunState:
        run = RunState(task=task, metadata=dict(metadata or {}))
        self.runs[run.run_id] = run
        return run

    async def start_turn(self, run: RunState, index: int) -> TurnState:
        current = self._require_run(run.run_id)
        if _is_terminal_run(current.status):
            raise HarnessError(f"cannot start turn for terminal run: {current.status}")
        turn = TurnState(run_id=run.run_id, index=index)
        self.turns.append(turn)
        return turn

    async def record_prompt(self, turn: TurnState, manifest: dict[str, Any]) -> None:
        self._require_active_turn(turn)
        self.prompts.append({"run_id": turn.run_id, "turn_id": turn.turn_id, "manifest": dict(manifest)})

    async def record_model_event(self, turn: TurnState, event: dict[str, Any]) -> None:
        self._require_active_turn(turn)
        self.model_events.append({"run_id": turn.run_id, "turn_id": turn.turn_id, **dict(event)})

    async def record_tool_call(self, turn: TurnState, event: dict[str, Any]) -> None:
        self._require_active_turn(turn)
        self.tool_calls.append({"run_id": turn.run_id, "turn_id": turn.turn_id, **dict(event)})

    async def checkpoint(self, turn: TurnState, state: dict[str, Any]) -> Checkpoint:
        self._require_active_turn(turn)
        checkpoint = Checkpoint(
            run_id=turn.run_id,
            turn_id=turn.turn_id,
            sequence=len([item for item in self.checkpoints if item.run_id == turn.run_id]) + 1,
            state=dict(state),
        )
        self.checkpoints.append(checkpoint)
        return checkpoint

    async def resume(self, token: ResumeToken) -> Checkpoint:
        for checkpoint in self.checkpoints:
            if checkpoint.run_id == token.run_id and checkpoint.checkpoint_id == token.checkpoint_id:
                sequence = token.metadata.get("sequence")
                if sequence is not None and int(sequence) != checkpoint.sequence:
                    raise ResumeError("resume token sequence does not match checkpoint")
                return checkpoint
        raise ResumeError(token.checkpoint_id)

    async def finish_run(self, run: RunState, status: str, result: dict[str, Any]) -> None:
        current = self._require_run(run.run_id)
        if _is_terminal_run(current.status):
            raise HarnessError(f"run already finished: {current.status}")
        if status not in TERMINAL_RUN_STATUSES:
            raise HarnessError(f"invalid terminal run status: {status}")
        finished = RunState(
            run_id=run.run_id,
            status=status,  # type: ignore[arg-type]
            task=current.task,
            created_at=current.created_at,
            metadata=dict(current.metadata),
        )
        self.runs[run.run_id] = finished
        self._finish_open_turns(run.run_id, _turn_status_for_run(status))
        self.finished.append(
            {
                "run_id": run.run_id,
                "status": status,
                "result": dict(result),
                "finished_at": utc_now_iso(),
            }
        )

    async def record_error(
        self,
        *,
        run: RunState | None = None,
        turn: TurnState | None = None,
        error: BaseException | str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        message = str(error)
        self.errors.append(
            {
                "run_id": (turn.run_id if turn is not None else run.run_id if run is not None else ""),
                "turn_id": turn.turn_id if turn is not None else "",
                "error_type": error.__class__.__name__ if isinstance(error, BaseException) else "Error",
                "message": message,
                "metadata": dict(metadata or {}),
                "created_at": utc_now_iso(),
            }
        )

    def latest_checkpoint(self, run_id: str) -> Checkpoint | None:
        matches = [checkpoint for checkpoint in self.checkpoints if checkpoint.run_id == run_id]
        if not matches:
            return None
        return sorted(matches, key=lambda item: item.sequence)[-1]

    def resume_token(self, checkpoint: Checkpoint) -> ResumeToken:
        return ResumeToken(
            run_id=checkpoint.run_id,
            checkpoint_id=checkpoint.checkpoint_id,
            metadata={"turn_id": checkpoint.turn_id, "sequence": checkpoint.sequence},
        )

    def resume_index(self, *, include_terminal: bool = True) -> ResumeIndex:
        return ResumeIndex.from_snapshot(
            self.snapshot(),
            include_terminal=include_terminal,
            metadata={"source": type(self).__name__},
        )

    def resumable_runs(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "run_id": candidate.run_id,
                "status": candidate.status,
                "task": candidate.task,
                "checkpoint_id": candidate.checkpoint_id,
                "checkpoint_sequence": candidate.checkpoint_sequence,
                "turn_id": candidate.turn_id,
                "created_at": candidate.checkpoint_created_at,
            }
            for candidate in self.resume_index().candidates
        )

    def run_manifest(self, run_id: str) -> dict[str, Any]:
        run = self._require_run(run_id)
        return {
            "run": _run_to_dict(run),
            "turns": [_turn_to_dict(turn) for turn in self.turns if turn.run_id == run_id],
            "prompts": [dict(item) for item in self.prompts if item.get("run_id") == run_id],
            "model_events": [dict(item) for item in self.model_events if item.get("run_id") == run_id],
            "tool_calls": [dict(item) for item in self.tool_calls if item.get("run_id") == run_id],
            "checkpoints": [
                _checkpoint_to_dict(checkpoint)
                for checkpoint in self.checkpoints
                if checkpoint.run_id == run_id
            ],
            "errors": [dict(item) for item in self.errors if item.get("run_id") == run_id],
            "finished": [dict(item) for item in self.finished if item.get("run_id") == run_id],
        }

    def snapshot(self) -> AgentJournalSnapshot:
        return AgentJournalSnapshot(
            runs=tuple(_run_to_dict(run) for run in sorted(self.runs.values(), key=lambda item: item.created_at)),
            turns=tuple(_turn_to_dict(turn) for turn in self.turns),
            prompts=tuple(dict(item) for item in self.prompts),
            model_events=tuple(dict(item) for item in self.model_events),
            tool_calls=tuple(dict(item) for item in self.tool_calls),
            checkpoints=tuple(_checkpoint_to_dict(checkpoint) for checkpoint in self.checkpoints),
            errors=tuple(dict(item) for item in self.errors),
            finished=tuple(dict(item) for item in self.finished),
        )

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> "InMemoryAgentJournal":
        return cls(_snapshot_from_manifest(manifest))

    def _load_snapshot(self, snapshot: AgentJournalSnapshot) -> None:
        for item in snapshot.runs:
            run = RunState(
                run_id=str(item.get("run_id") or ""),
                status=_run_status(str(item.get("status") or "running")),
                task=str(item.get("task") or ""),
                created_at=str(item.get("created_at") or utc_now_iso()),
                metadata=dict(item.get("metadata") or {}),
            )
            self.runs[run.run_id] = run
        self.turns.extend(
            TurnState(
                run_id=str(item.get("run_id") or ""),
                turn_id=str(item.get("turn_id") or ""),
                index=int(item.get("index") or 0),
                status=_turn_status(str(item.get("status") or "running")),
                created_at=str(item.get("created_at") or utc_now_iso()),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in snapshot.turns
        )
        self.prompts.extend(dict(item) for item in snapshot.prompts)
        self.model_events.extend(dict(item) for item in snapshot.model_events)
        self.tool_calls.extend(dict(item) for item in snapshot.tool_calls)
        self.checkpoints.extend(
            Checkpoint(
                run_id=str(item.get("run_id") or ""),
                checkpoint_id=str(item.get("checkpoint_id") or ""),
                turn_id=str(item.get("turn_id") or ""),
                sequence=int(item.get("sequence") or 0),
                state=dict(item.get("state") or {}),
                created_at=str(item.get("created_at") or utc_now_iso()),
            )
            for item in snapshot.checkpoints
        )
        self.errors.extend(dict(item) for item in snapshot.errors)
        self.finished.extend(dict(item) for item in snapshot.finished)

    def _require_run(self, run_id: str) -> RunState:
        run = self.runs.get(run_id)
        if run is None:
            raise HarnessError(f"unknown run: {run_id}")
        return run

    def _require_active_turn(self, turn: TurnState) -> TurnState:
        self._require_run(turn.run_id)
        current = next(
            (item for item in self.turns if item.run_id == turn.run_id and item.turn_id == turn.turn_id),
            None,
        )
        if current is None:
            raise HarnessError(f"unknown turn: {turn.turn_id}")
        if _is_terminal_turn(current.status):
            raise HarnessError(f"turn already finished: {current.status}")
        if _is_terminal_run(self.runs[turn.run_id].status):
            raise HarnessError(f"run already finished: {self.runs[turn.run_id].status}")
        return current

    def _finish_open_turns(self, run_id: str, status: TurnStatus) -> None:
        self.turns = [
            TurnState(
                run_id=turn.run_id,
                turn_id=turn.turn_id,
                index=turn.index,
                status=status if turn.run_id == run_id and not _is_terminal_turn(turn.status) else turn.status,
                created_at=turn.created_at,
                metadata=dict(turn.metadata),
            )
            for turn in self.turns
        ]


class InMemoryJournalStore(AgentJournalStorePort):
    """Snapshot store useful for tests and in-process SDK embeddings."""

    def __init__(self, snapshot: AgentJournalSnapshot | None = None) -> None:
        self._snapshot = _copy_snapshot(snapshot) if snapshot is not None else None

    def load_snapshot(self) -> AgentJournalSnapshot | None:
        return _copy_snapshot(self._snapshot) if self._snapshot is not None else None

    def save_snapshot(self, snapshot: AgentJournalSnapshot) -> None:
        self._snapshot = _copy_snapshot(snapshot)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-in-memory-journal-store/v1",
            "has_snapshot": self._snapshot is not None,
        }


class SQLiteJournalStore(AgentJournalStorePort):
    """SQLite snapshot store for durable local SDK runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def load_snapshot(self) -> AgentJournalSnapshot | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                """
                SELECT manifest_json
                FROM agent_journal_state
                WHERE id = 1
                """
            ).fetchone()
        if row is None:
            return None
        try:
            manifest = json.loads(str(row[0] or "{}"))
        except json.JSONDecodeError as exc:
            raise ResumeError(f"invalid SQLite journal snapshot: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ResumeError("invalid SQLite journal snapshot")
        return _snapshot_from_manifest(manifest)

    def save_snapshot(self, snapshot: AgentJournalSnapshot) -> None:
        raw = json.dumps(snapshot.manifest(), ensure_ascii=False, sort_keys=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO agent_journal_state(id, manifest_json, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    manifest_json = excluded.manifest_json,
                    updated_at = excluded.updated_at
                """,
                (raw, utc_now_iso()),
            )
            conn.commit()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-sqlite-journal-store/v1",
            "path": str(self.path),
        }

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_journal_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    manifest_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()


class MarkdownJournalStore(AgentJournalStorePort):
    """Markdown-backed journal snapshot store for inspectable local runs."""

    _START = "<!-- agent-journal-snapshot"
    _END = "-->"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_snapshot(self) -> AgentJournalSnapshot | None:
        if not self.path.exists():
            return None
        text = self.path.read_text(encoding="utf-8")
        start = text.find(self._START)
        if start < 0:
            return None
        payload_start = start + len(self._START)
        end = text.find(self._END, payload_start)
        if end < 0:
            raise ResumeError("invalid Markdown journal snapshot")
        raw = text[payload_start:end].strip()
        try:
            manifest = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise ResumeError(f"invalid Markdown journal snapshot: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ResumeError("invalid Markdown journal snapshot")
        return _snapshot_from_manifest(manifest)

    def save_snapshot(self, snapshot: AgentJournalSnapshot) -> None:
        raw = json.dumps(snapshot.manifest(), ensure_ascii=False, indent=2, sort_keys=True)
        body = (
            "# Agent Journal Snapshot\n\n"
            "This file is managed by raven_heart. The latest resumable state is stored below.\n\n"
            f"{self._START}\n"
            f"{raw}\n"
            f"{self._END}\n"
        )
        self.path.write_text(body, encoding="utf-8")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-markdown-journal-store/v1",
            "path": str(self.path),
        }


class PersistentAgentJournal(InMemoryAgentJournal):
    """Harness journal that persists through a pluggable snapshot store."""

    def __init__(
        self,
        store: AgentJournalStorePort,
        snapshot: AgentJournalSnapshot | None = None,
    ) -> None:
        self.store = store
        super().__init__(snapshot if snapshot is not None else store.load_snapshot())
        if snapshot is not None:
            self._persist()

    async def start_run(self, task: str, metadata: dict[str, Any] | None = None) -> RunState:
        run = await super().start_run(task, metadata)
        self._persist()
        return run

    async def start_turn(self, run: RunState, index: int) -> TurnState:
        turn = await super().start_turn(run, index)
        self._persist()
        return turn

    async def record_prompt(self, turn: TurnState, manifest: dict[str, Any]) -> None:
        await super().record_prompt(turn, manifest)
        self._persist()

    async def record_model_event(self, turn: TurnState, event: dict[str, Any]) -> None:
        await super().record_model_event(turn, event)
        self._persist()

    async def record_tool_call(self, turn: TurnState, event: dict[str, Any]) -> None:
        await super().record_tool_call(turn, event)
        self._persist()

    async def checkpoint(self, turn: TurnState, state: dict[str, Any]) -> Checkpoint:
        checkpoint = await super().checkpoint(turn, state)
        self._persist()
        return checkpoint

    async def finish_run(self, run: RunState, status: str, result: dict[str, Any]) -> None:
        await super().finish_run(run, status, result)
        self._persist()

    async def record_error(
        self,
        *,
        run: RunState | None = None,
        turn: TurnState | None = None,
        error: BaseException | str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await super().record_error(run=run, turn=turn, error=error, metadata=metadata)
        self._persist()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-persistent-journal/v1",
            "store": self.store.manifest(),
            "snapshot": self.snapshot().manifest(),
        }

    def _persist(self) -> None:
        self.store.save_snapshot(self.snapshot())


class SQLiteAgentJournal(PersistentAgentJournal):
    """Convenience wrapper for ``PersistentAgentJournal(SQLiteJournalStore(...))``."""

    def __init__(self, path: str | Path, snapshot: AgentJournalSnapshot | None = None) -> None:
        store = SQLiteJournalStore(path)
        self.path = store.path
        super().__init__(store, snapshot=snapshot)


def _run_to_dict(run: RunState) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "status": run.status,
        "task": run.task,
        "created_at": run.created_at,
        "metadata": dict(run.metadata),
    }


def _turn_to_dict(turn: TurnState) -> dict[str, Any]:
    return {
        "run_id": turn.run_id,
        "turn_id": turn.turn_id,
        "index": turn.index,
        "status": turn.status,
        "created_at": turn.created_at,
        "metadata": dict(turn.metadata),
    }


def _checkpoint_to_dict(checkpoint: Checkpoint) -> dict[str, Any]:
    return {
        "run_id": checkpoint.run_id,
        "checkpoint_id": checkpoint.checkpoint_id,
        "turn_id": checkpoint.turn_id,
        "sequence": checkpoint.sequence,
        "state": dict(checkpoint.state),
        "created_at": checkpoint.created_at,
    }


def _is_terminal_run(status: str) -> bool:
    return status in TERMINAL_RUN_STATUSES


def _is_terminal_turn(status: str) -> bool:
    return status in TERMINAL_TURN_STATUSES


def _run_status(value: str) -> RunStatus:
    if value not in {"running", *TERMINAL_RUN_STATUSES}:
        raise HarnessError(f"invalid run status: {value}")
    return value  # type: ignore[return-value]


def _turn_status(value: str) -> TurnStatus:
    if value not in {"running", *TERMINAL_TURN_STATUSES}:
        raise HarnessError(f"invalid turn status: {value}")
    return value  # type: ignore[return-value]


def _turn_status_for_run(status: str) -> TurnStatus:
    if status == "cancelled":
        return "cancelled"
    if status == "failed":
        return "failed"
    return "completed"


def _copy_snapshot(snapshot: AgentJournalSnapshot) -> AgentJournalSnapshot:
    return _snapshot_from_manifest(snapshot.manifest())


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

