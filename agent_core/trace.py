"""Trace records for prompt, tool, and cost observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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

