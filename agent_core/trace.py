"""Trace records for prompt, tool, and cost observations."""

from __future__ import annotations

import base64
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agent_core.backends import storage_backend_manifest
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
class StorageBackendTrace:
    """Run-level inventory of storage backends visible in trace manifests."""

    backends: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_trace_components(
        cls,
        **components: dict[str, Any],
    ) -> "StorageBackendTrace":
        entries: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
        for source, manifest in components.items():
            for backend, path in _iter_storage_backend_manifests(manifest, source):
                key = _storage_backend_key(backend)
                current = entries.get(key)
                if current is None:
                    current = dict(backend)
                    current["sources"] = []
                    entries[key] = current
                current["sources"] = sorted({*current.get("sources", ()), path})
        ordered = tuple(sorted(entries.values(), key=_storage_backend_sort_key))
        return cls(backends=ordered)

    def manifest(self) -> dict[str, Any]:
        backends = tuple(dict(item) for item in self.backends)
        role_counts = _count_backend_field(backends, "role")
        kind_counts = _count_backend_field(backends, "kind")
        return {
            "schema_version": "agent-core-storage-backend-trace/v1",
            "backend_count": len(backends),
            "core_builtin_count": sum(1 for item in backends if item.get("core_builtin") is True),
            "external_backend_count": sum(1 for item in backends if item.get("core_builtin") is False),
            "roles": role_counts,
            "kinds": kind_counts,
            "backends": list(backends),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ContextInjectionTrace:
    """Run-level inventory of prompt context injection decisions."""

    injections: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_prompt(cls, prompt: dict[str, Any]) -> "ContextInjectionTrace":
        metadata = prompt.get("metadata") if isinstance(prompt.get("metadata"), dict) else {}
        raw = metadata.get("context_injections") if isinstance(metadata, dict) else ()
        injections = tuple(dict(item) for item in raw or () if isinstance(item, dict))
        return cls(injections=injections)

    def manifest(self) -> dict[str, Any]:
        injections = tuple(dict(item) for item in self.injections)
        included = tuple(item for item in injections if item.get("included") is not False)
        excluded = tuple(item for item in injections if item.get("included") is False)
        trimmed = tuple(item for item in included if item.get("trimmed") is True)
        return {
            "schema_version": "agent-core-context-injection-trace/v1",
            "injection_count": len(injections),
            "included_count": len(included),
            "excluded_count": len(excluded),
            "trimmed_count": len(trimmed),
            "sources": _count_injection_field(injections, "source"),
            "targets": _count_injection_field(injections, "target"),
            "statuses": _count_injection_field(injections, "status"),
            "injections": list(injections),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MemoryGovernanceTrace:
    """Run-level inventory of memory write governance decisions."""

    decisions: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_session(cls, session: dict[str, Any]) -> "MemoryGovernanceTrace":
        memory = session.get("memory") if isinstance(session.get("memory"), dict) else {}
        governance = memory.get("governance") if isinstance(memory.get("governance"), dict) else {}
        raw = governance.get("decisions") if isinstance(governance, dict) else ()
        decisions = tuple(dict(item) for item in raw or () if isinstance(item, dict))
        return cls(decisions=decisions)

    def manifest(self) -> dict[str, Any]:
        decisions = tuple(dict(item) for item in self.decisions)
        allowed = tuple(item for item in decisions if item.get("allowed") is True)
        denied = tuple(item for item in decisions if item.get("decision") == "deny")
        rewritten = tuple(item for item in decisions if item.get("decision") == "rewrite")
        return {
            "schema_version": "agent-core-memory-governance-trace/v1",
            "decision_count": len(decisions),
            "allowed_count": len(allowed),
            "denied_count": len(denied),
            "rewritten_count": len(rewritten),
            "decisions_by_status": _count_injection_field(decisions, "decision"),
            "risk_levels": _count_injection_field(decisions, "risk_level"),
            "stores": _count_injection_field(decisions, "store"),
            "reasons": _count_injection_field(decisions, "reason"),
            "decisions": list(decisions),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceCorrelationEntry:
    """One normalized pointer into trace materials."""

    source: str
    kind: str
    run_id: str = ""
    turn_id: str = ""
    sequence: int = 0
    call_id: str = ""
    approval_id: str = ""
    decision_id: str = ""
    subject: str = ""
    status: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        parts = [self.source, self.kind]
        if self.run_id:
            parts.append(f"run={self.run_id}")
        if self.turn_id:
            parts.append(f"turn={self.turn_id}")
        if self.call_id:
            parts.append(f"call={self.call_id}")
        if self.approval_id:
            parts.append(f"approval={self.approval_id}")
        if self.decision_id:
            parts.append(f"decision={self.decision_id}")
        if self.sequence:
            parts.append(f"seq={self.sequence}")
        return "|".join(parts)

    def manifest(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "source": self.source,
            "kind": self.kind,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "sequence": self.sequence,
            "call_id": self.call_id,
            "approval_id": self.approval_id,
            "decision_id": self.decision_id,
            "subject": self.subject,
            "status": self.status,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceCorrelationIndex:
    """Cross-reference index for one run trace bundle."""

    entries: tuple[TraceCorrelationEntry, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_trace_components(
        cls,
        *,
        run_id: str,
        journal_replay: dict[str, Any] | None = None,
        provider: dict[str, Any] | None = None,
        tool_replay: dict[str, Any] | None = None,
        policy_decisions: dict[str, Any] | None = None,
        approvals: dict[str, Any] | None = None,
        event_log: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "TraceCorrelationIndex":
        entries: list[TraceCorrelationEntry] = []
        entries.extend(_correlate_journal(run_id, journal_replay or {}))
        entries.extend(_correlate_provider(run_id, provider or {}))
        entries.extend(_correlate_tool_replay(run_id, tool_replay or {}))
        entries.extend(_correlate_policy_decisions(run_id, policy_decisions or {}))
        entries.extend(_correlate_approvals(run_id, approvals or {}))
        entries.extend(_correlate_events(run_id, event_log or {}))
        return cls(
            entries=tuple(sorted(entries, key=_correlation_sort_key)),
            metadata=dict(metadata or {}),
        )

    def manifest(self) -> dict[str, Any]:
        entries = tuple(self.entries)
        return {
            "schema_version": "agent-core-trace-correlation-index/v1",
            "entry_count": len(entries),
            "entries": [entry.manifest() for entry in entries],
            "groups": {
                "turns": _group_entries(entries, lambda item: _join_id(item.run_id, item.turn_id)),
                "calls": _group_entries(entries, lambda item: item.call_id),
                "approvals": _group_entries(entries, lambda item: item.approval_id),
                "policy_decisions": _group_entries(entries, lambda item: item.decision_id),
                "subjects": _group_entries(entries, lambda item: item.subject),
            },
            "metadata": dict(self.metadata),
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
    resume_plan: dict[str, Any] = field(default_factory=dict)
    timeline_reduction: dict[str, Any] = field(default_factory=dict)
    capability_discovery: dict[str, Any] = field(default_factory=dict)
    memory_search: dict[str, Any] = field(default_factory=dict)
    storage_backends: dict[str, Any] = field(default_factory=dict)
    context_injections: dict[str, Any] = field(default_factory=dict)
    memory_governance: dict[str, Any] = field(default_factory=dict)
    correlation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        journal_ok = self.journal_replay.get("ok")
        storage_backends = self.storage_backends or StorageBackendTrace.from_trace_components(
            session=self.session,
            provider=self.provider,
            tool_replay=self.tool_replay,
            policy_decisions=self.policy_decisions,
            approvals=self.approvals,
            event_log=self.event_log,
            memory_search=self.memory_search,
        ).manifest()
        context_injections = self.context_injections or ContextInjectionTrace.from_prompt(
            self.prompt
        ).manifest()
        memory_governance = self.memory_governance or MemoryGovernanceTrace.from_session(
            self.session
        ).manifest()
        correlation = self.correlation or TraceCorrelationIndex.from_trace_components(
            run_id=self.run_id,
            journal_replay=self.journal_replay,
            provider=self.provider,
            tool_replay=self.tool_replay,
            policy_decisions=self.policy_decisions,
            approvals=self.approvals,
            event_log=self.event_log,
            metadata={"generated_by": "AgentRunTraceBundle"},
        ).manifest()
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
                "correlation_entry_count": int(correlation.get("entry_count") or 0),
                "has_resume": bool(self.resume),
                "has_resume_plan": bool(self.resume_plan),
                "resume_plan_ready": bool(self.resume_plan.get("ready"))
                if self.resume_plan
                else False,
                "has_timeline_reduction": bool(self.timeline_reduction),
                "capability_discovery_match_count": int(
                    self.capability_discovery.get("match_count") or 0
                ),
                "memory_search_hit_count": int(self.memory_search.get("hit_count") or 0),
                "storage_backend_count": int(storage_backends.get("backend_count") or 0),
                "external_storage_backend_count": int(
                    storage_backends.get("external_backend_count") or 0
                ),
                "context_injection_count": int(context_injections.get("injection_count") or 0),
                "context_injection_trimmed_count": int(
                    context_injections.get("trimmed_count") or 0
                ),
                "context_injection_excluded_count": int(
                    context_injections.get("excluded_count") or 0
                ),
                "memory_governance_decision_count": int(
                    memory_governance.get("decision_count") or 0
                ),
                "memory_governance_denied_count": int(
                    memory_governance.get("denied_count") or 0
                ),
                "memory_governance_rewritten_count": int(
                    memory_governance.get("rewritten_count") or 0
                ),
                "has_prompt_trim": bool(self.prompt.get("metadata", {}).get("trim")),
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
            "resume_plan": dict(self.resume_plan),
            "timeline_reduction": dict(self.timeline_reduction),
            "capability_discovery": dict(self.capability_discovery),
            "memory_search": dict(self.memory_search),
            "storage_backends": dict(storage_backends),
            "context_injections": dict(context_injections),
            "memory_governance": dict(memory_governance),
            "correlation": dict(correlation),
            "metadata": dict(self.metadata),
        }


def _iter_storage_backend_manifests(
    value: Any,
    source: str,
) -> tuple[tuple[dict[str, Any], str], ...]:
    found: list[tuple[dict[str, Any], str]] = []

    def walk(item: Any, path: str) -> None:
        if isinstance(item, dict):
            if item.get("schema_version") == "agent-core-storage-backend/v1":
                found.append((dict(item), path))
                return
            for key, child in item.items():
                walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")

    walk(value, source)
    return tuple(found)


def _storage_backend_key(backend: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(backend.get("role") or ""),
        str(backend.get("kind") or ""),
        str(backend.get("name") or ""),
        str(backend.get("namespace") or ""),
        str(backend.get("location") or ""),
    )


def _storage_backend_sort_key(backend: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return _storage_backend_key(backend)


def _count_backend_field(backends: tuple[dict[str, Any], ...], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for backend in backends:
        value = str(backend.get(field_name) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _count_injection_field(injections: tuple[dict[str, Any], ...], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for injection in injections:
        value = str(injection.get(field_name) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


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
        return {
            "schema_version": "agent-core-null-run-trace-store/v1",
            "backend": storage_backend_manifest(role="run_trace", kind="none"),
            "record_count": 0,
        }


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
            "backend": storage_backend_manifest(role="run_trace", kind="in_memory"),
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
            "backend": storage_backend_manifest(
                role="run_trace",
                kind="sqlite",
                location=str(self.path),
                capabilities=("save", "load", "records"),
            ),
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
            "backend": storage_backend_manifest(
                role="run_trace",
                kind="markdown",
                location=str(self.path),
                capabilities=("save", "load", "records"),
            ),
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


def _correlate_journal(run_id: str, manifest: dict[str, Any]) -> tuple[TraceCorrelationEntry, ...]:
    entries = []
    for event in _dict_items(manifest.get("events")):
        if run_id and str(event.get("run_id") or "") not in {"", run_id}:
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        entries.append(
            TraceCorrelationEntry(
                source="journal_replay",
                kind=str(event.get("event_type") or ""),
                run_id=str(event.get("run_id") or payload.get("run_id") or run_id),
                turn_id=str(event.get("turn_id") or payload.get("turn_id") or ""),
                sequence=int(event.get("sequence") or 0),
                call_id=_first_str(payload, ("call_id",)),
                approval_id=_approval_id_from_payload(payload),
                decision_id=_first_str(payload, ("decision_id",)),
                status=_first_str(payload, ("status",)),
                metadata={"payload_keys": sorted(str(key) for key in payload)},
            )
        )
    return tuple(entries)


def _correlate_provider(run_id: str, manifest: dict[str, Any]) -> tuple[TraceCorrelationEntry, ...]:
    entries = []
    for index, call in enumerate(_dict_items(manifest.get("calls")), start=1):
        metadata = call.get("metadata") if isinstance(call.get("metadata"), dict) else {}
        request = metadata.get("request") if isinstance(metadata.get("request"), dict) else {}
        request_metadata = request.get("metadata") if isinstance(request.get("metadata"), dict) else {}
        entries.append(
            TraceCorrelationEntry(
                source="provider",
                kind="llm_call",
                run_id=str(request_metadata.get("run_id") or run_id),
                turn_id=str(request_metadata.get("turn_id") or ""),
                sequence=index,
                status=str(call.get("status") or ""),
                metadata={
                    "provider_name": str(call.get("provider_name") or ""),
                    "model": str(call.get("model") or ""),
                    "attempt": int(call.get("attempt") or 0),
                    "streamed": bool(call.get("streamed")),
                    "retryable": bool(call.get("retryable")),
                },
            )
        )
    return tuple(entries)


def _correlate_tool_replay(run_id: str, manifest: dict[str, Any]) -> tuple[TraceCorrelationEntry, ...]:
    entries = []
    for record in _dict_items(manifest.get("records")):
        invocation = record.get("invocation") if isinstance(record.get("invocation"), dict) else {}
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        entries.append(
            TraceCorrelationEntry(
                source="tool_replay",
                kind="tool_result",
                run_id=str(metadata.get("run_id") or run_id),
                turn_id=str(metadata.get("turn_id") or ""),
                call_id=str(result.get("call_id") or invocation.get("call_id") or ""),
                status=str(result.get("status") or ""),
                subject=str(result.get("tool_name") or invocation.get("tool_name") or ""),
                metadata={
                    "replay_key": str(record.get("replay_key") or ""),
                    "tool_name": str(result.get("tool_name") or invocation.get("tool_name") or ""),
                },
            )
        )
    return tuple(entries)


def _correlate_policy_decisions(run_id: str, manifest: dict[str, Any]) -> tuple[TraceCorrelationEntry, ...]:
    entries = []
    for record in _dict_items(manifest.get("records")):
        decision = record.get("decision") if isinstance(record.get("decision"), dict) else {}
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        entries.append(
            TraceCorrelationEntry(
                source="policy_decisions",
                kind="policy_decision",
                run_id=str(record.get("run_id") or run_id),
                turn_id=str(record.get("turn_id") or ""),
                sequence=int(record.get("sequence") or 0),
                call_id=str(metadata.get("call_id") or ""),
                approval_id=_approval_id_from_payload(metadata),
                decision_id=str(record.get("decision_id") or ""),
                subject=str(record.get("subject") or ""),
                status=str(decision.get("status") or ""),
                metadata={
                    "subject_kind": str(record.get("subject_kind") or ""),
                    "subject_name": str(record.get("subject_name") or ""),
                    "approval_resumed": bool(metadata.get("approval_resumed")),
                },
            )
        )
    return tuple(entries)


def _correlate_approvals(run_id: str, manifest: dict[str, Any]) -> tuple[TraceCorrelationEntry, ...]:
    entries = []
    for record in _dict_items(manifest.get("records")):
        request = record.get("request") if isinstance(record.get("request"), dict) else {}
        decision = record.get("decision") if isinstance(record.get("decision"), dict) else {}
        entries.append(
            TraceCorrelationEntry(
                source="approvals",
                kind="approval_record",
                run_id=str(record.get("run_id") or run_id),
                turn_id=str(record.get("turn_id") or ""),
                approval_id=str(record.get("approval_id") or ""),
                subject=str(request.get("subject") or ""),
                status=str(record.get("status") or ""),
                metadata={
                    "reason": str(request.get("reason") or ""),
                    "actor": str(decision.get("actor") or ""),
                    "decision_status": str(decision.get("status") or ""),
                },
            )
        )
    return tuple(entries)


def _correlate_events(run_id: str, manifest: dict[str, Any]) -> tuple[TraceCorrelationEntry, ...]:
    entries = []
    for event in _dict_items(manifest.get("events")):
        if run_id and str(event.get("run_id") or "") not in {"", run_id}:
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        entries.append(
            TraceCorrelationEntry(
                source="event_log",
                kind=str(event.get("type") or ""),
                run_id=str(event.get("run_id") or run_id),
                turn_id=str(event.get("turn_id") or ""),
                sequence=int(event.get("sequence") or 0),
                call_id=_first_str(payload, ("call_id",)),
                approval_id=_approval_id_from_payload(payload),
                decision_id=_first_str(payload, ("decision_id",)),
                subject=_subject_from_payload(payload),
                status=_first_str(payload, ("status",)),
                metadata={"payload_keys": sorted(str(key) for key in payload)},
            )
        )
    return tuple(entries)


def _dict_items(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _first_str(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return str(value)
    return ""


def _approval_id_from_payload(payload: dict[str, Any]) -> str:
    direct = _first_str(payload, ("approval_id",))
    if direct:
        return direct
    approval_record = payload.get("approval_record")
    if isinstance(approval_record, dict):
        value = _first_str(approval_record, ("approval_id",))
        if value:
            return value
    grant = payload.get("grant")
    if isinstance(grant, dict):
        return _first_str(grant, ("approval_id",))
    approval = payload.get("approval")
    if isinstance(approval, dict):
        record = approval.get("approval_record")
        if isinstance(record, dict):
            return _first_str(record, ("approval_id",))
    return ""


def _subject_from_payload(payload: dict[str, Any]) -> str:
    direct = _first_str(payload, ("subject",))
    if direct:
        return direct
    request = payload.get("request")
    if isinstance(request, dict):
        return _first_str(request, ("subject",))
    grant = payload.get("grant")
    if isinstance(grant, dict):
        return _first_str(grant, ("subject",))
    approval = payload.get("approval")
    if isinstance(approval, dict):
        return _first_str(approval, ("subject",))
    return ""


def _correlation_sort_key(entry: TraceCorrelationEntry) -> tuple[str, str, int, str, str]:
    return (entry.run_id, entry.turn_id, entry.sequence, entry.source, entry.key)


def _join_id(first: str, second: str) -> str:
    if not first and not second:
        return ""
    return f"{first}:{second}"


def _group_entries(
    entries: tuple[TraceCorrelationEntry, ...],
    key_fn: Any,
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for entry in entries:
        key = str(key_fn(entry) or "")
        if not key:
            continue
        groups.setdefault(key, []).append(entry.key)
    return groups


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

