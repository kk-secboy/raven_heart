"""Durable AgentRunner session acceptance checks for SDK-local stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from agent_core.config import AgentProfile, CapabilitySet, RuntimeBudget
from agent_core.context import (
    AgentContextPack,
    ContextMaterial,
    ContextMaterialCenter,
    ContextMaterialQuery,
    ContextMaterialSelectionRequest,
    DefaultContextMaterialSelector,
    MarkdownContextMaterialStore,
    SQLiteContextMaterialStore,
)
from agent_core.events import MarkdownEventSink, SQLiteEventSink
from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.harness import MarkdownJournalStore, PersistentAgentJournal, SQLiteJournalStore
from agent_core.memory import (
    MarkdownMemoryStore,
    MemoryCenter,
    MemoryQuery,
    MemoryWrite,
    SQLiteMemoryStore,
)
from agent_core.policy import AllowAllPolicy, MarkdownPolicyDecisionStore, SQLitePolicyDecisionStore
from agent_core.providers import LLMProviderCenter, LLMRequest, LLMResponse
from agent_core.runner import (
    AgentRunQuery,
    AgentRunRequest,
    AgentSession,
    AgentSessionManager,
    MarkdownAgentRunStore,
    SQLiteAgentRunStore,
)
from agent_core.tools import (
    MarkdownToolReplayStore,
    PersistentToolReplay,
    SQLiteToolReplayStore,
    ToolCenter,
    ToolRegistry,
)
from agent_core.trace import MarkdownRunTraceStore, RunTraceQuery, SQLiteRunTraceStore


DurableSessionBackend = Literal["sqlite", "markdown"]


@dataclass(frozen=True)
class AgentCoreDurableSessionAcceptanceIssue:
    """One blocking issue from durable session acceptance."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-durable-session-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreDurableSessionAcceptanceReport:
    """Prompt-safe durable session acceptance report."""

    status: str
    sqlite_roundtrip: dict[str, Any] = field(default_factory=dict)
    markdown_roundtrip: dict[str, Any] = field(default_factory=dict)
    trace_eval: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreDurableSessionAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-durable-session-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "sqlite_roundtrip": dict(self.sqlite_roundtrip),
            "markdown_roundtrip": dict(self.markdown_roundtrip),
            "trace_eval": dict(self.trace_eval),
            "metadata": dict(self.metadata),
        }


class DurableSessionProvider:
    """Deterministic provider for durable session acceptance."""

    def __init__(self, *, backend: DurableSessionBackend) -> None:
        self.backend = backend
        self.requests: list[LLMRequest] = []
        self._responses: list[LLMResponse] = [
            LLMResponse(
                action={
                    "action": "call_tool",
                    "arguments": {
                        "tool_name": "lookup_durable_context",
                        "arguments": {"query": backend},
                    },
                },
                metadata={"phase": "tool_request", "backend": backend},
            ),
            LLMResponse(
                action={
                    "action": "finish",
                    "arguments": {"output": f"{backend}:durable-ok"},
                },
                metadata={"phase": "finish", "backend": backend},
            ),
        ]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(
            action={
                "action": "finish",
                "arguments": {"output": f"{self.backend}:durable-ok"},
            },
            metadata={"phase": "fallback_finish", "backend": self.backend},
        )


@dataclass(frozen=True)
class AgentCoreDurableSessionAcceptanceHarness:
    """Run an integrated durable AgentRunner session over SQLite and Markdown stores."""

    task: str = "inspect durable context and finish"
    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreDurableSessionAcceptanceReport:
        with TemporaryDirectory(
            prefix="agent-core-durable-session-",
            ignore_cleanup_errors=True,
        ) as raw_root:
            root = Path(raw_root)
            sqlite_roundtrip, sqlite_trace_eval = await _run_backend_roundtrip(
                "sqlite",
                root / "sqlite",
                task=self.task,
                metadata=dict(self.metadata),
            )
            markdown_roundtrip, markdown_trace_eval = await _run_backend_roundtrip(
                "markdown",
                root / "markdown",
                task=self.task,
                metadata=dict(self.metadata),
            )
        issues = _durable_session_issues(
            sqlite_roundtrip=sqlite_roundtrip,
            markdown_roundtrip=markdown_roundtrip,
            trace_evals=(sqlite_trace_eval, markdown_trace_eval),
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreDurableSessionAcceptanceReport(
            status=status,
            sqlite_roundtrip=sqlite_roundtrip,
            markdown_roundtrip=markdown_roundtrip,
            trace_eval={
                "schema_version": "agent-core-durable-session-trace-eval-summary/v1",
                "sqlite": sqlite_trace_eval,
                "markdown": markdown_trace_eval,
            },
            issues=issues,
            metadata={"scenario": "durable_session_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_durable_session_acceptance(
    *,
    task: str = "inspect durable context and finish",
    metadata: dict[str, Any] | None = None,
) -> AgentCoreDurableSessionAcceptanceReport:
    """Run the default integrated durable session acceptance scenario."""

    return await AgentCoreDurableSessionAcceptanceHarness(
        task=task,
        metadata=dict(metadata or {}),
    ).run()


async def _run_backend_roundtrip(
    backend: DurableSessionBackend,
    root: Path,
    *,
    task: str,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    paths = _backend_paths(root, backend)
    provider = DurableSessionProvider(backend=backend)
    session = await _durable_session(backend=backend, paths=paths, provider=provider)
    run_store = _run_store(backend, paths["run_state"])
    manager = AgentSessionManager(run_store=run_store)
    session_name = manager.register(session, name=f"durable-{backend}")
    request = AgentRunRequest(
        task=f"{task} using {backend}",
        context=AgentContextPack(workspace=f"{backend} durable workspace"),
        metadata={"acceptance": "durable_session", "backend": backend, **metadata},
        context_material_query=ContextMaterialQuery(
            query=f"{backend} durable context",
            limit=3,
            filters={"stores": (f"{backend}-context",)},
            tags=("durable", backend),
        ),
        context_material_selection=ContextMaterialSelectionRequest(
            task=f"{task} using {backend}",
            max_materials=2,
            max_bytes=1200,
        ),
    )
    outcome = await manager.run(session_name, request)
    trace_eval = DefaultTraceEvaluator().evaluate(
        outcome.trace_manifest,
        _durable_session_trace_spec(backend),
    ).manifest()
    reopened = await _reopen_backend(backend, paths, outcome.result.run_id)
    return (
        {
            "schema_version": "agent-core-durable-session-roundtrip/v1",
            "backend": backend,
            "run_id": outcome.result.run_id,
            "status": outcome.result.status,
            "output": outcome.result.output,
            "provider_request_count": len(provider.requests),
            "trace_summary": dict(outcome.trace_manifest.get("summary") or {}),
            "manager_run_count": len(run_store.list()),
            "reopened": reopened,
        },
        trace_eval,
    )


async def _durable_session(
    *,
    backend: DurableSessionBackend,
    paths: dict[str, Path],
    provider: DurableSessionProvider,
) -> AgentSession:
    provider_center = LLMProviderCenter(default_provider=f"durable-{backend}")
    provider_center.register(
        f"durable-{backend}",
        provider,
        default_model=f"durable-{backend}-mini",
    )
    tools = ToolCenter()
    registry = ToolRegistry()

    @registry.register_function(
        description="Lookup durable context through SDK tool center",
        tags=("durable", backend),
    )
    def lookup_durable_context(query: str) -> str:
        return f"{backend}:tool-result:{query}:context-found"

    tools.mount("local", registry, tags=("durable", backend))

    memory = MemoryCenter(default_store=f"{backend}-memory")
    memory_store = _memory_store(backend, paths["memory"])
    memory.register(
        f"{backend}-memory",
        memory_store,
        priority=10,
        backend_kind=backend,
        namespaces=("durable",),
        location=str(paths["memory"]),
        tags=("durable", backend),
    )
    await memory.write(
        MemoryWrite(
            content=f"{backend} durable memory says inspect persisted context before finish.",
            source=f"{backend}-memory-seed",
            metadata={"store": f"{backend}-memory", "kind": "seed"},
        )
    )

    context_store = ContextMaterialCenter(default_store=f"{backend}-context")
    context_store.register(
        f"{backend}-context",
        _context_store(backend, paths["context"]),
        priority=10,
        backend_kind=backend,
        location=str(paths["context"]),
        namespace="durable",
        tags=("durable", backend),
    )
    await context_store.write(
        ContextMaterial(
            name=f"{backend}-context-material",
            content=f"{backend} durable context material selected for the agent prompt.",
            role="dynamic",
            priority=20,
            metadata={
                "store": f"{backend}-context",
                "source": f"{backend}-context-seed",
                "namespace": "durable",
                "tags": (backend, "durable"),
            },
        )
    )

    return AgentSession(
        profile=AgentProfile(
            name=f"durable-{backend}",
            model=f"durable-{backend}-mini",
            instructions="Use durable memory and selected context before finishing.",
            capabilities=CapabilitySet(memory_enabled=True),
            budget=RuntimeBudget(max_iterations=4, max_cost_usd=0.1),
        ),
        provider=provider_center,
        tools=tools,
        harness=PersistentAgentJournal(_journal_store(backend, paths["journal"])),
        memory=memory,
        context_material_selector=DefaultContextMaterialSelector(),
        context_material_store=context_store,
        event_sink=_event_sink(backend, paths["events"]),
        policy=AllowAllPolicy(),
        policy_decision_store=_policy_store(backend, paths["policy"]),
        tool_replay=PersistentToolReplay(_tool_replay_store(backend, paths["tool_replay"])),
        trace_store=_trace_store(backend, paths["trace"]),
        metadata={"scenario": "durable_session_acceptance", "backend": backend},
    )


async def _reopen_backend(
    backend: DurableSessionBackend,
    paths: dict[str, Path],
    run_id: str,
) -> dict[str, Any]:
    journal = PersistentAgentJournal(_journal_store(backend, paths["journal"]))
    trace_store = _trace_store(backend, paths["trace"])
    event_sink = _event_sink(backend, paths["events"])
    tool_replay = PersistentToolReplay(_tool_replay_store(backend, paths["tool_replay"]))
    memory_store = _memory_store(backend, paths["memory"])
    context_store = _context_store(backend, paths["context"])
    policy_store = _policy_store(backend, paths["policy"])
    run_store = _run_store(backend, paths["run_state"])
    loaded_trace = await trace_store.load(run_id)
    trace_records = await trace_store.query(RunTraceQuery(run_ids=(run_id,)))
    memory_hits = await memory_store.search(MemoryQuery(query=f"{backend} durable", limit=5))
    context_materials = await context_store.search(
        ContextMaterialQuery(query=f"{backend} durable context", limit=5)
    )
    tool_records = await tool_replay.records()
    policy_records = await policy_store.records()
    run_records = run_store.query(AgentRunQuery(session_names=(f"durable-{backend}",)))
    return {
        "schema_version": "agent-core-durable-session-reopened/v1",
        "journal_run_count": len(journal.snapshot().runs),
        "journal_checkpoint_count": len(journal.snapshot().checkpoints),
        "journal_finished_count": len(journal.snapshot().finished),
        "trace_count": len(trace_records),
        "event_count": len(event_sink.records(run_id=run_id)),
        "tool_replay_count": len(tool_records),
        "memory_hit_count": len(memory_hits),
        "context_material_count": len(context_materials),
        "policy_decision_count": len(policy_records),
        "manager_run_count": len(run_records),
        "manager_completed_count": sum(1 for run in run_records if run.status == "completed"),
        "loaded_trace_loaded": loaded_trace is not None,
    }


def _durable_session_trace_spec(backend: DurableSessionBackend) -> TraceEvalSpec:
    return TraceEvalSpec(
        name=f"agent-core-durable-session-{backend}",
        expected_status="completed",
        max_iterations=4,
        max_provider_calls=3,
        required_provider_names=(f"durable-{backend}",),
        required_provider_models=(f"durable-{backend}-mini",),
        require_tool_center=True,
        required_tool_center_selected_mounts=("local",),
        required_tool_center_selected_tools=("lookup_durable_context",),
        required_tool_center_requested_tools=("lookup_durable_context",),
        require_tool_center_ready_routes=True,
        max_tool_center_failed_calls=0,
        require_event_log=False,
        require_terminal_event=False,
        require_event_sequence_monotonic=False,
        require_storage_backends=True,
        required_storage_backend_roles=(
            "memory",
            "journal",
            "tool_replay",
            "run_trace",
            "context_material",
            "policy_decision",
            "event_log",
        ),
        required_storage_backend_kinds=(backend,),
        require_context_injections=True,
        required_context_injection_names=("memory_recall", f"{backend}-context-material"),
        required_context_injection_sources=("memory", f"{backend}-context-seed"),
        require_context_material_selection=True,
        required_selected_context_material_names=(f"{backend}-context-material",),
        required_context_material_statuses=("selected",),
        max_dropped_context_materials=0,
    )


def _durable_session_issues(
    *,
    sqlite_roundtrip: dict[str, Any],
    markdown_roundtrip: dict[str, Any],
    trace_evals: tuple[dict[str, Any], dict[str, Any]],
) -> tuple[AgentCoreDurableSessionAcceptanceIssue, ...]:
    issues: list[AgentCoreDurableSessionAcceptanceIssue] = []
    for backend, report in (("sqlite", sqlite_roundtrip), ("markdown", markdown_roundtrip)):
        reopened = dict(report.get("reopened") or {})
        for key in (
            "journal_run_count",
            "journal_checkpoint_count",
            "journal_finished_count",
            "trace_count",
            "event_count",
            "tool_replay_count",
            "memory_hit_count",
            "context_material_count",
            "policy_decision_count",
            "manager_run_count",
            "manager_completed_count",
        ):
            if int(reopened.get(key) or 0) <= 0:
                issues.append(
                    AgentCoreDurableSessionAcceptanceIssue(
                        source=backend,
                        code=f"{key}_missing",
                        message=f"{backend} durable roundtrip did not reopen {key}.",
                        metadata={"reopened": reopened},
                    )
                )
        if not bool(reopened.get("loaded_trace_loaded")):
            issues.append(
                AgentCoreDurableSessionAcceptanceIssue(
                    source=backend,
                    code="loaded_trace_missing",
                    message=f"{backend} durable roundtrip did not load the stored trace.",
                    metadata={"reopened": reopened},
                )
            )
        if report.get("status") != "completed" or not str(report.get("output") or "").endswith(
            "durable-ok"
        ):
            issues.append(
                AgentCoreDurableSessionAcceptanceIssue(
                    source=backend,
                    code="run_not_completed",
                    message=f"{backend} durable run did not complete with expected output.",
                    metadata={"status": report.get("status"), "output": report.get("output")},
                )
            )
    for trace_eval in trace_evals:
        if not bool(trace_eval.get("ok")):
            issues.append(
                AgentCoreDurableSessionAcceptanceIssue(
                    source=str(trace_eval.get("name") or "trace_eval"),
                    code="trace_eval_failed",
                    message="Durable session trace eval failed.",
                    metadata={"trace_eval": trace_eval},
                )
            )
    return tuple(issues)


def _backend_paths(root: Path, backend: DurableSessionBackend) -> dict[str, Path]:
    if backend == "sqlite":
        return {
            "journal": root / "journal.sqlite",
            "trace": root / "trace.sqlite",
            "events": root / "events.sqlite",
            "tool_replay": root / "tool_replay.sqlite",
            "memory": root / "memory.sqlite",
            "context": root / "context.sqlite",
            "policy": root / "policy.sqlite",
            "run_state": root / "run_state.sqlite",
        }
    return {
        "journal": root / "journal.md",
        "trace": root / "trace.md",
        "events": root / "events.md",
        "tool_replay": root / "tool_replay.md",
        "memory": root / "memory",
        "context": root / "context.md",
        "policy": root / "policy.md",
        "run_state": root / "run_state.md",
    }


def _journal_store(backend: DurableSessionBackend, path: Path) -> Any:
    return SQLiteJournalStore(path) if backend == "sqlite" else MarkdownJournalStore(path)


def _trace_store(backend: DurableSessionBackend, path: Path) -> Any:
    return SQLiteRunTraceStore(path) if backend == "sqlite" else MarkdownRunTraceStore(path)


def _event_sink(backend: DurableSessionBackend, path: Path) -> Any:
    return SQLiteEventSink(path) if backend == "sqlite" else MarkdownEventSink(path)


def _tool_replay_store(backend: DurableSessionBackend, path: Path) -> Any:
    return SQLiteToolReplayStore(path) if backend == "sqlite" else MarkdownToolReplayStore(path)


def _memory_store(backend: DurableSessionBackend, path: Path) -> Any:
    return SQLiteMemoryStore(path) if backend == "sqlite" else MarkdownMemoryStore(path)


def _context_store(backend: DurableSessionBackend, path: Path) -> Any:
    return (
        SQLiteContextMaterialStore(path)
        if backend == "sqlite"
        else MarkdownContextMaterialStore(path)
    )


def _policy_store(backend: DurableSessionBackend, path: Path) -> Any:
    return (
        SQLitePolicyDecisionStore(path)
        if backend == "sqlite"
        else MarkdownPolicyDecisionStore(path)
    )


def _run_store(backend: DurableSessionBackend, path: Path) -> Any:
    return SQLiteAgentRunStore(path) if backend == "sqlite" else MarkdownAgentRunStore(path)

