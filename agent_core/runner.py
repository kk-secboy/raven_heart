"""Composable runner entrypoint for the provider-neutral agent core."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from agent_core.actions import ActionRegistry, ActionVerifierPort
from agent_core.approvals import ApprovalResumeContext, ApprovalStorePort, NullApprovalStore
from agent_core.artifacts import ArtifactStorePort
from agent_core.backends import storage_backend_catalog_from_components, storage_backend_manifest
from agent_core.capabilities import CapabilityCatalog, CapabilityQuery, CapabilityRecallRequest
from agent_core.config import AgentProfile, RuntimeBudget
from agent_core.context import (
    AgentContextPack,
    AgentPromptBuilder,
    ContextInjection,
    ContextInjectionPolicy,
    ContextMaterial,
    ContextMaterialQuery,
    ContextMaterialSelectionRequest,
    ContextMaterialSelectionResult,
    ContextMaterialSelectorPort,
    ContextMaterialStorePort,
    DefaultContextMaterialSelector,
)
from agent_core.events import (
    AgentEvent,
    EventSinkPort,
    EventStreamBatch,
    EventStreamCursor,
    EventStreamTail,
)
from agent_core.errors import ResumeError
from agent_core.harness import (
    AgentHarness,
    CancelToken,
    InMemoryAgentJournal,
    ResumeCandidate,
    ResumeIndex,
    ResumePlan,
    ResumeToken,
)
from agent_core.lifecycle import AgentLifecycleEvent, AgentLifecycleHookCenter, NullLifecycleHooks
from agent_core.loop_guard import LoopGuard
from agent_core.knowledge import (
    DefaultKnowledgeRecall,
    DefaultMidtermTimelineRecall,
    KnowledgeRecallPort,
    KnowledgeRecallRequest,
    MidtermTimelineRecallPort,
    MidtermTimelineRecallRequest,
)
from agent_core.memory import (
    MemoryFlushBuffer,
    MemoryFlushSignal,
    MemoryHit,
    MemoryPort,
    MemoryQuery,
    MemoryWrite,
    NullMemory,
    build_memory_injection,
    infer_memory_recall_intent,
)
from agent_core.mcp import MCPCenter, MCPContextMaterialRequest
from agent_core.policy import NullPolicyDecisionStore, PolicyDecisionStorePort, PolicyPort
from agent_core.preflight import (
    AgentRunPreflightCenter,
    AgentRunPreflightReport,
    AgentRunPreflightRequest,
    AgentRunPreflightRequirements,
    default_agent_run_preflight_center,
)
from agent_core.providers import LLMRequest, LLMProviderPort
from agent_core.prompt import (
    PromptBucketBudgetPolicy,
    PromptBucketRole,
    PromptIR,
    PromptSemanticReducerPort,
    PromptSemanticTrimRequest,
)
from agent_core.react import ReActConfig, ReActExecutor, ReActResult
from agent_core.reducer import ContextReducerPort, ReducerRequest, apply_reduction_to_timeline
from agent_core.skills import SkillsContext
from agent_core.structured import StructuredOutputSpec, StructuredOutputValidatorPort
from agent_core.task_contract import AgentTaskContract
from agent_core.timeline import TimelineBudget, TimelineStore, TimelineStorePort
from agent_core.tools import NullToolReplay, ToolReplayPort, ToolRuntimePort
from agent_core.trace import AgentJournalReplay, AgentRunTraceBundle, NullRunTraceStore, RunTraceStorePort
from agent_core.turn_runtime import (
    CapabilityRefreshPort,
    DeterministicPerceptionEvaluator,
    LoopStateFrame,
    NullCapabilityRefresh,
    PerceptionDownstreamScheduler,
    PerceptionControllerPort,
    ToolResultEvent,
    TurnCompletedEvent,
    TurnContextRefresherPort,
    TurnRefreshRequest,
    TurnRefreshResult,
    YaklangStylePerceptionController,
)


@dataclass
class AgentSession:
    """All core-level components needed to run one agent profile."""

    profile: AgentProfile
    provider: LLMProviderPort
    tools: ToolRuntimePort
    actions: ActionRegistry = field(default_factory=ActionRegistry)
    harness: AgentHarness = field(default_factory=InMemoryAgentJournal)
    skills: SkillsContext | None = None
    mcp: MCPCenter | None = None
    memory: MemoryPort = field(default_factory=NullMemory)
    timeline: TimelineStorePort = field(default_factory=TimelineStore)
    context_reducer: ContextReducerPort | None = None
    context_material_selector: ContextMaterialSelectorPort | None = None
    context_material_store: ContextMaterialStorePort | None = None
    context_injection_policy: ContextInjectionPolicy = field(default_factory=ContextInjectionPolicy)
    prompt_bucket_budget_policy: PromptBucketBudgetPolicy | None = None
    prompt_semantic_reducer: PromptSemanticReducerPort | None = None
    event_sink: EventSinkPort | None = None
    policy: PolicyPort | None = None
    policy_decision_store: PolicyDecisionStorePort = field(default_factory=NullPolicyDecisionStore)
    approval_store: ApprovalStorePort = field(default_factory=NullApprovalStore)
    tool_replay: ToolReplayPort = field(default_factory=NullToolReplay)
    trace_store: RunTraceStorePort = field(default_factory=NullRunTraceStore)
    lifecycle_hooks: AgentLifecycleHookCenter = field(default_factory=NullLifecycleHooks)
    preflight: AgentRunPreflightCenter = field(default_factory=default_agent_run_preflight_center)
    action_verifier: ActionVerifierPort | None = None
    structured_output_validator: StructuredOutputValidatorPort | None = None
    loop_guard: LoopGuard | None = None
    artifact_store: ArtifactStorePort | None = None
    turn_refresher: TurnContextRefresherPort | None = None
    perception_controller: PerceptionControllerPort | None = None
    capability_refresher: CapabilityRefreshPort | None = None
    knowledge_recall: KnowledgeRecallPort | None = None
    midterm_timeline_recall: MidtermTimelineRecallPort | None = None
    cancel_token: CancelToken = field(default_factory=CancelToken)
    native_tool_calls: bool = False
    stream: bool = False
    provider_cache_mode: str = "strip"
    provider_cache_min_segment_bytes: int = 1024
    metadata: dict[str, Any] = field(default_factory=dict)

    def reset_cancel_token(self) -> None:
        self.cancel_token = CancelToken()

    def capability_catalog(self) -> CapabilityCatalog:
        return CapabilityCatalog(
            actions=self.actions,
            tools=self.tools,
            skills=self.skills,
            mcp=self.mcp,
            metadata={"profile": self.profile.name, **self.metadata},
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-session/v1",
            "profile": {
                "name": self.profile.name,
                "model": self.profile.model,
                "instructions": self.profile.instructions,
                "budget": _budget_manifest(self.profile.budget),
                "capabilities": {
                    "actions": list(self.profile.capabilities.actions),
                    "tools": list(self.profile.capabilities.tools),
                    "skills": list(self.profile.capabilities.skills),
                    "memory_enabled": self.profile.capabilities.memory_enabled,
                    "human_in_loop_enabled": self.profile.capabilities.human_in_loop_enabled,
                },
            },
            "provider": _component_manifest_sync(self.provider),
            "tools": _component_manifest_sync(self.tools),
            "harness": _component_manifest_sync(self.harness),
            "capabilities": self.capability_catalog().manifest(),
            "memory": _component_manifest_sync(self.memory),
            "timeline": _component_manifest_sync(self.timeline),
            "tool_replay": _component_manifest_sync(self.tool_replay),
            "policy_decisions": _component_manifest_sync(self.policy_decision_store),
            "approvals": _component_manifest_sync(self.approval_store),
            "event_log": _component_manifest_sync(self.event_sink),
            "trace_store": _component_manifest_sync(self.trace_store),
            "lifecycle_hooks": self.lifecycle_hooks.manifest(),
            "preflight": self.preflight.manifest(),
            "artifact_store": _component_manifest_sync(self.artifact_store),
            "turn_refresher": _component_manifest_sync(self.turn_refresher)
            or {"enabled": self.turn_refresher is not None},
            "perception_controller": _component_manifest_sync(self.perception_controller)
            or {"enabled": self.perception_controller is not None},
            "capability_refresher": _component_manifest_sync(self.capability_refresher)
            or {"enabled": self.capability_refresher is not None},
            "knowledge_recall": _component_manifest_sync(self.knowledge_recall)
            or {"enabled": self.knowledge_recall is not None},
            "midterm_timeline_recall": _component_manifest_sync(self.midterm_timeline_recall)
            or {"enabled": self.midterm_timeline_recall is not None},
            "native_tool_calls": self.native_tool_calls,
            "stream": self.stream,
            "provider_cache": {
                "mode": self.provider_cache_mode,
                "min_segment_bytes": self.provider_cache_min_segment_bytes,
            },
            "context_reducer": _context_reducer_manifest(self.context_reducer),
            "context_material_selector": _context_material_selector_manifest(
                self.context_material_selector
            ),
            "context_material_store": _component_manifest_sync(self.context_material_store),
            "context_injection_policy": self.context_injection_policy.manifest(),
            "prompt_bucket_budget_policy": _prompt_bucket_budget_policy_manifest(
                self.prompt_bucket_budget_policy
            ),
            "prompt_semantic_reducer": _prompt_semantic_reducer_manifest(
                self.prompt_semantic_reducer
            ),
            "timeline_items": len(self.timeline.items),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentRunRequest:
    task: str
    context: AgentContextPack | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    refresh: bool = False
    resume_token: ResumeToken | None = None
    approval_resume: ApprovalResumeContext | None = None
    structured_output: StructuredOutputSpec | None = None
    context_materials: tuple[ContextMaterial, ...] = ()
    context_material_query: ContextMaterialQuery | None = None
    context_material_selection: ContextMaterialSelectionRequest | None = None
    mcp_context_materials: MCPContextMaterialRequest | None = None
    preflight_requirements: AgentRunPreflightRequirements | None = None
    task_contract: AgentTaskContract | None = None
    timeout_seconds: float | None = None
    native_tool_calls: bool | None = None
    stream: bool | None = None


@dataclass(frozen=True)
class AgentResumeRequest:
    task: str = ""
    run_id: str = ""
    include_terminal: bool = True
    allow_terminal: bool = True
    context: AgentContextPack | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    refresh: bool = False
    approval_resume: ApprovalResumeContext | None = None
    structured_output: StructuredOutputSpec | None = None
    context_materials: tuple[ContextMaterial, ...] = ()
    context_material_query: ContextMaterialQuery | None = None
    context_material_selection: ContextMaterialSelectionRequest | None = None
    mcp_context_materials: MCPContextMaterialRequest | None = None
    preflight_requirements: AgentRunPreflightRequirements | None = None
    task_contract: AgentTaskContract | None = None
    timeout_seconds: float | None = None
    native_tool_calls: bool | None = None
    stream: bool | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-resume-request/v1",
            "task": self.task,
            "run_id": self.run_id,
            "include_terminal": self.include_terminal,
            "allow_terminal": self.allow_terminal,
            "metadata": dict(self.metadata),
            "refresh": self.refresh,
            "has_context": self.context is not None,
            "has_approval_resume": self.approval_resume is not None,
            "has_structured_output": self.structured_output is not None,
            "context_material_count": len(self.context_materials),
            "has_context_material_query": self.context_material_query is not None,
            "has_context_material_selection": self.context_material_selection is not None,
            "has_mcp_context_materials": self.mcp_context_materials is not None,
            "has_preflight_requirements": self.preflight_requirements is not None,
            "has_task_contract": self.task_contract is not None,
            "timeout_seconds": self.timeout_seconds,
            "native_tool_calls": self.native_tool_calls,
            "stream": self.stream,
        }


@dataclass(frozen=True)
class AgentRunOutcome:
    result: ReActResult
    session_manifest: dict[str, Any]
    prompt_manifest: dict[str, Any]
    resume_manifest: dict[str, Any] = field(default_factory=dict)
    resume_plan_manifest: dict[str, Any] = field(default_factory=dict)
    timeline_reduction_manifest: dict[str, Any] = field(default_factory=dict)
    capability_discovery_manifest: dict[str, Any] = field(default_factory=dict)
    memory_search_manifest: dict[str, Any] = field(default_factory=dict)
    knowledge_recall_manifest: dict[str, Any] = field(default_factory=dict)
    midterm_timeline_recall_manifest: dict[str, Any] = field(default_factory=dict)
    downstream_plan_manifest: dict[str, Any] = field(default_factory=dict)
    loop_state_manifest: dict[str, Any] = field(default_factory=dict)
    preflight_manifest: dict[str, Any] = field(default_factory=dict)
    trace_manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentPromptBudgetPlan:
    """Effective prompt budget after profile and provider capability checks."""

    profile_max_prompt_bytes: int
    target_prompt_bytes: int
    provider_name: str = ""
    model: str = ""
    context_window_tokens: int = 0
    reserved_output_tokens: int = 0
    provider_input_budget_bytes: int = 0
    source: str = "profile_budget"
    route_plan: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def provider_limited(self) -> bool:
        return bool(
            self.provider_input_budget_bytes
            and self.target_prompt_bytes < self.profile_max_prompt_bytes
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-prompt-budget-plan/v1",
            "profile_max_prompt_bytes": self.profile_max_prompt_bytes,
            "target_prompt_bytes": self.target_prompt_bytes,
            "provider_limited": self.provider_limited,
            "provider_name": self.provider_name,
            "model": self.model,
            "context_window_tokens": self.context_window_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "provider_input_budget_bytes": self.provider_input_budget_bytes,
            "source": self.source,
            "route_plan": dict(self.route_plan),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentMemoryRecall:
    injections: tuple[ContextInjection, ...] = ()
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentContextMaterialSelection:
    injections: tuple[ContextInjection, ...] = ()
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManagedAgentRun:
    run_key: str
    session_name: str
    task: str
    status: str = "queued"
    result_run_id: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-managed-run/v1",
            "run_key": self.run_key,
            "session_name": self.session_name,
            "task": self.task,
            "status": self.status,
            "result_run_id": self.result_run_id,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentRunQuery:
    """Portable manager-run query for SDK and runtime-owned run stores."""

    run_keys: tuple[str, ...] = ()
    session_names: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    reverse: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_keys", tuple(str(item) for item in self.run_keys if str(item)))
        object.__setattr__(
            self,
            "session_names",
            tuple(str(item) for item in self.session_names if str(item)),
        )
        object.__setattr__(self, "statuses", tuple(str(item) for item in self.statuses if str(item)))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.limit is not None:
            object.__setattr__(self, "limit", max(0, int(self.limit)))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-run-query/v1",
            "run_keys": list(self.run_keys),
            "session_names": list(self.session_names),
            "statuses": list(self.statuses),
            "metadata": dict(self.metadata),
            "limit": self.limit,
            "reverse": self.reverse,
        }


@dataclass(frozen=True)
class AgentManagerConcurrencyPolicy:
    max_active_runs: int | None = None
    max_active_runs_per_session: int = 1
    reject_when_full: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_active_runs is not None:
            object.__setattr__(self, "max_active_runs", max(0, int(self.max_active_runs)))
        object.__setattr__(
            self,
            "max_active_runs_per_session",
            max(1, int(self.max_active_runs_per_session)),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-manager-concurrency-policy/v1",
            "max_active_runs": self.max_active_runs,
            "max_active_runs_per_session": self.max_active_runs_per_session,
            "reject_when_full": self.reject_when_full,
            "metadata": dict(self.metadata),
        }


class AgentManagerCapacityError(RuntimeError):
    def __init__(self, message: str, *, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata = dict(metadata or {})


@dataclass(frozen=True)
class AgentManagerCapacityStatus:
    session_name: str
    active_run_keys: tuple[str, ...] = ()
    manager_active_run_count: int = 0
    max_active_runs: int | None = None
    max_active_runs_per_session: int = 1
    session_available: bool = True
    manager_available: bool = True
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def active_run_count(self) -> int:
        return len(self.active_run_keys)

    @property
    def available(self) -> bool:
        return self.session_available and self.manager_available

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-manager-capacity-status/v1",
            "session_name": self.session_name,
            "available": self.available,
            "session_available": self.session_available,
            "manager_available": self.manager_available,
            "reason": self.reason,
            "active_run_count": self.active_run_count,
            "active_run_keys": list(self.active_run_keys),
            "manager_active_run_count": self.manager_active_run_count,
            "max_active_runs": self.max_active_runs,
            "max_active_runs_per_session": self.max_active_runs_per_session,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentManagerScheduleSnapshot:
    sessions: tuple[str, ...] = ()
    runs: tuple[ManagedAgentRun, ...] = ()
    active_by_session: dict[str, tuple[str, ...]] = field(default_factory=dict)
    queued_by_session: dict[str, tuple[str, ...]] = field(default_factory=dict)
    capacity_by_session: tuple[AgentManagerCapacityStatus, ...] = ()
    concurrency_policy: AgentManagerConcurrencyPolicy = field(default_factory=AgentManagerConcurrencyPolicy)
    run_store: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def status_counts(self) -> dict[str, int]:
        return dict(Counter(run.status for run in self.runs))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-manager-schedule-snapshot/v1",
            "sessions": list(self.sessions),
            "run_count": len(self.runs),
            "runs": [run.manifest() for run in self.runs],
            "status_counts": self.status_counts(),
            "active_by_session": {
                name: list(keys)
                for name, keys in sorted(self.active_by_session.items(), key=lambda item: item[0])
            },
            "active_run_count": sum(len(keys) for keys in self.active_by_session.values()),
            "queued_by_session": {
                name: list(keys)
                for name, keys in sorted(self.queued_by_session.items(), key=lambda item: item[0])
            },
            "queued_run_count": sum(len(keys) for keys in self.queued_by_session.values()),
            "capacity_by_session": [
                status.manifest() for status in self.capacity_by_session
            ],
            "concurrency_policy": self.concurrency_policy.manifest(),
            "run_store": dict(self.run_store),
            "metadata": dict(self.metadata),
        }


class AgentRunStorePort(Protocol):
    """Persistence boundary for manager-level run state."""

    def save(self, run: ManagedAgentRun) -> None:
        """Persist or replace one managed run."""

    def get(self, run_key: str) -> ManagedAgentRun | None:
        """Return one run by key if present."""

    def list(self) -> tuple[ManagedAgentRun, ...]:
        """Return all known runs."""

    def query(self, query: AgentRunQuery) -> tuple[ManagedAgentRun, ...]:
        """Return runs matching a portable query."""

    def delete(self, run_key: str) -> bool:
        """Delete one run state record."""

    def manifest(self) -> dict[str, Any]:
        """Return prompt-safe store metadata."""


class InMemoryAgentRunStore:
    def __init__(self, runs: tuple[ManagedAgentRun, ...] = ()) -> None:
        self._runs = {run.run_key: run for run in runs}

    def save(self, run: ManagedAgentRun) -> None:
        self._runs[run.run_key] = run

    def get(self, run_key: str) -> ManagedAgentRun | None:
        return self._runs.get(run_key)

    def list(self) -> tuple[ManagedAgentRun, ...]:
        return tuple(sorted(self._runs.values(), key=lambda item: item.run_key))

    def query(self, query: AgentRunQuery) -> tuple[ManagedAgentRun, ...]:
        return _query_managed_runs(self.list(), query)

    def delete(self, run_key: str) -> bool:
        return self._runs.pop(run_key, None) is not None

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-in-memory-run-store/v1",
            "backend": storage_backend_manifest(
                role="run_state",
                kind="in_memory",
                capabilities=("save", "get", "list", "query", "delete"),
            ),
            "run_count": len(self._runs),
        }


class SQLiteAgentRunStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS managed_runs (
                    run_key TEXT PRIMARY KEY,
                    session_name TEXT NOT NULL,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_run_id TEXT NOT NULL,
                    error TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )

    def save(self, run: ManagedAgentRun) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO managed_runs(
                    run_key, session_name, task, status, result_run_id, error, metadata_json
                )
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_key) DO UPDATE SET
                    session_name=excluded.session_name,
                    task=excluded.task,
                    status=excluded.status,
                    result_run_id=excluded.result_run_id,
                    error=excluded.error,
                    metadata_json=excluded.metadata_json
                """,
                (
                    run.run_key,
                    run.session_name,
                    run.task,
                    run.status,
                    run.result_run_id,
                    run.error,
                    json.dumps(run.metadata, ensure_ascii=False, sort_keys=True),
                ),
            )

    def get(self, run_key: str) -> ManagedAgentRun | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                """
                SELECT run_key, session_name, task, status, result_run_id, error, metadata_json
                FROM managed_runs
                WHERE run_key = ?
                """,
                (run_key,),
            ).fetchone()
        return _managed_run_from_row(row) if row else None

    def list(self) -> tuple[ManagedAgentRun, ...]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """
                SELECT run_key, session_name, task, status, result_run_id, error, metadata_json
                FROM managed_runs
                ORDER BY run_key
                """
            ).fetchall()
        return tuple(_managed_run_from_row(row) for row in rows)

    def query(self, query: AgentRunQuery) -> tuple[ManagedAgentRun, ...]:
        return _query_managed_runs(self.list(), query)

    def delete(self, run_key: str) -> bool:
        with sqlite3.connect(self.path) as conn:
            cursor = conn.execute("DELETE FROM managed_runs WHERE run_key = ?", (run_key,))
            return cursor.rowcount > 0

    def manifest(self) -> dict[str, Any]:
        with sqlite3.connect(self.path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM managed_runs").fetchone()[0]
        return {
            "schema_version": "agent-core-sqlite-run-store/v1",
            "backend": storage_backend_manifest(
                role="run_state",
                kind="sqlite",
                location=str(self.path),
                capabilities=("save", "get", "list", "query", "delete"),
            ),
            "path": str(self.path),
            "run_count": int(count),
        }


class MarkdownAgentRunStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("# Agent Run Store\n\n", encoding="utf-8")

    def save(self, run: ManagedAgentRun) -> None:
        runs = {item.run_key: item for item in self.list()}
        runs[run.run_key] = run
        self._write(tuple(sorted(runs.values(), key=lambda item: item.run_key)))

    def get(self, run_key: str) -> ManagedAgentRun | None:
        for run in self.list():
            if run.run_key == run_key:
                return run
        return None

    def list(self) -> tuple[ManagedAgentRun, ...]:
        text = self.path.read_text(encoding="utf-8")
        runs = []
        for match in _RUN_MARKDOWN_RE.finditer(text):
            runs.append(_managed_run_from_payload(_decode_run_payload(match.group("payload"))))
        return tuple(sorted(runs, key=lambda item: item.run_key))

    def query(self, query: AgentRunQuery) -> tuple[ManagedAgentRun, ...]:
        return _query_managed_runs(self.list(), query)

    def delete(self, run_key: str) -> bool:
        current = self.list()
        runs = tuple(run for run in current if run.run_key != run_key)
        existed = len(runs) != len(current)
        if existed:
            self._write(runs)
        return existed

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-markdown-run-store/v1",
            "backend": storage_backend_manifest(
                role="run_state",
                kind="markdown",
                location=str(self.path),
                capabilities=("save", "get", "list", "query", "delete"),
            ),
            "path": str(self.path),
            "run_count": len(self.list()),
        }

    def _write(self, runs: tuple[ManagedAgentRun, ...]) -> None:
        lines = ["# Agent Run Store", ""]
        for run in runs:
            lines.extend(
                [
                    f"## Run {run.run_key}",
                    "",
                    f"- session: {run.session_name}",
                    f"- status: {run.status}",
                    f"- result_run_id: {run.result_run_id or '-'}",
                    f"<!-- agent-core-run {_encode_run_payload(run.manifest())} -->",
                    "",
                ]
            )
        self.path.write_text("\n".join(lines), encoding="utf-8")


class AgentRunner:
    """SDK-like runner that wires core centers into ReAct execution."""

    def __init__(self, session: AgentSession) -> None:
        self.session = session

    async def refresh(self) -> None:
        refresh_tools = getattr(self.session.tools, "refresh", None)
        if callable(refresh_tools):
            await refresh_tools()
        if self.session.mcp is not None:
            await self.session.mcp.refresh(fail_fast=False)
            try:
                await self.session.mcp.refresh_resources()
            except Exception as exc:
                self.session.metadata["mcp_resource_refresh_error"] = str(exc)
            try:
                await self.session.mcp.refresh_prompts()
            except Exception as exc:
                self.session.metadata["mcp_prompt_refresh_error"] = str(exc)

    async def run(self, request: AgentRunRequest | str) -> AgentRunOutcome:
        run_request = request if isinstance(request, AgentRunRequest) else AgentRunRequest(task=str(request))
        await self._emit_lifecycle_event(
            AgentLifecycleEvent(
                type="run_starting",
                session_name=self.session.profile.name,
                task=run_request.task,
                metadata={"request_metadata": dict(run_request.metadata)},
            )
        )
        try:
            if run_request.refresh:
                await self.refresh()
            preflight = await self._preflight(run_request)
            if not preflight.ok:
                return await self._preflight_blocked_outcome(run_request, preflight)
            resume_manifest = await self._resume_manifest(run_request.resume_token)
            bootstrap_prompt = PromptIR.from_parts(
                dynamic=run_request.task,
                metadata={
                    "schema_version": "agent-core-bootstrap-prompt/v1",
                    "profile": self.session.profile.name,
                    "request_metadata": dict(run_request.metadata),
                    "resume": resume_manifest,
                },
            )
            turn_refresher = self.session.turn_refresher or _RunnerTurnContextRefresher(
                self,
                run_request=run_request,
                resume_manifest=resume_manifest,
            )
            executor = self._executor(
                run_request.approval_resume,
                run_request.structured_output,
                run_request.timeout_seconds,
                run_request.native_tool_calls,
                run_request.stream,
                turn_refresher=turn_refresher,
            )
            result = await executor.run(run_request.task, bootstrap_prompt)
            prompt_manifest = dict(getattr(executor, "last_prompt_manifest", {}) or {})
            capability_discovery_manifest = dict(
                getattr(turn_refresher, "last_capability_manifest", {}) or {}
            )
            memory_search_manifest = dict(getattr(turn_refresher, "last_memory_manifest", {}) or {})
            knowledge_recall_manifest = dict(
                getattr(turn_refresher, "last_knowledge_manifest", {}) or {}
            )
            midterm_timeline_recall_manifest = dict(
                getattr(turn_refresher, "last_midterm_timeline_manifest", {}) or {}
            )
            downstream_plan_manifest = dict(
                getattr(turn_refresher, "last_downstream_plan_manifest", {}) or {}
            )
            loop_state_manifest = dict(getattr(turn_refresher, "last_loop_state_manifest", {}) or {})
            timeline_reduction_manifest = dict(
                getattr(turn_refresher, "last_timeline_reduction_manifest", {}) or {}
            )
            await self._emit_lifecycle_event(
                AgentLifecycleEvent(
                    type="run_completed",
                    session_name=self.session.profile.name,
                    run_id=result.run_id,
                    task=run_request.task,
                    status=result.status,
                    metadata={
                        "iterations": result.iterations,
                        "output_bytes": len(result.output.encode("utf-8")),
                    },
                )
            )
            session_manifest = self.session.manifest()
            trace_manifest = await self._trace_manifest(
                result,
                session_manifest=session_manifest,
                prompt_manifest=prompt_manifest,
                resume_manifest=resume_manifest,
                resume_plan_manifest=_request_resume_plan_manifest(run_request),
                timeline_reduction_manifest=timeline_reduction_manifest,
                capability_discovery_manifest=capability_discovery_manifest,
                memory_search_manifest=memory_search_manifest,
                preflight_manifest=preflight.manifest(),
                request=run_request,
            )
            await self.session.trace_store.save(trace_manifest)
            return AgentRunOutcome(
                result=result,
                session_manifest=session_manifest,
                prompt_manifest=prompt_manifest,
                resume_manifest=resume_manifest,
                resume_plan_manifest=_request_resume_plan_manifest(run_request),
                timeline_reduction_manifest=timeline_reduction_manifest,
                capability_discovery_manifest=capability_discovery_manifest,
                memory_search_manifest=memory_search_manifest,
                knowledge_recall_manifest=knowledge_recall_manifest,
                midterm_timeline_recall_manifest=midterm_timeline_recall_manifest,
                downstream_plan_manifest=downstream_plan_manifest,
                loop_state_manifest=loop_state_manifest,
                preflight_manifest=preflight.manifest(),
                trace_manifest=trace_manifest,
            )
        except Exception as exc:
            await self._emit_lifecycle_event(
                AgentLifecycleEvent(
                    type="run_failed",
                    session_name=self.session.profile.name,
                    task=run_request.task,
                    status="failed",
                    error=str(exc),
                    metadata={"request_metadata": dict(run_request.metadata)},
                )
            )
            raise

    def resume_index(self, *, include_terminal: bool = True) -> ResumeIndex:
        return _resume_index_for_harness(self.session.harness, include_terminal=include_terminal)

    def resume_candidate(self, request: AgentResumeRequest | str | None = None) -> ResumeCandidate:
        resume_request = _normalize_resume_request(request)
        return _resume_candidate_for_request(self.session.harness, resume_request)

    def resume_plan(self, request: AgentResumeRequest | str | None = None) -> ResumePlan:
        resume_request = _normalize_resume_request(request)
        return _resume_plan_for_request(self.session.harness, resume_request)

    async def resume(self, request: AgentResumeRequest | str | None = None) -> AgentRunOutcome:
        resume_request = _normalize_resume_request(request)
        plan = _resume_plan_for_request(self.session.harness, resume_request)
        candidate = _resume_candidate_from_plan(plan, resume_request)
        return await self.run(_run_request_from_resume(candidate, resume_request, plan=plan))

    def _prompt_builder(self) -> AgentPromptBuilder:
        return AgentPromptBuilder(
            tools=self.session.tools,
            skills=self.session.skills,
            timeline=self.session.timeline,
            timeline_budget=self._timeline_budget(),
            capabilities=self.session.capability_catalog(),
            injection_policy=self.session.context_injection_policy,
        )

    async def _emit_lifecycle_event(self, event: AgentLifecycleEvent) -> None:
        await self.session.lifecycle_hooks.emit(event)

    def _timeline_budget(self) -> TimelineBudget:
        return TimelineBudget(max_bytes=self.session.profile.budget.max_timeline_bytes)

    async def _preflight(self, request: AgentRunRequest) -> AgentRunPreflightReport:
        return await self.session.preflight.check(self._preflight_request(request))

    def _preflight_request(self, request: AgentRunRequest) -> AgentRunPreflightRequest:
        requirements = _preflight_requirements_for_request(request)
        return AgentRunPreflightRequest(
            task=request.task,
            session_name=self.session.profile.name,
            requirements=requirements,
            available_actions=tuple(spec.name for spec in self.session.actions.specs()),
            available_tools=tuple(spec.name for spec in self.session.tools.specs() if spec.enabled),
            available_skills=_available_skill_names(self.session.skills),
            available_mcp_servers=_available_mcp_server_names(self.session.mcp),
            memory_enabled=bool(self.session.profile.capabilities.memory_enabled),
            provider_route_plan=_provider_route_plan_manifest(self.session, request),
            storage_backend_preflight=self._storage_backend_preflight(requirements),
            metadata={
                "request_metadata": dict(request.metadata),
                "task_contract": _task_contract_manifest(request),
            },
        )

    def _storage_backend_preflight(
        self,
        requirements: AgentRunPreflightRequirements,
    ) -> dict[str, Any]:
        if not requirements.storage_backend_requirements:
            return {}
        return storage_backend_catalog_from_components(self.session.manifest()).preflight(
            requirements.storage_backend_requirements,
            metadata={"session_name": self.session.profile.name},
        ).manifest()

    async def _preflight_blocked_outcome(
        self,
        request: AgentRunRequest,
        report: AgentRunPreflightReport,
    ) -> AgentRunOutcome:
        result = ReActResult(
            run_id=uuid4().hex,
            status="denied",
            output="run blocked by preflight",
            iterations=0,
            metadata={"preflight": report.manifest()},
        )
        await self._emit_lifecycle_event(
            AgentLifecycleEvent(
                type="run_completed",
                session_name=self.session.profile.name,
                run_id=result.run_id,
                task=request.task,
                status=result.status,
                metadata={"preflight": report.manifest()},
            )
        )
        session_manifest = self.session.manifest()
        trace_manifest = await self._trace_manifest(
            result,
            session_manifest=session_manifest,
            prompt_manifest={},
            resume_manifest={},
            resume_plan_manifest=_request_resume_plan_manifest(request),
            timeline_reduction_manifest={},
            capability_discovery_manifest={},
            memory_search_manifest={},
            preflight_manifest=report.manifest(),
            request=request,
        )
        await self.session.trace_store.save(trace_manifest)
        return AgentRunOutcome(
            result=result,
            session_manifest=session_manifest,
            prompt_manifest={},
            resume_plan_manifest=_request_resume_plan_manifest(request),
            preflight_manifest=report.manifest(),
            trace_manifest=trace_manifest,
        )

    async def _resume_manifest(self, token: ResumeToken | None) -> dict[str, Any]:
        if token is None:
            return {}
        checkpoint = await self.session.harness.resume(token)
        return {
            "schema_version": "agent-core-resume/v1",
            "run_id": checkpoint.run_id,
            "turn_id": checkpoint.turn_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "sequence": checkpoint.sequence,
            "created_at": checkpoint.created_at,
            "state": dict(checkpoint.state),
            "token_metadata": dict(token.metadata),
        }

    def _context_for(
        self,
        request: AgentRunRequest,
        *,
        resume_manifest: dict[str, Any] | None = None,
        injections: tuple[ContextInjection, ...] = (),
        context_material_selection_manifest: dict[str, Any] | None = None,
        timeline_reduction_manifest: dict[str, Any] | None = None,
    ) -> AgentContextPack:
        base = request.context or AgentContextPack()
        system = base.system or self.session.profile.instructions
        dynamic_task = base.dynamic_task or request.task
        resume_manifest = resume_manifest or {}
        timeline_reduction_manifest = timeline_reduction_manifest or {}
        context_injections = (*base.injections, *injections)
        if resume_manifest:
            context_injections = (
                *context_injections,
                ContextInjection(
                    name="resume_checkpoint",
                    content=_resume_context_block(resume_manifest),
                    target=PromptBucketRole.TIMELINE_OPEN,
                    source="harness",
                    priority=100,
                    metadata={
                        "run_id": resume_manifest.get("run_id"),
                        "checkpoint_id": resume_manifest.get("checkpoint_id"),
                        "sequence": resume_manifest.get("sequence"),
                    },
                ),
            )
        approval_resume_manifest = _approval_resume_manifest(request.approval_resume)
        if approval_resume_manifest:
            context_injections = (
                *context_injections,
                ContextInjection(
                    name="approval_resume",
                    content=_approval_resume_context_block(approval_resume_manifest),
                    target=PromptBucketRole.TIMELINE_OPEN,
                    source="approval",
                    priority=95,
                    metadata={
                        "grant_count": approval_resume_manifest.get("grant_count"),
                        "approved_count": approval_resume_manifest.get("approved_count"),
                    },
                ),
            )
        metadata = {
            **base.metadata,
            **request.metadata,
            "profile": self.session.profile.name,
        }
        context_material_selection_manifest = context_material_selection_manifest or {}
        if context_material_selection_manifest:
            metadata["context_material_selection"] = context_material_selection_manifest
        if resume_manifest:
            metadata["resume"] = resume_manifest
        if approval_resume_manifest:
            metadata["approval_resume"] = approval_resume_manifest
        if timeline_reduction_manifest:
            metadata["timeline_reduction"] = timeline_reduction_manifest
        if request.timeout_seconds is not None:
            metadata["timeout_seconds"] = request.timeout_seconds
        schema = base.schema
        if request.structured_output is not None:
            metadata["structured_output"] = request.structured_output.manifest()
            schema = _append_context_block(schema, request.structured_output.render_prompt())
        return AgentContextPack(
            system=system,
            task_instruction=base.task_instruction,
            schema=schema,
            output_example=base.output_example,
            recent_tools_cache=base.recent_tools_cache,
            user_history=base.user_history,
            workspace=base.workspace,
            current_time=base.current_time,
            dynamic_task=dynamic_task,
            injections=context_injections,
            metadata=metadata,
        )

    async def _reduce_timeline_if_needed(self, request: AgentRunRequest) -> dict[str, Any]:
        reducer = self.session.context_reducer
        if reducer is None:
            return {}
        budget = self._timeline_budget()
        total_bytes = self.session.timeline.total_bytes()
        if total_bytes <= budget.max_bytes:
            return {}
        reducer_request = ReducerRequest(
            items=self.session.timeline.items,
            max_bytes=budget.max_bytes,
            recent_keep_ratio=budget.recent_keep_ratio,
            metadata={
                "profile": self.session.profile.name,
                "request_metadata": dict(request.metadata),
            },
        )
        result = await reducer.reduce(reducer_request)
        view = apply_reduction_to_timeline(self.session.timeline, result)
        manifest = result.manifest()
        manifest["request"] = reducer_request.manifest()
        manifest["view"] = {
            "open_item_count": len(view.open_items),
            "compressed_head_bytes": len(view.compressed_head.encode("utf-8")),
            "archive_refs": list(view.archive_refs),
        }
        return manifest

    def _capability_discovery_manifest(
        self,
        request: AgentRunRequest,
        *,
        query_text: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        catalog = self.session.capability_catalog()
        resolved_query = query_text or (request.context.dynamic_task if request.context else "") or request.task
        manifest = catalog.discover(
            CapabilityQuery(
                query=resolved_query,
                limit=16,
            )
        ).manifest()
        if metadata:
            manifest["metadata"] = {**dict(manifest.get("metadata") or {}), **dict(metadata)}
        return manifest

    async def _memory_injections(self, request: AgentRunRequest) -> tuple[ContextInjection, ...]:
        return (await self._memory_recall(request)).injections

    async def _context_material_selection(
        self,
        request: AgentRunRequest,
        *,
        query: ContextMaterialQuery | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentContextMaterialSelection:
        store_materials, store_manifest = await self._stored_context_materials(request, query=query)
        mcp_materials, mcp_manifest = await self._mcp_context_materials(request)
        extra_metadata = dict(metadata or {})
        if store_manifest:
            extra_metadata["context_material_store"] = store_manifest
        if mcp_manifest:
            extra_metadata["mcp_context_materials"] = mcp_manifest
        selection_request = _context_material_selection_request(
            request,
            extra_materials=(*store_materials, *mcp_materials),
            extra_metadata=extra_metadata,
            task_override=query.query if query is not None else "",
        )
        if not selection_request.materials:
            return AgentContextMaterialSelection()
        selector = self.session.context_material_selector or DefaultContextMaterialSelector()
        result = selector.select(selection_request)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, ContextMaterialSelectionResult):
            manifest = getattr(result, "manifest", None)
            injections = tuple(getattr(result, "injections", ()))
            return AgentContextMaterialSelection(
                injections=tuple(
                    item for item in injections if isinstance(item, ContextInjection)
                ),
                manifest=dict(manifest()) if callable(manifest) else {},
            )
        return AgentContextMaterialSelection(
            injections=result.injections,
            manifest=result.manifest(),
        )

    async def _stored_context_materials(
        self,
        request: AgentRunRequest,
        *,
        query: ContextMaterialQuery | None = None,
    ) -> tuple[tuple[ContextMaterial, ...], dict[str, Any]]:
        context_query = query or request.context_material_query
        if context_query is None or self.session.context_material_store is None:
            return (), {}
        store = self.session.context_material_store
        plan_search = getattr(store, "plan_search", None)
        plan_manifest = {}
        if callable(plan_search):
            plan = plan_search(context_query)
            plan_manifest_method = getattr(plan, "manifest", None)
            if callable(plan_manifest_method):
                plan_manifest = plan_manifest_method()
        try:
            materials = await store.search(context_query)
        except Exception as exc:
            return (), {
                "schema_version": "agent-core-context-material-store-selection/v1",
                "status": "failed",
                "material_count": 0,
                "query": context_query.manifest(),
                "plan": plan_manifest,
                "error": str(exc),
            }
        return materials, {
            "schema_version": "agent-core-context-material-store-selection/v1",
            "status": "completed",
            "material_count": len(materials),
            "query": context_query.manifest(),
            "plan": plan_manifest,
            "materials": [material.manifest() for material in materials],
        }

    async def _mcp_context_materials(
        self,
        request: AgentRunRequest,
    ) -> tuple[tuple[ContextMaterial, ...], dict[str, Any]]:
        if request.mcp_context_materials is None or self.session.mcp is None:
            return (), {}
        try:
            result = await self.session.mcp.context_materials(request.mcp_context_materials)
        except Exception as exc:
            return (), {
                "schema_version": "agent-core-mcp-context-material-selection/v1",
                "status": "failed",
                "request": request.mcp_context_materials.manifest(),
                "material_count": 0,
                "error": str(exc),
            }
        return result.materials, result.manifest()

    async def _memory_recall(
        self,
        request: AgentRunRequest,
        *,
        query_text: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AgentMemoryRecall:
        if not self.session.profile.capabilities.memory_enabled:
            return AgentMemoryRecall(
                manifest={
                    "schema_version": "agent-core-memory-recall/v1",
                    "enabled": False,
                    "reason": "profile_memory_disabled",
                    "hit_count": 0,
                }
            )
        context = request.context or AgentContextPack()
        resolved_query_text = query_text or context.dynamic_task or request.task
        query = MemoryQuery(
            query=resolved_query_text,
            limit=5,
        )
        plan_manifest = _memory_search_plan_manifest(self.session.memory, query)
        fallback_manifest: dict[str, Any] = {}
        try:
            hits = await asyncio.wait_for(self.session.memory.search(query), timeout=0.2)
            timed_out = False
        except asyncio.TimeoutError:
            timed_out = True
            fallback_query = MemoryQuery(
                query=resolved_query_text,
                limit=5,
                mode="keyword",
            )
            try:
                hits = await asyncio.wait_for(
                    self.session.memory.search(fallback_query),
                    timeout=0.05,
                )
                fallback_timed_out = False
            except asyncio.TimeoutError:
                hits = ()
                fallback_timed_out = True
            fallback_manifest = {
                "schema_version": "agent-core-memory-recall-fallback/v1",
                "reason": "quick_semantic_timeout",
                "query": fallback_query.manifest(),
                "hit_count": len(hits),
                "timed_out": fallback_timed_out,
            }
        manifest = {
            "schema_version": "agent-core-memory-recall/v1",
            "enabled": True,
            "query": query.manifest(),
            "plan": plan_manifest,
            "hit_count": len(hits),
            "hits": [_memory_hit_manifest(hit) for hit in hits],
            "metadata": {
                **dict(metadata or {}),
                **({"quick_timeout_seconds": 0.2, "timed_out": True} if timed_out else {}),
                **({"fallback": fallback_manifest} if fallback_manifest else {}),
            },
        }
        return self._memory_recall_from_hits(
            hits,
            resolved_query_text=resolved_query_text,
            manifest=manifest,
            metadata=metadata,
        )

    async def _keyword_memory_recall(
        self,
        request: AgentRunRequest,
        *,
        query_text: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AgentMemoryRecall:
        if not self.session.profile.capabilities.memory_enabled:
            return AgentMemoryRecall(
                manifest={
                    "schema_version": "agent-core-memory-recall/v1",
                    "enabled": False,
                    "reason": "profile_memory_disabled",
                    "hit_count": 0,
                }
            )
        context = request.context or AgentContextPack()
        resolved_query_text = query_text or context.dynamic_task or request.task
        query = MemoryQuery(
            query=resolved_query_text,
            limit=5,
            mode="keyword",
        )
        plan_manifest = _memory_search_plan_manifest(self.session.memory, query)
        try:
            hits = await asyncio.wait_for(self.session.memory.search(query), timeout=0.2)
            timed_out = False
        except asyncio.TimeoutError:
            hits = ()
            timed_out = True
        manifest = {
            "schema_version": "agent-core-memory-recall/v1",
            "enabled": True,
            "query": query.manifest(),
            "plan": plan_manifest,
            "hit_count": len(hits),
            "hits": [_memory_hit_manifest(hit) for hit in hits],
            "metadata": {
                **dict(metadata or {}),
                "strategy": "yaklang_fast_keyword",
                "timed_out": timed_out,
            },
        }
        return self._memory_recall_from_hits(
            hits,
            resolved_query_text=resolved_query_text,
            manifest=manifest,
            metadata=metadata,
        )

    def _memory_recall_from_hits(
        self,
        hits: tuple[MemoryHit, ...],
        *,
        resolved_query_text: str,
        manifest: dict[str, Any],
        metadata: dict[str, Any] | None,
    ) -> AgentMemoryRecall:
        if not hits:
            return AgentMemoryRecall(manifest=manifest)
        topics = _string_sequence((metadata or {}).get("perception_topics"))
        keywords = _string_sequence((metadata or {}).get("perception_keywords"))
        downstream_plan = (metadata or {}).get("downstream_plan")
        if isinstance(downstream_plan, dict):
            topics = (*topics, *_string_sequence(downstream_plan.get("topics")))
            keywords = (*keywords, *_string_sequence(downstream_plan.get("keywords")))
        intent = infer_memory_recall_intent(
            resolved_query_text,
            topics=topics,
            keywords=keywords,
            metadata=dict(metadata or {}),
        )
        injection = build_memory_injection(
            hits,
            query=resolved_query_text,
            intent=intent,
        )
        manifest["injection"] = injection.manifest()
        return AgentMemoryRecall(
            injections=(
                ContextInjection(
                    name="memory_recall",
                    content=injection.content or _memory_context_block(hits),
                    target=PromptBucketRole.TIMELINE_OPEN,
                    source="memory",
                    priority=80,
                    metadata={
                        "query": resolved_query_text,
                        "hit_count": len(hits),
                        "intent": intent,
                        "routes": list(injection.manifest().get("routes") or ()),
                        "sources": [hit.source for hit in hits if hit.source],
                        "memory_injection": injection.manifest(),
                        **dict(metadata or {}),
                    },
                ),
            ),
            manifest=manifest,
        )

    def _executor(
        self,
        approval_resume: ApprovalResumeContext | None = None,
        structured_output: StructuredOutputSpec | None = None,
        timeout_seconds: float | None = None,
        native_tool_calls: bool | None = None,
        stream: bool | None = None,
        turn_refresher: TurnContextRefresherPort | None = None,
    ) -> ReActExecutor:
        budget = self.session.profile.budget
        effective_native_tool_calls = (
            self.session.native_tool_calls
            if native_tool_calls is None
            else bool(native_tool_calls)
        )
        effective_stream = self.session.stream if stream is None else bool(stream)
        return ReActExecutor(
            provider=self.session.provider,
            tool_runtime=self.session.tools,
            action_registry=self.session.actions,
            harness=self.session.harness,
            event_sink=self.session.event_sink,
            policy=self.session.policy,
            policy_decision_store=self.session.policy_decision_store,
            approval_store=self.session.approval_store,
            approval_resume=approval_resume,
            memory=self.session.memory,
            skills=self.session.skills,
            mcp=self.session.mcp,
            capability_catalog=self.session.capability_catalog(),
            timeline=self.session.timeline,
            tool_replay=self.session.tool_replay,
            action_verifier=self.session.action_verifier,
            structured_output_validator=self.session.structured_output_validator,
            loop_guard=self.session.loop_guard,
            artifact_store=self.session.artifact_store,
            cancel_token=self.session.cancel_token,
            turn_refresher=turn_refresher,
            config=ReActConfig(
                model=self.session.profile.model,
                max_iterations=budget.max_iterations,
                budget=budget,
                structured_output=structured_output,
                timeout_seconds=timeout_seconds,
                stream=effective_stream,
                native_tool_calls=effective_native_tool_calls,
                provider_cache_mode=self.session.provider_cache_mode,
                provider_cache_min_segment_bytes=self.session.provider_cache_min_segment_bytes,
            ),
        )

    async def _trace_manifest(
        self,
        result: ReActResult,
        *,
        session_manifest: dict[str, Any],
        prompt_manifest: dict[str, Any],
        resume_manifest: dict[str, Any],
        resume_plan_manifest: dict[str, Any],
        timeline_reduction_manifest: dict[str, Any],
        capability_discovery_manifest: dict[str, Any],
        memory_search_manifest: dict[str, Any],
        preflight_manifest: dict[str, Any],
        request: AgentRunRequest,
    ) -> dict[str, Any]:
        bundle = AgentRunTraceBundle(
            run_id=result.run_id,
            status=result.status,
            iterations=result.iterations,
            output_bytes=len(result.output.encode("utf-8")),
            session=session_manifest,
            prompt=prompt_manifest,
            journal_replay=self._journal_replay_manifest(result.run_id),
            provider=await _component_manifest(self.session.provider),
            tool_replay=await _component_manifest(self.session.tool_replay),
            policy_decisions=await _component_manifest(self.session.policy_decision_store),
            approvals=await _component_manifest(self.session.approval_store),
            event_log=await _component_manifest(self.session.event_sink),
            resume=resume_manifest,
            resume_plan=resume_plan_manifest,
            timeline_reduction=timeline_reduction_manifest,
            capability_discovery=capability_discovery_manifest,
            memory_search=memory_search_manifest,
            preflight=preflight_manifest,
            metadata={
                "profile": self.session.profile.name,
                "request_metadata": dict(request.metadata),
                "prompt_trim": dict(prompt_manifest.get("metadata", {}).get("trim") or {}),
                "prompt_budget": dict(prompt_manifest.get("metadata", {}).get("prompt_budget") or {}),
                "prompt_semantic_trim": dict(
                    prompt_manifest.get("metadata", {}).get("semantic_trim") or {}
                ),
                "native_tool_calls": self._native_tool_calls_for_request(request),
                "stream": self._stream_for_request(request),
            },
        )
        return bundle.manifest()

    def _native_tool_calls_for_request(self, request: AgentRunRequest) -> bool:
        if request.native_tool_calls is None:
            return self.session.native_tool_calls
        return bool(request.native_tool_calls)

    def _stream_for_request(self, request: AgentRunRequest) -> bool:
        if request.stream is None:
            return self.session.stream
        return bool(request.stream)

    def _journal_replay_manifest(self, run_id: str) -> dict[str, Any]:
        snapshot = getattr(self.session.harness, "snapshot", None)
        if not callable(snapshot):
            return {}
        return AgentJournalReplay.from_snapshot(snapshot(), run_id=run_id).manifest()

    async def _semantic_trim_prompt_if_needed(
        self,
        request: AgentRunRequest,
        prompt: PromptIR,
        budget_plan: AgentPromptBudgetPlan,
    ) -> PromptIR:
        reducer = self.session.prompt_semantic_reducer
        if reducer is None:
            return prompt
        reducer_request = PromptSemanticTrimRequest(
            prompt=prompt,
            task=request.task,
            target_bytes=budget_plan.target_prompt_bytes,
            metadata={
                "profile": self.session.profile.name,
                "request_metadata": dict(request.metadata),
                "prompt_budget": budget_plan.manifest(),
            },
        )
        result = await reducer.reduce(reducer_request)
        return result.prompt

    def _prompt_budget_plan(self, request: AgentRunRequest) -> AgentPromptBudgetPlan:
        profile_budget = max(1, int(self.session.profile.budget.max_prompt_bytes))
        route_plan = _provider_route_plan_manifest(self.session, request)
        selected = route_plan.get("selected_route") if isinstance(route_plan, dict) else {}
        selected = selected if isinstance(selected, dict) else {}
        route_metadata = selected.get("metadata") if isinstance(selected.get("metadata"), dict) else {}
        capabilities = route_metadata.get("model_capabilities")
        capabilities = capabilities if isinstance(capabilities, dict) else {}
        context_window = int(capabilities.get("context_window_tokens") or 0)
        max_output = int(capabilities.get("max_output_tokens") or 0)
        reserved_output = _prompt_reserved_output_tokens(request, max_output)
        provider_input_budget = 0
        if context_window > 0:
            input_tokens = max(1, context_window - reserved_output)
            provider_input_budget = max(1, input_tokens * 4)
        target = min(profile_budget, provider_input_budget) if provider_input_budget else profile_budget
        return AgentPromptBudgetPlan(
            profile_max_prompt_bytes=profile_budget,
            target_prompt_bytes=target,
            provider_name=str(selected.get("provider_name") or ""),
            model=str(selected.get("model") or self.session.profile.model or ""),
            context_window_tokens=context_window,
            reserved_output_tokens=reserved_output,
            provider_input_budget_bytes=provider_input_budget,
            source="provider_context_window" if provider_input_budget and target < profile_budget else "profile_budget",
            route_plan=route_plan,
            metadata={
                "profile": self.session.profile.name,
                "request_metadata": dict(request.metadata),
                "stream": self._stream_for_request(request),
            },
        )


class _DefaultRunnerCapabilityRefresh:
    """Default turn-time capability recall backed by CapabilityCatalog."""

    def __init__(self, runner: AgentRunner) -> None:
        self.runner = runner
        self._injections: tuple[ContextInjection, ...] = ()
        self.last_manifest: dict[str, Any] = {"enabled": False, "reason": "not_started"}

    async def refresh(self, request: TurnRefreshRequest) -> dict[str, Any]:
        metadata = dict(request.metadata)
        downstream = metadata.get("perception_downstream_refresh")
        perception_state = metadata.get("perception_state")
        state = {}
        if isinstance(perception_state, dict):
            state = dict(perception_state.get("state") or perception_state)
        query = str(metadata.get("perception_recall_query") or request.task or "").strip()
        perception_terms = (
            *_string_sequence(state.get("topics")),
            *_string_sequence(state.get("keywords")),
        )
        action_history = _metadata_sequence(metadata.get("action_history"))
        loaded_capabilities = _loaded_capabilities_from_action_history(action_history)
        recall = self.runner.session.capability_catalog().recall(
            CapabilityRecallRequest(
                query=query,
                limit=12,
                perception_terms=perception_terms,
                recent_tools=(
                    *_recent_tools_from_action_history(action_history),
                    *tuple(
                        item.get("name", "")
                        for item in loaded_capabilities
                        if item.get("kind") in {"tool", "mcp_tool"}
                    ),
                ),
                failed_tools=_failed_tools_from_action_history(action_history),
                scenario_whitelist=_string_sequence(metadata.get("scenario_whitelist")),
                metadata={
                    "turn_id": request.turn_id,
                    "iteration": request.iteration,
                    "perception_downstream_refresh": metadata.get(
                        "perception_downstream_refresh"
                    ),
                    "loaded_capabilities": loaded_capabilities,
                },
            )
        )
        manifest = recall.manifest()
        self.last_manifest = manifest
        rendered = recall.render_prompt()
        should_inject = bool(downstream) or bool(recall.failed_tools) or bool(loaded_capabilities)
        if should_inject and (recall.matches or recall.failed_tools):
            self._injections = (
                ContextInjection(
                    name="capability_recall",
                    content=rendered,
                    target=PromptBucketRole.TIMELINE_OPEN,
                    source="capability",
                    priority=76,
                    metadata=manifest,
                ),
            )
        else:
            self._injections = ()
        return manifest

    def context_injections(self) -> tuple[ContextInjection, ...]:
        return self._injections

    def manifest(self) -> dict[str, Any]:
        return self.last_manifest


class _RunnerTurnContextRefresher:
    """Internal default refresher that moves run-start context work into the loop."""

    def __init__(
        self,
        runner: AgentRunner,
        *,
        run_request: AgentRunRequest,
        resume_manifest: dict[str, Any],
    ) -> None:
        self.runner = runner
        self.run_request = run_request
        self.resume_manifest = dict(resume_manifest)
        self.perception = runner.session.perception_controller or YaklangStylePerceptionController(
            evaluator=DeterministicPerceptionEvaluator()
        )
        self.capability = runner.session.capability_refresher or _DefaultRunnerCapabilityRefresh(runner)
        self.knowledge = runner.session.knowledge_recall or DefaultKnowledgeRecall(
            store=runner.session.context_material_store,
            mcp=runner.session.mcp,
        )
        self.midterm_timeline = (
            runner.session.midterm_timeline_recall
            or DefaultMidtermTimelineRecall(runner.session.timeline)
        )
        self.downstream_scheduler = PerceptionDownstreamScheduler()
        self.perception_injections: tuple[ContextInjection, ...] = ()
        self.capability_injections: tuple[ContextInjection, ...] = ()
        self.knowledge_injections: tuple[ContextInjection, ...] = ()
        self.midterm_timeline_injections: tuple[ContextInjection, ...] = ()
        self.task_state_injections: tuple[ContextInjection, ...] = ()
        self.memory_flush_buffer = MemoryFlushBuffer()
        self._pending_memory_recall: asyncio.Task[AgentMemoryRecall] | None = None
        self.last_memory_manifest: dict[str, Any] = {}
        self.last_completed_memory_recall: AgentMemoryRecall | None = None
        self.last_context_material_manifest: dict[str, Any] = {}
        self.last_timeline_reduction_manifest: dict[str, Any] = {}
        self.last_capability_manifest: dict[str, Any] = {}
        self.last_knowledge_manifest: dict[str, Any] = {}
        self.last_midterm_timeline_manifest: dict[str, Any] = {}
        self.last_memory_flush_manifest: dict[str, Any] = {}
        self.last_downstream_plan_manifest: dict[str, Any] = {}
        self.last_loop_state_manifest: dict[str, Any] = {}

    async def before_model_call(self, request: TurnRefreshRequest) -> TurnRefreshResult:
        self._drain_perception_observation_to_timeline()
        timeline_diff = self.runner.session.timeline.diff_since(request.timeline_cursor)
        perception_manifest = self.perception.manifest()
        downstream_refresh = _consume_perception_downstream_refresh(self.perception)
        downstream_plan = self.downstream_scheduler.plan(
            downstream_refresh,
            fallback_query=(self.run_request.context.dynamic_task if self.run_request.context else "")
            or self.run_request.task,
        )
        downstream_intents = downstream_plan.intents
        perception_recall_query = downstream_plan.query or _perception_recall_query(
            downstream_refresh=downstream_refresh,
            fallback=(self.run_request.context.dynamic_task if self.run_request.context else "")
            or self.run_request.task,
        )
        perception_query_metadata = _perception_query_metadata(downstream_refresh)
        perception_query_metadata["downstream_plan"] = downstream_plan.manifest()
        memory_recall = await self._fast_memory_recall(
            enabled=downstream_plan.intent_enabled("memory_recall", default=True),
            query_text=perception_recall_query,
            metadata=perception_query_metadata,
        )
        context_material_query = _perception_context_material_query(
            downstream_refresh=downstream_refresh,
            fallback=self.run_request.context_material_query,
        )
        context_material_selection = (
            await self.runner._context_material_selection(
                self.run_request,
                query=context_material_query,
                metadata=(
                    {"perception_downstream_refresh": downstream_refresh}
                    if downstream_refresh
                    else {}
                ),
            )
            if downstream_plan.intent_enabled("context_material_recall", default=True)
            else AgentContextMaterialSelection(
                manifest={
                    "schema_version": "agent-core-context-material-selection/v1",
                    "enabled": False,
                    "reason": "perception_intent_disabled",
                }
            )
        )
        timeline_reduction_manifest = await self.runner._reduce_timeline_if_needed(self.run_request)
        capability_request = TurnRefreshRequest(
            task=request.task,
            iteration=request.iteration,
            run_id=request.run_id,
            turn_id=request.turn_id,
            bootstrap_prompt=request.bootstrap_prompt,
            token_budget=request.token_budget,
            timeline_cursor=request.timeline_cursor,
            compact_delta=request.compact_delta,
            metadata={
                **dict(request.metadata),
                "perception_state": perception_manifest,
                "perception_downstream_refresh": downstream_refresh,
                "perception_downstream_intents": downstream_intents,
                "perception_recall_query": perception_recall_query,
                "timeline_diff": timeline_diff.manifest(),
                "action_history": _metadata_sequence(request.metadata.get("action_history")),
                "recent_tools": _recent_tools_from_action_history(
                    _metadata_sequence(request.metadata.get("action_history"))
                ),
                "failed_tools": _failed_tools_from_action_history(
                    _metadata_sequence(request.metadata.get("action_history"))
                ),
                "scenario_whitelist": (
                    *_string_sequence(self.runner.session.metadata.get("scenario_whitelist")),
                    *_string_sequence(self.run_request.metadata.get("scenario_whitelist")),
                ),
            },
        )
        capability_manifest = (
            await self.capability.refresh(capability_request)
            if downstream_plan.intent_enabled("capability_search", default=True)
            else {"enabled": False, "reason": "perception_intent_disabled"}
        )
        if downstream_plan.intent_enabled("capability_search", default=True) and not capability_manifest.get("enabled"):
            capability_query = _perception_capability_query(
                downstream_refresh=downstream_refresh,
                fallback=perception_recall_query,
            )
            capability_manifest = self.runner._capability_discovery_manifest(
                self.run_request,
                query_text=capability_query,
                metadata={
                    "perception_downstream_refresh": downstream_refresh,
                    "perception_state": perception_manifest,
                },
            )

        knowledge_result = (
            await self.knowledge.recall(
                KnowledgeRecallRequest(
                    query=perception_recall_query,
                    topics=downstream_plan.topics,
                    keywords=downstream_plan.keywords,
                    limit=5,
                    metadata={
                        "turn_id": request.turn_id,
                        "iteration": request.iteration,
                        "downstream_plan": downstream_plan.manifest(),
                    },
                )
            )
            if downstream_plan.intent_enabled("knowledge_search", default=False)
            else None
        )
        if knowledge_result is not None:
            self.knowledge_injections = knowledge_result.injections
            knowledge_manifest = knowledge_result.manifest()
        else:
            self.knowledge_injections = ()
            knowledge_manifest = {"enabled": False, "reason": "perception_intent_disabled"}

        midterm_result = (
            await self.midterm_timeline.recall(
                MidtermTimelineRecallRequest(
                    query=perception_recall_query,
                    topics=downstream_plan.topics,
                    keywords=downstream_plan.keywords,
                    limit=6,
                    metadata={
                        "turn_id": request.turn_id,
                        "iteration": request.iteration,
                        "downstream_plan": downstream_plan.manifest(),
                    },
                )
            )
            if downstream_plan.intent_enabled("midterm_timeline_recall", default=False)
            else None
        )
        if midterm_result is not None:
            self.midterm_timeline_injections = midterm_result.injections
            midterm_manifest = midterm_result.manifest()
        else:
            self.midterm_timeline_injections = ()
            midterm_manifest = {"enabled": False, "reason": "perception_intent_disabled"}

        current_perception_injections = _perception_context_injections(self.perception)
        if current_perception_injections:
            self.perception_injections = current_perception_injections
        current_capability_injections = _capability_context_injections(self.capability)
        if current_capability_injections:
            self.capability_injections = current_capability_injections
        recent_tool_injections = _recent_tools_cache_context_injections(
            _metadata_sequence(request.metadata.get("action_history"))
        )
        self.task_state_injections = _task_state_context_injections(
            self.run_request,
            request=request,
            timeline_diff=timeline_diff,
            context_material_selection=context_material_selection.manifest,
            perception_state=perception_manifest,
            memory_flush=self.last_memory_flush_manifest,
        )
        self.last_loop_state_manifest = (
            dict(self.task_state_injections[0].metadata.get("loop_state") or {})
            if self.task_state_injections
            else {}
        )

        self.last_memory_manifest = memory_recall.manifest
        self.last_context_material_manifest = context_material_selection.manifest
        self.last_timeline_reduction_manifest = timeline_reduction_manifest
        self.last_capability_manifest = capability_manifest
        self.last_knowledge_manifest = knowledge_manifest
        self.last_midterm_timeline_manifest = midterm_manifest
        self.last_downstream_plan_manifest = downstream_plan.manifest()

        context = self.runner._context_for(
            self.run_request,
            resume_manifest=self.resume_manifest,
            injections=(
                *memory_recall.injections,
                *context_material_selection.injections,
                *self.capability_injections,
                *recent_tool_injections,
                *self.knowledge_injections,
                *self.midterm_timeline_injections,
                *self.perception_injections,
                *self.task_state_injections,
            ),
            context_material_selection_manifest=context_material_selection.manifest,
            timeline_reduction_manifest=timeline_reduction_manifest,
        )
        prompt = self.runner._prompt_builder().build(context)
        prompt_budget_plan = self.runner._prompt_budget_plan(self.run_request)
        prompt = _prompt_with_budget_plan(prompt, prompt_budget_plan)
        if self.runner.session.prompt_bucket_budget_policy is not None:
            prompt = self.runner.session.prompt_bucket_budget_policy.apply(prompt)
        prompt = await self.runner._semantic_trim_prompt_if_needed(
            self.run_request,
            prompt,
            prompt_budget_plan,
        )
        prompt = prompt.trim_to_budget(prompt_budget_plan.target_prompt_bytes)
        metadata = {
            "turn_refresh": request.manifest(),
            "capability_selection": capability_manifest,
            "knowledge_recall": knowledge_manifest,
            "midterm_timeline_recall": midterm_manifest,
            "memory_recall": memory_recall.manifest,
            "context_material_selection": context_material_selection.manifest,
            "timeline_reduction": timeline_reduction_manifest,
            "perception": perception_manifest,
            "perception_downstream_refresh": downstream_refresh,
            "perception_downstream_plan": downstream_plan.manifest(),
            "loop_state": self.last_loop_state_manifest,
            "memory_flush": self.last_memory_flush_manifest,
        }
        prompt = PromptIR(buckets=prompt.buckets, metadata={**prompt.metadata, **metadata})
        return TurnRefreshResult(
            prompt=prompt,
            timeline_cursor=timeline_diff.next_cursor,
            timeline_diff=timeline_diff,
            memory_hits=tuple(memory_recall.manifest.get("hits") or ()),
            context_injections=(
                *memory_recall.injections,
                *context_material_selection.injections,
                *self.capability_injections,
                *recent_tool_injections,
                *self.knowledge_injections,
                *self.midterm_timeline_injections,
                *self.perception_injections,
                *self.task_state_injections,
            ),
            perception_state=self.perception.manifest(),
            capability_selection=capability_manifest,
            metadata=metadata,
        )

    async def after_model_response(self, event: Any) -> None:
        return None

    async def after_tool_result(self, event: ToolResultEvent) -> None:
        injections = await self.perception.after_tool_result(event)
        if injections:
            self.perception_injections = (*self.perception_injections, *injections)[-8:]
        self._drain_perception_observation_to_timeline()

    async def after_turn(self, event: TurnCompletedEvent) -> None:
        if not self.runner.session.profile.capabilities.memory_enabled:
            return
        diff = event.timeline_diff
        if diff is None or not diff.changed:
            if event.status in {
                "finished",
                "provider_tool_finished",
                "loop_stalled",
                "output_validation_failed",
                "max_iterations",
            }:
                writes = await self.memory_flush_buffer.flush(reason=event.status)
                self.last_memory_flush_manifest = dict(self.memory_flush_buffer.last_manifest)
                for write in writes:
                    await self.runner.session.memory.write(write)
            return
        rendered = diff.render().strip()
        if not rendered:
            return
        writes = await self.memory_flush_buffer.observe(
            MemoryFlushSignal(
                content=rendered,
                run_id=event.run_id,
                turn_id=event.turn_id,
                iteration=event.iteration,
                status=event.status,
                is_done=event.status
                in {"finished", "provider_tool_finished", "loop_stalled", "output_validation_failed"},
                metadata={"timeline_diff": diff.manifest()},
            )
        )
        self.last_memory_flush_manifest = dict(self.memory_flush_buffer.last_manifest)
        for write in writes:
            await self.runner.session.memory.write(write)

    def _drain_perception_observation_to_timeline(self) -> None:
        observation = _consume_perception_observation(self.perception)
        if not observation or not observation.get("updated"):
            return
        content = _render_perception_observation(observation)
        if not content:
            return
        self.runner.session.timeline.add(
            content,
            kind="perception",
            perception=observation,
        )

    def _harvest_pending_memory_recall(self) -> AgentMemoryRecall | None:
        task = self._pending_memory_recall
        if task is None or not task.done():
            return None
        self._pending_memory_recall = None
        try:
            recall = task.result()
        except Exception as exc:  # pragma: no cover - defensive guard for async background tasks
            recall = AgentMemoryRecall(
                manifest={
                    "schema_version": "agent-core-memory-recall/v1",
                    "enabled": True,
                    "hit_count": 0,
                    "strategy": "yaklang_fast_background",
                    "error": str(exc),
                }
            )
        self.last_completed_memory_recall = recall
        self.last_memory_manifest = recall.manifest
        return recall

    async def _fast_memory_recall(
        self,
        *,
        enabled: bool,
        query_text: str,
        metadata: dict[str, Any],
    ) -> AgentMemoryRecall:
        if not enabled:
            self._pending_memory_recall = None
            self.last_completed_memory_recall = None
            return AgentMemoryRecall(
                manifest={
                    "schema_version": "agent-core-memory-recall/v1",
                    "enabled": False,
                    "reason": "perception_intent_disabled",
                    "hit_count": 0,
                    "metadata": metadata,
                }
            )

        completed = self._harvest_pending_memory_recall()
        if completed is not None:
            manifest = {
                **completed.manifest,
                "strategy": "yaklang_fast_background",
                "used_from_background": True,
            }
            return AgentMemoryRecall(injections=completed.injections, manifest=manifest)

        if self._pending_memory_recall is None:
            self._pending_memory_recall = asyncio.create_task(
                self.runner._memory_recall(
                    self.run_request,
                    query_text=query_text,
                    metadata={**metadata, "strategy": "yaklang_fast_background"},
                )
            )

        keyword_task: asyncio.Task[AgentMemoryRecall] | None = None
        wait_tasks: set[asyncio.Task[AgentMemoryRecall]] = {self._pending_memory_recall}
        if self._local_keyword_memory_recall_enabled():
            keyword_task = asyncio.create_task(
                self.runner._keyword_memory_recall(
                    self.run_request,
                    query_text=query_text,
                    metadata={**metadata, "strategy": "yaklang_fast_keyword"},
                )
            )
            wait_tasks.add(keyword_task)
        done, _pending = await asyncio.wait(
            wait_tasks,
            timeout=0.2,
        )
        keyword_recall: AgentMemoryRecall | None = None
        if keyword_task is not None and keyword_task in done:
            keyword_recall = keyword_task.result()
        elif keyword_task is not None and not keyword_task.done():
            keyword_task.cancel()

        if done:
            recall = None
            if self._pending_memory_recall in done:
                recall = self._harvest_pending_memory_recall()
            selected = recall
            selected_hit_count = int(selected.manifest.get("hit_count") or 0) if selected else 0
            keyword_hit_count = (
                int(keyword_recall.manifest.get("hit_count") or 0) if keyword_recall else 0
            )
            if keyword_recall is not None and (
                selected is None
                or (not selected_hit_count and keyword_hit_count)
            ):
                selected = keyword_recall
            if selected is not None:
                manifest = {
                    **selected.manifest,
                    "strategy": "yaklang_fast_background",
                    "wait_seconds": 0.2,
                    "timed_out": False,
                }
                if selected is keyword_recall and self._pending_memory_recall is not None:
                    manifest["semantic_pending"] = True
                    manifest["strategy"] = "yaklang_fast_keyword"
                return AgentMemoryRecall(injections=selected.injections, manifest=manifest)

        cached = self.last_completed_memory_recall
        if cached is not None:
            return AgentMemoryRecall(
                injections=cached.injections,
                manifest={
                    **cached.manifest,
                    "strategy": "yaklang_fast_background",
                    "timed_out": True,
                    "used_cached": True,
                    "wait_seconds": 0.2,
                    "pending": True,
                },
            )
        return AgentMemoryRecall(
            manifest={
                "schema_version": "agent-core-memory-recall/v1",
                "enabled": True,
                "hit_count": 0,
                "query": {
                    "query": query_text,
                    "limit": 5,
                    "mode": "hybrid",
                },
                "strategy": "yaklang_fast_background",
                "timed_out": True,
                "wait_seconds": 0.2,
                "pending": True,
                "metadata": metadata,
            }
        )

    def _local_keyword_memory_recall_enabled(self) -> bool:
        manifest_fn = getattr(self.runner.session.memory, "manifest", None)
        if not callable(manifest_fn):
            return True
        try:
            manifest = manifest_fn()
        except Exception:
            return False
        if str(manifest.get("schema_version") or "") == "agent-core-external-memory-store/v1":
            return False
        return str(manifest.get("backend_kind") or "") in {"in_memory", "sqlite", "markdown"}


class AgentSessionManager:
    """Manage multiple agent sessions and background runs."""

    def __init__(
        self,
        *,
        run_store: AgentRunStorePort | None = None,
        concurrency_policy: AgentManagerConcurrencyPolicy | None = None,
        mark_restored_active_interrupted: bool = True,
    ) -> None:
        self.run_store = run_store or InMemoryAgentRunStore()
        self.concurrency_policy = concurrency_policy or AgentManagerConcurrencyPolicy()
        self._sessions: dict[str, AgentSession] = {}
        self._runs: dict[str, ManagedAgentRun] = {
            run.run_key: _restored_run(run, mark_interrupted=mark_restored_active_interrupted)
            for run in self.run_store.list()
        }
        self._outcomes: dict[str, AgentRunOutcome] = {}
        self._tasks: dict[str, asyncio.Task[AgentRunOutcome]] = {}
        self._active_by_session: dict[str, list[str]] = {}
        for run in self._runs.values():
            self.run_store.save(run)

    def register(self, session: AgentSession, *, name: str | None = None, replace: bool = False) -> str:
        session_name = (name or session.profile.name).strip()
        if not session_name:
            raise ValueError("session name is required")
        if session_name in self._sessions and not replace:
            raise ValueError(f"session already registered: {session_name}")
        self._sessions[session_name] = session
        return session_name

    def unregister(self, name: str) -> bool:
        if self._active_by_session.get(name):
            raise RuntimeError(f"session has active run: {name}")
        return self._sessions.pop(name, None) is not None

    def session(self, name: str) -> AgentSession:
        try:
            return self._sessions[name]
        except KeyError as exc:
            raise KeyError(f"unknown session: {name}") from exc

    def sessions(self) -> tuple[str, ...]:
        return tuple(sorted(self._sessions))

    async def run(self, session_name: str, request: AgentRunRequest | str) -> AgentRunOutcome:
        run_key = self._new_run_key(session_name)
        return await self._run_once(run_key, session_name, self._normalize_request(request))

    def resume_index(self, session_name: str, *, include_terminal: bool = True) -> ResumeIndex:
        return _resume_index_for_harness(self.session(session_name).harness, include_terminal=include_terminal)

    def resume_plan(
        self,
        session_name: str,
        request: AgentResumeRequest | str | None = None,
    ) -> ResumePlan:
        resume_request = _normalize_resume_request(request)
        return _resume_plan_for_request(self.session(session_name).harness, resume_request)

    async def resume(
        self,
        session_name: str,
        request: AgentResumeRequest | str | None = None,
    ) -> AgentRunOutcome:
        resume_request = _normalize_resume_request(request)
        plan = _resume_plan_for_request(self.session(session_name).harness, resume_request)
        candidate = _resume_candidate_from_plan(plan, resume_request)
        return await self.run(session_name, _run_request_from_resume(candidate, resume_request, plan=plan))

    def start(self, session_name: str, request: AgentRunRequest | str) -> str:
        run_request = self._normalize_request(request)
        run_key = self._new_run_key(session_name)
        capacity = self._capacity_status(session_name)
        if not capacity.available:
            if self.concurrency_policy.reject_when_full:
                self._raise_capacity_error(session_name, capacity)
            self._save_run(
                ManagedAgentRun(
                    run_key=run_key,
                    session_name=session_name,
                    task=run_request.task,
                    status="queued",
                    metadata={
                        **dict(run_request.metadata),
                        "queued_for_capacity": True,
                        "capacity_status": capacity.manifest(),
                    },
                )
            )
            task = asyncio.create_task(self._run_when_capacity(run_key, session_name, run_request))
            self._tasks[run_key] = task
            return run_key
        self._save_run(ManagedAgentRun(
            run_key=run_key,
            session_name=session_name,
            task=run_request.task,
            metadata=dict(run_request.metadata),
        ))
        self._claim_run(session_name, run_key)
        task = asyncio.create_task(self._run_once(run_key, session_name, run_request, preclaimed=True))
        self._tasks[run_key] = task
        return run_key

    def start_resume(
        self,
        session_name: str,
        request: AgentResumeRequest | str | None = None,
    ) -> str:
        resume_request = _normalize_resume_request(request)
        plan = _resume_plan_for_request(self.session(session_name).harness, resume_request)
        candidate = _resume_candidate_from_plan(plan, resume_request)
        return self.start(session_name, _run_request_from_resume(candidate, resume_request, plan=plan))

    def cancel(self, run_key: str, reason: str = "cancelled by manager") -> bool:
        run = self._runs.get(run_key)
        if run is None or run.status in {"completed", "cancelled", "timeout", "failed"}:
            return False
        if run.status == "queued" and not self._run_is_claimed(run_key):
            cancelled = _replace_run(run, status="cancelled", error=reason)
            self._save_run(cancelled)
            self._outcomes[run_key] = self._queued_cancelled_outcome(cancelled, reason)
            return True
        session = self._sessions.get(run.session_name)
        if session is not None:
            session.cancel_token.cancel(reason)
        self._save_run(_replace_run(run, status="cancelling", error=reason))
        return True

    async def wait(self, run_key: str) -> AgentRunOutcome:
        task = self._tasks.get(run_key)
        if task is None:
            outcome = self._outcomes.get(run_key)
            if outcome is None:
                raise KeyError(run_key)
            return outcome
        return await task

    def outcome(self, run_key: str) -> AgentRunOutcome | None:
        return self._outcomes.get(run_key)

    def run_state(self, run_key: str) -> ManagedAgentRun:
        try:
            return self._runs[run_key]
        except KeyError as exc:
            raise KeyError(f"unknown run: {run_key}") from exc

    def runs(self) -> tuple[ManagedAgentRun, ...]:
        return tuple(sorted(self._runs.values(), key=lambda item: item.run_key))

    def query_runs(self, query: AgentRunQuery) -> tuple[ManagedAgentRun, ...]:
        return _query_managed_runs(self.runs(), query)

    def active_runs(self) -> tuple[ManagedAgentRun, ...]:
        active_keys = {
            run_key
            for keys in self._active_by_session.values()
            for run_key in keys
        }
        return tuple(run for run in self.runs() if run.run_key in active_keys)

    def capacity_status(self, session_name: str) -> AgentManagerCapacityStatus:
        if session_name not in self._sessions:
            raise KeyError(f"unknown session: {session_name}")
        return self._capacity_status(session_name)

    def schedule_snapshot(self) -> AgentManagerScheduleSnapshot:
        return AgentManagerScheduleSnapshot(
            sessions=self.sessions(),
            runs=self.runs(),
            active_by_session={
                name: tuple(keys)
                for name, keys in sorted(self._active_by_session.items(), key=lambda item: item[0])
            },
            queued_by_session=self._queued_by_session(),
            capacity_by_session=tuple(
                self._capacity_status(name) for name in self.sessions()
            ),
            concurrency_policy=self.concurrency_policy,
            run_store=self.run_store.manifest(),
            metadata={"source": type(self).__name__},
        )

    def event_batch(
        self,
        run_key: str,
        cursor: EventStreamCursor | None = None,
    ) -> EventStreamBatch:
        run = self.run_state(run_key)
        session = self.session(run.session_name)
        log = session.event_sink
        request = cursor or EventStreamCursor(
            run_id=run.result_run_id,
            run_key=run.run_key,
            session_name=run.session_name,
        )
        if not _is_event_log(log):
            return EventStreamBatch(cursor=request)
        return EventStreamBatch.from_log(log, request)

    def event_tail(
        self,
        run_key: str,
        cursor: EventStreamCursor | None = None,
        *,
        max_batches: int = 10,
        stop_at_terminal: bool = True,
    ) -> EventStreamTail:
        run = self.run_state(run_key)
        session = self.session(run.session_name)
        log = session.event_sink
        request = cursor or EventStreamCursor(
            run_id=run.result_run_id,
            run_key=run.run_key,
            session_name=run.session_name,
        )
        if not _is_event_log(log):
            return EventStreamTail(
                start_cursor=request,
                max_batches=max(1, int(max_batches)),
                stop_at_terminal=stop_at_terminal,
            )
        return EventStreamTail.from_log(
            log,
            request,
            max_batches=max_batches,
            stop_at_terminal=stop_at_terminal,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-session-manager/v1",
            "sessions": {
                name: session.manifest()
                for name, session in sorted(self._sessions.items(), key=lambda item: item[0])
            },
            "runs": [run.manifest() for run in self.runs()],
            "active_by_session": {name: list(keys) for name, keys in sorted(self._active_by_session.items())},
            "active_run_count": len(self.active_runs()),
            "queued_by_session": {
                name: list(keys)
                for name, keys in sorted(self._queued_by_session().items(), key=lambda item: item[0])
            },
            "queued_run_count": sum(len(keys) for keys in self._queued_by_session().values()),
            "schedule": self.schedule_snapshot().manifest(),
            "concurrency_policy": self.concurrency_policy.manifest(),
            "run_store": self.run_store.manifest(),
        }

    def _new_run_key(self, session_name: str) -> str:
        if session_name not in self._sessions:
            raise KeyError(f"unknown session: {session_name}")
        return uuid4().hex

    async def _run_once(
        self,
        run_key: str,
        session_name: str,
        request: AgentRunRequest,
        *,
        preclaimed: bool = False,
    ) -> AgentRunOutcome:
        if not preclaimed:
            self._ensure_capacity(session_name)
            self._claim_run(session_name, run_key)
            self._save_run(ManagedAgentRun(
                run_key=run_key,
                session_name=session_name,
                task=request.task,
                status="queued",
                metadata=dict(request.metadata),
            ))
        session = self.session(session_name)
        current_run = self._runs[run_key]
        if session.cancel_token.cancelled and current_run.status != "cancelling":
            session.reset_cancel_token()
        self._save_run(_replace_run(self._runs[run_key], status="running"))
        try:
            runner_session = _session_with_managed_event_sink(
                session,
                run_key=run_key,
                session_name=session_name,
                on_run_id=lambda run_id: self._record_result_run_id(run_key, run_id),
            )
            outcome = await AgentRunner(runner_session).run(request)
        except Exception as exc:
            run = self._runs[run_key]
            self._save_run(_replace_run(run, status="failed", error=str(exc)))
            record_error = getattr(session.harness, "record_error", None)
            if callable(record_error):
                await record_error(error=exc, metadata={"run_key": run_key, "session_name": session_name})
            raise
        finally:
            self._release_run(session_name, run_key)
            self._tasks.pop(run_key, None)

        status = outcome.result.status
        if status == "cancelled":
            managed_status = "cancelled"
        elif status == "completed":
            managed_status = "completed"
        else:
            managed_status = status
        self._save_run(_replace_run(
            self._runs[run_key],
            status=managed_status,
            result_run_id=outcome.result.run_id,
        ))
        self._outcomes[run_key] = outcome
        return outcome

    def _record_result_run_id(self, run_key: str, run_id: str) -> None:
        if not run_id:
            return
        run = self._runs.get(run_key)
        if run is None or run.result_run_id:
            return
        self._save_run(_replace_run(run, result_run_id=run_id))

    async def _run_when_capacity(
        self,
        run_key: str,
        session_name: str,
        request: AgentRunRequest,
    ) -> AgentRunOutcome:
        while True:
            if run_key not in self._runs:
                raise KeyError(run_key)
            run = self._runs[run_key]
            if run.status == "cancelled":
                self._tasks.pop(run_key, None)
                return self._outcomes.get(run_key) or self._queued_cancelled_outcome(
                    run,
                    run.error or "queued run cancelled",
                )
            capacity = self._capacity_status(session_name)
            if capacity.available:
                self._claim_run(session_name, run_key)
                self._save_run(
                    _replace_run(
                        self._runs[run_key],
                        metadata={
                            **dict(self._runs[run_key].metadata),
                            "queued_for_capacity": False,
                            "dequeued_capacity_status": capacity.manifest(),
                        },
                    )
                )
                return await self._run_once(
                    run_key,
                    session_name,
                    request,
                    preclaimed=True,
                )
            await asyncio.sleep(0.01)

    def _ensure_capacity(self, session_name: str) -> None:
        status = self._capacity_status(session_name)
        if not status.available:
            self._raise_capacity_error(session_name, status)

    def _raise_capacity_error(
        self,
        session_name: str,
        status: AgentManagerCapacityStatus,
    ) -> None:
        if not status.session_available:
            raise AgentManagerCapacityError(
                f"session active run capacity exceeded: {session_name}",
                metadata={
                    "session_name": session_name,
                    "active_run_keys": list(status.active_run_keys),
                    "max_active_runs_per_session": status.max_active_runs_per_session,
                    "capacity_status": status.manifest(),
                },
            )
        if not status.manager_available:
            raise AgentManagerCapacityError(
                "manager active run capacity exceeded",
                metadata={
                    "active_run_count": status.manager_active_run_count,
                    "max_active_runs": status.max_active_runs,
                    "capacity_status": status.manifest(),
                },
            )

    def _capacity_status(self, session_name: str) -> AgentManagerCapacityStatus:
        policy = self.concurrency_policy
        active_keys = tuple(self._active_by_session.get(session_name, ()))
        manager_active = len(self.active_runs())
        session_available = len(active_keys) < policy.max_active_runs_per_session
        manager_available = (
            policy.max_active_runs is None
            or manager_active < policy.max_active_runs
        )
        if not session_available:
            reason = "session_capacity_exceeded"
        elif not manager_available:
            reason = "manager_capacity_exceeded"
        else:
            reason = ""
        return AgentManagerCapacityStatus(
            session_name=session_name,
            active_run_keys=active_keys,
            manager_active_run_count=manager_active,
            max_active_runs=policy.max_active_runs,
            max_active_runs_per_session=policy.max_active_runs_per_session,
            session_available=session_available,
            manager_available=manager_available,
            reason=reason,
        )

    def _claim_run(self, session_name: str, run_key: str) -> None:
        self._active_by_session.setdefault(session_name, []).append(run_key)

    def _release_run(self, session_name: str, run_key: str) -> None:
        active = self._active_by_session.get(session_name)
        if not active:
            return
        self._active_by_session[session_name] = [key for key in active if key != run_key]
        if not self._active_by_session[session_name]:
            self._active_by_session.pop(session_name, None)

    def _run_is_claimed(self, run_key: str) -> bool:
        return any(run_key in keys for keys in self._active_by_session.values())

    def _queued_by_session(self) -> dict[str, tuple[str, ...]]:
        queued: dict[str, list[str]] = {}
        active_keys = {
            run_key
            for keys in self._active_by_session.values()
            for run_key in keys
        }
        for run in self.runs():
            if run.status != "queued" or run.run_key in active_keys:
                continue
            queued.setdefault(run.session_name, []).append(run.run_key)
        return {name: tuple(keys) for name, keys in sorted(queued.items())}

    def _queued_cancelled_outcome(self, run: ManagedAgentRun, reason: str) -> AgentRunOutcome:
        session = self._sessions.get(run.session_name)
        return AgentRunOutcome(
            result=ReActResult(
                run_id="",
                status="cancelled",
                output="",
                iterations=0,
                metadata={
                    "interrupt": {
                        "kind": "cancel",
                        "reason": reason,
                        "run_key": run.run_key,
                        "queued": True,
                    }
                },
            ),
            session_manifest=session.manifest() if session is not None else {},
            prompt_manifest={},
            trace_manifest={
                "schema_version": "agent-core-queued-run-cancelled/v1",
                "run": run.manifest(),
                "reason": reason,
            },
        )

    @staticmethod
    def _normalize_request(request: AgentRunRequest | str) -> AgentRunRequest:
        return request if isinstance(request, AgentRunRequest) else AgentRunRequest(task=str(request))

    def _save_run(self, run: ManagedAgentRun) -> None:
        self._runs[run.run_key] = run
        self.run_store.save(run)


def _budget_manifest(budget: RuntimeBudget) -> dict[str, Any]:
    return {
        "max_iterations": budget.max_iterations,
        "max_prompt_bytes": budget.max_prompt_bytes,
        "max_timeline_bytes": budget.max_timeline_bytes,
        "max_tool_result_bytes": budget.max_tool_result_bytes,
        "max_cost_usd": budget.max_cost_usd,
    }


class _ManagedRunEventSink:
    def __init__(
        self,
        base: EventSinkPort,
        *,
        run_key: str,
        session_name: str,
        on_run_id: Any,
    ) -> None:
        self.base = base
        self.run_key = run_key
        self.session_name = session_name
        self.on_run_id = on_run_id

    async def emit(self, event: AgentEvent) -> None:
        if event.run_id:
            self.on_run_id(event.run_id)
        payload = {
            **dict(event.payload),
            "run_key": self.run_key,
            "session_name": self.session_name,
        }
        await self.base.emit(replace(event, payload=payload))

    def records(self, *, run_id: str | None = None) -> tuple[AgentEvent, ...]:
        records = getattr(self.base, "records", None)
        if not callable(records):
            return ()
        return records(run_id=run_id)

    def manifest(self) -> dict[str, Any]:
        base_manifest = _component_manifest_sync(self.base)
        return {
            "schema_version": "agent-core-managed-run-event-sink/v1",
            "run_key": self.run_key,
            "session_name": self.session_name,
            "base": base_manifest,
        }


def _session_with_managed_event_sink(
    session: AgentSession,
    *,
    run_key: str,
    session_name: str,
    on_run_id: Any,
) -> AgentSession:
    if session.event_sink is None:
        return session
    return replace(
        session,
        event_sink=_ManagedRunEventSink(
            session.event_sink,
            run_key=run_key,
            session_name=session_name,
            on_run_id=on_run_id,
        ),
    )


def _is_event_log(value: Any) -> bool:
    return callable(getattr(value, "records", None))


def _prompt_with_budget_plan(prompt: PromptIR, plan: AgentPromptBudgetPlan) -> PromptIR:
    return PromptIR(
        buckets=prompt.buckets,
        metadata={**prompt.metadata, "prompt_budget": plan.manifest()},
    )


def _provider_route_plan_manifest(
    session: AgentSession,
    request: AgentRunRequest,
) -> dict[str, Any]:
    route_plan = getattr(session.provider, "route_plan", None)
    if not callable(route_plan):
        return {}
    route_request = LLMRequest(
        messages=[],
        model=session.profile.model,
        metadata={
            "profile": session.profile.name,
            "provider": str(request.metadata.get("provider") or ""),
            "requires_streaming": request.stream
            if request.stream is not None
            else session.stream,
            **_structured_output_route_metadata(request),
        },
    )
    try:
        value = route_plan(
            route_request,
            streamed=request.stream if request.stream is not None else session.stream,
        )
    except Exception as exc:
        return {
            "schema_version": "agent-core-provider-prompt-budget-route-plan-error/v1",
            "error": str(exc),
        }
    manifest = getattr(value, "manifest", None)
    if callable(manifest):
        result = manifest()
        return dict(result) if isinstance(result, dict) else {}
    return dict(value) if isinstance(value, dict) else {}


def _structured_output_route_metadata(request: AgentRunRequest) -> dict[str, Any]:
    if request.structured_output is None:
        return {}
    return {"requires_structured_output": True}


def _prompt_reserved_output_tokens(request: AgentRunRequest, max_output_tokens: int) -> int:
    for key in ("reserved_output_tokens", "max_output_tokens", "estimated_output_tokens"):
        value = request.metadata.get(key)
        if value is None:
            continue
        try:
            explicit = max(0, int(value))
        except (TypeError, ValueError):
            continue
        if max_output_tokens > 0:
            return min(max_output_tokens, explicit)
        return explicit
    return max(0, int(max_output_tokens))


def _context_reducer_manifest(reducer: ContextReducerPort | None) -> dict[str, Any]:
    if reducer is None:
        return {"enabled": False}
    reducer_manifest = getattr(reducer, "manifest", None)
    if callable(reducer_manifest):
        manifest = reducer_manifest()
        return {"enabled": True, **dict(manifest)}
    return {
        "enabled": True,
        "type": type(reducer).__name__,
    }


def _prompt_bucket_budget_policy_manifest(
    policy: PromptBucketBudgetPolicy | None,
) -> dict[str, Any]:
    if policy is None:
        return {"enabled": False}
    return {"enabled": True, **policy.manifest()}


def _prompt_semantic_reducer_manifest(
    reducer: PromptSemanticReducerPort | None,
) -> dict[str, Any]:
    if reducer is None:
        return {"enabled": False}
    reducer_manifest = getattr(reducer, "manifest", None)
    if callable(reducer_manifest):
        manifest = reducer_manifest()
        return {"enabled": True, **dict(manifest)}
    return {
        "enabled": True,
        "type": type(reducer).__name__,
    }


def _context_material_selector_manifest(
    selector: ContextMaterialSelectorPort | None,
) -> dict[str, Any]:
    if selector is None:
        return {"enabled": False}
    selector_manifest = getattr(selector, "manifest", None)
    if callable(selector_manifest):
        manifest = selector_manifest()
        return {"enabled": True, **dict(manifest)}
    return {
        "enabled": True,
        "type": type(selector).__name__,
    }


def _context_material_selection_request(
    request: AgentRunRequest,
    *,
    extra_materials: tuple[ContextMaterial, ...] = (),
    extra_metadata: dict[str, Any] | None = None,
    task_override: str = "",
) -> ContextMaterialSelectionRequest:
    base = request.context_material_selection
    materials = (*tuple(request.context_materials), *tuple(extra_materials))
    task = str(task_override or request.task)
    metadata = {
        "request_metadata": dict(request.metadata),
        **dict(extra_metadata or {}),
    }
    if base is None:
        return ContextMaterialSelectionRequest(
            task=task,
            materials=materials,
            metadata=metadata,
        )
    return ContextMaterialSelectionRequest(
        task=task_override or base.task or request.task,
        materials=(*tuple(base.materials or request.context_materials), *tuple(extra_materials)),
        max_materials=base.max_materials,
        max_bytes=base.max_bytes,
        allowed_targets=tuple(base.allowed_targets),
        min_score=base.min_score,
        metadata={
            **base.metadata,
            **metadata,
        },
    )


def _available_skill_names(skills: SkillsContext | None) -> tuple[str, ...]:
    if skills is None:
        return ()
    names = {skill.name for skill in skills.loaded()}
    registry = getattr(skills, "registry", None)
    listing = getattr(registry, "list", None)
    if callable(listing):
        names.update(skill.name for skill in listing())
    return tuple(sorted(names))


def _available_mcp_server_names(mcp: MCPCenter | None) -> tuple[str, ...]:
    if mcp is None:
        return ()
    return tuple(server.name for server in mcp.servers())


def _approval_resume_manifest(resume: ApprovalResumeContext | None) -> dict[str, Any]:
    if resume is None or resume.empty:
        return {}
    return resume.manifest()


def _normalize_resume_request(request: AgentResumeRequest | str | None) -> AgentResumeRequest:
    if request is None:
        return AgentResumeRequest()
    if isinstance(request, AgentResumeRequest):
        return request
    return AgentResumeRequest(task=str(request))


def _resume_index_for_harness(harness: AgentHarness, *, include_terminal: bool = True) -> ResumeIndex:
    resume_index = getattr(harness, "resume_index", None)
    if callable(resume_index):
        value = resume_index(include_terminal=include_terminal)
        if isinstance(value, ResumeIndex):
            return value
    snapshot = getattr(harness, "snapshot", None)
    if callable(snapshot):
        return ResumeIndex.from_snapshot(snapshot(), include_terminal=include_terminal)
    raise ResumeError("harness does not expose resumable state")


def _resume_candidate_for_request(
    harness: AgentHarness,
    request: AgentResumeRequest,
) -> ResumeCandidate:
    plan = _resume_plan_for_request(harness, request)
    return _resume_candidate_from_plan(plan, request)


def _resume_plan_for_request(
    harness: AgentHarness,
    request: AgentResumeRequest,
) -> ResumePlan:
    index = _resume_index_for_harness(harness, include_terminal=request.include_terminal)
    return ResumePlan.from_index(index, run_id=request.run_id, request=request.manifest())


def _resume_candidate_from_plan(
    plan: ResumePlan,
    request: AgentResumeRequest,
) -> ResumeCandidate:
    candidate = plan.candidate
    if candidate is None:
        if request.run_id:
            raise ResumeError(f"no resumable checkpoint for run: {request.run_id}")
        raise ResumeError("no resumable checkpoint available")
    if candidate.token is None:
        raise ResumeError(f"resume candidate has no token: {candidate.run_id}")
    if not plan.ready:
        issues = ", ".join(issue.code for issue in plan.issues if issue.severity == "error")
        raise ResumeError(f"resume plan is not ready: {issues or 'unknown'}")
    return candidate


def _run_request_from_resume(
    candidate: ResumeCandidate,
    request: AgentResumeRequest,
    *,
    plan: ResumePlan | None = None,
) -> AgentRunRequest:
    if candidate.token is None:
        raise ResumeError(f"resume candidate has no token: {candidate.run_id}")
    metadata = {
        **request.metadata,
        "resume": {
            "source_run_id": candidate.run_id,
            "checkpoint_id": candidate.checkpoint_id,
            "checkpoint_sequence": candidate.checkpoint_sequence,
            "terminal": candidate.terminal,
            "auto_selected": not bool(request.run_id),
        },
    }
    if plan is not None:
        metadata["resume_plan"] = plan.summary_manifest()
    task = request.task or candidate.task
    return AgentRunRequest(
        task=task,
        context=request.context,
        metadata=metadata,
        refresh=request.refresh,
        resume_token=candidate.token,
        approval_resume=request.approval_resume,
        structured_output=request.structured_output,
        context_materials=request.context_materials,
        context_material_query=request.context_material_query,
        context_material_selection=request.context_material_selection,
        mcp_context_materials=request.mcp_context_materials,
        preflight_requirements=request.preflight_requirements,
        task_contract=request.task_contract,
        timeout_seconds=request.timeout_seconds,
        native_tool_calls=request.native_tool_calls,
        stream=request.stream,
    )


def _request_resume_plan_manifest(request: AgentRunRequest) -> dict[str, Any]:
    value = request.metadata.get("resume_plan")
    return dict(value) if isinstance(value, dict) else {}


def _preflight_requirements_for_request(
    request: AgentRunRequest,
) -> AgentRunPreflightRequirements:
    if request.preflight_requirements is not None:
        return request.preflight_requirements
    if request.task_contract is not None:
        return request.task_contract.preflight_requirements(
            metadata={"request_metadata": dict(request.metadata)}
        )
    return AgentRunPreflightRequirements()


def _task_contract_manifest(request: AgentRunRequest) -> dict[str, Any]:
    if request.task_contract is None:
        return {}
    return request.task_contract.manifest()


async def _component_manifest(component: Any) -> dict[str, Any]:
    if component is None:
        return {}
    manifest = getattr(component, "manifest", None)
    if not callable(manifest):
        return {}
    value = manifest()
    if inspect.isawaitable(value):
        value = await value
    return dict(value) if isinstance(value, dict) else {}


def _component_manifest_sync(component: Any) -> dict[str, Any]:
    manifest = getattr(component, "manifest", None)
    if not callable(manifest) or inspect.iscoroutinefunction(manifest):
        return {}
    value = manifest()
    return dict(value) if isinstance(value, dict) else {}


def _resume_context_block(manifest: dict[str, Any]) -> str:
    state = manifest.get("state") if isinstance(manifest.get("state"), dict) else {}
    state_lines = []
    for key, value in sorted(state.items(), key=lambda item: str(item[0])):
        state_lines.append(f"- {key}: {value}")
    rendered_state = "\n".join(state_lines) if state_lines else "- empty"
    return (
        "== Resumed Checkpoint ==\n"
        f"run_id: {manifest.get('run_id')}\n"
        f"turn_id: {manifest.get('turn_id')}\n"
        f"checkpoint_id: {manifest.get('checkpoint_id')}\n"
        f"sequence: {manifest.get('sequence')}\n"
        "state:\n"
        f"{rendered_state}"
    )


def _approval_resume_context_block(manifest: dict[str, Any]) -> str:
    lines = [
        "== Approval Resume ==",
        f"grant_count: {manifest.get('grant_count')}",
        f"approved_count: {manifest.get('approved_count')}",
    ]
    for grant in manifest.get("grants") or ():
        if not isinstance(grant, dict):
            continue
        lines.append(
            "- "
            f"{grant.get('subject')} "
            f"status={grant.get('status')} "
            f"approval_id={grant.get('approval_id')} "
            f"actor={grant.get('actor')}"
        )
    return "\n".join(lines)


def _memory_context_block(hits: tuple[MemoryHit, ...]) -> str:
    lines = []
    for hit in hits:
        source = hit.source or "memory"
        lines.append(f"- {source} ({hit.score:.3f}): {hit.content}")
    return "[memory]\n" + "\n".join(lines)


def _memory_search_plan_manifest(memory: MemoryPort, query: MemoryQuery) -> dict[str, Any]:
    plan_search = getattr(memory, "plan_search", None)
    if not callable(plan_search):
        return {
            "schema_version": "agent-core-memory-search-plan-unavailable/v1",
            "query": query.manifest(),
            "memory_type": type(memory).__name__,
        }
    plan = plan_search(query)
    manifest = getattr(plan, "manifest", None)
    if callable(manifest):
        value = manifest()
        return dict(value) if isinstance(value, dict) else {}
    return {}


def _memory_hit_manifest(hit: MemoryHit) -> dict[str, Any]:
    content_bytes = len(hit.content.encode("utf-8"))
    return {
        "source": hit.source,
        "score": hit.score,
        "content_bytes": content_bytes,
        "content_sha256": hashlib.sha256(hit.content.encode("utf-8")).hexdigest()
        if hit.content
        else "",
        "metadata": dict(hit.metadata),
    }


def _consume_perception_downstream_refresh(perception: Any) -> dict[str, Any]:
    consume = getattr(perception, "consume_downstream_refresh", None)
    if not callable(consume):
        return {}
    refresh = consume()
    return dict(refresh) if isinstance(refresh, dict) else {}


def _consume_perception_observation(perception: Any) -> dict[str, Any]:
    consume = getattr(perception, "consume_observation", None)
    if not callable(consume):
        return {}
    observation = consume()
    return dict(observation) if isinstance(observation, dict) else {}


def _render_perception_observation(observation: dict[str, Any]) -> str:
    state = observation.get("state")
    if not isinstance(state, dict):
        return ""
    summary = str(state.get("summary") or "").strip()
    topics = _string_sequence(state.get("topics"))
    keywords = _string_sequence(state.get("keywords"))
    epoch = state.get("epoch")
    trigger = str(observation.get("trigger") or state.get("last_trigger") or "").strip()
    parts: list[str] = []
    prefix = "Perception"
    if epoch not in (None, ""):
        prefix += f" epoch={epoch}"
    if trigger:
        prefix += f" trigger={trigger}"
    if summary:
        parts.append(f"{prefix}: {summary}")
    else:
        parts.append(prefix)
    if topics:
        parts.append("topics=" + ", ".join(topics))
    if keywords:
        parts.append("keywords=" + ", ".join(keywords))
    intent_shift = str(state.get("intent_shift") or "").strip()
    if intent_shift:
        parts.append(f"intent_shift={intent_shift}")
    return " | ".join(parts).strip()


def _string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    try:
        return tuple(str(item).strip() for item in value or () if str(item).strip())
    except TypeError:
        return ()


def _perception_context_injections(perception: Any) -> tuple[ContextInjection, ...]:
    context_injections = getattr(perception, "context_injections", None)
    if not callable(context_injections):
        return ()
    injections = context_injections()
    try:
        return tuple(item for item in injections or () if isinstance(item, ContextInjection))
    except TypeError:
        return ()


def _capability_context_injections(capability: Any) -> tuple[ContextInjection, ...]:
    context_injections = getattr(capability, "context_injections", None)
    if not callable(context_injections):
        return ()
    injections = context_injections()
    try:
        return tuple(item for item in injections or () if isinstance(item, ContextInjection))
    except TypeError:
        return ()


def _perception_downstream_intents(downstream_refresh: dict[str, Any]) -> tuple[str, ...]:
    values = downstream_refresh.get("intents") if isinstance(downstream_refresh, dict) else ()
    intents = _string_sequence(values)
    if intents:
        return intents
    intent = downstream_refresh.get("intent") if isinstance(downstream_refresh, dict) else {}
    if isinstance(intent, dict):
        intents = _string_sequence(intent.get("intents"))
    return intents


def _intent_enabled(intents: tuple[str, ...], intent: str, *, default: bool) -> bool:
    if not intents:
        return default
    return intent in set(intents)


def _metadata_sequence(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _recent_tools_from_action_history(history: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    tools: list[str] = []
    for item in history:
        if item.get("ok") is False:
            continue
        tool_name = str(item.get("tool_name") or "").strip()
        if tool_name:
            tools.append(tool_name)
    return tuple(dict.fromkeys(reversed(tools)))


def _loaded_capabilities_from_action_history(
    history: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    loaded: list[dict[str, Any]] = []
    for item in history:
        if item.get("ok") is False:
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        result_metadata = metadata.get("tool_result_metadata")
        if not isinstance(result_metadata, dict):
            continue
        load = result_metadata.get("load_capability")
        if not isinstance(load, dict):
            continue
        match = load.get("match")
        if isinstance(match, dict) and match.get("name"):
            loaded.append(dict(match))
    deduped: dict[str, dict[str, Any]] = {}
    for item in loaded:
        key = f"{item.get('kind')}:{item.get('name')}"
        deduped[key] = item
    return tuple(deduped.values())


def _recent_tools_cache_context_injections(
    history: tuple[dict[str, Any], ...],
) -> tuple[ContextInjection, ...]:
    recent = _recent_tools_from_action_history(history)
    if not recent:
        return ()
    failed = set(_failed_tools_from_action_history(history))
    lines = [
        "[recent_tools_cache]",
        "Fast Tool Routing:",
        "- Prefer call_tool for an exact recent tool when it fits the current task.",
        "- Use search_capabilities or load_capability when the needed tool is not listed.",
        "Recent tools:",
    ]
    for name in recent[:8]:
        suffix = " status=recent_failed" if name in failed else " status=recent_ok"
        lines.append(f"- {name}{suffix}")
    content = "\n".join(lines)
    return (
        ContextInjection(
            name="recent_tools_cache",
            content=content,
            target=PromptBucketRole.SEMI_DYNAMIC_1,
            source="capability",
            priority=77,
            metadata={
                "schema_version": "agent-core-recent-tools-cache/v1",
                "recent_tools": list(recent[:8]),
                "failed_tools": list(failed),
            },
        ),
    )


def _failed_tools_from_action_history(history: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    tools: list[str] = []
    for item in history:
        if item.get("ok") is not False:
            continue
        tool_name = str(item.get("tool_name") or "").strip()
        if tool_name:
            tools.append(tool_name)
    return tuple(dict.fromkeys(reversed(tools)))


def _task_state_context_injections(
    run_request: AgentRunRequest,
    *,
    request: TurnRefreshRequest,
    timeline_diff: Any,
    context_material_selection: dict[str, Any],
    perception_state: dict[str, Any],
    memory_flush: dict[str, Any],
) -> tuple[ContextInjection, ...]:
    task_metadata = dict(run_request.metadata.get("task_state") or {})
    action_history = _metadata_sequence(request.metadata.get("action_history"))
    timeline_manifest = (
        timeline_diff.manifest()
        if timeline_diff is not None and getattr(timeline_diff, "changed", False)
        else {}
    )
    frame = LoopStateFrame(
        task=request.task,
        iteration=request.iteration,
        run_id=request.run_id,
        turn_id=request.turn_id,
        root_task=str(task_metadata.get("root_task", run_request.metadata.get("root_task") or "")),
        parent_task=str(task_metadata.get("parent_task", run_request.metadata.get("parent_task") or "")),
        current_task=str(task_metadata.get("current_task", run_request.metadata.get("current_task") or "")),
        plan_context=str(task_metadata.get("plan_context", run_request.metadata.get("plan_context") or "")),
        current_objective=str(
            task_metadata.get("current_objective", run_request.metadata.get("current_objective") or "")
        ),
        next_movement=str(task_metadata.get("next_movement", run_request.metadata.get("next_movement") or "")),
        dynamic_feedback=str(
            task_metadata.get("dynamic_feedback", run_request.metadata.get("dynamic_feedback") or "")
        ),
        repair_context=str(task_metadata.get("repair_context", run_request.metadata.get("repair_context") or "")),
        approval_context=str(
            task_metadata.get("approval_context", run_request.metadata.get("approval_context") or "")
        ),
        verification_state=str(
            task_metadata.get("verification_state", run_request.metadata.get("verification_state") or "")
        ),
        action_history=action_history,
        failed_tools=_failed_tools_from_action_history(action_history),
        perception_state=perception_state,
        memory_flush=memory_flush,
        timeline_diff=timeline_manifest,
        context_material_selection=context_material_selection,
        metadata={"request_metadata": dict(run_request.metadata)},
    )
    content = frame.render_prompt()
    if not content:
        return ()
    manifest = frame.manifest()
    return (
        ContextInjection(
            name="task_state_frame",
            content="[task_state_frame]\n" + content,
            target=PromptBucketRole.TIMELINE_OPEN,
            source="runtime",
            priority=70,
            metadata={
                "schema_version": "agent-core-task-state-frame/v1",
                "iteration": request.iteration,
                "turn_id": request.turn_id,
                "has_plan_context": bool(frame.plan_context),
                "action_history_count": len(action_history),
                "loop_state": manifest,
            },
        ),
    )


def _perception_capability_query(
    *,
    downstream_refresh: dict[str, Any],
    fallback: str,
) -> str:
    state = downstream_refresh.get("state") if isinstance(downstream_refresh, dict) else {}
    if not isinstance(state, dict):
        return fallback
    parts: list[str] = []
    summary = str(state.get("summary") or "").strip()
    if summary:
        parts.append(summary)
    for key in ("topics", "keywords"):
        values = state.get(key)
        if isinstance(values, str):
            parts.append(values)
            continue
        try:
            parts.extend(str(item) for item in values or () if str(item).strip())
        except TypeError:
            continue
    return " ".join(parts).strip() or fallback


def _perception_recall_query(
    *,
    downstream_refresh: dict[str, Any],
    fallback: str,
) -> str:
    state = downstream_refresh.get("state") if isinstance(downstream_refresh, dict) else {}
    if not isinstance(state, dict):
        return fallback
    parts: list[str] = []
    summary = str(state.get("summary") or "").strip()
    if summary:
        parts.append(summary)
    topics = _string_sequence(state.get("topics"))
    if topics:
        parts.append("Topics: " + ", ".join(topics))
    keywords = _string_sequence(state.get("keywords"))
    if keywords:
        parts.append("Keywords: " + ", ".join(keywords))
    return "\n".join(parts).strip() or fallback


def _perception_context_material_query(
    *,
    downstream_refresh: dict[str, Any],
    fallback: ContextMaterialQuery | None,
) -> ContextMaterialQuery | None:
    query_text = _perception_recall_query(downstream_refresh=downstream_refresh, fallback="")
    if not query_text:
        return fallback
    filters = {
        **(dict(fallback.filters) if fallback is not None else {}),
    }
    if fallback is not None:
        return replace(
            fallback,
            query=query_text,
            filters=filters,
            mode=fallback.mode or "hybrid",
        )
    return ContextMaterialQuery(
        query=query_text,
        limit=8,
        mode="hybrid",
        filters=filters,
    )


def _perception_query_metadata(downstream_refresh: dict[str, Any]) -> dict[str, Any]:
    if not downstream_refresh:
        return {}
    state = downstream_refresh.get("state") if isinstance(downstream_refresh, dict) else {}
    state_manifest = dict(state) if isinstance(state, dict) else {}
    metadata: dict[str, Any] = {
        "query_source": "perception_downstream_refresh",
        "perception_trigger": downstream_refresh.get("trigger"),
        "perception_reason": downstream_refresh.get("reason"),
    }
    if state_manifest:
        metadata["perception_epoch"] = state_manifest.get("epoch")
        metadata["perception_topics"] = list(_string_sequence(state_manifest.get("topics")))
        metadata["perception_keywords"] = list(_string_sequence(state_manifest.get("keywords")))
        metadata["perception_summary"] = str(state_manifest.get("summary") or "")
    return metadata


def _append_context_block(existing: str, block: str) -> str:
    if not existing:
        return block
    return existing.rstrip() + "\n\n" + block


_ACTIVE_MANAGED_RUN_STATUSES = {"queued", "running", "cancelling"}
_RUN_MARKDOWN_RE = re.compile(r"<!-- agent-core-run (?P<payload>[A-Za-z0-9+/=]+) -->")


def _restored_run(run: ManagedAgentRun, *, mark_interrupted: bool) -> ManagedAgentRun:
    if not mark_interrupted or run.status not in _ACTIVE_MANAGED_RUN_STATUSES:
        return run
    reason = (
        "queued run was restored without an executable request"
        if run.status == "queued"
        else "run was active when manager state was restored"
    )
    return _replace_run(
        run,
        status="interrupted",
        error=run.error or reason,
        metadata={
            **dict(run.metadata),
            "restored": True,
            "restored_from_status": run.status,
            "restore_action": "marked_interrupted",
            "restore_reason": reason,
            "restored_queued_for_capacity": bool(run.metadata.get("queued_for_capacity")),
        },
    )


def _managed_run_from_row(row: tuple[Any, ...]) -> ManagedAgentRun:
    metadata = json.loads(str(row[6] or "{}"))
    return ManagedAgentRun(
        run_key=str(row[0] or ""),
        session_name=str(row[1] or ""),
        task=str(row[2] or ""),
        status=str(row[3] or "queued"),
        result_run_id=str(row[4] or ""),
        error=str(row[5] or ""),
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _managed_run_from_payload(payload: dict[str, Any]) -> ManagedAgentRun:
    metadata = payload.get("metadata")
    return ManagedAgentRun(
        run_key=str(payload.get("run_key") or ""),
        session_name=str(payload.get("session_name") or ""),
        task=str(payload.get("task") or ""),
        status=str(payload.get("status") or "queued"),
        result_run_id=str(payload.get("result_run_id") or ""),
        error=str(payload.get("error") or ""),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def _query_managed_runs(
    runs: tuple[ManagedAgentRun, ...],
    query: AgentRunQuery,
) -> tuple[ManagedAgentRun, ...]:
    results = []
    for run in runs:
        if query.run_keys and run.run_key not in query.run_keys:
            continue
        if query.session_names and run.session_name not in query.session_names:
            continue
        if query.statuses and run.status not in query.statuses:
            continue
        if not _run_metadata_matches(run.metadata, query.metadata):
            continue
        results.append(run)
    results = sorted(results, key=lambda item: item.run_key, reverse=query.reverse)
    if query.limit is not None:
        results = results[: query.limit]
    return tuple(results)


def _run_metadata_matches(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if metadata.get(key) != value:
            return False
    return True


def _encode_run_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _decode_run_payload(payload: str) -> dict[str, Any]:
    raw = base64.b64decode(payload.encode("ascii")).decode("utf-8")
    data = json.loads(raw)
    return dict(data) if isinstance(data, dict) else {}


def _replace_run(
    run: ManagedAgentRun,
    *,
    status: str | None = None,
    result_run_id: str | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ManagedAgentRun:
    return ManagedAgentRun(
        run_key=run.run_key,
        session_name=run.session_name,
        task=run.task,
        status=status if status is not None else run.status,
        result_run_id=result_run_id if result_run_id is not None else run.result_run_id,
        error=error if error is not None else run.error,
        metadata=dict(metadata) if metadata is not None else dict(run.metadata),
    )

