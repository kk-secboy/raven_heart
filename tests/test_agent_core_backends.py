from __future__ import annotations

import agent_core
from agent_core import (
    InMemoryAgentRunStore,
    InMemoryApprovalStore,
    InMemoryArtifactStore,
    InMemoryJournalStore,
    InMemoryMemoryStore,
    InMemoryPlannerStore,
    InMemoryPolicyDecisionStore,
    InMemoryRunTraceStore,
    InMemoryToolReplayStore,
    ListEventSink,
    MarkdownAgentRunStore,
    MarkdownApprovalStore,
    MarkdownArtifactStore,
    MarkdownEventSink,
    MarkdownJournalStore,
    MarkdownMemoryStore,
    MarkdownPlannerStore,
    MarkdownPolicyDecisionStore,
    MarkdownRunTraceStore,
    MarkdownToolReplayStore,
    NullApprovalStore,
    NullPolicyDecisionStore,
    NullRunTraceStore,
    SQLiteAgentRunStore,
    SQLiteApprovalStore,
    SQLiteArtifactStore,
    SQLiteEventSink,
    SQLiteJournalStore,
    SQLiteMemoryStore,
    SQLitePlannerStore,
    SQLitePolicyDecisionStore,
    SQLiteRunTraceStore,
    SQLiteToolReplayStore,
    StorageBackendSpec,
    storage_backend_manifest,
)


def test_storage_backend_contract_exports_runtime_backend_shape() -> None:
    manifest = storage_backend_manifest(
        role="memory",
        kind="postgres",
        name="tenant-memory",
        namespace="tenant-a",
        core_builtin=False,
        capabilities=("keyword", "semantic", "vector"),
        metadata={"driver": "runtime-owned"},
    )

    assert manifest == StorageBackendSpec(
        role="memory",
        kind="postgres",
        name="tenant-memory",
        namespace="tenant-a",
        durable=True,
        inspectable=False,
        queryable=True,
        transactional=True,
        core_builtin=False,
        capabilities=("keyword", "semantic", "vector"),
        metadata={"driver": "runtime-owned"},
    ).manifest()
    assert manifest["schema_version"] == "agent-core-storage-backend/v1"
    assert manifest["kind"] == "postgres"


def test_core_store_manifests_include_unified_backend_metadata(tmp_path) -> None:
    stores = (
        (InMemoryMemoryStore(), "memory", "in_memory"),
        (SQLiteMemoryStore(tmp_path / "memory.sqlite"), "memory", "sqlite"),
        (MarkdownMemoryStore(tmp_path / "memory-md"), "memory", "markdown"),
        (InMemoryJournalStore(), "journal", "in_memory"),
        (SQLiteJournalStore(tmp_path / "journal.sqlite"), "journal", "sqlite"),
        (MarkdownJournalStore(tmp_path / "journal.md"), "journal", "markdown"),
        (NullRunTraceStore(), "run_trace", "none"),
        (InMemoryRunTraceStore(), "run_trace", "in_memory"),
        (SQLiteRunTraceStore(tmp_path / "trace.sqlite"), "run_trace", "sqlite"),
        (MarkdownRunTraceStore(tmp_path / "trace.md"), "run_trace", "markdown"),
        (InMemoryAgentRunStore(), "run_state", "in_memory"),
        (SQLiteAgentRunStore(tmp_path / "runs.sqlite"), "run_state", "sqlite"),
        (MarkdownAgentRunStore(tmp_path / "runs.md"), "run_state", "markdown"),
        (InMemoryToolReplayStore(), "tool_replay", "in_memory"),
        (SQLiteToolReplayStore(tmp_path / "tools.sqlite"), "tool_replay", "sqlite"),
        (MarkdownToolReplayStore(tmp_path / "tools.md"), "tool_replay", "markdown"),
        (NullApprovalStore(), "approval", "none"),
        (InMemoryApprovalStore(), "approval", "in_memory"),
        (SQLiteApprovalStore(tmp_path / "approvals.sqlite"), "approval", "sqlite"),
        (MarkdownApprovalStore(tmp_path / "approvals.md"), "approval", "markdown"),
        (NullPolicyDecisionStore(), "policy_decision", "none"),
        (InMemoryPolicyDecisionStore(), "policy_decision", "in_memory"),
        (SQLitePolicyDecisionStore(tmp_path / "policy.sqlite"), "policy_decision", "sqlite"),
        (MarkdownPolicyDecisionStore(tmp_path / "policy.md"), "policy_decision", "markdown"),
        (InMemoryPlannerStore(), "planner_state", "in_memory"),
        (SQLitePlannerStore(tmp_path / "plans.sqlite"), "planner_state", "sqlite"),
        (MarkdownPlannerStore(tmp_path / "plans.md"), "planner_state", "markdown"),
        (ListEventSink(), "event_log", "in_memory"),
        (SQLiteEventSink(tmp_path / "events.sqlite"), "event_log", "sqlite"),
        (MarkdownEventSink(tmp_path / "events.md"), "event_log", "markdown"),
        (InMemoryArtifactStore(), "artifact", "in_memory"),
        (SQLiteArtifactStore(tmp_path / "artifacts.sqlite"), "artifact", "sqlite"),
        (MarkdownArtifactStore(tmp_path / "artifacts.md"), "artifact", "markdown"),
    )

    for store, role, kind in stores:
        backend = store.manifest()["backend"]
        assert backend["schema_version"] == "agent-core-storage-backend/v1"
        assert backend["role"] == role
        assert backend["kind"] == kind
        assert backend["core_builtin"] is True


def test_agent_core_package_exports_storage_backend_contracts() -> None:
    assert agent_core.StorageBackendSpec is StorageBackendSpec
    assert agent_core.storage_backend_manifest is storage_backend_manifest
    assert "StorageBackendKind" in agent_core.__all__
