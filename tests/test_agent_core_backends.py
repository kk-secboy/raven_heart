from __future__ import annotations

import agent_core
from agent_core import (
    ContextMaterialCenter,
    ExternalContextMaterialStore,
    ExternalMemoryStore,
    InMemoryAgentRunStore,
    InMemoryApprovalStore,
    InMemoryArtifactStore,
    InMemoryContextMaterialStore,
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
    MarkdownContextMaterialStore,
    MarkdownEventSink,
    MarkdownJournalStore,
    MarkdownMemoryStore,
    MarkdownPlannerStore,
    MarkdownPolicyDecisionStore,
    MarkdownRunTraceStore,
    MarkdownToolReplayStore,
    MemoryCenter,
    NullApprovalStore,
    NullPolicyDecisionStore,
    NullRunTraceStore,
    SQLiteAgentRunStore,
    SQLiteApprovalStore,
    SQLiteArtifactStore,
    SQLiteContextMaterialStore,
    SQLiteEventSink,
    SQLiteJournalStore,
    SQLiteMemoryStore,
    SQLitePlannerStore,
    SQLitePolicyDecisionStore,
    SQLiteRunTraceStore,
    SQLiteToolReplayStore,
    StorageBackendCatalog,
    StorageBackendPreflightReport,
    StorageBackendRequirement,
    StorageBackendSpec,
    storage_backend_catalog_from_components,
    storage_backend_manifests_from_components,
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
        (InMemoryContextMaterialStore(), "context_material", "in_memory"),
        (SQLiteContextMaterialStore(tmp_path / "context.sqlite"), "context_material", "sqlite"),
        (MarkdownContextMaterialStore(tmp_path / "context-md"), "context_material", "markdown"),
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


def test_storage_backend_catalog_selects_runtime_owned_backends_without_adapters() -> None:
    catalog = StorageBackendCatalog(
        (
            storage_backend_manifest(
                role="memory",
                kind="sqlite",
                name="local-memory",
                capabilities=("keyword",),
            ),
            storage_backend_manifest(
                role="memory",
                kind="postgres",
                name="tenant-memory",
                namespace="tenant-a",
                core_builtin=False,
                capabilities=("keyword", "semantic", "vector"),
                location="postgres://runtime-owned",
            ),
            storage_backend_manifest(
                role="journal",
                kind="markdown",
                name="local-journal",
            ),
            {
                "role": "artifact",
                "kind": "object_storage",
                "name": "runtime-artifacts",
                "core_builtin": False,
            },
        )
    )

    selection = catalog.select(
        StorageBackendRequirement(
            role="memory",
            allowed_kinds=("postgres", "vector"),
            required_capabilities=("semantic", "vector"),
            namespace="tenant-a",
            require_durable=True,
            require_queryable=True,
        )
    )
    manifest = selection.manifest()

    assert selection.ready is True
    assert selection.selected is not None
    assert selection.selected.backend.name == "tenant-memory"
    assert manifest["schema_version"] == "agent-core-storage-backend-selection/v1"
    assert manifest["selected"]["backend"]["core_builtin"] is False
    assert manifest["selected"]["backend"]["kind"] == "postgres"
    assert manifest["candidate_count"] == 2
    assert catalog.manifest()["external_backend_count"] == 2
    assert catalog.manifest()["roles"]["memory"] == 2
    artifacts = catalog.select(role="artifact", allowed_kinds=("object_storage",), require_durable=True)
    assert artifacts.ready is True
    assert artifacts.selected is not None
    assert artifacts.selected.backend.durable is True


def test_storage_backend_catalog_records_missing_selection_reasons() -> None:
    catalog = StorageBackendCatalog(
        (
            storage_backend_manifest(
                role="memory",
                kind="sqlite",
                name="local-memory",
                capabilities=("keyword",),
            ),
            storage_backend_manifest(
                role="memory",
                kind="postgres",
                name="external-memory",
                core_builtin=False,
                capabilities=("keyword", "semantic"),
            ),
        )
    )

    selection = catalog.select(
        role="memory",
        allowed_kinds=("postgres",),
        required_capabilities=("graph",),
        allow_external=False,
    )
    manifest = selection.manifest()

    assert selection.ready is False
    assert manifest["status"] == "missing"
    assert manifest["selected"] == {}
    reasons = {candidate["backend"]["name"]: candidate["reason"] for candidate in manifest["candidates"]}
    assert "kind_not_allowed" in reasons["local-memory"]
    assert "missing_capabilities" in reasons["external-memory"]
    assert "external_backend_not_allowed" in reasons["external-memory"]
    external = next(candidate for candidate in manifest["candidates"] if candidate["backend"]["name"] == "external-memory")
    assert external["missing_capabilities"] == ["graph"]


def test_storage_backend_catalog_preflights_multi_role_runtime_backends() -> None:
    catalog = StorageBackendCatalog(
        (
            storage_backend_manifest(
                role="memory",
                kind="postgres",
                name="tenant-memory",
                namespace="tenant-a",
                core_builtin=False,
                capabilities=("keyword", "semantic", "vector"),
            ),
            storage_backend_manifest(
                role="context_material",
                kind="vector",
                name="tenant-context",
                namespace="tenant-a",
                core_builtin=False,
                capabilities=("semantic", "hybrid"),
            ),
            storage_backend_manifest(
                role="run_trace",
                kind="sqlite",
                name="local-traces",
            ),
        )
    )

    report = catalog.preflight(
        (
            StorageBackendRequirement(
                role="memory",
                allowed_kinds=("postgres", "vector"),
                required_capabilities=("semantic", "vector"),
                namespace="tenant-a",
                require_durable=True,
                require_queryable=True,
            ),
            StorageBackendRequirement(
                role="context_material",
                allowed_kinds=("vector", "graph"),
                required_capabilities=("semantic",),
                namespace="tenant-a",
            ),
            StorageBackendRequirement(
                role="run_trace",
                allowed_kinds=("postgres",),
                require_durable=True,
            ),
        ),
        metadata={"tenant": "tenant-a"},
    )
    manifest = report.manifest()

    assert isinstance(report, StorageBackendPreflightReport)
    assert report.ready is False
    assert manifest["schema_version"] == "agent-core-storage-backend-preflight/v1"
    assert manifest["status"] == "blocked"
    assert manifest["selection_count"] == 2
    assert manifest["blocking_count"] == 1
    assert manifest["selected_roles"] == ["memory", "context_material"]
    assert manifest["selected_kinds"] == ["postgres", "vector"]
    assert manifest["missing_roles"] == ["run_trace"]
    assert "kind_not_allowed" in manifest["blocking_reasons"]
    assert manifest["metadata"]["tenant"] == "tenant-a"


def test_storage_backend_catalog_builds_from_external_store_component_manifests() -> None:
    memory = ExternalMemoryStore(
        InMemoryMemoryStore(),
        name="tenant-memory",
        backend_kind="postgres",
        namespaces=("tenant-a",),
        supports_vector=True,
        supports_graph=True,
        location="postgres://runtime-owned",
    )
    memory_center = MemoryCenter(default_store="tenant-memory")
    memory_center.register_spec(memory.spec, memory)
    context = ExternalContextMaterialStore(
        InMemoryContextMaterialStore(),
        name="tenant-context",
        backend_kind="vector",
        namespace="tenant-a",
        supports_vector=True,
        location="vector://runtime-owned",
    )
    context_center = ContextMaterialCenter(default_store="tenant-context")
    context_center.register_spec(context.spec, context)

    manifests = storage_backend_manifests_from_components(
        memory_center.manifest(),
        context_center.manifest(),
    )
    catalog = storage_backend_catalog_from_components(
        memory_center.manifest(),
        context_center.manifest(),
    )
    preflight = catalog.preflight(
        (
            StorageBackendRequirement(
                role="memory",
                allowed_kinds=("postgres", "vector"),
                required_capabilities=("semantic", "vector", "graph"),
                namespace="tenant-a",
            ),
            StorageBackendRequirement(
                role="context_material",
                allowed_kinds=("vector", "graph"),
                required_capabilities=("semantic", "vector"),
                namespace="tenant-a",
            ),
        )
    ).manifest()

    assert [manifest["role"] for manifest in manifests] == ["memory", "context_material"]
    assert [manifest["kind"] for manifest in manifests] == ["postgres", "vector"]
    assert catalog.manifest()["external_backend_count"] == 2
    assert preflight["ready"] is True
    assert preflight["selected_roles"] == ["memory", "context_material"]
    assert preflight["selected_kinds"] == ["postgres", "vector"]
    assert preflight["blocking_count"] == 0


def test_agent_core_package_exports_storage_backend_contracts() -> None:
    assert agent_core.StorageBackendSpec is StorageBackendSpec
    assert agent_core.StorageBackendCatalog is StorageBackendCatalog
    assert agent_core.StorageBackendPreflightReport is StorageBackendPreflightReport
    assert agent_core.storage_backend_catalog_from_components is storage_backend_catalog_from_components
    assert (
        agent_core.storage_backend_manifests_from_components
        is storage_backend_manifests_from_components
    )
    assert agent_core.storage_backend_manifest is storage_backend_manifest
    assert "StorageBackendKind" in agent_core.__all__
