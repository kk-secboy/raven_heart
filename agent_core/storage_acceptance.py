"""SDK-level storage backend acceptance checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_core.approvals import InMemoryApprovalStore, MarkdownApprovalStore, NullApprovalStore, SQLiteApprovalStore
from agent_core.artifacts import InMemoryArtifactStore, MarkdownArtifactStore, SQLiteArtifactStore
from agent_core.backends import (
    StorageBackendCatalog,
    StorageBackendRequirement,
    storage_backend_catalog_from_components,
    storage_backend_manifest,
    storage_backend_manifests_from_components,
)
from agent_core.context import (
    ContextMaterial,
    ContextMaterialCenter,
    ContextMaterialQuery,
    InMemoryContextMaterialStore,
    MarkdownContextMaterialStore,
    SQLiteContextMaterialStore,
)
from agent_core.events import ListEventSink, MarkdownEventSink, SQLiteEventSink
from agent_core.harness import InMemoryJournalStore, MarkdownJournalStore, SQLiteJournalStore
from agent_core.manifest import BUILTIN_STORAGE_KINDS, CORE_STORAGE_ROLES, EXTERNAL_STORAGE_KINDS
from agent_core.memory import (
    InMemoryMemoryStore,
    MarkdownMemoryStore,
    MemoryCenter,
    MemoryQuery,
    MemoryWrite,
    SQLiteMemoryStore,
)
from agent_core.planner import InMemoryPlannerStore, MarkdownPlannerStore, SQLitePlannerStore
from agent_core.policy import (
    InMemoryPolicyDecisionStore,
    MarkdownPolicyDecisionStore,
    NullPolicyDecisionStore,
    SQLitePolicyDecisionStore,
)
from agent_core.runner import InMemoryAgentRunStore, MarkdownAgentRunStore, SQLiteAgentRunStore
from agent_core.tools import InMemoryToolReplayStore, MarkdownToolReplayStore, SQLiteToolReplayStore
from agent_core.trace import InMemoryRunTraceStore, MarkdownRunTraceStore, NullRunTraceStore, SQLiteRunTraceStore


@dataclass(frozen=True)
class AgentCoreStorageAcceptanceIssue:
    """One blocking storage-backend acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-storage-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreStorageAcceptanceReport:
    """Prompt-safe storage backend acceptance report."""

    status: str
    builtin_matrix: dict[str, Any] = field(default_factory=dict)
    external_contracts: dict[str, Any] = field(default_factory=dict)
    memory_roundtrip: dict[str, Any] = field(default_factory=dict)
    context_roundtrip: dict[str, Any] = field(default_factory=dict)
    component_extraction: dict[str, Any] = field(default_factory=dict)
    preflight: dict[str, Any] = field(default_factory=dict)
    blocked_preflight: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreStorageAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-storage-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "builtin_matrix": dict(self.builtin_matrix),
            "external_contracts": dict(self.external_contracts),
            "memory_roundtrip": dict(self.memory_roundtrip),
            "context_roundtrip": dict(self.context_roundtrip),
            "component_extraction": dict(self.component_extraction),
            "preflight": dict(self.preflight),
            "blocked_preflight": dict(self.blocked_preflight),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreStorageAcceptanceHarness:
    """Run deterministic storage backend portability checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreStorageAcceptanceReport:
        with TemporaryDirectory(prefix="agent-core-storage-", ignore_cleanup_errors=True) as raw_root:
            root = Path(raw_root)
            store_manifests = _builtin_store_manifests(root)
            builtin_matrix = _builtin_matrix(store_manifests)
            memory_roundtrip, memory_center = await _memory_roundtrip(root)
            context_roundtrip, context_center = await _context_roundtrip(root)
            external_contracts = _external_contracts()
            component_backends = storage_backend_manifests_from_components(
                memory_center.manifest(),
                context_center.manifest(),
                *store_manifests,
                external_contracts,
            )
            catalog = StorageBackendCatalog(component_backends)
            preflight = catalog.preflight(
                _storage_requirements(),
                metadata={"scenario": "storage_acceptance"},
            ).manifest()
            blocked_preflight = catalog.preflight(
                (
                    StorageBackendRequirement(
                        role="memory",
                        allowed_kinds=("graph",),
                        required_capabilities=("graph",),
                        allow_external=False,
                    ),
                ),
                metadata={"scenario": "storage_acceptance_blocked"},
            ).manifest()
            component_extraction = {
                "schema_version": "agent-core-storage-component-extraction/v1",
                "backend_count": len(component_backends),
                "catalog": catalog.manifest(),
            }
            issues = _storage_acceptance_issues(
                builtin_matrix=builtin_matrix,
                external_contracts=external_contracts,
                memory_roundtrip=memory_roundtrip,
                context_roundtrip=context_roundtrip,
                component_extraction=component_extraction,
                preflight=preflight,
                blocked_preflight=blocked_preflight,
            )
            status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
            return AgentCoreStorageAcceptanceReport(
                status=status,
                builtin_matrix=builtin_matrix,
                external_contracts=external_contracts,
                memory_roundtrip=memory_roundtrip,
                context_roundtrip=context_roundtrip,
                component_extraction=component_extraction,
                preflight=preflight,
                blocked_preflight=blocked_preflight,
                issues=issues,
                metadata={"scenario": "agent_core_storage_acceptance", **dict(self.metadata)},
            )


async def run_agent_core_storage_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreStorageAcceptanceReport:
    """Run the default storage backend acceptance checks."""

    return await AgentCoreStorageAcceptanceHarness(metadata=dict(metadata or {})).run()


def _builtin_store_manifests(root: Path) -> tuple[dict[str, Any], ...]:
    stores = (
        InMemoryMemoryStore(),
        SQLiteMemoryStore(root / "memory.sqlite"),
        MarkdownMemoryStore(root / "memory-md"),
        InMemoryContextMaterialStore(),
        SQLiteContextMaterialStore(root / "context.sqlite"),
        MarkdownContextMaterialStore(root / "context-md"),
        InMemoryJournalStore(),
        SQLiteJournalStore(root / "journal.sqlite"),
        MarkdownJournalStore(root / "journal.md"),
        InMemoryToolReplayStore(),
        SQLiteToolReplayStore(root / "tool-replay.sqlite"),
        MarkdownToolReplayStore(root / "tool-replay.md"),
        NullRunTraceStore(),
        InMemoryRunTraceStore(),
        SQLiteRunTraceStore(root / "traces.sqlite"),
        MarkdownRunTraceStore(root / "traces.md"),
        InMemoryAgentRunStore(),
        SQLiteAgentRunStore(root / "runs.sqlite"),
        MarkdownAgentRunStore(root / "runs.md"),
        NullApprovalStore(),
        InMemoryApprovalStore(),
        SQLiteApprovalStore(root / "approvals.sqlite"),
        MarkdownApprovalStore(root / "approvals.md"),
        NullPolicyDecisionStore(),
        InMemoryPolicyDecisionStore(),
        SQLitePolicyDecisionStore(root / "policy.sqlite"),
        MarkdownPolicyDecisionStore(root / "policy.md"),
        InMemoryPlannerStore(),
        SQLitePlannerStore(root / "plans.sqlite"),
        MarkdownPlannerStore(root / "plans.md"),
        ListEventSink(),
        SQLiteEventSink(root / "events.sqlite"),
        MarkdownEventSink(root / "events.md"),
        InMemoryArtifactStore(),
        SQLiteArtifactStore(root / "artifacts.sqlite"),
        MarkdownArtifactStore(root / "artifacts.md"),
    )
    return tuple(store.manifest() for store in stores)


def _builtin_matrix(store_manifests: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    backends = storage_backend_manifests_from_components(*store_manifests)
    roles: dict[str, set[str]] = {}
    kinds: dict[str, int] = {}
    for backend in backends:
        role = str(backend.get("role") or "")
        kind = str(backend.get("kind") or "")
        roles.setdefault(role, set()).add(kind)
        kinds[kind] = kinds.get(kind, 0) + 1
    return {
        "schema_version": "agent-core-storage-builtin-matrix/v1",
        "backend_count": len(backends),
        "roles": {role: sorted(values) for role, values in sorted(roles.items())},
        "kinds": dict(sorted(kinds.items())),
        "required_roles": list(CORE_STORAGE_ROLES),
        "required_builtin_kinds": list(BUILTIN_STORAGE_KINDS),
    }


async def _memory_roundtrip(root: Path) -> tuple[dict[str, Any], MemoryCenter]:
    sqlite = SQLiteMemoryStore(root / "roundtrip-memory.sqlite")
    markdown = MarkdownMemoryStore(root / "roundtrip-memory-md")
    center = MemoryCenter(default_store="sqlite")
    center.register(
        "sqlite",
        sqlite,
        priority=20,
        namespaces=("tenant-a",),
        tags=("durable", "keyword"),
    )
    center.register(
        "markdown",
        markdown,
        priority=10,
        namespaces=("tenant-a",),
        tags=("inspectable", "keyword"),
    )
    await center.write(
        MemoryWrite(
            content="sqlite memory stores semantic project context",
            source="sqlite",
            metadata={"store": "sqlite"},
        )
    )
    await center.write(
        MemoryWrite(
            content="markdown memory stores readable project context",
            source="markdown",
            metadata={"store": "markdown"},
        )
    )
    sqlite_reloaded = SQLiteMemoryStore(root / "roundtrip-memory.sqlite")
    markdown_reloaded = MarkdownMemoryStore(root / "roundtrip-memory-md")
    sqlite_hits = await sqlite_reloaded.search(MemoryQuery("semantic project", limit=5))
    markdown_hits = await markdown_reloaded.search(MemoryQuery("readable project", limit=5))
    center_hits = await center.search(
        MemoryQuery("project context", limit=5, filters={"stores": ("sqlite", "markdown")})
    )
    return (
        {
            "schema_version": "agent-core-storage-memory-roundtrip/v1",
            "sqlite_hit_count": len(sqlite_hits),
            "markdown_hit_count": len(markdown_hits),
            "center_hit_count": len(center_hits),
            "center_manifest": center.manifest(),
        },
        center,
    )


async def _context_roundtrip(root: Path) -> tuple[dict[str, Any], ContextMaterialCenter]:
    sqlite = SQLiteContextMaterialStore(root / "roundtrip-context.sqlite")
    markdown = MarkdownContextMaterialStore(root / "roundtrip-context-md")
    center = ContextMaterialCenter(default_store="sqlite")
    center.register(
        "sqlite",
        sqlite,
        priority=20,
        namespace="tenant-a",
        tags=("durable", "semantic"),
    )
    center.register(
        "markdown",
        markdown,
        priority=10,
        namespace="tenant-a",
        tags=("inspectable", "semantic"),
    )
    await center.write(
        ContextMaterial(
            name="sqlite-context",
            content="sqlite context material about project scope",
            role="memory",
            priority=10,
            metadata={"store": "sqlite"},
        )
    )
    await center.write(
        ContextMaterial(
            name="markdown-context",
            content="markdown context material about readable notes",
            role="memory",
            priority=5,
            metadata={"store": "markdown"},
        )
    )
    sqlite_reloaded = SQLiteContextMaterialStore(root / "roundtrip-context.sqlite")
    markdown_reloaded = MarkdownContextMaterialStore(root / "roundtrip-context-md")
    sqlite_hits = await sqlite_reloaded.search(ContextMaterialQuery("project scope", limit=5))
    markdown_hits = await markdown_reloaded.search(ContextMaterialQuery("readable notes", limit=5))
    center_hits = await center.search(
        ContextMaterialQuery(
            "context material",
            limit=5,
            filters={"stores": ("sqlite", "markdown")},
            namespace="tenant-a",
        )
    )
    return (
        {
            "schema_version": "agent-core-storage-context-roundtrip/v1",
            "sqlite_hit_count": len(sqlite_hits),
            "markdown_hit_count": len(markdown_hits),
            "center_hit_count": len(center_hits),
            "center_manifest": center.manifest(),
        },
        center,
    )


def _external_contracts() -> dict[str, Any]:
    backends = (
        storage_backend_manifest(
            role="memory",
            kind="postgres",
            name="tenant-memory",
            namespace="tenant-a",
            core_builtin=False,
            capabilities=("keyword", "semantic", "vector"),
            location="runtime://postgres/memory",
            metadata={"owner": "runtime"},
        ),
        storage_backend_manifest(
            role="context_material",
            kind="vector",
            name="tenant-context",
            namespace="tenant-a",
            core_builtin=False,
            capabilities=("semantic", "hybrid"),
            location="runtime://vector/context",
            metadata={"owner": "runtime"},
        ),
        storage_backend_manifest(
            role="artifact",
            kind="object_storage",
            name="tenant-artifacts",
            namespace="tenant-a",
            core_builtin=False,
            capabilities=("blob", "signed_uri"),
            location="runtime://object-storage/artifacts",
            metadata={"owner": "runtime"},
        ),
        storage_backend_manifest(
            role="run_state",
            kind="product",
            name="workflow-runs",
            namespace="tenant-a",
            core_builtin=False,
            capabilities=("scheduler", "query"),
            location="runtime://product/runs",
            metadata={"owner": "runtime"},
        ),
    )
    return {
        "schema_version": "agent-core-storage-external-contracts/v1",
        "backend_count": len(backends),
        "backends": [dict(backend) for backend in backends],
        "external_kinds": list(EXTERNAL_STORAGE_KINDS),
    }


def _storage_requirements() -> tuple[StorageBackendRequirement, ...]:
    return (
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
            allowed_kinds=("vector", "postgres"),
            required_capabilities=("semantic", "hybrid"),
            namespace="tenant-a",
            require_durable=True,
            require_queryable=True,
        ),
        StorageBackendRequirement(
            role="artifact",
            allowed_kinds=("object_storage",),
            namespace="tenant-a",
            require_durable=True,
            allow_external=True,
        ),
        StorageBackendRequirement(
            role="journal",
            allowed_kinds=("sqlite", "markdown"),
            require_durable=True,
            allow_external=False,
        ),
    )


def _storage_acceptance_issues(
    *,
    builtin_matrix: dict[str, Any],
    external_contracts: dict[str, Any],
    memory_roundtrip: dict[str, Any],
    context_roundtrip: dict[str, Any],
    component_extraction: dict[str, Any],
    preflight: dict[str, Any],
    blocked_preflight: dict[str, Any],
) -> tuple[AgentCoreStorageAcceptanceIssue, ...]:
    issues: list[AgentCoreStorageAcceptanceIssue] = []
    roles = {str(role): set(kinds) for role, kinds in (builtin_matrix.get("roles") or {}).items()}
    missing_roles = sorted(role for role in CORE_STORAGE_ROLES if role not in roles)
    if missing_roles:
        issues.append(
            AgentCoreStorageAcceptanceIssue(
                source="builtin_matrix",
                code="builtin_storage_role_missing",
                message="Built-in storage matrix is missing required roles.",
                metadata={"missing_roles": missing_roles},
            )
        )
    missing_builtin_kinds = sorted(
        kind for kind in BUILTIN_STORAGE_KINDS if int((builtin_matrix.get("kinds") or {}).get(kind) or 0) < 1
    )
    if missing_builtin_kinds:
        issues.append(
            AgentCoreStorageAcceptanceIssue(
                source="builtin_matrix",
                code="builtin_storage_kind_missing",
                message="Built-in storage matrix is missing required backend kinds.",
                metadata={"missing_kinds": missing_builtin_kinds},
            )
        )
    for name, report in (
        ("memory_roundtrip", memory_roundtrip),
        ("context_roundtrip", context_roundtrip),
    ):
        if int(report.get("sqlite_hit_count") or 0) < 1 or int(report.get("markdown_hit_count") or 0) < 1:
            issues.append(
                AgentCoreStorageAcceptanceIssue(
                    source=name,
                    code=f"{name}_failed",
                    message=f"{name} did not prove SQLite and Markdown persistence.",
                    metadata={"report": dict(report)},
                )
            )
        if int(report.get("center_hit_count") or 0) < 2:
            issues.append(
                AgentCoreStorageAcceptanceIssue(
                    source=name,
                    code=f"{name}_center_routing_failed",
                    message=f"{name} did not prove center fan-out across stores.",
                    metadata={"report": dict(report)},
                )
            )
    external_kinds = {
        str(backend.get("kind") or "")
        for backend in external_contracts.get("backends") or ()
        if isinstance(backend, dict) and backend.get("core_builtin") is False
    }
    for kind in ("postgres", "vector", "object_storage", "product"):
        if kind not in external_kinds:
            issues.append(
                AgentCoreStorageAcceptanceIssue(
                    source="external_contracts",
                    code="external_backend_kind_missing",
                    message=f"External runtime-owned backend contract missing: {kind}",
                    metadata={"external_kinds": sorted(external_kinds)},
                )
            )
    if int(component_extraction.get("backend_count") or 0) < len(CORE_STORAGE_ROLES):
        issues.append(
            AgentCoreStorageAcceptanceIssue(
                source="component_extraction",
                code="component_backend_extraction_too_small",
                message="Component backend extraction did not find enough backend manifests.",
                metadata={"component_extraction": dict(component_extraction)},
            )
        )
    if preflight.get("ready") is not True:
        issues.append(
            AgentCoreStorageAcceptanceIssue(
                source="preflight",
                code="storage_preflight_not_ready",
                message="Storage backend preflight did not pass.",
                metadata={"preflight": dict(preflight)},
            )
        )
    selected_kinds = set(preflight.get("selected_kinds") or ())
    for kind in ("postgres", "vector", "object_storage"):
        if kind not in selected_kinds:
            issues.append(
                AgentCoreStorageAcceptanceIssue(
                    source="preflight",
                    code="runtime_backend_not_selected",
                    message=f"Storage preflight did not select expected runtime backend kind: {kind}",
                    metadata={"selected_kinds": sorted(selected_kinds)},
                )
            )
    if blocked_preflight.get("ready") is not False or blocked_preflight.get("status") != "blocked":
        issues.append(
            AgentCoreStorageAcceptanceIssue(
                source="blocked_preflight",
                code="blocked_storage_preflight_not_blocked",
                message="Negative storage preflight did not block as expected.",
                metadata={"blocked_preflight": dict(blocked_preflight)},
            )
        )
    return tuple(issues)
