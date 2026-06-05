"""SDK-level runtime-owned backend portability acceptance checks."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_core.backends import StorageBackendRequirement
from agent_core.config import AgentProfile, CapabilitySet
from agent_core.context import (
    ContextMaterial,
    ContextMaterialCenter,
    ContextMaterialQuery,
    ContextMaterialStorePort,
    DefaultContextMaterialSelector,
    ExternalContextMaterialStore,
    MarkdownContextMaterialStore,
    SQLiteContextMaterialStore,
)
from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.memory import (
    ExternalMemoryStore,
    MarkdownMemoryStore,
    MemoryCenter,
    MemoryHit,
    MemoryPort,
    MemoryQuery,
    MemoryWrite,
    SQLiteMemoryStore,
)
from agent_core.preflight import AgentRunPreflightRequirements
from agent_core.runner import AgentRunRequest, AgentRunner, AgentSession
from agent_core.testing import MockLLMProvider, MockToolRuntime


@dataclass(frozen=True)
class AgentCoreExternalBackendAcceptanceIssue:
    """One blocking external-backend portability acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-external-backend-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreExternalBackendAcceptanceReport:
    """Prompt-safe report for runtime-owned backend portability."""

    status: str
    run: dict[str, Any] = field(default_factory=dict)
    backend_matrix: dict[str, Any] = field(default_factory=dict)
    external_memory: dict[str, Any] = field(default_factory=dict)
    external_context_material: dict[str, Any] = field(default_factory=dict)
    storage_preflight: dict[str, Any] = field(default_factory=dict)
    trace_summary: dict[str, Any] = field(default_factory=dict)
    trace_eval: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreExternalBackendAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-external-backend-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "run": dict(self.run),
            "backend_matrix": dict(self.backend_matrix),
            "external_memory": dict(self.external_memory),
            "external_context_material": dict(self.external_context_material),
            "storage_preflight": dict(self.storage_preflight),
            "trace_summary": dict(self.trace_summary),
            "trace_eval": dict(self.trace_eval),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreExternalBackendAcceptanceHarness:
    """Run deterministic external backend portability checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreExternalBackendAcceptanceReport:
        provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}])
        external_memory_stores = (
            _external_memory_store(
                name="tenant-pg",
                backend_kind="postgres",
                source="runtime-pg",
                score=0.91,
                supports_vector=True,
                supports_graph=False,
                priority=30,
            ),
            _external_memory_store(
                name="tenant-graph",
                backend_kind="graph",
                source="runtime-graph",
                score=0.89,
                supports_vector=False,
                supports_graph=True,
                priority=25,
            ),
            _external_memory_store(
                name="tenant-product",
                backend_kind="product",
                source="runtime-product",
                score=0.87,
                supports_vector=True,
                supports_graph=True,
                priority=20,
            ),
            _external_memory_store(
                name="tenant-custom",
                backend_kind="custom",
                source="runtime-custom",
                score=0.82,
                supports_vector=False,
                supports_graph=False,
                priority=10,
            ),
        )
        memory = MemoryCenter(default_store="tenant-pg")
        for store in external_memory_stores:
            memory.register_spec(store.spec, store)

        external_context_stores = (
            ExternalContextMaterialStore(
                _RuntimeContextMaterialAdapter(
                    materials=(
                        ContextMaterial(
                            name="vector-note",
                            content="runtime vector context: csrf callback evidence",
                            role="evidence",
                            priority=5,
                            metadata={"namespace": "tenant-a", "tags": ("auth",)},
                        ),
                    )
                ),
                name="tenant-vector",
                backend_kind="vector",
                namespace="tenant-a",
                supports_vector=True,
                tags=("auth", "semantic"),
                location="runtime://vector/context",
                metadata={"adapter": "runtime-owned"},
            ),
            ExternalContextMaterialStore(
                _RuntimeContextMaterialAdapter(
                    materials=(
                        ContextMaterial(
                            name="object-note",
                            content="runtime object storage context: archived callback evidence",
                            role="evidence",
                            priority=4,
                            metadata={"namespace": "tenant-a", "tags": ("auth", "archive")},
                        ),
                    )
                ),
                name="tenant-object-storage",
                backend_kind="object_storage",
                namespace="tenant-a",
                supports_semantic=True,
                supports_vector=False,
                tags=("auth", "archive"),
                location="runtime://object-storage/context",
                metadata={"adapter": "runtime-owned"},
            ),
        )
        context_materials = ContextMaterialCenter(default_store="tenant-vector")
        for store in external_context_stores:
            context_materials.register_spec(store.spec, store)

        with tempfile.TemporaryDirectory(prefix="agent-core-backends-") as tmp:
            builtin_matrix = _builtin_backend_matrix(Path(tmp))
            session = AgentSession(
                profile=AgentProfile(
                    name="external-backend-acceptance",
                    capabilities=CapabilitySet(memory_enabled=True),
                ),
                provider=provider,
                tools=MockToolRuntime(),
                memory=memory,
                context_material_store=context_materials,
                context_material_selector=DefaultContextMaterialSelector(),
            )
            outcome = await AgentRunner(session).run(
                AgentRunRequest(
                    task="auth callback risk",
                    context_material_query=ContextMaterialQuery(
                        query="auth callback",
                        namespace="tenant-a",
                        filters={"tags": ("auth",)},
                    ),
                    preflight_requirements=AgentRunPreflightRequirements(
                        storage_backend_requirements=(
                            StorageBackendRequirement(
                                role="memory",
                                allowed_kinds=("postgres", "graph", "product"),
                                required_capabilities=("semantic", "graph"),
                                namespace="tenant-a",
                            ),
                            StorageBackendRequirement(
                                role="context_material",
                                allowed_kinds=("vector", "postgres"),
                                required_capabilities=("semantic", "vector"),
                                namespace="tenant-a",
                            ),
                        ),
                    ),
                    metadata={"acceptance": "external_backend"},
                )
            )
        trace_eval = DefaultTraceEvaluator().evaluate(
            outcome.trace_manifest,
            _external_backend_trace_spec(),
        ).manifest()
        summary = _external_backend_summary(dict(outcome.trace_manifest.get("summary") or {}))
        run = {
            "run_id": outcome.result.run_id,
            "status": outcome.result.status,
            "output": outcome.result.output,
            "provider_request_count": len(provider.requests),
        }
        storage_preflight = dict(outcome.trace_manifest.get("storage_backend_preflight") or {})
        backend_matrix = _backend_matrix(
            builtin=builtin_matrix,
            external_memory=tuple(store.manifest() for store in external_memory_stores),
            external_context=tuple(store.manifest() for store in external_context_stores),
        )
        external_memory_manifest = _external_memory_manifest(external_memory_stores)
        external_context_manifest = _external_context_manifest(external_context_stores)
        issues = _external_backend_acceptance_issues(
            run=run,
            backend_matrix=backend_matrix,
            external_memory=external_memory_manifest,
            external_context_material=external_context_manifest,
            storage_preflight=storage_preflight,
            trace_summary=summary,
            trace_eval=trace_eval,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreExternalBackendAcceptanceReport(
            status=status,
            run=run,
            backend_matrix=backend_matrix,
            external_memory=external_memory_manifest,
            external_context_material=external_context_manifest,
            storage_preflight=storage_preflight,
            trace_summary=summary,
            trace_eval=trace_eval,
            issues=issues,
            metadata={
                "scenario": "agent_core_external_backend_acceptance",
                **dict(self.metadata),
            },
        )


async def run_agent_core_external_backend_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreExternalBackendAcceptanceReport:
    """Run the default runtime-owned backend portability acceptance checks."""

    return await AgentCoreExternalBackendAcceptanceHarness(
        metadata=dict(metadata or {})
    ).run()


def _external_memory_store(
    *,
    name: str,
    backend_kind: str,
    source: str,
    score: float,
    supports_vector: bool,
    supports_graph: bool,
    priority: int,
) -> ExternalMemoryStore:
    return ExternalMemoryStore(
        _RuntimeMemoryAdapter(
            hits=(
                MemoryHit(
                    content=f"runtime {backend_kind} memory: auth callback signal",
                    score=score,
                    source=source,
                    metadata={"tenant": "tenant-a"},
                ),
            )
        ),
        name=name,
        backend_kind=backend_kind,  # type: ignore[arg-type]
        priority=priority,
        namespaces=("tenant-a",),
        supports_vector=supports_vector,
        supports_graph=supports_graph,
        tags=("tenant", backend_kind),
        location=f"runtime://{backend_kind}/memory",
        metadata={"adapter": "runtime-owned"},
    )


def _builtin_backend_matrix(root: Path) -> dict[str, Any]:
    sqlite_memory = SQLiteMemoryStore(root / "memory.sqlite")
    markdown_memory = MarkdownMemoryStore(root / "memory.md")
    sqlite_context = SQLiteContextMaterialStore(root / "context.sqlite")
    markdown_context = MarkdownContextMaterialStore(root / "context.md")
    return {
        "schema_version": "agent-core-builtin-backend-matrix/v1",
        "memory": [
            sqlite_memory.manifest(),
            markdown_memory.manifest(),
        ],
        "context_material": [
            sqlite_context.manifest(),
            markdown_context.manifest(),
        ],
        "kinds": ["sqlite", "markdown"],
    }


def _backend_matrix(
    *,
    builtin: dict[str, Any],
    external_memory: tuple[dict[str, Any], ...],
    external_context: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    external_memory_kinds = [
        str((item.get("store") or {}).get("backend_kind") or item.get("backend_kind") or "")
        for item in external_memory
    ]
    external_context_kinds = [
        str(((item.get("store") or {}).get("backend") or {}).get("kind") or "")
        for item in external_context
    ]
    external_kinds = sorted(set(external_memory_kinds + external_context_kinds))
    return {
        "schema_version": "agent-core-backend-portability-matrix/v1",
        "builtin": dict(builtin),
        "builtin_kinds": list(builtin.get("kinds") or ()),
        "external_memory": list(external_memory),
        "external_context_material": list(external_context),
        "external_kinds": external_kinds,
        "external_memory_kinds": sorted(set(external_memory_kinds)),
        "external_context_material_kinds": sorted(set(external_context_kinds)),
    }


def _external_memory_manifest(
    stores: tuple[ExternalMemoryStore, ...],
) -> dict[str, Any]:
    manifests = tuple(store.manifest() for store in stores)
    return {
        "schema_version": "agent-core-external-memory-portability/v1",
        "store_count": len(manifests),
        "call_count": sum(int(item.get("call_count") or 0) for item in manifests),
        "kinds": sorted(
            {
                str((item.get("store") or {}).get("backend_kind") or item.get("backend_kind") or "")
                for item in manifests
            }
        ),
        "stores": list(manifests),
    }


def _external_context_manifest(
    stores: tuple[ExternalContextMaterialStore, ...],
) -> dict[str, Any]:
    manifests = tuple(store.manifest() for store in stores)
    return {
        "schema_version": "agent-core-external-context-material-portability/v1",
        "store_count": len(manifests),
        "call_count": sum(int(item.get("call_count") or 0) for item in manifests),
        "kinds": sorted(
            {
                str(((item.get("store") or {}).get("backend") or {}).get("kind") or "")
                for item in manifests
            }
        ),
        "stores": list(manifests),
    }


class _RuntimeMemoryAdapter(MemoryPort):
    def __init__(self, *, hits: tuple[MemoryHit, ...]) -> None:
        self.hits = tuple(hits)
        self.queries: list[MemoryQuery] = []
        self.writes: list[MemoryWrite] = []

    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        self.queries.append(query)
        return self.hits[: query.limit]

    async def write(self, item: MemoryWrite) -> None:
        self.writes.append(item)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-runtime-memory-adapter/v1",
            "query_count": len(self.queries),
            "write_count": len(self.writes),
        }


class _RuntimeContextMaterialAdapter(ContextMaterialStorePort):
    def __init__(self, *, materials: tuple[ContextMaterial, ...]) -> None:
        self.materials = tuple(materials)
        self.queries: list[ContextMaterialQuery] = []
        self.writes: list[ContextMaterial] = []

    async def search(self, query: ContextMaterialQuery) -> tuple[ContextMaterial, ...]:
        self.queries.append(query)
        return self.materials[: query.limit]

    async def write(self, material: ContextMaterial) -> None:
        self.writes.append(material)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-runtime-context-material-adapter/v1",
            "query_count": len(self.queries),
            "write_count": len(self.writes),
        }


def _external_backend_trace_spec() -> TraceEvalSpec:
    return TraceEvalSpec(
        name="external-backend-acceptance",
        expected_status="completed",
        max_provider_calls=2,
        require_storage_backends=True,
        required_storage_backend_roles=("memory", "context_material"),
        required_storage_backend_kinds=(
            "postgres",
            "vector",
            "graph",
            "product",
            "object_storage",
            "custom",
        ),
        max_external_storage_backends=6,
        require_storage_backend_preflight=True,
        require_storage_backend_preflight_ready=True,
        required_storage_backend_preflight_roles=("memory", "context_material"),
        require_context_injections=True,
        required_context_injection_sources=("memory", "evidence"),
        require_context_material_selection=True,
        required_selected_context_material_names=("vector-note",),
        required_context_material_statuses=("selected",),
        required_context_material_targets=("dynamic",),
        max_dropped_context_materials=0,
        max_failure_count=0,
    )


def _external_backend_acceptance_issues(
    *,
    run: dict[str, Any],
    backend_matrix: dict[str, Any],
    external_memory: dict[str, Any],
    external_context_material: dict[str, Any],
    storage_preflight: dict[str, Any],
    trace_summary: dict[str, Any],
    trace_eval: dict[str, Any],
) -> tuple[AgentCoreExternalBackendAcceptanceIssue, ...]:
    issues: list[AgentCoreExternalBackendAcceptanceIssue] = []
    if run.get("status") != "completed" or run.get("output") != "done":
        issues.append(
            AgentCoreExternalBackendAcceptanceIssue(
                source="run",
                code="external_backend_run_not_completed",
                message="External backend acceptance run did not complete.",
                metadata=dict(run),
            )
        )
    builtin_kinds = set(str(kind) for kind in backend_matrix.get("builtin_kinds") or ())
    external_kinds = set(str(kind) for kind in backend_matrix.get("external_kinds") or ())
    if {"sqlite", "markdown"} - builtin_kinds:
        issues.append(
            AgentCoreExternalBackendAcceptanceIssue(
                source="backend_matrix",
                code="builtin_backend_kind_missing",
                message="Built-in backend matrix did not include SQLite and Markdown.",
                metadata={"builtin_kinds": sorted(builtin_kinds)},
            )
        )
    if {"postgres", "vector", "graph", "product", "object_storage", "custom"} - external_kinds:
        issues.append(
            AgentCoreExternalBackendAcceptanceIssue(
                source="backend_matrix",
                code="external_backend_kind_missing",
                message="External backend matrix did not include all runtime-owned backend kinds.",
                metadata={"external_kinds": sorted(external_kinds)},
            )
        )
    if int(external_memory.get("call_count") or 0) < 4:
        issues.append(
            AgentCoreExternalBackendAcceptanceIssue(
                source="external_memory",
                code="external_memory_not_called",
                message="Runtime-owned external memory adapters were not all called.",
                metadata={"call_count": external_memory.get("call_count")},
            )
        )
    if int(external_context_material.get("call_count") or 0) < 2:
        issues.append(
            AgentCoreExternalBackendAcceptanceIssue(
                source="external_context_material",
                code="external_context_not_called",
                message="Runtime-owned external context material adapters were not all called.",
                metadata={"call_count": external_context_material.get("call_count")},
            )
        )
    if storage_preflight.get("ready") is not True:
        issues.append(
            AgentCoreExternalBackendAcceptanceIssue(
                source="storage_preflight",
                code="external_storage_preflight_not_ready",
                message="External storage backend preflight was not ready.",
                metadata={
                    "status": storage_preflight.get("status"),
                    "blocking_reasons": list(storage_preflight.get("blocking_reasons") or ()),
                },
            )
        )
    if int(trace_summary.get("external_storage_backend_count") or 0) < 6:
        issues.append(
            AgentCoreExternalBackendAcceptanceIssue(
                source="trace_summary",
                code="external_storage_backend_trace_missing",
                message="Trace did not record all runtime-owned external backend kinds.",
                metadata=dict(trace_summary),
            )
        )
    if int(trace_summary.get("memory_search_hit_count") or 0) < 1:
        issues.append(
            AgentCoreExternalBackendAcceptanceIssue(
                source="trace_summary",
                code="external_memory_hit_missing",
                message="Trace did not record an external memory hit.",
                metadata=dict(trace_summary),
            )
        )
    if int(trace_summary.get("context_material_selected_count") or 0) < 1:
        issues.append(
            AgentCoreExternalBackendAcceptanceIssue(
                source="trace_summary",
                code="external_context_material_missing",
                message="Trace did not record selected external context material.",
                metadata=dict(trace_summary),
            )
        )
    if trace_eval.get("ok") is not True:
        issues.append(
            AgentCoreExternalBackendAcceptanceIssue(
                source="trace_eval",
                code="external_backend_trace_eval_failed",
                message="Trace evaluator did not accept external backend portability contract.",
                metadata={"trace_eval": dict(trace_eval)},
            )
        )
    return tuple(issues)


def _external_backend_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "memory_search_hit_count",
        "storage_backend_count",
        "external_storage_backend_count",
        "storage_backend_roles",
        "storage_backend_kinds",
        "has_storage_backend_preflight",
        "storage_backend_preflight_ready",
        "storage_backend_preflight_blocking_count",
        "context_injection_count",
        "context_injection_names",
        "context_injection_sources",
        "included_context_injection_sources",
        "context_material_selection_count",
        "context_material_selected_count",
        "context_material_dropped_count",
        "selected_context_material_names",
        "context_material_statuses",
        "context_material_targets",
        "selected_context_material_bytes",
        "failure_count",
        "failure_sources",
        "failure_kinds",
    )
    return {key: summary.get(key) for key in keys if key in summary}
