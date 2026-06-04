"""Package-level SDK capability and boundary manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from typing import Any, Literal, Sequence


SDKCapabilityLayer = Literal[
    "harness",
    "agent_loop",
    "provider",
    "tooling",
    "context",
    "memory",
    "storage",
    "policy",
    "trace",
    "eval",
    "coordination",
    "boundary",
]
SDKCapabilityStatus = Literal["mvp", "stable_contract", "external_contract", "excluded"]
SDKReadinessStatus = Literal["ready", "blocked"]
SDKReadinessIssueSeverity = Literal["error", "warning"]


CORE_STORAGE_ROLES: tuple[str, ...] = (
    "memory",
    "journal",
    "tool_replay",
    "run_trace",
    "run_state",
    "planner_state",
    "artifact",
    "context_material",
    "approval",
    "policy_decision",
    "event_log",
)
BUILTIN_STORAGE_KINDS: tuple[str, ...] = ("in_memory", "sqlite", "markdown")
EXTERNAL_STORAGE_KINDS: tuple[str, ...] = (
    "postgres",
    "object_storage",
    "vector",
    "graph",
    "product",
    "external",
    "custom",
)
FORBIDDEN_RUNTIME_DEPENDENCIES: tuple[str, ...] = (
    "app.openai_agents_runtime",
    "app.tools",
    "app.graph",
    "app.knowledge",
    "app.api",
    "openai",
    "agents",
    "fastapi",
    "graphiti",
    "mcp",
    "redis",
    "sqlalchemy",
)
FORBIDDEN_RUNTIME_PACKAGES: tuple[str, ...] = (
    "adapters",
    "integrations",
    "openai_agents_runtime",
    "graphiti",
    "raven_runtime",
)
RUNTIME_OWNED_CONCERNS: tuple[str, ...] = (
    "product orchestration",
    "domain-specific prompts",
    "concrete LLM clients and credentials",
    "real tools and sandboxing",
    "MCP server deployment and secrets",
    "Graphiti, RAG, vector, graph, and product stores",
    "approval UI, identity, and permissions",
    "API routes, UI streams, metrics, and deployment",
    "runtime adapters for RavenStorm or other hosts",
)
STABLE_PUBLIC_API: tuple[str, ...] = (
    "AgentRunner",
    "AgentSession",
    "AgentRunRequest",
    "AgentRunOutcome",
    "AgentSessionManager",
    "AgentHarness",
    "ReActExecutor",
    "ReActConfig",
    "ReActResult",
    "LLMProviderPort",
    "LLMProviderCenter",
    "LLMRequest",
    "LLMResponse",
    "LLMMessage",
    "LLMProviderSpec",
    "LLMRetryPolicy",
    "ToolRuntimePort",
    "ToolRegistry",
    "ToolCenter",
    "ToolSpec",
    "ToolInvocation",
    "ToolResult",
    "ToolRetryPolicy",
    "SkillRegistry",
    "SkillsContext",
    "SkillSpec",
    "MCPCenter",
    "MCPConnectorPort",
    "MemoryPort",
    "MemoryCenter",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryWrite",
    "MemoryStoreSpec",
    "ContextMaterialCenter",
    "ContextMaterialStorePort",
    "ContextInjectionPolicy",
    "PromptIR",
    "PromptBucket",
    "PromptBucketBudgetPolicy",
    "PromptSemanticReducerPort",
    "StorageBackendSpec",
    "StorageBackendRequirement",
    "StorageBackendCatalog",
    "StorageBackendPreflightReport",
    "PolicyDecision",
    "PolicyDecisionStorePort",
    "ApprovalCenter",
    "ApprovalStorePort",
    "AgentRunTraceBundle",
    "RunTraceStorePort",
    "TraceEvalSpec",
    "TraceEvalHarness",
    "TraceReplayHarness",
    "AgentCoreSDKManifest",
    "AgentCoreReadinessProfile",
    "AgentCoreReadinessReport",
    "AgentCoreAcceptanceHarness",
    "AgentCoreAcceptanceReport",
    "AgentCoreApprovalAcceptanceHarness",
    "AgentCoreApprovalAcceptanceReport",
    "AgentCoreContextAcceptanceHarness",
    "AgentCoreContextAcceptanceReport",
    "AgentCoreCoordinationAcceptanceHarness",
    "AgentCoreCoordinationAcceptanceReport",
    "AgentCoreEventAcceptanceHarness",
    "AgentCoreEventAcceptanceReport",
    "AgentCoreExternalBackendAcceptanceHarness",
    "AgentCoreExternalBackendAcceptanceReport",
    "AgentCoreGuardrailAcceptanceHarness",
    "AgentCoreGuardrailAcceptanceReport",
    "AgentCoreLifecycleAcceptanceHarness",
    "AgentCoreLifecycleAcceptanceReport",
    "AgentCoreOrchestrationAcceptanceHarness",
    "AgentCoreOrchestrationAcceptanceReport",
    "AgentCoreProviderAcceptanceHarness",
    "AgentCoreProviderAcceptanceReport",
    "AgentCoreRecoveryHarness",
    "AgentCoreRecoveryReport",
    "AgentCoreResumeAcceptanceHarness",
    "AgentCoreResumeAcceptanceReport",
    "AgentCoreStorageAcceptanceHarness",
    "AgentCoreStorageAcceptanceReport",
    "AgentCoreRuntimeBoundaryReport",
    "AgentCoreValidationSuite",
    "AgentCoreValidationReport",
    "agent_core_sdk_manifest",
    "evaluate_agent_core_readiness",
    "evaluate_agent_core_runtime_boundary",
    "run_agent_core_acceptance",
    "run_agent_core_approval_acceptance",
    "run_agent_core_context_acceptance",
    "run_agent_core_coordination_acceptance",
    "run_agent_core_event_acceptance",
    "run_agent_core_external_backend_acceptance",
    "run_agent_core_guardrail_acceptance",
    "run_agent_core_lifecycle_acceptance",
    "run_agent_core_orchestration_acceptance",
    "run_agent_core_provider_acceptance",
    "run_agent_core_recovery_acceptance",
    "run_agent_core_resume_acceptance",
    "run_agent_core_storage_acceptance",
    "run_agent_core_validation",
)


@dataclass(frozen=True)
class AgentCoreCapability:
    """One package-level capability exposed by the SDK base."""

    name: str
    layer: SDKCapabilityLayer
    status: SDKCapabilityStatus = "mvp"
    summary: str = ""
    public_contracts: tuple[str, ...] = ()
    runtime_notes: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-sdk-capability/v1",
            "name": self.name,
            "layer": self.layer,
            "status": self.status,
            "summary": self.summary,
            "public_contracts": list(self.public_contracts),
            "runtime_notes": list(self.runtime_notes),
        }


@dataclass(frozen=True)
class AgentCoreRuntimeBoundary:
    """Machine-readable boundary between the SDK base and host runtimes."""

    dependency_direction: str = "runtime imports agent_core"
    core_owns: tuple[str, ...] = (
        "agent loop semantics",
        "harness lifecycle and checkpoint/resume contracts",
        "provider-neutral LLM protocols and routing",
        "tool, skill, MCP, memory, policy, trace, replay, and eval contracts",
        "prompt bucket IR, semantic trimming, and context injection policies",
        "storage backend manifests and preflight requirements",
    )
    runtime_owns: tuple[str, ...] = RUNTIME_OWNED_CONCERNS
    forbidden_dependencies: tuple[str, ...] = FORBIDDEN_RUNTIME_DEPENDENCIES
    forbidden_packages: tuple[str, ...] = FORBIDDEN_RUNTIME_PACKAGES

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-runtime-boundary/v1",
            "dependency_direction": self.dependency_direction,
            "core_owns": list(self.core_owns),
            "runtime_owns": list(self.runtime_owns),
            "forbidden_dependencies": list(self.forbidden_dependencies),
            "forbidden_packages": list(self.forbidden_packages),
        }


@dataclass(frozen=True)
class AgentCoreSDKManifest:
    """Package-level capability, storage, API, and boundary manifest."""

    package_name: str = "raven-heart"
    package_version: str = "0.1.0"
    api_stability: str = "mvp"
    capabilities: tuple[AgentCoreCapability, ...] = ()
    public_api: tuple[str, ...] = ()
    storage_roles: tuple[str, ...] = CORE_STORAGE_ROLES
    builtin_storage_kinds: tuple[str, ...] = BUILTIN_STORAGE_KINDS
    external_storage_kinds: tuple[str, ...] = EXTERNAL_STORAGE_KINDS
    runtime_boundary: AgentCoreRuntimeBoundary = field(default_factory=AgentCoreRuntimeBoundary)
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        layers: dict[str, int] = {}
        statuses: dict[str, int] = {}
        for capability in self.capabilities:
            layers[capability.layer] = layers.get(capability.layer, 0) + 1
            statuses[capability.status] = statuses.get(capability.status, 0) + 1
        return {
            "schema_version": "agent-core-sdk-manifest/v1",
            "package": self.package_name,
            "version": self.package_version,
            "api_stability": self.api_stability,
            "capability_count": len(self.capabilities),
            "capability_layers": layers,
            "capability_statuses": statuses,
            "capabilities": [capability.manifest() for capability in self.capabilities],
            "storage_backend_interfaces": {
                "roles": list(self.storage_roles),
                "builtin_kinds": list(self.builtin_storage_kinds),
                "external_kinds": list(self.external_storage_kinds),
            },
            "runtime_boundary": self.runtime_boundary.manifest(),
            "public_api_count": len(self.public_api),
            "public_api": list(self.public_api),
            "api_contract": agent_core_api_contract(self.public_api).manifest(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreAPIContract:
    """Machine-readable public API stability contract."""

    stable_api: tuple[str, ...] = STABLE_PUBLIC_API
    mvp_api: tuple[str, ...] = ()
    experimental_api: tuple[str, ...] = ()
    deprecated_api: tuple[str, ...] = ()
    version_policy: str = (
        "Stable API names should not be removed or change manifest shape without "
        "a minor-version migration note; MVP names may still evolve before 1.0."
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-api-contract/v1",
            "stable_api_count": len(self.stable_api),
            "mvp_api_count": len(self.mvp_api),
            "experimental_api_count": len(self.experimental_api),
            "deprecated_api_count": len(self.deprecated_api),
            "stable_api": list(self.stable_api),
            "mvp_api": list(self.mvp_api),
            "experimental_api": list(self.experimental_api),
            "deprecated_api": list(self.deprecated_api),
            "version_policy": self.version_policy,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreAPIStabilityReport:
    """Report for checking a package manifest against the stable public API."""

    status: SDKReadinessStatus
    missing_stable_api: tuple[str, ...] = ()
    present_stable_api: tuple[str, ...] = ()
    deprecated_public_api: tuple[str, ...] = ()
    mvp_public_api: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-api-stability-report/v1",
            "status": self.status,
            "ready": self.ready,
            "missing_stable_api_count": len(self.missing_stable_api),
            "deprecated_public_api_count": len(self.deprecated_public_api),
            "mvp_public_api_count": len(self.mvp_public_api),
            "missing_stable_api": list(self.missing_stable_api),
            "present_stable_api": list(self.present_stable_api),
            "deprecated_public_api": list(self.deprecated_public_api),
            "mvp_public_api": list(self.mvp_public_api),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreReadinessIssue:
    """One missing or contradictory SDK readiness contract."""

    code: str
    message: str
    severity: SDKReadinessIssueSeverity = "error"
    expected: str = ""
    actual: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-readiness-issue/v1",
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True)
class AgentCoreReadinessReport:
    """Result of checking an SDK manifest against an integration profile."""

    profile_name: str
    status: SDKReadinessStatus
    issues: tuple[AgentCoreReadinessIssue, ...] = ()
    matched_capabilities: tuple[str, ...] = ()
    matched_public_api: tuple[str, ...] = ()
    matched_storage_roles: tuple[str, ...] = ()
    matched_builtin_storage_kinds: tuple[str, ...] = ()
    matched_external_storage_kinds: tuple[str, ...] = ()
    matched_forbidden_dependencies: tuple[str, ...] = ()
    matched_forbidden_packages: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-readiness-report/v1",
            "profile_name": self.profile_name,
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "matched": {
                "capabilities": list(self.matched_capabilities),
                "public_api": list(self.matched_public_api),
                "storage_roles": list(self.matched_storage_roles),
                "builtin_storage_kinds": list(self.matched_builtin_storage_kinds),
                "external_storage_kinds": list(self.matched_external_storage_kinds),
                "forbidden_dependencies": list(self.matched_forbidden_dependencies),
                "forbidden_packages": list(self.matched_forbidden_packages),
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreReadinessProfile:
    """Machine-readable SDK readiness requirements for host-runtime migration."""

    name: str
    purpose: str = ""
    required_capabilities: tuple[str, ...] = ()
    required_public_api: tuple[str, ...] = ()
    required_storage_roles: tuple[str, ...] = ()
    required_builtin_storage_kinds: tuple[str, ...] = ()
    required_external_storage_kinds: tuple[str, ...] = ()
    required_forbidden_dependencies: tuple[str, ...] = ()
    required_forbidden_packages: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, sdk_manifest: AgentCoreSDKManifest | dict[str, Any]) -> AgentCoreReadinessReport:
        manifest = (
            sdk_manifest.manifest()
            if isinstance(sdk_manifest, AgentCoreSDKManifest)
            else dict(sdk_manifest)
        )
        capabilities = {
            str(capability.get("name") or "")
            for capability in manifest.get("capabilities", ())
            if isinstance(capability, dict)
        }
        public_api = {str(name) for name in manifest.get("public_api", ())}
        storage = manifest.get("storage_backend_interfaces") or {}
        storage_roles = {str(name) for name in storage.get("roles", ())}
        builtin_kinds = {str(name) for name in storage.get("builtin_kinds", ())}
        external_kinds = {str(name) for name in storage.get("external_kinds", ())}
        boundary = manifest.get("runtime_boundary") or {}
        forbidden_dependencies = {str(name) for name in boundary.get("forbidden_dependencies", ())}
        forbidden_packages = {str(name) for name in boundary.get("forbidden_packages", ())}

        issues: list[AgentCoreReadinessIssue] = []
        matched_capabilities = _match_required(
            issues,
            required=self.required_capabilities,
            actual=capabilities,
            code="required_capability_missing",
            label="capability",
        )
        matched_public_api = _match_required(
            issues,
            required=self.required_public_api,
            actual=public_api,
            code="required_public_api_missing",
            label="public API",
        )
        matched_storage_roles = _match_required(
            issues,
            required=self.required_storage_roles,
            actual=storage_roles,
            code="required_storage_role_missing",
            label="storage role",
        )
        matched_builtin_storage_kinds = _match_required(
            issues,
            required=self.required_builtin_storage_kinds,
            actual=builtin_kinds,
            code="required_builtin_storage_kind_missing",
            label="built-in storage kind",
        )
        matched_external_storage_kinds = _match_required(
            issues,
            required=self.required_external_storage_kinds,
            actual=external_kinds,
            code="required_external_storage_kind_missing",
            label="external storage kind",
        )
        matched_forbidden_dependencies = _match_required(
            issues,
            required=self.required_forbidden_dependencies,
            actual=forbidden_dependencies,
            code="required_boundary_dependency_missing",
            label="forbidden dependency declaration",
        )
        matched_forbidden_packages = _match_required(
            issues,
            required=self.required_forbidden_packages,
            actual=forbidden_packages,
            code="required_boundary_package_missing",
            label="forbidden package declaration",
        )
        status: SDKReadinessStatus = "blocked" if any(
            issue.severity == "error" for issue in issues
        ) else "ready"
        return AgentCoreReadinessReport(
            profile_name=self.name,
            status=status,
            issues=tuple(issues),
            matched_capabilities=matched_capabilities,
            matched_public_api=matched_public_api,
            matched_storage_roles=matched_storage_roles,
            matched_builtin_storage_kinds=matched_builtin_storage_kinds,
            matched_external_storage_kinds=matched_external_storage_kinds,
            matched_forbidden_dependencies=matched_forbidden_dependencies,
            matched_forbidden_packages=matched_forbidden_packages,
            metadata={
                "purpose": self.purpose,
                "sdk_manifest_schema": str(manifest.get("schema_version") or ""),
                **dict(self.metadata),
            },
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-readiness-profile/v1",
            "name": self.name,
            "purpose": self.purpose,
            "required_capabilities": list(self.required_capabilities),
            "required_public_api": list(self.required_public_api),
            "required_storage_roles": list(self.required_storage_roles),
            "required_builtin_storage_kinds": list(self.required_builtin_storage_kinds),
            "required_external_storage_kinds": list(self.required_external_storage_kinds),
            "required_forbidden_dependencies": list(self.required_forbidden_dependencies),
            "required_forbidden_packages": list(self.required_forbidden_packages),
            "metadata": dict(self.metadata),
        }


def default_agent_core_capabilities() -> tuple[AgentCoreCapability, ...]:
    """Return the SDK base capability matrix without runtime adapter imports."""

    return (
        AgentCoreCapability(
            name="harness_lifecycle",
            layer="harness",
            summary="Run, turn, checkpoint, resume, cancel, timeout, and journal contracts.",
            public_contracts=("AgentHarness", "AgentRunner", "AgentSessionManager"),
        ),
        AgentCoreCapability(
            name="react_loop",
            layer="agent_loop",
            summary="Provider-neutral ReAct execution with action parsing and tool execution.",
            public_contracts=("ReActExecutor", "ReActConfig", "ReActResult"),
        ),
        AgentCoreCapability(
            name="llm_provider_center",
            layer="provider",
            summary="LLM provider ports, routing, request shape planning, streaming, and call audit.",
            public_contracts=("LLMProviderPort", "LLMProviderCenter", "TransportLLMProvider"),
        ),
        AgentCoreCapability(
            name="tool_skill_mcp_centers",
            layer="tooling",
            summary="Tool registry, tool center, skill registry, and MCP inventory contracts.",
            public_contracts=("ToolRegistry", "ToolCenter", "SkillRegistry", "MCPCenter"),
        ),
        AgentCoreCapability(
            name="prompt_context_semantics",
            layer="context",
            summary="Prompt buckets, budget policy, semantic trimming, context materials, and injection policy.",
            public_contracts=(
                "PromptIR",
                "PromptBucketBudgetPolicy",
                "PromptSemanticReducerPort",
                "ContextInjectionPolicy",
                "ContextMaterialCenter",
            ),
        ),
        AgentCoreCapability(
            name="memory_governance",
            layer="memory",
            summary="Memory center, governance decisions, and in-memory/SQLite/Markdown stores.",
            public_contracts=("MemoryCenter", "MemoryGovernancePort", "MemoryPort"),
            runtime_notes=("Postgres, graph, vector, and product memory stores implement ports outside core.",),
        ),
        AgentCoreCapability(
            name="storage_backend_contracts",
            layer="storage",
            status="stable_contract",
            summary="Backend manifests, catalog selection, and preflight requirements for all core store roles.",
            public_contracts=(
                "StorageBackendSpec",
                "StorageBackendCatalog",
                "StorageBackendRequirement",
                "StorageBackendPreflightReport",
            ),
        ),
        AgentCoreCapability(
            name="api_stability_contract",
            layer="boundary",
            status="stable_contract",
            summary="Package-root public API stability contract and compatibility report.",
            public_contracts=("AgentCoreAPIContract", "AgentCoreAPIStabilityReport"),
        ),
        AgentCoreCapability(
            name="runtime_boundary_audit",
            layer="boundary",
            status="stable_contract",
            summary="Package self-audit for forbidden runtime imports and adapter packages.",
            public_contracts=(
                "AgentCoreRuntimeBoundaryReport",
                "evaluate_agent_core_runtime_boundary",
            ),
        ),
        AgentCoreCapability(
            name="policy_approval",
            layer="policy",
            summary="Policy decisions, approval queues, decision stores, and resume context.",
            public_contracts=("PolicyDecision", "ApprovalCenter", "ApprovalResumeContext"),
            runtime_notes=("Approval UI, identity, and permission workflow stay in the runtime.",),
        ),
        AgentCoreCapability(
            name="trace_replay_eval",
            layer="trace",
            summary="Run trace bundle, trace stores, replay harness, eval contracts, and failure summaries.",
            public_contracts=("AgentRunTraceBundle", "TraceReplayHarness", "TraceEvalHarness"),
        ),
        AgentCoreCapability(
            name="replacement_acceptance_harness",
            layer="eval",
            summary="Pure-SDK end-to-end acceptance scenario for host-runtime migration gates.",
            public_contracts=("AgentCoreAcceptanceHarness", "AgentCoreAcceptanceReport"),
            runtime_notes=("Runtime-specific adapters still need separate host acceptance tests.",),
        ),
        AgentCoreCapability(
            name="context_acceptance_harness",
            layer="eval",
            summary="Pure-SDK context selection, injection, budget, semantic trim, and trace acceptance checks.",
            public_contracts=(
                "AgentCoreContextAcceptanceHarness",
                "AgentCoreContextAcceptanceReport",
            ),
            runtime_notes=("Runtime-specific context sources and ranking models stay outside core.",),
        ),
        AgentCoreCapability(
            name="approval_acceptance_harness",
            layer="eval",
            summary="Pure-SDK policy gate, approval queue, approval resume, and trace acceptance checks.",
            public_contracts=(
                "AgentCoreApprovalAcceptanceHarness",
                "AgentCoreApprovalAcceptanceReport",
            ),
            runtime_notes=("Operator UX, identity, notifications, and workflow routing stay outside core.",),
        ),
        AgentCoreCapability(
            name="orchestration_acceptance_harness",
            layer="eval",
            summary="Pure-SDK action, ToolCenter, skill, MCP, capability discovery, and context-material orchestration checks.",
            public_contracts=(
                "AgentCoreOrchestrationAcceptanceHarness",
                "AgentCoreOrchestrationAcceptanceReport",
            ),
            runtime_notes=("Concrete tools, MCP processes, secrets, and product orchestration stay outside core.",),
        ),
        AgentCoreCapability(
            name="coordination_acceptance_harness",
            layer="eval",
            summary="Pure-SDK planner, handoff, agent-as-tool, artifact, trace, and eval coordination checks.",
            public_contracts=(
                "AgentCoreCoordinationAcceptanceHarness",
                "AgentCoreCoordinationAcceptanceReport",
            ),
            runtime_notes=("Runtime-specific scheduling, worker pools, and product workflows stay outside core.",),
        ),
        AgentCoreCapability(
            name="event_acceptance_harness",
            layer="eval",
            summary="Pure-SDK streaming run, event log, event paging, event tail, and trace-eval checks.",
            public_contracts=(
                "AgentCoreEventAcceptanceHarness",
                "AgentCoreEventAcceptanceReport",
            ),
            runtime_notes=("Runtime-specific SSE, WebSocket, metrics, and observability sinks stay outside core.",),
        ),
        AgentCoreCapability(
            name="external_backend_acceptance_harness",
            layer="eval",
            summary="Pure-SDK runtime-owned memory/context backend portability, preflight, trace, and eval checks.",
            public_contracts=(
                "AgentCoreExternalBackendAcceptanceHarness",
                "AgentCoreExternalBackendAcceptanceReport",
            ),
            runtime_notes=("Concrete PostgreSQL, vector, graph, product, and Graphiti clients stay outside core.",),
        ),
        AgentCoreCapability(
            name="guardrail_acceptance_harness",
            layer="eval",
            summary="Pure-SDK input preflight, policy denial, structured-output repair, trace, and eval checks.",
            public_contracts=(
                "AgentCoreGuardrailAcceptanceHarness",
                "AgentCoreGuardrailAcceptanceReport",
            ),
            runtime_notes=("Product-specific safety policy, operator escalation, and domain guardrail suites stay outside core.",),
        ),
        AgentCoreCapability(
            name="lifecycle_acceptance_harness",
            layer="eval",
            summary="Pure-SDK scheduling, queueing, cancellation, timeout, interrupt, and trace-eval checks.",
            public_contracts=(
                "AgentCoreLifecycleAcceptanceHarness",
                "AgentCoreLifecycleAcceptanceReport",
            ),
            runtime_notes=("Distributed schedulers, leases, worker process control, and UI controls stay outside core.",),
        ),
        AgentCoreCapability(
            name="provider_acceptance_harness",
            layer="eval",
            summary="Pure-SDK provider compatibility matrix for routing, codecs, streaming, and transport fallback.",
            public_contracts=(
                "AgentCoreProviderAcceptanceHarness",
                "AgentCoreProviderAcceptanceReport",
            ),
            runtime_notes=("Concrete provider clients, credentials, and rate limits stay outside core.",),
        ),
        AgentCoreCapability(
            name="recovery_acceptance_harness",
            layer="eval",
            summary="Pure-SDK provider fallback and tool retry acceptance checks.",
            public_contracts=("AgentCoreRecoveryHarness", "AgentCoreRecoveryReport"),
            runtime_notes=("Runtime-specific retry budgets and circuit breakers remain host policy.",),
        ),
        AgentCoreCapability(
            name="resume_acceptance_harness",
            layer="eval",
            summary="Pure-SDK checkpoint/resume acceptance checks and trace eval gate.",
            public_contracts=(
                "AgentCoreResumeAcceptanceHarness",
                "AgentCoreResumeAcceptanceReport",
            ),
            runtime_notes=("Runtime-specific worker scheduling and UI recovery stay outside core.",),
        ),
        AgentCoreCapability(
            name="storage_acceptance_harness",
            layer="eval",
            summary="Pure-SDK storage backend portability checks for built-in and runtime-owned backend contracts.",
            public_contracts=(
                "AgentCoreStorageAcceptanceHarness",
                "AgentCoreStorageAcceptanceReport",
            ),
            runtime_notes=("Concrete PostgreSQL, vector, graph, object-store, and product clients stay outside core.",),
        ),
        AgentCoreCapability(
            name="validation_suite",
            layer="eval",
            status="stable_contract",
            summary="Aggregate SDK validation suite for readiness, API, run, context, approval, orchestration, coordination, events, lifecycle, provider, storage, recovery, and resume gates.",
            public_contracts=("AgentCoreValidationSuite", "AgentCoreValidationReport"),
            runtime_notes=("Runtime-specific adapter acceptance suites run after this SDK gate.",),
        ),
        AgentCoreCapability(
            name="planner_handoff_artifacts",
            layer="coordination",
            summary="Planner protocol, handoff records, agent-as-tool contracts, and artifact stores.",
            public_contracts=("PlannerPort", "HandoffRouter", "AgentToolRuntime", "ArtifactStorePort"),
        ),
        AgentCoreCapability(
            name="runtime_boundary",
            layer="boundary",
            status="excluded",
            summary="Raven/OpenAI Agents SDK/Graphiti/FastAPI adapters are intentionally excluded.",
            runtime_notes=RUNTIME_OWNED_CONCERNS,
        ),
    )


def agent_core_replacement_readiness_profile() -> AgentCoreReadinessProfile:
    """Readiness profile for replacing a generic hosted agent SDK core."""

    return AgentCoreReadinessProfile(
        name="agent-core-replacement-readiness",
        purpose=(
            "Check that the package exposes the generic SDK contracts needed by "
            "Raven, code-agent, ops-agent, or other runtimes before adapter work."
        ),
        required_capabilities=(
            "harness_lifecycle",
            "react_loop",
            "llm_provider_center",
            "tool_skill_mcp_centers",
            "prompt_context_semantics",
            "memory_governance",
            "storage_backend_contracts",
            "api_stability_contract",
            "runtime_boundary_audit",
            "policy_approval",
            "trace_replay_eval",
            "replacement_acceptance_harness",
            "context_acceptance_harness",
            "approval_acceptance_harness",
            "orchestration_acceptance_harness",
            "coordination_acceptance_harness",
            "event_acceptance_harness",
            "external_backend_acceptance_harness",
            "guardrail_acceptance_harness",
            "lifecycle_acceptance_harness",
            "provider_acceptance_harness",
            "storage_acceptance_harness",
            "recovery_acceptance_harness",
            "resume_acceptance_harness",
            "validation_suite",
            "planner_handoff_artifacts",
            "runtime_boundary",
        ),
        required_public_api=(
            "AgentRunner",
            "AgentSession",
            "AgentSessionManager",
            "ReActExecutor",
            "LLMProviderPort",
            "LLMProviderCenter",
            "ToolRegistry",
            "ToolCenter",
            "SkillRegistry",
            "MCPCenter",
            "MemoryCenter",
            "MemoryPort",
            "PromptIR",
            "PromptBucketBudgetPolicy",
            "PromptSemanticReducerPort",
            "ContextMaterialCenter",
            "ContextInjectionPolicy",
            "StorageBackendCatalog",
            "StorageBackendRequirement",
            "AgentRunTraceBundle",
            "TraceReplayHarness",
            "TraceEvalHarness",
            "AgentCoreSDKManifest",
            "AgentCoreAPIContract",
            "AgentCoreAPIStabilityReport",
            "agent_core_api_contract",
            "evaluate_agent_core_api_stability",
            "AgentCoreRuntimeBoundaryReport",
            "evaluate_agent_core_runtime_boundary",
            "AgentCoreAcceptanceHarness",
            "AgentCoreAcceptanceReport",
            "run_agent_core_acceptance",
            "AgentCoreApprovalAcceptanceHarness",
            "AgentCoreApprovalAcceptanceReport",
            "run_agent_core_approval_acceptance",
            "AgentCoreContextAcceptanceHarness",
            "AgentCoreContextAcceptanceReport",
            "run_agent_core_context_acceptance",
            "AgentCoreOrchestrationAcceptanceHarness",
            "AgentCoreOrchestrationAcceptanceReport",
            "run_agent_core_orchestration_acceptance",
            "AgentCoreCoordinationAcceptanceHarness",
            "AgentCoreCoordinationAcceptanceReport",
            "run_agent_core_coordination_acceptance",
            "AgentCoreEventAcceptanceHarness",
            "AgentCoreEventAcceptanceReport",
            "run_agent_core_event_acceptance",
            "AgentCoreExternalBackendAcceptanceHarness",
            "AgentCoreExternalBackendAcceptanceReport",
            "run_agent_core_external_backend_acceptance",
            "AgentCoreGuardrailAcceptanceHarness",
            "AgentCoreGuardrailAcceptanceReport",
            "run_agent_core_guardrail_acceptance",
            "AgentCoreLifecycleAcceptanceHarness",
            "AgentCoreLifecycleAcceptanceReport",
            "run_agent_core_lifecycle_acceptance",
            "AgentCoreProviderAcceptanceHarness",
            "AgentCoreProviderAcceptanceReport",
            "run_agent_core_provider_acceptance",
            "AgentCoreStorageAcceptanceHarness",
            "AgentCoreStorageAcceptanceReport",
            "run_agent_core_storage_acceptance",
            "AgentCoreRecoveryHarness",
            "AgentCoreRecoveryReport",
            "run_agent_core_recovery_acceptance",
            "AgentCoreResumeAcceptanceHarness",
            "AgentCoreResumeAcceptanceReport",
            "run_agent_core_resume_acceptance",
            "AgentCoreValidationSuite",
            "AgentCoreValidationReport",
            "run_agent_core_validation",
        ),
        required_storage_roles=CORE_STORAGE_ROLES,
        required_builtin_storage_kinds=BUILTIN_STORAGE_KINDS,
        required_external_storage_kinds=EXTERNAL_STORAGE_KINDS,
        required_forbidden_dependencies=FORBIDDEN_RUNTIME_DEPENDENCIES,
        required_forbidden_packages=FORBIDDEN_RUNTIME_PACKAGES,
        metadata={"runtime_adapters_in_scope": False},
    )


def agent_core_api_contract(
    public_api: Sequence[str] = (),
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreAPIContract:
    """Build the public API stability contract for a package-root export set."""

    public = set(str(name) for name in public_api)
    stable_set = set(STABLE_PUBLIC_API)
    mvp = tuple(sorted(public - stable_set))
    return AgentCoreAPIContract(
        stable_api=STABLE_PUBLIC_API,
        mvp_api=mvp,
        metadata=dict(metadata or {}),
    )


def evaluate_agent_core_api_stability(
    sdk_manifest: AgentCoreSDKManifest | dict[str, Any],
    *,
    contract: AgentCoreAPIContract | None = None,
) -> AgentCoreAPIStabilityReport:
    """Check that a package manifest still exposes the stable public API."""

    manifest = (
        sdk_manifest.manifest()
        if isinstance(sdk_manifest, AgentCoreSDKManifest)
        else sdk_manifest
    )
    public_api = {str(name) for name in manifest.get("public_api", ())}
    api_contract = contract or AgentCoreAPIContract()
    stable = tuple(api_contract.stable_api)
    deprecated = tuple(api_contract.deprecated_api)
    missing = tuple(name for name in stable if name not in public_api)
    present = tuple(name for name in stable if name in public_api)
    deprecated_public = tuple(name for name in deprecated if name in public_api)
    stable_set = set(stable)
    deprecated_set = set(deprecated)
    mvp_public = tuple(sorted(public_api - stable_set - deprecated_set))
    status: SDKReadinessStatus = "blocked" if missing else "ready"
    return AgentCoreAPIStabilityReport(
        status=status,
        missing_stable_api=missing,
        present_stable_api=present,
        deprecated_public_api=deprecated_public,
        mvp_public_api=mvp_public,
        metadata={
            "sdk_manifest_schema": str(manifest.get("schema_version") or ""),
            "package": str(manifest.get("package") or ""),
            "version": str(manifest.get("version") or ""),
        },
    )


def agent_core_sdk_manifest(
    *,
    public_api: Sequence[str] = (),
    package_version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreSDKManifest:
    """Build the SDK package manifest for runtime preflight and migration checks."""

    return AgentCoreSDKManifest(
        package_version=package_version or _package_version(),
        capabilities=default_agent_core_capabilities(),
        public_api=tuple(sorted(set(public_api))),
        metadata=dict(metadata or {}),
    )


def evaluate_agent_core_readiness(
    sdk_manifest: AgentCoreSDKManifest | dict[str, Any],
    profile: AgentCoreReadinessProfile | None = None,
) -> AgentCoreReadinessReport:
    """Evaluate an SDK manifest against the default replacement-readiness profile."""

    return (profile or agent_core_replacement_readiness_profile()).evaluate(sdk_manifest)


def _package_version() -> str:
    try:
        return importlib_metadata.version("raven-heart")
    except importlib_metadata.PackageNotFoundError:
        return "0.1.0"


def _match_required(
    issues: list[AgentCoreReadinessIssue],
    *,
    required: tuple[str, ...],
    actual: set[str],
    code: str,
    label: str,
) -> tuple[str, ...]:
    matched: list[str] = []
    for item in required:
        if item in actual:
            matched.append(item)
            continue
        issues.append(
            AgentCoreReadinessIssue(
                code=code,
                message=f"Required {label} is missing: {item}",
                expected=item,
                actual=",".join(sorted(actual)),
            )
        )
    return tuple(matched)
