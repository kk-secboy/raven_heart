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
    "ContextWindowBuilder",
    "ContextWindowPolicy",
    "ContextWindowReport",
    "PromptIR",
    "PromptBucket",
    "PromptBucketBudgetPolicy",
    "PromptSemanticReducerPort",
    "StorageBackendSpec",
    "StorageBackendRequirement",
    "StorageBackendCatalog",
    "StorageBackendPreflightReport",
    "AgentStateBundle",
    "AgentStateBundleBuilder",
    "AgentStateBundlePolicy",
    "PolicyDecision",
    "PolicyDecisionStorePort",
    "ApprovalCenter",
    "ApprovalStorePort",
    "AgentRunTraceBundle",
    "RunTraceStorePort",
    "TraceExportBuilder",
    "TraceExportBundle",
    "TraceExportPolicy",
    "TraceExportRecord",
    "TraceEvalSpec",
    "TraceEvalHarness",
    "TraceReplayHarness",
    "AgentCoreSDKManifest",
    "AgentCoreAPILifecyclePolicy",
    "AgentCoreAPILifecycleReport",
    "AgentCoreReadinessProfile",
    "AgentCoreReadinessReport",
    "AgentCoreAcceptanceHarness",
    "AgentCoreAcceptanceReport",
    "AgentCoreApprovalAcceptanceHarness",
    "AgentCoreApprovalAcceptanceReport",
    "AgentCoreBudgetAcceptanceHarness",
    "AgentCoreBudgetAcceptanceReport",
    "AgentCoreContextAcceptanceHarness",
    "AgentCoreContextAcceptanceReport",
    "AgentCoreContextWindowAcceptanceHarness",
    "AgentCoreContextWindowAcceptanceReport",
    "AgentCoreCoordinationAcceptanceHarness",
    "AgentCoreCoordinationAcceptanceReport",
    "AgentCoreDurableSessionAcceptanceHarness",
    "AgentCoreDurableSessionAcceptanceReport",
    "AgentCoreEvalSuiteAcceptanceHarness",
    "AgentCoreEvalSuiteAcceptanceReport",
    "AgentCoreEventAcceptanceHarness",
    "AgentCoreEventAcceptanceReport",
    "AgentCoreExternalBackendAcceptanceHarness",
    "AgentCoreExternalBackendAcceptanceReport",
    "AgentCoreGuardrailAcceptanceHarness",
    "AgentCoreGuardrailAcceptanceReport",
    "AgentCoreInteractionAcceptanceHarness",
    "AgentCoreInteractionAcceptanceReport",
    "AgentCoreLifecycleAcceptanceHarness",
    "AgentCoreLifecycleAcceptanceReport",
    "AgentCoreNativeToolAcceptanceHarness",
    "AgentCoreNativeToolAcceptanceReport",
    "AgentCoreOrchestrationAcceptanceHarness",
    "AgentCoreOrchestrationAcceptanceReport",
    "AgentCorePackagingAcceptanceHarness",
    "AgentCorePackagingAcceptanceReport",
    "AgentCoreProviderAcceptanceHarness",
    "AgentCoreProviderAcceptanceReport",
    "AgentCoreProviderConformanceHarness",
    "AgentCoreProviderConformanceReport",
    "AgentCoreProviderConformanceSpec",
    "AgentCoreProviderResilienceAcceptanceHarness",
    "AgentCoreProviderResilienceAcceptanceReport",
    "AgentCoreRedactionAcceptanceHarness",
    "AgentCoreRedactionAcceptanceReport",
    "AgentCoreRecoveryHarness",
    "AgentCoreRecoveryReport",
    "AgentCoreResumeAcceptanceHarness",
    "AgentCoreResumeAcceptanceReport",
    "AgentCoreStorageAcceptanceHarness",
    "AgentCoreStorageAcceptanceReport",
    "AgentCoreStateBundleAcceptanceHarness",
    "AgentCoreStateBundleAcceptanceReport",
    "AgentCoreTaskProfileAcceptanceHarness",
    "AgentCoreTaskProfileAcceptanceReport",
    "AgentCoreTraceExportAcceptanceHarness",
    "AgentCoreTraceExportAcceptanceReport",
    "AgentCoreRuntimeBoundaryReport",
    "AgentCoreValidationSuite",
    "AgentCoreValidationReport",
    "RedactionPolicy",
    "RedactionResult",
    "TraceEvalCase",
    "TraceEvalSuite",
    "TraceEvalSuiteRunner",
    "export_trace_bundle",
    "agent_core_sdk_manifest",
    "build_agent_state_bundle",
    "run_agent_core_state_bundle_acceptance",
    "run_agent_core_trace_export_acceptance",
    "evaluate_agent_core_api_lifecycle",
    "evaluate_agent_core_readiness",
    "evaluate_agent_core_runtime_boundary",
    "run_agent_core_acceptance",
    "run_agent_core_approval_acceptance",
    "run_agent_core_budget_acceptance",
    "run_agent_core_context_acceptance",
    "build_context_window_report",
    "run_agent_core_context_window_acceptance",
    "run_agent_core_coordination_acceptance",
    "run_agent_core_durable_session_acceptance",
    "run_agent_core_eval_suite_acceptance",
    "run_agent_core_event_acceptance",
    "run_agent_core_external_backend_acceptance",
    "run_agent_core_guardrail_acceptance",
    "run_agent_core_interaction_acceptance",
    "run_agent_core_lifecycle_acceptance",
    "run_agent_core_native_tool_acceptance",
    "run_agent_core_orchestration_acceptance",
    "run_agent_core_packaging_acceptance",
    "run_agent_core_provider_acceptance",
    "run_agent_core_provider_conformance",
    "run_agent_core_provider_resilience_acceptance",
    "run_agent_core_redaction_acceptance",
    "run_agent_core_recovery_acceptance",
    "run_agent_core_resume_acceptance",
    "run_agent_core_storage_acceptance",
    "run_agent_core_task_profile_acceptance",
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
            "api_lifecycle_policy": AgentCoreAPILifecyclePolicy().manifest(),
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
class AgentCoreAPILifecyclePolicy:
    """Machine-readable API lifecycle and compatibility policy."""

    stable_api_change_policy: str = (
        "Stable API names require compatibility-preserving changes or a documented "
        "minor-version migration path."
    )
    mvp_api_change_policy: str = (
        "MVP API names may evolve before 1.0, but must remain visible in the API "
        "contract so runtime migrations can audit changes."
    )
    experimental_api_change_policy: str = (
        "Experimental API names are opt-in and must not be required by the default "
        "replacement-readiness profile."
    )
    deprecated_api_change_policy: str = (
        "Deprecated API names must remain exported through at least one minor "
        "release before removal."
    )
    min_deprecation_minor_versions: int = 1
    allow_mvp_breaking_changes_before_1: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-api-lifecycle-policy/v1",
            "stable_api_change_policy": self.stable_api_change_policy,
            "mvp_api_change_policy": self.mvp_api_change_policy,
            "experimental_api_change_policy": self.experimental_api_change_policy,
            "deprecated_api_change_policy": self.deprecated_api_change_policy,
            "min_deprecation_minor_versions": self.min_deprecation_minor_versions,
            "allow_mvp_breaking_changes_before_1": self.allow_mvp_breaking_changes_before_1,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreAPILifecycleIssue:
    """One API lifecycle policy issue."""

    code: str
    message: str
    severity: SDKReadinessIssueSeverity = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-api-lifecycle-issue/v1",
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreAPILifecycleReport:
    """Report for package-root API lifecycle and compatibility policy."""

    status: SDKReadinessStatus
    package: str = ""
    version: str = ""
    pre_1_0: bool = True
    policy: AgentCoreAPILifecyclePolicy = field(default_factory=AgentCoreAPILifecyclePolicy)
    stable_api_count: int = 0
    mvp_api_count: int = 0
    experimental_api_count: int = 0
    deprecated_api_count: int = 0
    public_api_count: int = 0
    unknown_public_api: tuple[str, ...] = ()
    issues: tuple[AgentCoreAPILifecycleIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-api-lifecycle-report/v1",
            "status": self.status,
            "ready": self.ready,
            "error_count": self.error_count,
            "issue_count": len(self.issues),
            "package": self.package,
            "version": self.version,
            "pre_1_0": self.pre_1_0,
            "policy": self.policy.manifest(),
            "stable_api_count": self.stable_api_count,
            "mvp_api_count": self.mvp_api_count,
            "experimental_api_count": self.experimental_api_count,
            "deprecated_api_count": self.deprecated_api_count,
            "public_api_count": self.public_api_count,
            "unknown_public_api_count": len(self.unknown_public_api),
            "unknown_public_api": list(self.unknown_public_api),
            "issues": [issue.manifest() for issue in self.issues],
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
            name="context_window_report",
            layer="context",
            summary="Unified prompt-safe report for selected, injected, trimmed, and excluded context window material.",
            public_contracts=(
                "ContextWindowBuilder",
                "ContextWindowPolicy",
                "ContextWindowReport",
                "build_context_window_report",
                "AgentCoreContextWindowAcceptanceHarness",
                "AgentCoreContextWindowAcceptanceReport",
            ),
            runtime_notes=(
                "Runtime-specific context sources, ranking models, and UI views stay outside core.",
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
            name="state_bundle_contracts",
            layer="storage",
            summary="Prompt-safe portable state bundles for SDK migration, archive, and recovery preflight.",
            public_contracts=(
                "AgentStateBundle",
                "AgentStateBundleBuilder",
                "AgentStateBundlePolicy",
                "build_agent_state_bundle",
                "AgentCoreStateBundleAcceptanceHarness",
                "AgentCoreStateBundleAcceptanceReport",
            ),
            runtime_notes=(
                "Concrete PG, object-store, product workflow, and cross-tenant migration jobs stay outside core.",
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
            name="api_lifecycle_policy",
            layer="boundary",
            status="stable_contract",
            summary="Package-root API lifecycle policy for stable, MVP, experimental, and deprecated exports.",
            public_contracts=(
                "AgentCoreAPILifecyclePolicy",
                "AgentCoreAPILifecycleReport",
                "evaluate_agent_core_api_lifecycle",
            ),
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
            name="trace_export_bundle",
            layer="trace",
            summary="Prompt-safe trace export bundles with redaction audit, byte counts, and payload digests.",
            public_contracts=(
                "TraceExportBuilder",
                "TraceExportBundle",
                "TraceExportPolicy",
                "TraceExportRecord",
                "export_trace_bundle",
                "AgentCoreTraceExportAcceptanceHarness",
                "AgentCoreTraceExportAcceptanceReport",
            ),
            runtime_notes=(
                "Runtime-specific sinks, dashboards, retention jobs, and tenant export workflows stay outside core.",
            ),
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
            name="durable_session_acceptance_harness",
            layer="eval",
            summary="Pure-SDK integrated AgentRunner session durable roundtrip over SQLite and Markdown stores.",
            public_contracts=(
                "AgentCoreDurableSessionAcceptanceHarness",
                "AgentCoreDurableSessionAcceptanceReport",
            ),
            runtime_notes=(
                "Production PostgreSQL, object storage, vector, graph, and product stores stay outside core.",
            ),
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
            name="interaction_acceptance_harness",
            layer="eval",
            summary="Pure-SDK mixed streaming, native tool-call, structured-output, event, trace, and eval checks.",
            public_contracts=(
                "AgentCoreInteractionAcceptanceHarness",
                "AgentCoreInteractionAcceptanceReport",
            ),
            runtime_notes=("Concrete provider credentials, runtime streams, and UI delivery stay outside core.",),
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
            name="native_tool_acceptance_harness",
            layer="eval",
            summary="Pure-SDK provider-native tool-call execution, tool-result message, journal, trace, and eval checks.",
            public_contracts=(
                "AgentCoreNativeToolAcceptanceHarness",
                "AgentCoreNativeToolAcceptanceReport",
            ),
            runtime_notes=("Provider-specific function-calling payload formats and HTTP clients stay outside core.",),
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
            name="provider_conformance_harness",
            layer="eval",
            summary="Provider-neutral conformance checks for external LLMProviderPort implementations.",
            public_contracts=(
                "AgentCoreProviderConformanceHarness",
                "AgentCoreProviderConformanceReport",
                "AgentCoreProviderConformanceSpec",
            ),
            runtime_notes=(
                "Real provider HTTP clients, credentials, deployment names, and rate limits stay outside core.",
            ),
        ),
        AgentCoreCapability(
            name="provider_resilience_acceptance_harness",
            layer="eval",
            summary="Pure-SDK provider retry, retry-after, fallback, explicit-provider fail-closed, stream-fallback, trace, and eval checks.",
            public_contracts=(
                "AgentCoreProviderResilienceAcceptanceHarness",
                "AgentCoreProviderResilienceAcceptanceReport",
            ),
            runtime_notes=(
                "Runtime-specific retry budgets, provider credentials, and circuit breakers remain host policy.",
            ),
        ),
        AgentCoreCapability(
            name="budget_acceptance_harness",
            layer="policy",
            summary="Pure-SDK provider call, token, and cost budget checks for LLMProviderCenter.",
            public_contracts=(
                "AgentCoreBudgetAcceptanceHarness",
                "AgentCoreBudgetAcceptanceReport",
            ),
            runtime_notes=(
                "Vendor quotas, tenant rate limits, billing APIs, and operator policy stay outside core.",
            ),
        ),
        AgentCoreCapability(
            name="redaction_contracts",
            layer="boundary",
            summary="Prompt-safe recursive redaction policy, result manifests, and acceptance gate.",
            public_contracts=(
                "RedactionPolicy",
                "RedactionResult",
                "AgentCoreRedactionAcceptanceHarness",
                "AgentCoreRedactionAcceptanceReport",
            ),
            runtime_notes=(
                "Credential sources, secret stores, tenant policy, and incident workflow stay outside core.",
            ),
        ),
        AgentCoreCapability(
            name="eval_suite_runner",
            layer="eval",
            summary="Provider-neutral multi-case trace eval suites for SDK/runtime regression gates.",
            public_contracts=(
                "TraceEvalCase",
                "TraceEvalSuite",
                "TraceEvalSuiteRunner",
                "AgentCoreEvalSuiteAcceptanceHarness",
                "AgentCoreEvalSuiteAcceptanceReport",
            ),
            runtime_notes=(
                "Runtime-specific datasets, dashboards, and domain scoring policy stay outside core.",
            ),
        ),
        AgentCoreCapability(
            name="packaging_acceptance_harness",
            layer="boundary",
            summary="Pure-SDK package metadata, build-system, public import, repository file, and example smoke checks.",
            public_contracts=(
                "AgentCorePackagingAcceptanceHarness",
                "AgentCorePackagingAcceptanceReport",
            ),
            runtime_notes=("Release automation, package indexes, signing, and adapter wheels stay outside core.",),
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
            name="task_profile_acceptance_harness",
            layer="eval",
            summary="Pure-SDK code, ops, and security task-profile portability checks through one agent core.",
            public_contracts=(
                "AgentCoreTaskProfileAcceptanceHarness",
                "AgentCoreTaskProfileAcceptanceReport",
            ),
            runtime_notes=("Domain prompts, concrete tools, production schedulers, and product workflows stay outside core.",),
        ),
        AgentCoreCapability(
            name="validation_suite",
            layer="eval",
            status="stable_contract",
            summary="Aggregate SDK validation suite for readiness, API, run, context, approval, orchestration, coordination, events, lifecycle, provider, provider conformance, eval suite, storage, recovery, and resume gates.",
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
            "context_window_report",
            "memory_governance",
            "storage_backend_contracts",
            "state_bundle_contracts",
            "api_stability_contract",
            "api_lifecycle_policy",
            "runtime_boundary_audit",
            "policy_approval",
            "trace_replay_eval",
            "trace_export_bundle",
            "replacement_acceptance_harness",
            "context_acceptance_harness",
            "approval_acceptance_harness",
            "orchestration_acceptance_harness",
            "coordination_acceptance_harness",
            "durable_session_acceptance_harness",
            "event_acceptance_harness",
            "external_backend_acceptance_harness",
            "guardrail_acceptance_harness",
            "interaction_acceptance_harness",
            "lifecycle_acceptance_harness",
            "native_tool_acceptance_harness",
            "packaging_acceptance_harness",
            "provider_acceptance_harness",
            "provider_conformance_harness",
            "provider_resilience_acceptance_harness",
            "budget_acceptance_harness",
            "redaction_contracts",
            "eval_suite_runner",
            "storage_acceptance_harness",
            "task_profile_acceptance_harness",
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
            "ContextWindowBuilder",
            "ContextWindowPolicy",
            "ContextWindowReport",
            "build_context_window_report",
            "StorageBackendCatalog",
            "StorageBackendRequirement",
            "AgentStateBundle",
            "AgentStateBundleBuilder",
            "AgentStateBundlePolicy",
            "build_agent_state_bundle",
            "AgentRunTraceBundle",
            "TraceExportBuilder",
            "TraceExportBundle",
            "TraceExportPolicy",
            "TraceExportRecord",
            "export_trace_bundle",
            "TraceReplayHarness",
            "TraceEvalHarness",
            "TraceEvalCase",
            "TraceEvalSuite",
            "TraceEvalSuiteRunner",
            "AgentCoreSDKManifest",
            "AgentCoreAPIContract",
            "AgentCoreAPILifecyclePolicy",
            "AgentCoreAPILifecycleReport",
            "evaluate_agent_core_api_lifecycle",
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
            "AgentCoreContextWindowAcceptanceHarness",
            "AgentCoreContextWindowAcceptanceReport",
            "run_agent_core_context_window_acceptance",
            "AgentCoreOrchestrationAcceptanceHarness",
            "AgentCoreOrchestrationAcceptanceReport",
            "run_agent_core_orchestration_acceptance",
            "AgentCoreCoordinationAcceptanceHarness",
            "AgentCoreCoordinationAcceptanceReport",
            "run_agent_core_coordination_acceptance",
            "AgentCoreDurableSessionAcceptanceHarness",
            "AgentCoreDurableSessionAcceptanceReport",
            "run_agent_core_durable_session_acceptance",
            "AgentCoreEventAcceptanceHarness",
            "AgentCoreEventAcceptanceReport",
            "run_agent_core_event_acceptance",
            "AgentCoreExternalBackendAcceptanceHarness",
            "AgentCoreExternalBackendAcceptanceReport",
            "run_agent_core_external_backend_acceptance",
            "AgentCoreGuardrailAcceptanceHarness",
            "AgentCoreGuardrailAcceptanceReport",
            "run_agent_core_guardrail_acceptance",
            "AgentCoreInteractionAcceptanceHarness",
            "AgentCoreInteractionAcceptanceReport",
            "run_agent_core_interaction_acceptance",
            "AgentCoreLifecycleAcceptanceHarness",
            "AgentCoreLifecycleAcceptanceReport",
            "run_agent_core_lifecycle_acceptance",
            "AgentCoreNativeToolAcceptanceHarness",
            "AgentCoreNativeToolAcceptanceReport",
            "run_agent_core_native_tool_acceptance",
            "AgentCorePackagingAcceptanceHarness",
            "AgentCorePackagingAcceptanceReport",
            "run_agent_core_packaging_acceptance",
            "AgentCoreProviderAcceptanceHarness",
            "AgentCoreProviderAcceptanceReport",
            "run_agent_core_provider_acceptance",
            "AgentCoreProviderConformanceHarness",
            "AgentCoreProviderConformanceReport",
            "AgentCoreProviderConformanceSpec",
            "run_agent_core_provider_conformance",
            "AgentCoreProviderResilienceAcceptanceHarness",
            "AgentCoreProviderResilienceAcceptanceReport",
            "run_agent_core_provider_resilience_acceptance",
            "AgentCoreBudgetAcceptanceHarness",
            "AgentCoreBudgetAcceptanceReport",
            "run_agent_core_budget_acceptance",
            "RedactionPolicy",
            "RedactionResult",
            "AgentCoreRedactionAcceptanceHarness",
            "AgentCoreRedactionAcceptanceReport",
            "run_agent_core_redaction_acceptance",
            "AgentCoreEvalSuiteAcceptanceHarness",
            "AgentCoreEvalSuiteAcceptanceReport",
            "run_agent_core_eval_suite_acceptance",
            "AgentCoreStorageAcceptanceHarness",
            "AgentCoreStorageAcceptanceReport",
            "run_agent_core_storage_acceptance",
            "AgentCoreStateBundleAcceptanceHarness",
            "AgentCoreStateBundleAcceptanceReport",
            "run_agent_core_state_bundle_acceptance",
            "AgentCoreTaskProfileAcceptanceHarness",
            "AgentCoreTaskProfileAcceptanceReport",
            "run_agent_core_task_profile_acceptance",
            "AgentCoreTraceExportAcceptanceHarness",
            "AgentCoreTraceExportAcceptanceReport",
            "run_agent_core_trace_export_acceptance",
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


def evaluate_agent_core_api_lifecycle(
    sdk_manifest: AgentCoreSDKManifest | dict[str, Any],
    *,
    contract: AgentCoreAPIContract | None = None,
    policy: AgentCoreAPILifecyclePolicy | None = None,
) -> AgentCoreAPILifecycleReport:
    """Evaluate package-root API lifecycle and compatibility policy."""

    manifest = (
        sdk_manifest.manifest()
        if isinstance(sdk_manifest, AgentCoreSDKManifest)
        else sdk_manifest
    )
    public_api = {str(name) for name in manifest.get("public_api", ())}
    api_contract = contract or agent_core_api_contract(tuple(sorted(public_api)))
    lifecycle_policy = policy or AgentCoreAPILifecyclePolicy()
    stable = set(api_contract.stable_api)
    mvp = set(api_contract.mvp_api)
    experimental = set(api_contract.experimental_api)
    deprecated = set(api_contract.deprecated_api)
    issues: list[AgentCoreAPILifecycleIssue] = []

    missing_stable = tuple(sorted(stable - public_api))
    for name in missing_stable:
        issues.append(
            AgentCoreAPILifecycleIssue(
                code="stable_api_missing",
                message=f"Stable API is missing from package root: {name}",
                metadata={"api_name": name},
            )
        )

    deprecated_public = tuple(sorted(deprecated & public_api))
    for name in deprecated_public:
        issues.append(
            AgentCoreAPILifecycleIssue(
                code="deprecated_api_exported",
                message=f"Deprecated API is still exported: {name}",
                severity="warning",
                metadata={"api_name": name},
            )
        )

    groups = {
        "stable": stable,
        "mvp": mvp,
        "experimental": experimental,
        "deprecated": deprecated,
    }
    for left_name, left in groups.items():
        for right_name, right in groups.items():
            if left_name >= right_name:
                continue
            overlap = tuple(sorted(left & right))
            if overlap:
                issues.append(
                    AgentCoreAPILifecycleIssue(
                        code="api_lifecycle_group_overlap",
                        message=f"API lifecycle groups overlap: {left_name}/{right_name}",
                        metadata={
                            "left": left_name,
                            "right": right_name,
                            "api_names": list(overlap),
                        },
                    )
                )

    known = stable | mvp | experimental | deprecated
    unknown_public = tuple(sorted(public_api - known))
    if unknown_public:
        issues.append(
            AgentCoreAPILifecycleIssue(
                code="unknown_public_api",
                message="Public API names are not classified by lifecycle contract.",
                metadata={"api_names": list(unknown_public)},
            )
        )

    version = str(manifest.get("version") or "")
    parsed_version = _parse_version_triplet(version)
    if parsed_version is None:
        issues.append(
            AgentCoreAPILifecycleIssue(
                code="package_version_unparseable",
                message="Package version is not parseable as major.minor.patch.",
                metadata={"version": version},
            )
        )
        pre_1_0 = True
    else:
        pre_1_0 = parsed_version[0] == 0

    status: SDKReadinessStatus = (
        "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
    )
    return AgentCoreAPILifecycleReport(
        status=status,
        package=str(manifest.get("package") or ""),
        version=version,
        pre_1_0=pre_1_0,
        policy=lifecycle_policy,
        stable_api_count=len(stable),
        mvp_api_count=len(mvp),
        experimental_api_count=len(experimental),
        deprecated_api_count=len(deprecated),
        public_api_count=len(public_api),
        unknown_public_api=unknown_public,
        issues=tuple(issues),
        metadata={
            "sdk_manifest_schema": str(manifest.get("schema_version") or ""),
            "api_contract_schema": api_contract.manifest()["schema_version"],
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


def _parse_version_triplet(version: str) -> tuple[int, int, int] | None:
    core = version.split("+", 1)[0].split("-", 1)[0]
    parts = core.split(".")
    if len(parts) < 2:
        return None
    try:
        major = int(parts[0])
        minor = int(parts[1])
        patch = int(parts[2]) if len(parts) > 2 else 0
    except ValueError:
        return None
    return (major, minor, patch)


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
