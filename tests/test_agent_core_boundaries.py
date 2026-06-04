from __future__ import annotations

import ast
from pathlib import Path


AGENT_CORE = Path(__file__).resolve().parents[1] / "agent_core"
ROOT = Path(__file__).resolve().parents[1]


def test_agent_core_has_no_runtime_or_provider_imports() -> None:
    forbidden_prefixes = (
        "app.openai_agents_runtime",
        "app.tools",
        "app.graph",
        "app.knowledge",
        "app.api",
        "app.core.redis_client",
        "openai",
        "agents",
        "fastapi",
        "mcp",
        "redis",
        "sqlalchemy",
    )
    offenders: list[tuple[str, str]] = []

    for path in AGENT_CORE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes):
                    offenders.append((path.name, name))

    assert offenders == []


def test_agent_core_package_root_exports_stable_base_api() -> None:
    import agent_core

    expected = {
        "AgentRunner",
        "AgentCoreAcceptanceHarness",
        "AgentCoreAcceptanceIssue",
        "AgentCoreAcceptanceReport",
        "AgentCoreAPIContract",
        "AgentCoreAPIStabilityReport",
        "AgentCoreCapability",
        "AgentCoreReadinessIssue",
        "AgentCoreReadinessProfile",
        "AgentCoreReadinessReport",
        "AgentCoreRecoveryHarness",
        "AgentCoreRecoveryIssue",
        "AgentCoreRecoveryReport",
        "AgentCoreRuntimeBoundary",
        "AgentCoreSDKManifest",
        "AgentSession",
        "AgentHarness",
        "AgentManagerCapacityError",
        "AgentManagerConcurrencyPolicy",
        "AgentPromptBudgetPlan",
        "AgentPromptBuilder",
        "AgentResumeRequest",
        "ApprovalCenter",
        "ApprovalDecisionRecord",
        "ApprovalGrant",
        "ApprovalQueueFilter",
        "ApprovalQueueView",
        "ApprovalRecord",
        "ApprovalResolution",
        "ApprovalResumeContext",
        "ApprovalStorePort",
        "AgentRunQuery",
        "AgentRunStorePort",
        "AgentRunTraceBundle",
        "AgentRunPreflightCenter",
        "AgentRunPreflightCheckPort",
        "AgentRunPreflightCheckRecord",
        "AgentRunPreflightIssue",
        "AgentRunPreflightReport",
        "AgentRunPreflightRequest",
        "AgentRunPreflightRequirements",
        "AgentToolRuntime",
        "AgentToolSpec",
        "AgentToolTrace",
        "CapabilityDiscoveryResult",
        "CapabilityKind",
        "CapabilityMatch",
        "CapabilityQuery",
        "ContextInjection",
        "ContextInjectionDecision",
        "ContextInjectionDecisionStatus",
        "ContextInjectionPolicy",
        "ContextInjectionTrace",
        "ContextMaterialSelectionTrace",
        "ContextMaterial",
        "ContextMaterialCallRecord",
        "ContextMaterialCenter",
        "ContextMaterialQuery",
        "ContextMaterialQueryMode",
        "ContextMaterialRoute",
        "ContextMaterialSearchPlan",
        "ContextMaterialSelection",
        "ContextMaterialSelectionRequest",
        "ContextMaterialSelectionResult",
        "ContextMaterialSelectionStatus",
        "ContextMaterialSelectorPort",
        "ContextMaterialStoreNotFoundError",
        "ContextMaterialStorePort",
        "ContextMaterialStoreSpec",
        "ContextReducerPort",
        "DefaultLLMProviderCodec",
        "DefaultContextReducer",
        "DefaultContextMaterialSelector",
        "DefaultPromptSemanticReducer",
        "DefaultTraceEvaluator",
        "DeterministicEmbeddingProvider",
        "EmbeddingProviderCenter",
        "EmbeddingProviderPort",
        "EmbeddingRequest",
        "EmbeddingResponse",
        "ExternalMemoryCallRecord",
        "ExternalContextMaterialStore",
        "ExternalMemoryStore",
        "HandoffDecision",
        "HandoffRecord",
        "HandoffRequest",
        "HandoffRouter",
        "HandoffRouterPort",
        "HandoffSpec",
        "HandoffTrace",
        "ReActExecutor",
        "InMemoryPlanner",
        "InMemoryPlannerStore",
        "JsonStructuredOutputValidator",
        "InMemoryApprovalStore",
        "InMemoryAgentRunStore",
        "InMemoryArtifactStore",
        "InMemoryContextMaterialStore",
        "InMemoryRunTraceStore",
        "MarkdownApprovalStore",
        "MarkdownAgentRunStore",
        "MarkdownArtifactStore",
        "MarkdownPlannerStore",
        "MarkdownRunTraceStore",
        "InMemoryToolReplayStore",
        "ErrorClassification",
        "EventLogPort",
        "EventStreamBatch",
        "EventStreamCursor",
        "EventStreamTail",
        "FailureKind",
        "ListEventSink",
        "MarkdownEventSink",
        "NullApprovalStore",
        "NullRunTraceStore",
        "NullToolReplay",
        "NullEventSink",
        "classify_error",
        "LLMProviderCenter",
        "LLMCallRecord",
        "LLMModelCapabilities",
        "LLMProviderCodecPort",
        "LLMProviderRouteCandidate",
        "LLMProviderRoutePlan",
        "LLMRequestShapePlan",
        "LLMContentPart",
        "LLMMessage",
        "LLMResponseFormat",
        "LLMToolChoice",
        "LLMToolContract",
        "AgentLifecycleEvent",
        "AgentLifecycleHookCenter",
        "AgentLifecycleHookPort",
        "AgentLifecycleHookRecord",
        "LifecycleEventType",
        "LifecycleHookStatus",
        "NullLifecycleHooks",
        "ToolCenter",
        "ToolCenterCallRecord",
        "ToolCenterTrace",
        "ToolRegistry",
        "ToolRouteCandidate",
        "ToolRoutePlan",
        "SkillRegistry",
        "MCPCenter",
        "MCPInventoryRefreshResult",
        "MCPStdioJSONRPCConnector",
        "MemoryBackendKind",
        "MemoryCenter",
        "MemoryGovernanceTrace",
        "MemoryQueryMode",
        "MemoryRoute",
        "MemorySearchPlan",
        "MultiAgentCoordinator",
        "AgentJournalStorePort",
        "InMemoryJournalStore",
        "MarkdownJournalStore",
        "PersistentAgentJournal",
        "PersistentPlanner",
        "AgentJournalReplay",
        "AgentReplayEvent",
        "ReplayIssue",
        "SQLiteAgentJournal",
        "SQLiteEventSink",
        "SQLiteApprovalStore",
        "SQLiteAgentRunStore",
        "SQLiteArtifactStore",
        "SQLiteJournalStore",
        "SQLiteRunTraceStore",
        "SQLiteToolReplayStore",
        "SQLiteMemoryStore",
        "SQLiteContextMaterialStore",
        "SQLitePlannerStore",
        "MarkdownMemoryStore",
        "MarkdownContextMaterialStore",
        "MarkdownToolReplayStore",
        "StorageBackendKind",
        "StorageBackendPreflightReport",
        "StorageBackendCandidate",
        "StorageBackendCatalog",
        "StorageBackendRequirement",
        "StorageBackendRole",
        "StorageBackendSelection",
        "StorageBackendSpec",
        "StorageBackendTrace",
        "storage_backend_catalog_from_components",
        "storage_backend_manifests_from_components",
        "RuleBasedPolicy",
        "RuleBasedMemoryGovernance",
        "PersistentToolReplay",
        "PlanExecutionReport",
        "PlanExecutionStatus",
        "PlanExecutionStep",
        "PlanExecutor",
        "PlannerStorePort",
        "DEFAULT_PROMPT_TRIM_ORDER",
        "DEFAULT_PROMPT_TRIM_RULES",
        "LLMRetryPolicy",
        "LLMStreamEvent",
        "LLMUsageLimits",
        "LLMTransportPort",
        "LLMToolCall",
        "PromptBucketBudgetDecision",
        "PromptBucketBudgetPolicy",
        "PromptBucketBudgetResult",
        "PromptBucketBudgetRule",
        "PromptSemanticReducerPort",
        "PromptSemanticTrimDecision",
        "PromptSemanticTrimRequest",
        "PromptSemanticTrimResult",
        "PromptTrimPlan",
        "PromptTrimResult",
        "PromptTrimRule",
        "PromptTrimStep",
        "ReducerRequest",
        "ReducerResult",
        "ResumeCandidate",
        "ResumeIndex",
        "RetryHint",
        "RunTraceStorePort",
        "RunTraceQuery",
        "RunInterrupt",
        "RunInterruptKind",
        "SchemaValidationIssue",
        "SchemaValidationResult",
        "StructuredOutputResult",
        "StructuredOutputSpec",
        "StructuredOutputValidatorPort",
        "StaticRunPreflightCheck",
        "default_agent_run_preflight_center",
        "default_agent_core_capabilities",
        "agent_core_api_contract",
        "agent_core_sdk_manifest",
        "agent_core_replacement_readiness_profile",
        "evaluate_agent_core_api_stability",
        "evaluate_agent_core_readiness",
        "run_agent_core_acceptance",
        "run_agent_core_recovery_acceptance",
        "UsageInfo",
        "ToolReplayRecord",
        "ToolReplayStorePort",
        "TraceCorrelationEntry",
        "TraceCorrelationIndex",
        "TraceEvalHarness",
        "TraceEvalIssue",
        "TraceEvalReport",
        "TraceEvalSpec",
        "TraceEvaluatorPort",
        "TraceReplayComparator",
        "TraceReplayDiffIssue",
        "TraceReplayDiffReport",
        "TraceReplayDiffSpec",
        "TraceReplayHarness",
        "TraceReplayResult",
        "TraceReplayStep",
        "TransportLLMProvider",
        "agent_tool_spec_from_session",
        "apply_reduction_to_timeline",
        "handoff_spec_from_session",
        "storage_backend_manifest",
        "validate_json_schema_subset",
        "validate_tool_arguments",
    }

    assert expected <= set(agent_core.__all__)
    for name in expected:
        assert getattr(agent_core, name) is not None


def test_agent_core_sdk_manifest_declares_capabilities_and_runtime_boundary() -> None:
    import agent_core

    manifest = agent_core.agent_core_sdk_manifest().manifest()
    capability_names = {capability["name"] for capability in manifest["capabilities"]}
    boundary = manifest["runtime_boundary"]
    storage = manifest["storage_backend_interfaces"]

    assert manifest["schema_version"] == "agent-core-sdk-manifest/v1"
    assert manifest["package"] == "raven-heart"
    assert manifest["public_api_count"] == len(set(agent_core.__all__))
    assert manifest["api_contract"]["schema_version"] == "agent-core-api-contract/v1"
    assert "AgentRunner" in manifest["api_contract"]["stable_api"]
    assert "AgentRunner" in manifest["public_api"]
    assert "harness_lifecycle" in capability_names
    assert "react_loop" in capability_names
    assert "llm_provider_center" in capability_names
    assert "prompt_context_semantics" in capability_names
    assert "memory_governance" in capability_names
    assert "api_stability_contract" in capability_names
    assert "trace_replay_eval" in capability_names
    assert "replacement_acceptance_harness" in capability_names
    assert "recovery_acceptance_harness" in capability_names
    assert "runtime_boundary" in capability_names
    assert manifest["capability_statuses"]["excluded"] == 1
    assert boundary["dependency_direction"] == "runtime imports agent_core"
    assert "openai_agents_runtime" in boundary["forbidden_packages"]
    assert "openai" in boundary["forbidden_dependencies"]
    assert "runtime adapters for RavenStorm or other hosts" in boundary["runtime_owns"]
    assert {"memory", "journal", "run_trace", "event_log"} <= set(storage["roles"])
    assert {"in_memory", "sqlite", "markdown"} <= set(storage["builtin_kinds"])
    assert {"postgres", "vector", "graph", "product", "custom"} <= set(
        storage["external_kinds"]
    )


def test_agent_core_boundary_scan_uses_manifest_forbidden_dependencies() -> None:
    import agent_core

    boundary = agent_core.agent_core_sdk_manifest().manifest()["runtime_boundary"]
    forbidden_from_manifest = set(boundary["forbidden_dependencies"])
    scanned_prefixes = {
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
    }

    assert scanned_prefixes <= forbidden_from_manifest


def test_agent_core_replacement_readiness_profile_passes_current_sdk_manifest() -> None:
    import agent_core

    profile = agent_core.agent_core_replacement_readiness_profile()
    report = agent_core.evaluate_agent_core_readiness(profile)
    manifest = report.manifest()

    assert profile.manifest()["schema_version"] == "agent-core-readiness-profile/v1"
    assert manifest["schema_version"] == "agent-core-readiness-report/v1"
    assert manifest["profile_name"] == "agent-core-replacement-readiness"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert "AgentRunner" in manifest["matched"]["public_api"]
    assert "api_stability_contract" in manifest["matched"]["capabilities"]
    assert "prompt_context_semantics" in manifest["matched"]["capabilities"]
    assert "replacement_acceptance_harness" in manifest["matched"]["capabilities"]
    assert "recovery_acceptance_harness" in manifest["matched"]["capabilities"]
    assert "postgres" in manifest["matched"]["external_storage_kinds"]
    assert "openai_agents_runtime" in manifest["matched"]["forbidden_packages"]


def test_agent_core_readiness_report_blocks_missing_required_contracts() -> None:
    import agent_core

    sdk_manifest = agent_core.agent_core_sdk_manifest().manifest()
    sdk_manifest["capabilities"] = [
        capability
        for capability in sdk_manifest["capabilities"]
        if capability["name"] != "react_loop"
    ]
    sdk_manifest["public_api"] = [
        name for name in sdk_manifest["public_api"] if name != "ReActExecutor"
    ]

    report = agent_core.agent_core_replacement_readiness_profile().evaluate(sdk_manifest)
    issue_codes = {issue["code"] for issue in report.manifest()["issues"]}
    expected_values = {issue["expected"] for issue in report.manifest()["issues"]}

    assert report.ready is False
    assert report.manifest()["status"] == "blocked"
    assert "required_capability_missing" in issue_codes
    assert "required_public_api_missing" in issue_codes
    assert {"react_loop", "ReActExecutor"} <= expected_values


def test_agent_core_api_stability_report_tracks_stable_package_root() -> None:
    import agent_core

    contract = agent_core.agent_core_api_contract()
    report = agent_core.evaluate_agent_core_api_stability()
    manifest = report.manifest()

    assert contract.manifest()["schema_version"] == "agent-core-api-contract/v1"
    assert "AgentRunner" in contract.manifest()["stable_api"]
    assert manifest["schema_version"] == "agent-core-api-stability-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["missing_stable_api"] == []
    assert "AgentRunner" in manifest["present_stable_api"]
    assert manifest["mvp_public_api_count"] > 0


def test_agent_core_api_stability_report_blocks_missing_stable_api() -> None:
    import agent_core
    from agent_core.manifest import evaluate_agent_core_api_stability

    sdk_manifest = agent_core.agent_core_sdk_manifest().manifest()
    sdk_manifest["public_api"] = [
        name for name in sdk_manifest["public_api"] if name != "AgentRunner"
    ]

    report = evaluate_agent_core_api_stability(
        sdk_manifest,
        contract=agent_core.AgentCoreAPIContract(),
    )

    assert report.ready is False
    assert report.manifest()["status"] == "blocked"
    assert report.manifest()["missing_stable_api"] == ["AgentRunner"]


def test_repository_does_not_ship_runtime_adapter_packages() -> None:
    forbidden_dirs = {
        "adapters",
        "integrations",
        "openai_agents_runtime",
        "graphiti",
        "raven_runtime",
    }
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.iterdir()
        if path.is_dir() and path.name in forbidden_dirs
    ]

    assert offenders == []

