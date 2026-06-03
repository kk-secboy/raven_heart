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
        "AgentSession",
        "AgentHarness",
        "AgentManagerCapacityError",
        "AgentManagerConcurrencyPolicy",
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
        "AgentRunStorePort",
        "AgentRunTraceBundle",
        "CapabilityDiscoveryResult",
        "CapabilityKind",
        "CapabilityMatch",
        "CapabilityQuery",
        "ContextInjection",
        "ContextInjectionDecision",
        "ContextInjectionDecisionStatus",
        "ContextInjectionPolicy",
        "ContextInjectionTrace",
        "ContextReducerPort",
        "DefaultLLMProviderCodec",
        "DefaultContextReducer",
        "DefaultTraceEvaluator",
        "DeterministicEmbeddingProvider",
        "EmbeddingProviderCenter",
        "EmbeddingProviderPort",
        "EmbeddingRequest",
        "EmbeddingResponse",
        "ExternalMemoryStore",
        "HandoffDecision",
        "HandoffRecord",
        "HandoffRequest",
        "HandoffRouter",
        "HandoffRouterPort",
        "HandoffSpec",
        "ReActExecutor",
        "InMemoryPlanner",
        "InMemoryPlannerStore",
        "JsonStructuredOutputValidator",
        "InMemoryApprovalStore",
        "InMemoryAgentRunStore",
        "InMemoryArtifactStore",
        "InMemoryRunTraceStore",
        "MarkdownApprovalStore",
        "MarkdownAgentRunStore",
        "MarkdownArtifactStore",
        "MarkdownPlannerStore",
        "MarkdownRunTraceStore",
        "InMemoryToolReplayStore",
        "EventLogPort",
        "ListEventSink",
        "MarkdownEventSink",
        "NullApprovalStore",
        "NullRunTraceStore",
        "NullToolReplay",
        "NullEventSink",
        "LLMProviderCenter",
        "LLMCallRecord",
        "LLMModelCapabilities",
        "LLMProviderCodecPort",
        "LLMMessage",
        "ToolCenter",
        "ToolRegistry",
        "SkillRegistry",
        "MCPCenter",
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
        "SQLitePlannerStore",
        "MarkdownMemoryStore",
        "MarkdownToolReplayStore",
        "StorageBackendKind",
        "StorageBackendRole",
        "StorageBackendSpec",
        "StorageBackendTrace",
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
        "PromptBucketBudgetDecision",
        "PromptBucketBudgetPolicy",
        "PromptBucketBudgetResult",
        "PromptBucketBudgetRule",
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
        "RunInterrupt",
        "RunInterruptKind",
        "SchemaValidationIssue",
        "SchemaValidationResult",
        "StructuredOutputResult",
        "StructuredOutputSpec",
        "StructuredOutputValidatorPort",
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
        "apply_reduction_to_timeline",
        "handoff_spec_from_session",
        "storage_backend_manifest",
        "validate_json_schema_subset",
        "validate_tool_arguments",
    }

    assert expected <= set(agent_core.__all__)
    for name in expected:
        assert getattr(agent_core, name) is not None


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

