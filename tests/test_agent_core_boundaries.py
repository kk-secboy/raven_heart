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
        "AgentPromptBuilder",
        "ApprovalDecisionRecord",
        "ApprovalGrant",
        "ApprovalRecord",
        "ApprovalResumeContext",
        "ApprovalStorePort",
        "AgentRunStorePort",
        "AgentRunTraceBundle",
        "ContextInjection",
        "ContextReducerPort",
        "DefaultContextReducer",
        "DefaultTraceEvaluator",
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
        "ListEventSink",
        "NullApprovalStore",
        "NullRunTraceStore",
        "NullToolReplay",
        "NullEventSink",
        "LLMProviderCenter",
        "LLMCallRecord",
        "LLMMessage",
        "ToolCenter",
        "ToolRegistry",
        "SkillRegistry",
        "MCPCenter",
        "MCPStdioJSONRPCConnector",
        "MemoryCenter",
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
        "RuleBasedPolicy",
        "RuleBasedMemoryGovernance",
        "PersistentToolReplay",
        "PlanExecutionReport",
        "PlanExecutionStatus",
        "PlanExecutionStep",
        "PlanExecutor",
        "PlannerStorePort",
        "DEFAULT_PROMPT_TRIM_ORDER",
        "LLMStreamEvent",
        "ReducerRequest",
        "ReducerResult",
        "RetryHint",
        "RunTraceStorePort",
        "StructuredOutputResult",
        "StructuredOutputSpec",
        "StructuredOutputValidatorPort",
        "UsageInfo",
        "ToolReplayRecord",
        "ToolReplayStorePort",
        "TraceEvalHarness",
        "TraceEvalIssue",
        "TraceEvalReport",
        "TraceEvalSpec",
        "TraceEvaluatorPort",
        "TraceReplayHarness",
        "TraceReplayResult",
        "TraceReplayStep",
        "apply_reduction_to_timeline",
        "handoff_spec_from_session",
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

