"""End-to-end SDK acceptance scenario for host-runtime migration checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.artifacts import InMemoryArtifactStore
from agent_core.config import AgentProfile, CapabilitySet, RuntimeBudget
from agent_core.context import AgentContextPack
from agent_core.events import ListEventSink
from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.harness import InMemoryJournalStore, PersistentAgentJournal
from agent_core.manifest import (
    AgentCoreReadinessReport,
    AgentCoreSDKManifest,
    agent_core_sdk_manifest,
    evaluate_agent_core_readiness,
)
from agent_core.memory import InMemoryMemoryStore, MemoryCenter, MemoryRecord
from agent_core.policy import InMemoryPolicyDecisionStore
from agent_core.providers import LLMProviderCenter, LLMRequest, LLMResponse
from agent_core.runner import AgentRunRequest, AgentRunner, AgentSession
from agent_core.skills import SkillRegistry, SkillSpec, SkillsContext
from agent_core.tools import (
    InMemoryToolReplayStore,
    PersistentToolReplay,
    ToolCenter,
    ToolRegistry,
)
from agent_core.trace import InMemoryRunTraceStore


@dataclass(frozen=True)
class AgentCoreAcceptanceIssue:
    """One blocking issue from the SDK acceptance scenario."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreAcceptanceReport:
    """Prompt-safe summary of a deterministic SDK acceptance run."""

    status: str
    readiness: dict[str, Any] = field(default_factory=dict)
    trace_eval: dict[str, Any] = field(default_factory=dict)
    trace_summary: dict[str, Any] = field(default_factory=dict)
    run_summary: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "readiness": dict(self.readiness),
            "trace_eval": dict(self.trace_eval),
            "trace_summary": dict(self.trace_summary),
            "run_summary": dict(self.run_summary),
            "metadata": dict(self.metadata),
        }


class AcceptanceLLMProvider:
    """Deterministic provider used by the SDK acceptance scenario."""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self._responses: list[LLMResponse] = [
            LLMResponse(
                action={
                    "action": "call_tool",
                    "arguments": {
                        "tool_name": "lookup",
                        "arguments": {"query": "admin"},
                    },
                }
            ),
            LLMResponse(action={"action": "finish", "arguments": {"output": "accepted"}}),
        ]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(action={"action": "finish", "arguments": {"output": "accepted"}})


@dataclass(frozen=True)
class AgentCoreAcceptanceHarness:
    """Run the pure-core replacement-readiness acceptance scenario."""

    task: str = "inspect target"
    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(
        self,
        *,
        sdk_manifest: AgentCoreSDKManifest | dict[str, Any] | None = None,
    ) -> AgentCoreAcceptanceReport:
        readiness = evaluate_agent_core_readiness(sdk_manifest or agent_core_sdk_manifest())
        provider = AcceptanceLLMProvider()
        session = _acceptance_session(provider=provider)
        outcome = await AgentRunner(session).run(
            AgentRunRequest(
                task=self.task,
                context=AgentContextPack(workspace="workspace acceptance context"),
                metadata={"acceptance": "agent_core_replacement", **dict(self.metadata)},
            )
        )
        trace_eval = DefaultTraceEvaluator().evaluate(
            outcome.trace_manifest,
            _acceptance_trace_spec(),
        )
        issues = _acceptance_issues(readiness, trace_eval.manifest(), outcome.trace_manifest)
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        trace_summary = dict(outcome.trace_manifest.get("summary") or {})
        return AgentCoreAcceptanceReport(
            status=status,
            readiness=readiness.manifest(),
            trace_eval=trace_eval.manifest(),
            trace_summary=trace_summary,
            run_summary={
                "run_id": outcome.result.run_id,
                "status": outcome.result.status,
                "output": outcome.result.output,
                "provider_request_count": len(provider.requests),
                "tool_call_count": int(trace_summary.get("tool_center_call_count") or 0),
                "memory_hit_count": int(trace_summary.get("memory_search_hit_count") or 0),
                "context_injection_count": int(trace_summary.get("context_injection_count") or 0),
            },
            issues=issues,
            metadata={"scenario": "agent_core_replacement", **dict(self.metadata)},
        )


async def run_agent_core_acceptance(
    *,
    sdk_manifest: AgentCoreSDKManifest | dict[str, Any] | None = None,
    task: str = "inspect target",
    metadata: dict[str, Any] | None = None,
) -> AgentCoreAcceptanceReport:
    """Run the default pure-SDK acceptance scenario."""

    return await AgentCoreAcceptanceHarness(task=task, metadata=dict(metadata or {})).run(
        sdk_manifest=sdk_manifest
    )


def _acceptance_session(*, provider: AcceptanceLLMProvider) -> AgentSession:
    center = LLMProviderCenter(default_provider="acceptance")
    center.register("acceptance", provider, default_model="acceptance-mini")

    registry = ToolRegistry()

    @registry.register_function(description="Lookup target context", tags=("context", "lookup"))
    def lookup(query: str) -> str:
        return f"lookup:{query}:admin panel found"

    tools = ToolCenter()
    tools.mount("local", registry, tags=("acceptance",))

    skills = SkillRegistry()
    skills.register(
        SkillSpec(
            name="acceptance-skill",
            description="Acceptance skill",
            prompt="Use retrieved context before deciding.",
        )
    )
    skill_context = SkillsContext(skills)
    skill_context.load("acceptance-skill")

    memory = MemoryCenter(default_store="local")
    memory.register(
        "local",
        InMemoryMemoryStore(
            (
                MemoryRecord(
                    content="Remember to inspect the admin panel before finishing.",
                    source="acceptance-memory",
                ),
            )
        ),
    )

    return AgentSession(
        profile=AgentProfile(
            name="agent-core-acceptance",
            model="acceptance-mini",
            instructions="Run the SDK acceptance scenario.",
            capabilities=CapabilitySet(memory_enabled=True),
            budget=RuntimeBudget(max_iterations=4, max_cost_usd=0.1),
        ),
        provider=center,
        tools=tools,
        harness=PersistentAgentJournal(InMemoryJournalStore()),
        skills=skill_context,
        memory=memory,
        event_sink=ListEventSink(),
        tool_replay=PersistentToolReplay(InMemoryToolReplayStore()),
        trace_store=InMemoryRunTraceStore(),
        artifact_store=InMemoryArtifactStore(),
        policy_decision_store=InMemoryPolicyDecisionStore(),
        metadata={"scenario": "agent_core_acceptance"},
    )


def _acceptance_trace_spec() -> TraceEvalSpec:
    return TraceEvalSpec(
        name="agent-core-replacement-acceptance",
        expected_status="completed",
        max_iterations=4,
        max_provider_calls=3,
        required_provider_names=("acceptance",),
        required_provider_models=("acceptance-mini",),
        require_tool_center=True,
        required_tool_center_selected_mounts=("local",),
        required_tool_center_selected_tools=("lookup",),
        required_tool_center_requested_tools=("lookup",),
        require_tool_center_ready_routes=True,
        max_tool_center_failed_calls=0,
        require_event_log=True,
        require_terminal_event=True,
        require_event_sequence_monotonic=True,
        require_storage_backends=True,
        required_storage_backend_roles=(
            "memory",
            "journal",
            "tool_replay",
            "run_trace",
            "event_log",
            "artifact",
            "policy_decision",
        ),
        require_context_injections=True,
        required_context_injection_sources=("memory",),
        required_included_context_injection_sources=("memory",),
        max_excluded_context_injections=0,
        max_trimmed_context_injections=0,
        max_failure_count=0,
    )


def _acceptance_issues(
    readiness: AgentCoreReadinessReport,
    trace_eval: dict[str, Any],
    trace: dict[str, Any],
) -> tuple[AgentCoreAcceptanceIssue, ...]:
    issues: list[AgentCoreAcceptanceIssue] = []
    if not readiness.ready:
        for issue in readiness.issues:
            issues.append(
                AgentCoreAcceptanceIssue(
                    source="readiness",
                    code=issue.code,
                    message=issue.message,
                    severity=issue.severity,
                    metadata={"expected": issue.expected, "actual": issue.actual},
                )
            )
    if trace_eval.get("ok") is not True:
        for issue in trace_eval.get("issues") or ():
            if not isinstance(issue, dict):
                continue
            issues.append(
                AgentCoreAcceptanceIssue(
                    source="trace_eval",
                    code=str(issue.get("code") or "trace_eval_issue"),
                    message=str(issue.get("message") or ""),
                    severity=str(issue.get("severity") or "error"),
                    metadata=dict(issue.get("metadata") or {}),
                )
            )
    summary = trace.get("summary") or {}
    if int(summary.get("tool_center_call_count") or 0) < 1:
        issues.append(
            AgentCoreAcceptanceIssue(
                source="trace_summary",
                code="tool_call_missing",
                message="Acceptance run did not record a ToolCenter call.",
            )
        )
    if int(summary.get("memory_search_hit_count") or 0) < 1:
        issues.append(
            AgentCoreAcceptanceIssue(
                source="trace_summary",
                code="memory_hit_missing",
                message="Acceptance run did not record a memory hit.",
            )
        )
    if int(summary.get("context_injection_count") or 0) < 1:
        issues.append(
            AgentCoreAcceptanceIssue(
                source="trace_summary",
                code="context_injection_missing",
                message="Acceptance run did not record a context injection.",
            )
        )
    return tuple(issues)
