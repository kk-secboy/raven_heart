"""SDK-level context semantics acceptance checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.config import AgentProfile, CapabilitySet, RuntimeBudget
from agent_core.context import (
    AgentContextPack,
    ContextInjectionPolicy,
    ContextMaterial,
    ContextMaterialCenter,
    ContextMaterialQuery,
    ContextMaterialSelectionRequest,
    DefaultContextMaterialSelector,
    InMemoryContextMaterialStore,
)
from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.events import ListEventSink
from agent_core.harness import InMemoryJournalStore, PersistentAgentJournal
from agent_core.memory import InMemoryMemoryStore, MemoryCenter, MemoryRecord
from agent_core.policy import InMemoryPolicyDecisionStore
from agent_core.prompt import (
    DefaultPromptSemanticReducer,
    PromptBucketBudgetPolicy,
    PromptBucketBudgetRule,
    PromptBucketRole,
)
from agent_core.providers import (
    LLMModelCapabilities,
    LLMProviderCenter,
    LLMRequest,
    LLMResponse,
)
from agent_core.runner import AgentRunRequest, AgentRunner, AgentSession
from agent_core.tools import InMemoryToolReplayStore, PersistentToolReplay, ToolRegistry
from agent_core.trace import InMemoryRunTraceStore


@dataclass(frozen=True)
class AgentCoreContextAcceptanceIssue:
    """One blocking issue from the SDK context acceptance scenario."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-context-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreContextAcceptanceReport:
    """Prompt-safe context semantics acceptance report."""

    status: str
    trace_eval: dict[str, Any] = field(default_factory=dict)
    trace_summary: dict[str, Any] = field(default_factory=dict)
    context_summary: dict[str, Any] = field(default_factory=dict)
    run_summary: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreContextAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-context-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "trace_eval": dict(self.trace_eval),
            "trace_summary": dict(self.trace_summary),
            "context_summary": dict(self.context_summary),
            "run_summary": dict(self.run_summary),
            "metadata": dict(self.metadata),
        }


class ContextAcceptanceProvider:
    """Deterministic provider used by the context acceptance scenario."""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(action={"action": "finish", "arguments": {"output": "context-ok"}})


@dataclass(frozen=True)
class AgentCoreContextAcceptanceHarness:
    """Run a deterministic Yaklang-style context shaping acceptance scenario."""

    task: str = "investigate admin auth csrf risk"
    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreContextAcceptanceReport:
        provider = ContextAcceptanceProvider()
        session = _context_acceptance_session(provider=provider)
        request = _context_acceptance_request(self.task, self.metadata)
        outcome = await AgentRunner(session).run(request)
        trace_eval = DefaultTraceEvaluator().evaluate(
            outcome.trace_manifest,
            _context_acceptance_trace_spec(),
        )
        trace_summary = dict(outcome.trace_manifest.get("summary") or {})
        context_summary = _context_summary(outcome.trace_manifest)
        issues = _context_acceptance_issues(
            trace_eval=trace_eval.manifest(),
            trace=outcome.trace_manifest,
            provider_request_count=len(provider.requests),
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreContextAcceptanceReport(
            status=status,
            trace_eval=trace_eval.manifest(),
            trace_summary=trace_summary,
            context_summary=context_summary,
            run_summary={
                "run_id": outcome.result.run_id,
                "status": outcome.result.status,
                "output": outcome.result.output,
                "provider_request_count": len(provider.requests),
                "prompt_bytes": int(outcome.prompt_manifest.get("prompt_bytes") or 0),
            },
            issues=issues,
            metadata={"scenario": "agent_core_context_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_context_acceptance(
    *,
    task: str = "investigate admin auth csrf risk",
    metadata: dict[str, Any] | None = None,
) -> AgentCoreContextAcceptanceReport:
    """Run the default context semantics acceptance scenario."""

    return await AgentCoreContextAcceptanceHarness(
        task=task,
        metadata=dict(metadata or {}),
    ).run()


def _context_acceptance_session(*, provider: ContextAcceptanceProvider) -> AgentSession:
    center = LLMProviderCenter(default_provider="context")
    center.register(
        "context",
        provider,
        default_model="context-mini",
        default_capabilities=LLMModelCapabilities(
            context_window_tokens=420,
            max_output_tokens=80,
        ),
    )

    context_store = ContextMaterialCenter(default_store="local")
    context_store.register(
        "local",
        InMemoryContextMaterialStore(
            (
                ContextMaterial(
                    name="auth_trace",
                    content=(
                        "admin auth callback failed csrf token validation\n\n"
                        + "admin control plane csrf mismatch evidence " * 36
                    ),
                    role="timeline",
                    priority=9,
                    metadata={"source": "trace", "tags": ("auth",), "namespace": "tenant-a"},
                ),
                ContextMaterial(
                    name="risk_schema",
                    content='{"required":["risk","evidence","next_step"]}',
                    role="schema",
                    priority=8,
                    metadata={"source": "runtime", "tags": ("auth",), "namespace": "tenant-a"},
                ),
                ContextMaterial(
                    name="operator_hint",
                    content="operator asks for concise exploitability and next step " * 18,
                    role="runtime",
                    priority=7,
                    metadata={"source": "runtime", "tags": ("auth",), "namespace": "tenant-a"},
                ),
                ContextMaterial(
                    name="denied_static",
                    content="this must not be injected into high static rules",
                    role="system",
                    priority=10,
                    metadata={"source": "runtime", "tags": ("auth",), "namespace": "tenant-a"},
                ),
                ContextMaterial(
                    name="old_dns_note",
                    content="dns propagation completed without auth relevance",
                    role="memory",
                    priority=1,
                    metadata={"source": "memory", "tags": ("auth",), "namespace": "tenant-a"},
                ),
            )
        ),
        backend_kind="markdown",
        namespace="tenant-a",
        tags=("auth", "local"),
    )

    memory = MemoryCenter(default_store="local")
    memory.register(
        "local",
        InMemoryMemoryStore(
            (
                MemoryRecord(
                    content=(
                        "Previous admin auth csrf investigation kept this finding. "
                        + "csrf admin token mismatch and exploitability evidence " * 26
                    ),
                    source="context-memory",
                ),
            )
        ),
    )

    return AgentSession(
        profile=AgentProfile(
            name="context-acceptance",
            model="context-mini",
            instructions="Use selected context, preserve current task, and summarize risk.",
            capabilities=CapabilitySet(memory_enabled=True),
            budget=RuntimeBudget(max_iterations=2, max_prompt_bytes=1600, max_cost_usd=0.1),
        ),
        provider=center,
        tools=ToolRegistry(),
        harness=PersistentAgentJournal(InMemoryJournalStore()),
        memory=memory,
        context_material_store=context_store,
        context_material_selector=DefaultContextMaterialSelector(),
        context_injection_policy=ContextInjectionPolicy(
            max_injection_bytes=260,
            max_total_bytes=760,
            allowed_targets=(
                PromptBucketRole.SEMI_DYNAMIC_1,
                PromptBucketRole.SEMI_DYNAMIC_2,
                PromptBucketRole.TIMELINE_OPEN,
                PromptBucketRole.DYNAMIC,
            ),
        ),
        prompt_bucket_budget_policy=PromptBucketBudgetPolicy(
            rules=(
                PromptBucketBudgetRule(
                    role=PromptBucketRole.SEMI_DYNAMIC_1,
                    max_bytes=180,
                    min_keep_bytes=80,
                    reason="cap recalled memory before global trim",
                ),
            )
        ),
        prompt_semantic_reducer=DefaultPromptSemanticReducer(),
        event_sink=ListEventSink(),
        tool_replay=PersistentToolReplay(InMemoryToolReplayStore()),
        trace_store=InMemoryRunTraceStore(),
        policy_decision_store=InMemoryPolicyDecisionStore(),
        metadata={"scenario": "context_acceptance"},
    )


def _context_acceptance_request(
    task: str,
    metadata: dict[str, Any],
) -> AgentRunRequest:
    return AgentRunRequest(
        task=task,
        context=AgentContextPack(
            workspace="\n\n".join(
                (
                    "backup job completed successfully " + ("backup " * 80),
                    "admin auth csrf callback token mismatch on control plane "
                    + ("admin csrf control plane " * 70),
                    "unrelated package inventory finished cleanly " + ("inventory " * 80),
                    "admin auth exploitability depends on token reuse evidence "
                    + ("auth token evidence " * 70),
                )
            ),
            dynamic_task=task,
        ),
        context_material_query=ContextMaterialQuery(
            query="admin auth csrf risk",
            limit=5,
            namespace="tenant-a",
            filters={"tags": ("auth",)},
        ),
        context_material_selection=ContextMaterialSelectionRequest(
            task=task,
            max_materials=3,
            max_bytes=4096,
            allowed_targets=(
                PromptBucketRole.SEMI_DYNAMIC_1,
                PromptBucketRole.SEMI_DYNAMIC_2,
                PromptBucketRole.TIMELINE_OPEN,
                PromptBucketRole.DYNAMIC,
            ),
        ),
        metadata={"acceptance": "context_semantics", **dict(metadata)},
    )


def _context_acceptance_trace_spec() -> TraceEvalSpec:
    return TraceEvalSpec(
        name="agent-core-context-acceptance",
        expected_status="completed",
        max_iterations=2,
        max_provider_calls=1,
        required_provider_names=("context",),
        required_provider_models=("context-mini",),
        require_provider_model_capabilities=True,
        require_provider_route_plan=True,
        required_provider_route_selected_names=("context",),
        require_provider_route_preflight=True,
        require_provider_route_preflight_ready=True,
        require_provider_request_shape_plan=True,
        required_provider_request_shape_provider_names=("context",),
        max_provider_request_shape_final_output_tokens=80,
        require_prompt_budget=True,
        required_prompt_budget_sources=("provider_context_window",),
        required_prompt_budget_provider_names=("context",),
        max_prompt_budget_target_bytes=1360,
        require_prompt_bucket_budget=True,
        required_prompt_bucket_budget_roles=("semi_dynamic_1",),
        required_prompt_bucket_budget_statuses=("trimmed",),
        max_prompt_bucket_budget_trimmed=1,
        forbid_over_budget_prompt_buckets=True,
        require_prompt_semantic_trim=True,
        required_prompt_semantic_trim_roles=("timeline_open",),
        required_prompt_semantic_trim_statuses=("trimmed",),
        max_prompt_semantic_trimmed=1,
        max_prompt_semantic_trim_final_bytes=1360,
        require_context_material_selection=True,
        required_selected_context_material_names=("auth_trace", "risk_schema"),
        required_context_material_statuses=("selected", "target_denied", "count_exceeded"),
        required_context_material_targets=("timeline_open", "semi_dynamic_2"),
        max_dropped_context_materials=2,
        require_context_injections=True,
        required_context_injection_sources=("memory", "trace", "runtime"),
        required_context_injection_targets=("semi_dynamic_1", "timeline_open", "semi_dynamic_2"),
        required_context_injection_statuses=("trimmed", "included"),
        required_included_context_injection_sources=("memory", "trace", "runtime"),
        required_trimmed_context_injection_sources=("memory", "trace"),
        max_trimmed_context_injections=3,
        max_excluded_context_injections=0,
        require_event_log=True,
        require_terminal_event=True,
        require_storage_backends=True,
        required_storage_backend_roles=(
            "memory",
            "journal",
            "tool_replay",
            "run_trace",
            "event_log",
            "policy_decision",
            "context_material",
        ),
        max_failure_count=0,
    )


def _context_summary(trace: dict[str, Any]) -> dict[str, Any]:
    summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
    context_material = (
        trace.get("context_material_selection")
        if isinstance(trace.get("context_material_selection"), dict)
        else {}
    )
    injections = (
        trace.get("context_injections")
        if isinstance(trace.get("context_injections"), dict)
        else {}
    )
    semantic_trim = (
        trace.get("prompt_semantic_trim")
        if isinstance(trace.get("prompt_semantic_trim"), dict)
        else {}
    )
    bucket_budget = (
        trace.get("prompt_bucket_budget")
        if isinstance(trace.get("prompt_bucket_budget"), dict)
        else {}
    )
    return {
        "schema_version": "agent-core-context-acceptance-summary/v1",
        "context_material_selection_count": int(
            summary.get("context_material_selection_count") or 0
        ),
        "context_material_selected_count": int(
            summary.get("context_material_selected_count") or 0
        ),
        "context_material_dropped_count": int(
            summary.get("context_material_dropped_count") or 0
        ),
        "context_injection_count": int(summary.get("context_injection_count") or 0),
        "context_injection_trimmed_count": int(
            summary.get("context_injection_trimmed_count") or 0
        ),
        "prompt_bucket_budget_trimmed_count": int(
            summary.get("prompt_bucket_budget_trimmed_count") or 0
        ),
        "prompt_semantic_trimmed_count": int(
            summary.get("prompt_semantic_trimmed_count") or 0
        ),
        "selection_statuses": dict(context_material.get("statuses") or {}),
        "injection_statuses": dict(injections.get("statuses") or {}),
        "semantic_trim_statuses": {
            str(item.get("role") or ""): str(item.get("status") or "")
            for item in semantic_trim.get("decisions", ())
            if isinstance(item, dict)
        },
        "bucket_budget_statuses": {
            str(item.get("role") or ""): str(item.get("status") or "")
            for item in bucket_budget.get("decisions", ())
            if isinstance(item, dict)
        },
    }


def _context_acceptance_issues(
    *,
    trace_eval: dict[str, Any],
    trace: dict[str, Any],
    provider_request_count: int,
) -> tuple[AgentCoreContextAcceptanceIssue, ...]:
    issues: list[AgentCoreContextAcceptanceIssue] = []
    if trace_eval.get("ok") is not True:
        for issue in trace_eval.get("issues") or ():
            if not isinstance(issue, dict):
                continue
            issues.append(
                AgentCoreContextAcceptanceIssue(
                    source="trace_eval",
                    code=str(issue.get("code") or "trace_eval_issue"),
                    message=str(issue.get("message") or ""),
                    severity=str(issue.get("severity") or "error"),
                    metadata=dict(issue.get("metadata") or {}),
                )
            )
    summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
    required_counts = {
        "context_material_selected_count": 2,
        "context_injection_trimmed_count": 1,
        "prompt_bucket_budget_trimmed_count": 1,
        "prompt_semantic_trimmed_count": 1,
    }
    for key, minimum in required_counts.items():
        if int(summary.get(key) or 0) < minimum:
            issues.append(
                AgentCoreContextAcceptanceIssue(
                    source="trace_summary",
                    code=f"{key}_too_low",
                    message=f"Context acceptance summary did not reach {key}.",
                    metadata={"actual": int(summary.get(key) or 0), "minimum": minimum},
                )
            )
    if provider_request_count != 1:
        issues.append(
            AgentCoreContextAcceptanceIssue(
                source="provider",
                code="provider_request_count_unexpected",
                message="Context acceptance should complete in one provider request.",
                metadata={"actual": provider_request_count},
            )
        )
    return tuple(issues)
