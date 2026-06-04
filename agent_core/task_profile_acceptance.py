"""SDK-level cross-runtime task profile acceptance checks."""

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
from agent_core.prompt import PromptBucketRole
from agent_core.providers import LLMModelCapabilities, LLMProviderCenter, LLMRequest, LLMResponse
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
class AgentCoreTaskProfileAcceptanceIssue:
    """One blocking cross-runtime task profile acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-task-profile-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreTaskProfileAcceptanceReport:
    """Prompt-safe report for code/ops/security task-profile portability."""

    status: str
    profiles: tuple[dict[str, Any], ...] = ()
    matrix: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreTaskProfileAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-task-profile-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "profiles": [dict(profile) for profile in self.profiles],
            "matrix": dict(self.matrix),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TaskProfileSpec:
    """One deterministic task profile used to prove SDK runtime portability."""

    name: str
    task: str
    skill_name: str
    skill_prompt: str
    context_name: str
    context_content: str
    memory_content: str
    tool_result: str


class TaskProfileProvider:
    """Deterministic provider that exercises tool execution before finishing."""

    def __init__(self, profile: TaskProfileSpec) -> None:
        self.profile = profile
        self.requests: list[LLMRequest] = []
        self._responses: list[LLMResponse] = [
            LLMResponse(
                action={
                    "action": "call_tool",
                    "arguments": {
                        "tool_name": "inspect_profile",
                        "arguments": {"profile": profile.name},
                    },
                }
            ),
            LLMResponse(
                action={
                    "action": "finish",
                    "arguments": {"output": f"{profile.name}-profile-ready"},
                }
            ),
        ]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(
            action={
                "action": "finish",
                "arguments": {"output": f"{self.profile.name}-profile-ready"},
            }
        )


@dataclass(frozen=True)
class AgentCoreTaskProfileAcceptanceHarness:
    """Run deterministic code/ops/security profile checks through one SDK core."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreTaskProfileAcceptanceReport:
        profile_reports = []
        issues: list[AgentCoreTaskProfileAcceptanceIssue] = []
        for spec in _task_profiles():
            report = await _run_task_profile(spec, self.metadata)
            profile_reports.append(report)
            issues.extend(_task_profile_issues(report))
        matrix = _task_profile_matrix(profile_reports)
        issues.extend(_task_profile_matrix_issues(matrix))
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreTaskProfileAcceptanceReport(
            status=status,
            profiles=tuple(profile_reports),
            matrix=matrix,
            issues=tuple(issues),
            metadata={"scenario": "agent_core_task_profile_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_task_profile_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreTaskProfileAcceptanceReport:
    """Run the default cross-runtime task profile acceptance checks."""

    return await AgentCoreTaskProfileAcceptanceHarness(metadata=dict(metadata or {})).run()


async def _run_task_profile(
    spec: TaskProfileSpec,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    provider = TaskProfileProvider(spec)
    session = _task_profile_session(spec, provider)
    request = _task_profile_request(spec, metadata)
    outcome = await AgentRunner(session).run(request)
    trace_eval = DefaultTraceEvaluator().evaluate(
        outcome.trace_manifest,
        _task_profile_trace_spec(spec),
    ).manifest()
    summary = dict(outcome.trace_manifest.get("summary") or {})
    return {
        "schema_version": "agent-core-task-profile-run/v1",
        "name": spec.name,
        "status": outcome.result.status,
        "output": outcome.result.output,
        "provider_request_count": len(provider.requests),
        "provider_name": f"{spec.name}-provider",
        "model": f"{spec.name}-mini",
        "tool_call_count": int(summary.get("tool_center_call_count") or 0),
        "memory_hit_count": int(summary.get("memory_search_hit_count") or 0),
        "context_material_selected_count": int(
            summary.get("context_material_selected_count") or 0
        ),
        "context_injection_count": int(summary.get("context_injection_count") or 0),
        "loaded_skill": spec.skill_name,
        "trace_eval": trace_eval,
        "summary": summary,
    }


def _task_profile_session(
    spec: TaskProfileSpec,
    provider: TaskProfileProvider,
) -> AgentSession:
    providers = LLMProviderCenter(default_provider=f"{spec.name}-provider")
    providers.register(
        f"{spec.name}-provider",
        provider,
        default_model=f"{spec.name}-mini",
        default_capabilities=LLMModelCapabilities(
            context_window_tokens=1200,
            max_output_tokens=96,
            supports_tool_calls=True,
        ),
    )

    registry = ToolRegistry()

    @registry.register_function(
        description="Inspect one generic runtime task profile.",
        tags=("task-profile", spec.name),
    )
    def inspect_profile(profile: str) -> str:
        return f"{profile}:{spec.tool_result}"

    tools = ToolCenter()
    tools.mount("local", registry, tags=("task-profile", spec.name))

    skills = SkillRegistry()
    skills.register(
        SkillSpec(
            name=spec.skill_name,
            description=f"{spec.name} task profile skill",
            prompt=spec.skill_prompt,
            tags=(spec.name,),
            priority=10,
        )
    )
    skill_context = SkillsContext(skills)
    skill_context.load(spec.skill_name)

    memory = MemoryCenter(default_store="local")
    memory.register(
        "local",
        InMemoryMemoryStore(
            (
                MemoryRecord(content=spec.memory_content, source=f"{spec.name}-memory"),
            )
        ),
        backend_kind="in_memory",
        namespaces=(spec.name,),
        tags=(spec.name,),
    )

    context_store = ContextMaterialCenter(default_store="local")
    context_store.register(
        "local",
        InMemoryContextMaterialStore(
            (
                ContextMaterial(
                    name=spec.context_name,
                    content=spec.context_content,
                    role="runtime",
                    priority=9,
                    metadata={"source": "runtime", "namespace": spec.name, "tags": (spec.name,)},
                ),
                ContextMaterial(
                    name=f"{spec.name}_low_priority",
                    content="irrelevant background that should lose priority",
                    role="timeline",
                    priority=1,
                    metadata={"source": "runtime", "namespace": spec.name, "tags": (spec.name,)},
                ),
            )
        ),
        backend_kind="in_memory",
        namespace=spec.name,
        tags=(spec.name,),
    )

    return AgentSession(
        profile=AgentProfile(
            name=f"{spec.name}-task-profile",
            model=f"{spec.name}-mini",
            instructions="Use loaded skills, recalled memory, selected context, and tools.",
            capabilities=CapabilitySet(memory_enabled=True, skills=(spec.skill_name,)),
            budget=RuntimeBudget(max_iterations=4, max_prompt_bytes=2200, max_cost_usd=0.1),
        ),
        provider=providers,
        tools=tools,
        skills=skill_context,
        memory=memory,
        context_material_store=context_store,
        context_material_selector=DefaultContextMaterialSelector(),
        context_injection_policy=ContextInjectionPolicy(
            max_injection_bytes=360,
            max_total_bytes=900,
            allowed_targets=(
                PromptBucketRole.SEMI_DYNAMIC_1,
                PromptBucketRole.SEMI_DYNAMIC_2,
                PromptBucketRole.TIMELINE_OPEN,
                PromptBucketRole.DYNAMIC,
            ),
        ),
        harness=PersistentAgentJournal(InMemoryJournalStore()),
        event_sink=ListEventSink(),
        tool_replay=PersistentToolReplay(InMemoryToolReplayStore()),
        trace_store=InMemoryRunTraceStore(),
        policy_decision_store=InMemoryPolicyDecisionStore(),
        metadata={"scenario": "task_profile_acceptance", "profile": spec.name},
    )


def _task_profile_request(
    spec: TaskProfileSpec,
    metadata: dict[str, Any],
) -> AgentRunRequest:
    return AgentRunRequest(
        task=spec.task,
        context=AgentContextPack(
            workspace=(
                f"{spec.name} workspace evidence: {spec.context_content}\n\n"
                "generic host runtime should provide concrete tools separately."
            ),
            dynamic_task=spec.task,
        ),
        context_material_query=ContextMaterialQuery(
            query=spec.task,
            limit=2,
            namespace=spec.name,
            filters={"tags": (spec.name,)},
        ),
        context_material_selection=ContextMaterialSelectionRequest(
            task=spec.task,
            max_materials=1,
            max_bytes=512,
            allowed_targets=(
                PromptBucketRole.SEMI_DYNAMIC_2,
                PromptBucketRole.TIMELINE_OPEN,
                PromptBucketRole.DYNAMIC,
            ),
        ),
        metadata={
            "acceptance": "task_profile",
            "profile": spec.name,
            **dict(metadata),
        },
    )


def _task_profile_trace_spec(spec: TaskProfileSpec) -> TraceEvalSpec:
    return TraceEvalSpec(
        name=f"agent-core-task-profile-{spec.name}",
        expected_status="completed",
        max_iterations=4,
        max_provider_calls=3,
        required_provider_names=(f"{spec.name}-provider",),
        required_provider_models=(f"{spec.name}-mini",),
        require_provider_model_capabilities=True,
        require_provider_route_plan=True,
        required_provider_route_selected_names=(f"{spec.name}-provider",),
        require_tool_center=True,
        required_tool_center_selected_mounts=("local",),
        required_tool_center_selected_tools=("inspect_profile",),
        required_tool_center_requested_tools=("inspect_profile",),
        require_tool_center_ready_routes=True,
        max_tool_center_failed_calls=0,
        require_skill_center=True,
        required_loaded_skills=(spec.skill_name,),
        require_context_material_selection=True,
        required_selected_context_material_names=(spec.context_name,),
        required_context_material_statuses=("selected",),
        require_context_injections=True,
        required_context_injection_sources=("memory", "runtime"),
        required_included_context_injection_sources=("memory", "runtime"),
        max_excluded_context_injections=0,
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
            "policy_decision",
            "context_material",
        ),
        max_failure_count=0,
    )


def _task_profiles() -> tuple[TaskProfileSpec, ...]:
    return (
        TaskProfileSpec(
            name="code",
            task="review changed parser code for correctness and missing tests",
            skill_name="code-review",
            skill_prompt="Prioritize correctness, regression risk, and focused tests.",
            context_name="code_diff_context",
            context_content="parser change touches error handling and needs regression coverage",
            memory_content="Previous code agent runs preserved unrelated user changes.",
            tool_result="diff inspected with focused risk notes",
        ),
        TaskProfileSpec(
            name="ops",
            task="triage failing deployment health check and propose next action",
            skill_name="ops-triage",
            skill_prompt="Prioritize blast radius, rollback state, and observable symptoms.",
            context_name="ops_incident_context",
            context_content="deployment health check failed after config rollout on one region",
            memory_content="Ops agent should prefer reversible checks before remediation.",
            tool_result="health check summarized with rollback candidate",
        ),
        TaskProfileSpec(
            name="security",
            task="assess login csrf evidence and decide verification path",
            skill_name="security-review",
            skill_prompt="Prioritize exploitability, evidence quality, and safe verification.",
            context_name="security_finding_context",
            context_content="login csrf token mismatch observed on admin callback",
            memory_content="Security agent should preserve evidence chain and avoid destructive steps.",
            tool_result="csrf evidence inspected with safe verification path",
        ),
    )


def _task_profile_issues(
    report: dict[str, Any],
) -> tuple[AgentCoreTaskProfileAcceptanceIssue, ...]:
    issues: list[AgentCoreTaskProfileAcceptanceIssue] = []
    name = str(report.get("name") or "")
    if report.get("status") != "completed":
        issues.append(
            AgentCoreTaskProfileAcceptanceIssue(
                source="profile",
                code="profile_run_not_completed",
                message=f"Task profile {name} did not complete.",
                metadata=dict(report),
            )
        )
    if report.get("trace_eval", {}).get("ok") is not True:
        for issue in report.get("trace_eval", {}).get("issues") or ():
            if not isinstance(issue, dict):
                continue
            issues.append(
                AgentCoreTaskProfileAcceptanceIssue(
                    source=f"profile:{name}",
                    code=str(issue.get("code") or "trace_eval_issue"),
                    message=str(issue.get("message") or ""),
                    severity=str(issue.get("severity") or "error"),
                    metadata=dict(issue.get("metadata") or {}),
                )
            )
    required_counts = {
        "tool_call_count": 1,
        "memory_hit_count": 1,
        "context_material_selected_count": 1,
        "context_injection_count": 2,
    }
    for key, minimum in required_counts.items():
        if int(report.get(key) or 0) < minimum:
            issues.append(
                AgentCoreTaskProfileAcceptanceIssue(
                    source=f"profile:{name}",
                    code=f"{key}_too_low",
                    message=f"Task profile {name} did not reach required {key}.",
                    metadata={"actual": int(report.get(key) or 0), "minimum": minimum},
                )
            )
    return tuple(issues)


def _task_profile_matrix(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    names = [str(report.get("name") or "") for report in reports]
    return {
        "schema_version": "agent-core-task-profile-matrix/v1",
        "profile_count": len(reports),
        "profile_names": names,
        "completed_profiles": [
            str(report.get("name") or "")
            for report in reports
            if report.get("status") == "completed"
        ],
        "tool_enabled_profiles": [
            str(report.get("name") or "")
            for report in reports
            if int(report.get("tool_call_count") or 0) >= 1
        ],
        "memory_enabled_profiles": [
            str(report.get("name") or "")
            for report in reports
            if int(report.get("memory_hit_count") or 0) >= 1
        ],
        "context_enabled_profiles": [
            str(report.get("name") or "")
            for report in reports
            if int(report.get("context_material_selected_count") or 0) >= 1
        ],
        "skill_names": [str(report.get("loaded_skill") or "") for report in reports],
        "provider_names": [str(report.get("provider_name") or "") for report in reports],
    }


def _task_profile_matrix_issues(
    matrix: dict[str, Any],
) -> tuple[AgentCoreTaskProfileAcceptanceIssue, ...]:
    issues: list[AgentCoreTaskProfileAcceptanceIssue] = []
    expected = {"code", "ops", "security"}
    profile_names = set(matrix.get("profile_names") or ())
    if profile_names != expected:
        issues.append(
            AgentCoreTaskProfileAcceptanceIssue(
                source="matrix",
                code="task_profiles_missing",
                message="Task profile matrix must cover code, ops, and security.",
                metadata={"expected": sorted(expected), "actual": sorted(profile_names)},
            )
        )
    for key in (
        "completed_profiles",
        "tool_enabled_profiles",
        "memory_enabled_profiles",
        "context_enabled_profiles",
    ):
        values = set(matrix.get(key) or ())
        if values != expected:
            issues.append(
                AgentCoreTaskProfileAcceptanceIssue(
                    source="matrix",
                    code=f"{key}_incomplete",
                    message=f"Task profile matrix did not satisfy {key}.",
                    metadata={"expected": sorted(expected), "actual": sorted(values)},
                )
            )
    return tuple(issues)
