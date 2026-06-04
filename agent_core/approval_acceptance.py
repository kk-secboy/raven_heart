"""SDK-level policy and approval acceptance checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.approvals import (
    ApprovalCenter,
    ApprovalDecisionRecord,
    ApprovalResumeContext,
    InMemoryApprovalStore,
)
from agent_core.config import AgentProfile, RuntimeBudget
from agent_core.events import ListEventSink
from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.harness import InMemoryAgentJournal
from agent_core.policy import InMemoryPolicyDecisionStore, PolicyRule, RuleBasedPolicy
from agent_core.providers import LLMRequest, LLMResponse
from agent_core.runner import AgentRunRequest, AgentRunner, AgentSession
from agent_core.tools import ToolInvocation, ToolRegistry, ToolResult, ToolSpec
from agent_core.trace import InMemoryRunTraceStore


@dataclass(frozen=True)
class AgentCoreApprovalAcceptanceIssue:
    """One blocking policy/approval acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-approval-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreApprovalAcceptanceReport:
    """Prompt-safe policy/approval acceptance report."""

    status: str
    blocked_run: dict[str, Any] = field(default_factory=dict)
    approved_run: dict[str, Any] = field(default_factory=dict)
    approval_queue: dict[str, Any] = field(default_factory=dict)
    approval_resume: dict[str, Any] = field(default_factory=dict)
    policy_decisions: dict[str, Any] = field(default_factory=dict)
    blocked_trace_eval: dict[str, Any] = field(default_factory=dict)
    approved_trace_eval: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreApprovalAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-approval-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "blocked_run": dict(self.blocked_run),
            "approved_run": dict(self.approved_run),
            "approval_queue": dict(self.approval_queue),
            "approval_resume": dict(self.approval_resume),
            "policy_decisions": dict(self.policy_decisions),
            "blocked_trace_eval": dict(self.blocked_trace_eval),
            "approved_trace_eval": dict(self.approved_trace_eval),
            "metadata": dict(self.metadata),
        }


class ApprovalAcceptanceProvider:
    """Deterministic provider for approval gate checks."""

    def __init__(self, output: str = "done") -> None:
        self.requests: list[LLMRequest] = []
        self._responses: list[LLMResponse] = [
            LLMResponse(action={"action": "deploy", "arguments": {}}),
            LLMResponse(action={"action": "finish", "arguments": {"output": output}}),
        ]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(action={"action": "finish", "arguments": {"output": "done"}})


@dataclass(frozen=True)
class AgentCoreApprovalAcceptanceHarness:
    """Run deterministic policy/approval acceptance checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreApprovalAcceptanceReport:
        approval_store = InMemoryApprovalStore()
        approval_center = ApprovalCenter(approval_store)
        policy_decision_store = InMemoryPolicyDecisionStore()

        blocked_provider = ApprovalAcceptanceProvider(output="blocked-awaiting-approval")
        blocked_outcome = await AgentRunner(
            _approval_session(
                provider=blocked_provider,
                approval_store=approval_store,
                policy_decision_store=policy_decision_store,
            )
        ).run(
            AgentRunRequest(
                task="deploy release",
                metadata={"acceptance": "approval_block", **dict(self.metadata)},
            )
        )
        pending = await approval_store.pending()
        blocked_eval = DefaultTraceEvaluator().evaluate(
            blocked_outcome.trace_manifest,
            _blocked_trace_spec(),
        )

        resolved = None
        if pending:
            resolved = await approval_store.decide(
                pending[0].approval_id,
                ApprovalDecisionRecord(
                    status="approved",
                    actor="operator",
                    reason="SDK approval acceptance",
                    metadata={"scenario": "agent_core_approval_acceptance"},
                ),
            )
        resume = ApprovalResumeContext.from_records((resolved,)) if resolved else ApprovalResumeContext()

        approved_provider = ApprovalAcceptanceProvider(output="approved-deployed")
        approved_outcome = await AgentRunner(
            _approval_session(
                provider=approved_provider,
                approval_store=approval_store,
                policy_decision_store=policy_decision_store,
            )
        ).run(
            AgentRunRequest(
                task="deploy release after approval",
                approval_resume=resume,
                metadata={"acceptance": "approval_resume", **dict(self.metadata)},
            )
        )
        approved_eval = DefaultTraceEvaluator().evaluate(
            approved_outcome.trace_manifest,
            _approved_trace_spec(),
        )
        approval_queue = await approval_center.manifest()
        policy_manifest = policy_decision_store.manifest()
        issues = _approval_acceptance_issues(
            blocked_trace_eval=blocked_eval.manifest(),
            approved_trace_eval=approved_eval.manifest(),
            blocked_trace=blocked_outcome.trace_manifest,
            approved_trace=approved_outcome.trace_manifest,
            approval_queue=approval_queue,
            approval_resume=resume.manifest(),
            policy_decisions=policy_manifest,
            blocked_provider_request_count=len(blocked_provider.requests),
            approved_provider_request_count=len(approved_provider.requests),
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreApprovalAcceptanceReport(
            status=status,
            blocked_run=_run_summary(blocked_outcome.trace_manifest, blocked_provider),
            approved_run=_run_summary(approved_outcome.trace_manifest, approved_provider),
            approval_queue=approval_queue,
            approval_resume=resume.manifest(),
            policy_decisions=policy_manifest,
            blocked_trace_eval=blocked_eval.manifest(),
            approved_trace_eval=approved_eval.manifest(),
            issues=issues,
            metadata={"scenario": "agent_core_approval_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_approval_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreApprovalAcceptanceReport:
    """Run the default policy/approval acceptance checks."""

    return await AgentCoreApprovalAcceptanceHarness(metadata=dict(metadata or {})).run()


def _approval_session(
    *,
    provider: ApprovalAcceptanceProvider,
    approval_store: InMemoryApprovalStore,
    policy_decision_store: InMemoryPolicyDecisionStore,
) -> AgentSession:
    registry = ToolRegistry()

    async def deploy(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            content="release deployed",
        )

    registry.register(ToolSpec(name="deploy", tags=("release",)), deploy)
    return AgentSession(
        profile=AgentProfile(
            name="approval-acceptance",
            model="approval-mini",
            instructions="Run approval acceptance safely.",
            budget=RuntimeBudget(max_iterations=3),
        ),
        provider=provider,
        tools=registry,
        harness=InMemoryAgentJournal(),
        event_sink=ListEventSink(),
        trace_store=InMemoryRunTraceStore(),
        approval_store=approval_store,
        policy_decision_store=policy_decision_store,
        policy=RuleBasedPolicy(
            (
                PolicyRule(
                    name="release-approval",
                    status="approval_required",
                    tool_names=("deploy",),
                    reason="release deployment requires approval",
                ),
            )
        ),
        metadata={"scenario": "agent_core_approval_acceptance"},
    )


def _blocked_trace_spec() -> TraceEvalSpec:
    return TraceEvalSpec(
        name="agent-core-approval-block-acceptance",
        expected_status="completed",
        max_provider_calls=2,
        require_approvals=True,
        required_approval_statuses=("pending",),
        required_approval_subjects=("tool:deploy",),
        required_approval_subject_kinds=("tool",),
        max_rejected_approvals=0,
        require_event_log=True,
        required_event_log_types=("approval_requested",),
        require_event_sequence_monotonic=True,
        require_terminal_event=True,
        max_failure_count=1,
    )


def _approved_trace_spec() -> TraceEvalSpec:
    return TraceEvalSpec(
        name="agent-core-approval-resume-acceptance",
        expected_status="completed",
        max_provider_calls=2,
        require_approvals=True,
        required_approval_statuses=("approved",),
        required_approval_subjects=("tool:deploy",),
        required_approval_subject_kinds=("tool",),
        require_approved_approval_subjects=("tool:deploy",),
        max_pending_approvals=0,
        max_rejected_approvals=0,
        require_context_injections=True,
        required_context_injection_names=("approval_resume",),
        required_context_injection_sources=("approval",),
        required_included_context_injection_sources=("approval",),
        require_event_log=True,
        required_event_log_types=("approval_resumed", "tool_finished"),
        require_event_sequence_monotonic=True,
        require_terminal_event=True,
        max_failure_count=0,
    )


def _run_summary(trace: dict[str, Any], provider: ApprovalAcceptanceProvider) -> dict[str, Any]:
    summary = dict(trace.get("summary") or {})
    return {
        "run_id": trace.get("run_id", "") or str(summary.get("run_id") or ""),
        "status": trace.get("status", "") or str(summary.get("status") or ""),
        "provider_request_count": len(provider.requests),
        "approval_record_count": int(summary.get("approval_record_count") or 0),
        "approval_pending_count": int(summary.get("approval_pending_count") or 0),
        "approval_approved_count": int(summary.get("approval_approved_count") or 0),
        "context_injection_count": int(summary.get("context_injection_count") or 0),
        "failure_count": int(summary.get("failure_count") or 0),
    }


def _approval_acceptance_issues(
    *,
    blocked_trace_eval: dict[str, Any],
    approved_trace_eval: dict[str, Any],
    blocked_trace: dict[str, Any],
    approved_trace: dict[str, Any],
    approval_queue: dict[str, Any],
    approval_resume: dict[str, Any],
    policy_decisions: dict[str, Any],
    blocked_provider_request_count: int,
    approved_provider_request_count: int,
) -> tuple[AgentCoreApprovalAcceptanceIssue, ...]:
    issues: list[AgentCoreApprovalAcceptanceIssue] = []
    _extend_eval_issues(issues, "blocked_trace_eval", blocked_trace_eval)
    _extend_eval_issues(issues, "approved_trace_eval", approved_trace_eval)
    blocked_summary = dict(blocked_trace.get("summary") or {})
    approved_summary = dict(approved_trace.get("summary") or {})
    if int(blocked_summary.get("approval_pending_count") or 0) < 1:
        issues.append(
            AgentCoreApprovalAcceptanceIssue(
                source="approval_queue",
                code="pending_approval_missing",
                message="Blocked approval run did not leave a pending approval.",
                metadata={"summary": blocked_summary},
            )
        )
    if int(approved_summary.get("approval_approved_count") or 0) < 1:
        issues.append(
            AgentCoreApprovalAcceptanceIssue(
                source="approval_queue",
                code="approved_record_missing",
                message="Approved run did not expose an approved approval record.",
                metadata={"summary": approved_summary},
            )
        )
    if int(approved_summary.get("context_injection_count") or 0) < 1:
        issues.append(
            AgentCoreApprovalAcceptanceIssue(
                source="approval_resume",
                code="approval_resume_not_injected",
                message="Approved run did not inject approval resume context.",
                metadata={"summary": approved_summary},
            )
        )
    if int(approval_resume.get("approved_count") or 0) < 1:
        issues.append(
            AgentCoreApprovalAcceptanceIssue(
                source="approval_resume",
                code="approval_resume_grant_missing",
                message="Approval resume context did not contain an approved grant.",
                metadata={"approval_resume": dict(approval_resume)},
            )
        )
    queue = dict(approval_queue.get("queue") or {})
    if int(queue.get("approved_count") or 0) < 1:
        issues.append(
            AgentCoreApprovalAcceptanceIssue(
                source="approval_center",
                code="approval_center_approved_count_missing",
                message="Approval center did not expose the approved decision.",
                metadata={"queue": queue},
            )
        )
    if int(queue.get("pending_count") or 0) != 0:
        issues.append(
            AgentCoreApprovalAcceptanceIssue(
                source="approval_center",
                code="approval_center_pending_not_cleared",
                message="Approval center still has pending approvals after decision.",
                metadata={"queue": queue},
            )
        )
    if int(policy_decisions.get("record_count") or 0) < 2:
        issues.append(
            AgentCoreApprovalAcceptanceIssue(
                source="policy_decision",
                code="policy_decision_records_missing",
                message="Policy decision audit did not record both approval checks.",
                metadata={"policy_decisions": dict(policy_decisions)},
            )
        )
    if blocked_provider_request_count < 2 or approved_provider_request_count < 2:
        issues.append(
            AgentCoreApprovalAcceptanceIssue(
                source="provider",
                code="provider_loop_incomplete",
                message="Approval acceptance provider loop did not complete both turns.",
                metadata={
                    "blocked_provider_request_count": blocked_provider_request_count,
                    "approved_provider_request_count": approved_provider_request_count,
                },
            )
        )
    return tuple(issues)


def _extend_eval_issues(
    issues: list[AgentCoreApprovalAcceptanceIssue],
    source: str,
    trace_eval: dict[str, Any],
) -> None:
    if trace_eval.get("ok") is True:
        return
    for issue in trace_eval.get("issues") or ():
        if not isinstance(issue, dict):
            continue
        issues.append(
            AgentCoreApprovalAcceptanceIssue(
                source=source,
                code=str(issue.get("code") or f"{source}_issue"),
                message=str(issue.get("message") or ""),
                severity=str(issue.get("severity") or "error"),
                metadata=dict(issue.get("metadata") or {}),
            )
        )
    if not any(issue.source == source for issue in issues):
        issues.append(
            AgentCoreApprovalAcceptanceIssue(
                source=source,
                code=f"{source}_failed",
                message=f"{source} did not pass.",
                metadata={"trace_eval": dict(trace_eval)},
            )
        )
