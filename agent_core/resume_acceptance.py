"""SDK-level checkpoint/resume acceptance checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.config import AgentProfile
from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.harness import InMemoryJournalStore, PersistentAgentJournal
from agent_core.providers import LLMRequest, LLMResponse
from agent_core.runner import AgentResumeRequest, AgentRunner, AgentSession
from agent_core.tools import ToolRegistry


@dataclass(frozen=True)
class AgentCoreResumeAcceptanceIssue:
    """One blocking checkpoint/resume acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-resume-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreResumeAcceptanceReport:
    """Prompt-safe checkpoint/resume acceptance report."""

    status: str
    resume_index: dict[str, Any] = field(default_factory=dict)
    resume_plan: dict[str, Any] = field(default_factory=dict)
    resume_manifest: dict[str, Any] = field(default_factory=dict)
    trace_eval: dict[str, Any] = field(default_factory=dict)
    trace_summary: dict[str, Any] = field(default_factory=dict)
    run_summary: dict[str, Any] = field(default_factory=dict)
    terminal_rejection: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreResumeAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-resume-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "resume_index": dict(self.resume_index),
            "resume_plan": dict(self.resume_plan),
            "resume_manifest": dict(self.resume_manifest),
            "trace_eval": dict(self.trace_eval),
            "trace_summary": dict(self.trace_summary),
            "run_summary": dict(self.run_summary),
            "terminal_rejection": dict(self.terminal_rejection),
            "metadata": dict(self.metadata),
        }


class ResumeAcceptanceProvider:
    """Deterministic provider for the resume acceptance scenario."""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(action={"action": "finish", "arguments": {"output": "resumed-ok"}})


def _request_prompt_text(request: LLMRequest) -> str:
    return "\n\n".join(
        message.content
        for message in request.messages
        if message.metadata.get("agent_core_prompt")
    )


@dataclass(frozen=True)
class AgentCoreResumeAcceptanceHarness:
    """Run a deterministic checkpoint/resume acceptance scenario."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreResumeAcceptanceReport:
        journal = PersistentAgentJournal(InMemoryJournalStore())
        source_run = await journal.start_run(
            "initial checkpoint task",
            metadata={"scenario": "resume_acceptance"},
        )
        source_turn = await journal.start_turn(source_run, 0)
        checkpoint = await journal.checkpoint(
            source_turn,
            {
                "observation": "admin UI found",
                "last_tool": "lookup",
                "next_step": "summarize resumed state",
            },
        )
        provider = ResumeAcceptanceProvider()
        runner = AgentRunner(
            AgentSession(
                profile=AgentProfile(name="resume-acceptance", model="resume-mini"),
                provider=provider,
                tools=ToolRegistry(),
                harness=journal,
            )
        )
        request = AgentResumeRequest(
            run_id=source_run.run_id,
            task="continue from checkpoint",
            metadata={"acceptance": "resume", **dict(self.metadata)},
        )
        plan = runner.resume_plan(request)
        outcome = await runner.resume(request)
        trace_eval = DefaultTraceEvaluator().evaluate(
            outcome.trace_manifest,
            _resume_acceptance_trace_spec(checkpoint.checkpoint_id),
        )
        terminal_rejection = await _terminal_rejection_manifest(journal)
        issues = _resume_acceptance_issues(
            plan=plan.summary_manifest(),
            resume=outcome.resume_manifest,
            trace_eval=trace_eval.manifest(),
            trace=outcome.trace_manifest,
            terminal_rejection=terminal_rejection,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        trace_summary = dict(outcome.trace_manifest.get("summary") or {})
        return AgentCoreResumeAcceptanceReport(
            status=status,
            resume_index=runner.resume_index().manifest(),
            resume_plan=plan.summary_manifest(),
            resume_manifest=outcome.resume_manifest,
            trace_eval=trace_eval.manifest(),
            trace_summary=trace_summary,
            run_summary={
                "source_run_id": source_run.run_id,
                "resumed_run_id": outcome.result.run_id,
                "status": outcome.result.status,
                "output": outcome.result.output,
                "checkpoint_id": checkpoint.checkpoint_id,
                "provider_request_count": len(provider.requests),
                "resume_prompt_injected": bool(
                    provider.requests
                    and "== Resumed Checkpoint ==" in _request_prompt_text(provider.requests[0])
                ),
            },
            terminal_rejection=terminal_rejection,
            issues=issues,
            metadata={"scenario": "agent_core_resume_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_resume_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreResumeAcceptanceReport:
    """Run the default checkpoint/resume acceptance scenario."""

    return await AgentCoreResumeAcceptanceHarness(metadata=dict(metadata or {})).run()


def _resume_acceptance_trace_spec(checkpoint_id: str) -> TraceEvalSpec:
    return TraceEvalSpec(
        name="agent-core-resume-acceptance",
        expected_status="completed",
        max_provider_calls=1,
        require_resume=True,
        require_resume_plan=True,
        require_resume_plan_ready=True,
        expected_resume_checkpoint_id=checkpoint_id,
        require_context_injections=True,
        required_context_injection_names=("resume_checkpoint",),
        required_context_injection_sources=("harness",),
        required_included_context_injection_sources=("harness",),
        require_journal_ok=True,
        max_failure_count=0,
    )


async def _terminal_rejection_manifest(journal: PersistentAgentJournal) -> dict[str, Any]:
    terminal_run = await journal.start_run("terminal checkpoint task")
    terminal_turn = await journal.start_turn(terminal_run, 0)
    terminal_checkpoint = await journal.checkpoint(terminal_turn, {"terminal": True})
    await journal.finish_run(terminal_run, "completed", {"output": "done"})
    plan = AgentRunner(
        AgentSession(
            profile=AgentProfile(name="resume-terminal-check"),
            provider=ResumeAcceptanceProvider(),
            tools=ToolRegistry(),
            harness=journal,
        )
    ).resume_plan(
        AgentResumeRequest(
            run_id=terminal_run.run_id,
            task="strict terminal resume",
            allow_terminal=False,
        )
    )
    summary = plan.summary_manifest()
    return {
        "schema_version": "agent-core-resume-terminal-rejection/v1",
        "checkpoint_id": terminal_checkpoint.checkpoint_id,
        "status": summary.get("status"),
        "ready": summary.get("ready"),
        "issue_codes": list(summary.get("issue_codes") or ()),
        "plan": plan.manifest(),
    }


def _resume_acceptance_issues(
    *,
    plan: dict[str, Any],
    resume: dict[str, Any],
    trace_eval: dict[str, Any],
    trace: dict[str, Any],
    terminal_rejection: dict[str, Any],
) -> tuple[AgentCoreResumeAcceptanceIssue, ...]:
    issues: list[AgentCoreResumeAcceptanceIssue] = []
    if plan.get("ready") is not True:
        issues.append(
            AgentCoreResumeAcceptanceIssue(
                source="resume_plan",
                code="resume_plan_not_ready",
                message="Resume plan was not ready.",
                metadata={"plan": dict(plan)},
            )
        )
    if not resume.get("checkpoint_id"):
        issues.append(
            AgentCoreResumeAcceptanceIssue(
                source="resume_manifest",
                code="resume_checkpoint_missing",
                message="Resume manifest did not include a checkpoint id.",
            )
        )
    if trace_eval.get("ok") is not True:
        for issue in trace_eval.get("issues") or ():
            if not isinstance(issue, dict):
                continue
            issues.append(
                AgentCoreResumeAcceptanceIssue(
                    source="trace_eval",
                    code=str(issue.get("code") or "trace_eval_issue"),
                    message=str(issue.get("message") or ""),
                    severity=str(issue.get("severity") or "error"),
                    metadata=dict(issue.get("metadata") or {}),
                )
            )
    summary = trace.get("summary") or {}
    if summary.get("resume_plan_ready") is not True:
        issues.append(
            AgentCoreResumeAcceptanceIssue(
                source="trace_summary",
                code="trace_resume_plan_not_ready",
                message="Trace summary did not record a ready resume plan.",
            )
        )
    if terminal_rejection.get("ready") is not False:
        issues.append(
            AgentCoreResumeAcceptanceIssue(
                source="terminal_rejection",
                code="terminal_resume_not_rejected",
                message="Strict terminal resume plan was not rejected.",
            )
        )
    if "terminal_checkpoint_not_allowed" not in set(terminal_rejection.get("issue_codes") or ()):
        issues.append(
            AgentCoreResumeAcceptanceIssue(
                source="terminal_rejection",
                code="terminal_resume_issue_missing",
                message="Strict terminal resume did not report terminal_checkpoint_not_allowed.",
            )
        )
    return tuple(issues)
