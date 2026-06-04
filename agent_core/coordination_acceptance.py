"""SDK-level planner, handoff, agent-tool, and artifact acceptance checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.artifacts import InMemoryArtifactStore
from agent_core.config import AgentProfile, CapabilitySet
from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.handoff import AgentToolRuntime, HandoffRequest, MultiAgentCoordinator
from agent_core.planner import InMemoryPlanner, PlanExecutor
from agent_core.runner import AgentSession, AgentSessionManager
from agent_core.testing import MockLLMProvider, MockToolRuntime
from agent_core.tools import ToolInvocation
from agent_core.trace import AgentRunTraceBundle, AgentToolTrace, ArtifactTrace, HandoffTrace, PlannerTrace


@dataclass(frozen=True)
class AgentCoreCoordinationAcceptanceIssue:
    """One blocking multi-agent coordination acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-coordination-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreCoordinationAcceptanceReport:
    """Prompt-safe coordination acceptance report."""

    status: str
    planner: dict[str, Any] = field(default_factory=dict)
    handoff: dict[str, Any] = field(default_factory=dict)
    agent_tool: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    traces: dict[str, Any] = field(default_factory=dict)
    trace_bundle: dict[str, Any] = field(default_factory=dict)
    trace_eval: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreCoordinationAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-coordination-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "planner": dict(self.planner),
            "handoff": dict(self.handoff),
            "agent_tool": dict(self.agent_tool),
            "artifacts": dict(self.artifacts),
            "traces": dict(self.traces),
            "trace_bundle": dict(self.trace_bundle),
            "trace_eval": dict(self.trace_eval),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreCoordinationAcceptanceHarness:
    """Run deterministic SDK coordination checks without host-runtime adapters."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreCoordinationAcceptanceReport:
        manager = _coordination_manager()
        planner = InMemoryPlanner()
        executor = PlanExecutor(planner=planner, manager=manager, default_session="worker")
        plan_report = await executor.execute(
            "ship sdk coordination gate",
            context={
                "plan_id": "coordination-plan",
                "steps": (
                    {"step_id": "audit", "goal": "Audit current SDK coordination"},
                    {
                        "step_id": "implement",
                        "goal": "Implement coordination acceptance",
                        "depends_on": ("audit",),
                    },
                ),
            },
        )
        coordinator = MultiAgentCoordinator.from_manager(manager)
        handoff_record = await coordinator.handoff(
            HandoffRequest(
                task="review current SDK coordination patch",
                source_session="planner",
                required_tags=("review",),
                required_tools=("diff",),
                required_skills=("review",),
                metadata={"scenario": "coordination_acceptance"},
            )
        )
        agent_tools = AgentToolRuntime.from_manager(manager)
        agent_tool_result = await agent_tools.invoke(
            ToolInvocation(
                tool_name="agent_code_reviewer",
                arguments={"task": "review private patch"},
            )
        )
        artifact_store = InMemoryArtifactStore()
        await artifact_store.put_text(
            "coordination evidence: planner completed, reviewer selected, agent tool completed",
            content_type="text/markdown",
            metadata={
                "kind": "coordination_evidence",
                "tool_name": "agent_code_reviewer",
                "status": agent_tool_result.status,
            },
        )

        planner_trace = PlannerTrace.from_session(
            {"planner": planner.manifest(), "plan_executor": executor.manifest()}
        ).manifest()
        handoff_trace = HandoffTrace.from_prompt(
            {"metadata": {"handoff": handoff_record.decision.manifest()}}
        ).manifest()
        agent_tool_trace = AgentToolTrace.from_tool_center(
            {"calls": [{"result": agent_tool_result.manifest()}]}
        ).manifest()
        artifact_trace = ArtifactTrace.from_session(
            {"artifact_store": artifact_store.manifest()}
        ).manifest()
        trace_bundle = AgentRunTraceBundle(
            run_id="coordination-acceptance",
            status="completed",
            journal_replay={"ok": True, "event_count": 0, "events": []},
            planner_trace=planner_trace,
            handoff_trace=handoff_trace,
            agent_tool_trace=agent_tool_trace,
            artifact_trace=artifact_trace,
            prompt={"metadata": {"handoff": handoff_record.decision.manifest()}},
            session={
                "tools": {"calls": [{"result": agent_tool_result.manifest()}]},
                "artifact_store": artifact_store.manifest(),
            },
            metadata={"scenario": "agent_core_coordination_acceptance"},
        ).manifest()
        trace_eval = DefaultTraceEvaluator().evaluate(
            trace_bundle,
            TraceEvalSpec(
                name="coordination-acceptance",
                expected_status="completed",
                require_journal_ok=True,
                require_planner_trace=True,
                required_plan_ids=("coordination-plan",),
                required_plan_step_ids=("audit", "implement"),
                required_plan_step_statuses=("completed",),
                required_plan_execution_statuses=("completed",),
                max_failed_plan_steps=0,
                max_blocked_plan_reports=0,
                require_handoff=True,
                required_handoff_statuses=("selected",),
                required_handoff_selected_sessions=("code-reviewer",),
                required_handoff_source_sessions=("planner",),
                max_handoff_denied=0,
                max_handoff_not_found=0,
                require_agent_tools=True,
                required_agent_tool_names=("agent_code_reviewer",),
                required_agent_tool_sessions=("code-reviewer",),
                required_agent_tool_statuses=("completed",),
                max_agent_tool_failures=0,
                require_artifacts=True,
                required_artifact_kinds=("coordination_evidence",),
                required_artifact_tool_names=("agent_code_reviewer",),
                required_artifact_content_types=("text/markdown",),
                max_artifact_count=1,
                max_artifact_total_bytes=512,
                max_artifact_size_bytes=512,
            ),
        ).manifest()

        planner_manifest = plan_report.manifest()
        handoff_manifest = handoff_record.manifest()
        agent_tool_manifest = {
            "schema_version": "agent-core-coordination-agent-tool/v1",
            "runtime": agent_tools.manifest(),
            "result": agent_tool_result.manifest(),
        }
        artifact_manifest = artifact_store.manifest()
        traces = {
            "planner": planner_trace,
            "handoff": handoff_trace,
            "agent_tool": agent_tool_trace,
            "artifact": artifact_trace,
        }
        issues = _coordination_acceptance_issues(
            planner=planner_manifest,
            handoff=handoff_manifest,
            agent_tool=agent_tool_manifest,
            artifacts=artifact_manifest,
            traces=traces,
            trace_bundle=trace_bundle,
            trace_eval=trace_eval,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreCoordinationAcceptanceReport(
            status=status,
            planner=planner_manifest,
            handoff=handoff_manifest,
            agent_tool=agent_tool_manifest,
            artifacts=artifact_manifest,
            traces=traces,
            trace_bundle=trace_bundle,
            trace_eval=trace_eval,
            issues=issues,
            metadata={"scenario": "agent_core_coordination_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_coordination_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreCoordinationAcceptanceReport:
    """Run the default planner/handoff/agent-tool/artifact acceptance checks."""

    return await AgentCoreCoordinationAcceptanceHarness(metadata=dict(metadata or {})).run()


def _coordination_manager() -> AgentSessionManager:
    manager = AgentSessionManager()
    manager.register(
        AgentSession(
            profile=AgentProfile(name="worker"),
            provider=MockLLMProvider(
                [
                    {"action": "finish", "arguments": {"output": "audit done"}},
                    {"action": "finish", "arguments": {"output": "implement done"}},
                ]
            ),
            tools=MockToolRuntime(),
        ),
        name="worker",
    )
    manager.register(
        AgentSession(
            profile=AgentProfile(
                name="code-reviewer",
                instructions="Review code changes and return prompt-safe findings.",
                capabilities=CapabilitySet(tools=("diff",), skills=("review",)),
            ),
            provider=MockLLMProvider(
                [
                    {"action": "finish", "arguments": {"output": "reviewed handoff"}},
                    {"action": "finish", "arguments": {"output": "reviewed tool"}},
                ]
            ),
            tools=MockToolRuntime({"diff": "patch ok"}),
            metadata={
                "tags": ("code", "review"),
                "handoff_priority": 10,
            },
        ),
        name="code-reviewer",
    )
    return manager


def _coordination_acceptance_issues(
    *,
    planner: dict[str, Any],
    handoff: dict[str, Any],
    agent_tool: dict[str, Any],
    artifacts: dict[str, Any],
    traces: dict[str, Any],
    trace_bundle: dict[str, Any],
    trace_eval: dict[str, Any],
) -> tuple[AgentCoreCoordinationAcceptanceIssue, ...]:
    issues: list[AgentCoreCoordinationAcceptanceIssue] = []
    if planner.get("status") != "completed":
        issues.append(
            AgentCoreCoordinationAcceptanceIssue(
                source="planner",
                code="plan_execution_not_completed",
                message="Plan executor did not complete the acceptance plan.",
                metadata={"planner": dict(planner)},
            )
        )
    plan = planner.get("plan") if isinstance(planner.get("plan"), dict) else {}
    status_counts = plan.get("status_counts") if isinstance(plan.get("status_counts"), dict) else {}
    if int(status_counts.get("completed") or 0) < 2:
        issues.append(
            AgentCoreCoordinationAcceptanceIssue(
                source="planner",
                code="plan_steps_not_completed",
                message="Plan trace did not record both coordination steps as completed.",
                metadata={"status_counts": dict(status_counts)},
            )
        )
    decision = handoff.get("decision") if isinstance(handoff.get("decision"), dict) else {}
    if decision.get("status") != "selected" or decision.get("selected_session") != "code-reviewer":
        issues.append(
            AgentCoreCoordinationAcceptanceIssue(
                source="handoff",
                code="handoff_target_not_selected",
                message="Handoff did not select the expected reviewer session.",
                metadata={"decision": dict(decision)},
            )
        )
    outcome = handoff.get("outcome") if isinstance(handoff.get("outcome"), dict) else {}
    outcome_result = outcome.get("result") if isinstance(outcome.get("result"), dict) else {}
    if outcome_result.get("status") != "completed":
        issues.append(
            AgentCoreCoordinationAcceptanceIssue(
                source="handoff",
                code="handoff_child_run_not_completed",
                message="Selected handoff session did not complete.",
                metadata={"outcome": dict(outcome)},
            )
        )
    agent_result = agent_tool.get("result") if isinstance(agent_tool.get("result"), dict) else {}
    if agent_result.get("status") != "completed":
        issues.append(
            AgentCoreCoordinationAcceptanceIssue(
                source="agent_tool",
                code="agent_tool_not_completed",
                message="Agent-as-tool invocation did not complete.",
                metadata={"result": dict(agent_result)},
            )
        )
    runtime = agent_tool.get("runtime") if isinstance(agent_tool.get("runtime"), dict) else {}
    if int(runtime.get("record_count") or 0) != 1:
        issues.append(
            AgentCoreCoordinationAcceptanceIssue(
                source="agent_tool",
                code="agent_tool_record_missing",
                message="AgentToolRuntime did not record exactly one delegated call.",
                metadata={"runtime": dict(runtime)},
            )
        )
    artifact_records = tuple(
        dict(item) for item in artifacts.get("artifacts") or () if isinstance(item, dict)
    )
    if len(artifact_records) != 1:
        issues.append(
            AgentCoreCoordinationAcceptanceIssue(
                source="artifacts",
                code="artifact_count_unexpected",
                message="Artifact store did not record exactly one coordination artifact.",
                metadata={"artifact_count": len(artifact_records)},
            )
        )
    elif artifact_records[0].get("content_type") != "text/markdown":
        issues.append(
            AgentCoreCoordinationAcceptanceIssue(
                source="artifacts",
                code="artifact_content_type_unexpected",
                message="Coordination artifact content type is not text/markdown.",
                metadata={"artifact": dict(artifact_records[0])},
            )
        )
    _extend_trace_issues(issues, traces=traces, trace_bundle=trace_bundle, trace_eval=trace_eval)
    return tuple(issues)


def _extend_trace_issues(
    issues: list[AgentCoreCoordinationAcceptanceIssue],
    *,
    traces: dict[str, Any],
    trace_bundle: dict[str, Any],
    trace_eval: dict[str, Any],
) -> None:
    planner_trace = dict(traces.get("planner") or {})
    if int(planner_trace.get("plan_count") or 0) < 1 or int(
        planner_trace.get("execution_report_count") or 0
    ) < 1:
        issues.append(
            AgentCoreCoordinationAcceptanceIssue(
                source="planner_trace",
                code="planner_trace_missing",
                message="PlannerTrace did not summarize the acceptance plan.",
                metadata={"trace": planner_trace},
            )
        )
    handoff_trace = dict(traces.get("handoff") or {})
    if int(handoff_trace.get("selected_count") or 0) < 1:
        issues.append(
            AgentCoreCoordinationAcceptanceIssue(
                source="handoff_trace",
                code="handoff_trace_missing_selection",
                message="HandoffTrace did not record the selected reviewer.",
                metadata={"trace": handoff_trace},
            )
        )
    agent_tool_trace = dict(traces.get("agent_tool") or {})
    if int(agent_tool_trace.get("completed_count") or 0) < 1:
        issues.append(
            AgentCoreCoordinationAcceptanceIssue(
                source="agent_tool_trace",
                code="agent_tool_trace_missing_completion",
                message="AgentToolTrace did not record the completed child agent call.",
                metadata={"trace": agent_tool_trace},
            )
        )
    artifact_trace = dict(traces.get("artifact") or {})
    if int(artifact_trace.get("artifact_count") or 0) != 1:
        issues.append(
            AgentCoreCoordinationAcceptanceIssue(
                source="artifact_trace",
                code="artifact_trace_count_unexpected",
                message="ArtifactTrace did not record exactly one artifact.",
                metadata={"trace": artifact_trace},
            )
        )
    summary = trace_bundle.get("summary") if isinstance(trace_bundle.get("summary"), dict) else {}
    if int(summary.get("planner_execution_report_count") or 0) < 1:
        issues.append(
            AgentCoreCoordinationAcceptanceIssue(
                source="trace_bundle",
                code="trace_bundle_planner_missing",
                message="Run trace bundle did not include planner coordination evidence.",
                metadata={"summary": dict(summary)},
            )
        )
    if trace_eval.get("ok") is not True:
        issues.append(
            AgentCoreCoordinationAcceptanceIssue(
                source="trace_eval",
                code="coordination_trace_eval_failed",
                message="Trace evaluator did not accept the coordination trace contract.",
                metadata={"trace_eval": dict(trace_eval)},
            )
        )
