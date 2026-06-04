"""SDK-level input, policy, and output guardrail acceptance checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.config import AgentProfile
from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.policy import InMemoryPolicyDecisionStore, PolicyRule, RuleBasedPolicy
from agent_core.preflight import AgentRunPreflightRequirements
from agent_core.runner import AgentRunRequest, AgentRunner, AgentSession
from agent_core.structured import JsonStructuredOutputValidator, StructuredOutputSpec
from agent_core.testing import MockLLMProvider, MockToolRuntime
from agent_core.tools import ToolInvocation, ToolRegistry, ToolResult, ToolSpec


@dataclass(frozen=True)
class AgentCoreGuardrailAcceptanceIssue:
    """One blocking guardrail acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-guardrail-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreGuardrailAcceptanceReport:
    """Prompt-safe guardrail acceptance report."""

    status: str
    preflight: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    structured_output: dict[str, Any] = field(default_factory=dict)
    trace_evals: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreGuardrailAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-guardrail-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "preflight": dict(self.preflight),
            "policy": dict(self.policy),
            "structured_output": dict(self.structured_output),
            "trace_evals": dict(self.trace_evals),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreGuardrailAcceptanceHarness:
    """Run deterministic SDK guardrail checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreGuardrailAcceptanceReport:
        preflight = await _preflight_guardrail_acceptance()
        policy = await _policy_guardrail_acceptance()
        structured_output = await _structured_output_guardrail_acceptance()
        trace_evals = {
            "preflight": DefaultTraceEvaluator()
            .evaluate(preflight["trace"], _preflight_trace_spec())
            .manifest(),
            "policy": DefaultTraceEvaluator()
            .evaluate(policy["trace"], _policy_trace_spec())
            .manifest(),
            "structured_output": DefaultTraceEvaluator()
            .evaluate(structured_output["trace"], _structured_output_trace_spec())
            .manifest(),
        }
        issues = _guardrail_acceptance_issues(
            preflight=preflight,
            policy=policy,
            structured_output=structured_output,
            trace_evals=trace_evals,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreGuardrailAcceptanceReport(
            status=status,
            preflight=_prompt_safe(preflight),
            policy=_prompt_safe(policy),
            structured_output=_prompt_safe(structured_output),
            trace_evals=trace_evals,
            issues=issues,
            metadata={"scenario": "agent_core_guardrail_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_guardrail_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreGuardrailAcceptanceReport:
    """Run the default SDK guardrail acceptance checks."""

    return await AgentCoreGuardrailAcceptanceHarness(metadata=dict(metadata or {})).run()


async def _preflight_guardrail_acceptance() -> dict[str, Any]:
    session = AgentSession(
        profile=AgentProfile(name="guardrail-preflight"),
        provider=MockLLMProvider([]),
        tools=MockToolRuntime(),
    )
    outcome = await AgentRunner(session).run(
        AgentRunRequest(
            task="oversized",
            preflight_requirements=AgentRunPreflightRequirements(
                max_task_bytes=4,
                required_tools=("scan",),
            ),
        )
    )
    summary = dict(outcome.trace_manifest.get("summary") or {})
    preflight = dict(outcome.trace_manifest.get("preflight") or {})
    return {
        "schema_version": "agent-core-guardrail-preflight-acceptance/v1",
        "status": outcome.result.status,
        "output": outcome.result.output,
        "preflight_status": preflight.get("status"),
        "blocking_codes": list(preflight.get("blocking_codes") or ()),
        "blocking_count": int(preflight.get("blocking_count") or 0),
        "summary": _guardrail_summary(summary),
        "trace": outcome.trace_manifest,
    }


async def _policy_guardrail_acceptance() -> dict[str, Any]:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="delete_target",
            description="Delete a target file.",
            tags=("destructive",),
        ),
        _delete_target,
    )
    provider = MockLLMProvider(
        [
            {
                "action": "delete_target",
                "arguments": {"path": "target.txt"},
            },
            {"action": "finish", "arguments": {"output": "safe"}},
        ]
    )
    policy_store = InMemoryPolicyDecisionStore()
    session = AgentSession(
        profile=AgentProfile(name="guardrail-policy"),
        provider=provider,
        tools=registry,
        policy=RuleBasedPolicy(
            (
                PolicyRule(
                    name="deny-destructive",
                    status="deny",
                    tool_tags=("destructive",),
                    reason="destructive tools are blocked",
                ),
            )
        ),
        policy_decision_store=policy_store,
    )
    outcome = await AgentRunner(session).run(AgentRunRequest(task="delete target"))
    records = list((outcome.trace_manifest.get("policy_decisions") or {}).get("records") or ())
    denied = [
        record
        for record in records
        if isinstance(record, dict)
        and (record.get("decision") or {}).get("status") == "deny"
    ]
    summary = dict(outcome.trace_manifest.get("summary") or {})
    return {
        "schema_version": "agent-core-guardrail-policy-acceptance/v1",
        "status": outcome.result.status,
        "output": outcome.result.output,
        "provider_request_count": len(provider.requests),
        "policy_record_count": len(records),
        "denied_count": len(denied),
        "denied_subjects": [
            str(record.get("subject") or "") for record in denied if isinstance(record, dict)
        ],
        "executed_tool_count": int(summary.get("tool_execution_count") or 0),
        "summary": _guardrail_summary(summary),
        "trace": outcome.trace_manifest,
    }


async def _structured_output_guardrail_acceptance() -> dict[str, Any]:
    provider = MockLLMProvider(
        [
            {"action": "finish", "arguments": {"output": "plain text"}},
            {
                "action": "finish",
                "arguments": {"output": '{"summary":"safe","risk":1}'},
            },
        ]
    )
    spec = StructuredOutputSpec(
        name="risk_summary",
        description="Guardrail acceptance risk summary.",
        schema={
            "type": "object",
            "required": ["summary", "risk"],
            "properties": {
                "summary": {"type": "string"},
                "risk": {"type": "integer"},
            },
        },
        max_repairs=1,
    )
    session = AgentSession(
        profile=AgentProfile(name="guardrail-structured-output"),
        provider=provider,
        tools=MockToolRuntime(),
        structured_output_validator=JsonStructuredOutputValidator(),
    )
    outcome = await AgentRunner(session).run(
        AgentRunRequest(task="produce risk summary", structured_output=spec)
    )
    summary = dict(outcome.trace_manifest.get("summary") or {})
    structured_trace = dict(outcome.trace_manifest.get("structured_output_trace") or {})
    return {
        "schema_version": "agent-core-guardrail-structured-output-acceptance/v1",
        "status": outcome.result.status,
        "output": outcome.result.output,
        "provider_request_count": len(provider.requests),
        "record_count": int(structured_trace.get("record_count") or 0),
        "repair_count": int(structured_trace.get("repair_count") or 0),
        "ok_count": int(structured_trace.get("ok_count") or 0),
        "failed_count": int(structured_trace.get("failed_count") or 0),
        "schema_names": dict(structured_trace.get("schema_names") or {}),
        "summary": _guardrail_summary(summary),
        "trace": outcome.trace_manifest,
    }


async def _delete_target(invocation: ToolInvocation) -> ToolResult:
    return ToolResult(
        call_id=invocation.call_id,
        tool_name=invocation.tool_name,
        content="deleted",
    )


def _preflight_trace_spec() -> TraceEvalSpec:
    return TraceEvalSpec(
        name="guardrail-preflight-acceptance",
        expected_status="denied",
        require_preflight=True,
        require_preflight_blocked=True,
        required_preflight_issue_codes=("missing_tool", "task_bytes_exceeded"),
        require_failure_summary=True,
        required_failure_sources=("preflight", "run"),
        required_failure_kinds=("policy_denied",),
        max_failure_count=3,
    )


def _policy_trace_spec() -> TraceEvalSpec:
    return TraceEvalSpec(
        name="guardrail-policy-acceptance",
        expected_status="completed",
        max_provider_calls=3,
        require_preflight=True,
        require_preflight_passed=True,
        max_failure_count=0,
    )


def _structured_output_trace_spec() -> TraceEvalSpec:
    return TraceEvalSpec(
        name="guardrail-structured-output-acceptance",
        expected_status="completed",
        max_provider_calls=3,
        require_structured_output=True,
        require_structured_output_ok=True,
        required_structured_output_schema_names=("risk_summary",),
        max_structured_output_repairs=1,
        max_structured_output_failures=1,
        require_failure_summary=True,
        required_failure_sources=("structured_output",),
        required_failure_kinds=("schema_invalid",),
        max_failure_count=1,
    )


def _guardrail_acceptance_issues(
    *,
    preflight: dict[str, Any],
    policy: dict[str, Any],
    structured_output: dict[str, Any],
    trace_evals: dict[str, Any],
) -> tuple[AgentCoreGuardrailAcceptanceIssue, ...]:
    issues: list[AgentCoreGuardrailAcceptanceIssue] = []
    _check_preflight(issues, preflight)
    _check_policy(issues, policy)
    _check_structured_output(issues, structured_output)
    for name, trace_eval in trace_evals.items():
        if isinstance(trace_eval, dict) and trace_eval.get("ok") is True:
            continue
        issues.append(
            AgentCoreGuardrailAcceptanceIssue(
                source=f"{name}_trace_eval",
                code="guardrail_trace_eval_failed",
                message=f"{name} guardrail trace eval failed.",
                metadata={"trace_eval": dict(trace_eval) if isinstance(trace_eval, dict) else {}},
            )
        )
    return tuple(issues)


def _check_preflight(
    issues: list[AgentCoreGuardrailAcceptanceIssue],
    preflight: dict[str, Any],
) -> None:
    codes = set(str(code) for code in preflight.get("blocking_codes") or ())
    if preflight.get("status") != "denied" or preflight.get("preflight_status") != "blocked":
        issues.append(
            AgentCoreGuardrailAcceptanceIssue(
                source="preflight",
                code="preflight_not_blocked",
                message="Input preflight guardrail did not block the unsafe request.",
                metadata=_prompt_safe(preflight),
            )
        )
    if {"missing_tool", "task_bytes_exceeded"} - codes:
        issues.append(
            AgentCoreGuardrailAcceptanceIssue(
                source="preflight",
                code="preflight_codes_missing",
                message="Input preflight guardrail did not report required blocking codes.",
                metadata={"blocking_codes": sorted(codes)},
            )
        )


def _check_policy(
    issues: list[AgentCoreGuardrailAcceptanceIssue],
    policy: dict[str, Any],
) -> None:
    if policy.get("status") != "completed" or policy.get("output") != "safe":
        issues.append(
            AgentCoreGuardrailAcceptanceIssue(
                source="policy",
                code="policy_run_not_recovered",
                message="Policy-denied tool run did not continue to a safe final answer.",
                metadata=_prompt_safe(policy),
            )
        )
    if int(policy.get("denied_count") or 0) != 1:
        issues.append(
            AgentCoreGuardrailAcceptanceIssue(
                source="policy",
                code="policy_denial_missing",
                message="Policy guardrail did not record one denied dangerous tool decision.",
                metadata=_prompt_safe(policy),
            )
        )
    if int(policy.get("executed_tool_count") or 0) != 0:
        issues.append(
            AgentCoreGuardrailAcceptanceIssue(
                source="policy",
                code="policy_denied_tool_executed",
                message="Policy-denied dangerous tool was executed.",
                metadata=_prompt_safe(policy),
            )
        )


def _check_structured_output(
    issues: list[AgentCoreGuardrailAcceptanceIssue],
    structured_output: dict[str, Any],
) -> None:
    if structured_output.get("status") != "completed":
        issues.append(
            AgentCoreGuardrailAcceptanceIssue(
                source="structured_output",
                code="structured_output_not_completed",
                message="Structured output guardrail did not finish after repair.",
                metadata=_prompt_safe(structured_output),
            )
        )
    if int(structured_output.get("repair_count") or 0) != 1:
        issues.append(
            AgentCoreGuardrailAcceptanceIssue(
                source="structured_output",
                code="structured_output_repair_missing",
                message="Structured output guardrail did not record one repair.",
                metadata=_prompt_safe(structured_output),
            )
        )
    if int(structured_output.get("ok_count") or 0) != 1:
        issues.append(
            AgentCoreGuardrailAcceptanceIssue(
                source="structured_output",
                code="structured_output_ok_missing",
                message="Structured output guardrail did not record a valid output.",
                metadata=_prompt_safe(structured_output),
            )
        )


def _guardrail_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "iterations",
        "provider_call_count",
        "has_preflight",
        "preflight_blocked",
        "preflight_issue_count",
        "preflight_blocking_count",
        "policy_decision_record_count",
        "tool_execution_count",
        "tool_names",
        "structured_output_record_count",
        "structured_output_repair_count",
        "structured_output_failed_count",
        "failure_count",
        "failure_sources",
        "failure_kinds",
    )
    return {key: summary.get(key) for key in keys if key in summary}


def _prompt_safe(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "trace"}
