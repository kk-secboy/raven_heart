"""SDK-level provider-native tool-call acceptance checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.config import AgentProfile
from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.providers import LLMProviderCenter, LLMResponse, LLMToolCall
from agent_core.runner import AgentRunRequest, AgentRunner, AgentSession
from agent_core.testing import MockLLMProvider, MockToolRuntime


@dataclass(frozen=True)
class AgentCoreNativeToolAcceptanceIssue:
    """One blocking provider-native tool-call acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-native-tool-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreNativeToolAcceptanceReport:
    """Prompt-safe report for provider-native tool-call portability."""

    status: str
    run: dict[str, Any] = field(default_factory=dict)
    provider_request: dict[str, Any] = field(default_factory=dict)
    tool_runtime: dict[str, Any] = field(default_factory=dict)
    journal_tool_call: dict[str, Any] = field(default_factory=dict)
    trace_summary: dict[str, Any] = field(default_factory=dict)
    trace_eval: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreNativeToolAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-native-tool-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "run": dict(self.run),
            "provider_request": dict(self.provider_request),
            "tool_runtime": dict(self.tool_runtime),
            "journal_tool_call": dict(self.journal_tool_call),
            "trace_summary": dict(self.trace_summary),
            "trace_eval": dict(self.trace_eval),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreNativeToolAcceptanceHarness:
    """Run deterministic provider-native tool-call acceptance checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreNativeToolAcceptanceReport:
        provider = MockLLMProvider(
            [
                LLMResponse(
                    content="checking",
                    tool_calls=(
                        LLMToolCall(
                            tool_name="lookup",
                            arguments={"query": "target"},
                            call_id="call-1",
                        ),
                    ),
                ),
                {"action": "finish", "arguments": {"output": "done"}},
            ]
        )
        center = LLMProviderCenter(default_provider="mock")
        center.register("mock", provider, default_model="mock-mini")
        tools = MockToolRuntime({"lookup": "found"})
        session = AgentSession(
            profile=AgentProfile(name="native-tool-acceptance", model="mock-mini"),
            provider=center,
            tools=tools,
        )
        outcome = await AgentRunner(session).run(
            AgentRunRequest(
                task="inspect target",
                native_tool_calls=True,
                metadata={"acceptance": "native_tool"},
            )
        )
        trace_eval = DefaultTraceEvaluator().evaluate(
            outcome.trace_manifest,
            _native_tool_trace_spec(),
        ).manifest()
        summary = _native_tool_summary(dict(trace_eval.get("summary") or {}))
        provider_request = _provider_request_manifest(provider.requests)
        journal_tool_call = _journal_tool_call(outcome.trace_manifest)
        tool_runtime = {
            "schema_version": "agent-core-native-tool-runtime-acceptance/v1",
            "invocation_count": len(tools.invocations),
            "tool_names": [item.tool_name for item in tools.invocations],
            "call_ids": [item.call_id for item in tools.invocations],
            "argument_keys": [
                sorted(str(key) for key in item.arguments) for item in tools.invocations
            ],
        }
        run = {
            "run_id": outcome.result.run_id,
            "status": outcome.result.status,
            "output": outcome.result.output,
            "provider_request_count": len(provider.requests),
            "native_tool_calls": bool(outcome.trace_manifest.get("metadata", {}).get("native_tool_calls")),
        }
        issues = _native_tool_acceptance_issues(
            run=run,
            provider_request=provider_request,
            tool_runtime=tool_runtime,
            journal_tool_call=journal_tool_call,
            trace_summary=summary,
            trace_eval=trace_eval,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreNativeToolAcceptanceReport(
            status=status,
            run=run,
            provider_request=provider_request,
            tool_runtime=tool_runtime,
            journal_tool_call=journal_tool_call,
            trace_summary=summary,
            trace_eval=trace_eval,
            issues=issues,
            metadata={"scenario": "agent_core_native_tool_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_native_tool_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreNativeToolAcceptanceReport:
    """Run the default provider-native tool-call acceptance checks."""

    return await AgentCoreNativeToolAcceptanceHarness(metadata=dict(metadata or {})).run()


def _native_tool_trace_spec() -> TraceEvalSpec:
    return TraceEvalSpec(
        name="native-tool-acceptance",
        expected_status="completed",
        max_provider_calls=2,
        require_provider_tool_calls=True,
        required_provider_tool_call_names=("lookup",),
        max_provider_tool_calls=1,
        require_provider_tool_results=True,
        require_provider_tool_result_execution=True,
        required_provider_tool_result_names=("lookup",),
        required_provider_tool_result_statuses=("completed",),
        max_provider_tool_result_failures=0,
        require_tool_schema_validation=True,
        required_tool_schema_validation_names=("lookup",),
        required_tool_schema_valid_names=("lookup",),
        max_tool_schema_invalid=0,
        max_failure_count=0,
    )


def _provider_request_manifest(requests: list[Any]) -> dict[str, Any]:
    first = requests[0] if requests else None
    second = requests[1] if len(requests) > 1 else None
    return {
        "schema_version": "agent-core-native-tool-provider-request-acceptance/v1",
        "request_count": len(requests),
        "first_tool_names": [tool.name for tool in (first.tools if first else ())],
        "first_tool_choice": first.tool_choice.manifest() if first and first.tool_choice else {},
        "second_last_message": second.messages[-1].manifest() if second and second.messages else {},
    }


def _journal_tool_call(trace: dict[str, Any]) -> dict[str, Any]:
    for event in trace.get("journal_replay", {}).get("events", ()):
        if isinstance(event, dict) and event.get("event_type") == "tool_call":
            return dict(event.get("payload") or {})
    return {}


def _native_tool_acceptance_issues(
    *,
    run: dict[str, Any],
    provider_request: dict[str, Any],
    tool_runtime: dict[str, Any],
    journal_tool_call: dict[str, Any],
    trace_summary: dict[str, Any],
    trace_eval: dict[str, Any],
) -> tuple[AgentCoreNativeToolAcceptanceIssue, ...]:
    issues: list[AgentCoreNativeToolAcceptanceIssue] = []
    if run.get("status") != "completed" or run.get("output") != "done":
        issues.append(
            AgentCoreNativeToolAcceptanceIssue(
                source="run",
                code="native_tool_run_not_completed",
                message="Native tool acceptance run did not complete.",
                metadata=dict(run),
            )
        )
    if run.get("native_tool_calls") is not True:
        issues.append(
            AgentCoreNativeToolAcceptanceIssue(
                source="run",
                code="native_tool_flag_missing",
                message="Run trace did not preserve native_tool_calls=true.",
                metadata=dict(run),
            )
        )
    if provider_request.get("first_tool_names") != ["lookup"]:
        issues.append(
            AgentCoreNativeToolAcceptanceIssue(
                source="provider_request",
                code="native_tool_contract_missing",
                message="Provider request did not include the native lookup tool contract.",
                metadata=dict(provider_request),
            )
        )
    if (provider_request.get("second_last_message") or {}).get("role") != "tool":
        issues.append(
            AgentCoreNativeToolAcceptanceIssue(
                source="provider_request",
                code="native_tool_result_message_missing",
                message="Provider did not receive a tool role result message.",
                metadata=dict(provider_request),
            )
        )
    if tool_runtime.get("invocation_count") != 1 or tool_runtime.get("call_ids") != ["call-1"]:
        issues.append(
            AgentCoreNativeToolAcceptanceIssue(
                source="tool_runtime",
                code="native_tool_runtime_invocation_missing",
                message="Native provider tool call was not executed through tool runtime.",
                metadata=dict(tool_runtime),
            )
        )
    provider_tool_call = (
        (journal_tool_call.get("metadata") or {}).get("provider_tool_call")
        if isinstance(journal_tool_call.get("metadata"), dict)
        else {}
    )
    if provider_tool_call.get("call_id") != "call-1":
        issues.append(
            AgentCoreNativeToolAcceptanceIssue(
                source="journal",
                code="native_tool_journal_metadata_missing",
                message="Journal tool call did not preserve provider tool-call metadata.",
                metadata=dict(journal_tool_call),
            )
        )
    if int(trace_summary.get("provider_tool_result_execution_count") or 0) != 1:
        issues.append(
            AgentCoreNativeToolAcceptanceIssue(
                source="trace_summary",
                code="native_tool_trace_summary_missing",
                message="Trace summary did not record native provider tool result execution.",
                metadata=dict(trace_summary),
            )
        )
    if trace_eval.get("ok") is not True:
        issues.append(
            AgentCoreNativeToolAcceptanceIssue(
                source="trace_eval",
                code="native_tool_trace_eval_failed",
                message="Trace evaluator did not accept native provider tool-call contract.",
                metadata={"trace_eval": dict(trace_eval)},
            )
        )
    return tuple(issues)


def _native_tool_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "provider_tool_call_count",
        "provider_tool_call_names",
        "provider_tool_result_count",
        "provider_tool_result_names",
        "provider_tool_result_statuses",
        "provider_tool_result_failure_count",
        "provider_tool_result_execution_count",
        "tool_names",
        "tool_execution_count",
        "tool_schema_validation_count",
        "tool_schema_invalid_count",
        "tool_schema_validated_names",
        "tool_schema_valid_names",
        "failure_count",
        "failure_sources",
        "failure_kinds",
    )
    return {key: summary.get(key) for key in keys if key in summary}
