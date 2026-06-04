"""SDK-level combined interaction acceptance checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from agent_core.config import AgentProfile
from agent_core.events import ListEventSink
from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.providers import (
    LLMProviderCenter,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    LLMToolCall,
    UsageInfo,
)
from agent_core.runner import AgentRunRequest, AgentRunner, AgentSession
from agent_core.structured import JsonStructuredOutputValidator, StructuredOutputSpec
from agent_core.testing import MockToolRuntime


@dataclass(frozen=True)
class AgentCoreInteractionAcceptanceIssue:
    """One blocking mixed interaction acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-interaction-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreInteractionAcceptanceReport:
    """Prompt-safe report for mixed stream/tool/structured interactions."""

    status: str
    run: dict[str, Any] = field(default_factory=dict)
    provider_request: dict[str, Any] = field(default_factory=dict)
    provider_stream: dict[str, Any] = field(default_factory=dict)
    tool_runtime: dict[str, Any] = field(default_factory=dict)
    structured_output: dict[str, Any] = field(default_factory=dict)
    event_log: dict[str, Any] = field(default_factory=dict)
    trace_summary: dict[str, Any] = field(default_factory=dict)
    trace_eval: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreInteractionAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-interaction-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "run": dict(self.run),
            "provider_request": dict(self.provider_request),
            "provider_stream": dict(self.provider_stream),
            "tool_runtime": dict(self.tool_runtime),
            "structured_output": dict(self.structured_output),
            "event_log": dict(self.event_log),
            "trace_summary": dict(self.trace_summary),
            "trace_eval": dict(self.trace_eval),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreInteractionAcceptanceHarness:
    """Run deterministic mixed SDK interaction checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreInteractionAcceptanceReport:
        provider = _InteractionAcceptanceProvider()
        center = LLMProviderCenter(default_provider="interaction")
        center.register("interaction", provider, default_model="interaction-mini")
        tools = MockToolRuntime({"lookup": "asset=gateway risk=2"})
        events = ListEventSink()
        structured_spec = StructuredOutputSpec(
            name="risk_summary",
            description="Interaction acceptance risk summary.",
            schema={
                "type": "object",
                "required": ["summary", "risk"],
                "properties": {
                    "summary": {"type": "string"},
                    "risk": {"type": "integer"},
                },
            },
        )
        session = AgentSession(
            profile=AgentProfile(name="interaction-acceptance", model="interaction-mini"),
            provider=center,
            tools=tools,
            structured_output_validator=JsonStructuredOutputValidator(),
            event_sink=events,
        )
        outcome = await AgentRunner(session).run(
            AgentRunRequest(
                task="inspect gateway and return structured risk",
                stream=True,
                native_tool_calls=True,
                structured_output=structured_spec,
                metadata={"acceptance": "interaction"},
            )
        )
        trace_eval = DefaultTraceEvaluator().evaluate(
            outcome.trace_manifest,
            _interaction_trace_spec(),
        ).manifest()
        run = {
            "schema_version": "agent-core-interaction-run-acceptance/v1",
            "run_id": outcome.result.run_id,
            "status": outcome.result.status,
            "output": outcome.result.output,
            "provider_request_count": len(provider.requests),
            "stream_request_count": provider.stream_request_count,
            "native_tool_calls": bool(
                outcome.trace_manifest.get("metadata", {}).get("native_tool_calls")
            ),
            "stream_requested": bool(outcome.trace_manifest.get("metadata", {}).get("stream")),
        }
        provider_request = _provider_request_manifest(provider.requests)
        provider_stream = _provider_stream_manifest(outcome.trace_manifest)
        tool_runtime = {
            "schema_version": "agent-core-interaction-tool-runtime-acceptance/v1",
            "invocation_count": len(tools.invocations),
            "tool_names": [item.tool_name for item in tools.invocations],
            "call_ids": [item.call_id for item in tools.invocations],
            "argument_keys": [
                sorted(str(key) for key in item.arguments) for item in tools.invocations
            ],
        }
        structured_output = dict(outcome.trace_manifest.get("structured_output_trace") or {})
        event_log = events.manifest()
        summary = _interaction_summary(dict(trace_eval.get("summary") or {}))
        issues = _interaction_acceptance_issues(
            run=run,
            provider_request=provider_request,
            provider_stream=provider_stream,
            tool_runtime=tool_runtime,
            structured_output=structured_output,
            event_log=event_log,
            trace_eval=trace_eval,
            trace_summary=summary,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreInteractionAcceptanceReport(
            status=status,
            run=run,
            provider_request=provider_request,
            provider_stream=provider_stream,
            tool_runtime=tool_runtime,
            structured_output=structured_output,
            event_log=_event_log_summary(event_log),
            trace_summary=summary,
            trace_eval=trace_eval,
            issues=issues,
            metadata={"scenario": "agent_core_interaction_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_interaction_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreInteractionAcceptanceReport:
    """Run the default mixed stream/tool/structured interaction checks."""

    return await AgentCoreInteractionAcceptanceHarness(metadata=dict(metadata or {})).run()


class _InteractionAcceptanceProvider:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self.stream_request_count = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("interaction acceptance must use streaming provider calls")

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self.requests.append(request)
        self.stream_request_count += 1
        if self.stream_request_count == 1:
            yield LLMStreamEvent(type="message_start")
            yield LLMStreamEvent(type="delta", delta="looking up gateway")
            yield LLMStreamEvent(
                type="tool_call",
                tool_call=LLMToolCall(
                    tool_name="lookup",
                    arguments={"query": "gateway"},
                    call_id="interaction-call-1",
                ),
            )
            yield LLMStreamEvent(type="usage", usage=UsageInfo(total_tokens=7))
            yield LLMStreamEvent(type="message_end")
            return

        yield LLMStreamEvent(type="message_start")
        yield LLMStreamEvent(type="delta", delta="final structured answer")
        yield LLMStreamEvent(
            type="action",
            action={
                "action": "finish",
                "arguments": {"output": '{"summary":"gateway reachable","risk":2}'},
            },
        )
        yield LLMStreamEvent(type="usage", usage=UsageInfo(total_tokens=9))
        yield LLMStreamEvent(type="message_end")


def _interaction_trace_spec() -> TraceEvalSpec:
    return TraceEvalSpec(
        name="interaction-acceptance",
        expected_status="completed",
        max_provider_calls=3,
        require_provider_streaming=True,
        required_provider_stream_event_types=(
            "message_start",
            "delta",
            "tool_call",
            "usage",
            "action",
            "message_end",
        ),
        max_provider_stream_errors=0,
        require_provider_tool_calls=True,
        required_provider_tool_call_names=("lookup",),
        max_provider_tool_calls=1,
        require_provider_tool_results=True,
        require_provider_tool_result_execution=True,
        required_provider_tool_result_names=("lookup",),
        required_provider_tool_result_statuses=("completed",),
        max_provider_tool_result_failures=0,
        require_structured_output=True,
        require_structured_output_ok=True,
        required_structured_output_schema_names=("risk_summary",),
        max_structured_output_repairs=0,
        max_structured_output_failures=0,
        require_event_log=True,
        required_event_log_types=(
            "run_started",
            "model_stream",
            "model_response",
            "run_finished",
        ),
        require_terminal_event=True,
        require_event_sequence_monotonic=True,
        max_duplicate_event_sequences=0,
        max_failure_count=0,
    )


def _provider_request_manifest(requests: list[LLMRequest]) -> dict[str, Any]:
    first = requests[0] if requests else None
    second = requests[1] if len(requests) > 1 else None
    return {
        "schema_version": "agent-core-interaction-provider-request-acceptance/v1",
        "request_count": len(requests),
        "first_tool_names": [tool.name for tool in (first.tools if first else ())],
        "first_tool_choice": first.tool_choice.manifest() if first and first.tool_choice else {},
        "first_response_format": (
            first.response_format.manifest() if first and first.response_format else {}
        ),
        "second_last_message": second.messages[-1].manifest()
        if second and second.messages
        else {},
        "second_response_format": (
            second.response_format.manifest() if second and second.response_format else {}
        ),
    }


def _provider_stream_manifest(trace: dict[str, Any]) -> dict[str, Any]:
    calls = list((trace.get("provider") or {}).get("calls") or ())
    summaries = [
        dict((call.get("metadata") or {}).get("stream_summary") or {})
        for call in calls
        if isinstance(call, dict)
    ]
    event_types: list[str] = []
    tool_names: list[str] = []
    for summary in summaries:
        event_types.extend(str(item) for item in summary.get("event_types") or ())
        for tool_call in summary.get("tool_calls") or ():
            if isinstance(tool_call, dict):
                tool_names.append(str(tool_call.get("tool_name") or ""))
    return {
        "schema_version": "agent-core-interaction-provider-stream-acceptance/v1",
        "call_count": len(calls),
        "stream_summary_count": len(summaries),
        "event_types": event_types,
        "tool_call_names": [name for name in tool_names if name],
        "error_count": sum(1 for summary in summaries if summary.get("error")),
    }


def _event_log_summary(event_log: dict[str, Any]) -> dict[str, Any]:
    events = [event for event in event_log.get("events") or () if isinstance(event, dict)]
    return {
        "schema_version": "agent-core-interaction-event-log-acceptance/v1",
        "event_count": len(events),
        "event_types": [str(event.get("type") or "") for event in events],
        "model_stream_event_count": sum(1 for event in events if event.get("type") == "model_stream"),
        "terminal": bool(events and events[-1].get("type") == "run_finished"),
    }


def _interaction_acceptance_issues(
    *,
    run: dict[str, Any],
    provider_request: dict[str, Any],
    provider_stream: dict[str, Any],
    tool_runtime: dict[str, Any],
    structured_output: dict[str, Any],
    event_log: dict[str, Any],
    trace_eval: dict[str, Any],
    trace_summary: dict[str, Any],
) -> tuple[AgentCoreInteractionAcceptanceIssue, ...]:
    issues: list[AgentCoreInteractionAcceptanceIssue] = []
    if run.get("status") != "completed":
        issues.append(
            AgentCoreInteractionAcceptanceIssue(
                source="run",
                code="interaction_run_not_completed",
                message="Mixed interaction acceptance run did not complete.",
                metadata=dict(run),
            )
        )
    if run.get("stream_request_count") != 2 or run.get("stream_requested") is not True:
        issues.append(
            AgentCoreInteractionAcceptanceIssue(
                source="run",
                code="interaction_stream_not_used",
                message="Mixed interaction run did not use streaming provider calls.",
                metadata=dict(run),
            )
        )
    if run.get("native_tool_calls") is not True:
        issues.append(
            AgentCoreInteractionAcceptanceIssue(
                source="run",
                code="interaction_native_tool_flag_missing",
                message="Mixed interaction run did not preserve native_tool_calls=true.",
                metadata=dict(run),
            )
        )
    if provider_request.get("first_tool_names") != ["lookup"]:
        issues.append(
            AgentCoreInteractionAcceptanceIssue(
                source="provider_request",
                code="interaction_tool_contract_missing",
                message="First provider request did not include the lookup tool contract.",
                metadata=dict(provider_request),
            )
        )
    if (provider_request.get("first_response_format") or {}).get("name") != "risk_summary":
        issues.append(
            AgentCoreInteractionAcceptanceIssue(
                source="provider_request",
                code="interaction_response_format_missing",
                message="Provider request did not carry the structured output response format.",
                metadata=dict(provider_request),
            )
        )
    if (provider_request.get("second_last_message") or {}).get("role") != "tool":
        issues.append(
            AgentCoreInteractionAcceptanceIssue(
                source="provider_request",
                code="interaction_tool_result_message_missing",
                message="Second provider request did not include the provider tool result message.",
                metadata=dict(provider_request),
            )
        )
    required_events = {"delta", "tool_call", "usage", "action", "message_end"}
    actual_events = set(str(item) for item in provider_stream.get("event_types") or ())
    if required_events - actual_events:
        issues.append(
            AgentCoreInteractionAcceptanceIssue(
                source="provider_stream",
                code="interaction_stream_events_missing",
                message="Provider stream did not expose all required mixed event types.",
                metadata={"missing": sorted(required_events - actual_events), **dict(provider_stream)},
            )
        )
    if provider_stream.get("tool_call_names") != ["lookup"]:
        issues.append(
            AgentCoreInteractionAcceptanceIssue(
                source="provider_stream",
                code="interaction_stream_tool_call_missing",
                message="Provider stream did not preserve the streamed lookup tool call.",
                metadata=dict(provider_stream),
            )
        )
    if tool_runtime.get("invocation_count") != 1 or tool_runtime.get("call_ids") != [
        "interaction-call-1"
    ]:
        issues.append(
            AgentCoreInteractionAcceptanceIssue(
                source="tool_runtime",
                code="interaction_tool_runtime_invocation_missing",
                message="Streamed provider tool call was not executed through tool runtime.",
                metadata=dict(tool_runtime),
            )
        )
    if int(structured_output.get("ok_count") or 0) != 1 or dict(
        structured_output.get("schema_names") or {}
    ).get("risk_summary") != 1:
        issues.append(
            AgentCoreInteractionAcceptanceIssue(
                source="structured_output",
                code="interaction_structured_output_missing",
                message="Mixed interaction run did not produce valid structured output.",
                metadata=dict(structured_output),
            )
        )
    if int(event_log.get("event_count") or 0) < 1:
        issues.append(
            AgentCoreInteractionAcceptanceIssue(
                source="event_log",
                code="interaction_event_log_missing",
                message="Mixed interaction run did not emit SDK events.",
                metadata=dict(event_log),
            )
        )
    if int(trace_summary.get("provider_stream_call_count") or 0) != 2:
        issues.append(
            AgentCoreInteractionAcceptanceIssue(
                source="trace_summary",
                code="interaction_trace_stream_summary_missing",
                message="Trace summary did not record two streamed provider calls.",
                metadata=dict(trace_summary),
            )
        )
    if trace_eval.get("ok") is not True:
        issues.append(
            AgentCoreInteractionAcceptanceIssue(
                source="trace_eval",
                code="interaction_trace_eval_failed",
                message="Trace evaluator did not accept mixed stream/tool/structured contract.",
                metadata={"trace_eval": dict(trace_eval)},
            )
        )
    return tuple(issues)


def _interaction_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "provider_call_count",
        "provider_stream_call_count",
        "provider_stream_summary_count",
        "provider_stream_event_types",
        "provider_stream_error_count",
        "provider_tool_call_count",
        "provider_tool_call_names",
        "provider_tool_result_count",
        "provider_tool_result_names",
        "provider_tool_result_statuses",
        "provider_tool_result_execution_count",
        "structured_output_record_count",
        "structured_output_ok",
        "structured_output_schema_names",
        "structured_output_repair_count",
        "structured_output_failure_count",
        "event_log_count",
        "event_log_types",
        "terminal_event_seen",
        "failure_count",
        "failure_sources",
        "failure_kinds",
    )
    return {key: summary.get(key) for key in keys if key in summary}
