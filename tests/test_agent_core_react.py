from __future__ import annotations

import asyncio

import pytest

from agent_core.actions import (
    ActionError,
    ActionRegistry,
    ActionSpec,
    ActionVerification,
    ParsedAction,
)
from agent_core.events import AgentEvent, ListEventSink
from agent_core.artifacts import InMemoryArtifactStore
from agent_core.harness import CancelToken
from agent_core.loop_guard import LoopGuardConfig
from agent_core.memory import MemoryHit
from agent_core.prompt import PromptIR
from agent_core.providers import LLMMessage, LLMRequest, LLMResponse, LLMToolCall
from agent_core.react import (
    ReActConfig,
    ReActExecutor,
    _message_bytes,
    _messages_bytes,
    _native_tool_contracts_bytes,
)
from agent_core.config import RuntimeBudget
from agent_core.skills import SkillRegistry, SkillsContext
from agent_core.timeline import TimelineStore
from agent_core.tools import (
    InMemoryToolReplay,
    InMemoryToolReplayStore,
    MarkdownToolReplayStore,
    PersistentToolReplay,
    SQLiteToolReplayStore,
    ToolInvocation,
    ToolRegistry,
    ToolResult,
    ToolRetryPolicy,
    ToolSpec,
)
from agent_core.testing import InMemoryHarness, MockLLMProvider, MockMemory, MockToolRuntime


class _BlockingProvider:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self.release = asyncio.Event()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        await self.release.wait()
        return LLMResponse(action={"action": "finish", "arguments": {"output": "late"}})


class _BlockingToolRuntime(MockToolRuntime):
    def __init__(self) -> None:
        super().__init__({"lookup": ""})
        self.release = asyncio.Event()

    async def invoke(self, invocation: ToolInvocation) -> ToolResult:
        self.invocations.append(invocation)
        await self.release.wait()
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            status="completed",
            content="late",
        )


@pytest.mark.asyncio
async def test_list_event_sink_assigns_sequence_for_external_events() -> None:
    events = ListEventSink()

    await events.emit(AgentEvent(type="run_started"))
    await events.emit(AgentEvent(type="run_finished"))

    assert [event.sequence for event in events.events] == [1, 2]
    assert events.manifest()["events"][1]["type"] == "run_finished"


@pytest.mark.asyncio
async def test_react_executor_runs_tool_then_finish_with_mock_ports() -> None:
    provider = MockLLMProvider(
        [
            {
                "action": "call_tool",
                "arguments": {"tool_name": "lookup", "arguments": {"query": "target"}},
            },
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    tools = MockToolRuntime({"lookup": "tool says target is ready"})
    harness = InMemoryHarness()
    events = ListEventSink()

    executor = ReActExecutor(
        provider=provider,
        tool_runtime=tools,
        action_registry=ActionRegistry(),
        harness=harness,
        event_sink=events,
        config=ReActConfig(max_iterations=4),
    )

    result = await executor.run("inspect target", PromptIR.from_parts(high_static="rules"))

    assert result.status == "completed"
    assert result.output == "done"
    assert result.iterations == 2
    assert len(provider.requests) == 2
    assert len(tools.invocations) == 1
    assert tools.invocations[0].tool_name == "lookup"
    assert harness.prompts
    assert harness.checkpoints[-1].state["status"] == "finished"
    assert [event.type for event in events.events if event.type == "tool_finished"]
    assert [event.sequence for event in events.events] == list(range(1, len(events.events) + 1))
    manifest = events.manifest()
    assert manifest["schema_version"] == "agent-core-event-log/v1"
    assert manifest["event_count"] == len(events.events)
    assert manifest["events"][0]["type"] == "run_started"


@pytest.mark.asyncio
async def test_react_executor_feedbacks_action_schema_errors() -> None:
    provider = MockLLMProvider(
        [
            {"action": "call_tool", "arguments": {}},
            {"action": "finish", "arguments": {"output": "recovered"}},
        ]
    )
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime({"lookup": "ok"}),
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        config=ReActConfig(max_iterations=3),
    )

    result = await executor.run("task", PromptIR.from_parts(high_static="rules"))

    assert result.status == "completed"
    assert result.output == "recovered"
    assert len(provider.requests) == 2
    assert provider.requests[1].messages[-1].content.startswith('{"feedback": "action_error"')


def test_react_executor_provider_messages_apply_total_request_budget() -> None:
    executor = ReActExecutor(
        provider=MockLLMProvider([]),
        tool_runtime=MockToolRuntime({}),
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        config=ReActConfig(
            budget=RuntimeBudget(max_prompt_bytes=900),
            loop_delta_max_bytes=700,
        ),
    )
    prompt = PromptIR.from_parts(
        dynamic="fresh prompt fact\n" + ("x" * 850),
    )
    compact_delta = [
        LLMMessage(role="assistant", content="previous answer " + ("a" * 280)),
        LLMMessage(role="user", name="lookup", content="tool output " + ("b" * 280)),
    ]

    messages = executor._provider_messages(prompt, compact_delta)
    total_bytes = sum(_message_bytes(message) for message in messages)

    assert total_bytes <= 900
    assert messages[0].metadata["agent_core_prompt"] is True
    assert len(messages[0].content.encode("utf-8")) < len(prompt.render().encode("utf-8"))


def test_react_executor_provider_messages_split_prompt_like_yaklang_cache_sections() -> None:
    executor = ReActExecutor(
        provider=MockLLMProvider([]),
        tool_runtime=MockToolRuntime({}),
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        config=ReActConfig(budget=RuntimeBudget(max_prompt_bytes=4000)),
    )
    prompt = PromptIR.from_parts(
        high_static="system rules",
        frozen="tool inventory",
        semi_dynamic_1="skills context",
        semi_dynamic_2="schema context",
        timeline_open="latest tool fact",
        dynamic="current task",
    )

    messages = executor._provider_messages(prompt, [])

    assert [message.role for message in messages] == ["system", "user", "user", "user", "user"]
    assert [
        message.metadata["agent_core_prompt_segment"] for message in messages
    ] == [
        "high_static",
        "frozen",
        "semi_dynamic_1",
        "semi_dynamic_2",
        "timeline_open_dynamic",
    ]
    assert messages[0].metadata["cache_hint"]["cacheable"] is True
    assert messages[2].metadata["cache_hint"]["cacheable"] is False
    assert messages[3].metadata["cache_hint"]["cacheable"] is True
    assert messages[-1].metadata["cache_hint"]["cacheable"] is False
    assert "latest tool fact" in messages[-1].content
    assert "current task" in messages[-1].content
    assert "skills context" in messages[2].content
    assert "schema context" in messages[3].content

    next_prompt = PromptIR.from_parts(
        high_static="system rules",
        frozen="tool inventory",
        semi_dynamic_1="different skills context",
        semi_dynamic_2="schema context",
        timeline_open="different latest tool fact",
        dynamic="different current task",
    )
    next_messages = executor._provider_messages(next_prompt, [])

    assert [message.content for message in next_messages[:2]] == [
        message.content for message in messages[:2]
    ]
    assert next_messages[2].content != messages[2].content
    assert next_messages[3].content == messages[3].content
    stable_prefix = "\n\n".join(message.content for message in next_messages[:4])
    assert "different latest tool fact" not in stable_prefix
    assert "different current task" not in stable_prefix
    assert "different skills context" in stable_prefix


@pytest.mark.asyncio
async def test_react_executor_provider_request_can_enable_provider_cache_policy() -> None:
    provider = MockLLMProvider(
        [{"action": "finish", "arguments": {"output": "done"}}]
    )
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime({}),
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        config=ReActConfig(
            budget=RuntimeBudget(max_prompt_bytes=4000),
            provider_cache_mode="ephemeral",
            provider_cache_min_segment_bytes=4,
        ),
    )

    result = await executor.run(
        "task",
        PromptIR.from_parts(high_static="stable rules", dynamic="task"),
    )

    assert result.status == "completed"
    assert provider.requests[0].metadata["provider_cache_policy"] == {
        "mode": "ephemeral",
        "min_segment_bytes": 4,
        "cache_control": {"type": "ephemeral"},
    }


def test_react_executor_provider_messages_budget_drops_unbounded_delta_first() -> None:
    executor = ReActExecutor(
        provider=MockLLMProvider([]),
        tool_runtime=MockToolRuntime({}),
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        config=ReActConfig(
            budget=RuntimeBudget(max_prompt_bytes=700),
            loop_delta_max_bytes=1200,
        ),
    )
    prompt = PromptIR.from_parts(dynamic="task\n" + ("x" * 500))
    compact_delta = [
        LLMMessage(role="assistant", content="older " + ("a" * 900)),
        LLMMessage(role="user", name="lookup", content="newer " + ("b" * 900)),
    ]

    messages = executor._provider_messages(prompt, compact_delta)

    assert sum(_message_bytes(message) for message in messages) <= 700
    assert [message.metadata.get("agent_core_prompt") for message in messages] == [True]
    assert messages[0].metadata["agent_core_prompt_segment"] == "timeline_open_dynamic"


def test_react_executor_provider_budget_can_trim_context_injection_view() -> None:
    executor = ReActExecutor(
        provider=MockLLMProvider([]),
        tool_runtime=MockToolRuntime({}),
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        config=ReActConfig(budget=RuntimeBudget(max_prompt_bytes=900)),
    )
    prompt = PromptIR.from_parts(
        timeline_open="[context_injection:memory source=memory]\n" + ("memory fact " * 300),
        dynamic="current task",
    )

    default_trimmed = prompt.trim_to_budget(900)
    provider_messages = executor._provider_messages(prompt, [])

    assert default_trimmed.manifest()["metadata"]["trim"]["converged"] is False
    assert sum(_message_bytes(message) for message in provider_messages) <= 900


def test_action_registry_validates_registered_json_schema_subset() -> None:
    registry = ActionRegistry()
    registry.register(
        ActionSpec(
            name="ask",
            parameters_schema={
                "type": "object",
                "required": ["question"],
                "properties": {"question": {"type": "string"}},
            },
        )
    )

    parsed = registry.parse({"action": "ask", "arguments": {"question": "continue?"}})

    assert parsed.name == "ask"
    with pytest.raises(ActionError):
        registry.parse({"action": "ask", "arguments": {"question": 123}})


def test_action_registry_accepts_top_level_action_arguments() -> None:
    registry = ActionRegistry()
    registry.register(ActionSpec(name="call_tool"))
    registry.register(ActionSpec(name="finish", terminal=True))

    tool_action = registry.parse('{"action":"call_tool","tool_name":"exec_echo"}')
    finish_action = registry.parse(
        '```json\n{"action":"finish","finding":"stable result"}\n```'
    )

    assert tool_action.arguments == {"tool_name": "exec_echo"}
    assert finish_action.arguments == {"finding": "stable result"}


@pytest.mark.asyncio
async def test_react_executor_stops_at_max_iterations() -> None:
    provider = MockLLMProvider(
        [
            LLMResponse(
                action={
                    "action": "call_tool",
                    "arguments": {"tool_name": "lookup", "arguments": {}},
                }
            ),
            LLMResponse(
                action={
                    "action": "call_tool",
                    "arguments": {"tool_name": "lookup", "arguments": {}},
                }
            ),
        ]
    )
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime({"lookup": "ok"}),
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        config=ReActConfig(max_iterations=2),
    )

    result = await executor.run("loop", PromptIR.from_parts(dynamic="task"))

    assert result.status == "max_iterations"
    assert result.iterations == 2


@pytest.mark.asyncio
async def test_react_executor_stalls_repeated_tool_action_result_pattern() -> None:
    provider = MockLLMProvider(
        [
            {"action": "call_tool", "arguments": {"tool_name": "lookup", "arguments": {"q": "same"}}},
            {"action": "call_tool", "arguments": {"tool_name": "lookup", "arguments": {"q": "same"}}},
            {"action": "call_tool", "arguments": {"tool_name": "lookup", "arguments": {"q": "same"}}},
            {"action": "finish", "arguments": {"output": "should not be reached"}},
        ]
    )
    tools = MockToolRuntime({"lookup": "same result"})
    harness = InMemoryHarness()
    events = ListEventSink()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=tools,
        action_registry=ActionRegistry(),
        harness=harness,
        event_sink=events,
        config=ReActConfig(max_iterations=5),
    )

    result = await executor.run("loop", PromptIR.from_parts(dynamic="task"))

    assert result.status == "stalled"
    assert result.iterations == 3
    assert len(tools.invocations) == 3
    assert harness.finished[-1]["status"] == "stalled"
    assert any(checkpoint.state["status"] == "loop_warning" for checkpoint in harness.checkpoints)
    assert harness.checkpoints[-1].state["status"] == "loop_stalled"
    assert provider.requests[2].messages[-1].content.startswith('{"feedback": "loop_warning"')
    assert any(event.type == "loop_warning" for event in events.events)


@pytest.mark.asyncio
async def test_react_executor_loop_guard_covers_direct_tool_actions() -> None:
    tools = ToolRegistry()
    calls: list[str] = []

    @tools.register_function(aliases=("probe",))
    def http_probe(target: str) -> str:
        calls.append(target)
        return "unchanged"

    provider = MockLLMProvider(
        [
            {"action": "probe", "arguments": {"target": "example.test"}},
            {"action": "probe", "arguments": {"target": "example.test"}},
            {"action": "probe", "arguments": {"target": "example.test"}},
        ]
    )
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=tools,
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        config=ReActConfig(max_iterations=4),
    )

    result = await executor.run("direct loop", PromptIR.from_parts(dynamic="task"))

    assert result.status == "stalled"
    assert calls == ["example.test", "example.test", "example.test"]
    assert result.metadata["loop_guard"]["action"] == "probe"
    assert result.metadata["loop_guard"]["tool_name"] == "http_probe"


@pytest.mark.asyncio
async def test_react_executor_loop_guard_can_be_disabled() -> None:
    provider = MockLLMProvider(
        [
            {"action": "call_tool", "arguments": {"tool_name": "lookup", "arguments": {"q": "same"}}},
            {"action": "call_tool", "arguments": {"tool_name": "lookup", "arguments": {"q": "same"}}},
            {"action": "call_tool", "arguments": {"tool_name": "lookup", "arguments": {"q": "same"}}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime({"lookup": "same result"}),
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        config=ReActConfig(
            max_iterations=5,
            loop_guard=LoopGuardConfig(enabled=False),
        ),
    )

    result = await executor.run("allowed loop", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert result.output == "done"


@pytest.mark.asyncio
async def test_react_executor_honors_cancel_token_before_first_turn() -> None:
    cancel = CancelToken(cancelled=True, reason="operator stopped")
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "nope"}}])
    harness = InMemoryHarness()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime(),
        action_registry=ActionRegistry(),
        harness=harness,
        cancel_token=cancel,
    )

    result = await executor.run("cancelled task", PromptIR.from_parts(dynamic="task"))

    assert result.status == "cancelled"
    assert result.output == "operator stopped"
    assert result.iterations == 0
    assert provider.requests == []
    assert harness.finished[-1]["status"] == "cancelled"


def test_cancel_token_records_interrupt_manifest() -> None:
    cancel = CancelToken()

    cancel.timeout(
        "provider call timed out",
        timeout_seconds=0.25,
        metadata={"phase": "provider"},
    )

    manifest = cancel.manifest()
    assert manifest["cancelled"] is True
    assert manifest["reason"] == "provider call timed out"
    assert manifest["interrupt"]["schema_version"] == "agent-core-run-interrupt/v1"
    assert manifest["interrupt"]["kind"] == "timeout"
    assert manifest["interrupt"]["timeout_seconds"] == 0.25
    assert manifest["interrupt"]["metadata"]["phase"] == "provider"


@pytest.mark.asyncio
async def test_react_executor_times_out_provider_call_with_interrupt_manifest() -> None:
    provider = _BlockingProvider()
    harness = InMemoryHarness()
    events = ListEventSink()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime(),
        action_registry=ActionRegistry(),
        harness=harness,
        event_sink=events,
        config=ReActConfig(timeout_seconds=0.01),
    )

    result = await executor.run("slow provider", PromptIR.from_parts(dynamic="task"))

    assert result.status == "timeout"
    assert result.iterations == 1
    assert result.metadata["interrupt"]["kind"] == "timeout"
    assert harness.finished[-1]["status"] == "timeout"
    assert harness.finished[-1]["result"]["interrupt"]["metadata"]["phase"] == "provider"
    assert harness.checkpoints[-1].state["status"] == "timeout"
    assert harness.checkpoints[-1].state["phase"] == "provider"
    assert any(event.type == "run_timeout" for event in events.events)


@pytest.mark.asyncio
async def test_react_executor_times_out_tool_call_with_interrupt_manifest() -> None:
    provider = MockLLMProvider(
        [
            {
                "action": "call_tool",
                "arguments": {"tool_name": "lookup", "arguments": {"query": "slow"}},
            }
        ]
    )
    tools = _BlockingToolRuntime()
    harness = InMemoryHarness()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=tools,
        action_registry=ActionRegistry(),
        harness=harness,
        config=ReActConfig(timeout_seconds=0.1),
    )

    result = await executor.run("slow tool", PromptIR.from_parts(dynamic="task"))

    assert result.status == "timeout"
    assert result.output == "tool call timed out"
    assert result.metadata["interrupt"]["metadata"]["phase"] == "tool"
    assert harness.finished[-1]["status"] == "timeout"
    assert harness.checkpoints[-1].state["phase"] == "tool"
    assert tools.invocations[0].tool_name == "lookup"


@pytest.mark.asyncio
async def test_react_executor_injects_memory_hits_into_provider_request() -> None:
    memory = MockMemory([MemoryHit(content="prior fact", score=0.9, source="kb")])
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}])
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime(),
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        memory=memory,
    )

    result = await executor.run("use memory", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert memory.queries[0].query == "use memory"
    assert provider.requests[0].messages[-1].name == "memory"
    assert "prior fact" in provider.requests[0].messages[-1].content


@pytest.mark.asyncio
async def test_react_executor_repairs_native_finish_without_output() -> None:
    provider = MockLLMProvider(
        [
            LLMResponse(
                tool_calls=(
                    LLMToolCall(
                        tool_name="finish",
                        arguments={"terminal": True},
                        call_id="finish-empty",
                    ),
                ),
                finish_reason="tool_calls",
            ),
            "final answer after repair",
        ]
    )
    harness = InMemoryHarness()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime(),
        action_registry=ActionRegistry(),
        harness=harness,
        config=ReActConfig(max_iterations=2, native_tool_calls=True),
    )

    result = await executor.run("finish with text", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert result.output == "final answer after repair"
    assert len(provider.requests) == 2
    assert "finish requires a non-empty output string" in provider.requests[1].messages[-1].content
    assert harness.checkpoints[0].state["status"] == "finish_error"
    assert harness.finished[-1]["result"]["output"] == "final answer after repair"


@pytest.mark.asyncio
async def test_react_executor_budgets_native_tool_messages_and_tool_inventory() -> None:
    provider = MockLLMProvider(
        [
            LLMResponse(
                tool_calls=(
                    LLMToolCall(
                        tool_name="finish",
                        arguments={"output": "done"},
                        call_id="finish-ok",
                    ),
                ),
                finish_reason="tool_calls",
            ),
        ]
    )
    tools = ToolRegistry()

    async def noop(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            content="unused",
        )

    tools.register(
        ToolSpec(
            name="inspect_alpha",
            description="Return alpha callback-state nonce evidence",
            parameters_schema={"type": "object", "properties": {}},
            tags=("alpha", "callback"),
        ),
        noop,
    )
    for index in range(29):
        tools.register(
            ToolSpec(
                name=f"irrelevant_{index:02d}",
                description="Unrelated verbose tool " + ("x" * 240),
                parameters_schema={
                    "type": "object",
                    "properties": {"payload": {"type": "string", "description": "x" * 80}},
                },
                tags=("noise",),
            ),
            noop,
        )
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=tools,
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        config=ReActConfig(
            max_iterations=1,
            native_tool_calls=True,
            budget=RuntimeBudget(max_prompt_bytes=1400),
            loop_delta_max_bytes=512,
        ),
    )

    result = await executor.run(
        "Alpha task: use inspect_alpha for callback nonce review.",
        PromptIR.from_parts(
            high_static="stable rules",
            dynamic="Alpha task: inspect callback-state nonce handling. " + ("context " * 80),
        ),
    )

    request = provider.requests[0]
    tool_names = [tool.name for tool in request.tools]
    assert result.status == "completed"
    assert _messages_bytes(request.messages) <= 1400
    assert _messages_bytes(request.messages) + _native_tool_contracts_bytes(request.tools) <= 1400
    assert "inspect_alpha" in tool_names
    assert len(tool_names) < 30


@pytest.mark.asyncio
async def test_mock_provider_stream_emits_action_and_message_end() -> None:
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}])

    events = [
        event
        async for event in provider.stream(request=LLMRequest(messages=[]))
    ]

    assert [event.type for event in events] == ["action", "message_end"]
    assert events[0].action == {"action": "finish", "arguments": {"output": "done"}}


@pytest.mark.asyncio
async def test_react_executor_can_drive_loop_from_stream_events() -> None:
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "streamed"}}])
    events = ListEventSink()
    harness = InMemoryHarness()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime(),
        action_registry=ActionRegistry(),
        harness=harness,
        event_sink=events,
        config=ReActConfig(stream=True),
    )

    result = await executor.run("stream task", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert result.output == "streamed"
    assert harness.model_events[0]["metadata"]["streamed"] is True
    assert harness.model_events[0]["metadata"]["stream"]["has_action"] is True
    assert [event.type for event in events.events if event.type == "model_stream"]


@pytest.mark.asyncio
async def test_react_executor_retries_retryable_tool_failures() -> None:
    attempts = 0
    registry = ToolRegistry()

    async def lookup(invocation: ToolInvocation) -> ToolResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ToolResult(
                call_id=invocation.call_id,
                tool_name=invocation.tool_name,
                status="failed",
                error="temporary",
                metadata={"retryable": True},
            )
        return ToolResult(call_id=invocation.call_id, tool_name=invocation.tool_name, content="ok")

    registry.register(ToolSpec(name="lookup"), lookup)
    provider = MockLLMProvider(
        [
            {"action": "call_tool", "arguments": {"tool_name": "lookup", "arguments": {}}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    events = ListEventSink()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=registry,
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        event_sink=events,
        config=ReActConfig(
            max_iterations=3,
            tool_retry_policy=ToolRetryPolicy(max_attempts=2),
        ),
    )

    result = await executor.run("retry tool", PromptIR.from_parts(dynamic="task"))
    tool_finished = [event for event in events.events if event.type == "tool_finished"][-1]

    assert result.status == "completed"
    assert attempts == 2
    assert provider.requests[1].messages[-1].content == "ok"
    assert tool_finished.payload["execution"]["attempt_count"] == 2
    assert tool_finished.payload["execution"]["retried"] is True


@pytest.mark.asyncio
async def test_react_executor_replays_duplicate_tool_invocation() -> None:
    provider = MockLLMProvider(
        [
            {
                "action": "call_tool",
                "arguments": {"tool_name": "lookup", "arguments": {"q": "secret-target-value"}},
            },
            {
                "action": "call_tool",
                "arguments": {"tool_name": "lookup", "arguments": {"q": "secret-target-value"}},
            },
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    tools = MockToolRuntime({"lookup": "cached result"})
    events = ListEventSink()
    replay = InMemoryToolReplay()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=tools,
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        event_sink=events,
        tool_replay=replay,
        config=ReActConfig(max_iterations=4),
    )

    result = await executor.run("replay task", PromptIR.from_parts(dynamic="task"))
    manifest = replay.manifest()

    assert result.status == "completed"
    assert len(tools.invocations) == 1
    assert any(event.payload.get("replayed") for event in events.events if event.type == "tool_finished")
    assert manifest["record_count"] == 1
    assert manifest["records"][0]["invocation"]["argument_keys"] == ["q"]
    assert "secret-target-value" not in str(manifest["records"][0]["invocation"])


@pytest.mark.asyncio
async def test_persistent_tool_replay_uses_store_port_and_manifests_records() -> None:
    store = InMemoryToolReplayStore()
    replay = PersistentToolReplay(store)
    invocation = ToolInvocation(tool_name="lookup", arguments={"query": "target"})
    result = ToolResult(call_id=invocation.call_id, tool_name="lookup", content="found")

    await replay.put(invocation, result)
    replayed = await replay.get(invocation)
    manifest = await replay.manifest()

    assert replayed == result
    assert manifest["schema_version"] == "agent-core-persistent-tool-replay/v1"
    assert manifest["store"]["schema_version"] == "agent-core-in-memory-tool-replay-store/v1"
    assert manifest["records"][0]["result"]["content_sha256"]


@pytest.mark.asyncio
async def test_sqlite_tool_replay_store_persists_records_across_instances(tmp_path) -> None:
    path = tmp_path / "tool_replay.sqlite"
    invocation = ToolInvocation(tool_name="lookup", arguments={"query": "target"})
    result = ToolResult(
        call_id=invocation.call_id,
        tool_name="lookup",
        content="found",
        data={"ok": True},
    )
    first = PersistentToolReplay(SQLiteToolReplayStore(path))

    await first.put(invocation, result)
    second = PersistentToolReplay(SQLiteToolReplayStore(path))
    replayed = await second.get(invocation)
    manifest = await second.manifest()

    assert replayed == result
    assert manifest["store"]["schema_version"] == "agent-core-sqlite-tool-replay-store/v1"
    assert manifest["record_count"] == 1
    assert manifest["records"][0]["invocation"]["argument_keys"] == ["query"]


@pytest.mark.asyncio
async def test_markdown_tool_replay_store_persists_records_and_survives_comment_markers(tmp_path) -> None:
    path = tmp_path / "tool_replay.md"
    invocation = ToolInvocation(tool_name="lookup", arguments={"query": "target"})
    result = ToolResult(
        call_id=invocation.call_id,
        tool_name="lookup",
        content="found --> still stored",
        metadata={"source": "test"},
    )
    first = PersistentToolReplay(MarkdownToolReplayStore(path))

    await first.put(invocation, result)
    second = PersistentToolReplay(MarkdownToolReplayStore(path))
    replayed = await second.get(invocation)
    manifest = await second.manifest()

    assert replayed == result
    assert "found --> still stored" not in path.read_text(encoding="utf-8")
    assert manifest["store"]["schema_version"] == "agent-core-markdown-tool-replay-store/v1"
    assert manifest["record_count"] == 1
    assert manifest["records"][0]["result"]["content_sha256"]


@pytest.mark.asyncio
async def test_react_executor_compacts_large_tool_result_for_prompt() -> None:
    provider = MockLLMProvider(
        [
            {"action": "call_tool", "arguments": {"tool_name": "dump", "arguments": {}}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    tools = MockToolRuntime({"dump": "A" * 2000})
    harness = InMemoryHarness()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=tools,
        action_registry=ActionRegistry(),
        harness=harness,
        config=ReActConfig(max_iterations=3, budget=RuntimeBudget(max_tool_result_bytes=160)),
    )

    result = await executor.run("large tool", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert "[tool_result_compacted" in provider.requests[1].messages[-1].content
    assert harness.tool_calls[0]["metadata"]["compacted"] is True


@pytest.mark.asyncio
async def test_react_executor_stores_large_tool_result_as_artifact() -> None:
    large = "full-result-" + ("A" * 2000)
    provider = MockLLMProvider(
        [
            {"action": "call_tool", "arguments": {"tool_name": "dump", "arguments": {}}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    artifacts = InMemoryArtifactStore()
    harness = InMemoryHarness()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime({"dump": large}),
        action_registry=ActionRegistry(),
        harness=harness,
        artifact_store=artifacts,
        config=ReActConfig(max_iterations=3, budget=RuntimeBudget(max_tool_result_bytes=160)),
    )

    result = await executor.run("large artifact", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert len(artifacts.records) == 1
    artifact = next(iter(artifacts.records.values()))
    assert artifact.text() == large
    prompt_content = provider.requests[1].messages[-1].content
    assert "[tool_result_compacted" in prompt_content
    assert "[artifact_ref uri=artifact://" in prompt_content
    assert large not in prompt_content
    assert harness.tool_calls[0]["metadata"]["artifact"]["uri"] == artifact.uri
    assert harness.tool_calls[0]["metadata"]["artifact"]["sha256"] == artifact.sha256


@pytest.mark.asyncio
async def test_react_executor_records_timeline_items() -> None:
    provider = MockLLMProvider(
        [
            {"action": "call_tool", "arguments": {"tool_name": "lookup", "arguments": {}}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    timeline = TimelineStore()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime({"lookup": "tool output"}),
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        timeline=timeline,
        config=ReActConfig(max_iterations=3),
    )

    result = await executor.run("timeline task", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    kinds = [item.kind for item in timeline.items]
    assert "task" in kinds
    assert "model" in kinds
    assert "tool" in kinds


@pytest.mark.asyncio
async def test_react_executor_supports_search_tools_core_action() -> None:
    tools = ToolRegistry()

    async def handler(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(call_id=invocation.call_id, tool_name=invocation.tool_name)

    tools.register(ToolSpec(name="http_probe", description="Probe HTTP", tags=("web",)), handler)
    provider = MockLLMProvider(
        [
            {"action": "search_tools", "arguments": {"query": "http web"}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=tools,
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        config=ReActConfig(max_iterations=3),
    )

    result = await executor.run("find tool", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert "http_probe" in provider.requests[1].messages[-1].content


@pytest.mark.asyncio
async def test_react_executor_dispatches_direct_tool_action() -> None:
    tools = ToolRegistry()

    @tools.register_function(aliases=("probe",))
    def http_probe(target: str) -> str:
        return f"probed {target}"

    provider = MockLLMProvider(
        [
            {"action": "probe", "arguments": {"target": "example.test"}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=tools,
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        config=ReActConfig(max_iterations=3),
    )

    result = await executor.run("direct tool", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert provider.requests[1].messages[-1].name == "http_probe"
    assert provider.requests[1].messages[-1].content == "probed example.test"


@pytest.mark.asyncio
async def test_react_executor_feedbacks_action_verifier_errors() -> None:
    class RejectFirstLookup:
        def __init__(self) -> None:
            self.calls = 0

        async def verify(self, action: ParsedAction) -> ActionVerification:
            self.calls += 1
            if self.calls == 1 and action.name == "call_tool":
                return ActionVerification(ok=False, message="lookup needs a safer query")
            return ActionVerification()

    provider = MockLLMProvider(
        [
            {"action": "call_tool", "arguments": {"tool_name": "lookup", "arguments": {"query": ""}}},
            {
                "action": "call_tool",
                "arguments": {"tool_name": "lookup", "arguments": {"query": "safe"}},
            },
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    tools = MockToolRuntime({"lookup": "ok"})
    harness = InMemoryHarness()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=tools,
        action_registry=ActionRegistry(),
        harness=harness,
        action_verifier=RejectFirstLookup(),
        config=ReActConfig(max_iterations=4),
    )

    result = await executor.run("verify tool", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert len(tools.invocations) == 1
    assert tools.invocations[0].arguments == {"query": "safe"}
    assert harness.checkpoints[0].state["status"] == "action_verifier_error"
    assert provider.requests[1].messages[-1].content.startswith(
        '{"feedback": "action_verifier_error"'
    )


@pytest.mark.asyncio
async def test_react_executor_supports_skill_load_and_resource_core_actions(tmp_path) -> None:
    skill_dir = tmp_path / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: review
description: Review code
tags: [code]
---
# Review

Use references.
""",
        encoding="utf-8",
    )
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "rules.md").write_text("\n".join(f"rule {i}" for i in range(20)), encoding="utf-8")
    registry = SkillRegistry()
    registry.discover_markdown_skills(tmp_path / "skills")
    skills = SkillsContext(registry)
    provider = MockLLMProvider(
        [
            {"action": "load_skill", "arguments": {"name": "review"}},
            {
                "action": "load_skill_resource",
                "arguments": {"ref": "@review/references/rules.md", "offset": 2},
            },
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime(),
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        skills=skills,
        config=ReActConfig(max_iterations=4),
    )

    result = await executor.run("use skill", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert "[skill:review]" in provider.requests[1].messages[-1].content
    resource_message = provider.requests[2].messages[-1]
    assert "VIEW_WINDOW" in resource_message.content
    assert "2 | rule 1" in resource_message.content

