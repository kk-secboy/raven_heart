from __future__ import annotations

import pytest

from agent_core.actions import (
    ActionError,
    ActionRegistry,
    ActionSpec,
    ActionVerification,
    ParsedAction,
)
from agent_core.events import ListEventSink
from agent_core.artifacts import InMemoryArtifactStore
from agent_core.harness import CancelToken
from agent_core.loop_guard import LoopGuardConfig
from agent_core.memory import MemoryHit
from agent_core.prompt import PromptIR
from agent_core.providers import LLMRequest, LLMResponse
from agent_core.react import ReActConfig, ReActExecutor
from agent_core.config import RuntimeBudget
from agent_core.skills import SkillRegistry, SkillsContext
from agent_core.timeline import TimelineStore
from agent_core.tools import InMemoryToolReplay, ToolInvocation, ToolRegistry, ToolResult, ToolSpec
from agent_core.testing import InMemoryHarness, MockLLMProvider, MockMemory, MockToolRuntime


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
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime(),
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        event_sink=events,
        config=ReActConfig(stream=True),
    )

    result = await executor.run("stream task", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert result.output == "streamed"
    assert [event.type for event in events.events if event.type == "model_stream"]


@pytest.mark.asyncio
async def test_react_executor_replays_duplicate_tool_invocation() -> None:
    provider = MockLLMProvider(
        [
            {"action": "call_tool", "arguments": {"tool_name": "lookup", "arguments": {"q": "x"}}},
            {"action": "call_tool", "arguments": {"tool_name": "lookup", "arguments": {"q": "x"}}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    tools = MockToolRuntime({"lookup": "cached result"})
    events = ListEventSink()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=tools,
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        event_sink=events,
        tool_replay=InMemoryToolReplay(),
        config=ReActConfig(max_iterations=4),
    )

    result = await executor.run("replay task", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert len(tools.invocations) == 1
    assert any(event.payload.get("replayed") for event in events.events if event.type == "tool_finished")


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

