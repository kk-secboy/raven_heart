from __future__ import annotations

import pytest

from agent_core.config import AgentProfile, CapabilitySet
from agent_core.handoff import (
    AgentToolRuntime,
    HandoffRequest,
    HandoffRouter,
    HandoffSpec,
    MultiAgentCoordinator,
    agent_tool_spec_from_session,
    handoff_spec_from_session,
)
from agent_core.runner import AgentRunner, AgentSession, AgentSessionManager
from agent_core.testing import MockLLMProvider, MockToolRuntime
from agent_core.tools import ToolInvocation


def test_handoff_router_selects_highest_priority_matching_agent() -> None:
    router = HandoffRouter(
        (
            HandoffSpec(
                session_name="generalist",
                tags=("code",),
                tools=("read",),
                priority=1,
            ),
            HandoffSpec(
                session_name="reviewer",
                tags=("code", "review"),
                tools=("read", "diff"),
                priority=10,
            ),
        )
    )

    decision = router.decide(
        HandoffRequest(
            task="review patch",
            required_tags=("code",),
            required_tools=("diff",),
        )
    )

    assert decision.selected
    assert decision.selected_session == "reviewer"
    assert [candidate.session_name for candidate in decision.candidates] == ["reviewer"]
    assert decision.manifest()["schema_version"] == "agent-core-handoff-decision/v1"


def test_handoff_router_rejects_explicit_target_that_misses_requirements() -> None:
    router = HandoffRouter(
        (
            HandoffSpec(
                session_name="ops",
                tags=("ops",),
                tools=("logs",),
            ),
        )
    )

    decision = router.decide(
        HandoffRequest(
            task="review code",
            target_session="ops",
            required_tags=("code",),
        )
    )

    assert not decision.selected
    assert decision.status == "denied"
    assert "does not satisfy" in decision.reason


@pytest.mark.asyncio
async def test_multi_agent_coordinator_runs_selected_session_and_records_handoff() -> None:
    manager = AgentSessionManager()
    manager.register(
        AgentSession(
            profile=AgentProfile(
                name="code-reviewer",
                capabilities=CapabilitySet(tools=("diff",), skills=("review",)),
            ),
            provider=MockLLMProvider([{"action": "finish", "arguments": {"output": "reviewed"}}]),
            tools=MockToolRuntime(),
            metadata={"tags": ("code", "review"), "handoff_priority": 5},
        )
    )
    manager.register(
        AgentSession(
            profile=AgentProfile(name="ops"),
            provider=MockLLMProvider([{"action": "finish", "arguments": {"output": "ops"}}]),
            tools=MockToolRuntime(),
            metadata={"tags": ("ops",)},
        )
    )
    coordinator = MultiAgentCoordinator.from_manager(manager)

    record = await coordinator.handoff(
        HandoffRequest(
            task="review the current patch",
            source_session="planner",
            required_tags=("review",),
            required_tools=("diff",),
        )
    )

    assert record.outcome is not None
    assert record.decision.selected_session == "code-reviewer"
    assert record.outcome.result.output == "reviewed"
    assert record.manifest()["outcome"]["result"]["status"] == "completed"
    assert record.outcome.trace_manifest["summary"]["handoff_selected_count"] == 1
    assert record.outcome.trace_manifest["handoff_trace"]["selected_sessions"] == {
        "code-reviewer": 1
    }
    assert coordinator.manifest()["record_count"] == 1


def test_handoff_spec_from_session_exports_profile_capabilities() -> None:
    session = AgentSession(
        profile=AgentProfile(
            name="specialist",
            instructions="Handle code edits.",
            capabilities=CapabilitySet(tools=("read",), skills=("python",)),
        ),
        provider=MockLLMProvider([]),
        tools=MockToolRuntime(),
        metadata={"tags": ("code",), "handoff_priority": 3},
    )

    spec = handoff_spec_from_session(session, session_name="specialist-v1")

    assert spec.session_name == "specialist-v1"
    assert spec.description == "Handle code edits."
    assert spec.tags == ("code",)
    assert spec.tools == ("read",)
    assert spec.skills == ("python",)
    assert spec.priority == 3


@pytest.mark.asyncio
async def test_agent_tool_runtime_invokes_managed_agent_session() -> None:
    manager = AgentSessionManager()
    manager.register(
        AgentSession(
            profile=AgentProfile(name="code-reviewer", instructions="Review code."),
            provider=MockLLMProvider(
                [{"action": "finish", "arguments": {"output": "reviewed patch"}}]
            ),
            tools=MockToolRuntime(),
            metadata={"tags": ("code", "review")},
        )
    )
    runtime = AgentToolRuntime.from_manager(manager)

    result = await runtime.invoke(
        ToolInvocation("agent_code_reviewer", {"task": "review private patch"})
    )
    manifest = runtime.manifest()

    assert result.status == "completed"
    assert result.content == "reviewed patch"
    assert result.data["status"] == "completed"
    assert result.metadata["tool"]["session_name"] == "code-reviewer"
    assert manifest["schema_version"] == "agent-core-agent-tool-runtime/v1"
    assert manifest["record_count"] == 1
    assert manifest["records"][0]["task_bytes"] == len("review private patch")
    assert "review private patch" not in str(manifest)


@pytest.mark.asyncio
async def test_parent_agent_can_call_child_agent_as_tool() -> None:
    child = AgentSession(
        profile=AgentProfile(name="code-reviewer", instructions="Review code."),
        provider=MockLLMProvider(
            [{"action": "finish", "arguments": {"output": "child reviewed"}}]
        ),
        tools=MockToolRuntime(),
    )
    manager = AgentSessionManager()
    manager.register(child)
    agent_tools = AgentToolRuntime.from_manager(manager)
    parent = AgentSession(
        profile=AgentProfile(name="planner"),
        provider=MockLLMProvider(
            [
                {
                    "action": "call_tool",
                    "arguments": {
                        "tool_name": "agent_code_reviewer",
                        "arguments": {"task": "review patch"},
                    },
                },
                {"action": "finish", "arguments": {"output": "parent done"}},
            ]
        ),
        tools=agent_tools,
    )

    outcome = await AgentRunner(parent).run("delegate review")

    assert outcome.result.status == "completed"
    assert outcome.result.output == "parent done"
    assert agent_tools.records[0]["result"]["status"] == "completed"
    assert agent_tools.records[0]["result"]["output_bytes"] == len("child reviewed")
    assert (
        parent.harness.tool_calls[0]["metadata"]["tool"]["session_name"]
        == "code-reviewer"
    )


def test_agent_tool_spec_from_session_exports_tool_schema() -> None:
    session = AgentSession(
        profile=AgentProfile(name="ops-helper", instructions="Handle ops checks."),
        provider=MockLLMProvider([]),
        tools=MockToolRuntime(),
        metadata={"tags": ("ops",), "agent_tool_enabled": False},
    )

    spec = agent_tool_spec_from_session(session, session_name="ops-helper")
    tool = spec.tool_spec()

    assert spec.tool_name == "agent_ops_helper"
    assert spec.enabled is False
    assert tool.name == "agent_ops_helper"
    assert tool.enabled is False
    assert tool.parameters_schema["required"] == ["task"]
    assert "agent_tool" in tool.tags
