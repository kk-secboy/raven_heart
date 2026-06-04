from __future__ import annotations

import pytest

from agent_core.config import AgentProfile, CapabilitySet
from agent_core.handoff import (
    HandoffRequest,
    HandoffRouter,
    HandoffSpec,
    MultiAgentCoordinator,
    handoff_spec_from_session,
)
from agent_core.runner import AgentSession, AgentSessionManager
from agent_core.testing import MockLLMProvider, MockToolRuntime


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
