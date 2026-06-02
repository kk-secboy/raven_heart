from __future__ import annotations

import asyncio

import pytest

from agent_core.config import AgentProfile, CapabilitySet, RuntimeBudget
from agent_core.context import AgentContextPack
from agent_core.harness import InMemoryAgentJournal
from agent_core.memory import InMemoryMemoryStore, MemoryRecord
from agent_core.mcp import MCPCenter, MCPServerSpec
from agent_core.providers import LLMProviderCenter
from agent_core.providers import LLMRequest, LLMResponse
from agent_core.runner import AgentRunner, AgentRunRequest, AgentSession, AgentSessionManager
from agent_core.skills import SkillRegistry, SkillsContext, SkillSpec
from agent_core.testing import MockLLMProvider, MockToolRuntime
from agent_core.tools import ToolInvocation, ToolRegistry, ToolResult, ToolSpec


class _BlockingProvider:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        await self.release.wait()
        return LLMResponse(action={"action": "finish", "arguments": {"output": "done"}})


@pytest.mark.asyncio
async def test_agent_runner_executes_react_with_profile_context_and_manifest() -> None:
    provider = MockLLMProvider(
        [
            {
                "action": "call_tool",
                "arguments": {"tool_name": "lookup", "arguments": {"query": "x"}},
            },
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    center = LLMProviderCenter(default_provider="mock")
    center.register("mock", provider, default_model="mock-mini")
    tools = MockToolRuntime({"lookup": "found"})
    journal = InMemoryAgentJournal()
    registry = SkillRegistry()
    registry.register(SkillSpec(name="recon", description="Recon skill", prompt="Use passive recon."))
    skills = SkillsContext(registry)
    skills.load("recon")
    memory = InMemoryMemoryStore(
        (
            MemoryRecord(content="Inspect target admin UI at /admin", source="memory"),
        )
    )
    session = AgentSession(
        profile=AgentProfile(
            name="core-test",
            model="mock-mini",
            instructions="Follow core rules.",
            capabilities=CapabilitySet(memory_enabled=True),
            budget=RuntimeBudget(max_iterations=4, max_cost_usd=0.5),
        ),
        provider=center,
        tools=tools,
        harness=journal,
        skills=skills,
        memory=memory,
        metadata={"runtime": "test"},
    )

    outcome = await AgentRunner(session).run(
        AgentRunRequest(
            task="inspect target",
            context=AgentContextPack(workspace="workspace note"),
            metadata={"request_id": "r1"},
        )
    )

    assert outcome.result.status == "completed"
    assert outcome.result.output == "done"
    assert provider.requests[0].model == "mock-mini"
    assert provider.requests[0].metadata["max_cost_usd"] == 0.5
    assert "Follow core rules." in provider.requests[0].messages[0].content
    assert "workspace note" in provider.requests[0].messages[0].content
    assert "target admin UI" in provider.requests[0].messages[-1].content
    assert tools.invocations[0].tool_name == "lookup"
    assert outcome.session_manifest["profile"]["name"] == "core-test"
    assert outcome.session_manifest["capabilities"]["skills"]["loaded_skills"][0]["name"] == "recon"
    assert outcome.prompt_manifest["metadata"]["request_id"] == "r1"
    assert journal.finished[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_agent_runner_respects_profile_memory_disabled() -> None:
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}])
    memory = InMemoryMemoryStore((MemoryRecord(content="hidden memory", source="memory"),))
    session = AgentSession(
        profile=AgentProfile(
            name="no-memory",
            capabilities=CapabilitySet(memory_enabled=False),
        ),
        provider=provider,
        tools=MockToolRuntime(),
        memory=memory,
    )

    outcome = await AgentRunner(session).run("task")

    assert outcome.result.status == "completed"
    assert all(message.name != "memory" for message in provider.requests[0].messages)


@pytest.mark.asyncio
async def test_agent_runner_trims_prompt_to_profile_budget() -> None:
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}])
    session = AgentSession(
        profile=AgentProfile(
            name="trimmed",
            instructions="stable rules",
            budget=RuntimeBudget(max_prompt_bytes=900),
        ),
        provider=provider,
        tools=MockToolRuntime(),
    )

    outcome = await AgentRunner(session).run(
        AgentRunRequest(
            task="current task",
            context=AgentContextPack(workspace="old observation " + ("o" * 1200) + " latest"),
        )
    )

    prompt_text = provider.requests[0].messages[0].content
    trim = outcome.prompt_manifest["metadata"]["trim"]

    assert outcome.result.status == "completed"
    assert len(prompt_text.encode("utf-8")) <= 900
    assert "stable rules" in prompt_text
    assert "current task" in prompt_text
    assert "[...trimmed...]" in prompt_text
    assert trim["target_bytes"] == 900


@pytest.mark.asyncio
async def test_agent_runner_injects_resume_checkpoint_context() -> None:
    journal = InMemoryAgentJournal()
    original_run = await journal.start_run("original task")
    original_turn = await journal.start_turn(original_run, 0)
    checkpoint = await journal.checkpoint(
        original_turn,
        {
            "status": "tool_finished",
            "last_tool": "lookup",
            "observation": "admin UI found",
        },
    )
    token = journal.resume_token(checkpoint)
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "resumed"}}])
    session = AgentSession(
        profile=AgentProfile(name="resumable"),
        provider=provider,
        tools=MockToolRuntime(),
        harness=journal,
    )

    outcome = await AgentRunner(session).run(
        AgentRunRequest(task="continue task", resume_token=token)
    )

    assert outcome.result.status == "completed"
    assert outcome.result.output == "resumed"
    assert outcome.resume_manifest["checkpoint_id"] == checkpoint.checkpoint_id
    assert outcome.resume_manifest["state"]["observation"] == "admin UI found"
    assert outcome.prompt_manifest["metadata"]["resume"]["checkpoint_id"] == checkpoint.checkpoint_id
    prompt_text = provider.requests[0].messages[0].content
    assert "== Resumed Checkpoint ==" in prompt_text
    assert "admin UI found" in prompt_text
    assert "last_tool: lookup" in prompt_text


@pytest.mark.asyncio
async def test_agent_runner_refreshes_tools_and_tolerates_partial_mcp_refresh() -> None:
    registry = ToolRegistry()

    async def lookup(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(call_id=invocation.call_id, tool_name=invocation.tool_name, content="ok")

    registry.register(ToolSpec(name="lookup"), lookup)
    mcp = MCPCenter()
    mcp.register_server(MCPServerSpec(name="missing", transport="stdio"))
    session = AgentSession(
        profile=AgentProfile(name="refresh"),
        provider=MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}]),
        tools=registry,
        mcp=mcp,
    )

    outcome = await AgentRunner(session).run(AgentRunRequest(task="task", refresh=True))

    assert outcome.result.status == "completed"
    assert session.metadata["mcp_resource_refresh_error"].startswith("no MCP connector")
    assert session.metadata["mcp_prompt_refresh_error"].startswith("no MCP connector")
    assert outcome.session_manifest["capabilities"]["tools"]["tools"][0]["name"] == "lookup"


@pytest.mark.asyncio
async def test_agent_session_manager_runs_and_indexes_sessions() -> None:
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}])
    session = AgentSession(
        profile=AgentProfile(name="managed"),
        provider=provider,
        tools=MockToolRuntime(),
    )
    manager = AgentSessionManager()
    manager.register(session)

    outcome = await manager.run("managed", AgentRunRequest(task="task", metadata={"request_id": "r1"}))
    run = manager.runs()[0]

    assert outcome.result.status == "completed"
    assert run.status == "completed"
    assert run.result_run_id == outcome.result.run_id
    assert manager.outcome(run.run_key) == outcome
    assert manager.manifest()["sessions"]["managed"]["profile"]["name"] == "managed"
    assert manager.manifest()["runs"][0]["metadata"]["request_id"] == "r1"


@pytest.mark.asyncio
async def test_agent_session_manager_runs_with_resume_token() -> None:
    journal = InMemoryAgentJournal()
    original_run = await journal.start_run("managed original")
    original_turn = await journal.start_turn(original_run, 0)
    checkpoint = await journal.checkpoint(original_turn, {"step": "halfway"})
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "resumed"}}])
    session = AgentSession(
        profile=AgentProfile(name="managed-resume"),
        provider=provider,
        tools=MockToolRuntime(),
        harness=journal,
    )
    manager = AgentSessionManager()
    manager.register(session)

    outcome = await manager.run(
        "managed-resume",
        AgentRunRequest(task="continue", resume_token=journal.resume_token(checkpoint)),
    )
    run = manager.runs()[0]

    assert outcome.result.status == "completed"
    assert outcome.resume_manifest["state"]["step"] == "halfway"
    assert run.status == "completed"
    assert run.result_run_id == outcome.result.run_id


@pytest.mark.asyncio
async def test_agent_session_manager_background_run_rejects_same_session_concurrency() -> None:
    provider = _BlockingProvider()
    session = AgentSession(
        profile=AgentProfile(name="slow"),
        provider=provider,
        tools=MockToolRuntime(),
    )
    manager = AgentSessionManager()
    manager.register(session)

    run_key = manager.start("slow", "task")
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError):
        manager.start("slow", "second task")

    assert manager.run_state(run_key).status == "running"
    provider.release.set()
    outcome = await manager.wait(run_key)

    assert outcome.result.status == "completed"
    assert not manager.active_runs()


@pytest.mark.asyncio
async def test_agent_session_manager_cancels_background_run() -> None:
    provider = _BlockingProvider()
    session = AgentSession(
        profile=AgentProfile(name="cancel"),
        provider=provider,
        tools=MockToolRuntime(),
    )
    manager = AgentSessionManager()
    manager.register(session)

    run_key = manager.start("cancel", "task")
    assert manager.cancel(run_key, "stop now") is True
    provider.release.set()
    outcome = await manager.wait(run_key)

    assert outcome.result.status == "cancelled"
    assert manager.run_state(run_key).status == "cancelled"
    assert manager.cancel(run_key) is False

