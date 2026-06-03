from __future__ import annotations

import asyncio

import pytest

from agent_core.approvals import ApprovalDecisionRecord, ApprovalResumeContext, InMemoryApprovalStore
from agent_core.config import AgentProfile, CapabilitySet, RuntimeBudget
from agent_core.context import AgentContextPack, ContextInjectionPolicy
from agent_core.errors import ResumeError
from agent_core.harness import InMemoryAgentJournal
from agent_core.memory import InMemoryMemoryStore, MemoryCenter, MemoryRecord
from agent_core.mcp import MCPCenter, MCPServerSpec
from agent_core.providers import LLMProviderCenter
from agent_core.providers import LLMRequest, LLMResponse, LLMToolCall
from agent_core.prompt import PromptBucketBudgetPolicy, PromptBucketBudgetRule, PromptBucketRole
from agent_core.reducer import DefaultContextReducer
from agent_core.runner import (
    AgentManagerCapacityError,
    AgentManagerConcurrencyPolicy,
    AgentResumeRequest,
    AgentRunner,
    AgentRunRequest,
    AgentSession,
    AgentSessionManager,
    InMemoryAgentRunStore,
    ManagedAgentRun,
    MarkdownAgentRunStore,
    SQLiteAgentRunStore,
)
from agent_core.skills import SkillRegistry, SkillsContext, SkillSpec
from agent_core.testing import MockLLMProvider, MockToolRuntime
from agent_core.timeline import TimelineStore
from agent_core.tools import ToolInvocation, ToolRegistry, ToolResult, ToolSpec
from agent_core.policy import ApprovalRequest, PolicyRule, RuleBasedPolicy


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
    memory = MemoryCenter(default_store="local")
    memory.register(
        "local",
        InMemoryMemoryStore(
            (
                MemoryRecord(content="Inspect target admin UI at /admin", source="memory"),
            )
        ),
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
    prompt_text = provider.requests[0].messages[0].content
    assert "Follow core rules." in prompt_text
    assert "workspace note" in prompt_text
    assert "[context_injection:memory_recall source=memory]" in prompt_text
    assert "target admin UI" in prompt_text
    assert all(message.name != "memory" for message in provider.requests[0].messages)
    assert tools.invocations[0].tool_name == "lookup"
    assert outcome.session_manifest["profile"]["name"] == "core-test"
    assert outcome.session_manifest["capabilities"]["skills"]["loaded_skills"][0]["name"] == "recon"
    assert outcome.prompt_manifest["metadata"]["request_id"] == "r1"
    memory_injection = outcome.prompt_manifest["metadata"]["context_injections"][0]
    assert memory_injection["name"] == "memory_recall"
    assert memory_injection["source"] == "memory"
    assert memory_injection["metadata"]["hit_count"] == 1
    assert outcome.memory_search_manifest["hit_count"] == 1
    assert outcome.memory_search_manifest["plan"]["schema_version"] == "agent-core-memory-search-plan/v1"
    assert outcome.trace_manifest["summary"]["memory_search_hit_count"] == 1
    assert outcome.trace_manifest["memory_search"]["hits"][0]["content_sha256"]
    assert journal.finished[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_agent_runner_can_enable_provider_native_tool_calls_per_request() -> None:
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
        profile=AgentProfile(name="native-tools", model="mock-mini"),
        provider=center,
        tools=tools,
    )

    outcome = await AgentRunner(session).run(
        AgentRunRequest(task="inspect", native_tool_calls=True)
    )

    assert outcome.result.status == "completed"
    assert provider.requests[0].tools[0].name == "lookup"
    assert provider.requests[0].tool_choice is not None
    assert provider.requests[1].messages[-1].role == "tool"
    assert provider.requests[1].messages[-1].content == "found"
    assert tools.invocations[0].call_id == "call-1"
    assert tools.invocations[0].arguments == {"query": "target"}
    assert outcome.trace_manifest["metadata"]["native_tool_calls"] is True
    assert center.calls[0].metadata["response"]["tool_call_count"] == 1


@pytest.mark.asyncio
async def test_agent_runner_native_tool_call_session_default_can_be_overridden() -> None:
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}])
    session = AgentSession(
        profile=AgentProfile(name="native-default"),
        provider=provider,
        tools=MockToolRuntime({"lookup": "found"}),
        native_tool_calls=True,
    )

    default_outcome = await AgentRunner(session).run("default native")
    override_outcome = await AgentRunner(session).run(
        AgentRunRequest(task="override native", native_tool_calls=False)
    )

    assert default_outcome.session_manifest["native_tool_calls"] is True
    assert default_outcome.trace_manifest["metadata"]["native_tool_calls"] is True
    assert provider.requests[0].tools[0].name == "lookup"
    assert override_outcome.trace_manifest["metadata"]["native_tool_calls"] is False
    assert provider.requests[1].tools == ()


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
async def test_agent_runner_applies_context_injection_policy_to_memory_recall() -> None:
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}])
    memory = InMemoryMemoryStore(
        (
            MemoryRecord(
                content="target admin UI " * 80,
                source="memory",
                metadata={"scope": "project"},
            ),
        )
    )
    session = AgentSession(
        profile=AgentProfile(name="injection-policy"),
        provider=provider,
        tools=MockToolRuntime(),
        memory=memory,
        context_injection_policy=ContextInjectionPolicy(max_injection_bytes=120),
    )

    outcome = await AgentRunner(session).run("inspect target")
    injection = outcome.prompt_manifest["metadata"]["context_injections"][0]
    prompt_text = provider.requests[0].messages[0].content

    assert outcome.result.status == "completed"
    assert injection["name"] == "memory_recall"
    assert injection["status"] == "trimmed"
    assert injection["included"] is True
    assert injection["final_bytes"] <= 120
    assert "[...context injection trimmed...]" in prompt_text
    assert outcome.session_manifest["context_injection_policy"]["max_injection_bytes"] == 120
    assert outcome.trace_manifest["prompt"]["metadata"]["context_injections"][0]["status"] == "trimmed"


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
    assert outcome.trace_manifest["summary"]["has_prompt_trim"] is True
    assert outcome.trace_manifest["metadata"]["prompt_trim"]["target_bytes"] == 900


@pytest.mark.asyncio
async def test_agent_runner_applies_prompt_bucket_budget_policy_before_global_trim() -> None:
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}])
    session = AgentSession(
        profile=AgentProfile(name="bucket-budget", budget=RuntimeBudget(max_prompt_bytes=4000)),
        provider=provider,
        tools=MockToolRuntime(),
        prompt_bucket_budget_policy=PromptBucketBudgetPolicy(
            rules=(
                PromptBucketBudgetRule(
                    role=PromptBucketRole.SEMI_DYNAMIC_1,
                    max_bytes=140,
                    min_keep_bytes=80,
                    reason="cap memory and skills before global prompt trim",
                ),
                PromptBucketBudgetRule(
                    role=PromptBucketRole.DYNAMIC,
                    max_bytes=80,
                    protected=True,
                    reason="preserve current task",
                ),
            )
        ),
    )

    outcome = await AgentRunner(session).run(
        AgentRunRequest(
            task="inspect",
            context=AgentContextPack(
                recent_tools_cache="recent tool " + ("x" * 600),
                dynamic_task="current task " + ("d" * 120),
            ),
        )
    )
    prompt_text = provider.requests[0].messages[0].content
    bucket_budget = outcome.prompt_manifest["metadata"]["bucket_budget"]

    assert outcome.result.status == "completed"
    assert bucket_budget["trimmed_count"] == 1
    assert bucket_budget["protected_count"] == 1
    assert "[...bucket budget trimmed...]" in prompt_text
    assert "current task" in prompt_text
    assert outcome.session_manifest["prompt_bucket_budget_policy"]["enabled"] is True
    assert outcome.trace_manifest["summary"]["has_prompt_bucket_budget"] is True
    assert outcome.trace_manifest["summary"]["prompt_bucket_budget_trimmed_count"] == 1
    assert outcome.trace_manifest["prompt_bucket_budget"]["trimmed_count"] == 1


@pytest.mark.asyncio
async def test_agent_runner_applies_optional_timeline_reducer_before_prompt_build() -> None:
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}])
    timeline = TimelineStore()
    old = timeline.add("old observation " + ("a" * 500), kind="observation")
    middle = timeline.add("middle observation " + ("b" * 500), kind="observation")
    latest = timeline.add("latest observation should remain visible", kind="observation")
    session = AgentSession(
        profile=AgentProfile(
            name="timeline-reduced",
            budget=RuntimeBudget(max_timeline_bytes=320),
        ),
        provider=provider,
        tools=MockToolRuntime(),
        timeline=timeline,
        context_reducer=DefaultContextReducer(),
    )

    outcome = await AgentRunner(session).run("summarize timeline")

    prompt_text = provider.requests[0].messages[0].content
    reduction = outcome.timeline_reduction_manifest

    assert outcome.result.status == "completed"
    assert reduction["metadata"]["compressed_item_count"] == 2
    assert reduction["request"]["max_bytes"] == 320
    assert reduction["view"]["open_item_count"] == 1
    assert timeline.items[0].deleted is True
    assert timeline.items[1].deleted is True
    assert timeline.items[2].item_id == latest.item_id
    assert "[compressed_head]" in prompt_text
    assert f"timeline:{old.item_id}" in prompt_text
    assert f"timeline:{middle.item_id}" in prompt_text
    assert "latest observation should remain visible" in prompt_text
    assert outcome.prompt_manifest["metadata"]["timeline_reduction"]["metadata"][
        "compressed_item_count"
    ] == 2
    assert outcome.session_manifest["context_reducer"]["enabled"] is True
    assert outcome.session_manifest["context_reducer"]["type"] == "DefaultContextReducer"


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
    injection = outcome.prompt_manifest["metadata"]["context_injections"][0]
    assert injection["name"] == "resume_checkpoint"
    assert injection["source"] == "harness"
    assert injection["metadata"]["checkpoint_id"] == checkpoint.checkpoint_id
    prompt_text = provider.requests[0].messages[0].content
    assert "[context_injection:resume_checkpoint source=harness]" in prompt_text
    assert "== Resumed Checkpoint ==" in prompt_text
    assert "admin UI found" in prompt_text
    assert "last_tool: lookup" in prompt_text


@pytest.mark.asyncio
async def test_agent_runner_resume_selects_latest_checkpoint_candidate() -> None:
    journal = InMemoryAgentJournal()
    first_run = await journal.start_run("first task")
    first_turn = await journal.start_turn(first_run, 0)
    await journal.checkpoint(first_turn, {"step": "old"})
    second_run = await journal.start_run("second task")
    second_turn = await journal.start_turn(second_run, 0)
    latest = await journal.checkpoint(second_turn, {"step": "latest"})
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "resumed"}}])
    session = AgentSession(
        profile=AgentProfile(name="runner-resume"),
        provider=provider,
        tools=MockToolRuntime(),
        harness=journal,
    )

    runner = AgentRunner(session)
    plan = runner.resume_plan(AgentResumeRequest(task="continue latest", metadata={"request_id": "r1"}))
    outcome = await runner.resume(AgentResumeRequest(task="continue latest", metadata={"request_id": "r1"}))

    assert plan.ready is True
    assert plan.summary_manifest()["checkpoint_id"] == latest.checkpoint_id
    assert outcome.result.status == "completed"
    assert outcome.resume_manifest["checkpoint_id"] == latest.checkpoint_id
    assert outcome.resume_plan_manifest["checkpoint_id"] == latest.checkpoint_id
    assert outcome.resume_plan_manifest["status"] == "ready"
    assert outcome.resume_manifest["state"]["step"] == "latest"
    assert outcome.prompt_manifest["metadata"]["request_id"] == "r1"
    assert outcome.prompt_manifest["metadata"]["resume"]["checkpoint_id"] == latest.checkpoint_id
    assert outcome.prompt_manifest["metadata"]["resume_plan"]["checkpoint_id"] == latest.checkpoint_id
    assert provider.requests[0].messages[0].content.count("== Resumed Checkpoint ==") == 1


@pytest.mark.asyncio
async def test_agent_runner_resume_can_target_specific_run() -> None:
    journal = InMemoryAgentJournal()
    first_run = await journal.start_run("first task")
    first_turn = await journal.start_turn(first_run, 0)
    first_checkpoint = await journal.checkpoint(first_turn, {"step": "first"})
    second_run = await journal.start_run("second task")
    second_turn = await journal.start_turn(second_run, 0)
    await journal.checkpoint(second_turn, {"step": "second"})
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "targeted"}}])
    session = AgentSession(
        profile=AgentProfile(name="targeted-resume"),
        provider=provider,
        tools=MockToolRuntime(),
        harness=journal,
    )

    outcome = await AgentRunner(session).resume(
        AgentResumeRequest(run_id=first_run.run_id, task="continue first")
    )

    assert outcome.result.output == "targeted"
    assert outcome.resume_manifest["checkpoint_id"] == first_checkpoint.checkpoint_id
    assert outcome.resume_manifest["state"]["step"] == "first"


@pytest.mark.asyncio
async def test_agent_runner_passes_approval_resume_context_to_executor_and_prompt() -> None:
    registry = ToolRegistry()

    async def deploy(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(call_id=invocation.call_id, tool_name=invocation.tool_name, content="deployed")

    registry.register(ToolSpec(name="deploy", tags=("release",)), deploy)
    approval_store = InMemoryApprovalStore()
    approval = await approval_store.submit(
        ApprovalRequest(
            reason="deployment requires approval",
            subject="tool:deploy",
            metadata={"rule": "release-approval"},
        )
    )
    resolved = await approval_store.decide(
        approval.approval_id,
        ApprovalDecisionRecord(status="approved", actor="operator"),
    )
    provider = MockLLMProvider(
        [
            {"action": "deploy", "arguments": {}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    session = AgentSession(
        profile=AgentProfile(name="approval-runner"),
        provider=provider,
        tools=registry,
        approval_store=approval_store,
        policy=RuleBasedPolicy(
            [
                PolicyRule(
                    name="release-approval",
                    status="approval_required",
                    tool_names=("deploy",),
                    reason="deployment requires approval",
                )
            ]
        ),
    )

    outcome = await AgentRunner(session).run(
        AgentRunRequest(
            task="deploy",
            approval_resume=ApprovalResumeContext.from_records((resolved,)),
        )
    )

    prompt_text = provider.requests[0].messages[0].content
    injection = outcome.prompt_manifest["metadata"]["context_injections"][0]

    assert outcome.result.status == "completed"
    assert outcome.result.output == "done"
    assert provider.requests[1].messages[-1].content == "deployed"
    assert "== Approval Resume ==" in prompt_text
    assert "tool:deploy" in prompt_text
    assert outcome.prompt_manifest["metadata"]["approval_resume"]["approved_count"] == 1
    assert injection["name"] == "approval_resume"
    assert injection["source"] == "approval"


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
async def test_agent_session_manager_persists_run_state_to_store() -> None:
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}])
    store = InMemoryAgentRunStore()
    manager = AgentSessionManager(run_store=store)
    manager.register(
        AgentSession(
            profile=AgentProfile(name="stored"),
            provider=provider,
            tools=MockToolRuntime(),
        )
    )

    outcome = await manager.run("stored", AgentRunRequest(task="task", metadata={"request_id": "r1"}))
    run = manager.runs()[0]
    stored = store.get(run.run_key)
    restored = AgentSessionManager(run_store=store)

    assert outcome.result.status == "completed"
    assert stored is not None
    assert stored.status == "completed"
    assert stored.result_run_id == outcome.result.run_id
    assert restored.run_state(run.run_key).status == "completed"
    assert manager.manifest()["run_store"]["schema_version"] == "agent-core-in-memory-run-store/v1"


@pytest.mark.asyncio
async def test_sqlite_agent_run_store_persists_manager_runs_across_instances(tmp_path) -> None:
    path = tmp_path / "runs.sqlite"
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}])
    manager = AgentSessionManager(run_store=SQLiteAgentRunStore(path))
    manager.register(
        AgentSession(
            profile=AgentProfile(name="sqlite-runs"),
            provider=provider,
            tools=MockToolRuntime(),
        )
    )

    outcome = await manager.run("sqlite-runs", "task")
    restored = AgentSessionManager(run_store=SQLiteAgentRunStore(path))
    run = restored.runs()[0]

    assert restored.manifest()["run_store"]["schema_version"] == "agent-core-sqlite-run-store/v1"
    assert run.status == "completed"
    assert run.result_run_id == outcome.result.run_id


def test_markdown_agent_run_store_marks_restored_active_runs_interrupted(tmp_path) -> None:
    path = tmp_path / "runs.md"
    store = MarkdownAgentRunStore(path)
    store.save(
        ManagedAgentRun(
            run_key="run-1",
            session_name="session",
            task="long task",
            status="running",
            metadata={"request_id": "r1"},
        )
    )

    restored = AgentSessionManager(run_store=MarkdownAgentRunStore(path))
    run = restored.run_state("run-1")
    text = path.read_text(encoding="utf-8")

    assert run.status == "interrupted"
    assert "active when manager state was restored" in run.error
    assert run.metadata["restored"] is True
    assert run.metadata["restored_from_status"] == "running"
    assert run.metadata["restore_action"] == "marked_interrupted"
    assert "long task" not in text
    assert "<!-- agent-core-run " in text


def test_agent_run_store_marks_restored_queued_capacity_run_interrupted_with_reason(tmp_path) -> None:
    path = tmp_path / "runs.md"
    store = MarkdownAgentRunStore(path)
    store.save(
        ManagedAgentRun(
            run_key="queued-1",
            session_name="session",
            task="queued task",
            status="queued",
            metadata={"queued_for_capacity": True},
        )
    )

    restored = AgentSessionManager(run_store=MarkdownAgentRunStore(path))
    run = restored.run_state("queued-1")
    snapshot = restored.schedule_snapshot().manifest()

    assert run.status == "interrupted"
    assert "queued run was restored without an executable request" in run.error
    assert run.metadata["restored"] is True
    assert run.metadata["restored_from_status"] == "queued"
    assert run.metadata["restored_queued_for_capacity"] is True
    assert snapshot["queued_run_count"] == 0
    assert snapshot["status_counts"]["interrupted"] == 1


def test_agent_session_manager_schedule_snapshot_restores_interrupted_runs_without_capacity_claim(
    tmp_path,
) -> None:
    path = tmp_path / "runs.md"
    store = MarkdownAgentRunStore(path)
    store.save(
        ManagedAgentRun(
            run_key="run-1",
            session_name="session",
            task="long task",
            status="running",
        )
    )
    manager = AgentSessionManager(
        run_store=MarkdownAgentRunStore(path),
        concurrency_policy=AgentManagerConcurrencyPolicy(max_active_runs=1),
    )
    manager.register(
        AgentSession(
            profile=AgentProfile(name="session"),
            provider=MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}]),
            tools=MockToolRuntime(),
        )
    )

    snapshot = manager.schedule_snapshot().manifest()
    capacity = manager.capacity_status("session").manifest()

    assert snapshot["schema_version"] == "agent-core-manager-schedule-snapshot/v1"
    assert snapshot["status_counts"]["interrupted"] == 1
    assert snapshot["active_run_count"] == 0
    assert capacity["available"] is True
    assert capacity["manager_active_run_count"] == 0


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
async def test_agent_session_manager_resumes_from_latest_checkpoint() -> None:
    journal = InMemoryAgentJournal()
    original_run = await journal.start_run("managed original")
    original_turn = await journal.start_turn(original_run, 0)
    checkpoint = await journal.checkpoint(original_turn, {"step": "manager"})
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "resumed"}}])
    manager = AgentSessionManager()
    manager.register(
        AgentSession(
            profile=AgentProfile(name="managed-auto-resume"),
            provider=provider,
            tools=MockToolRuntime(),
            harness=journal,
        )
    )

    plan = manager.resume_plan("managed-auto-resume", "continue managed")
    outcome = await manager.resume("managed-auto-resume", "continue managed")
    run = manager.runs()[0]

    assert plan.ready is True
    assert plan.summary_manifest()["checkpoint_id"] == checkpoint.checkpoint_id
    assert outcome.result.status == "completed"
    assert outcome.resume_manifest["checkpoint_id"] == checkpoint.checkpoint_id
    assert outcome.resume_plan_manifest["checkpoint_id"] == checkpoint.checkpoint_id
    assert run.metadata["resume"]["checkpoint_id"] == checkpoint.checkpoint_id
    assert run.metadata["resume_plan"]["checkpoint_id"] == checkpoint.checkpoint_id
    assert run.metadata["resume"]["auto_selected"] is True


@pytest.mark.asyncio
async def test_agent_session_manager_can_reject_terminal_resume_candidates() -> None:
    journal = InMemoryAgentJournal()
    original_run = await journal.start_run("managed original")
    original_turn = await journal.start_turn(original_run, 0)
    await journal.checkpoint(original_turn, {"step": "done"})
    await journal.finish_run(original_run, "completed", {"output": "done"})
    manager = AgentSessionManager()
    manager.register(
        AgentSession(
            profile=AgentProfile(name="managed-strict-resume"),
            provider=MockLLMProvider([{"action": "finish", "arguments": {"output": "resumed"}}]),
            tools=MockToolRuntime(),
            harness=journal,
        )
    )
    request = AgentResumeRequest(
        run_id=original_run.run_id,
        task="continue managed",
        allow_terminal=False,
    )

    plan = manager.resume_plan("managed-strict-resume", request)

    assert plan.ready is False
    assert plan.summary_manifest()["issue_codes"] == ["terminal_checkpoint_not_allowed"]
    with pytest.raises(ResumeError, match="resume plan is not ready"):
        await manager.resume("managed-strict-resume", request)


@pytest.mark.asyncio
async def test_agent_session_manager_start_resume_runs_in_background() -> None:
    journal = InMemoryAgentJournal()
    original_run = await journal.start_run("background original")
    original_turn = await journal.start_turn(original_run, 0)
    checkpoint = await journal.checkpoint(original_turn, {"step": "background"})
    provider = _BlockingProvider()
    manager = AgentSessionManager()
    manager.register(
        AgentSession(
            profile=AgentProfile(name="background-resume"),
            provider=provider,
            tools=MockToolRuntime(),
            harness=journal,
        )
    )

    run_key = manager.start_resume(
        "background-resume",
        AgentResumeRequest(run_id=original_run.run_id, task="continue background"),
    )
    await asyncio.sleep(0)

    assert manager.run_state(run_key).status == "running"
    assert manager.run_state(run_key).metadata["resume"]["checkpoint_id"] == checkpoint.checkpoint_id
    provider.release.set()
    outcome = await manager.wait(run_key)

    assert outcome.result.status == "completed"
    assert outcome.resume_manifest["state"]["step"] == "background"


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

    with pytest.raises(AgentManagerCapacityError) as exc_info:
        manager.start("slow", "second task")

    assert exc_info.value.metadata["max_active_runs_per_session"] == 1
    assert exc_info.value.metadata["capacity_status"]["reason"] == "session_capacity_exceeded"
    assert manager.run_state(run_key).status == "running"
    assert manager.capacity_status("slow").available is False
    provider.release.set()
    outcome = await manager.wait(run_key)

    assert outcome.result.status == "completed"
    assert not manager.active_runs()
    assert manager.capacity_status("slow").available is True


@pytest.mark.asyncio
async def test_agent_session_manager_queues_when_capacity_full_and_rejection_disabled() -> None:
    first_provider = _BlockingProvider()
    second_provider = MockLLMProvider(
        [{"action": "finish", "arguments": {"output": "queued done"}}]
    )
    session = AgentSession(
        profile=AgentProfile(name="queued"),
        provider=first_provider,
        tools=MockToolRuntime(),
    )
    manager = AgentSessionManager(
        concurrency_policy=AgentManagerConcurrencyPolicy(reject_when_full=False)
    )
    manager.register(session)

    first_key = manager.start("queued", "first task")
    await asyncio.sleep(0)
    session.provider = second_provider
    second_key = manager.start("queued", "second task")
    await asyncio.sleep(0.02)
    snapshot = manager.schedule_snapshot().manifest()
    manifest = manager.manifest()

    assert manager.run_state(first_key).status == "running"
    assert manager.run_state(second_key).status == "queued"
    assert manager.run_state(second_key).metadata["queued_for_capacity"] is True
    assert snapshot["queued_by_session"]["queued"] == [second_key]
    assert snapshot["queued_run_count"] == 1
    assert snapshot["active_run_count"] == 1
    assert manifest["queued_by_session"]["queued"] == [second_key]
    assert manager.capacity_status("queued").available is False
    assert not second_provider.requests

    first_provider.release.set()
    first = await manager.wait(first_key)
    second = await manager.wait(second_key)

    assert first.result.status == "completed"
    assert second.result.status == "completed"
    assert second.result.output == "queued done"
    assert manager.run_state(second_key).metadata["queued_for_capacity"] is False
    assert "dequeued_capacity_status" in manager.run_state(second_key).metadata
    assert not manager.active_runs()
    assert manager.schedule_snapshot().manifest()["queued_run_count"] == 0


@pytest.mark.asyncio
async def test_agent_session_manager_cancels_unclaimed_queued_run_without_touching_active_session() -> None:
    first_provider = _BlockingProvider()
    second_provider = MockLLMProvider(
        [{"action": "finish", "arguments": {"output": "should not run"}}]
    )
    session = AgentSession(
        profile=AgentProfile(name="queued-cancel"),
        provider=first_provider,
        tools=MockToolRuntime(),
    )
    manager = AgentSessionManager(
        concurrency_policy=AgentManagerConcurrencyPolicy(reject_when_full=False)
    )
    manager.register(session)

    first_key = manager.start("queued-cancel", "first task")
    await asyncio.sleep(0)
    session.provider = second_provider
    second_key = manager.start("queued-cancel", "second task")
    await asyncio.sleep(0)

    assert manager.cancel(second_key, "drop queued") is True
    cancelled = await manager.wait(second_key)
    assert cancelled.result.status == "cancelled"
    assert cancelled.result.metadata["interrupt"]["queued"] is True
    assert manager.run_state(second_key).status == "cancelled"
    assert manager.schedule_snapshot().manifest()["queued_run_count"] == 0
    assert not second_provider.requests

    first_provider.release.set()
    first = await manager.wait(first_key)

    assert first.result.status == "completed"
    assert first.result.metadata.get("interrupt") is None
    assert not second_provider.requests
    assert not manager.active_runs()


@pytest.mark.asyncio
async def test_agent_session_manager_allows_configured_session_concurrency() -> None:
    first_provider = _BlockingProvider()
    second_provider = _BlockingProvider()
    session = AgentSession(
        profile=AgentProfile(name="parallel"),
        provider=first_provider,
        tools=MockToolRuntime(),
    )
    manager = AgentSessionManager(
        concurrency_policy=AgentManagerConcurrencyPolicy(max_active_runs_per_session=2)
    )
    manager.register(session)

    first_key = manager.start("parallel", "first task")
    await asyncio.sleep(0)
    session.provider = second_provider
    second_key = manager.start("parallel", "second task")
    await asyncio.sleep(0)

    manifest = manager.manifest()
    snapshot = manager.schedule_snapshot().manifest()

    assert set(manifest["active_by_session"]["parallel"]) == {first_key, second_key}
    assert set(manifest["schedule"]["active_by_session"]["parallel"]) == {first_key, second_key}
    assert manifest["concurrency_policy"]["max_active_runs_per_session"] == 2
    assert snapshot["capacity_by_session"][0]["active_run_count"] == 2
    assert snapshot["capacity_by_session"][0]["available"] is False

    first_provider.release.set()
    second_provider.release.set()
    first = await manager.wait(first_key)
    second = await manager.wait(second_key)

    assert first.result.status == "completed"
    assert second.result.status == "completed"
    assert not manager.active_runs()


@pytest.mark.asyncio
async def test_agent_session_manager_enforces_global_active_capacity() -> None:
    first_provider = _BlockingProvider()
    second_provider = _BlockingProvider()
    manager = AgentSessionManager(
        concurrency_policy=AgentManagerConcurrencyPolicy(max_active_runs=1)
    )
    manager.register(
        AgentSession(
            profile=AgentProfile(name="first"),
            provider=first_provider,
            tools=MockToolRuntime(),
        )
    )
    manager.register(
        AgentSession(
            profile=AgentProfile(name="second"),
            provider=second_provider,
            tools=MockToolRuntime(),
        )
    )

    run_key = manager.start("first", "task")
    await asyncio.sleep(0)

    with pytest.raises(AgentManagerCapacityError) as exc_info:
        manager.start("second", "blocked")

    assert exc_info.value.metadata["max_active_runs"] == 1
    assert exc_info.value.metadata["capacity_status"]["reason"] == "manager_capacity_exceeded"
    assert manager.manifest()["active_run_count"] == 1
    assert manager.capacity_status("second").manager_available is False

    first_provider.release.set()
    outcome = await manager.wait(run_key)

    assert outcome.result.status == "completed"
    assert not second_provider.requests


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


@pytest.mark.asyncio
async def test_agent_runner_passes_request_timeout_to_react_executor() -> None:
    provider = _BlockingProvider()
    journal = InMemoryAgentJournal()
    session = AgentSession(
        profile=AgentProfile(name="timeout-runner"),
        provider=provider,
        tools=MockToolRuntime(),
        harness=journal,
    )

    outcome = await AgentRunner(session).run(
        AgentRunRequest(task="slow task", timeout_seconds=0.01)
    )

    assert outcome.result.status == "timeout"
    assert outcome.result.metadata["interrupt"]["kind"] == "timeout"
    assert outcome.prompt_manifest["metadata"]["timeout_seconds"] == 0.01
    assert outcome.trace_manifest["run"]["status"] == "timeout"
    assert journal.finished[-1]["status"] == "timeout"


@pytest.mark.asyncio
async def test_agent_session_manager_marks_timeout_run_terminal() -> None:
    provider = _BlockingProvider()
    manager = AgentSessionManager()
    manager.register(
        AgentSession(
            profile=AgentProfile(name="managed-timeout"),
            provider=provider,
            tools=MockToolRuntime(),
        )
    )

    outcome = await manager.run(
        "managed-timeout",
        AgentRunRequest(task="slow task", timeout_seconds=0.01),
    )
    run = manager.runs()[0]

    assert outcome.result.status == "timeout"
    assert run.status == "timeout"
    assert manager.cancel(run.run_key) is False

