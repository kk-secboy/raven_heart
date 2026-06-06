from __future__ import annotations

import pytest

from agent_core.config import AgentProfile, RuntimeBudget
from agent_core.context import ContextMaterial, InMemoryContextMaterialStore
from agent_core.mcp import (
    MCPCenter,
    MCPPromptContent,
    MCPPromptSpec,
    MCPResourceContent,
    MCPResourceSpec,
    MCPServerSpec,
    MCPToolReference,
    MCPToolSpec,
)
from agent_core.memory import InMemoryMemoryStore
from agent_core.providers import LLMRequest
from agent_core.runner import AgentRunner, AgentRunRequest, AgentSession
from agent_core.skills import SkillRegistry, SkillSpec, SkillsContext
from agent_core.testing import InMemoryHarness, MockLLMProvider, MockToolRuntime
from agent_core.timeline import TimelineStore
from agent_core.tools import ToolResult
from agent_core.turn_runtime import (
    DeterministicPerceptionEvaluator,
    PerceptionRuntimeConfig,
    YaklangStylePerceptionController,
)


def _prompt_text(request: LLMRequest) -> str:
    return "\n\n".join(
        message.content
        for message in request.messages
        if message.metadata.get("agent_core_prompt")
    )


def _request_text(request: LLMRequest) -> str:
    return "\n\n".join(message.content for message in request.messages)


def _prompt_segments(request: LLMRequest) -> tuple[str, ...]:
    return tuple(
        str(message.metadata.get("agent_core_prompt_segment") or "")
        for message in request.messages
        if message.metadata.get("agent_core_prompt")
    )


def _segment_text(request: LLMRequest, segment: str) -> str:
    return "\n\n".join(
        message.content
        for message in request.messages
        if message.metadata.get("agent_core_prompt_segment") == segment
    )


def _frozen_capability_prefix(request: LLMRequest) -> str:
    frozen = _segment_text(request, "frozen")
    prefix = frozen.split("\n\n[timeline_frozen]\n", 1)[0]
    return prefix.replace("</prompt_materials>", "").strip()


def _request_bytes(request: LLMRequest) -> int:
    return sum(len(message.content.encode("utf-8")) for message in request.messages)


def _compact_delta_bytes(request: LLMRequest) -> int:
    return sum(
        len(message.content.encode("utf-8"))
        for message in request.messages
        if not message.metadata.get("agent_core_prompt")
    )


def _kv_cache_prefix_text(request: LLMRequest) -> str:
    """Return the byte-stable provider prefix that can benefit from KV cache."""

    parts: list[str] = []
    for message in request.messages:
        if not message.metadata.get("agent_core_prompt"):
            break
        segment = str(message.metadata.get("agent_core_prompt_segment") or "")
        if segment == "high_static":
            parts.append(f"{message.role}\n{message.content}")
            continue
        if segment == "frozen":
            parts.append(f"{message.role}\n{message.content}")
            continue
        if segment == "semi_dynamic_1":
            parts.append(f"{message.role}\n{message.content}")
            continue
        if segment == "semi_dynamic_2":
            parts.append(f"{message.role}\n{message.content}")
            continue
        break
    return "\n\n".join(part for part in parts if part)


def _provider_wire_text(request: LLMRequest) -> str:
    return "\n\n".join(f"{message.role}\n{message.content}" for message in request.messages)


def _longest_common_prefix_bytes(values: tuple[str, ...]) -> int:
    if not values:
        return 0
    prefix = values[0]
    for value in values[1:]:
        size = 0
        for left, right in zip(prefix, value, strict=False):
            if left != right:
                break
            size += 1
        prefix = prefix[:size]
        if not prefix:
            break
    return len(prefix.encode("utf-8"))


class _LifecycleMCPConnector:
    async def list_tools(self, server: MCPServerSpec) -> tuple[MCPToolSpec, ...]:
        return (
            MCPToolSpec(
                reference=MCPToolReference(
                    server_name=server.name,
                    tool_name="read_alpha_design",
                ),
                description="Read alpha callback design notes",
                tags=("alpha", "docs"),
            ),
        )

    async def invoke_tool(
        self,
        server: MCPServerSpec,
        tool_name: str,
        arguments: dict,
    ) -> ToolResult:
        return ToolResult(
            call_id="mcp-alpha",
            tool_name=tool_name,
            content="mcp alpha design says callback-state nonce must be replay checked",
        )

    async def list_resources(self, server: MCPServerSpec) -> tuple[MCPResourceSpec, ...]:
        return (
            MCPResourceSpec(
                server_name=server.name,
                uri="memory://alpha-callback-design",
                name="Alpha callback design",
                description="alpha callback-state nonce and replay validation design",
                mime_type="text/markdown",
                tags=("alpha", "callback", "docs"),
            ),
        )

    async def read_resource(self, server: MCPServerSpec, uri: str) -> MCPResourceContent:
        return MCPResourceContent(
            server_name=server.name,
            uri=uri,
            text="MCP doc: alpha callback-state nonce must be stored and replay checked before token exchange.",
            mime_type="text/markdown",
        )

    async def list_prompts(self, server: MCPServerSpec) -> tuple[MCPPromptSpec, ...]:
        return (
            MCPPromptSpec(
                server_name=server.name,
                name="alpha_review_prompt",
                description="alpha callback review prompt",
                tags=("alpha", "review"),
            ),
        )

    async def get_prompt(
        self,
        server: MCPServerSpec,
        name: str,
        arguments: dict,
    ) -> MCPPromptContent:
        return MCPPromptContent(
            server_name=server.name,
            name=name,
            messages=(
                {
                    "role": "user",
                    "content": "When reviewing alpha callback, check nonce persistence and replay windows.",
                },
            ),
        )


@pytest.mark.asyncio
async def test_load_capability_changes_next_turn_capability_surface() -> None:
    provider = MockLLMProvider(
        [
            {"action": "load_capability", "arguments": {"name": "lookup_alpha"}},
            {"action": "finish", "arguments": {"output": "loaded"}},
        ]
    )
    tools = MockToolRuntime({"lookup_alpha": "alpha lookup result"})
    timeline = TimelineStore()
    session = AgentSession(
        profile=AgentProfile(
            name="capability-load-loop",
            instructions="Use loaded capabilities when they match the task.",
            budget=RuntimeBudget(max_iterations=3, max_prompt_bytes=9000),
        ),
        provider=provider,
        tools=tools,
        harness=InMemoryHarness(),
        memory=InMemoryMemoryStore(),
        timeline=timeline,
        perception_controller=YaklangStylePerceptionController(
            evaluator=DeterministicPerceptionEvaluator(),
            config=PerceptionRuntimeConfig(sync_triggers=True),
        ),
    )

    outcome = await AgentRunner(session).run("Alpha task: load lookup_alpha and continue")

    assert outcome.result.status == "completed"
    assert len(provider.requests) == 2
    second_prompt = _prompt_text(provider.requests[1])
    assert "[capability_recall]" in second_prompt
    assert "[recent_tools_cache]" in second_prompt
    assert "lookup_alpha" in second_prompt
    assert "load_capability" in "\n".join(item.content for item in timeline.items)


@pytest.mark.asyncio
async def test_yaklang_style_functional_topic_switch_recall_and_prompt_blocks() -> None:
    provider = MockLLMProvider(
        [
            {"action": "call_tool", "arguments": {"tool_name": "lookup_alpha"}},
            {"action": "finish", "arguments": {"output": "alpha done"}},
            {"action": "call_tool", "arguments": {"tool_name": "lookup_beta"}},
            {"action": "finish", "arguments": {"output": "beta done"}},
            {"action": "finish", "arguments": {"output": "alpha resumed"}},
        ]
    )
    tools = MockToolRuntime(
        {
            "lookup_alpha": (
                "ALPHA_FACT callback-state nonce evidence: nonce alpha-42 must be "
                "persisted before redirect and replay checked on return."
            ),
            "lookup_beta": (
                "BETA_FACT billing rate-limit evidence: tenant burst budget is "
                "30 requests per minute and retries need jitter."
            ),
        }
    )
    skills = SkillRegistry()
    skills.register(
        SkillSpec(
            name="alpha-callback-review",
            description="Review alpha callback-state nonce handling",
            prompt="Check callback-state nonce persistence, replay checks, and token exchange ordering.",
            tags=("alpha", "callback"),
            priority=10,
        )
    )
    skills.register(
        SkillSpec(
            name="beta-billing-review",
            description="Review beta billing rate limit behavior",
            prompt="Check burst budget, retry jitter, and per-tenant limit isolation.",
            tags=("beta", "billing"),
            priority=8,
        )
    )
    mcp = MCPCenter()
    mcp.register_server(MCPServerSpec(name="docs", transport="memory", tags=("alpha", "docs")))
    mcp.register_connector("memory", _LifecycleMCPConnector())
    context_store = InMemoryContextMaterialStore(
        (
            ContextMaterial(
                name="alpha-kb-callback-state",
                content=(
                    "Knowledge doc: alpha callback-state uses nonce alpha-42; "
                    "the verifier must persist nonce before redirect and reject replay."
                ),
                role="knowledge",
                priority=90,
                metadata={"tags": ("alpha", "callback"), "source": "embedded-doc"},
            ),
            ContextMaterial(
                name="beta-kb-rate-limit",
                content=(
                    "Knowledge doc: beta billing rate limit allows 30 requests per minute "
                    "and requires retry jitter for 429 responses."
                ),
                role="knowledge",
                priority=80,
                metadata={"tags": ("beta", "billing"), "source": "embedded-doc"},
            ),
        )
    )
    memory = InMemoryMemoryStore()
    timeline = TimelineStore()
    harness = InMemoryHarness()
    session = AgentSession(
        profile=AgentProfile(
            name="yaklang-functional",
            instructions="You are a loop agent. Use timeline, memory, skills, MCP, and knowledge context.",
            budget=RuntimeBudget(max_iterations=3, max_prompt_bytes=12000, max_timeline_bytes=4096),
        ),
        provider=provider,
        tools=tools,
        harness=harness,
        skills=SkillsContext(skills),
        mcp=mcp,
        memory=memory,
        timeline=timeline,
        context_material_store=context_store,
        perception_controller=YaklangStylePerceptionController(
            evaluator=DeterministicPerceptionEvaluator(),
            config=PerceptionRuntimeConfig(sync_triggers=True),
        ),
    )
    runner = AgentRunner(session)

    alpha = await runner.run(
        AgentRunRequest(
            task="Alpha task: inspect callback-state nonce handling",
            refresh=True,
        )
    )
    beta = await runner.run("Beta task: inspect billing rate-limit behavior")
    alpha_return = await runner.run("Alpha task again: continue callback-state nonce analysis")

    assert alpha.result.status == "completed"
    assert beta.result.status == "completed"
    assert alpha_return.result.status == "completed"
    assert [call.tool_name for call in tools.invocations] == ["lookup_alpha", "lookup_beta"]
    assert len(provider.requests) == 5

    alpha_second_prompt = _prompt_text(provider.requests[1])
    beta_first_prompt = _prompt_text(provider.requests[2])
    alpha_return_prompt = _prompt_text(provider.requests[4])

    assert "ALPHA_FACT callback-state nonce evidence" in alpha_second_prompt
    assert "[knowledge_recall]" in alpha_second_prompt
    assert "alpha-kb-callback-state" in alpha_second_prompt
    assert "mcp_resource:memory://alpha-callback-design" in alpha_second_prompt
    assert "alpha-callback-review" in alpha_second_prompt

    assert "ALPHA_FACT callback-state nonce evidence" in beta_first_prompt
    assert "Beta task: inspect billing rate-limit behavior" in beta_first_prompt
    assert "BETA_FACT billing rate-limit evidence" not in beta_first_prompt

    assert "[memory]" in alpha_return_prompt
    assert "ALPHA_FACT callback-state nonce evidence" in alpha_return_prompt
    assert "callback-state nonce" in alpha_return_prompt
    assert "[midterm_timeline_recall]" not in _prompt_text(provider.requests[0])

    for request in provider.requests:
        segments = _prompt_segments(request)
        assert "high_static" in segments
        assert "frozen" in segments
        assert "timeline_open_dynamic" in segments
        assert sum(
            len(message.content.encode("utf-8"))
            for message in request.messages
        ) <= 12000

    high_static_hashes = {
        _segment_text(request, "high_static")
        for request in provider.requests
    }
    assert len(high_static_hashes) == 1

    timeline_text = "\n".join(item.content for item in timeline.items if not item.deleted)
    assert "ALPHA_FACT callback-state nonce evidence" in timeline_text
    assert "BETA_FACT billing rate-limit evidence" in timeline_text
    assert any("ALPHA_FACT callback-state nonce evidence" in record.content for record in memory.records)


@pytest.mark.asyncio
async def test_yaklang_style_long_loop_keeps_prompt_blocks_stable_and_recalls_old_topic() -> None:
    provider = MockLLMProvider(
        [
            {"action": "call_tool", "arguments": {"tool_name": "lookup_alpha_plan"}},
            {"action": "finish", "arguments": {"output": "alpha plan done"}},
            {"action": "call_tool", "arguments": {"tool_name": "lookup_alpha_replay"}},
            {"action": "finish", "arguments": {"output": "alpha replay done"}},
            {"action": "call_tool", "arguments": {"tool_name": "lookup_alpha_token"}},
            {"action": "finish", "arguments": {"output": "alpha token done"}},
            {"action": "call_tool", "arguments": {"tool_name": "lookup_beta"}},
            {"action": "finish", "arguments": {"output": "beta done"}},
            {"action": "finish", "arguments": {"output": "alpha resumed from memory"}},
        ]
    )
    tools = MockToolRuntime(
        {
            "lookup_alpha_plan": (
                "ALPHA_FACT plan: callback-state nonce alpha-42 is generated and stored "
                "before redirect."
            ),
            "lookup_alpha_replay": (
                "ALPHA_FACT replay: callback handler rejects alpha-42 nonce replay "
                "before token exchange."
            ),
            "lookup_alpha_token": (
                "ALPHA_FACT token: token exchange is blocked until callback-state nonce "
                "alpha-42 passes replay validation."
            ),
            "lookup_beta": (
                "BETA_FACT billing: tenant webhook retry budget is 30 RPM and 429 "
                "responses need jitter plus idempotency keys."
            ),
        }
    )
    skills = SkillRegistry()
    skills.register(
        SkillSpec(
            name="alpha-callback-review",
            description="Review alpha callback-state nonce handling",
            prompt="Track alpha nonce generation, persistence, replay rejection, and token exchange order.",
            tags=("alpha", "callback"),
            priority=10,
        )
    )
    skills.register(
        SkillSpec(
            name="beta-billing-review",
            description="Review beta billing retry behavior",
            prompt="Track tenant budgets, 429 retry jitter, and idempotency.",
            tags=("beta", "billing"),
            priority=8,
        )
    )
    mcp = MCPCenter()
    mcp.register_server(MCPServerSpec(name="docs", transport="memory", tags=("alpha", "docs")))
    mcp.register_connector("memory", _LifecycleMCPConnector())
    context_store = InMemoryContextMaterialStore(
        (
            ContextMaterial(
                name="alpha-kb-long-callback",
                content=(
                    "Knowledge doc: alpha callback-state nonce alpha-42 is persisted "
                    "before redirect and replay checked before token exchange."
                ),
                role="knowledge",
                priority=90,
                metadata={"tags": ("alpha", "callback"), "source": "embedded-doc"},
            ),
            ContextMaterial(
                name="beta-kb-long-billing",
                content=(
                    "Knowledge doc: beta webhook delivery uses a 30 RPM tenant budget, "
                    "jittered retry, and idempotency keys."
                ),
                role="knowledge",
                priority=80,
                metadata={"tags": ("beta", "billing"), "source": "embedded-doc"},
            ),
            ContextMaterial(
                name="noise-kb-cache",
                content="Unrelated CDN cache eviction policy for image thumbnails.",
                role="knowledge",
                priority=1,
                metadata={"tags": ("cache",), "source": "embedded-doc"},
            ),
        )
    )
    memory = InMemoryMemoryStore()
    timeline = TimelineStore()
    session = AgentSession(
        profile=AgentProfile(
            name="yaklang-long-loop",
            instructions="You are a long-running loop agent. Keep stable instructions stable.",
            budget=RuntimeBudget(
                max_iterations=3,
                max_prompt_bytes=14000,
                max_timeline_bytes=1800,
            ),
        ),
        provider=provider,
        tools=tools,
        harness=InMemoryHarness(),
        skills=SkillsContext(skills),
        mcp=mcp,
        memory=memory,
        timeline=timeline,
        context_material_store=context_store,
        perception_controller=YaklangStylePerceptionController(
            evaluator=DeterministicPerceptionEvaluator(),
            config=PerceptionRuntimeConfig(sync_triggers=True),
        ),
    )
    runner = AgentRunner(session)

    outcomes = [
        await runner.run(AgentRunRequest(task="Alpha long task step 1: map nonce storage", refresh=True)),
        await runner.run("Alpha long task step 2: verify replay rejection"),
        await runner.run("Alpha long task step 3: verify token exchange ordering"),
        await runner.run("Beta interruption: inspect billing webhook retry"),
        await runner.run("Alpha return: continue callback-state nonce analysis from memory"),
    ]

    assert [outcome.result.status for outcome in outcomes] == ["completed"] * 5
    assert [call.tool_name for call in tools.invocations] == [
        "lookup_alpha_plan",
        "lookup_alpha_replay",
        "lookup_alpha_token",
        "lookup_beta",
    ]
    assert outcomes[-1].result.output == "alpha resumed from memory"
    assert len(provider.requests) == 9

    prompts = [_prompt_text(request) for request in provider.requests]
    alpha_return_prompt = prompts[-1]
    beta_after_tool_prompt = prompts[7]

    assert "[memory]" in alpha_return_prompt
    assert "ALPHA_FACT" in alpha_return_prompt
    assert "callback-state nonce" in alpha_return_prompt
    assert "BETA_FACT billing" in _request_text(provider.requests[7])
    assert (
        "[knowledge_recall]" in beta_after_tool_prompt
        or "[midterm_timeline_recall]" in beta_after_tool_prompt
    )
    assert "mcp_resource:memory://alpha-callback-design" in "\n".join(prompts)
    assert "alpha-callback-review" in "\n".join(prompts)

    high_static_hashes = {_segment_text(request, "high_static") for request in provider.requests}
    frozen_capability_prefixes = {
        _frozen_capability_prefix(request) for request in provider.requests
    }
    frozen_segments = {_segment_text(request, "frozen") for request in provider.requests}
    semi_dynamic_1_hashes = {_segment_text(request, "semi_dynamic_1") for request in provider.requests}
    semi_dynamic_2_hashes = {_segment_text(request, "semi_dynamic_2") for request in provider.requests}
    timeline_open_hashes = {
        _segment_text(request, "timeline_open_dynamic") for request in provider.requests
    }
    assert len(high_static_hashes) == 1
    assert len(frozen_capability_prefixes) == 1
    assert len(frozen_segments) == 1
    assert 1 <= len(semi_dynamic_2_hashes) <= 2
    assert len(semi_dynamic_1_hashes) > 1
    assert len(timeline_open_hashes) > 1

    kv_prefixes = tuple(_kv_cache_prefix_text(request) for request in provider.requests)
    assert all(prefix for prefix in kv_prefixes)
    assert _longest_common_prefix_bytes(kv_prefixes) > 0
    for request, prefix in zip(provider.requests, kv_prefixes, strict=True):
        assert _prompt_segments(request)[:4] == (
            "high_static",
            "frozen",
            "semi_dynamic_1",
            "semi_dynamic_2",
        )
        assert "[timeline_frozen]" not in prefix
        assert "[timeline_open]" not in prefix
        assert "[memory]" not in prefix
        assert "[knowledge_recall]" not in prefix
        assert "[midterm_timeline_recall]" not in prefix
        dynamic_injections = [
            line
            for line in prefix.splitlines()
            if line.startswith("[context_injection:")
            and not line.startswith("[context_injection:recent_tools_cache ")
        ]
        assert dynamic_injections == []
        assert "Alpha long task" not in prefix
        assert "Beta interruption" not in prefix
        assert "Alpha return" not in prefix
        assert "ALPHA_FACT" not in prefix
        assert "BETA_FACT" not in prefix

    assert max(_request_bytes(request) for request in provider.requests) <= 14000
    assert max(_compact_delta_bytes(request) for request in provider.requests) <= 1600
    assert all(len(_prompt_segments(request)) <= 5 for request in provider.requests)

    timeline_text = "\n".join(item.content for item in timeline.items if not item.deleted)
    assert "ALPHA_FACT plan" in timeline_text
    assert "ALPHA_FACT replay" in timeline_text
    assert "ALPHA_FACT token" in timeline_text
    assert "BETA_FACT billing" in timeline_text
    assert any("ALPHA_FACT token" in record.content for record in memory.records)
