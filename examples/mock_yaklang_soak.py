from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

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


def _request_bytes(request: LLMRequest) -> int:
    return sum(len(message.content.encode("utf-8")) for message in request.messages)


def _compact_delta_bytes(request: LLMRequest) -> int:
    return sum(
        len(message.content.encode("utf-8"))
        for message in request.messages
        if not message.metadata.get("agent_core_prompt")
    )


def _segment_hashes(requests: list[LLMRequest], segment_name: str) -> list[str]:
    hashes: list[str] = []
    for request in requests:
        content = "\n\n".join(
            message.content
            for message in request.messages
            if message.metadata.get("agent_core_prompt_segment") == segment_name
        )
        hashes.append(hashlib.sha256(content.encode("utf-8")).hexdigest() if content else "")
    return hashes


class _SoakMCPConnector:
    async def list_tools(self, server: MCPServerSpec) -> tuple[MCPToolSpec, ...]:
        return (
            MCPToolSpec(
                reference=MCPToolReference(server_name=server.name, tool_name="read_alpha_soak"),
                description="Read alpha soak design note",
                tags=("alpha", "callback"),
            ),
        )

    async def invoke_tool(self, server: MCPServerSpec, tool_name: str, arguments: dict) -> ToolResult:
        return ToolResult(
            call_id="mcp-soak-alpha",
            tool_name=tool_name,
            content="MCP_SOAK_ALPHA: nonce alpha-42 requires replay validation.",
        )

    async def list_resources(self, server: MCPServerSpec) -> tuple[MCPResourceSpec, ...]:
        return (
            MCPResourceSpec(
                server_name=server.name,
                uri="memory://soak/alpha-callback.md",
                name="Soak alpha callback design",
                description="Alpha nonce persistence and replay validation",
                tags=("alpha", "callback"),
            ),
        )

    async def read_resource(self, server: MCPServerSpec, uri: str) -> MCPResourceContent:
        return MCPResourceContent(
            server_name=server.name,
            uri=uri,
            text="MCP doc: alpha callback nonce alpha-42 is persisted and replay checked.",
        )

    async def list_prompts(self, server: MCPServerSpec) -> tuple[MCPPromptSpec, ...]:
        return (
            MCPPromptSpec(
                server_name=server.name,
                name="soak_alpha_review",
                description="Alpha callback review prompt",
                tags=("alpha", "callback"),
            ),
        )

    async def get_prompt(self, server: MCPServerSpec, name: str, arguments: dict) -> MCPPromptContent:
        return MCPPromptContent(
            server_name=server.name,
            name=name,
            messages=({"role": "user", "content": "Check nonce persistence and replay windows."},),
        )


async def run_soak(output: Path) -> dict[str, Any]:
    alpha_steps = 10
    responses: list[dict[str, Any]] = []
    tool_results: dict[str, str] = {}
    tasks: list[AgentRunRequest | str] = []

    for index in range(alpha_steps):
        tool_name = f"lookup_alpha_{index}"
        responses.append({"action": "call_tool", "arguments": {"tool_name": tool_name}})
        responses.append({"action": "finish", "arguments": {"output": f"alpha step {index} done"}})
        tool_results[tool_name] = (
            f"ALPHA_FACT step-{index}: callback-state nonce alpha-42 invariant {index}; "
            "persist before redirect and reject replay before token exchange."
        )
        task = f"Alpha long-running task step {index}: verify nonce invariant {index}"
        tasks.append(AgentRunRequest(task=task, refresh=True) if index == 0 else task)

    responses.extend(
        [
            {"action": "call_tool", "arguments": {"tool_name": "lookup_beta"}},
            {"action": "finish", "arguments": {"output": "beta interruption done"}},
            {"action": "finish", "arguments": {"output": "alpha return recalled from memory"}},
        ]
    )
    tool_results["lookup_beta"] = (
        "BETA_FACT soak: billing webhook retry has 30 RPM tenant budget, jitter, "
        "and idempotency keys."
    )
    tasks.append("Beta interruption: inspect billing webhook retry while alpha work is paused")
    tasks.append("Alpha return: resume nonce alpha-42 work from memory without calling tools")

    provider = MockLLMProvider(responses)
    tools = MockToolRuntime(tool_results)
    skills = SkillRegistry()
    skills.register(
        SkillSpec(
            name="alpha-callback-review",
            description="Review alpha callback-state nonce handling",
            prompt="Track nonce persistence, replay rejection, and token exchange ordering.",
            tags=("alpha", "callback"),
            priority=20,
        )
    )
    skills.register(
        SkillSpec(
            name="beta-billing-review",
            description="Review beta webhook retry behavior",
            prompt="Track rate limits, retry jitter, and idempotency.",
            tags=("beta", "billing"),
            priority=15,
        )
    )
    mcp = MCPCenter()
    mcp.register_server(MCPServerSpec(name="soak-docs", transport="memory", tags=("alpha", "docs")))
    mcp.register_connector("memory", _SoakMCPConnector())
    context_store = InMemoryContextMaterialStore(
        (
            ContextMaterial(
                name="soak-alpha-doc",
                content=(
                    "Knowledge doc: alpha callback-state nonce alpha-42 must be persisted "
                    "before redirect and replay checked before token exchange."
                ),
                role="knowledge",
                priority=90,
                metadata={"tags": ("alpha", "callback"), "source": "mock-doc"},
            ),
            ContextMaterial(
                name="soak-beta-doc",
                content=(
                    "Knowledge doc: beta billing webhooks use a 30 RPM tenant budget, "
                    "jittered retry, and idempotency keys."
                ),
                role="knowledge",
                priority=85,
                metadata={"tags": ("beta", "billing"), "source": "mock-doc"},
            ),
        )
    )
    memory = InMemoryMemoryStore()
    timeline = TimelineStore()
    session = AgentSession(
        profile=AgentProfile(
            name="mock-yaklang-soak",
            instructions="You are a long-running loop agent. Keep stable context stable.",
            budget=RuntimeBudget(
                max_iterations=3,
                max_prompt_bytes=18000,
                max_timeline_bytes=2200,
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
    await mcp.refresh(fail_fast=False)
    await mcp.refresh_resources()
    await mcp.refresh_prompts()
    runner = AgentRunner(session)
    outcomes = [await runner.run(task) for task in tasks]
    prompt_texts = [_prompt_text(request) for request in provider.requests]
    request_bytes = [_request_bytes(request) for request in provider.requests]
    compact_delta_bytes = [_compact_delta_bytes(request) for request in provider.requests]
    tail_request_bytes = request_bytes[-6:]
    tail_request_byte_range = (
        max(tail_request_bytes, default=0) - min(tail_request_bytes, default=0)
    )
    high_static_hashes = _segment_hashes(provider.requests, "high_static")
    frozen_hashes = _segment_hashes(provider.requests, "frozen")
    semi_dynamic_hashes = _segment_hashes(provider.requests, "semi_dynamic_1")
    semi_dynamic_2_hashes = _segment_hashes(provider.requests, "semi_dynamic_2")
    timeline_dynamic_hashes = _segment_hashes(provider.requests, "timeline_open_dynamic")
    report = {
        "schema_version": "raven-heart-mock-yaklang-soak/v1",
        "status": "completed",
        "task_count": len(tasks),
        "provider_request_count": len(provider.requests),
        "tool_invocation_count": len(tools.invocations),
        "outcomes": [
            {"status": outcome.result.status, "output": outcome.result.output}
            for outcome in outcomes
        ],
        "request_bytes": request_bytes,
        "compact_delta_bytes": compact_delta_bytes,
        "max_request_bytes": max(request_bytes, default=0),
        "max_compact_delta_bytes": max(compact_delta_bytes, default=0),
        "tail_request_bytes": tail_request_bytes,
        "tail_request_byte_range": tail_request_byte_range,
        "request_growth_bounded": max(request_bytes, default=0) <= 18000
        and tail_request_byte_range <= 2500,
        "compact_delta_bounded": max(compact_delta_bytes, default=0) <= 1024,
        "high_static_stable": len(set(high_static_hashes)) == 1,
        "frozen_changed": len(set(frozen_hashes)) > 1,
        "semi_dynamic_2_stable": len(set(semi_dynamic_2_hashes)) <= 1,
        "semi_dynamic_changed": len(set(semi_dynamic_hashes)) > 1,
        "timeline_open_dynamic_changed": len(set(timeline_dynamic_hashes)) > 1,
        "alpha_return_contains_memory": "[memory]" in prompt_texts[-1],
        "alpha_return_contains_alpha_fact": "ALPHA_FACT" in prompt_texts[-1],
        "beta_request_contains_beta_fact": any("BETA_FACT" in text for text in prompt_texts),
        "skill_context_visible": any("alpha-callback-review" in text for text in prompt_texts),
        "mcp_context_visible": any("mcp_resource:" in text or "[mcp_servers]" in text for text in prompt_texts),
        "timeline_item_count": len(timeline.items),
        "memory_record_count": len(memory.records),
        "memory_contains_alpha": any("ALPHA_FACT" in record.content for record in memory.records),
        "memory_contains_beta": any("BETA_FACT" in record.content for record in memory.records),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(Path(".tmp") / "mock-yaklang-soak.json"),
        help="Where to write the deterministic soak report.",
    )
    args = parser.parse_args()
    report = asyncio.run(run_soak(Path(args.output)))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
