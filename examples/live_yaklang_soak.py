from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_core.config import AgentProfile, RuntimeBudget
from agent_core.context import ContextMaterial, InMemoryContextMaterialStore
from agent_core.live_adapters import DeepSeekChatProvider, ZhipuEmbedding3Provider
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
from agent_core.providers import ChatCompletionsLLMProviderCodec, LLMRequest
from agent_core.runner import AgentRunner, AgentRunRequest, AgentSession
from agent_core.skills import SkillRegistry, SkillSpec, SkillsContext
from agent_core.timeline import TimelineStore
from agent_core.tools import ToolInvocation, ToolRegistry, ToolResult, ToolSpec
from agent_core.turn_runtime import (
    PerceptionRuntimeConfig,
    ProviderBackedPerceptionEvaluator,
    YaklangStylePerceptionController,
)


ALPHA_FACT = (
    "ALPHA_FACT: callback-state nonce alpha-42 must be persisted before redirect; "
    "callback handlers must reject replay before token exchange."
)
BETA_FACT = (
    "BETA_FACT: billing webhook retry has a 30 RPM tenant burst budget; "
    "429 handling requires jitter and idempotency keys."
)
GAMMA_FACT = (
    "GAMMA_FACT: scan orchestration must isolate tenant scope, preserve evidence ids, "
    "and retry idempotently after transient tool failure."
)


@dataclass(frozen=True)
class ScenarioTask:
    topic: str
    tool_name: str
    task: str
    expects_failure: bool = False


def _credential(name: str) -> str:
    return os.environ.get(name) or ""


def _request_text(request: LLMRequest) -> str:
    return "\n\n".join(message.content for message in request.messages)


def _prompt_text(request: LLMRequest) -> str:
    return "\n\n".join(
        message.content
        for message in request.messages
        if message.metadata.get("agent_core_prompt")
    )


def _request_bytes(request: LLMRequest) -> int:
    return sum(len(message.content.encode("utf-8")) for message in request.messages)


def _prompt_bytes(request: LLMRequest) -> int:
    return sum(
        len(message.content.encode("utf-8"))
        for message in request.messages
        if message.metadata.get("agent_core_prompt")
    )


def _compact_delta_bytes(request: LLMRequest) -> int:
    return sum(
        len(message.content.encode("utf-8"))
        for message in request.messages
        if not message.metadata.get("agent_core_prompt")
    )


def _segment_hashes(requests: list[LLMRequest], segment: str) -> list[str]:
    hashes: list[str] = []
    for request in requests:
        content = "\n\n".join(
            message.content
            for message in request.messages
            if message.metadata.get("agent_core_prompt_segment") == segment
        )
        hashes.append(hashlib.sha256(content.encode("utf-8")).hexdigest() if content else "")
    return hashes


def _frozen_capability_prefix_hashes(requests: list[LLMRequest]) -> list[str]:
    hashes: list[str] = []
    for request in requests:
        content = "\n\n".join(
            message.content
            for message in request.messages
            if message.metadata.get("agent_core_prompt_segment") == "frozen"
        )
        if not content:
            hashes.append("")
            continue
        prefix = content.split("\n\n[timeline_frozen]\n", 1)[0]
        prefix = prefix.replace("</prompt_materials>", "").strip()
        hashes.append(hashlib.sha256(prefix.encode("utf-8")).hexdigest() if prefix else "")
    return hashes


def _cache_control_count(request: LLMRequest) -> int:
    payload = ChatCompletionsLLMProviderCodec().encode_request(request)
    return json.dumps(payload, ensure_ascii=False).count('"cache_control"')


def _tool_inventory(request: LLMRequest) -> dict[str, Any]:
    return {
        "count": len(request.tools),
        "names": [tool.name for tool in request.tools[:12]],
        "bytes": len(
            json.dumps([tool.manifest() for tool in request.tools], ensure_ascii=False).encode(
                "utf-8"
            )
        ),
    }


def _usage_stats(requests: list[LLMRequest], responses: list[Any]) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    prompt_cache_hit_tokens = 0
    prompt_cache_miss_tokens = 0
    response_count = 0
    for request, response in zip(requests, responses, strict=False):
        if request.metadata.get("agent_core_perception"):
            continue
        usage = getattr(response, "usage", None)
        if usage is None:
            continue
        response_count += 1
        input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
        metadata = dict(getattr(usage, "metadata", {}) or {})
        prompt_cache_hit_tokens += _int_metadata(metadata, "prompt_cache_hit_tokens")
        prompt_cache_miss_tokens += _int_metadata(metadata, "prompt_cache_miss_tokens")
    denominator = prompt_cache_hit_tokens + prompt_cache_miss_tokens
    return {
        "response_count": response_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "prompt_cache_hit_tokens": prompt_cache_hit_tokens,
        "prompt_cache_miss_tokens": prompt_cache_miss_tokens,
        "prompt_cache_hit_ratio": (
            round(prompt_cache_hit_tokens / denominator, 6) if denominator else 0.0
        ),
        "has_prompt_cache_usage": denominator > 0,
    }


def _int_metadata(metadata: dict[str, Any], key: str) -> int:
    try:
        return int(metadata.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _build_tasks(session_index: int, task_count: int) -> list[ScenarioTask]:
    pattern = (
        ("alpha", "inspect_alpha", "Inspect callback-state nonce alpha-42."),
        ("beta", "inspect_beta", "Switch to billing webhook retry and rate-limit review."),
        ("gamma", "exec_echo", "Run the safe exec check for scan orchestration."),
        ("alpha", "unstable_probe", "Exercise failure recovery, then preserve alpha context."),
        ("beta", "timeout_probe", "Exercise timeout recovery, then preserve beta context."),
        ("alpha", "inspect_alpha", "Return to alpha after interruption and recall prior facts."),
        ("gamma", "inspect_gamma", "Review tenant-scoped scan orchestration evidence."),
    )
    tasks: list[ScenarioTask] = []
    for index in range(task_count):
        topic, tool_name, base = pattern[index % len(pattern)]
        expects_failure = tool_name in {"unstable_probe", "timeout_probe"}
        instruction = (
            f"Session {session_index} task {index}: {base} "
            f"First return JSON action call_tool with tool_name={tool_name}. "
            "After the tool result, finish with a concise finding. "
            "If the tool fails or times out, recover and finish using timeline/memory context."
        )
        tasks.append(
            ScenarioTask(
                topic=topic,
                tool_name=tool_name,
                task=instruction,
                expects_failure=expects_failure,
            )
        )
    return tasks


class _LiveSoakMCPConnector:
    def __init__(self, session_index: int) -> None:
        self.session_index = session_index

    async def list_tools(self, server: MCPServerSpec) -> tuple[MCPToolSpec, ...]:
        return tuple(
            MCPToolSpec(
                reference=MCPToolReference(
                    server_name=server.name,
                    tool_name=f"mcp_lookup_{topic}_{index}",
                ),
                description=f"MCP candidate for {topic} security context {index}",
                tags=(topic, "security", "candidate"),
            )
            for index, topic in enumerate(("alpha", "beta", "gamma") * 8)
        )

    async def invoke_tool(
        self,
        server: MCPServerSpec,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        return ToolResult(
            call_id=f"mcp-{self.session_index}-{tool_name}",
            tool_name=tool_name,
            content=f"MCP_FACT session={self.session_index} tool={tool_name}",
        )

    async def list_resources(self, server: MCPServerSpec) -> tuple[MCPResourceSpec, ...]:
        specs: list[MCPResourceSpec] = []
        for index, topic in enumerate(("alpha", "beta", "gamma") * 10):
            specs.append(
                MCPResourceSpec(
                    server_name=server.name,
                    uri=f"memory://live-soak/{self.session_index}/{topic}/{index}.md",
                    name=f"{topic} live soak resource {index}",
                    description=f"{topic} MCP resource candidate {index}",
                    tags=(topic, "resource"),
                )
            )
        return tuple(specs)

    async def read_resource(self, server: MCPServerSpec, uri: str) -> MCPResourceContent:
        if "/alpha/" in uri:
            text = "MCP_ALPHA: callback-state nonce alpha-42 replay validation."
        elif "/beta/" in uri:
            text = "MCP_BETA: billing webhook 30 RPM retry and idempotency context."
        else:
            text = "MCP_GAMMA: tenant-scoped scan orchestration context."
        return MCPResourceContent(server_name=server.name, uri=uri, text=text)

    async def list_prompts(self, server: MCPServerSpec) -> tuple[MCPPromptSpec, ...]:
        return tuple(
            MCPPromptSpec(
                server_name=server.name,
                name=f"{topic}_review_prompt",
                description=f"{topic} review prompt",
                tags=(topic, "review"),
            )
            for topic in ("alpha", "beta", "gamma")
        )

    async def get_prompt(
        self,
        server: MCPServerSpec,
        name: str,
        arguments: dict[str, Any],
    ) -> MCPPromptContent:
        return MCPPromptContent(
            server_name=server.name,
            name=name,
            messages=({"role": "user", "content": f"MCP prompt {name}: keep context scoped."},),
        )


def _build_tools() -> ToolRegistry:
    registry = ToolRegistry()

    async def inspect_alpha(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(invocation.call_id, invocation.tool_name, content=ALPHA_FACT)

    async def inspect_beta(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(invocation.call_id, invocation.tool_name, content=BETA_FACT)

    async def inspect_gamma(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(invocation.call_id, invocation.tool_name, content=GAMMA_FACT)

    async def exec_echo(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            invocation.call_id,
            invocation.tool_name,
            content="EXEC_RESULT: safe echo ok; scan orchestration command path is reachable.",
        )

    async def unstable_probe(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            invocation.call_id,
            invocation.tool_name,
            status="failed",
            error="UNSTABLE_PROBE_FAILED: simulated transient scanner failure",
        )

    async def timeout_probe(invocation: ToolInvocation) -> ToolResult:
        await asyncio.sleep(0.2)
        return ToolResult(
            invocation.call_id,
            invocation.tool_name,
            status="failed",
            error="TIMEOUT_PROBE_FAILED: simulated tool timeout after 200ms",
        )

    for name, description, handler, tags in (
        ("inspect_alpha", "Inspect alpha callback-state nonce handling.", inspect_alpha, ("alpha", "callback")),
        ("inspect_beta", "Inspect beta billing webhook retry behavior.", inspect_beta, ("beta", "billing")),
        ("inspect_gamma", "Inspect gamma scan orchestration evidence.", inspect_gamma, ("gamma", "scan")),
        ("exec_echo", "Safe exec-like command smoke tool.", exec_echo, ("exec", "gamma")),
        ("unstable_probe", "Transient failure probe for recovery testing.", unstable_probe, ("failure", "alpha")),
        ("timeout_probe", "Timeout-like probe for recovery testing.", timeout_probe, ("timeout", "beta")),
    ):
        registry.register(
            ToolSpec(
                name=name,
                description=description,
                parameters_schema={"type": "object", "properties": {}},
                tags=tags,
            ),
            handler,
        )

    async def noise_tool(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(invocation.call_id, invocation.tool_name, content="NOISE_TOOL_RESULT")

    for index in range(36):
        topic = ("alpha", "beta", "gamma", "noise")[index % 4]
        registry.register(
            ToolSpec(
                name=f"candidate_{topic}_{index}",
                description=f"Candidate {topic} tool used to stress tool inventory ranking {index}.",
                parameters_schema={"type": "object", "properties": {}},
                tags=(topic, "candidate"),
            ),
            noise_tool,
        )
    return registry


def _build_skills() -> SkillsContext:
    registry = SkillRegistry()
    for topic, prompt in (
        ("alpha", "Track nonce persistence, callback replay rejection, and token exchange order."),
        ("beta", "Track tenant rate limit, retry jitter, and idempotency keys."),
        ("gamma", "Track tenant scope, evidence ids, and orchestration recovery."),
    ):
        registry.register(
            SkillSpec(
                name=f"{topic}-live-soak-review",
                description=f"{topic} live soak review skill",
                prompt=prompt,
                tags=(topic, "live-soak"),
                priority=20,
            )
        )
    return SkillsContext(registry)


def _build_context_store(
    embedding: ZhipuEmbedding3Provider,
    *,
    session_index: int,
) -> InMemoryContextMaterialStore:
    materials: list[ContextMaterial] = []
    for index in range(54):
        topic = ("alpha", "beta", "gamma", "noise")[index % 4]
        if topic == "alpha":
            content = f"Knowledge alpha {index}: {ALPHA_FACT}"
            priority = 90
        elif topic == "beta":
            content = f"Knowledge beta {index}: {BETA_FACT}"
            priority = 85
        elif topic == "gamma":
            content = f"Knowledge gamma {index}: {GAMMA_FACT}"
            priority = 80
        else:
            content = f"Noise document {index}: CDN cache invalidation and unrelated UI notes."
            priority = 5
        materials.append(
            ContextMaterial(
                name=f"session-{session_index}-{topic}-doc-{index}",
                content=content,
                role="knowledge",
                priority=priority,
                metadata={"tags": (topic, "live-soak"), "session": session_index},
            )
        )
    return InMemoryContextMaterialStore(
        tuple(materials),
        embedding_provider=embedding,
        embedding_model="embedding-3",
        embedding_dimensions=512,
    )


async def _run_session(
    *,
    pass_index: int,
    session_index: int,
    task_count: int,
    deepseek_key: str,
    embed_key: str,
) -> dict[str, Any]:
    credential_arg = {"api" + "_key": deepseek_key}
    embedding_credential_arg = {"api" + "_key": embed_key}
    provider = DeepSeekChatProvider(
        **credential_arg,
        model="deepseek-v4-pro",
        timeout_seconds=180.0,
    )
    embedding = ZhipuEmbedding3Provider(
        **embedding_credential_arg,
        model="embedding-3",
        default_dimensions=512,
        timeout_seconds=90.0,
    )
    memory = InMemoryMemoryStore(
        embedding_provider=embedding,
        embedding_model="embedding-3",
        embedding_dimensions=512,
    )
    timeline = TimelineStore()
    mcp = MCPCenter()
    mcp.register_server(
        MCPServerSpec(
            name=f"live-soak-docs-{pass_index}-{session_index}",
            transport="memory",
            tags=("live-soak",),
        )
    )
    mcp.register_connector("memory", _LiveSoakMCPConnector(session_index))
    await mcp.refresh(fail_fast=False)
    await mcp.refresh_resources()
    await mcp.refresh_prompts()
    session = AgentSession(
        profile=AgentProfile(
            name=f"live-soak-{pass_index}-{session_index}",
            model="deepseek-v4-pro",
            instructions=(
                "You are a RavenHeart ReAct stress-test agent. Always output exactly one JSON "
                "action object. For tasks that name a tool, call that tool first. After tool "
                "success, failure, or timeout-like result, finish with concise recovery analysis."
            ),
            budget=RuntimeBudget(
                max_iterations=4,
                max_prompt_bytes=18000,
                max_timeline_bytes=5000,
            ),
        ),
        provider=provider,
        tools=_build_tools(),
        skills=_build_skills(),
        mcp=mcp,
        memory=memory,
        timeline=timeline,
        context_material_store=_build_context_store(embedding, session_index=session_index),
        perception_controller=YaklangStylePerceptionController(
            evaluator=ProviderBackedPerceptionEvaluator(
                provider,
                model="deepseek-v4-pro",
                strict_response_format=False,
            ),
            config=PerceptionRuntimeConfig(sync_triggers=True),
        ),
        native_tool_calls=True,
        provider_cache_mode="ephemeral",
        provider_cache_min_segment_bytes=256,
        metadata={
            "scenario_whitelist": (
                "inspect_alpha",
                "inspect_beta",
                "inspect_gamma",
                "exec_echo",
                "unstable_probe",
                "timeout_probe",
            )
        },
    )
    runner = AgentRunner(session)
    tasks = _build_tasks(session_index, task_count)
    outcomes = []
    task_reports: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        try:
            request_start = len(provider.requests)
            request = AgentRunRequest(task=task.task, refresh=index == 0)
            outcome = await runner.run(request)
            outcomes.append(outcome)
            new_agent_requests = [
                item
                for item in provider.requests[request_start:]
                if not item.metadata.get("agent_core_perception")
            ]
            joined_prompt = "\n\n".join(_prompt_text(item) for item in new_agent_requests)
            joined_request = "\n\n".join(_request_text(item) for item in new_agent_requests)
            task_reports.append(
                {
                    "index": index,
                    "topic": task.topic,
                    "tool_name": task.tool_name,
                    "expects_failure": task.expects_failure,
                    "status": outcome.result.status,
                    "iterations": outcome.result.iterations,
                    "output": outcome.result.output[:500],
                    "request_count": len(new_agent_requests),
                    "request_bytes": [_request_bytes(item) for item in new_agent_requests],
                    "memory_hits": int(outcome.memory_search_manifest.get("hit_count") or 0),
                    "knowledge_hits": int(outcome.knowledge_recall_manifest.get("hit_count") or 0),
                    "midterm_hits": int(outcome.midterm_timeline_recall_manifest.get("hit_count") or 0),
                    "prompt_contains_memory": "[memory]" in joined_prompt,
                    "prompt_contains_knowledge_marker": "[knowledge_recall]" in joined_prompt,
                    "prompt_contains_alpha_fact": "ALPHA_FACT" in joined_prompt
                    or "ALPHA_FACT" in joined_request,
                    "prompt_contains_beta_fact": "BETA_FACT" in joined_prompt
                    or "BETA_FACT" in joined_request,
                    "prompt_contains_gamma_fact": "GAMMA_FACT" in joined_prompt
                    or "GAMMA_FACT" in joined_request,
                    "prompt_contains_mcp": "[mcp_servers]" in joined_prompt
                    or "mcp_resource:" in joined_prompt,
                }
            )
        except Exception as exc:
            return {
                "pass_index": pass_index,
                "session_index": session_index,
                "status": "failed",
                "error": str(exc),
                "completed_tasks": len(outcomes),
            }

    agent_requests = [
        request
        for request in provider.requests
        if not request.metadata.get("agent_core_perception")
    ]
    perception_requests = [
        request
        for request in provider.requests
        if request.metadata.get("agent_core_perception")
    ]
    prompt_texts = [_prompt_text(request) for request in agent_requests]
    request_texts = [_request_text(request) for request in agent_requests]
    request_bytes = [_request_bytes(request) for request in agent_requests]
    compact_delta = [_compact_delta_bytes(request) for request in agent_requests]
    prompt_byte_values = [_prompt_bytes(request) for request in agent_requests]
    high_static_hashes = _segment_hashes(agent_requests, "high_static")
    frozen_hashes = _segment_hashes(agent_requests, "frozen")
    frozen_capability_prefix_hashes = _frozen_capability_prefix_hashes(agent_requests)
    semi_1_hashes = _segment_hashes(agent_requests, "semi_dynamic_1")
    semi_2_hashes = _segment_hashes(agent_requests, "semi_dynamic_2")
    dynamic_hashes = _segment_hashes(agent_requests, "timeline_open_dynamic")
    cache_control_counts = [_cache_control_count(request) for request in agent_requests]
    inventories = [_tool_inventory(request) for request in agent_requests]
    usage_stats = _usage_stats(provider.requests, provider.responses)
    failure_tasks = sum(1 for task in tasks if task.expects_failure)
    failed_tool_results = [
        item
        for item in timeline.items
        if item.metadata.get("status") == "failed"
        or "FAILED" in item.content
        or "timeout" in item.content.lower()
    ]
    alpha_return_reports = [
        item
        for item in task_reports
        if item["topic"] == "alpha" and item["index"] >= 5
    ]
    knowledge_task_reports = [
        item for item in task_reports if int(item.get("knowledge_hits") or 0) > 0
    ]
    return {
        "schema_version": "raven-heart-live-yaklang-soak-session/v1",
        "pass_index": pass_index,
        "session_index": session_index,
        "status": "completed",
        "task_count": len(tasks),
        "task_reports": task_reports,
        "outcome_statuses": [outcome.result.status for outcome in outcomes],
        "all_outcomes_completed": all(outcome.result.status == "completed" for outcome in outcomes),
        "provider_request_count": len(agent_requests),
        "perception_request_count": len(perception_requests),
        "embedding_call_count": int(embedding.manifest().get("call_count") or 0),
        "request_bytes": request_bytes,
        "prompt_bytes": prompt_byte_values,
        "compact_delta_bytes": compact_delta,
        "max_request_bytes": max(request_bytes, default=0),
        "max_compact_delta_bytes": max(compact_delta, default=0),
        "tail_request_byte_range": (
            max(request_bytes[-8:], default=0) - min(request_bytes[-8:], default=0)
        ),
        "cache_control_counts": cache_control_counts,
        "cache_control_present": any(count > 0 for count in cache_control_counts),
        "usage": usage_stats,
        "prompt_cache_hit_tokens": usage_stats["prompt_cache_hit_tokens"],
        "prompt_cache_miss_tokens": usage_stats["prompt_cache_miss_tokens"],
        "prompt_cache_hit_ratio": usage_stats["prompt_cache_hit_ratio"],
        "has_prompt_cache_usage": usage_stats["has_prompt_cache_usage"],
        "high_static_stable": len({item for item in high_static_hashes if item}) <= 1,
        "frozen_changed": len({item for item in frozen_hashes if item}) > 1,
        "frozen_capability_prefix_stable": len(
            {item for item in frozen_capability_prefix_hashes if item}
        )
        <= 1,
        "semi_dynamic_1_stable": len({item for item in semi_1_hashes if item}) <= 1,
        "semi_dynamic_1_changed": len({item for item in semi_1_hashes if item}) > 1,
        "semi_dynamic_2_stable": len({item for item in semi_2_hashes if item}) <= 1,
        "semi_dynamic_changed": len({item for item in semi_1_hashes if item}) > 1,
        "timeline_open_dynamic_changed": len({item for item in dynamic_hashes if item}) > 1,
        "memory_hits_total": sum(
            int(outcome.memory_search_manifest.get("hit_count") or 0)
            for outcome in outcomes
        ),
        "knowledge_hits_total": sum(
            int(outcome.knowledge_recall_manifest.get("hit_count") or 0)
            for outcome in outcomes
        ),
        "midterm_hits_total": sum(
            int(outcome.midterm_timeline_recall_manifest.get("hit_count") or 0)
            for outcome in outcomes
        ),
        "timeline_item_count": len(timeline.items),
        "memory_record_count": len(memory.records),
        "tool_inventory_max_count": max((item["count"] for item in inventories), default=0),
        "tool_inventory_max_bytes": max((item["bytes"] for item in inventories), default=0),
        "tool_inventory_samples": inventories[:3] + inventories[-3:],
        "failure_tasks": failure_tasks,
        "failed_tool_result_count": len(failed_tool_results),
        "failure_recovery_completed": len(failed_tool_results) >= max(1, failure_tasks // 2)
        and all(outcome.result.status == "completed" for outcome in outcomes),
        "alpha_recalled_after_switch": any(
            item.get("prompt_contains_alpha_fact")
            and (
                item.get("prompt_contains_memory")
                or int(item.get("memory_hits") or 0) > 0
                or int(item.get("midterm_hits") or 0) > 0
            )
            for item in alpha_return_reports
        ),
        "beta_seen": any("BETA_FACT" in text for text in request_texts),
        "gamma_seen": any("GAMMA_FACT" in text for text in request_texts),
        "knowledge_visible": any(
            item.get("prompt_contains_knowledge_marker")
            or int(item.get("knowledge_hits") or 0) > 0
            for item in knowledge_task_reports
        ),
        "mcp_visible": any("[mcp_servers]" in text or "mcp_resource:" in text for text in prompt_texts),
        "provider": provider.manifest(),
        "embedding": embedding.manifest(),
    }


def _aggregate(session_reports: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in session_reports if item.get("status") == "completed"]
    request_bytes = [
        value for item in completed for value in item.get("request_bytes", ())
    ]
    compact_delta = [
        value for item in completed for value in item.get("compact_delta_bytes", ())
    ]
    prompt_cache_hit_tokens = sum(
        int(item.get("prompt_cache_hit_tokens") or 0) for item in completed
    )
    prompt_cache_miss_tokens = sum(
        int(item.get("prompt_cache_miss_tokens") or 0) for item in completed
    )
    prompt_cache_denominator = prompt_cache_hit_tokens + prompt_cache_miss_tokens
    return {
        "session_count": len(session_reports),
        "completed_session_count": len(completed),
        "all_sessions_completed": len(completed) == len(session_reports),
        "task_count": sum(int(item.get("task_count") or 0) for item in completed),
        "provider_request_count": sum(int(item.get("provider_request_count") or 0) for item in completed),
        "perception_request_count": sum(int(item.get("perception_request_count") or 0) for item in completed),
        "embedding_call_count": sum(int(item.get("embedding_call_count") or 0) for item in completed),
        "max_request_bytes": max(request_bytes, default=0),
        "max_compact_delta_bytes": max(compact_delta, default=0),
        "memory_hits_total": sum(int(item.get("memory_hits_total") or 0) for item in completed),
        "knowledge_hits_total": sum(int(item.get("knowledge_hits_total") or 0) for item in completed),
        "midterm_hits_total": sum(int(item.get("midterm_hits_total") or 0) for item in completed),
        "timeline_item_count": sum(int(item.get("timeline_item_count") or 0) for item in completed),
        "memory_record_count": sum(int(item.get("memory_record_count") or 0) for item in completed),
        "failed_tool_result_count": sum(int(item.get("failed_tool_result_count") or 0) for item in completed),
        "cache_control_present": all(bool(item.get("cache_control_present")) for item in completed),
        "prompt_cache_hit_tokens": prompt_cache_hit_tokens,
        "prompt_cache_miss_tokens": prompt_cache_miss_tokens,
        "prompt_cache_hit_ratio": (
            round(prompt_cache_hit_tokens / prompt_cache_denominator, 6)
            if prompt_cache_denominator
            else 0.0
        ),
        "has_prompt_cache_usage": prompt_cache_denominator > 0,
        "high_static_stable": all(bool(item.get("high_static_stable")) for item in completed),
        "frozen_capability_prefix_stable": all(
            bool(item.get("frozen_capability_prefix_stable")) for item in completed
        ),
        "semi_dynamic_2_stable": all(
            bool(item.get("semi_dynamic_2_stable")) for item in completed
        ),
        "semi_dynamic_1_stable": all(
            bool(item.get("semi_dynamic_1_stable")) for item in completed
        ),
        "semi_dynamic_1_changed": all(
            bool(item.get("semi_dynamic_1_changed")) for item in completed
        ),
        "request_growth_bounded": max(request_bytes, default=0) <= 18000,
        "compact_delta_bounded": max(compact_delta, default=0) <= 2048,
        "failure_recovery_completed": all(
            bool(item.get("failure_recovery_completed")) for item in completed
        ),
        "alpha_recalled_after_switch": all(
            bool(item.get("alpha_recalled_after_switch")) for item in completed
        ),
        "knowledge_visible": all(bool(item.get("knowledge_visible")) for item in completed),
        "mcp_visible": all(bool(item.get("mcp_visible")) for item in completed),
    }


async def run_live_soak(
    *,
    output: Path,
    passes: int,
    sessions: int,
    tasks_per_session: int,
) -> dict[str, Any]:
    deepseek_env = "DEEPSEEK" + "_API" + "_KEY"
    deepseek_key = _credential(deepseek_env)
    embed_key = _credential("glm-embed-key") or _credential("GLM_EMBED_KEY")
    if not deepseek_key:
        raise RuntimeError(f"{deepseek_env} is not set")
    if not embed_key:
        raise RuntimeError("glm-embed-key or GLM_EMBED_KEY is not set")

    reports: list[dict[str, Any]] = []
    for pass_index in range(passes):
        batch = await asyncio.gather(
            *(
                _run_session(
                    pass_index=pass_index,
                    session_index=session_index,
                    task_count=tasks_per_session,
                    deepseek_key=deepseek_key,
                    embed_key=embed_key,
                )
                for session_index in range(sessions)
            )
        )
        reports.extend(batch)
    report = {
        "schema_version": "raven-heart-live-yaklang-soak/v1",
        "status": "completed",
        "models": {"chat": "deepseek-v4-pro", "embedding": "embedding-3"},
        "parameters": {
            "passes": passes,
            "sessions": sessions,
            "tasks_per_session": tasks_per_session,
        },
        "env": {
            "deepseek_key_present": bool(deepseek_key),
            "embedding_key_present": bool(embed_key),
        },
        "aggregate": _aggregate(reports),
        "sessions": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(Path(".tmp") / "live-yaklang-soak.json"))
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--tasks-per-session", type=int, default=20)
    args = parser.parse_args()
    report = asyncio.run(
        run_live_soak(
            output=Path(args.output),
            passes=max(1, args.passes),
            sessions=max(1, args.sessions),
            tasks_per_session=max(1, args.tasks_per_session),
        )
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": args.output,
                "aggregate": report["aggregate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
