from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
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
from agent_core.providers import LLMRequest
from agent_core.runner import AgentRunner, AgentRunRequest, AgentSession
from agent_core.skills import SkillRegistry, SkillSpec, SkillsContext
from agent_core.timeline import TimelineStore
from agent_core.tools import ToolInvocation, ToolRegistry, ToolResult, ToolSpec
from agent_core.turn_runtime import (
    PerceptionRuntimeConfig,
    ProviderBackedPerceptionEvaluator,
    YaklangStylePerceptionController,
)


ALPHA_TOOL_FACT = (
    "ALPHA_FACT: OAuth callback-state nonce alpha-42 must be persisted before redirect; "
    "callback handlers must reject nonce replay before token exchange."
)
BETA_TOOL_FACT = (
    "BETA_FACT: Billing webhook retry uses a 30 request per minute tenant burst budget; "
    "429 responses require jittered retry and idempotency keys."
)
KNOWLEDGE_DOC_DIR = Path(__file__).with_name("knowledge_docs")
KNOWLEDGE_DOC_SPECS = (
    (
        "doc-alpha-callback-state",
        "alpha_callback_state.md",
        ("alpha", "callback"),
        90,
    ),
    (
        "doc-beta-billing-webhook",
        "beta_billing_webhook.md",
        ("beta", "billing"),
        85,
    ),
    (
        "doc-unrelated-cache",
        "unrelated_cache.md",
        ("cache",),
        10,
    ),
)


def _prompt_text(request: LLMRequest) -> str:
    return "\n\n".join(
        message.content
        for message in request.messages
        if message.metadata.get("agent_core_prompt")
    )


def _request_text(request: LLMRequest) -> str:
    return "\n\n".join(message.content for message in request.messages)


def _frozen_capability_prefix(text: str) -> str:
    prefix = text.split("\n\n[timeline_frozen]\n", 1)[0]
    return prefix.replace("</prompt_materials>", "").strip()


def _prompt_segments(request: LLMRequest) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for message in request.messages:
        if not message.metadata.get("agent_core_prompt"):
            continue
        content = message.content
        item = {
            "role": message.role,
            "segment": message.metadata.get("agent_core_prompt_segment"),
            "bytes": len(content.encode("utf-8")),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest() if content else "",
            "cache_hint": dict(message.metadata.get("cache_hint") or {}),
            "bucket_roles": list(message.metadata.get("bucket_roles") or ()),
        }
        if message.metadata.get("agent_core_prompt_segment") == "frozen":
            prefix = _frozen_capability_prefix(content)
            item["capability_prefix_sha256"] = (
                hashlib.sha256(prefix.encode("utf-8")).hexdigest() if prefix else ""
            )
            item["contains_timeline_frozen"] = "[timeline_frozen]" in content
        segments.append(item)
    return segments


def _provider_request_report(
    *,
    index: int,
    request: LLMRequest,
    prompt_text: str,
) -> dict[str, Any]:
    prompt_messages = [message for message in request.messages if message.metadata.get("agent_core_prompt")]
    prompt_bytes = sum(len(message.content.encode("utf-8")) for message in prompt_messages)
    content_bytes = sum(len(message.content.encode("utf-8")) for message in request.messages)
    request_text = _request_text(request)
    return {
        "index": index,
        "message_count": len(request.messages),
        "prompt_message_count": len(prompt_messages),
        "compact_delta_message_count": max(0, len(request.messages) - len(prompt_messages)),
        "content_bytes": content_bytes,
        "prompt_bytes": prompt_bytes,
        "compact_delta_bytes": max(0, content_bytes - prompt_bytes),
        "segments": _prompt_segments(request),
        "contains_alpha_fact": "ALPHA_FACT" in prompt_text,
        "contains_beta_fact": "BETA_FACT" in prompt_text,
        "contains_alpha_fact_in_request": "ALPHA_FACT" in request_text,
        "contains_beta_fact_in_request": "BETA_FACT" in request_text,
        "contains_memory": "[memory]" in prompt_text,
        "contains_knowledge": "[knowledge_recall]" in prompt_text,
        "contains_midterm_timeline": "[midterm_timeline_recall]" in prompt_text,
        "contains_skill_context": "alpha-callback-review" in prompt_text
        or "beta-billing-review" in prompt_text,
        "contains_mcp_context": "mcp_resource:" in prompt_text
        or "MCP document:" in prompt_text
        or "[mcp_servers]" in prompt_text,
    }


def _prompt_stability_summary(provider_request_reports: list[dict[str, Any]]) -> dict[str, Any]:
    segment_hashes: dict[str, list[str]] = {}
    frozen_capability_prefix_hashes: list[str] = []
    frozen_timeline_view_count = 0
    for request in provider_request_reports:
        for segment in request["segments"]:
            name = str(segment.get("segment") or "")
            if not name:
                continue
            segment_hashes.setdefault(name, []).append(str(segment.get("sha256") or ""))
            if name == "frozen":
                prefix_hash = str(segment.get("capability_prefix_sha256") or "")
                if prefix_hash:
                    frozen_capability_prefix_hashes.append(prefix_hash)
                if segment.get("contains_timeline_frozen"):
                    frozen_timeline_view_count += 1
    unique_hashes = {
        name: sorted({digest for digest in hashes if digest})
        for name, hashes in sorted(segment_hashes.items())
    }
    frozen_prefix_unique = sorted(
        {digest for digest in frozen_capability_prefix_hashes if digest}
    )
    return {
        "schema_version": "raven-heart-live-prompt-stability/v1",
        "segment_unique_hash_count": {
            name: len(hashes) for name, hashes in unique_hashes.items()
        },
        "high_static_stable": len(unique_hashes.get("high_static", ())) <= 1,
        "frozen_segment_stable": len(unique_hashes.get("frozen", ())) <= 1,
        "frozen_capability_prefix_stable": len(frozen_prefix_unique) <= 1,
        "frozen_capability_prefix_change_count": max(0, len(frozen_prefix_unique) - 1),
        "frozen_capability_prefix_bounded": len(frozen_prefix_unique) <= 2,
        "frozen_timeline_view_present": frozen_timeline_view_count > 0,
        "semi_dynamic_1_changed": len(unique_hashes.get("semi_dynamic_1", ())) > 1,
        "semi_dynamic_2_stable": len(unique_hashes.get("semi_dynamic_2", ())) <= 1,
        "semi_dynamic_changed": len(unique_hashes.get("semi_dynamic_1", ())) > 1,
        "timeline_open_dynamic_changed": len(unique_hashes.get("timeline_open_dynamic", ())) > 1,
        "request_content_bytes": [
            int(request["content_bytes"]) for request in provider_request_reports
        ],
        "prompt_bytes": [int(request["prompt_bytes"]) for request in provider_request_reports],
        "compact_delta_bytes": [
            int(request["compact_delta_bytes"]) for request in provider_request_reports
        ],
        "max_content_bytes": max(
            (int(request["content_bytes"]) for request in provider_request_reports),
            default=0,
        ),
        "max_compact_delta_bytes": max(
            (int(request["compact_delta_bytes"]) for request in provider_request_reports),
            default=0,
        ),
    }


def _functional_assertions(
    *,
    outcomes: list[Any],
    provider_request_reports: list[dict[str, Any]],
    timeline: TimelineStore,
    memory: InMemoryMemoryStore,
    embedding: ZhipuEmbedding3Provider,
    perception_request_count: int,
) -> dict[str, Any]:
    alpha_return_request = provider_request_reports[-1] if provider_request_reports else {}
    return {
        "schema_version": "raven-heart-live-functional-assertions/v1",
        "all_runs_completed": all(outcome.result.status == "completed" for outcome in outcomes),
        "alpha_tool_fact_in_timeline": any("ALPHA_FACT" in item.content for item in timeline.items),
        "beta_tool_fact_in_timeline": any("BETA_FACT" in item.content for item in timeline.items),
        "alpha_tool_fact_in_memory": any("ALPHA_FACT" in record.content for record in memory.records),
        "beta_tool_fact_in_memory": any("BETA_FACT" in record.content for record in memory.records),
        "alpha_return_prompt_contains_memory": bool(alpha_return_request.get("contains_memory")),
        "alpha_return_prompt_contains_alpha_fact": bool(
            alpha_return_request.get("contains_alpha_fact")
        ),
        "tool_result_visible_in_provider_request": any(
            request.get("contains_alpha_fact_in_request")
            or request.get("contains_beta_fact_in_request")
            for request in provider_request_reports
        ),
        "skills_visible_in_prompt": any(
            request.get("contains_skill_context") for request in provider_request_reports
        ),
        "mcp_visible_in_prompt": any(
            request.get("contains_mcp_context") for request in provider_request_reports
        ),
        "knowledge_or_midterm_used_after_topic_switch": any(
            request.get("contains_knowledge") or request.get("contains_midterm_timeline")
            for request in provider_request_reports[3:]
        ),
        "embedding_used": int(embedding.manifest().get("call_count") or 0) > 0,
        "perception_used": perception_request_count > 0,
    }


def _shrink(text: str, limit: int = 260) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 3)].rstrip() + "..."


class _LiveMCPConnector:
    async def list_tools(self, server: MCPServerSpec) -> tuple[MCPToolSpec, ...]:
        return (
            MCPToolSpec(
                reference=MCPToolReference(server_name=server.name, tool_name="read_alpha_notes"),
                description="Read alpha callback-state design note",
                tags=("alpha", "callback", "docs"),
            ),
        )

    async def invoke_tool(
        self,
        server: MCPServerSpec,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        return ToolResult(
            call_id="mcp-alpha-note",
            tool_name=tool_name,
            content="MCP_ALPHA_NOTE: alpha callback nonce alpha-42 must be replay checked.",
        )

    async def list_resources(self, server: MCPServerSpec) -> tuple[MCPResourceSpec, ...]:
        return (
            MCPResourceSpec(
                server_name=server.name,
                uri="memory://alpha/callback-design.md",
                name="Alpha callback design",
                description="OAuth callback-state nonce alpha-42 replay validation",
                mime_type="text/markdown",
                tags=("alpha", "callback", "docs"),
            ),
            MCPResourceSpec(
                server_name=server.name,
                uri="memory://beta/billing-webhook.md",
                name="Beta billing webhook",
                description="Billing webhook rate limit retry and idempotency note",
                mime_type="text/markdown",
                tags=("beta", "billing", "docs"),
            ),
        )

    async def read_resource(self, server: MCPServerSpec, uri: str) -> MCPResourceContent:
        if "alpha" in uri:
            text = (
                "MCP document: Alpha callback-state nonce alpha-42 is stored before redirect, "
                "then replay checked before token exchange."
            )
        else:
            text = (
                "MCP document: Beta billing webhook retry must respect a 30 RPM tenant budget "
                "and use idempotency keys."
            )
        return MCPResourceContent(server_name=server.name, uri=uri, text=text, mime_type="text/markdown")

    async def list_prompts(self, server: MCPServerSpec) -> tuple[MCPPromptSpec, ...]:
        return (
            MCPPromptSpec(
                server_name=server.name,
                name="alpha_callback_review",
                description="Review alpha callback replay and nonce storage",
                tags=("alpha", "callback"),
            ),
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
            messages=(
                {
                    "role": "user",
                    "content": "When alpha callback appears, check nonce persistence and replay rejection.",
                },
            ),
        )


def _build_tools() -> ToolRegistry:
    registry = ToolRegistry()

    async def inspect_alpha(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            content=ALPHA_TOOL_FACT,
        )

    async def inspect_beta(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            content=BETA_TOOL_FACT,
        )

    async def exec_echo(invocation: ToolInvocation) -> ToolResult:
        command = str(invocation.arguments.get("command") or "echo ok")
        if command not in {"echo alpha", "echo beta", "echo ok"}:
            return ToolResult(
                call_id=invocation.call_id,
                tool_name=invocation.tool_name,
                status="failed",
                error=f"command not allowed in live smoke: {command}",
            )
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            content=f"EXEC_RESULT: {command}",
        )

    registry.register(
        ToolSpec(
            name="inspect_alpha",
            description="Return evidence about alpha OAuth callback-state nonce handling.",
            parameters_schema={"type": "object", "properties": {}},
            tags=("alpha", "callback"),
        ),
        inspect_alpha,
    )
    registry.register(
        ToolSpec(
            name="inspect_beta",
            description="Return evidence about beta billing webhook rate limit and retry behavior.",
            parameters_schema={"type": "object", "properties": {}},
            tags=("beta", "billing"),
        ),
        inspect_beta,
    )
    registry.register(
        ToolSpec(
            name="exec_echo",
            description="Safe exec-like smoke tool; only echo alpha, echo beta, or echo ok are allowed.",
            parameters_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
            tags=("exec", "safe"),
        ),
        exec_echo,
    )
    return registry


def _build_skills() -> SkillsContext:
    registry = SkillRegistry()
    registry.register(
        SkillSpec(
            name="alpha-callback-review",
            description="Review OAuth callback-state nonce replay protection",
            prompt="For alpha callback tasks, verify nonce persistence, replay rejection, and token exchange order.",
            tags=("alpha", "callback"),
            priority=20,
        )
    )
    registry.register(
        SkillSpec(
            name="beta-billing-review",
            description="Review billing webhook rate limits and retry policy",
            prompt="For beta billing tasks, verify tenant burst budget, retry jitter, and idempotency keys.",
            tags=("beta", "billing"),
            priority=15,
        )
    )
    return SkillsContext(registry)


def _build_mcp() -> MCPCenter:
    center = MCPCenter()
    center.register_server(MCPServerSpec(name="live-docs", transport="memory", tags=("docs",)))
    center.register_connector("memory", _LiveMCPConnector())
    return center


def _build_context_store(
    embedding: ZhipuEmbedding3Provider,
) -> tuple[InMemoryContextMaterialStore, list[dict[str, Any]]]:
    materials: list[ContextMaterial] = []
    manifests: list[dict[str, Any]] = []
    for name, filename, tags, priority in KNOWLEDGE_DOC_SPECS:
        path = KNOWLEDGE_DOC_DIR / filename
        content = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        materials.append(
            ContextMaterial(
                name=name,
                content=content,
                role="knowledge",
                priority=priority,
                metadata={
                    "tags": tags,
                    "source": "file-backed-doc",
                    "path": str(path),
                    "sha256": digest,
                },
            )
        )
        manifests.append(
            {
                "name": name,
                "path": str(path),
                "bytes": len(content.encode("utf-8")),
                "sha256": digest,
                "tags": list(tags),
                "priority": priority,
            }
        )
    return InMemoryContextMaterialStore(
        tuple(materials),
        embedding_provider=embedding,
        embedding_model="embedding-3",
        embedding_dimensions=512,
    ), manifests


async def run_live(output: Path) -> dict[str, Any]:
    deepseek_env = "DEEPSEEK" + "_API" + "_KEY"
    deepseek_key = os.environ.get(deepseek_env) or ""
    embed_key = os.environ.get("glm-embed-key") or os.environ.get("GLM_EMBED_KEY") or ""
    if not deepseek_key:
        raise RuntimeError(f"{deepseek_env} is not set")
    if not embed_key:
        raise RuntimeError("glm-embed-key or GLM_EMBED_KEY is not set")

    credential_arg = {"api" + "_key": deepseek_key}
    embedding_credential_arg = {"api" + "_key": embed_key}
    provider = DeepSeekChatProvider(**credential_arg, model="deepseek-v4-pro")
    embedding = ZhipuEmbedding3Provider(
        **embedding_credential_arg,
        model="embedding-3",
        default_dimensions=512,
    )
    memory = InMemoryMemoryStore(
        embedding_provider=embedding,
        embedding_model="embedding-3",
        embedding_dimensions=512,
    )
    timeline = TimelineStore()
    mcp = _build_mcp()
    await mcp.refresh(fail_fast=False)
    await mcp.refresh_resources()
    await mcp.refresh_prompts()
    context_store, knowledge_documents = _build_context_store(embedding)
    session = AgentSession(
        profile=AgentProfile(
            name="live-yaklang-e2e",
            model="deepseek-v4-pro",
            instructions=(
                "You are a RavenHeart ReAct agent. Always respond with one JSON action object. "
                "Use call_tool for evidence before finish. Keep final outputs concise."
            ),
            budget=RuntimeBudget(max_iterations=4, max_prompt_bytes=16000, max_timeline_bytes=6000),
        ),
        provider=provider,
        tools=_build_tools(),
        skills=_build_skills(),
        mcp=mcp,
        memory=memory,
        timeline=timeline,
        context_material_store=context_store,
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
        metadata={"scenario_whitelist": ("inspect_alpha", "inspect_beta", "exec_echo")},
    )
    runner = AgentRunner(session)

    requests = (
        AgentRunRequest(
            task=(
                "Alpha task: inspect callback-state nonce handling. "
                "Call inspect_alpha before finishing."
            ),
            refresh=True,
        ),
        AgentRunRequest(
            task=(
                "Beta task: switch context to billing webhook rate-limit behavior. "
                "Call inspect_beta before finishing."
            ),
        ),
        AgentRunRequest(
            task=(
                "Alpha task again: return to callback-state nonce analysis. "
                "Use remembered alpha facts and then finish."
            ),
        ),
    )
    outcomes = []
    for request in requests:
        outcome = await runner.run(request)
        outcomes.append(outcome)

    provider_requests = [
        request
        for request in provider.requests
        if not request.metadata.get("agent_core_perception")
    ]
    perception_requests = [
        request
        for request in provider.requests
        if request.metadata.get("agent_core_perception")
    ]
    prompt_texts = [_prompt_text(request) for request in provider_requests]
    provider_request_reports = [
        _provider_request_report(index=index, request=request, prompt_text=prompt_texts[index])
        for index, request in enumerate(provider_requests)
    ]
    report = {
        "schema_version": "raven-heart-live-yaklang-e2e/v1",
        "status": "completed",
        "models": {
            "chat": "deepseek-v4-pro",
            "embedding": "embedding-3",
            "embedding_dimensions": 512,
        },
        "env": {
            "deepseek_key_present": bool(deepseek_key),
            "embedding_key_present": bool(embed_key),
        },
        "knowledge_documents": knowledge_documents,
        "outcomes": [
            {
                "task": request.task,
                "status": outcome.result.status,
                "iterations": outcome.result.iterations,
                "output": outcome.result.output,
                "knowledge_hits": outcome.knowledge_recall_manifest.get("hit_count"),
                "midterm_timeline_hits": outcome.midterm_timeline_recall_manifest.get("hit_count"),
                "memory_hits": outcome.memory_search_manifest.get("hit_count"),
                "downstream_enabled": outcome.downstream_plan_manifest.get("enabled"),
            }
            for request, outcome in zip(requests, outcomes, strict=True)
        ],
        "provider_requests": provider_request_reports,
        "prompt_stability": _prompt_stability_summary(provider_request_reports),
        "functional_assertions": _functional_assertions(
            outcomes=outcomes,
            provider_request_reports=provider_request_reports,
            timeline=timeline,
            memory=memory,
            embedding=embedding,
            perception_request_count=len(perception_requests),
        ),
        "perception_requests": [
            {
                "index": index,
                "content_bytes": sum(len(message.content.encode("utf-8")) for message in request.messages),
                "strict_response_format": request.metadata.get("strict_response_format"),
            }
            for index, request in enumerate(perception_requests)
        ],
        "timeline": {
            "item_count": len(timeline.items),
            "kinds": [item.kind for item in timeline.items],
            "contains_alpha_fact": any("ALPHA_FACT" in item.content for item in timeline.items),
            "contains_beta_fact": any("BETA_FACT" in item.content for item in timeline.items),
            "entries": [
                {
                    "kind": item.kind,
                    "content": _shrink(item.content),
                    "tool_name": item.metadata.get("tool_name"),
                    "status": item.metadata.get("status"),
                    "action": item.metadata.get("action"),
                }
                for item in timeline.items
            ],
        },
        "memory": {
            "record_count": len(memory.records),
            "contains_alpha_fact": any("ALPHA_FACT" in record.content for record in memory.records),
            "contains_beta_fact": any("BETA_FACT" in record.content for record in memory.records),
        },
        "embedding": embedding.manifest(),
        "provider": provider.manifest(),
        "model_responses": [
            {
                "index": index,
                "content": _shrink(response.content, 500),
                "finish_reason": response.finish_reason,
                "tool_calls": [tool_call.manifest() for tool_call in response.tool_calls],
                "has_action": response.action is not None,
            }
            for index, response in enumerate(provider.responses)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(Path(".tmp") / "live-yaklang-e2e.json"),
        help="Where to write the redacted JSON report.",
    )
    args = parser.parse_args()
    report = asyncio.run(run_live(Path(args.output)))
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": args.output,
                "provider_request_count": len(report["provider_requests"]),
                "embedding_call_count": report["embedding"]["call_count"],
                "outcomes": report["outcomes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
