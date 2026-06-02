from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import pytest

from agent_core.config import AgentProfile, RuntimeBudget
from agent_core.memory import MemoryQuery, MemoryWrite
from agent_core.providers import LLMMessage, LLMProviderCenter, LLMRequest
from agent_core.runner import AgentRunner
from agent_core.testing import MockLLMProvider
from agent_core.tools import ToolInvocation
from integrations.raven.agent_core_adapter import (
    RuntimeAgentCoreSessionConfig,
    RuntimeMemoryAdapter,
    RuntimeMemoryAdapterConfig,
    RuntimeTaskTreeSessionConfig,
    RuntimeLLMProviderAdapter,
    RuntimeLLMProviderAdapterConfig,
    SDKToolRuntimeAdapter,
    build_agent_core_runtime_session,
    build_agent_core_session_from_sdk_tools,
    build_agent_core_tasktree_runtime_session,
    inspect_agent_core_tasktree_runtime,
)
from app.openai_agents_runtime.tool_manifest import ToolManifest, bind_tool_manifest


@dataclass
class FakeSDKTool:
    name: str
    description: str
    params_json_schema: dict[str, Any]
    agent_core_coroutine: Any | None = None


class _OpenAIMessage:
    content = '{"action":"finish","arguments":{"output":"openai-like"}}'


class _OpenAIChoice:
    message = _OpenAIMessage()
    finish_reason = "stop"


class _OpenAIUsage:
    prompt_tokens = 3
    completion_tokens = 4
    total_tokens = 7


class _OpenAIResponse:
    choices = [_OpenAIChoice()]
    usage = _OpenAIUsage()


class _OpenAICompatibleCreate:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def __call__(self, **kwargs: Any) -> _OpenAIResponse:
        self.kwargs = kwargs
        return _OpenAIResponse()


class _OpenAICompatibleClient:
    def __init__(self) -> None:
        create = _OpenAICompatibleCreate()
        self.create = create
        self.chat = type("Chat", (), {"completions": type("Completions", (), {"create": create})()})()


class _FakeWorkingMemoryBackend:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.write_calls: list[dict[str, Any]] = []

    async def search(
        self,
        query: str,
        *,
        project_id: str,
        run_id: str | None = None,
        task_id: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        self.search_calls.append(
            {
                "query": query,
                "project_id": project_id,
                "run_id": run_id,
                "task_id": task_id,
                "limit": limit,
            }
        )
        return {
            "status": "ok",
            "policy": "temporary_memory_not_graphiti_until_verified",
            "results": [
                {
                    "content": "Inspect target admin UI at /admin",
                    "kind": "tool_observation",
                    "score": 0.8,
                    "memory_id": "m1",
                    "recall_backend": "hash_fallback",
                    "metadata": {"target": "web"},
                }
            ],
        }

    async def write(self, event: dict[str, Any]) -> dict[str, Any]:
        self.write_calls.append(event)
        return {"status": "stored"}


class _SequencedRuntimeCompletion:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        if not self.responses:
            return {"content": '{"action":"finish","arguments":{"output":"done"}}'}
        return {"content": json.dumps(self.responses.pop(0), ensure_ascii=False)}


async def _lookup(**kwargs: Any) -> str:
    return f"lookup:{kwargs['query']}"


def _fake_tool() -> FakeSDKTool:
    tool = FakeSDKTool(
        name="lookup",
        description="Lookup target",
        params_json_schema={
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}},
            "additionalProperties": False,
        },
        agent_core_coroutine=_lookup,
    )
    bind_tool_manifest(
        tool,
        ToolManifest(
            name="lookup",
            capability="research",
            risk_level="low",
            side_effects=(),
            scope_fields=("query",),
            idempotent=True,
            approval_policy="allow",
            required_sandbox="none",
        ),
    )
    return tool


def _fake_tasktree_tool(name: str, description: str, coroutine: Any) -> FakeSDKTool:
    tool = FakeSDKTool(
        name=name,
        description=description,
        params_json_schema={"type": "object", "properties": {}, "additionalProperties": True},
        agent_core_coroutine=coroutine,
    )
    bind_tool_manifest(
        tool,
        ToolManifest(
            name=name,
            capability="task_tree",
            risk_level="low",
            side_effects=("plan_state",),
            scope_fields=("task_id",),
            idempotent=name == "get_plan_status",
            approval_policy="allow",
            required_sandbox="none",
        ),
    )
    return tool


@pytest.mark.asyncio
async def test_sdk_tool_runtime_adapter_exposes_specs_and_invokes_tool() -> None:
    adapter = SDKToolRuntimeAdapter([_fake_tool()])

    specs = adapter.specs()
    result = await adapter.invoke(ToolInvocation(tool_name="lookup", arguments={"query": "target"}))
    missing = await adapter.invoke(ToolInvocation(tool_name="missing"))

    assert specs[0].name == "lookup"
    assert specs[0].parameters_schema["required"] == ["query"]
    assert specs[0].tags == ("research",)
    assert specs[0].metadata["manifest"]["approval_policy"] == "allow"
    assert result.status == "completed"
    assert result.content == "lookup:target"
    assert missing.status == "failed"


@pytest.mark.asyncio
async def test_runtime_llm_provider_adapter_wraps_callable_completion() -> None:
    calls: list[dict[str, Any]] = []

    async def complete(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        return {
            "action": {"action": "finish", "arguments": {"output": "done"}},
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7, "cost_usd": 0.01},
        }

    provider = RuntimeLLMProviderAdapter(
        complete,
        config=RuntimeLLMProviderAdapterConfig(
            provider_name="raven",
            model="raven-mini",
            temperature=0.2,
            max_output_tokens=128,
        ),
    )

    response = await provider.complete(
        LLMRequest(messages=[LLMMessage(role="user", content="task")], metadata={"trace": "t1"})
    )

    assert response.action == {"action": "finish", "arguments": {"output": "done"}}
    assert response.usage.total_tokens == 7
    assert response.metadata["provider"] == "raven"
    assert calls[0]["model"] == "raven-mini"
    assert calls[0]["temperature"] == 0.2
    assert calls[0]["max_tokens"] == 128
    assert calls[0]["metadata"]["trace"] == "t1"


@pytest.mark.asyncio
async def test_runtime_llm_provider_adapter_accepts_direct_action_fields() -> None:
    provider = RuntimeLLMProviderAdapter(
        lambda payload: {"action": "finish", "arguments": {"output": "direct"}},
        config=RuntimeLLMProviderAdapterConfig(model="runtime-model"),
    )

    response = await provider.complete(LLMRequest(messages=[LLMMessage(role="user", content="task")]))

    assert response.action == {"action": "finish", "arguments": {"output": "direct"}}
    assert response.content == ""


@pytest.mark.asyncio
async def test_runtime_llm_provider_adapter_wraps_openai_compatible_client() -> None:
    client = _OpenAICompatibleClient()
    provider = RuntimeLLMProviderAdapter(
        client,
        config=RuntimeLLMProviderAdapterConfig(model="compat-model"),
    )

    response = await provider.complete(LLMRequest(messages=[LLMMessage(role="user", content="task")]))

    assert response.content == '{"action":"finish","arguments":{"output":"openai-like"}}'
    assert response.usage.input_tokens == 3
    assert response.finish_reason == "stop"
    assert client.create.kwargs["model"] == "compat-model"
    assert client.create.kwargs["messages"][0]["content"] == "task"


@pytest.mark.asyncio
async def test_runtime_memory_adapter_wraps_working_memory_shape() -> None:
    backend = _FakeWorkingMemoryBackend()
    memory = RuntimeMemoryAdapter(
        backend,
        config=RuntimeMemoryAdapterConfig(
            source="working_memory",
            project_id="p1",
            run_id="r1",
            task_id="t1",
        ),
    )

    hits = await memory.search(MemoryQuery(query="inspect target", limit=3))
    await memory.write(MemoryWrite(content="remember this", source="note", metadata={"kind": "operator"}))

    assert hits[0].content == "Inspect target admin UI at /admin"
    assert hits[0].source == "tool_observation"
    assert hits[0].metadata["memory_id"] == "m1"
    assert backend.search_calls[0]["project_id"] == "p1"
    assert backend.search_calls[0]["run_id"] == "r1"
    assert backend.write_calls[0]["content"] == "remember this"
    assert backend.write_calls[0]["project_id"] == "p1"
    assert backend.write_calls[0]["kind"] == "operator"


@pytest.mark.asyncio
async def test_agent_runner_can_call_existing_sdk_tool_through_adapter() -> None:
    raw_provider = MockLLMProvider(
        [
            {
                "action": "call_tool",
                "arguments": {"tool_name": "lookup", "arguments": {"query": "target"}},
            },
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    center = LLMProviderCenter(default_provider="mock")
    center.register("mock", raw_provider, default_model="mock-mini")
    session = build_agent_core_session_from_sdk_tools(
        profile=AgentProfile(
            name="adapter",
            model="mock-mini",
            budget=RuntimeBudget(max_iterations=3),
        ),
        provider=center,
        sdk_tools=[_fake_tool()],
        metadata={"source": "test"},
    )

    outcome = await AgentRunner(session).run("inspect target")

    assert outcome.result.status == "completed"
    assert session.manifest()["metadata"]["adapter"] == "openai_agents_runtime"
    assert outcome.session_manifest["capabilities"]["tools"]["tools"][0]["name"] == "lookup"


@pytest.mark.asyncio
async def test_agent_runner_can_use_runtime_llm_provider_adapter() -> None:
    async def complete(payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["messages"][0]["content"]
        return {"content": '{"action":"finish","arguments":{"output":"runtime-done"}}'}

    session = build_agent_core_session_from_sdk_tools(
        profile=AgentProfile(name="provider-adapter", model="runtime-model"),
        provider=RuntimeLLMProviderAdapter(
            complete,
            config=RuntimeLLMProviderAdapterConfig(model="runtime-model"),
        ),
        sdk_tools=[_fake_tool()],
    )

    outcome = await AgentRunner(session).run("inspect target")

    assert outcome.result.status == "completed"
    assert outcome.result.output == "runtime-done"


@pytest.mark.asyncio
async def test_agent_runner_injects_runtime_memory_adapter_hits() -> None:
    backend = _FakeWorkingMemoryBackend()
    provider = RuntimeLLMProviderAdapter(
        lambda payload: {"content": '{"action":"finish","arguments":{"output":"done"}}'},
        config=RuntimeLLMProviderAdapterConfig(model="runtime-model"),
    )
    session = build_agent_core_session_from_sdk_tools(
        profile=AgentProfile(name="memory-adapter", model="runtime-model"),
        provider=provider,
        sdk_tools=[_fake_tool()],
    )
    session.memory = RuntimeMemoryAdapter(
        backend,
        config=RuntimeMemoryAdapterConfig(source="working_memory", project_id="p1"),
    )

    outcome = await AgentRunner(session).run("inspect target")

    assert outcome.result.status == "completed"
    assert backend.search_calls[0]["query"] == "inspect target"
    assert "Inspect target admin UI" in provider.requests[0]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_agent_runner_runs_tasktree_equivalent_flow_through_runtime_adapters() -> None:
    state: dict[str, Any] = {"tasks": [], "completed": []}

    async def create_plan(tasks: list[dict[str, Any]], facts: str = "", evidence: str = "") -> str:
        state["tasks"] = list(tasks)
        state["facts"] = facts
        state["evidence"] = evidence
        return f"TaskTree plan created: {len(tasks)} task(s)"

    async def get_plan_status() -> str:
        return f"TaskTree phase=running total={len(state['tasks'])} completed={len(state['completed'])}"

    async def complete_task(task_id: str, result_summary: str = "") -> str:
        state["completed"].append({"task_id": task_id, "result_summary": result_summary})
        return f"TaskTree task completed: {task_id}"

    completion = _SequencedRuntimeCompletion(
        [
            {
                "action": "call_tool",
                "arguments": {
                    "tool_name": "create_plan",
                    "arguments": {
                        "tasks": [
                            {
                                "name": "Recon target",
                                "lane": "recon",
                                "goal": "Collect HTTP facts",
                            }
                        ],
                        "facts": "memory recalled /admin",
                    },
                },
            },
            {"action": "call_tool", "arguments": {"tool_name": "get_plan_status", "arguments": {}}},
            {
                "action": "call_tool",
                "arguments": {
                    "tool_name": "complete_task",
                    "arguments": {
                        "task_id": "Recon target",
                        "result_summary": "Found /admin from recalled memory and lookup.",
                    },
                },
            },
            {
                "action": "finish",
                "arguments": {"output": "TaskTree equivalent flow completed"},
            },
        ]
    )
    backend = _FakeWorkingMemoryBackend()
    session = build_agent_core_session_from_sdk_tools(
        profile=AgentProfile(
            name="tasktree-equivalent",
            model="runtime-model",
            budget=RuntimeBudget(max_iterations=6),
        ),
        provider=RuntimeLLMProviderAdapter(
            completion,
            config=RuntimeLLMProviderAdapterConfig(model="runtime-model"),
        ),
        sdk_tools=[
            _fake_tasktree_tool("create_plan", "Create TaskTree plan", create_plan),
            _fake_tasktree_tool("get_plan_status", "Get TaskTree status", get_plan_status),
            _fake_tasktree_tool("complete_task", "Complete TaskTree task", complete_task),
        ],
        metadata={"flow": "tasktree_equivalent"},
    )
    session.memory = RuntimeMemoryAdapter(
        backend,
        config=RuntimeMemoryAdapterConfig(source="working_memory", project_id="project-1"),
    )

    outcome = await AgentRunner(session).run("plan and complete recon for target")

    assert outcome.result.status == "completed"
    assert outcome.result.output == "TaskTree equivalent flow completed"
    assert state["tasks"][0]["name"] == "Recon target"
    assert state["completed"][0]["task_id"] == "Recon target"
    assert backend.search_calls[0]["query"] == "plan and complete recon for target"
    assert "Inspect target admin UI" in completion.calls[0]["messages"][-1]["content"]
    assert [
        call["tool_name"]
        for call in session.harness.snapshot().manifest()["tool_calls"]
    ] == ["create_plan", "get_plan_status", "complete_task"]
    assert outcome.session_manifest["metadata"]["flow"] == "tasktree_equivalent"


@pytest.mark.asyncio
async def test_runtime_session_factory_mounts_centers_and_runs_tasktree_flow() -> None:
    state: dict[str, Any] = {"tasks": [], "completed": []}

    async def create_plan(tasks: list[dict[str, Any]], facts: str = "", evidence: str = "") -> str:
        state["tasks"] = list(tasks)
        return f"TaskTree plan created: {len(tasks)} task(s)"

    async def complete_task(task_id: str, result_summary: str = "") -> str:
        state["completed"].append({"task_id": task_id, "result_summary": result_summary})
        return f"TaskTree task completed: {task_id}"

    completion = _SequencedRuntimeCompletion(
        [
            {
                "action": "call_tool",
                "arguments": {
                    "tool_name": "create_plan",
                    "arguments": {
                        "tasks": [{"name": "Recon target", "lane": "recon"}],
                        "facts": "factory memory recall",
                    },
                },
            },
            {
                "action": "call_tool",
                "arguments": {
                    "tool_name": "complete_task",
                    "arguments": {"task_id": "Recon target", "result_summary": "done"},
                },
            },
            {"action": "finish", "arguments": {"output": "factory done"}},
        ]
    )
    session = build_agent_core_runtime_session(
        profile=AgentProfile(
            name="factory",
            model="runtime-model",
            budget=RuntimeBudget(max_iterations=5),
        ),
        completion=completion,
        sdk_tools=[
            _fake_tasktree_tool("create_plan", "Create TaskTree plan", create_plan),
            _fake_tasktree_tool("complete_task", "Complete TaskTree task", complete_task),
        ],
        memory_backend=_FakeWorkingMemoryBackend(),
        config=RuntimeAgentCoreSessionConfig(
            provider_name="runtime",
            tool_mount_name="tasktree_tools",
            memory_store_name="working_memory",
            metadata={"flow": "factory"},
        ),
        memory_config=RuntimeMemoryAdapterConfig(source="working_memory", project_id="project-1"),
    )

    outcome = await AgentRunner(session).run("factory tasktree")
    manifest = outcome.session_manifest

    assert outcome.result.status == "completed"
    assert outcome.result.output == "factory done"
    assert state["tasks"][0]["name"] == "Recon target"
    assert state["completed"][0]["task_id"] == "Recon target"
    assert manifest["provider"]["default_provider"] == "runtime"
    assert manifest["capabilities"]["tools"]["mounts"][0]["name"] == "tasktree_tools"
    assert manifest["memory"]["stores"][0]["name"] == "working_memory"
    assert manifest["metadata"]["flow"] == "factory"


@pytest.mark.asyncio
async def test_tasktree_runtime_session_factory_uses_lazy_tool_builder_shape() -> None:
    state: dict[str, Any] = {"builder": {}, "tasks": [], "completed": []}

    async def create_plan(tasks: list[dict[str, Any]], facts: str = "", evidence: str = "") -> str:
        state["tasks"] = list(tasks)
        return f"TaskTree plan created: {len(tasks)} task(s)"

    async def complete_task(task_id: str, result_summary: str = "") -> str:
        state["completed"].append({"task_id": task_id, "result_summary": result_summary})
        return f"TaskTree task completed: {task_id}"

    def tool_builder(plan_md_path: str = "", project_id: str = ""):
        state["builder"] = {"plan_md_path": plan_md_path, "project_id": project_id}
        return (
            [
                _fake_tasktree_tool("create_plan", "Create TaskTree plan", create_plan),
                _fake_tasktree_tool("complete_task", "Complete TaskTree task", complete_task),
            ],
            lambda force=False: None,
            lambda task_id: None,
        )

    completion = _SequencedRuntimeCompletion(
        [
            {
                "action": "call_tool",
                "arguments": {
                    "tool_name": "create_plan",
                    "arguments": {"tasks": [{"name": "Recon target"}]},
                },
            },
            {
                "action": "call_tool",
                "arguments": {
                    "tool_name": "complete_task",
                    "arguments": {"task_id": "Recon target", "result_summary": "done"},
                },
            },
            {"action": "finish", "arguments": {"output": "tasktree factory done"}},
        ]
    )

    session = build_agent_core_tasktree_runtime_session(
        profile=AgentProfile(
            name="real-tasktree-entry",
            model="runtime-model",
            budget=RuntimeBudget(max_iterations=5),
        ),
        completion=completion,
        memory_backend=_FakeWorkingMemoryBackend(),
        config=RuntimeTaskTreeSessionConfig(
            project_id="project-1",
            plan_md_path="ignored.md",
            metadata={"flow": "tasktree_factory"},
        ),
        tool_builder=tool_builder,
    )

    outcome = await AgentRunner(session).run("tasktree factory task")
    manifest = outcome.session_manifest

    assert outcome.result.status == "completed"
    assert outcome.result.output == "tasktree factory done"
    assert state["builder"] == {"plan_md_path": "ignored.md", "project_id": "project-1"}
    assert state["tasks"][0]["name"] == "Recon target"
    assert state["completed"][0]["task_id"] == "Recon target"
    assert manifest["capabilities"]["tools"]["mounts"][0]["name"] == "tasktree_tools"
    assert manifest["memory"]["stores"][0]["metadata"]["adapter"] == "runtime_memory"
    assert manifest["metadata"]["tasktree"] is True
    assert manifest["metadata"]["flow"] == "tasktree_factory"


def test_tasktree_runtime_readiness_probe_reports_available_tool_surface() -> None:
    state: dict[str, Any] = {"builder": {}}

    async def create_plan(tasks: list[dict[str, Any]]) -> str:
        return f"created {len(tasks)}"

    async def complete_task(task_id: str) -> str:
        return f"completed {task_id}"

    def tool_builder(plan_md_path: str = "", project_id: str = ""):
        state["builder"] = {"plan_md_path": plan_md_path, "project_id": project_id}
        return (
            [
                _fake_tasktree_tool("create_plan", "Create TaskTree plan", create_plan),
                _fake_tasktree_tool("complete_task", "Complete TaskTree task", complete_task),
            ],
            lambda force=False: None,
            lambda task_id: None,
        )

    readiness = inspect_agent_core_tasktree_runtime(
        config=RuntimeTaskTreeSessionConfig(
            project_id="project-readiness",
            plan_md_path="plan.md",
            metadata={"flow": "readiness"},
        ),
        tool_builder=tool_builder,
    )

    payload = readiness.to_dict()

    assert readiness.status == "available"
    assert readiness.tool_count == 2
    assert readiness.tool_names == ("complete_task", "create_plan")
    assert state["builder"] == {"plan_md_path": "plan.md", "project_id": "project-readiness"}
    assert payload["metadata"]["project_id"] == "project-readiness"
    assert payload["metadata"]["flow"] == "readiness"
    assert payload["manifest"]["metadata"]["tasktree"] is True
    assert [tool["name"] for tool in payload["manifest"]["tools"]] == ["complete_task", "create_plan"]


def test_tasktree_runtime_readiness_probe_reports_missing_dependency_without_crashing() -> None:
    def missing_dependency_builder(plan_md_path: str = "", project_id: str = ""):
        raise ModuleNotFoundError("No module named 'pydantic'")

    readiness = inspect_agent_core_tasktree_runtime(
        config=RuntimeTaskTreeSessionConfig(project_id="project-missing"),
        tool_builder=missing_dependency_builder,
    )

    payload = readiness.to_dict()

    assert readiness.status == "unavailable"
    assert readiness.tool_count == 0
    assert readiness.tool_names == ()
    assert "pydantic" in readiness.reason
    assert payload["metadata"]["project_id"] == "project-missing"
    assert payload["metadata"]["error_type"] == "ModuleNotFoundError"

