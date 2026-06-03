from __future__ import annotations

import pytest

from agent_core.config import AgentProfile
from agent_core.events import ListEventSink
from agent_core.harness import InMemoryAgentJournal
from agent_core.policy import InMemoryPolicyDecisionStore
from agent_core.providers import LLMProviderCenter
from agent_core.runner import AgentRunner, AgentSession
from agent_core.testing import MockLLMProvider
from agent_core.tools import InMemoryToolReplay, ToolInvocation, ToolRegistry, ToolResult, ToolSpec
from agent_core.trace import (
    AgentRunTraceBundle,
    InMemoryRunTraceStore,
    MarkdownRunTraceStore,
    SQLiteRunTraceStore,
)


async def _lookup(invocation: ToolInvocation) -> ToolResult:
    return ToolResult(call_id=invocation.call_id, tool_name=invocation.tool_name, content="found")


def test_agent_run_trace_bundle_summarizes_core_manifests() -> None:
    bundle = AgentRunTraceBundle(
        run_id="run-1",
        status="completed",
        iterations=2,
        output_bytes=4,
        journal_replay={"ok": True, "event_count": 7},
        provider={"call_count": 2},
        tool_replay={"record_count": 1},
        policy_decisions={"record_count": 2},
        approvals={"record_count": 3},
        event_log={"event_count": 9},
        resume={"checkpoint_id": "c1"},
        timeline_reduction={"compressed_bytes": 10},
    )

    manifest = bundle.manifest()

    assert manifest["schema_version"] == "agent-core-run-trace-bundle/v1"
    assert manifest["run"]["run_id"] == "run-1"
    assert manifest["summary"]["journal_ok"] is True
    assert manifest["summary"]["journal_event_count"] == 7
    assert manifest["summary"]["provider_call_count"] == 2
    assert manifest["summary"]["tool_replay_record_count"] == 1
    assert manifest["summary"]["policy_decision_record_count"] == 2
    assert manifest["summary"]["approval_record_count"] == 3
    assert manifest["summary"]["event_log_count"] == 9
    assert manifest["summary"]["has_resume"] is True
    assert manifest["summary"]["has_timeline_reduction"] is True


@pytest.mark.asyncio
async def test_run_trace_stores_persist_bundle_manifests(tmp_path) -> None:
    bundle = AgentRunTraceBundle(
        run_id="run-1",
        status="completed",
        iterations=1,
        provider={"call_count": 1},
    ).manifest()
    memory = InMemoryRunTraceStore()
    sqlite = SQLiteRunTraceStore(tmp_path / "traces.sqlite")
    markdown = MarkdownRunTraceStore(tmp_path / "traces.md")

    for store in (memory, sqlite, markdown):
        await store.save(bundle)
        loaded = await store.load("run-1")
        records = await store.records()

        assert loaded is not None
        assert loaded["run"]["run_id"] == "run-1"
        assert records[0]["summary"]["provider_call_count"] == 1

    assert memory.manifest()["record_count"] == 1
    assert sqlite.manifest()["schema_version"] == "agent-core-sqlite-run-trace-store/v1"
    assert markdown.manifest()["schema_version"] == "agent-core-markdown-run-trace-store/v1"
    assert "agent-core-run-trace-bundle" not in (tmp_path / "traces.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_agent_runner_exports_run_trace_bundle() -> None:
    provider = MockLLMProvider(
        [
            {"action": "lookup", "arguments": {"query": "target"}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    center = LLMProviderCenter(default_provider="mock")
    center.register("mock", provider, default_model="mock-mini")
    tools = ToolRegistry()
    tools.register(ToolSpec(name="lookup"), _lookup)
    event_sink = ListEventSink()
    session = AgentSession(
        profile=AgentProfile(name="traceable", model="mock-mini"),
        provider=center,
        tools=tools,
        harness=InMemoryAgentJournal(),
        event_sink=event_sink,
        tool_replay=InMemoryToolReplay(),
        policy_decision_store=InMemoryPolicyDecisionStore(),
    )

    outcome = await AgentRunner(session).run("inspect")
    trace = outcome.trace_manifest

    assert outcome.result.status == "completed"
    assert trace["schema_version"] == "agent-core-run-trace-bundle/v1"
    assert trace["run"]["run_id"] == outcome.result.run_id
    assert trace["run"]["status"] == "completed"
    assert trace["summary"]["journal_ok"] is True
    assert trace["summary"]["journal_event_count"] > 0
    assert trace["summary"]["provider_call_count"] == 2
    assert trace["summary"]["tool_replay_record_count"] == 1
    assert trace["summary"]["policy_decision_record_count"] == 3
    assert trace["summary"]["event_log_count"] == event_sink.manifest()["event_count"]
    assert trace["journal_replay"]["ok"] is True
    assert trace["provider"]["calls"][0]["provider_name"] == "mock"
    assert trace["tool_replay"]["records"][0]["result"]["tool_name"] == "lookup"
    assert trace["policy_decisions"]["records"][1]["subject"] == "tool:lookup"
    assert trace["prompt"]["metadata"]["profile"] == "traceable"


@pytest.mark.asyncio
async def test_agent_runner_saves_trace_bundle_when_store_is_configured() -> None:
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}])
    trace_store = InMemoryRunTraceStore()
    session = AgentSession(
        profile=AgentProfile(name="trace-store"),
        provider=provider,
        tools=ToolRegistry(),
        harness=InMemoryAgentJournal(),
        trace_store=trace_store,
    )

    outcome = await AgentRunner(session).run("finish")
    saved = await trace_store.load(outcome.result.run_id)

    assert saved is not None
    assert saved["run"]["status"] == "completed"
    assert saved["prompt"]["metadata"]["profile"] == "trace-store"
    assert outcome.session_manifest["trace_store"]["record_count"] == 0
