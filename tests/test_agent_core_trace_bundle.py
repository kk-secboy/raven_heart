from __future__ import annotations

import pytest

from agent_core.config import AgentProfile
from agent_core.events import ListEventSink
from agent_core.harness import InMemoryAgentJournal, InMemoryJournalStore, PersistentAgentJournal
from agent_core.memory import InMemoryMemoryStore
from agent_core.policy import InMemoryPolicyDecisionStore
from agent_core.providers import LLMProviderCenter
from agent_core.runner import AgentRunner, AgentSession
from agent_core.testing import MockLLMProvider
from agent_core.tools import (
    InMemoryToolReplayStore,
    PersistentToolReplay,
    ToolInvocation,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from agent_core.trace import (
    AgentRunTraceBundle,
    InMemoryRunTraceStore,
    MarkdownRunTraceStore,
    SQLiteRunTraceStore,
    StorageBackendTrace,
    TraceCorrelationIndex,
)
from agent_core.artifacts import InMemoryArtifactStore
from agent_core.backends import storage_backend_manifest


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
        resume_plan={"ready": True, "checkpoint_id": "c1"},
        timeline_reduction={"compressed_bytes": 10},
        capability_discovery={"match_count": 4},
        memory_search={"hit_count": 2},
        prompt={"metadata": {"trim": {"target_bytes": 900}}},
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
    assert manifest["summary"]["correlation_entry_count"] == 0
    assert manifest["summary"]["has_resume"] is True
    assert manifest["summary"]["has_resume_plan"] is True
    assert manifest["summary"]["resume_plan_ready"] is True
    assert manifest["summary"]["has_timeline_reduction"] is True
    assert manifest["summary"]["capability_discovery_match_count"] == 4
    assert manifest["summary"]["memory_search_hit_count"] == 2
    assert manifest["summary"]["has_prompt_trim"] is True
    assert manifest["capability_discovery"]["match_count"] == 4
    assert manifest["memory_search"]["hit_count"] == 2
    assert manifest["resume_plan"]["checkpoint_id"] == "c1"


def test_trace_correlation_index_cross_references_trace_materials() -> None:
    index = TraceCorrelationIndex.from_trace_components(
        run_id="run-1",
        provider={
            "calls": [
                {
                    "provider_name": "mock",
                    "model": "mock-mini",
                    "attempt": 0,
                    "status": "completed",
                    "metadata": {
                        "request": {"metadata": {"run_id": "run-1", "turn_id": "turn-1"}}
                    },
                }
            ]
        },
        tool_replay={
            "records": [
                {
                    "replay_key": "lookup:{}",
                    "invocation": {"call_id": "call-1", "tool_name": "lookup"},
                    "result": {"call_id": "call-1", "tool_name": "lookup", "status": "completed"},
                }
            ]
        },
        policy_decisions={
            "records": [
                {
                    "decision_id": "decision-1",
                    "sequence": 1,
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "subject": "tool:lookup",
                    "decision": {"status": "allow"},
                    "metadata": {"call_id": "call-1"},
                }
            ]
        },
        approvals={
            "records": [
                {
                    "approval_id": "approval-1",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "status": "approved",
                    "request": {"subject": "tool:lookup", "reason": "ok"},
                    "decision": {"status": "approved", "actor": "operator"},
                }
            ]
        },
        event_log={
            "events": [
                {
                    "type": "tool_finished",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "sequence": 3,
                    "payload": {"call_id": "call-1", "status": "completed"},
                },
                {
                    "type": "approval_requested",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "sequence": 4,
                    "payload": {"approval_id": "approval-1", "request": {"subject": "tool:lookup"}},
                },
            ]
        },
    )
    manifest = index.manifest()

    assert manifest["schema_version"] == "agent-core-trace-correlation-index/v1"
    assert manifest["entry_count"] == 6
    assert "run-1:turn-1" in manifest["groups"]["turns"]
    assert any("tool_replay" in key for key in manifest["groups"]["calls"]["call-1"])
    assert manifest["groups"]["approvals"]["approval-1"]
    assert manifest["groups"]["policy_decisions"]["decision-1"]


def test_storage_backend_trace_collects_and_deduplicates_component_backends() -> None:
    memory_backend = storage_backend_manifest(
        role="memory",
        kind="postgres",
        name="tenant-memory",
        core_builtin=False,
    )
    duplicate_memory_backend = dict(memory_backend)
    trace = StorageBackendTrace.from_trace_components(
        session={
            "memory": {"stores": [{"backend": memory_backend}]},
            "trace_store": {"backend": storage_backend_manifest(role="run_trace", kind="sqlite")},
        },
        memory_search={"plan": {"selected_stores": [{"backend": duplicate_memory_backend}]}},
        event_log={"backend": storage_backend_manifest(role="event_log", kind="markdown")},
    ).manifest()

    assert trace["schema_version"] == "agent-core-storage-backend-trace/v1"
    assert trace["backend_count"] == 3
    assert trace["external_backend_count"] == 1
    assert trace["roles"] == {"event_log": 1, "memory": 1, "run_trace": 1}
    assert trace["kinds"] == {"markdown": 1, "postgres": 1, "sqlite": 1}
    memory = next(item for item in trace["backends"] if item["role"] == "memory")
    assert memory["core_builtin"] is False
    assert len(memory["sources"]) == 2


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
        harness=PersistentAgentJournal(InMemoryJournalStore()),
        memory=InMemoryMemoryStore(),
        event_sink=event_sink,
        tool_replay=PersistentToolReplay(InMemoryToolReplayStore()),
        trace_store=InMemoryRunTraceStore(),
        artifact_store=InMemoryArtifactStore(),
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
    assert trace["summary"]["correlation_entry_count"] > 0
    assert trace["summary"]["has_resume_plan"] is False
    assert trace["summary"]["resume_plan_ready"] is False
    assert trace["journal_replay"]["ok"] is True
    assert trace["provider"]["calls"][0]["provider_name"] == "mock"
    assert trace["provider"]["calls"][0]["metadata"]["request"]["metadata"]["run_id"] == outcome.result.run_id
    assert trace["tool_replay"]["records"][0]["result"]["tool_name"] == "lookup"
    assert trace["policy_decisions"]["records"][1]["subject"] == "tool:lookup"
    call_id = trace["tool_replay"]["records"][0]["result"]["call_id"]
    assert trace["correlation"]["groups"]["calls"][call_id]
    assert trace["prompt"]["metadata"]["profile"] == "traceable"
    assert trace["summary"]["capability_discovery_match_count"] == trace["capability_discovery"]["match_count"]
    assert trace["summary"]["memory_search_hit_count"] == 0
    assert trace["summary"]["storage_backend_count"] >= 7
    assert trace["storage_backends"]["roles"]["memory"] == 1
    assert trace["storage_backends"]["roles"]["journal"] == 1
    assert trace["storage_backends"]["roles"]["tool_replay"] == 1
    assert trace["storage_backends"]["roles"]["run_trace"] == 1
    assert trace["storage_backends"]["roles"]["event_log"] == 1
    assert trace["storage_backends"]["roles"]["artifact"] == 1
    assert trace["storage_backends"]["roles"]["policy_decision"] == 1
    assert trace["capability_discovery"]["schema_version"] == "agent-core-capability-discovery/v1"
    assert trace["memory_search"]["enabled"] is True


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
