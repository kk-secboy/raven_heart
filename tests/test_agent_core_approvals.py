from __future__ import annotations

import pytest

from agent_core import (
    ApprovalDecisionRecord,
    ApprovalRequest,
    InMemoryApprovalStore,
    ListEventSink,
)
from agent_core.actions import ActionRegistry
from agent_core.policy import PolicyRule, RuleBasedPolicy
from agent_core.prompt import PromptIR
from agent_core.react import ReActConfig, ReActExecutor
from agent_core.testing import InMemoryHarness, MockLLMProvider
from agent_core.tools import ToolInvocation, ToolRegistry, ToolResult, ToolSpec


async def _handler(invocation: ToolInvocation) -> ToolResult:
    return ToolResult(call_id=invocation.call_id, tool_name=invocation.tool_name, content="executed")


@pytest.mark.asyncio
async def test_in_memory_approval_store_tracks_pending_and_decision_manifest() -> None:
    store = InMemoryApprovalStore()
    request = ApprovalRequest(
        reason="release requires approval",
        subject="tool:deploy",
        metadata={"rule": "release"},
    )

    record = await store.submit(request, run_id="run-1", turn_id="turn-1")
    pending = await store.pending()
    resolved = await store.decide(
        record.approval_id,
        ApprovalDecisionRecord(status="approved", actor="operator", reason="ok"),
    )
    manifest = store.manifest()

    assert pending[0].approval_id == record.approval_id
    assert resolved.status == "approved"
    assert not resolved.pending
    assert resolved.decision is not None
    assert resolved.decision.actor == "operator"
    assert manifest["record_count"] == 1
    assert manifest["pending_count"] == 0
    assert manifest["records"][0]["request"]["subject"] == "tool:deploy"

    with pytest.raises(ValueError, match="approval already resolved"):
        await store.decide(record.approval_id, ApprovalDecisionRecord(status="rejected"))


@pytest.mark.asyncio
async def test_react_records_tool_approval_request_in_store_and_event_stream() -> None:
    registry = ToolRegistry()
    registry.register(ToolSpec(name="deploy", tags=("release",)), _handler)
    provider = MockLLMProvider(
        [
            {"action": "deploy", "arguments": {}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    approval_store = InMemoryApprovalStore()
    events = ListEventSink()
    harness = InMemoryHarness()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=registry,
        action_registry=ActionRegistry(),
        harness=harness,
        event_sink=events,
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
        config=ReActConfig(max_iterations=3),
    )

    result = await executor.run("deploy", PromptIR.from_parts(dynamic="task"))
    pending = await approval_store.pending()
    event_manifest = events.manifest()

    assert result.status == "completed"
    assert len(pending) == 1
    assert pending[0].request.subject == "tool:deploy"
    assert pending[0].metadata == {"kind": "tool", "tool_name": "deploy"}
    assert harness.tool_calls[0]["metadata"]["approval_record"]["approval_id"] == pending[0].approval_id
    assert any(event["type"] == "approval_requested" for event in event_manifest["events"])
