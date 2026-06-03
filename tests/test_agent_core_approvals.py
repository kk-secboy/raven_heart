from __future__ import annotations

import pytest

from agent_core import (
    ApprovalCenter,
    ApprovalDecisionRecord,
    ApprovalQueueFilter,
    ApprovalRequest,
    ApprovalResumeContext,
    InMemoryApprovalStore,
    ListEventSink,
    MarkdownApprovalStore,
    SQLiteApprovalStore,
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
async def test_approval_resume_context_exports_approved_subject_grants() -> None:
    store = InMemoryApprovalStore()
    request = ApprovalRequest(
        reason="release requires approval",
        subject="tool:deploy",
        metadata={"rule": "release"},
    )
    record = await store.submit(request, run_id="run-1", turn_id="turn-1")
    resolved = await store.decide(
        record.approval_id,
        ApprovalDecisionRecord(
            status="approved",
            actor="operator",
            reason="ship it",
            metadata={"ticket": "APP-1"},
        ),
    )

    resume = ApprovalResumeContext.from_records((resolved,), metadata={"source": "test"})
    grant = resume.approved_for(request)
    manifest = resume.manifest()

    assert grant is not None
    assert grant.approval_id == record.approval_id
    assert grant.subject == "tool:deploy"
    assert grant.actor == "operator"
    assert manifest["grant_count"] == 1
    assert manifest["approved_count"] == 1
    assert manifest["grants"][0]["decision_metadata"]["ticket"] == "APP-1"
    assert manifest["metadata"]["source"] == "test"


@pytest.mark.asyncio
async def test_approval_center_views_resolves_and_builds_resume_context() -> None:
    store = InMemoryApprovalStore()
    center = ApprovalCenter(store)
    deploy = await center.submit(
        ApprovalRequest(
            reason="release requires approval",
            subject="tool:deploy",
            metadata={"rule": "release"},
        ),
        run_id="run-1",
        turn_id="turn-1",
    )
    finish = await center.submit(
        ApprovalRequest(
            reason="final answer requires approval",
            subject="action:finish",
            metadata={"rule": "finish"},
        ),
        run_id="run-2",
        turn_id="turn-1",
    )

    pending_deploy = await center.view(ApprovalQueueFilter(subject="tool:deploy"))
    approved = await center.approve(deploy.approval_id, actor="operator", reason="ok")
    rejected = await center.reject(finish.approval_id, actor="operator", reason="not yet")
    resume = await center.resume_context(run_id="run-1")
    manifest = await center.manifest()

    assert pending_deploy.records[0].approval_id == deploy.approval_id
    assert approved.record.status == "approved"
    assert approved.resume_context.approved_for(deploy.request) is not None
    assert rejected.record.status == "rejected"
    assert rejected.resume_context.approved_for(finish.request) is None
    assert resume.approved_for(deploy.request) is not None
    assert resume.approved_for(finish.request) is None
    assert manifest["schema_version"] == "agent-core-approval-center/v1"
    assert manifest["queue"]["record_count"] == 2
    assert manifest["queue"]["approved_count"] == 1
    assert manifest["queue"]["rejected_count"] == 1


@pytest.mark.asyncio
async def test_approval_center_works_with_durable_stores(tmp_path) -> None:
    path = tmp_path / "approvals.sqlite"
    first = ApprovalCenter(SQLiteApprovalStore(path))
    record = await first.submit(
        ApprovalRequest(reason="release requires approval", subject="tool:deploy"),
        run_id="run-1",
    )
    await first.approve(record.approval_id, actor="operator", metadata={"ticket": "APP-1"})

    second = ApprovalCenter(SQLiteApprovalStore(path))
    view = await second.view(
        ApprovalQueueFilter(statuses=("approved",), run_id="run-1", subject="tool:deploy")
    )
    resume = await second.resume_context(approval_ids=(record.approval_id,))

    assert view.records[0].decision is not None
    assert view.records[0].decision.metadata["ticket"] == "APP-1"
    assert resume.grants[0].approval_id == record.approval_id
    assert resume.grants[0].approved is True


@pytest.mark.asyncio
async def test_sqlite_approval_store_persists_records_across_instances(tmp_path) -> None:
    path = tmp_path / "approvals.sqlite"
    first = SQLiteApprovalStore(path)
    request = ApprovalRequest(
        reason="release requires approval",
        subject="tool:deploy",
        metadata={"rule": "release"},
    )

    record = await first.submit(request, run_id="run-1", turn_id="turn-1")
    await first.decide(
        record.approval_id,
        ApprovalDecisionRecord(
            status="approved",
            actor="operator",
            reason="ok",
            metadata={"ticket": "APP-1"},
        ),
    )
    second = SQLiteApprovalStore(path)
    restored = await second.get(record.approval_id)
    manifest = second.manifest()

    assert restored is not None
    assert restored.status == "approved"
    assert restored.request.subject == "tool:deploy"
    assert restored.decision is not None
    assert restored.decision.metadata["ticket"] == "APP-1"
    assert await second.pending() == ()
    assert manifest["schema_version"] == "agent-core-sqlite-approval-store/v1"
    assert manifest["record_count"] == 1
    assert manifest["pending_count"] == 0


@pytest.mark.asyncio
async def test_markdown_approval_store_persists_records_and_hides_payload_text(tmp_path) -> None:
    path = tmp_path / "approvals.md"
    first = MarkdownApprovalStore(path)
    request = ApprovalRequest(
        reason="comment marker --> should not break markdown",
        subject="action:finish",
        metadata={"rule": "finish-approval"},
    )

    record = await first.submit(request, run_id="run-1", turn_id="turn-1")
    await first.decide(
        record.approval_id,
        ApprovalDecisionRecord(status="rejected", actor="operator", reason="not yet"),
    )
    second = MarkdownApprovalStore(path)
    restored = await second.get(record.approval_id)
    manifest = second.manifest()
    text = path.read_text(encoding="utf-8")

    assert restored is not None
    assert restored.status == "rejected"
    assert restored.decision is not None
    assert restored.decision.actor == "operator"
    assert "comment marker --> should not break markdown" not in text
    assert manifest["schema_version"] == "agent-core-markdown-approval-store/v1"
    assert manifest["record_count"] == 1
    assert manifest["records"][0]["request"]["subject"] == "action:finish"


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


@pytest.mark.asyncio
async def test_react_uses_approved_resume_context_to_execute_tool_gate() -> None:
    registry = ToolRegistry()
    registry.register(ToolSpec(name="deploy", tags=("release",)), _handler)
    approval_store = InMemoryApprovalStore()
    request = ApprovalRequest(
        reason="deployment requires approval",
        subject="tool:deploy",
        metadata={"rule": "release-approval"},
    )
    record = await approval_store.submit(request)
    resolved = await approval_store.decide(
        record.approval_id,
        ApprovalDecisionRecord(status="approved", actor="operator"),
    )
    events = ListEventSink()
    provider = MockLLMProvider(
        [
            {"action": "deploy", "arguments": {}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=registry,
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        event_sink=events,
        approval_store=approval_store,
        approval_resume=ApprovalResumeContext.from_records((resolved,)),
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
    event_manifest = events.manifest()

    assert result.status == "completed"
    assert result.output == "done"
    assert provider.requests[1].messages[-1].content == "executed"
    assert len(await approval_store.pending()) == 0
    assert any(event["type"] == "approval_resumed" for event in event_manifest["events"])
