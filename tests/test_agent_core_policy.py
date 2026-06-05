from __future__ import annotations

from typing import Any

import pytest

from agent_core.actions import ActionRegistry, ParsedAction
from agent_core.approvals import ApprovalDecisionRecord, ApprovalResumeContext, InMemoryApprovalStore
from agent_core.harness import InMemoryAgentJournal
from agent_core.mcp import MCPCenter, MCPServerSpec, MCPToolReference, MCPToolSpec
from agent_core.policy import (
    ApprovalRequest,
    CompositePolicy,
    InMemoryPolicyDecisionStore,
    MarkdownPolicyDecisionStore,
    PolicyDecision,
    PolicyRule,
    RuleBasedPolicy,
    SQLitePolicyDecisionStore,
)
from agent_core.prompt import PromptIR
from agent_core.react import ReActConfig, ReActExecutor
from agent_core.testing import InMemoryHarness, MockLLMProvider
from agent_core.tools import ToolCenter, ToolInvocation, ToolRegistry, ToolResult, ToolSpec


async def _handler(invocation: ToolInvocation) -> ToolResult:
    return ToolResult(
        call_id=invocation.call_id,
        tool_name=invocation.tool_name,
        content="executed",
    )


class FakeMCPConnector:
    def __init__(self) -> None:
        self.invocations = 0

    async def list_tools(self, server: MCPServerSpec) -> tuple[MCPToolSpec, ...]:
        return (
            MCPToolSpec(
                reference=MCPToolReference(server_name=server.name, tool_name="delete_file"),
                description="Delete a file through MCP",
                tags=("filesystem", "write"),
            ),
        )

    async def invoke_tool(
        self,
        server: MCPServerSpec,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        self.invocations += 1
        return ToolResult(call_id="remote", tool_name=tool_name, content="deleted")


@pytest.mark.asyncio
async def test_rule_based_policy_matches_action_names_and_precedence() -> None:
    policy = RuleBasedPolicy(
        [
            PolicyRule(name="approval", status="approval_required", action_names=("danger_*",)),
            PolicyRule(name="deny", status="deny", action_names=("danger_delete",)),
        ]
    )
    decision = await policy.check_action(ParsedAction(name="danger_delete"))

    assert decision.status == "deny"


@pytest.mark.asyncio
async def test_rule_based_policy_denies_tool_by_tag_before_execution() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="delete_target", description="Delete target", tags=("destructive",)),
        _handler,
    )
    provider = MockLLMProvider(
        [
            {"action": "call_tool", "arguments": {"tool_name": "delete_target", "arguments": {}}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    policy = RuleBasedPolicy(
        [
            PolicyRule(
                name="deny-destructive",
                status="deny",
                tool_tags=("destructive",),
                reason="destructive tools are blocked",
            )
        ]
    )
    harness = InMemoryHarness()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=registry,
        action_registry=ActionRegistry(),
        harness=harness,
        policy=policy,
        config=ReActConfig(max_iterations=3),
    )

    result = await executor.run("try delete", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert provider.requests[1].messages[-1].content == "destructive tools are blocked"
    assert harness.tool_calls[0]["status"] == "denied"


@pytest.mark.asyncio
async def test_rule_based_policy_denies_mcp_tool_by_server_metadata() -> None:
    connector = FakeMCPConnector()
    mcp = MCPCenter()
    mcp.register_server(MCPServerSpec(name="fs", transport="mock"))
    mcp.register_connector("mock", connector)
    await mcp.refresh()
    center = ToolCenter()
    center.mount("mcp", mcp)
    provider = MockLLMProvider(
        [
            {
                "action": "call_tool",
                "arguments": {
                    "tool_name": "mcp__fs__delete_file",
                    "arguments": {"path": "target.txt"},
                },
            },
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    policy = RuleBasedPolicy(
        [
            PolicyRule(
                name="deny-fs-mcp",
                status="deny",
                mcp_servers=("fs",),
                reason="filesystem MCP server is blocked",
            )
        ]
    )
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=center,
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        policy=policy,
        config=ReActConfig(max_iterations=3),
    )

    result = await executor.run("try mcp delete", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert connector.invocations == 0
    assert provider.requests[1].messages[-1].content == "filesystem MCP server is blocked"


@pytest.mark.asyncio
async def test_approval_required_policy_returns_tool_result_without_execution() -> None:
    registry = ToolRegistry()
    registry.register(ToolSpec(name="deploy", tags=("release",)), _handler)
    provider = MockLLMProvider(
        [
            {"action": "deploy", "arguments": {}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    policy = RuleBasedPolicy(
        [
            PolicyRule(
                name="release-approval",
                status="approval_required",
                tool_names=("deploy",),
                reason="deployment requires approval",
            )
        ]
    )
    harness = InMemoryHarness()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=registry,
        action_registry=ActionRegistry(),
        harness=harness,
        policy=policy,
        config=ReActConfig(max_iterations=3),
    )

    result = await executor.run("deploy", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert provider.requests[1].messages[-1].content == "deployment requires approval"
    assert harness.tool_calls[0]["status"] == "approval_required"
    assert harness.tool_calls[0]["metadata"]["approval"]["subject"] == "tool:deploy"


@pytest.mark.asyncio
async def test_approval_required_action_policy_stops_run_with_approval_metadata() -> None:
    provider = MockLLMProvider(
        [
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    policy = RuleBasedPolicy(
        [
            PolicyRule(
                name="finish-approval",
                status="approval_required",
                action_names=("finish",),
                reason="final answer needs approval",
            )
        ]
    )
    harness = InMemoryHarness()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=ToolRegistry(),
        action_registry=ActionRegistry(),
        harness=harness,
        policy=policy,
        config=ReActConfig(max_iterations=2),
    )

    result = await executor.run("needs approval", PromptIR.from_parts(dynamic="task"))

    assert result.status == "approval_required"
    assert result.output == "final answer needs approval"
    assert result.metadata["approval"]["subject"] == "action:finish"
    assert harness.finished[-1]["status"] == "approval_required"


@pytest.mark.asyncio
async def test_approved_resume_context_allows_action_policy_gate() -> None:
    approval_store = InMemoryApprovalStore()
    policy = RuleBasedPolicy(
        [
            PolicyRule(
                name="finish-approval",
                status="approval_required",
                action_names=("finish",),
                reason="final answer needs approval",
            )
        ]
    )
    first_provider = MockLLMProvider(
        [
            {"action": "finish", "arguments": {"output": "pending answer"}},
        ]
    )
    first_harness = InMemoryHarness()
    first_executor = ReActExecutor(
        provider=first_provider,
        tool_runtime=ToolRegistry(),
        action_registry=ActionRegistry(),
        harness=first_harness,
        approval_store=approval_store,
        policy=policy,
        config=ReActConfig(max_iterations=2),
    )
    first = await first_executor.run("needs approval", PromptIR.from_parts(dynamic="task"))
    pending = await approval_store.pending()
    resolved = await approval_store.decide(
        pending[0].approval_id,
        ApprovalDecisionRecord(status="approved", actor="operator"),
    )
    second_provider = MockLLMProvider(
        [
            {"action": "finish", "arguments": {"output": "approved answer"}},
        ]
    )
    second_harness = InMemoryHarness()
    second_executor = ReActExecutor(
        provider=second_provider,
        tool_runtime=ToolRegistry(),
        action_registry=ActionRegistry(),
        harness=second_harness,
        approval_store=approval_store,
        approval_resume=ApprovalResumeContext.from_records((resolved,)),
        policy=policy,
        config=ReActConfig(max_iterations=2),
    )

    second = await second_executor.run("resume approved answer", PromptIR.from_parts(dynamic="task"))

    assert first.status == "approval_required"
    assert pending[0].request.subject == "action:finish"
    assert second.status == "completed"
    assert second.output == "approved answer"
    assert second_harness.finished[-1]["status"] == "completed"


@pytest.mark.asyncio
async def test_policy_terminal_statuses_are_valid_for_core_journal() -> None:
    provider = MockLLMProvider(
        [
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    policy = RuleBasedPolicy(
        [
            PolicyRule(
                name="finish-deny",
                status="deny",
                action_names=("finish",),
                reason="finish denied",
            )
        ]
    )
    journal = InMemoryAgentJournal()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=ToolRegistry(),
        action_registry=ActionRegistry(),
        harness=journal,
        policy=policy,
        config=ReActConfig(max_iterations=2),
    )

    result = await executor.run("deny final", PromptIR.from_parts(dynamic="task"))

    assert result.status == "denied"
    assert journal.finished[-1]["status"] == "denied"
    assert journal.runs[result.run_id].status == "denied"


@pytest.mark.asyncio
async def test_composite_policy_denies_over_approval() -> None:
    policy = CompositePolicy(
        [
            RuleBasedPolicy([PolicyRule(name="approval", status="approval_required", tool_names=("scan",))]),
            RuleBasedPolicy([PolicyRule(name="deny", status="deny", tool_names=("scan",))]),
        ]
    )
    decision = await policy.check_tool(ToolInvocation(tool_name="scan"))

    assert decision.status == "deny"


@pytest.mark.asyncio
async def test_policy_decision_stores_persist_records(tmp_path) -> None:
    decision = PolicyDecision(
        status="approval_required",
        reason="release requires approval",
        approval=ApprovalRequest(
            reason="release requires approval",
            subject="tool:deploy",
            metadata={"rule": "release"},
        ),
    )
    stores = (
        InMemoryPolicyDecisionStore(),
        SQLitePolicyDecisionStore(tmp_path / "policy.sqlite"),
        MarkdownPolicyDecisionStore(tmp_path / "policy.md"),
    )

    for store in stores:
        record = await store.submit(
            decision,
            subject_kind="tool",
            subject_name="deploy",
            run_id="run-1",
            turn_id="turn-1",
            metadata={"argument_keys": ["target"]},
        )
        records = await store.records()

        assert records[0].decision_id == record.decision_id
        assert records[0].subject == "tool:deploy"
        assert records[0].decision.status == "approval_required"
        assert records[0].decision.approval is not None
        assert records[0].metadata["argument_keys"] == ["target"]
        assert store.manifest()["record_count"] == 1

    markdown_text = (tmp_path / "policy.md").read_text(encoding="utf-8")
    assert "agent-core-policy-decision-record-payload" not in markdown_text


@pytest.mark.asyncio
async def test_react_records_action_and_tool_policy_decisions() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="delete_target", description="Delete target", tags=("destructive",)),
        _handler,
    )
    provider = MockLLMProvider(
        [
            {"action": "delete_target", "arguments": {"path": "target.txt"}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    policy = RuleBasedPolicy(
        [
            PolicyRule(
                name="deny-destructive",
                status="deny",
                tool_tags=("destructive",),
                reason="destructive tools are blocked",
            )
        ]
    )
    store = InMemoryPolicyDecisionStore()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=registry,
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        policy=policy,
        policy_decision_store=store,
        config=ReActConfig(max_iterations=3),
    )

    result = await executor.run("try delete", PromptIR.from_parts(dynamic="task"))
    records = await store.records()

    assert result.status == "completed"
    assert [record.subject for record in records] == [
        "action:delete_target",
        "tool:delete_target",
        "action:finish",
    ]
    assert records[0].decision.status == "allow"
    assert records[1].decision.status == "deny"
    assert records[1].decision.reason == "destructive tools are blocked"
    assert records[1].metadata["argument_keys"] == ["path"]
    assert records[1].metadata["tool_spec"]["tags"] == ["destructive"]

