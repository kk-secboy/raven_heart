from __future__ import annotations

import pytest

from agent_core.actions import ActionRegistry, ActionSpec
from agent_core.capabilities import CapabilityCatalog
from agent_core.context import AgentContextPack, AgentPromptBuilder
from agent_core.mcp import MCPCenter, MCPServerSpec, MCPToolReference, MCPToolSpec
from agent_core.prompt import PromptBucketRole
from agent_core.skills import SkillRegistry, SkillsContext, SkillSpec
from agent_core.tools import ToolCenter, ToolInvocation, ToolRegistry, ToolResult, ToolSpec
from agent_core.trace import CapabilityTrace


@pytest.mark.asyncio
async def test_action_tool_skill_manifests_feed_capability_trace(tmp_path) -> None:
    actions = ActionRegistry()
    actions.register(
        ActionSpec(
            name="call_tool",
            description="Call a tool",
            parameters_schema={"type": "object", "required": ["tool_name"]},
        )
    )

    tools = ToolRegistry()

    async def handler(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(call_id=invocation.call_id, tool_name=invocation.tool_name)

    tools.register(
        ToolSpec(
            name="http_probe",
            description="Probe HTTP",
            aliases=("probe",),
            tags=("web",),
            parameters_schema={"type": "object"},
        ),
        handler,
    )

    skill_dir = tmp_path / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: review
description: Review code
tags: [code]
---
# Review

Read rules.
""",
        encoding="utf-8",
    )
    (skill_dir / "rules.md").write_text("rule one\nrule two", encoding="utf-8")
    skills = SkillRegistry()
    skills.discover_markdown_skills(tmp_path / "skills")
    context = SkillsContext(skills)
    context.load("review")
    view = context.load_resource("@review/rules.md")

    trace = CapabilityTrace(
        run_id="run-1",
        turn_id="turn-1",
        actions=actions.manifest(),
        tools=tools.select_inventory(max_tokens=100).manifest(),
        skills=context.snapshot(),
    ).manifest()

    assert trace["schema_version"] == "agent-core-capability-trace/v1"
    assert trace["actions"]["actions"][0]["name"] == "call_tool"
    assert trace["tools"]["visible_tools"][0]["aliases"] == ["probe"]
    assert trace["skills"]["loaded_skills"][0]["name"] == "review"
    assert trace["skills"]["views"][0]["view_id"] == view.view_id
    assert trace["mcp"] == {}
    assert trace["catalog"] == {}


def test_action_registry_renders_action_inventory() -> None:
    actions = ActionRegistry()
    actions.register(ActionSpec(name="finish", description="Finish run", terminal=True))
    actions.register(ActionSpec(name="search_tools", description="Search tools"))

    rendered = actions.render_actions()

    assert "- finish terminal=true: Finish run" in rendered
    assert "- search_tools: Search tools" in rendered


class FakeMCPConnector:
    async def list_tools(self, server: MCPServerSpec) -> tuple[MCPToolSpec, ...]:
        return (
            MCPToolSpec(
                reference=MCPToolReference(server_name=server.name, tool_name="read_file"),
                description="Read file",
                tags=("filesystem",),
            ),
        )

    async def invoke_tool(self, server: MCPServerSpec, tool_name: str, arguments: dict) -> ToolResult:
        return ToolResult(call_id="remote", tool_name=tool_name)


@pytest.mark.asyncio
async def test_capability_catalog_manifests_actions_tools_skills_and_mcp() -> None:
    actions = ActionRegistry()
    actions.register(ActionSpec(name="finish", description="Finish", terminal=True))
    tools = ToolRegistry()

    async def handler(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(call_id=invocation.call_id, tool_name=invocation.tool_name)

    tools.register(ToolSpec(name="lookup", description="Lookup", tags=("local",)), handler)
    skills_registry = SkillRegistry()
    skills_registry.register(SkillSpec(name="review", description="Review code", tags=("code",)))
    skills = SkillsContext(skills_registry)
    mcp = MCPCenter()
    mcp.register_server(MCPServerSpec(name="fs", transport="stdio"))
    mcp.register_connector("stdio", FakeMCPConnector())
    await mcp.refresh()
    center = ToolCenter()
    center.mount("local", tools)
    center.mount("mcp", mcp)
    catalog = CapabilityCatalog(actions=actions, tools=center, skills=skills, mcp=mcp)

    manifest = catalog.manifest()
    rendered = catalog.render_prompt(include_skills=True)
    trace = CapabilityTrace(run_id="r", turn_id="t", catalog=manifest).manifest()

    assert manifest["schema_version"] == "agent-core-capability-catalog/v1"
    assert manifest["actions"]["actions"][0]["name"] == "finish"
    assert [tool["name"] for tool in manifest["tools"]["tools"]] == ["lookup", "mcp__fs__read_file"]
    assert manifest["mcp"]["servers"][0]["state"]["status"] == "refreshed"
    assert "[action_inventory]" in rendered
    assert "[tool_inventory]" in rendered
    assert "[mcp_servers]" in rendered
    assert "[skills_context]" in rendered
    assert trace["catalog"]["mcp"]["servers"][0]["name"] == "fs"


@pytest.mark.asyncio
async def test_prompt_builder_places_capability_catalog_in_frozen_bucket() -> None:
    actions = ActionRegistry()
    actions.register(ActionSpec(name="finish", description="Finish", terminal=True))
    tools = ToolRegistry()

    async def handler(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(call_id=invocation.call_id, tool_name=invocation.tool_name)

    tools.register(ToolSpec(name="lookup", description="Lookup"), handler)
    registry = SkillRegistry()
    registry.register(SkillSpec(name="review", prompt="Review carefully."))
    skills = SkillsContext(registry)
    skills.load("review")
    catalog = CapabilityCatalog(actions=actions, tools=tools, skills=skills)

    prompt = AgentPromptBuilder(capabilities=catalog, skills=skills).build(
        AgentContextPack(system="system", dynamic_task="task")
    )

    frozen = prompt.bucket(PromptBucketRole.FROZEN).content
    semi_dynamic = prompt.bucket(PromptBucketRole.SEMI_DYNAMIC_1).content
    assert "[capability_catalog]" in frozen
    assert "[action_inventory]" in frozen
    assert "[tool_inventory]" in frozen
    assert "[skills_context]" not in frozen
    assert "[skills_context]" in semi_dynamic

