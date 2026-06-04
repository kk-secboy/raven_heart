"""SDK-level tool, skill, MCP, and capability orchestration acceptance checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.actions import ActionRegistry, ActionSpec
from agent_core.capabilities import CapabilityCatalog, CapabilityQuery
from agent_core.mcp import (
    MCPCenter,
    MCPContextMaterialRequest,
    MCPPromptContent,
    MCPPromptSpec,
    MCPResourceContent,
    MCPResourceSpec,
    MCPServerSpec,
    MCPToolReference,
    MCPToolSpec,
)
from agent_core.skills import SkillRegistry, SkillsContext, SkillSpec
from agent_core.tools import ToolCenter, ToolInvocation, ToolRegistry, ToolResult, ToolSpec
from agent_core.trace import MCPCenterTrace, SkillCenterTrace, ToolCenterTrace


@dataclass(frozen=True)
class AgentCoreOrchestrationAcceptanceIssue:
    """One blocking capability-orchestration acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-orchestration-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreOrchestrationAcceptanceReport:
    """Prompt-safe tool/skill/MCP orchestration acceptance report."""

    status: str
    capability_discovery: dict[str, Any] = field(default_factory=dict)
    capability_prompt: dict[str, Any] = field(default_factory=dict)
    tool_center: dict[str, Any] = field(default_factory=dict)
    mcp_inventory: dict[str, Any] = field(default_factory=dict)
    mcp_context_materials: dict[str, Any] = field(default_factory=dict)
    skill_center: dict[str, Any] = field(default_factory=dict)
    traces: dict[str, Any] = field(default_factory=dict)
    execution_summary: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreOrchestrationAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-orchestration-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "capability_discovery": dict(self.capability_discovery),
            "capability_prompt": dict(self.capability_prompt),
            "tool_center": dict(self.tool_center),
            "mcp_inventory": dict(self.mcp_inventory),
            "mcp_context_materials": dict(self.mcp_context_materials),
            "skill_center": dict(self.skill_center),
            "traces": dict(self.traces),
            "execution_summary": dict(self.execution_summary),
            "metadata": dict(self.metadata),
        }


class OrchestrationAcceptanceMCPConnector:
    """Deterministic in-memory MCP connector for SDK orchestration checks."""

    def __init__(self) -> None:
        self.invocations: list[tuple[str, str, dict[str, Any]]] = []

    async def list_tools(self, server: MCPServerSpec) -> tuple[MCPToolSpec, ...]:
        return (
            MCPToolSpec(
                reference=MCPToolReference(server_name=server.name, tool_name="read_file"),
                description="Read project documentation through MCP",
                parameters_schema={
                    "type": "object",
                    "required": ["path"],
                    "properties": {"path": {"type": "string"}},
                },
                tags=("filesystem", "docs"),
            ),
        )

    async def invoke_tool(
        self,
        server: MCPServerSpec,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        self.invocations.append((server.name, tool_name, dict(arguments)))
        return ToolResult(
            call_id="mcp-call",
            tool_name=tool_name,
            content=f"{server.name}:{tool_name}:{arguments.get('path', '')}",
        )

    async def list_resources(self, server: MCPServerSpec) -> tuple[MCPResourceSpec, ...]:
        return (
            MCPResourceSpec(
                server_name=server.name,
                uri="file://README.md",
                name="README",
                description="Raven Heart README",
                mime_type="text/markdown",
                tags=("docs",),
            ),
        )

    async def read_resource(self, server: MCPServerSpec, uri: str) -> MCPResourceContent:
        return MCPResourceContent(
            server_name=server.name,
            uri=uri,
            text=f"{server.name}:{uri}:context",
            mime_type="text/markdown",
        )

    async def list_prompts(self, server: MCPServerSpec) -> tuple[MCPPromptSpec, ...]:
        return (
            MCPPromptSpec(
                server_name=server.name,
                name="summarize",
                description="Summarize project context",
                arguments_schema={
                    "type": "object",
                    "required": ["target"],
                    "properties": {"target": {"type": "string"}},
                },
                tags=("analysis", "docs"),
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
                    "content": f"summarize {arguments.get('target', 'project')}",
                },
            ),
            description="Summarize project context",
        )


@dataclass(frozen=True)
class AgentCoreOrchestrationAcceptanceHarness:
    """Run deterministic capability orchestration acceptance checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreOrchestrationAcceptanceReport:
        actions = _acceptance_actions()
        local_tools = _acceptance_local_tools()
        mcp = _acceptance_mcp()
        inventory = await mcp.refresh_inventory()
        tools = ToolCenter()
        tools.mount("local", local_tools, tags=("builtin",))
        tools.mount("mcp", mcp, tags=("remote",))
        skills = _acceptance_skills()
        catalog = CapabilityCatalog(actions=actions, tools=tools, skills=skills, mcp=mcp)
        discovery = catalog.discover(
            CapabilityQuery(query="docs project review filesystem", tags=("docs",), limit=20)
        )
        prompt_text = catalog.render_prompt(include_skills=True)
        local_result = await tools.invoke(
            ToolInvocation(tool_name="lookup_project", arguments={"query": "docs"})
        )
        mcp_result = await tools.invoke(
            ToolInvocation(
                tool_name="mcp__fs__read_file",
                arguments={"path": "README.md"},
            )
        )
        context_materials = await mcp.context_materials(
            MCPContextMaterialRequest(
                query="docs project",
                prompt_arguments={"fs:summarize": {"target": "raven heart"}},
                limit=4,
                metadata={"acceptance": "orchestration"},
            )
        )
        tool_manifest = tools.manifest()
        mcp_manifest = mcp.manifest()
        skill_manifest = skills.snapshot()
        traces = {
            "tool_center": ToolCenterTrace.from_session({"tools": tool_manifest}),
            "mcp_center": MCPCenterTrace.from_session({"mcp": mcp_manifest}),
            "skill_center": SkillCenterTrace.from_session({"skills": skill_manifest}),
        }
        discovery_manifest = discovery.manifest()
        context_manifest = context_materials.manifest()
        execution_summary = {
            "local_tool_ok": local_result.ok,
            "local_tool_name": local_result.tool_name,
            "mcp_tool_ok": mcp_result.ok,
            "mcp_tool_name": mcp_result.tool_name,
            "capability_prompt_sections": _capability_prompt_sections(prompt_text),
        }
        issues = _orchestration_acceptance_issues(
            discovery=discovery_manifest,
            prompt_text=prompt_text,
            tool_center=tool_manifest,
            mcp_inventory=tuple(result.manifest() for result in inventory),
            mcp_context_materials=context_manifest,
            skill_center=skill_manifest,
            traces=traces,
            execution_summary=execution_summary,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreOrchestrationAcceptanceReport(
            status=status,
            capability_discovery=discovery_manifest,
            capability_prompt={
                "schema_version": "agent-core-capability-prompt-acceptance/v1",
                "bytes": len(prompt_text.encode("utf-8")),
                "sections": execution_summary["capability_prompt_sections"],
            },
            tool_center=tool_manifest,
            mcp_inventory={
                "schema_version": "agent-core-orchestration-mcp-inventory/v1",
                "results": [result.manifest() for result in inventory],
            },
            mcp_context_materials=context_manifest,
            skill_center=skill_manifest,
            traces=traces,
            execution_summary=execution_summary,
            issues=issues,
            metadata={"scenario": "agent_core_orchestration_acceptance", **dict(self.metadata)},
        )


async def run_agent_core_orchestration_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreOrchestrationAcceptanceReport:
    """Run the default tool/skill/MCP orchestration acceptance checks."""

    return await AgentCoreOrchestrationAcceptanceHarness(metadata=dict(metadata or {})).run()


def _acceptance_actions() -> ActionRegistry:
    actions = ActionRegistry()
    actions.register(ActionSpec(name="finish", description="Finish the run", terminal=True))
    actions.register(ActionSpec(name="search_tools", description="Search docs and tools"))
    return actions


def _acceptance_local_tools() -> ToolRegistry:
    registry = ToolRegistry()

    async def lookup(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            content=f"local:{invocation.arguments.get('query', '')}",
        )

    registry.register(
        ToolSpec(
            name="lookup_project",
            description="Lookup local project documentation",
            aliases=("lookup_docs",),
            tags=("local", "docs"),
            parameters_schema={
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
        ),
        lookup,
    )
    return registry


def _acceptance_mcp() -> MCPCenter:
    mcp = MCPCenter()
    mcp.register_server(MCPServerSpec(name="fs", transport="memory", tags=("docs",)))
    mcp.register_connector("memory", OrchestrationAcceptanceMCPConnector())
    return mcp


def _acceptance_skills() -> SkillsContext:
    registry = SkillRegistry()
    registry.register(
        SkillSpec(
            name="project-review",
            description="Review project context before deciding",
            prompt="Use project documentation and tool results before answering.",
            tags=("docs", "review"),
            priority=10,
        )
    )
    skills = SkillsContext(registry)
    skills.load("project-review")
    return skills


def _capability_prompt_sections(prompt_text: str) -> dict[str, bool]:
    return {
        "action_inventory": "[action_inventory]" in prompt_text,
        "tool_inventory": "[tool_inventory]" in prompt_text,
        "mcp_servers": "[mcp_servers]" in prompt_text,
        "skills_context": "[skills_context]" in prompt_text,
    }


def _orchestration_acceptance_issues(
    *,
    discovery: dict[str, Any],
    prompt_text: str,
    tool_center: dict[str, Any],
    mcp_inventory: tuple[dict[str, Any], ...],
    mcp_context_materials: dict[str, Any],
    skill_center: dict[str, Any],
    traces: dict[str, Any],
    execution_summary: dict[str, Any],
) -> tuple[AgentCoreOrchestrationAcceptanceIssue, ...]:
    issues: list[AgentCoreOrchestrationAcceptanceIssue] = []
    counts = dict(discovery.get("counts") or {})
    required_kinds = {
        "action",
        "tool",
        "skill",
        "mcp_tool",
        "mcp_resource",
        "mcp_prompt",
        "mcp_server",
    }
    missing = sorted(kind for kind in required_kinds if int(counts.get(kind) or 0) < 1)
    if missing:
        issues.append(
            AgentCoreOrchestrationAcceptanceIssue(
                source="capability_discovery",
                code="capability_kind_missing",
                message="Capability discovery did not include every required kind.",
                metadata={"missing": missing, "counts": counts},
            )
        )
    sections = _capability_prompt_sections(prompt_text)
    missing_sections = sorted(name for name, present in sections.items() if not present)
    if missing_sections:
        issues.append(
            AgentCoreOrchestrationAcceptanceIssue(
                source="capability_prompt",
                code="capability_prompt_section_missing",
                message="Capability prompt did not render every required section.",
                metadata={"missing": missing_sections, "sections": sections},
            )
        )
    if int(tool_center.get("call_count") or 0) < 2:
        issues.append(
            AgentCoreOrchestrationAcceptanceIssue(
                source="tool_center",
                code="tool_center_calls_missing",
                message="ToolCenter did not record both local and MCP calls.",
                metadata={"tool_center": dict(tool_center)},
            )
        )
    trace_tool = dict(traces.get("tool_center") or {})
    if int(trace_tool.get("failed_count") or 0) != 0:
        issues.append(
            AgentCoreOrchestrationAcceptanceIssue(
                source="tool_center_trace",
                code="tool_center_call_failed",
                message="ToolCenter trace recorded failed calls.",
                metadata={"trace": trace_tool},
            )
        )
    if not execution_summary.get("local_tool_ok") or not execution_summary.get("mcp_tool_ok"):
        issues.append(
            AgentCoreOrchestrationAcceptanceIssue(
                source="tool_execution",
                code="tool_execution_failed",
                message="Local or MCP tool execution did not complete.",
                metadata={"execution_summary": dict(execution_summary)},
            )
        )
    if not mcp_inventory or any(item.get("status") != "refreshed" for item in mcp_inventory):
        issues.append(
            AgentCoreOrchestrationAcceptanceIssue(
                source="mcp_inventory",
                code="mcp_inventory_not_refreshed",
                message="MCP inventory refresh did not finish cleanly.",
                metadata={"inventory": [dict(item) for item in mcp_inventory]},
            )
        )
    if int(mcp_context_materials.get("material_count") or 0) < 2:
        issues.append(
            AgentCoreOrchestrationAcceptanceIssue(
                source="mcp_context_materials",
                code="mcp_context_materials_missing",
                message="MCP resources/prompts were not exported as context materials.",
                metadata={"mcp_context_materials": dict(mcp_context_materials)},
            )
        )
    if int(mcp_context_materials.get("failed_count") or 0) != 0:
        issues.append(
            AgentCoreOrchestrationAcceptanceIssue(
                source="mcp_context_materials",
                code="mcp_context_materials_failed",
                message="MCP context material export recorded failures.",
                metadata={"mcp_context_materials": dict(mcp_context_materials)},
            )
        )
    if not skill_center.get("loaded_skills"):
        issues.append(
            AgentCoreOrchestrationAcceptanceIssue(
                source="skill_center",
                code="skill_not_loaded",
                message="Skill center did not expose a loaded skill.",
                metadata={"skill_center": dict(skill_center)},
            )
        )
    trace_mcp = dict(traces.get("mcp_center") or {})
    if "fs" not in set(trace_mcp.get("refreshed_servers") or ()):
        issues.append(
            AgentCoreOrchestrationAcceptanceIssue(
                source="mcp_center_trace",
                code="mcp_refreshed_server_missing",
                message="MCP trace did not record the refreshed server.",
                metadata={"trace": trace_mcp},
            )
        )
    trace_skill = dict(traces.get("skill_center") or {})
    if "project-review" not in set(trace_skill.get("loaded_skill_names") or ()):
        issues.append(
            AgentCoreOrchestrationAcceptanceIssue(
                source="skill_center_trace",
                code="skill_trace_missing",
                message="Skill trace did not record the loaded skill.",
                metadata={"trace": trace_skill},
            )
        )
    return tuple(issues)
