"""SDK-level capability visibility and failure-isolation acceptance checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.capabilities import CapabilityCatalog, CapabilityQuery
from agent_core.mcp import (
    MCPCenter,
    MCPPromptSpec,
    MCPResourceSpec,
    MCPServerSpec,
    MCPToolReference,
    MCPToolSpec,
)
from agent_core.skills import SkillRegistry, SkillsContext, SkillSpec
from agent_core.tools import ToolCenter, ToolInvocation, ToolRegistry, ToolResult, ToolSpec
from agent_core.trace import MCPCenterTrace, SkillCenterTrace, ToolCenterTrace


@dataclass(frozen=True)
class AgentCoreCapabilityGovernanceAcceptanceIssue:
    """One blocking capability governance acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-capability-governance-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreCapabilityGovernanceAcceptanceReport:
    """Prompt-safe capability visibility, disabled-route, and isolation report."""

    status: str
    tool_governance: dict[str, Any] = field(default_factory=dict)
    mcp_governance: dict[str, Any] = field(default_factory=dict)
    skill_governance: dict[str, Any] = field(default_factory=dict)
    capability_discovery: dict[str, Any] = field(default_factory=dict)
    prompt_governance: dict[str, Any] = field(default_factory=dict)
    traces: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreCapabilityGovernanceAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-capability-governance-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "tool_governance": dict(self.tool_governance),
            "mcp_governance": dict(self.mcp_governance),
            "skill_governance": dict(self.skill_governance),
            "capability_discovery": dict(self.capability_discovery),
            "prompt_governance": dict(self.prompt_governance),
            "traces": dict(self.traces),
            "metadata": dict(self.metadata),
        }


class _GovernanceMCPConnector:
    def __init__(self) -> None:
        self.invocations: list[tuple[str, str, dict[str, Any]]] = []

    async def list_tools(self, server: MCPServerSpec) -> tuple[MCPToolSpec, ...]:
        return (
            MCPToolSpec(
                reference=MCPToolReference(
                    server_name=server.name,
                    tool_name="read_file",
                ),
                description="Read a safe project file.",
                parameters_schema={"type": "object"},
                tags=("safe", "docs"),
            ),
            MCPToolSpec(
                reference=MCPToolReference(
                    server_name=server.name,
                    tool_name="delete_file",
                ),
                description="Delete a project file.",
                parameters_schema={"type": "object"},
                tags=("danger",),
                enabled=False,
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
            call_id="mcp-governance-call",
            tool_name=tool_name,
            content=f"{server.name}:{tool_name}:ok",
        )

    async def list_resources(self, server: MCPServerSpec) -> tuple[MCPResourceSpec, ...]:
        return (
            MCPResourceSpec(
                server_name=server.name,
                uri="file://SAFE.md",
                name="safe docs",
                description="Safe docs resource",
                tags=("safe", "docs"),
            ),
        )

    async def list_prompts(self, server: MCPServerSpec) -> tuple[MCPPromptSpec, ...]:
        return (
            MCPPromptSpec(
                server_name=server.name,
                name="safe_summary",
                description="Summarize safe docs.",
                tags=("safe", "docs"),
            ),
        )


class _FailingMCPConnector:
    async def list_tools(self, server: MCPServerSpec) -> tuple[MCPToolSpec, ...]:
        raise RuntimeError(f"{server.name} inventory unavailable")


@dataclass(frozen=True)
class AgentCoreCapabilityGovernanceAcceptanceHarness:
    """Run deterministic capability governance and isolation checks."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCoreCapabilityGovernanceAcceptanceReport:
        local_tools = _governance_local_tools()
        mcp = _governance_mcp()
        inventory = await mcp.refresh_inventory(fail_fast=False)
        tools = ToolCenter()
        tools.mount("local", local_tools, tags=("local",))
        tools.mount("mcp", mcp, tags=("remote",))
        skills = _governance_skills()
        catalog = CapabilityCatalog(tools=tools, skills=skills, mcp=mcp)

        safe_local = await tools.invoke(
            ToolInvocation(tool_name="safe_lookup", arguments={"query": "docs"})
        )
        safe_mcp = await tools.invoke(
            ToolInvocation(tool_name="mcp__fs__read_file", arguments={"path": "SAFE.md"})
        )
        disabled_local_plan = tools.route_plan("delete_target").manifest()
        disabled_mcp_plan = tools.route_plan("mcp__fs__delete_file").manifest()
        disabled_local = await tools.invoke(
            ToolInvocation(tool_name="delete_target", arguments={"target": "prod"})
        )
        disabled_mcp = await tools.invoke(
            ToolInvocation(tool_name="mcp__fs__delete_file", arguments={"path": "prod"})
        )
        prompt_text = catalog.render_prompt(include_skills=True)
        discovery = catalog.discover(
            CapabilityQuery(query="delete danger docs", tags=("danger",), limit=20)
        )
        traces = {
            "tool_center": ToolCenterTrace.from_session({"tools": tools.manifest()}),
            "mcp_center": MCPCenterTrace.from_session({"mcp": mcp.manifest()}),
            "skill_center": SkillCenterTrace.from_session({"skills": skills.snapshot()}),
        }
        tool_governance = {
            "schema_version": "agent-core-tool-governance-acceptance/v1",
            "visible_tool_names": [spec.name for spec in tools.specs()],
            "local_disabled_plan": disabled_local_plan,
            "mcp_disabled_plan": disabled_mcp_plan,
            "safe_local_status": safe_local.status,
            "safe_mcp_status": safe_mcp.status,
            "disabled_local_status": disabled_local.status,
            "disabled_local_error": disabled_local.error,
            "disabled_mcp_status": disabled_mcp.status,
            "disabled_mcp_error": disabled_mcp.error,
            "call_count": int(tools.manifest().get("call_count") or 0),
        }
        mcp_governance = {
            "schema_version": "agent-core-mcp-governance-acceptance/v1",
            "inventory_statuses": {
                result.server_name: result.status for result in inventory
            },
            "state_statuses": {
                state.server_name: state.status for state in mcp.states()
            },
            "visible_mcp_tools": [spec.name for spec in mcp.specs()],
            "audit_mcp_tools": [
                spec.name for spec in mcp.specs(include_disabled=True)
            ],
            "failed_servers": [
                state.server_name for state in mcp.states() if state.status == "failed"
            ],
            "disabled_servers": [
                state.server_name for state in mcp.states() if state.status == "disabled"
            ],
        }
        skill_governance = {
            "schema_version": "agent-core-skill-governance-acceptance/v1",
            "auto_selected": [
                skill.name for skill in skills.registry.select("delete production target")
            ],
            "safe_selected": [
                skill.name for skill in skills.registry.select("review docs safely")
            ],
            "loaded_skill_names": [skill.name for skill in skills.loaded()],
            "available_skill_names": [skill.name for skill in skills.registry.list()],
        }
        discovery_manifest = discovery.manifest()
        prompt_governance = {
            "schema_version": "agent-core-capability-prompt-governance-acceptance/v1",
            "bytes": len(prompt_text.encode("utf-8")),
            "contains_safe_lookup": "safe_lookup" in prompt_text,
            "contains_mcp_read_file": "mcp__fs__read_file" in prompt_text,
            "contains_delete_target": "delete_target" in prompt_text,
            "contains_mcp_delete_file": "mcp__fs__delete_file" in prompt_text,
            "contains_danger_admin": "danger-admin" in prompt_text,
        }
        issues = _capability_governance_issues(
            tool_governance=tool_governance,
            mcp_governance=mcp_governance,
            skill_governance=skill_governance,
            capability_discovery=discovery_manifest,
            prompt_governance=prompt_governance,
            traces=traces,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCoreCapabilityGovernanceAcceptanceReport(
            status=status,
            tool_governance=tool_governance,
            mcp_governance=mcp_governance,
            skill_governance=skill_governance,
            capability_discovery=discovery_manifest,
            prompt_governance=prompt_governance,
            traces=traces,
            issues=issues,
            metadata={
                "scenario": "agent_core_capability_governance_acceptance",
                **dict(self.metadata),
            },
        )


async def run_agent_core_capability_governance_acceptance(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreCapabilityGovernanceAcceptanceReport:
    """Run default capability governance and failure-isolation acceptance checks."""

    return await AgentCoreCapabilityGovernanceAcceptanceHarness(
        metadata=dict(metadata or {})
    ).run()


def _governance_local_tools() -> ToolRegistry:
    registry = ToolRegistry()

    async def safe_lookup(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            content=f"safe:{invocation.arguments.get('query', '')}",
        )

    async def delete_target(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            status="failed",
            error="should not execute disabled tool",
        )

    registry.register(
        ToolSpec(
            name="safe_lookup",
            description="Lookup safe project context.",
            tags=("safe", "docs"),
            parameters_schema={"type": "object"},
        ),
        safe_lookup,
    )
    registry.register(
        ToolSpec(
            name="delete_target",
            description="Dangerous deletion tool.",
            aliases=("rm_target",),
            tags=("danger",),
            parameters_schema={"type": "object"},
            enabled=False,
        ),
        delete_target,
    )
    return registry


def _governance_mcp() -> MCPCenter:
    mcp = MCPCenter()
    mcp.register_server(MCPServerSpec(name="fs", transport="memory", tags=("safe", "docs")))
    mcp.register_server(
        MCPServerSpec(name="disabled", transport="memory", tags=("danger",), enabled=False)
    )
    mcp.register_server(MCPServerSpec(name="broken", transport="broken", tags=("danger",)))
    mcp.register_connector("memory", _GovernanceMCPConnector())
    mcp.register_connector("broken", _FailingMCPConnector())
    return mcp


def _governance_skills() -> SkillsContext:
    registry = SkillRegistry()
    registry.register(
        SkillSpec(
            name="safe-review",
            description="Review safe docs before answering.",
            prompt="Use safe docs and tool outputs.",
            tags=("safe", "docs"),
            priority=10,
        )
    )
    registry.register(
        SkillSpec(
            name="danger-admin",
            description="Privileged dangerous administrative skill.",
            prompt="Requires explicit operator-controlled loading.",
            tags=("danger", "delete"),
            priority=100,
            disable_model_invocation=True,
        )
    )
    skills = SkillsContext(registry)
    skills.load("safe-review")
    return skills


def _capability_governance_issues(
    *,
    tool_governance: dict[str, Any],
    mcp_governance: dict[str, Any],
    skill_governance: dict[str, Any],
    capability_discovery: dict[str, Any],
    prompt_governance: dict[str, Any],
    traces: dict[str, Any],
) -> tuple[AgentCoreCapabilityGovernanceAcceptanceIssue, ...]:
    issues: list[AgentCoreCapabilityGovernanceAcceptanceIssue] = []
    visible_tools = set(tool_governance.get("visible_tool_names") or ())
    forbidden_visible = visible_tools.intersection(
        {"delete_target", "mcp__fs__delete_file"}
    )
    if forbidden_visible:
        issues.append(
            AgentCoreCapabilityGovernanceAcceptanceIssue(
                source="tool_governance",
                code="disabled_tool_visible",
                message="Disabled tools were visible in default tool inventory.",
                metadata={"visible_tools": sorted(visible_tools)},
            )
        )
    _check_disabled_plan(
        issues,
        "local_disabled_plan",
        tool_governance.get("local_disabled_plan"),
    )
    _check_disabled_plan(
        issues,
        "mcp_disabled_plan",
        tool_governance.get("mcp_disabled_plan"),
    )
    if tool_governance.get("safe_local_status") != "completed" or tool_governance.get("safe_mcp_status") != "completed":
        issues.append(
            AgentCoreCapabilityGovernanceAcceptanceIssue(
                source="tool_governance",
                code="safe_tool_execution_failed",
                message="Enabled local or MCP tool did not execute.",
                metadata=dict(tool_governance),
            )
        )
    for key in ("disabled_local", "disabled_mcp"):
        if tool_governance.get(f"{key}_status") != "failed" or "disabled" not in str(
            tool_governance.get(f"{key}_error") or ""
        ).lower():
            issues.append(
                AgentCoreCapabilityGovernanceAcceptanceIssue(
                    source="tool_governance",
                    code=f"{key}_not_blocked",
                    message=f"{key} did not fail closed as disabled.",
                    metadata=dict(tool_governance),
                )
            )
    statuses = mcp_governance.get("inventory_statuses") if isinstance(mcp_governance.get("inventory_statuses"), dict) else {}
    expected_statuses = {"fs": "refreshed", "disabled": "disabled", "broken": "failed"}
    for server_name, expected in expected_statuses.items():
        if statuses.get(server_name) != expected:
            issues.append(
                AgentCoreCapabilityGovernanceAcceptanceIssue(
                    source="mcp_governance",
                    code="mcp_server_status_unexpected",
                    message=f"MCP server status mismatch for {server_name}.",
                    metadata={"expected": expected, "actual": statuses.get(server_name), "statuses": dict(statuses)},
                )
            )
    visible_mcp_tools = set(mcp_governance.get("visible_mcp_tools") or ())
    audit_mcp_tools = set(mcp_governance.get("audit_mcp_tools") or ())
    if "mcp__fs__delete_file" in visible_mcp_tools or "mcp__fs__delete_file" not in audit_mcp_tools:
        issues.append(
            AgentCoreCapabilityGovernanceAcceptanceIssue(
                source="mcp_governance",
                code="mcp_disabled_tool_visibility_wrong",
                message="Disabled MCP tool was not separated between visible and audit inventories.",
                metadata=dict(mcp_governance),
            )
        )
    if "danger-admin" in set(skill_governance.get("auto_selected") or ()):
        issues.append(
            AgentCoreCapabilityGovernanceAcceptanceIssue(
                source="skill_governance",
                code="disabled_skill_auto_selected",
                message="Skill disabled for model invocation was auto-selected.",
                metadata=dict(skill_governance),
            )
        )
    if "safe-review" not in set(skill_governance.get("safe_selected") or ()):
        issues.append(
            AgentCoreCapabilityGovernanceAcceptanceIssue(
                source="skill_governance",
                code="safe_skill_not_selected",
                message="Safe skill was not selected for a safe docs task.",
                metadata=dict(skill_governance),
            )
        )
    _check_discovery(issues, capability_discovery)
    if prompt_governance.get("contains_delete_target") or prompt_governance.get("contains_mcp_delete_file"):
        issues.append(
            AgentCoreCapabilityGovernanceAcceptanceIssue(
                source="prompt_governance",
                code="disabled_tool_in_prompt",
                message="Capability prompt exposed a disabled tool.",
                metadata=dict(prompt_governance),
            )
        )
    trace_tool = traces.get("tool_center") if isinstance(traces.get("tool_center"), dict) else {}
    if int(trace_tool.get("failed_count") or 0) < 2:
        issues.append(
            AgentCoreCapabilityGovernanceAcceptanceIssue(
                source="trace",
                code="disabled_failures_not_traced",
                message="ToolCenter trace did not record disabled call failures.",
                metadata={"tool_center_trace": dict(trace_tool)},
            )
        )
    trace_mcp = traces.get("mcp_center") if isinstance(traces.get("mcp_center"), dict) else {}
    if "broken" not in set(trace_mcp.get("failed_servers") or ()):
        issues.append(
            AgentCoreCapabilityGovernanceAcceptanceIssue(
                source="trace",
                code="mcp_failed_server_not_traced",
                message="MCP trace did not record the failed server.",
                metadata={"mcp_center_trace": dict(trace_mcp)},
            )
        )
    trace_skill = traces.get("skill_center") if isinstance(traces.get("skill_center"), dict) else {}
    if "safe-review" not in set(trace_skill.get("loaded_skill_names") or ()):
        issues.append(
            AgentCoreCapabilityGovernanceAcceptanceIssue(
                source="trace",
                code="loaded_skill_not_traced",
                message="Skill trace did not record the loaded safe skill.",
                metadata={"skill_center_trace": dict(trace_skill)},
            )
        )
    return tuple(issues)


def _check_disabled_plan(
    issues: list[AgentCoreCapabilityGovernanceAcceptanceIssue],
    source: str,
    raw_plan: Any,
) -> None:
    plan = raw_plan if isinstance(raw_plan, dict) else {}
    candidates = [
        candidate
        for candidate in plan.get("candidates") or ()
        if isinstance(candidate, dict)
    ]
    if plan.get("ready") is True or not any(
        candidate.get("reason") == "disabled" for candidate in candidates
    ):
        issues.append(
            AgentCoreCapabilityGovernanceAcceptanceIssue(
                source=source,
                code="disabled_route_plan_missing",
                message="Disabled route plan did not expose disabled candidate without selecting it.",
                metadata={"route_plan": dict(plan)},
            )
        )


def _check_discovery(
    issues: list[AgentCoreCapabilityGovernanceAcceptanceIssue],
    discovery: dict[str, Any],
) -> None:
    matches = [
        match for match in discovery.get("matches") or () if isinstance(match, dict)
    ]
    disabled_tool_matches = [
        match
        for match in matches
        if match.get("kind") in {"tool", "mcp_tool"} and match.get("enabled") is False
    ]
    if disabled_tool_matches:
        issues.append(
            AgentCoreCapabilityGovernanceAcceptanceIssue(
                source="capability_discovery",
                code="disabled_tool_discovered",
                message="Capability discovery exposed disabled tool matches.",
                metadata={"matches": disabled_tool_matches},
            )
        )
    skill_matches = [
        match for match in matches if match.get("kind") == "skill" and match.get("name") == "danger-admin"
    ]
    if not skill_matches or skill_matches[0].get("enabled") is not False:
        issues.append(
            AgentCoreCapabilityGovernanceAcceptanceIssue(
                source="capability_discovery",
                code="disabled_skill_audit_missing",
                message="Capability discovery did not expose disabled skill as disabled audit metadata.",
                metadata={"matches": matches},
            )
        )
