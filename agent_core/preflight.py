"""Run preflight contracts for provider-neutral agent sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


PreflightSeverity = Literal["info", "warning", "error"]
PreflightStatus = Literal["passed", "blocked"]


@dataclass(frozen=True)
class AgentRunPreflightRequirements:
    max_task_bytes: int | None = None
    required_actions: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    required_skills: tuple[str, ...] = ()
    required_mcp_servers: tuple[str, ...] = ()
    require_memory: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-run-preflight-requirements/v1",
            "max_task_bytes": self.max_task_bytes,
            "required_actions": list(self.required_actions),
            "required_tools": list(self.required_tools),
            "required_skills": list(self.required_skills),
            "required_mcp_servers": list(self.required_mcp_servers),
            "require_memory": self.require_memory,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentRunPreflightRequest:
    task: str
    session_name: str = ""
    requirements: AgentRunPreflightRequirements = field(
        default_factory=AgentRunPreflightRequirements
    )
    available_actions: tuple[str, ...] = ()
    available_tools: tuple[str, ...] = ()
    available_skills: tuple[str, ...] = ()
    available_mcp_servers: tuple[str, ...] = ()
    memory_enabled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task_bytes(self) -> int:
        return len(self.task.encode("utf-8"))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-run-preflight-request/v1",
            "task_bytes": self.task_bytes,
            "session_name": self.session_name,
            "requirements": self.requirements.manifest(),
            "available_actions": list(self.available_actions),
            "available_tools": list(self.available_tools),
            "available_skills": list(self.available_skills),
            "available_mcp_servers": list(self.available_mcp_servers),
            "memory_enabled": self.memory_enabled,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentRunPreflightIssue:
    severity: PreflightSeverity
    code: str
    message: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return self.severity == "error"

    def manifest(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "blocking": self.blocking,
            "metadata": dict(self.metadata),
        }


class AgentRunPreflightCheckPort(Protocol):
    name: str

    async def check(self, request: AgentRunPreflightRequest) -> tuple[AgentRunPreflightIssue, ...]:
        """Return preflight issues for one run request."""


@dataclass(frozen=True)
class AgentRunPreflightCheckRecord:
    check_name: str
    issues: tuple[AgentRunPreflightIssue, ...] = ()

    @property
    def blocking_count(self) -> int:
        return sum(1 for issue in self.issues if issue.blocking)

    def manifest(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "issue_count": len(self.issues),
            "blocking_count": self.blocking_count,
            "issues": [issue.manifest() for issue in self.issues],
        }


@dataclass(frozen=True)
class AgentRunPreflightReport:
    request: AgentRunPreflightRequest
    checks: tuple[AgentRunPreflightCheckRecord, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def issues(self) -> tuple[AgentRunPreflightIssue, ...]:
        return tuple(issue for check in self.checks for issue in check.issues)

    @property
    def blocking_issues(self) -> tuple[AgentRunPreflightIssue, ...]:
        return tuple(issue for issue in self.issues if issue.blocking)

    @property
    def status(self) -> PreflightStatus:
        return "blocked" if self.blocking_issues else "passed"

    @property
    def ok(self) -> bool:
        return self.status == "passed"

    def manifest(self) -> dict[str, Any]:
        issues = self.issues
        blocking = self.blocking_issues
        return {
            "schema_version": "agent-core-run-preflight-report/v1",
            "status": self.status,
            "ok": self.ok,
            "check_count": len(self.checks),
            "issue_count": len(issues),
            "blocking_count": len(blocking),
            "codes": sorted({issue.code for issue in issues}),
            "blocking_codes": sorted({issue.code for issue in blocking}),
            "request": self.request.manifest(),
            "checks": [check.manifest() for check in self.checks],
            "metadata": dict(self.metadata),
        }


class StaticRunPreflightCheck:
    """Deterministic SDK-local checks that do not call providers or tools."""

    name = "static"

    async def check(self, request: AgentRunPreflightRequest) -> tuple[AgentRunPreflightIssue, ...]:
        issues: list[AgentRunPreflightIssue] = []
        requirements = request.requirements
        if not request.task.strip():
            issues.append(
                AgentRunPreflightIssue(
                    "warning",
                    "empty_task",
                    "run task is empty",
                    source=self.name,
                )
            )
        if requirements.max_task_bytes is not None and request.task_bytes > requirements.max_task_bytes:
            issues.append(
                AgentRunPreflightIssue(
                    "error",
                    "task_bytes_exceeded",
                    "run task exceeds configured preflight byte limit",
                    source=self.name,
                    metadata={
                        "actual": request.task_bytes,
                        "limit": int(requirements.max_task_bytes),
                    },
                )
            )
        issues.extend(
            _missing_requirements(
                "missing_action",
                "required action is unavailable",
                requirements.required_actions,
                request.available_actions,
                source=self.name,
            )
        )
        issues.extend(
            _missing_requirements(
                "missing_tool",
                "required tool is unavailable",
                requirements.required_tools,
                request.available_tools,
                source=self.name,
            )
        )
        issues.extend(
            _missing_requirements(
                "missing_skill",
                "required skill is unavailable",
                requirements.required_skills,
                request.available_skills,
                source=self.name,
            )
        )
        issues.extend(
            _missing_requirements(
                "missing_mcp_server",
                "required MCP server is unavailable",
                requirements.required_mcp_servers,
                request.available_mcp_servers,
                source=self.name,
            )
        )
        if requirements.require_memory and not request.memory_enabled:
            issues.append(
                AgentRunPreflightIssue(
                    "error",
                    "memory_required",
                    "run requires memory but the session does not expose memory",
                    source=self.name,
                )
            )
        return tuple(issues)


class AgentRunPreflightCenter:
    """Composable run preflight bus for SDK and runtime-owned checks."""

    def __init__(self, checks: tuple[AgentRunPreflightCheckPort, ...] = ()) -> None:
        self._checks: list[AgentRunPreflightCheckPort] = list(checks)

    @classmethod
    def default(cls) -> "AgentRunPreflightCenter":
        return cls((StaticRunPreflightCheck(),))

    def register(self, check: AgentRunPreflightCheckPort) -> None:
        self._checks.append(check)

    def checks(self) -> tuple[AgentRunPreflightCheckPort, ...]:
        return tuple(self._checks)

    async def check(self, request: AgentRunPreflightRequest) -> AgentRunPreflightReport:
        records: list[AgentRunPreflightCheckRecord] = []
        for check in self._checks:
            issues = await check.check(request)
            records.append(
                AgentRunPreflightCheckRecord(
                    check_name=getattr(check, "name", check.__class__.__name__),
                    issues=tuple(issues),
                )
            )
        return AgentRunPreflightReport(request=request, checks=tuple(records))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-run-preflight-center/v1",
            "check_count": len(self._checks),
            "checks": [
                getattr(check, "name", check.__class__.__name__) for check in self._checks
            ],
        }


def default_agent_run_preflight_center() -> AgentRunPreflightCenter:
    return AgentRunPreflightCenter.default()


def _missing_requirements(
    code: str,
    message: str,
    required: tuple[str, ...],
    available: tuple[str, ...],
    *,
    source: str,
) -> tuple[AgentRunPreflightIssue, ...]:
    available_set = set(available)
    return tuple(
        AgentRunPreflightIssue(
            "error",
            code,
            message,
            source=source,
            metadata={"name": name},
        )
        for name in required
        if name not in available_set
    )
