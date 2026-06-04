"""Multi-agent handoff contracts and lightweight coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from agent_core.runner import AgentRunOutcome, AgentRunRequest, AgentSession, AgentSessionManager
from agent_core.tools import ToolInvocation, ToolResult, ToolSpec


HandoffDecisionStatus = Literal["selected", "not_found", "denied"]


@dataclass(frozen=True)
class HandoffSpec:
    """Advertised handoff capability for one agent session."""

    session_name: str
    description: str = ""
    tags: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    priority: int = 0
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-handoff-spec/v1",
            "session_name": self.session_name,
            "description": self.description,
            "tags": list(self.tags),
            "tools": list(self.tools),
            "skills": list(self.skills),
            "priority": self.priority,
            "enabled": self.enabled,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class HandoffRequest:
    task: str
    source_session: str = ""
    target_session: str = ""
    required_tags: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    required_skills: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-handoff-request/v1",
            "task": self.task,
            "source_session": self.source_session,
            "target_session": self.target_session,
            "required_tags": list(self.required_tags),
            "required_tools": list(self.required_tools),
            "required_skills": list(self.required_skills),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class HandoffDecision:
    status: HandoffDecisionStatus
    request: HandoffRequest
    selected_session: str = ""
    reason: str = ""
    candidates: tuple[HandoffSpec, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selected(self) -> bool:
        return self.status == "selected" and bool(self.selected_session)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-handoff-decision/v1",
            "status": self.status,
            "selected_session": self.selected_session,
            "reason": self.reason,
            "request": self.request.manifest(),
            "candidates": [candidate.manifest() for candidate in self.candidates],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class HandoffRecord:
    decision: HandoffDecision
    outcome: AgentRunOutcome | None = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        outcome_manifest: dict[str, Any] = {}
        if self.outcome is not None:
            outcome_manifest = {
                "result": {
                    "run_id": self.outcome.result.run_id,
                    "status": self.outcome.result.status,
                    "iterations": self.outcome.result.iterations,
                    "output_bytes": len(self.outcome.result.output.encode("utf-8")),
                },
                "trace_run_id": self.outcome.trace_manifest.get("run", {}).get("run_id", ""),
            }
        return {
            "schema_version": "agent-core-handoff-record/v1",
            "decision": self.decision.manifest(),
            "outcome": outcome_manifest,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentToolSpec:
    """Expose one managed agent session as a provider-neutral tool."""

    tool_name: str
    session_name: str
    description: str = ""
    task_argument: str = "task"
    enabled: bool = True
    tags: tuple[str, ...] = ("agent",)
    metadata: dict[str, Any] = field(default_factory=dict)

    def tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name,
            description=self.description or f"Delegate a task to agent session {self.session_name}.",
            parameters_schema={
                "type": "object",
                "properties": {
                    self.task_argument: {
                        "type": "string",
                        "description": "Task for the delegated agent session.",
                    },
                },
                "required": [self.task_argument],
                "additionalProperties": True,
            },
            tags=tuple(dict.fromkeys((*self.tags, "agent_tool"))),
            enabled=self.enabled,
            metadata={
                **dict(self.metadata),
                "agent_tool": self.manifest(),
            },
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-agent-tool-spec/v1",
            "tool_name": self.tool_name,
            "session_name": self.session_name,
            "description": self.description,
            "task_argument": self.task_argument,
            "enabled": self.enabled,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


class HandoffRouterPort(Protocol):
    def decide(self, request: HandoffRequest) -> HandoffDecision:
        """Select a target session for one handoff request."""


class HandoffRouter(HandoffRouterPort):
    """Deterministic router over advertised session handoff specs."""

    def __init__(self, specs: tuple[HandoffSpec, ...] = ()) -> None:
        self._specs: dict[str, HandoffSpec] = {spec.session_name: spec for spec in specs}

    def register(self, spec: HandoffSpec) -> None:
        if not spec.session_name.strip():
            raise ValueError("handoff session_name is required")
        self._specs[spec.session_name] = spec

    def unregister(self, session_name: str) -> bool:
        return self._specs.pop(session_name, None) is not None

    def specs(self, *, include_disabled: bool = False) -> tuple[HandoffSpec, ...]:
        specs = self._specs.values() if include_disabled else (
            spec for spec in self._specs.values() if spec.enabled
        )
        return tuple(sorted(specs, key=lambda item: (-item.priority, item.session_name)))

    def decide(self, request: HandoffRequest) -> HandoffDecision:
        if request.target_session:
            spec = self._specs.get(request.target_session)
            if spec is None:
                return HandoffDecision("not_found", request, reason="target session is not registered")
            if not spec.enabled:
                return HandoffDecision(
                    "denied",
                    request,
                    reason="target session is disabled",
                    candidates=(spec,),
                )
            if not _matches(spec, request):
                return HandoffDecision(
                    "denied",
                    request,
                    reason="target session does not satisfy handoff requirements",
                    candidates=(spec,),
                )
            return HandoffDecision("selected", request, selected_session=spec.session_name, candidates=(spec,))

        candidates = tuple(spec for spec in self.specs() if _matches(spec, request))
        if not candidates:
            return HandoffDecision("not_found", request, reason="no matching handoff target")
        selected = candidates[0]
        return HandoffDecision(
            "selected",
            request,
            selected_session=selected.session_name,
            candidates=candidates,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-handoff-router/v1",
            "specs": [spec.manifest() for spec in self.specs(include_disabled=True)],
        }


class MultiAgentCoordinator:
    """Run tasks through a routed target `AgentSessionManager` session."""

    def __init__(
        self,
        *,
        manager: AgentSessionManager,
        router: HandoffRouterPort | None = None,
    ) -> None:
        self.manager = manager
        self.router = router or HandoffRouter()
        self.records: list[HandoffRecord] = []

    @classmethod
    def from_manager(cls, manager: AgentSessionManager) -> "MultiAgentCoordinator":
        router = HandoffRouter()
        for session_name in manager.sessions():
            router.register(handoff_spec_from_session(manager.session(session_name), session_name=session_name))
        return cls(manager=manager, router=router)

    async def handoff(
        self,
        request: HandoffRequest,
        *,
        run_request: AgentRunRequest | None = None,
    ) -> HandoffRecord:
        decision = self.router.decide(request)
        if not decision.selected:
            record = HandoffRecord(decision=decision, error=decision.reason)
            self.records.append(record)
            return record
        base_request = run_request or AgentRunRequest(
            task=request.task,
            metadata={
                **dict(request.metadata),
                "handoff": decision.manifest(),
            },
        )
        outcome = await self.manager.run(decision.selected_session, base_request)
        record = HandoffRecord(decision=decision, outcome=outcome)
        self.records.append(record)
        return record

    def manifest(self) -> dict[str, Any]:
        router_manifest = getattr(self.router, "manifest", None)
        return {
            "schema_version": "agent-core-multi-agent-coordinator/v1",
            "router": router_manifest() if callable(router_manifest) else {},
            "record_count": len(self.records),
            "records": [record.manifest() for record in self.records],
        }


class AgentToolRuntime:
    """Tool runtime that delegates tool invocations to managed agent sessions."""

    def __init__(
        self,
        *,
        manager: AgentSessionManager,
        specs: tuple[AgentToolSpec, ...] = (),
    ) -> None:
        self.manager = manager
        self._specs: dict[str, AgentToolSpec] = {}
        self.records: list[dict[str, Any]] = []
        for spec in specs:
            self.register(spec)

    @classmethod
    def from_manager(cls, manager: AgentSessionManager) -> "AgentToolRuntime":
        runtime = cls(manager=manager)
        for session_name in manager.sessions():
            session = manager.session(session_name)
            runtime.register(agent_tool_spec_from_session(session, session_name=session_name))
        return runtime

    def register(self, spec: AgentToolSpec) -> None:
        if not spec.tool_name.strip():
            raise ValueError("agent tool_name is required")
        if not spec.session_name.strip():
            raise ValueError("agent tool session_name is required")
        self._specs[spec.tool_name] = spec

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(
            spec.tool_spec()
            for spec in sorted(self._specs.values(), key=lambda item: item.tool_name)
            if spec.enabled
        )

    async def invoke(self, invocation: ToolInvocation) -> ToolResult:
        spec = self._specs.get(invocation.tool_name)
        if spec is None or not spec.enabled:
            return ToolResult(
                call_id=invocation.call_id,
                tool_name=invocation.tool_name,
                status="failed",
                error=f"agent tool not found: {invocation.tool_name}",
                metadata={"agent_tool_missing": True},
            )
        task = _agent_tool_task(invocation, spec)
        request = AgentRunRequest(
            task=task,
            metadata={
                "agent_tool": spec.manifest(),
                "parent_tool_invocation": invocation.manifest(),
            },
        )
        try:
            outcome = await self.manager.run(spec.session_name, request)
        except Exception as exc:
            record = _agent_tool_record(spec, invocation, task=task, error=str(exc))
            self.records.append(record)
            return ToolResult(
                call_id=invocation.call_id,
                tool_name=invocation.tool_name,
                status="failed",
                error=str(exc),
                metadata=record,
            )
        record = _agent_tool_record(spec, invocation, task=task, outcome=outcome)
        self.records.append(record)
        status = "completed" if outcome.result.status == "completed" else "failed"
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            status=status,
            content=outcome.result.output,
            data={
                "run_id": outcome.result.run_id,
                "status": outcome.result.status,
                "iterations": outcome.result.iterations,
            },
            error="" if status == "completed" else outcome.result.status,
            metadata=record,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-agent-tool-runtime/v1",
            "tool_count": len(self._specs),
            "tools": [
                spec.manifest()
                for spec in sorted(self._specs.values(), key=lambda item: item.tool_name)
            ],
            "record_count": len(self.records),
            "records": [dict(record) for record in self.records],
        }


def handoff_spec_from_session(session: AgentSession, *, session_name: str = "") -> HandoffSpec:
    capabilities = session.profile.capabilities
    metadata = {
        "profile": session.profile.name,
        **dict(session.metadata),
    }
    return HandoffSpec(
        session_name=session_name or session.profile.name,
        description=session.profile.instructions,
        tags=tuple(str(item) for item in metadata.get("tags", ()) or ()),
        tools=tuple(capabilities.tools),
        skills=tuple(capabilities.skills),
        priority=int(metadata.get("handoff_priority") or 0),
        enabled=bool(metadata.get("handoff_enabled", True)),
        metadata=metadata,
    )


def agent_tool_spec_from_session(
    session: AgentSession,
    *,
    session_name: str = "",
    tool_name: str = "",
) -> AgentToolSpec:
    resolved_session_name = session_name or session.profile.name
    metadata = {"profile": session.profile.name, **dict(session.metadata)}
    return AgentToolSpec(
        tool_name=tool_name or _agent_tool_name(resolved_session_name),
        session_name=resolved_session_name,
        description=session.profile.instructions,
        enabled=bool(metadata.get("agent_tool_enabled", True)),
        tags=tuple(str(item) for item in metadata.get("tags", ()) or ()),
        metadata=metadata,
    )


def _matches(spec: HandoffSpec, request: HandoffRequest) -> bool:
    if not spec.enabled:
        return False
    tags = set(spec.tags)
    tools = set(spec.tools)
    skills = set(spec.skills)
    return (
        set(request.required_tags) <= tags
        and set(request.required_tools) <= tools
        and set(request.required_skills) <= skills
    )


def _agent_tool_task(invocation: ToolInvocation, spec: AgentToolSpec) -> str:
    value = invocation.arguments.get(spec.task_argument)
    if value is None:
        value = invocation.arguments.get("task") or invocation.arguments.get("input") or ""
    return str(value)


def _agent_tool_record(
    spec: AgentToolSpec,
    invocation: ToolInvocation,
    *,
    task: str,
    outcome: AgentRunOutcome | None = None,
    error: str = "",
) -> dict[str, Any]:
    result = {}
    if outcome is not None:
        result = {
            "run_id": outcome.result.run_id,
            "status": outcome.result.status,
            "iterations": outcome.result.iterations,
            "output_bytes": len(outcome.result.output.encode("utf-8")),
            "trace_run_id": outcome.trace_manifest.get("run", {}).get("run_id", ""),
        }
    return {
        "schema_version": "agent-core-agent-tool-call/v1",
        "tool": spec.manifest(),
        "invocation": invocation.manifest(),
        "task_bytes": len(task.encode("utf-8")),
        "result": result,
        "error": error,
    }


def _agent_tool_name(session_name: str) -> str:
    normalized = "".join(char if char.isalnum() else "_" for char in session_name.lower())
    normalized = "_".join(part for part in normalized.split("_") if part)
    return f"agent_{normalized or 'session'}"
