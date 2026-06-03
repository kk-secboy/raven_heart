"""Multi-agent handoff contracts and lightweight coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from agent_core.runner import AgentRunOutcome, AgentRunRequest, AgentSession, AgentSessionManager


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
